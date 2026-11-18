"""Candidate generation + feature engineering + per-observation reference labels.

For every GPS observation we take the K nearest road segments (arcs) as candidates.
Each (observation, candidate arc) pair becomes one row; label = 1 if the candidate is
the reference arc for that observation.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from data_loader import clean_track, project

K_CANDIDATES = 8
SAMPLE_STEP_M = 15.0      # densify arcs for KD-tree candidate search
SEARCH_POOL = 60          # densified points queried before de-duplicating arcs

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min",
    "heading_diff", "heading_diff_cos", "proj_frac", "arc_len",
    "speed", "n_cand_30m", "dist_prev", "dist_next", "out_degree", "in_degree",
]


def build_geometry(trip):
    tr = clean_track(trip["track"])
    lon0, lat0 = tr.lon.mean(), tr.lat.mean()
    px, py = project(tr.lon, tr.lat, lon0, lat0)
    nx, ny = project(trip["nodes"].lon, trip["nodes"].lat, lon0, lat0)
    u = trip["arcs"].u.to_numpy(); v = trip["arcs"].v.to_numpy()
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]
    return dict(track=tr, px=px, py=py, ax=ax, ay=ay, bx=bx, by=by, u=u, v=v,
                n_nodes=len(nx), lon0=lon0, lat0=lat0)


def point_seg(px, py, ax, ay, bx, by):
    """Distance from points to segments (broadcast) + projection fraction along segment."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    f = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    f = np.clip(f, 0, 1)
    cx, cy = ax + f * dx, ay + f * dy
    return np.hypot(px - cx, py - cy), f


def arc_kdtree(g):
    ax, ay, bx, by = g["ax"], g["ay"], g["bx"], g["by"]
    L = np.hypot(bx - ax, by - ay)
    n = np.maximum(1, np.ceil(L / SAMPLE_STEP_M).astype(int)) + 1
    ids = np.repeat(np.arange(len(L)), n)
    fr = np.concatenate([np.linspace(0, 1, k) for k in n])
    sx = ax[ids] + fr * (bx - ax)[ids]
    sy = ay[ids] + fr * (by - ay)[ids]
    return cKDTree(np.c_[sx, sy]), ids


def reference_labels(g, route):
    """Monotonic alignment of GPS points to the route arc sequence (DP, min total distance).
    Gives one reference arc per observation that respects the route order."""
    px, py = g["px"], g["py"]
    r = np.asarray(route)
    C, _ = point_seg(px[:, None], py[:, None], g["ax"][r][None], g["ay"][r][None],
                     g["bx"][r][None], g["by"][r][None])
    n, m = C.shape
    D = C[0].copy()
    back = np.zeros((n, m), dtype=np.int32)
    for i in range(1, n):
        # best predecessor index k <= j  (running argmin)
        idx = np.arange(m)
        best = np.minimum.accumulate(D)
        new_min = np.r_[True, D[1:] < best[:-1]]
        back[i] = np.maximum.accumulate(np.where(new_min, idx, 0))
        D = C[i] + best
    j = int(np.argmin(D))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return r[path], C[np.arange(n), path]


def candidate_table(trip, k=K_CANDIDATES, with_labels=True):
    g = build_geometry(trip)
    tree, ids = arc_kdtree(g)
    px, py = g["px"], g["py"]
    n = len(px)
    _, nn = tree.query(np.c_[px, py], k=min(SEARCH_POOL, len(ids)))

    # GPS kinematics
    t = g["track"].t.to_numpy()
    dxp = np.gradient(px); dyp = np.gradient(py)
    heading = np.arctan2(dyp, dxp)
    dt = np.gradient(t); dt[dt == 0] = 1
    speed = np.hypot(dxp, dyp) / dt

    arc_head = np.arctan2(g["by"] - g["ay"], g["bx"] - g["ax"])
    arc_len = np.hypot(g["bx"] - g["ax"], g["by"] - g["ay"])
    out_deg = np.bincount(g["u"], minlength=g["n_nodes"])[g["v"]]
    in_deg = np.bincount(g["v"], minlength=g["n_nodes"])[g["u"]]

    if with_labels and trip["route"] is not None:
        ref, ref_dist = reference_labels(g, trip["route"])
    else:
        ref, ref_dist = np.full(n, -1), np.full(n, np.nan)

    rows = []
    for i in range(n):
        cand = pd.unique(ids[nn[i]])
        d, f = point_seg(px[i], py[i], g["ax"][cand], g["ay"][cand], g["bx"][cand], g["by"][cand])
        o = np.argsort(d)[:k]
        cand, d, f = cand[o], d[o], f[o]
        hd = np.abs((heading[i] - arc_head[cand] + np.pi) % (2 * np.pi) - np.pi)
        ip, inx = max(i - 1, 0), min(i + 1, n - 1)
        dprev, _ = point_seg(px[ip], py[ip], g["ax"][cand], g["ay"][cand], g["bx"][cand], g["by"][cand])
        dnext, _ = point_seg(px[inx], py[inx], g["ax"][cand], g["ay"][cand], g["bx"][cand], g["by"][cand])
        dmin = d.min()
        rows.append(pd.DataFrame({
            "trip": trip["id"], "obs": i, "arc": cand,
            "dist": d, "dist_rank": np.arange(len(cand)), "dist_minus_min": d - dmin,
            "dist_ratio_min": d / (dmin + 1.0), "heading_diff": hd, "heading_diff_cos": np.cos(hd),
            "proj_frac": f, "arc_len": arc_len[cand], "speed": speed[i],
            "n_cand_30m": int((d < 30).sum()), "dist_prev": dprev, "dist_next": dnext,
            "out_degree": out_deg[cand], "in_degree": in_deg[cand],
            "ref_arc": ref[i], "ref_dist": ref_dist[i], "label": (cand == ref[i]).astype(int),
        }))
    return pd.concat(rows, ignore_index=True), g
