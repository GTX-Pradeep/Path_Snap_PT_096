"""HMM + Viterbi route reconstruction on top of the candidate classifier.

Hidden states   : for GPS observation i, the K (=8) candidate arcs generated for it
                  (state j at time i  <->  arc cand[i, j]; the correspondence is never lost).
Emission        : log P(candidate is the true arc) from the classifier, weighted by W_EMIT.
Transition      : a vehicle moving between consecutive points travels along the network, so the
                  network distance between the two map-matched positions should be close to the
                  straight-line GPS step:
                      route_d = rest of arc a + shortest path(end of a -> start of b) + part of arc b
                      (same arc: signed (frac_b - frac_a) * len_a, so moving backwards is penalised)
                      log T(a -> b) = -| route_d - gps_d | / BETA
                  Shortest paths: Dijkstra on the directed road graph (arc length weights), limited
                  to MAX_ROUTE_M.
Disconnected    : b unreachable from a within MAX_ROUTE_M -> log T = UNREACHABLE (a large finite
                  penalty, not -inf), so the decoder can still "break" if the classifier is very sure,
                  and a path always exists.
GPS gaps        : if the time gap dt > GAP_S the chain is not constrained (log T = 0 for all pairs):
                  after a long dropout any road may be reached.
Missing cands   : observations with fewer than K candidates are padded with -inf emissions.
Decoding        : standard Viterbi recurrence  delta_{i+1}(b) = max_a [delta_i(a) + log T(a,b)] + w*log e_{i+1}(b)
                  with back-pointers; vectorised K x K per step.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DEFAULT_PARAMS = dict(beta=10.0, w_emit=1.0, gap_s=30.0, max_route_m=1500.0, unreachable=-30.0)


def road_graph(g):
    """Directed graph u -> v weighted by arc length (parallel arcs: keep the shortest)."""
    L = np.hypot(g["bx"] - g["ax"], g["by"] - g["ay"])
    e = pd.DataFrame({"u": g["u"], "v": g["v"], "w": np.maximum(L, 1e-3)}).groupby(["u", "v"]).w.min().reset_index()
    n = g["n_nodes"]
    return csr_matrix((e.w, (e.u, e.v)), shape=(n, n)), L


def candidate_arrays(c, k=8):
    """c: candidate rows of one trip (obs, arc, proj_frac, proba) -> padded [n, k] arrays."""
    c = c.sort_values(["obs", "dist_rank"]) if "dist_rank" in c else c.sort_values("obs")
    obs = np.sort(c.obs.unique())
    n = len(obs)
    arcs = np.full((n, k), -1, dtype=np.int64)
    frac = np.zeros((n, k)); prob = np.zeros((n, k))
    pos = c.groupby("obs").cumcount().to_numpy()
    row = np.searchsorted(obs, c.obs.to_numpy())
    keep = pos < k
    arcs[row[keep], pos[keep]] = c.arc.to_numpy()[keep]
    frac[row[keep], pos[keep]] = c.proj_frac.to_numpy()[keep]
    prob[row[keep], pos[keep]] = c.proba.to_numpy()[keep]
    return obs, arcs, frac, prob


def transition_mismatch(g, obs, arcs, frac, max_route_m):
    """|route distance - gps distance| for every step and candidate pair: array [n-1, k, k] (inf = unreachable)."""
    G, L = road_graph(g)
    valid = arcs >= 0
    A = np.unique(arcs[valid])
    src = np.unique(g["v"][A]); dst = np.unique(g["u"][A])
    D = np.empty((len(src), len(dst)), dtype=np.float32)
    for i in range(0, len(src), 256):  # chunked Dijkstra, keep only the needed columns
        D[i:i + 256] = dijkstra(G, indices=src[i:i + 256], limit=max_route_m)[:, dst]
    px, py = g["px"][obs], g["py"][obs]
    gps = np.hypot(np.diff(px), np.diff(py))
    a = np.where(valid, arcs, A[0])                   # padded slots -> any real arc (masked later)
    rs = np.searchsorted(src, g["v"][a])              # [n, k] row index of arc end node
    cd = np.searchsorted(dst, g["u"][a])              # [n, k] col index of arc start node
    la = L[a]
    net = D[rs[:-1, :, None], cd[1:, None, :]]        # [n-1, k, k]
    route = (1 - frac[:-1, :, None]) * la[:-1, :, None] + net + frac[1:, None, :] * la[1:, None, :]
    same = arcs[:-1, :, None] == arcs[1:, None, :]
    # same arc: SIGNED progress along the arc. Moving backwards gives a negative route distance and
    # is penalised, otherwise the decoder can "drive" the opposite-direction twin arc for free.
    route = np.where(same, (frac[1:, None, :] - frac[:-1, :, None]) * la[:-1, :, None], route)
    mism = np.abs(route - gps[:, None, None]).astype(np.float32)
    bad = ~(valid[:-1, :, None] & valid[1:, None, :])
    mism[bad] = np.inf
    return mism


def viterbi(prob, valid, mism, dt, beta, w_emit, gap_s, unreachable, **_):
    n, k = prob.shape
    loge = np.where(valid, w_emit * np.log(np.clip(prob, 1e-6, 1.0)), -np.inf)
    back = np.zeros((n, k), dtype=np.int16)
    delta = loge[0].copy()
    for i in range(n - 1):
        if dt[i] > gap_s:
            logT = np.zeros((k, k))
        else:
            logT = np.where(np.isfinite(mism[i]), -mism[i] / beta, unreachable)
        logT = np.where(valid[i][:, None] & valid[i + 1][None, :], logT, -np.inf)
        s = delta[:, None] + logT
        back[i + 1] = np.argmax(s, axis=0)
        delta = s[back[i + 1], np.arange(k)] + loge[i + 1]
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


class TripDecoder:
    """Precomputes the parameter-independent part once, so decoder parameters can be tuned cheaply."""

    def __init__(self, g, cands, params=None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.g = g
        self.obs, self.arcs, self.frac, self.prob = candidate_arrays(cands)
        self.valid = self.arcs >= 0
        t = g["track"].t.to_numpy()[self.obs]
        self.dt = np.diff(t)
        self.mism = transition_mismatch(g, self.obs, self.arcs, self.frac, self.p["max_route_m"])

    def classifier_only(self):
        return self.arcs[np.arange(len(self.obs)), np.argmax(np.where(self.valid, self.prob, -1), axis=1)]

    def decode(self, **override):
        p = {**self.p, **override}
        path = viterbi(self.prob, self.valid, self.mism, self.dt, **p)
        return self.arcs[np.arange(len(self.obs)), path]


# ---------------------------------------------------------------- metrics
def route_metrics(g, pred, ref=None, route=None):
    """Pointwise accuracy (vs per-observation reference arc), route continuity, route-level P/R/F1."""
    u, v = g["u"], g["v"]
    same = pred[1:] == pred[:-1]
    linked = v[pred[:-1]] == u[pred[1:]]
    out = {
        "n_obs": int(len(pred)),
        "route_continuity": float(np.mean(same | linked)) if len(pred) > 1 else 1.0,
        "n_breaks": int(np.sum(~(same | linked))),
    }
    if ref is not None:
        out["pointwise_accuracy"] = float(np.mean(pred == ref))
    if route is not None:
        seq = pred[np.r_[True, pred[1:] != pred[:-1]]]
        P, R = set(seq.tolist()), set(np.asarray(route).tolist())
        tp = len(P & R)
        prec, rec = tp / max(len(P), 1), tp / max(len(R), 1)
        out.update(route_precision=prec, route_recall=rec,
                   route_f1=0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return out


def predicted_sequence(pred):
    """Collapse consecutive duplicates -> ordered list of traversed arcs."""
    pred = np.asarray(pred)
    return pred[np.r_[True, pred[1:] != pred[:-1]]]
