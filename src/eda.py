"""Dataset exploration: inventory, GPS quality, sampling gaps, road-network connectivity, plots.

Outputs go to results/eda/ (CSV tables + PNG plots) and maps/ (Folium maps).
Partitions analysed: train (80 trips) and student_test (10 trips).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from data_loader import DATA_DIR, list_trips, load_trip, project

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def trip_stats(part, tid):
    trip = load_trip(part, tid)
    tr, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    d = DATA_DIR / part / tid
    dt = np.diff(tr.t.to_numpy())
    x, y = project(tr.lon, tr.lat, tr.lon.mean(), tr.lat.mean())
    step = np.hypot(np.diff(x), np.diff(y))
    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs.u, arcs.v)), shape=(n, n))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs.u, minlength=n) + np.bincount(arcs.v, minlength=n)
    route_ok = None
    if route is not None:  # are consecutive route arcs connected head-to-tail?
        u, v = arcs.u.to_numpy(), arcs.v.to_numpy()
        route_ok = float(np.mean(v[route[:-1]] == u[route[1:]])) if len(route) > 1 else 1.0
    return {
        "partition": part, "trip": tid,
        "track_kb": (d / f"{tid}.track").stat().st_size / 1024,
        "n_gps": len(tr), "n_nodes": n, "n_arcs": len(arcs),
        "route_len": None if route is None else len(route),
        "missing_values": int(tr.isna().sum().sum()),
        "invalid_coords": int((~tr.lon.between(-180, 180) | ~tr.lat.between(-90, 90)).sum()),
        "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0]),
        "non_monotonic_t": int((dt <= 0).sum()),
        "pct_dt_eq_1s": float(np.mean(dt == 1) * 100),
        "max_gap_s": float(dt.max()), "n_gaps_gt_5s": int((dt > 5).sum()),
        "max_jump_m": float(step.max()), "track_length_km": float(step.sum() / 1000),
        "mean_speed_kmh": float(step.sum() / max(tr.t.iloc[-1] - tr.t.iloc[0], 1) * 3.6),
        "lon_min": tr.lon.min(), "lon_max": tr.lon.max(), "lat_min": tr.lat.min(), "lat_max": tr.lat.max(),
        "weak_components": n_weak, "largest_weak_frac": float(np.bincount(lab_w).max() / n),
        "strong_components": n_strong, "largest_strong_frac": float(np.bincount(lab_s).max() / n),
        "isolated_nodes": int((deg == 0).sum()),
        "route_consecutive_connected": route_ok,
    }


def plots(inv):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
        s = inv[inv.partition == part]
        ax[0].hist(s.n_gps, bins=20, alpha=.6, label=part, color=c)
        ax[1].hist(s.max_gap_s, bins=20, alpha=.6, label=part, color=c)
        ax[2].hist(s.mean_speed_kmh, bins=20, alpha=.6, label=part, color=c)
    for a, t in zip(ax, ["GPS observations per trip", "Largest time gap per trip (s)", "Mean speed per trip (km/h)"]):
        a.set_title(t); a.legend()
    plt.tight_layout(); plt.savefig(OUT / "trip_distributions.png", dpi=120); plt.close()

    # sampling intervals + GPS coordinates over all train trips
    dts, fig = [], plt.figure(figsize=(7, 6))
    for tid in list_trips("train"):
        tr = load_trip("train", tid)["track"]
        dts.append(np.diff(tr.t.to_numpy()))
        plt.plot(tr.lon, tr.lat, lw=.6)
    plt.title("All training GPS trajectories (lon/lat)"); plt.xlabel("longitude"); plt.ylabel("latitude")
    plt.savefig(OUT / "train_trajectories.png", dpi=120); plt.close()
    dts = np.concatenate(dts)
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(dts, 0, 10), bins=np.arange(0, 11.5, .5))
    plt.yscale("log"); plt.title("Sampling interval (s, clipped at 10) — train")
    plt.savefig(OUT / "sampling_intervals.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(inv.n_arcs, inv.n_nodes, c=(inv.partition == "train"), cmap="coolwarm", s=15)
    plt.xlabel("# arcs (road segments)"); plt.ylabel("# nodes"); plt.title("Road-network size per trip")
    plt.savefig(OUT / "network_size.png", dpi=120); plt.close()
    return pd.Series(dts).describe()


def road_network_plot(part, tid):
    trip = load_trip(part, tid)
    n, a, tr = trip["nodes"], trip["arcs"], trip["track"]
    plt.figure(figsize=(8, 8))
    xs = np.c_[n.lon[a.u], n.lon[a.v], np.full(len(a), np.nan)].ravel()
    ys = np.c_[n.lat[a.u], n.lat[a.v], np.full(len(a), np.nan)].ravel()
    plt.plot(xs, ys, color="lightgrey", lw=.5, label="road network")
    if trip["route"] is not None:
        r = trip["route"]
        plt.plot(np.c_[n.lon[a.u[r]], n.lon[a.v[r]], np.full(len(r), np.nan)].ravel(),
                 np.c_[n.lat[a.u[r]], n.lat[a.v[r]], np.full(len(r), np.nan)].ravel(),
                 color="green", lw=2.5, label="reference route")
    plt.scatter(tr.lon, tr.lat, s=1, color="blue", label="GPS")
    plt.legend(); plt.title(f"Trip {tid} ({part})")
    plt.savefig(OUT / f"road_network_{part}_{tid}.png", dpi=120); plt.close()


if __name__ == "__main__":
    rows = [trip_stats(p, t) for p in ["train", "student_test"] for t in list_trips(p)]
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    summary = inv.groupby("partition")[["n_gps", "n_nodes", "n_arcs", "route_len", "missing_values",
                                         "pct_dt_eq_1s", "max_gap_s", "n_gaps_gt_5s", "max_jump_m",
                                         "track_length_km", "mean_speed_kmh", "weak_components",
                                         "isolated_nodes", "route_consecutive_connected"]].describe().T
    summary.to_csv(OUT / "summary_by_partition.csv")
    print(inv.groupby("partition").size().rename("trips"))
    print(summary.round(2).to_string())
    print("\nSampling interval stats (train):\n", plots(inv))
    road_network_plot("train", "00000010")
    anomalies = inv.sort_values("max_gap_s", ascending=False).head(10)[["partition", "trip", "max_gap_s", "max_jump_m", "n_gaps_gt_5s"]]
    anomalies.to_csv(OUT / "anomalies_top10.csv", index=False)
    print("\nTrips with the largest GPS gaps:\n", anomalies.to_string(index=False))
