"""Folium maps of predicted routes on student_test trips (+ static PNGs for the report).

Layers: raw GPS (blue dots), road network near the trip (grey), reference route (green),
classifier-only arcs (orange dashed), classifier + HMM predicted route (red), with a legend and
layer control. Uses the final model's student_test probabilities and the tuned decoder.

Usage: python route_maps.py [trip ids...]   (default: best, median and worst trip by HMM accuracy)
Outputs: maps/route_student_test_<trip>.html, results/hmm/route_<trip>.png
"""
import json
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import clean_track, load_trip
from features import build_geometry
from hmm import TripDecoder, predicted_sequence, route_metrics

ROOT = Path(__file__).resolve().parents[1]
RES, MAPS = ROOT / "results", ROOT / "maps"
CFG = json.load(open(ROOT / "models" / "final_config.json"))

LEGEND = """<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;
border:1px solid #bbb;border-radius:6px;font:13px sans-serif;line-height:1.7">
<b>Trip {trip} (student_test)</b><br>
<span style="color:#1c7ed6">&#9679;</span> Raw GPS<br>
<span style="color:#2b8a3e">&#9473;&#9473;</span> Reference route<br>
<span style="color:#f59f00">&#9476;&#9476;</span> Classifier-only arcs ({clf:.1%})<br>
<span style="color:#d6336c">&#9473;&#9473;</span> Predicted route, HMM ({hmm:.1%})<br>
<span style="color:#999">&#9472;&#9472;</span> Road network (toggle)</div>"""


def decode_trip(t, cands):
    trip = load_trip("student_test", t)
    g = build_geometry(trip)
    c = cands[cands.trip == t]
    d = TripDecoder(g, c, CFG["hmm_params"])
    ref = c.groupby("obs").ref_arc.first().loc[d.obs].to_numpy()
    clf, hmm = d.classifier_only(), d.decode()
    return trip, g, clf, hmm, route_metrics(g, clf, ref), route_metrics(g, hmm, ref)


def seg(trip, a):
    n, arcs = trip["nodes"], trip["arcs"]
    u, v = arcs.u[a], arcs.v[a]
    return [(n.lat[u], n.lon[u]), (n.lat[v], n.lon[v])]


def folium_map(t, trip, clf, hmm, mc, mh):
    tr = clean_track(trip["track"])
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    n, arcs = trip["nodes"], trip["arcs"]
    # only road segments whose midpoint is within ~100 m of a GPS point (keeps the HTML small)
    from scipy.spatial import cKDTree
    k = np.cos(np.radians(tr.lat.mean()))
    tree = cKDTree(np.c_[tr.lat, tr.lon * k])
    mid = np.c_[(n.lat.to_numpy()[arcs.u] + n.lat.to_numpy()[arcs.v]) / 2,
                (n.lon.to_numpy()[arcs.u] + n.lon.to_numpy()[arcs.v]) / 2 * k]
    near = tree.query(mid, distance_upper_bound=0.0009)[0] < np.inf
    net = folium.FeatureGroup("Road network (near trip)", show=False)
    for a in np.where(near)[0]:
        folium.PolyLine(seg(trip, a), color="#999", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route")
    for a in trip["route"]:
        folium.PolyLine(seg(trip, a), color="#2b8a3e", weight=8, opacity=0.45).add_to(ref)
    ref.add_to(m)
    fc = folium.FeatureGroup("Classifier-only arcs")
    for a in sorted(set(clf.tolist())):
        folium.PolyLine(seg(trip, a), color="#f59f00", weight=3, dash_array="6 6").add_to(fc)
    fc.add_to(m)
    fh = folium.FeatureGroup("Predicted route (HMM)")
    for a in predicted_sequence(hmm):
        folium.PolyLine(seg(trip, a), color="#d6336c", weight=4, opacity=0.9).add_to(fh)
    fh.add_to(m)
    gps = folium.FeatureGroup("Raw GPS")
    step = max(1, len(tr) // 1500)
    for lat, lon in zip(tr.lat[::step], tr.lon[::step]):
        folium.CircleMarker((lat, lon), radius=2, color="#1c7ed6", fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(LEGEND.format(trip=t, clf=mc["pointwise_accuracy"], hmm=mh["pointwise_accuracy"])))
    m.save(MAPS / f"route_student_test_{t}.html")


def static_png(t, trip, clf, hmm, mc, mh):
    n, arcs, tr = trip["nodes"], trip["arcs"], clean_track(trip["track"])
    def lines(a_list):
        a_list = np.asarray(list(a_list))
        return (np.c_[n.lon[arcs.u[a_list]], n.lon[arcs.v[a_list]], np.full(len(a_list), np.nan)].ravel(),
                np.c_[n.lat[arcs.u[a_list]], n.lat[arcs.v[a_list]], np.full(len(a_list), np.nan)].ravel())
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for a, pred, name, mm in [(ax[0], clf, "Classifier only", mc), (ax[1], hmm, "Classifier + HMM", mh)]:
        a.plot(*lines(range(len(arcs))), color="#ddd", lw=0.5)
        a.plot(*lines(trip["route"]), color="#2b8a3e", lw=5, alpha=0.4, label="reference route")
        a.plot(*lines(sorted(set(pred.tolist()))), color="#d6336c" if "HMM" in name else "#f59f00", lw=1.8, label="predicted arcs")
        a.scatter(tr.lon, tr.lat, s=1, color="#1c7ed6", label="GPS")
        a.set_title(f"{name}: accuracy {mm['pointwise_accuracy']:.1%}, continuity {mm['route_continuity']:.1%}, breaks {mm['n_breaks']}")
        a.set_xlim(tr.lon.min() - 0.003, tr.lon.max() + 0.003); a.set_ylim(tr.lat.min() - 0.002, tr.lat.max() + 0.002)
        a.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Trip {t} (student_test)")
    plt.tight_layout(); plt.savefig(RES / "hmm" / f"route_{t}.png", dpi=110); plt.close()


if __name__ == "__main__":
    cands = pd.read_parquet(RES / "candidates_student_test.parquet")
    pr = pd.read_parquet(RES / f"pred_test_{CFG['model_name']}.parquet")
    cands["proba"] = pr.proba.to_numpy()
    trips = sys.argv[1:]
    if not trips:
        pt = pd.read_csv(RES / "hmm" / "per_trip.csv")
        pt = pt[(pt.model == CFG["model_name"]) & (pt.method == "hmm")].sort_values("pointwise_accuracy")
        trips = [f"{int(x):08d}" for x in [pt.trip.iloc[-1], pt.trip.iloc[len(pt) // 2], pt.trip.iloc[0]]]
    for t in trips:
        trip, g, clf, hmm, mc, mh = decode_trip(t, cands)
        folium_map(t, trip, clf, hmm, mc, mh)
        static_png(t, trip, clf, hmm, mc, mh)
        print(f"{t}: classifier {mc['pointwise_accuracy']:.4f} -> HMM {mh['pointwise_accuracy']:.4f}  saved map + png")
