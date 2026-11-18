"""Folium map: raw GPS, nearby road network, reference route (and predicted route if given)."""
import sys
from pathlib import Path

import folium

from data_loader import load_trip, clean_track

OUT = Path(__file__).resolve().parents[1] / "maps"


def arc_line(trip, a):
    n, arcs = trip["nodes"], trip["arcs"]
    u, v = arcs.u[a], arcs.v[a]
    return [(n.lat[u], n.lon[u]), (n.lat[v], n.lon[v])]


def trip_map(part, tid, predicted=None):
    trip = load_trip(part, tid)
    tr = clean_track(trip["track"])
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")

    net = folium.FeatureGroup("Road network (sample)", show=False)
    for a in range(0, len(trip["arcs"]), 3):
        folium.PolyLine(arc_line(trip, a), color="#999", weight=1).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup("Raw GPS")
    for lat, lon in zip(tr.lat[::5], tr.lon[::5]):
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True).add_to(gps)
    gps.add_to(m)

    if trip["route"] is not None:
        ref = folium.FeatureGroup("Reference route")
        for a in trip["route"]:
            folium.PolyLine(arc_line(trip, a), color="green", weight=5, opacity=0.7).add_to(ref)
        ref.add_to(m)
    if predicted is not None:
        pr = folium.FeatureGroup("Predicted route")
        for a in predicted:
            folium.PolyLine(arc_line(trip, a), color="red", weight=3).add_to(pr)
        pr.add_to(m)
    folium.LayerControl().add_to(m)
    return m


if __name__ == "__main__":
    part, tid = (sys.argv[1:3] if len(sys.argv) > 2 else ("train", "00000010"))
    trip_map(part, tid).save(OUT / f"trip_{part}_{tid}.html")
    print("saved", OUT / f"trip_{part}_{tid}.html")
