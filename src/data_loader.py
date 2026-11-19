"""Raw data loading for PathSnap trips.

Each trip folder has: <id>.track (lon lat t), <id>.nodes (lon lat),
<id>.arcs (from to), <id>.route (arc ids, chronological).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", Path(__file__).resolve().parents[1] / "data"))
EARTH_R = 6371000.0


def list_trips(partition):
    d = DATA_DIR / partition
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def load_trip(partition, trip_id, base=None):
    """base: optional root folder (default DATA_DIR); the API passes the evaluation folder's parent."""
    d = Path(base or DATA_DIR) / partition / trip_id
    track = pd.read_csv(d / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(d / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(d / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["u", "v"])
    route_file = d / f"{trip_id}.route"
    route = None
    if route_file.exists():
        route = pd.read_csv(route_file, header=None, names=["arc"])["arc"].to_numpy()
    return {"id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}


def clean_track(track):
    """Drop NaN / implausible coordinates, sort by time, drop duplicate timestamps."""
    t = track.dropna()
    t = t[(t.lon.between(-180, 180)) & (t.lat.between(-90, 90))]
    t = t.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    return t


def project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres (fine at city scale)."""
    x = np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * EARTH_R
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y
