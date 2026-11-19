"""End-to-end inference pipeline used by the API.

predict(trip)  : raw trip (track + nodes + arcs, NO route) -> candidates -> features -> classifier
                 probabilities -> HMM/Viterbi -> one arc per GPS point + ordered route.
score(...)     : server-side only. Compares a prediction with the reference route (.route) of the
                 trip; the reference never enters predict().
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from features import candidate_table, reference_labels
from hmm import TripDecoder, predicted_sequence, route_metrics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


class PathSnapPredictor:
    def __init__(self, config_path=MODELS / "final_config.json"):
        self.cfg = json.load(open(config_path))
        self.model = joblib.load(MODELS / self.cfg["model_file"])
        self.features = self.cfg["features"]

    def predict(self, trip):
        assert trip.get("route") is None, "reference route must not enter the prediction pipeline"
        t0 = time.time()
        cands, g = candidate_table(trip, k=self.cfg["k_candidates"], with_labels=False)
        t1 = time.time()
        cands["proba"] = self.model.predict_proba(cands[self.features].to_numpy())[:, 1]
        t2 = time.time()
        dec = TripDecoder(g, cands, self.cfg["hmm_params"])
        hmm_arcs = dec.decode()
        clf_arcs = dec.classifier_only()
        t3 = time.time()
        return {
            "g": g, "obs": dec.obs, "hmm_arcs": hmm_arcs, "clf_arcs": clf_arcs,
            "route": predicted_sequence(hmm_arcs),
            "timing_s": {"candidates_features": t1 - t0, "classifier": t2 - t1,
                         "hmm_viterbi": t3 - t2, "total": t3 - t0},
        }


def score(pred, reference_route):
    """Server-side evaluation against the reference route (never sent to the model or the UI)."""
    g = pred["g"]
    ref_all, _ = reference_labels(g, reference_route)      # per-observation reference arc
    ref = ref_all[pred["obs"]]
    out = {"hmm": route_metrics(g, pred["hmm_arcs"], ref, reference_route),
           "classifier_only": route_metrics(g, pred["clf_arcs"], ref, reference_route)}
    return out


def arc_coords(trip, arcs):
    """[[ [lat, lon], [lat, lon] ], ...] for drawing arcs on a map."""
    n, a = trip["nodes"], trip["arcs"]
    u, v = a.u.to_numpy()[arcs], a.v.to_numpy()[arcs]
    return np.stack([np.c_[n.lat.to_numpy()[u], n.lon.to_numpy()[u]],
                     np.c_[n.lat.to_numpy()[v], n.lon.to_numpy()[v]]], axis=1).round(6).tolist()
