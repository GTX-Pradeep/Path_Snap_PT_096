"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn main:app --host 0.0.0.0 --port 8000      (from the api/ folder)
Env:  PATHSNAP_EVAL_DIR = folder with the evaluation trips (default: ../data/evaluator_test)

Endpoints
  GET  /health            -> status + loaded model + decoder parameters
  GET  /trips             -> the evaluation trip ids (+ number of GPS points); no reference data
  POST /predict           -> body {"trip_id": "00000059"}: runs the full pipeline on the server
                             (candidates -> features -> classifier -> HMM/Viterbi), then scores the
                             prediction against the server-side reference route.
                             Returns GPS points, predicted route geometry and metrics.
The reference route is read only by the scoring step, never by the model, and is never returned.
"""
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_loader import clean_track, load_trip  # noqa: E402
from pipeline import PathSnapPredictor, arc_coords, score  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "evaluator_test")).resolve()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = PathSnapPredictor()
log.info("Loaded model %s | HMM params %s | evaluation trips from %s",
         predictor.cfg["model_name"], predictor.cfg["hmm_params"], EVAL_DIR)


def trip_ids():
    if not EVAL_DIR.exists():
        return []
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_eval_trip(trip_id):
    if trip_id not in trip_ids():
        raise HTTPException(404, f"unknown trip id {trip_id}")
    trip = load_trip(EVAL_DIR.name, trip_id, base=EVAL_DIR.parent)
    reference = trip.pop("route")   # kept server-side for scoring only
    trip["route"] = None
    return trip, reference


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg["model_name"], "hmm_params": predictor.cfg["hmm_params"],
            "n_trips": len(trip_ids())}


@app.get("/trips")
def trips():
    out = []
    for t in trip_ids():
        n = sum(1 for _ in open(EVAL_DIR / t / f"{t}.track"))
        out.append({"trip_id": t, "n_gps": n})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    log.info("POST /predict trip=%s received", req.trip_id)
    trip, reference = load_eval_trip(req.trip_id)
    pred = predictor.predict(trip)
    metrics = score(pred, reference)
    tr = clean_track(trip["track"])
    total = time.time() - t0
    m = metrics["hmm"]
    log.info("POST /predict trip=%s | obs=%d | HMM accuracy=%.4f (classifier-only %.4f) | continuity=%.4f "
             "| route F1=%.4f | %.2fs", req.trip_id, m["n_obs"], m["pointwise_accuracy"],
             metrics["classifier_only"]["pointwise_accuracy"], m["route_continuity"], m["route_f1"], total)
    return {
        "trip_id": req.trip_id,
        "model": predictor.cfg["model_name"],
        "n_gps": int(len(tr)),
        "gps": tr[["lat", "lon"]].round(6).to_numpy().tolist(),
        "predicted_route": arc_coords(trip, pred["route"]),
        "predicted_route_arc_ids": pred["route"].tolist(),
        "classifier_only_arcs": arc_coords(trip, sorted(set(pred["clf_arcs"].tolist()))),
        "metrics": metrics,
        "timing_s": {**{k: round(v, 3) for k, v in pred["timing_s"].items()}, "request_total": round(total, 3)},
    }


# optional: also serve the UI from the API laptop (for single-machine testing)
if (ROOT / "ui").exists():
    app.mount("/ui", StaticFiles(directory=ROOT / "ui", html=True), name="ui")
