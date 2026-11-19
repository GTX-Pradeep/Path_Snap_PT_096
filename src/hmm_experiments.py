"""HMM / Viterbi experiments.

1. Out-of-fold (OOF) probabilities on the training trips for the final classifier
   (5-fold GroupKFold: each trip is scored by a model that never saw it), so the decoder is
   tuned on honest probabilities, not on over-confident in-sample ones.
2. Decoder parameter search (grid over BETA x W_EMIT) on 20 training trips, objective =
   pointwise accuracy (training observations with a reliable reference only).
3. Student-test comparison, for EVERY trained classifier: classifier-only (argmax per GPS point)
   vs classifier + HMM/Viterbi: pointwise accuracy, route continuity, breaks, route P/R/F1.
4. Writes models/final_config.json (final model + decoder parameters) used by the API.

Outputs: results/hmm/{decoder_tuning.csv, comparison.csv, per_trip.csv, *.png}
"""
import itertools
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from data_loader import load_trip
from features import FEATURES, K_CANDIDATES, build_geometry
from hmm import DEFAULT_PARAMS, TripDecoder, route_metrics
from model_zoo.lgbm import make_lgbm
from train_models import MAX_REF_DIST_M, load_tables

ROOT = Path(__file__).resolve().parents[1]
RES, OUT, MOD = ROOT / "results", ROOT / "results" / "hmm", ROOT / "models"
OUT.mkdir(parents=True, exist_ok=True)
FINAL_MODEL = "lgbm_tuned" if (MOD / "lgbm_tuned.joblib").exists() else "lgbm"
GRID = dict(beta=[0.5, 1, 2, 5, 10, 20, 40], w_emit=[0.25, 0.5, 1, 2, 4])
N_TUNE_TRIPS = 20
SEED = 42


def final_params():
    p = RES / "bayes_opt" / "best_params.json"
    return json.load(open(p))["best_params"] if FINAL_MODEL == "lgbm_tuned" and p.exists() else {}


def oof_probabilities(tr_all):
    tr_clean = tr_all[tr_all.ref_dist <= MAX_REF_DIST_M]
    proba = np.zeros(len(tr_all))
    trips = tr_all.trip.to_numpy()
    for a, b in GroupKFold(5).split(tr_all, groups=trips):
        fit_trips = set(trips[a])
        fit = tr_clean[tr_clean.trip.isin(fit_trips)]
        m = make_lgbm(**final_params()).fit(fit[FEATURES].to_numpy(), fit.label.to_numpy())
        proba[b] = m.predict_proba(tr_all.iloc[b][FEATURES].to_numpy())[:, 1]
    return proba


def tune_decoder():
    tr_all = pd.read_parquet(RES / "candidates_train.parquet")
    t0 = time.time()
    tr_all["proba"] = oof_probabilities(tr_all)
    print(f"OOF probabilities: {time.time() - t0:.0f}s", flush=True)
    # tuning trips: reliable reference (median ref distance <= 50 m), moderate length, fixed seed
    stats = tr_all.groupby("trip").agg(n=("obs", "nunique"), refd=("ref_dist", "median"))
    pool = stats[(stats.refd <= MAX_REF_DIST_M) & (stats.n <= 6000)].index.to_numpy()
    tune = np.random.default_rng(SEED).choice(pool, N_TUNE_TRIPS, replace=False)
    decs = []
    for t in tune:
        c = tr_all[tr_all.trip == t]
        trip = load_trip("train", t)
        d = TripDecoder(build_geometry(trip), c)
        r = c.groupby("obs").agg(ref=("ref_arc", "first"), rd=("ref_dist", "first")).loc[d.obs]
        decs.append((d, r.ref.to_numpy(), (r.rd <= MAX_REF_DIST_M).to_numpy()))
    rows = []
    base = np.mean(np.concatenate([(d.classifier_only() == ref)[ok] for d, ref, ok in decs]))
    for beta, w in itertools.product(GRID["beta"], GRID["w_emit"]):
        acc = np.mean(np.concatenate([(d.decode(beta=beta, w_emit=w) == ref)[ok] for d, ref, ok in decs]))
        rows.append(dict(beta=beta, w_emit=w, pointwise_accuracy=acc))
        print(f"  beta={beta:>3} w_emit={w:<3} acc={acc:.4f}", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "decoder_tuning.csv", index=False)
    best = grid.loc[grid.pointwise_accuracy.idxmax()]
    piv = grid.pivot(index="beta", columns="w_emit", values="pointwise_accuracy")
    plt.figure(figsize=(6, 4))
    plt.imshow(piv.values, cmap="viridis", aspect="auto")
    for (i, j), v in np.ndenumerate(piv.values):
        plt.text(j, i, f"{v:.3f}", ha="center", va="center", color="white", fontsize=8)
    plt.xticks(range(len(piv.columns)), piv.columns); plt.yticks(range(len(piv.index)), piv.index)
    plt.xlabel("w_emit (emission weight)"); plt.ylabel("beta (transition scale, m)")
    plt.title(f"Decoder tuning on {N_TUNE_TRIPS} train trips (OOF)\nclassifier-only = {base:.3f}")
    plt.tight_layout(); plt.savefig(OUT / "decoder_tuning.png", dpi=120); plt.close()
    return {"beta": float(best.beta), "w_emit": float(best.w_emit)}, list(map(str, tune)), float(base)


def compare_on_student_test(params):
    te = pd.read_parquet(RES / "candidates_student_test.parquet")
    models = sorted(p.stem.replace("pred_test_", "") for p in RES.glob("pred_test_*.parquet"))
    geoms = {t: (load_trip("student_test", t)) for t in te.trip.unique()}
    rows, per_trip = [], []
    for m in models:
        pr = pd.read_parquet(RES / f"pred_test_{m}.parquet")
        assert (pr.arc.to_numpy() == te.arc.to_numpy()).all()
        c_all = te.assign(proba=pr.proba.to_numpy())
        agg = {"clf": [], "hmm": []}
        t0 = time.time()
        for t, trip in geoms.items():
            g = build_geometry(trip)
            c = c_all[c_all.trip == t]
            d = TripDecoder(g, c, params)
            ref = c.groupby("obs").ref_arc.first().loc[d.obs].to_numpy()
            for name, pred in [("clf", d.classifier_only()), ("hmm", d.decode())]:
                r = route_metrics(g, pred, ref, trip["route"])
                agg[name].append(r)
                per_trip.append(dict(model=m, trip=t, method=name, **r))
        dec_time = time.time() - t0
        for name in ["clf", "hmm"]:
            df = pd.DataFrame(agg[name])
            w = df.n_obs / df.n_obs.sum()
            rows.append(dict(model=m, method={"clf": "classifier only", "hmm": "classifier + HMM"}[name],
                             pointwise_accuracy=float((df.pointwise_accuracy * w).sum()),
                             route_continuity=float((df.route_continuity * w).sum()),
                             breaks=int(df.n_breaks.sum()),
                             route_precision=df.route_precision.mean(), route_recall=df.route_recall.mean(),
                             route_f1=df.route_f1.mean(),
                             decode_time_s_10_trips=dec_time if name == "hmm" else None))
        print(f"  {m}: clf={rows[-2]['pointwise_accuracy']:.4f} hmm={rows[-1]['pointwise_accuracy']:.4f} "
              f"continuity {rows[-2]['route_continuity']:.3f}->{rows[-1]['route_continuity']:.3f}", flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison.csv", index=False)
    pd.DataFrame(per_trip).to_csv(OUT / "per_trip.csv", index=False)

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, col, title in zip(ax, ["pointwise_accuracy", "route_continuity", "route_f1"],
                             ["Pointwise accuracy", "Route continuity", "Route-level F1 (arc sets)"]):
        p = comp.pivot(index="model", columns="method", values=col)[["classifier only", "classifier + HMM"]]
        p.plot.bar(ax=a, rot=0); a.set_title(f"{title} — student_test"); a.set_ylim(max(0, p.values.min() - 0.1), 1)
        a.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "classifier_vs_hmm.png", dpi=120); plt.close()
    pt = pd.DataFrame(per_trip)
    pt = pt[pt.model == FINAL_MODEL].pivot(index="trip", columns="method", values="pointwise_accuracy")
    pt.plot.bar(figsize=(9, 4), rot=45, title=f"Per-trip pointwise accuracy, {FINAL_MODEL} (student_test)")
    plt.ylim(max(0, pt.values.min() - 0.05), 1); plt.tight_layout()
    plt.savefig(OUT / "per_trip_accuracy.png", dpi=120); plt.close()
    return comp


if __name__ == "__main__":
    print("final classifier:", FINAL_MODEL)
    best, tune_trips, base = tune_decoder()
    params = {**DEFAULT_PARAMS, **best}
    print("best decoder params:", params)
    comp = compare_on_student_test(params)
    print(comp.round(4).to_string(index=False))
    cfg = {"model_file": f"{FINAL_MODEL}.joblib", "model_name": FINAL_MODEL, "features": FEATURES,
           "k_candidates": K_CANDIDATES, "hmm_params": params, "decoder_tuned_on_train_trips": tune_trips,
           "decoder_tuning_classifier_only_accuracy": base}
    if FINAL_MODEL.startswith("lgbm"):  # version-independent copy of the final model for the API
        joblib.load(MOD / f"{FINAL_MODEL}.joblib").booster_.save_model(str(MOD / f"{FINAL_MODEL}.txt"))
        cfg["booster_file"] = f"{FINAL_MODEL}.txt"
    json.dump(cfg, open(MOD / "final_config.json", "w"), indent=2)
    print("wrote models/final_config.json")
