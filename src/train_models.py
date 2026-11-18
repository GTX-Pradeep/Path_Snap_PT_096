"""Train + evaluate candidate classifiers.

Usage: python train_models.py logreg rf knn lgbm mlp   (any subset; default = all)

Metrics:
  - CV Macro F1 (candidate level, 5-fold GroupKFold by trip on the 80 training trips)
  - Student-test Macro F1, precision, recall, Log Loss (candidate level)
  - Student-test observation-level Top-1 accuracy (highest-scoring candidate == reference arc)
  - Training / inference time
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from features import FEATURES
from model_zoo import discover

SEED = 42
MODELS = discover()  # every model file in src/model_zoo/

ROOT = Path(__file__).resolve().parents[1]
RES, MOD = ROOT / "results", ROOT / "models"
RES.mkdir(exist_ok=True); MOD.mkdir(exist_ok=True)


MAX_REF_DIST_M = 50.0  # training obs whose reference arc is >50 m from the GPS point have unreliable labels


def load_tables(clean_train=True):
    tr = pd.read_parquet(RES / "candidates_train.parquet")
    te = pd.read_parquet(RES / "candidates_student_test.parquet")
    if clean_train:  # student_test is never filtered
        tr = tr[tr.ref_dist <= MAX_REF_DIST_M].reset_index(drop=True)
    return tr, te


def top1_accuracy(df, proba):
    d = df[["trip", "obs", "label"]].copy()
    d["p"] = proba
    best = d.loc[d.groupby(["trip", "obs"]).p.idxmax()]
    return float(best.label.mean())


def subsample(idx, n, rng):
    return idx if n is None or len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def cv_macro_f1(make, X, y, groups, max_rows=None, n_folds=5, verbose_name=""):
    rng = np.random.default_rng(SEED)
    scores = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        a = subsample(a, max_rows, rng)
        m = make().fit(X[a], y[a])
        scores.append(f1_score(y[b], m.predict(X[b]), average="macro"))
        if verbose_name:
            print(f"  [{verbose_name}] fold {k}: macro F1={scores[-1]:.4f}", flush=True)
    return scores


def evaluate(name, make=None, family=None, max_rows="default", tag=None, tr=None, te=None):
    spec = MODELS.get(name, {})
    make = make or spec["make"]
    family = family or spec["family"]
    max_rows = spec.get("max_train_rows") if max_rows == "default" else max_rows
    tag = tag or name
    if tr is None:
        tr, te = load_tables()
    X, y, groups = tr[FEATURES].to_numpy(), tr.label.to_numpy(), tr.trip.to_numpy()

    print(f"== {tag} ({family})", flush=True)
    cv = cv_macro_f1(make, X, y, groups, max_rows, verbose_name=tag)

    rng = np.random.default_rng(SEED)
    idx = subsample(np.arange(len(y)), max_rows, rng)
    t0 = time.time()
    model = make().fit(X[idx], y[idx])
    train_time = time.time() - t0

    Xt, yt = te[FEATURES].to_numpy(), te.label.to_numpy()
    t0 = time.time()
    proba = model.predict_proba(Xt)[:, 1]
    infer_time = time.time() - t0
    pred = (proba >= 0.5).astype(int)
    train_pred = model.predict(X[idx])

    est = model.steps[-1][1] if hasattr(model, "steps") else model
    res = {
        "model": tag, "family": family,
        "config": {k: str(v) for k, v in est.get_params().items()},
        "train_rows": int(len(idx)),
        "cv_macro_f1_folds": [float(s) for s in cv],
        "cv_macro_f1_mean": float(np.mean(cv)), "cv_macro_f1_std": float(np.std(cv)),
        "train_macro_f1": float(f1_score(y[idx], train_pred, average="macro")),
        "test_macro_f1": float(f1_score(yt, pred, average="macro")),
        "test_precision": float(precision_score(yt, pred)),
        "test_recall": float(recall_score(yt, pred)),
        "test_log_loss": float(log_loss(yt, np.clip(proba, 1e-7, 1 - 1e-7))),
        "test_top1_accuracy": top1_accuracy(te, proba),
        "test_confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "train_time_s": train_time, "inference_time_s": infer_time,
    }
    if hasattr(est, "loss_curve_"):  # epoch-wise history (MLP only)
        res["loss_curve"] = [float(v) for v in est.loss_curve_]
        res["validation_scores"] = [float(v) for v in (est.validation_scores_ or [])]

    joblib.dump(model, MOD / f"{tag}.joblib", compress=3)
    te[["trip", "obs", "arc", "label"]].assign(proba=proba).to_parquet(RES / f"pred_test_{tag}.parquet")
    with open(RES / f"metrics_{tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  -> test top1={res['test_top1_accuracy']:.4f} macroF1={res['test_macro_f1']:.4f} "
          f"logloss={res['test_log_loss']:.4f}", flush=True)
    return res


def comparison_table():
    keep = ["model", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "train_macro_f1",
            "test_macro_f1", "test_precision", "test_recall", "test_log_loss",
            "test_top1_accuracy", "train_time_s", "inference_time_s"]
    rows = [{k: json.load(open(p)).get(k) for k in keep} for p in sorted(RES.glob("metrics_*.json"))]
    table = pd.DataFrame(rows).sort_values("test_top1_accuracy", ascending=False)
    table.to_csv(RES / "model_comparison.csv", index=False)
    return table


if __name__ == "__main__":
    tr, te = load_tables()
    for n in (sys.argv[1:] or list(MODELS)):
        evaluate(n, tr=tr, te=te)
    print(comparison_table().to_string(index=False))
