"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model: LightGBM.

Why LightGBM: best student-test Top-1 accuracy of the five models (86.04%), fast training
(~11 s) and fast inference (needed by the API), and good probability quality for the HMM.

Objective  : mean candidate-level Macro F1 over 3-fold GroupKFold by trip (80 training trips).
Sampler    : TPE (Tree-structured Parzen Estimator), seed 42, median pruner per fold.
Trials     : N_TRIALS (default 30).
Afterwards : best params are refitted and evaluated with the SAME protocol as every other model
             (5-fold group CV + student_test) -> results/metrics_lgbm_tuned.json.
Outputs    : results/bayes_opt/{trials.csv, best_params.json, tuned_vs_untuned.csv, *.png}
Usage      : python bayes_opt.py [n_trials]
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from features import FEATURES
from model_zoo.lgbm import make_lgbm
from train_models import evaluate, load_tables

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bayes_opt"
OUT.mkdir(parents=True, exist_ok=True)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_FOLDS = 3

SEARCH_SPACE = {
    "n_estimators": "int 100..600",
    "learning_rate": "log-uniform 0.01..0.2",
    "num_leaves": "int log 15..255",
    "min_child_samples": "int log 10..300",
    "subsample": "uniform 0.5..1.0",
    "colsample_bytree": "uniform 0.5..1.0",
    "reg_lambda": "log-uniform 1e-3..10",
    "reg_alpha": "log-uniform 1e-3..10",
    "is_unbalance": "categorical {True, False}",
}


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        is_unbalance=trial.suggest_categorical("is_unbalance", [True, False]),
    )


def main():
    tr, te = load_tables()
    X, y, g = tr[FEATURES].to_numpy(), tr.label.to_numpy(), tr.trip.to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, g))

    def objective(trial):
        params = suggest(trial)
        scores = []
        for k, (a, b) in enumerate(folds):
            m = make_lgbm(**params).fit(X[a], y[a])
            scores.append(f1_score(y[b], m.predict(X[b]), average="macro"))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name="lgbm_macro_f1",
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    # trial 0 = the untuned defaults, so the search starts from the baseline
    from model_zoo.lgbm import DEFAULTS
    study.enqueue_trial({k: DEFAULTS[k] for k in ["n_estimators", "learning_rate", "num_leaves",
                                                  "min_child_samples", "subsample", "colsample_bytree",
                                                  "is_unbalance"]} | {"reg_lambda": 1e-3, "reg_alpha": 1e-3})
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS,
                   callbacks=[lambda s, t: print(f"trial {t.number:2d} {t.state.name:8s} "
                                                 f"value={t.value} best={s.best_value:.4f}", flush=True)])
    search_time = time.time() - t0

    trials = study.trials_dataframe()
    trials.to_csv(OUT / "trials.csv", index=False)
    best = study.best_params
    json.dump({"best_params": best, "best_cv3_macro_f1": study.best_value, "n_trials": N_TRIALS,
               "n_complete": int((trials.state == "COMPLETE").sum()),
               "n_pruned": int((trials.state == "PRUNED").sum()),
               "search_time_s": search_time, "objective": f"mean macro F1, {N_FOLDS}-fold GroupKFold by trip",
               "sampler": "TPE (seed 42)", "pruner": "MedianPruner", "search_space": SEARCH_SPACE},
              open(OUT / "best_params.json", "w"), indent=2)

    # plots: optimisation history + parameter importances
    done = trials[trials.state == "COMPLETE"]
    plt.figure(figsize=(8, 4))
    plt.scatter(done.number, done.value, s=18, label="trial")
    plt.plot(done.number, done.value.cummax(), color="tab:red", label="best so far")
    plt.axhline(done.value.iloc[0], color="grey", ls=":", label="trial 0 = untuned defaults")
    plt.xlabel("trial"); plt.ylabel(f"{N_FOLDS}-fold group CV macro F1"); plt.legend()
    plt.title("Bayesian optimisation history (LightGBM, TPE)")
    plt.tight_layout(); plt.savefig(OUT / "optimisation_history.png", dpi=120); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        pd.Series(imp).sort_values().plot.barh(figsize=(7, 4), title="Hyperparameter importance (fANOVA)")
        plt.tight_layout(); plt.savefig(OUT / "param_importance.png", dpi=120); plt.close()
    except Exception as e:  # importance needs enough complete trials
        print("importance skipped:", e)

    # refit + evaluate with the same protocol as all models
    tuned = evaluate("lgbm", make=lambda: make_lgbm(**best), family="Boosting (Bayesian-tuned)",
                     tag="lgbm_tuned", tr=tr, te=te)
    untuned = json.load(open(ROOT / "results" / "metrics_lgbm.json"))
    cols = ["cv_macro_f1_mean", "test_top1_accuracy", "test_macro_f1", "test_log_loss",
            "train_time_s", "inference_time_s"]
    cmp = pd.DataFrame({"untuned": [untuned[c] for c in cols], "tuned": [tuned[c] for c in cols]}, index=cols)
    cmp["change"] = cmp.tuned - cmp.untuned
    cmp.to_csv(OUT / "tuned_vs_untuned.csv")
    print(cmp.round(4).to_string())
    print("best params:", best)


if __name__ == "__main__":
    main()
