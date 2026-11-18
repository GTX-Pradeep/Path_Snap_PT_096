"""Plots from actual experiment outputs (results/metrics_*.json).

  python plot_results.py            -> comparison plots for every trained model
  python plot_results.py rf         -> diagnostic plots for one model
Saved to results/plots/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_all():
    return [json.load(open(p)) for p in sorted(RES.glob("metrics_*.json"))]


def comparison():
    ms = load_all()
    names = [m["model"] for m in ms]
    x = np.arange(len(ms))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].bar(x, [m["cv_macro_f1_mean"] for m in ms], yerr=[m["cv_macro_f1_std"] for m in ms], capsize=4)
    ax[0].set_title("5-fold GroupKFold CV Macro F1 (train)")
    w = 0.38
    ax[1].bar(x - w / 2, [m["test_top1_accuracy"] for m in ms], w, label="Top-1 accuracy")
    ax[1].bar(x + w / 2, [m["test_macro_f1"] for m in ms], w, label="Macro F1")
    ax[1].set_title("Student-test performance"); ax[1].legend(loc="lower right")
    ax[2].bar(x, [m["test_log_loss"] for m in ms], color="tab:red")
    ax[2].set_title("Student-test log loss (lower is better)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(names)
    for a in ax[:2]:
        a.set_ylim(0.5, 1.0)
    plt.tight_layout(); plt.savefig(OUT / "model_comparison.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [m["train_time_s"] for m in ms], 0.4, label="train (s)")
    plt.bar(x + 0.2, [m["inference_time_s"] for m in ms], 0.4, label="inference on student_test (s)")
    plt.yscale("log"); plt.xticks(x, names); plt.legend(); plt.title("Computational cost")
    plt.tight_layout(); plt.savefig(OUT / "compute_cost.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 4))
    for m in ms:
        plt.plot(range(1, len(m["cv_macro_f1_folds"]) + 1), m["cv_macro_f1_folds"], "o-", label=m["model"])
    plt.xlabel("fold"); plt.ylabel("macro F1"); plt.title("Stability across CV folds"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "cv_folds.png", dpi=120); plt.close()
    print("saved comparison plots to", OUT)


def single(name):
    m = json.load(open(RES / f"metrics_{name}.json"))
    cm = np.array(m["test_confusion_matrix"])
    n_plots = 2 + ("loss_curve" in m)
    fig, ax = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    ax[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(j, i, f"{v}", ha="center", va="center")
    ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(["pred 0", "pred 1"]); ax[0].set_yticklabels(["true 0", "true 1"])
    ax[0].set_title(f"{name}: student-test confusion matrix")
    ax[1].bar(range(1, 6), m["cv_macro_f1_folds"])
    ax[1].axhline(m["test_macro_f1"], color="red", ls="--", label="student-test")
    ax[1].axhline(m["train_macro_f1"], color="green", ls=":", label="train (fit data)")
    ax[1].set_ylim(0.5, 1); ax[1].set_xlabel("CV fold"); ax[1].set_title("Macro F1: CV vs train vs test")
    ax[1].legend()
    if "loss_curve" in m:  # only for models with an epoch-wise history
        ax[2].plot(m["loss_curve"], label="training loss")
        ax2 = ax[2].twinx()
        ax2.plot(m["validation_scores"], color="orange", label="validation accuracy")
        ax[2].set_xlabel("epoch"); ax[2].set_title("MLP training history")
        ax[2].legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(OUT / f"{name}_diagnostics.png", dpi=120); plt.close()
    print("saved", OUT / f"{name}_diagnostics.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for n in sys.argv[1:]:
            single(n)
    else:
        comparison()
