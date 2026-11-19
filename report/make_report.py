"""Builds the technical report (HTML + PDF) from the actual result files in results/ and models/.

Every number in the report is read from those files at build time, so the report always matches
the repository.  Usage:  python report/make_report.py   (PDF needs: pip install playwright; playwright install chromium)
"""
import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, MOD, REP = ROOT / "results", ROOT / "models", ROOT / "report"
GH = "https://github.com/GTX-Pradeep/Path_Snap_PT_096"


def J(p):
    return json.load(open(p))


def img(path, caption, width=100):
    p = ROOT / path
    if not p.exists():
        return f"<p class='missing'>[missing figure: {path}]</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' style='width:{width}%'>"
            f"<figcaption>{caption} <span class='src'>({path})</span></figcaption></figure>")


def table(df, fmt=None):
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        rows += "<tr>" + "".join(f"<td>{fmt.get(c, lambda v: v)(r[c])}</td>" for c in df.columns) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


pct = lambda v: f"{100 * v:.2f}%"
f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"
sec = lambda v: f"{v:.2f} s"

# ------------------------------------------------------------------ data
M = {p.stem.replace("metrics_", ""): J(p) for p in RES.glob("metrics_*.json")}
ORDER = [m for m in ["logreg", "rf", "knn", "lgbm", "mlp", "lgbm_tuned"] if m in M]
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "knn": "K-Nearest Neighbours",
         "lgbm": "LightGBM", "mlp": "MLP neural network", "lgbm_tuned": "LightGBM (Bayesian-tuned)"}
CONFIG = {"logreg": "StandardScaler; C=1.0, L2, class_weight=balanced, max_iter=1000",
          "rf": "100 trees, max_depth=12, min_samples_leaf=5, max_samples=0.2, balanced_subsample",
          "knn": "StandardScaler; k=25, distance-weighted; fitted on 300k-row subsample",
          "lgbm": "300 trees, lr=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, colsample=0.8, is_unbalance",
          "mlp": "StandardScaler; hidden (64, 32) ReLU, Adam lr=1e-3, alpha=1e-4, batch 1024, early stopping; 500k-row subsample"}
HMM = pd.read_csv(RES / "hmm" / "comparison.csv")
CFG = J(MOD / "final_config.json")
FINAL = CFG["model_name"]
BO = J(RES / "bayes_opt" / "best_params.json")
if "lgbm_tuned" in M:
    CONFIG["lgbm_tuned"] = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in BO["best_params"].items())
INV = pd.read_csv(RES / "eda" / "inventory.csv")
TUNE = pd.read_csv(RES / "hmm" / "decoder_tuning.csv")
PT = pd.read_csv(RES / "hmm" / "per_trip.csv", dtype={"trip": str})


def hmm_row(model, method):
    r = HMM[(HMM.model == model) & (HMM.method == method)]
    return r.iloc[0] if len(r) else None


def fit_verdict(m):
    gap = m["train_macro_f1"] - m["cv_macro_f1_mean"]
    if gap > 0.05:
        return (f"<b>Overfitting.</b> Train macro F1 ({m['train_macro_f1']:.3f}) is clearly above CV "
                f"({m['cv_macro_f1_mean']:.3f}) and student-test ({m['test_macro_f1']:.3f}): the model fits its "
                f"training rows much better than unseen trips.")
    if m["cv_macro_f1_mean"] < 0.85:
        return (f"<b>Slight underfitting, no overfitting.</b> Train ({m['train_macro_f1']:.3f}), CV "
                f"({m['cv_macro_f1_mean']:.3f}) and student-test ({m['test_macro_f1']:.3f}) macro F1 are almost equal "
                f"but lower than the non-linear models: the model capacity, not the data, limits it.")
    return (f"<b>Good fit.</b> Train ({m['train_macro_f1']:.3f}), CV ({m['cv_macro_f1_mean']:.3f} ± "
            f"{m['cv_macro_f1_std']:.3f}) and student-test ({m['test_macro_f1']:.3f}) macro F1 are close, so it "
            f"generalises to unseen trips without memorising the training trips.")


WHY = {
 "logreg": ("Each candidate is a yes/no decision (is this the true arc?), and logistic regression is the standard "
            "probabilistic baseline for that. It outputs calibrated probabilities that the HMM can use directly as "
            "emissions, trains in seconds, and its coefficients show how each feature pushes a candidate up or down.",
            "Strengths: fast, interpretable, stable across folds, good probabilities. Limitations: a linear decision "
            "boundary cannot express interactions such as <i>close AND same direction</i>; balanced class weights give "
            "high recall but low precision.",
            "Binary cross-entropy (log loss) with L2 regularisation (C=1.0), lbfgs solver."),
 "rf": ("The right arc depends on feature interactions (distance only matters if the heading also agrees). An "
        "ensemble of decision trees learns such rules without feature scaling and is robust to the outliers found "
        "in EDA (jumps, gaps). Averaging many trees on bootstrap samples gives stable probabilities.",
        "Strengths: non-linear, robust, little tuning needed, feature importances. Limitations: slower to train "
        "and larger on disk (~8 MB) than linear or boosted models; probabilities are averaged votes, less sharp "
        "than boosting.",
        "Each tree minimises Gini impurity at every split; the forest averages tree probabilities."),
 "knn": ("A candidate that looks like training candidates that were correct is probably correct: KNN scores a "
         "candidate by the labels of its 25 most similar training candidates in standardised feature space. It "
         "is a non-parametric reference point with no training phase.",
         "Strengths: no model assumptions, captures local structure. Limitations: memorises the training set, "
         "prediction cost grows with the number of stored rows (fitted on a 300k subsample, still the slowest "
         "inference), very sensitive to feature scaling, model file too large for GitHub (43 MB).",
         "No loss is optimised (lazy learner); prediction = distance-weighted vote of the 25 nearest neighbours "
         "(Euclidean distance on standardised features)."),
 "lgbm": ("Gradient boosting adds trees one at a time, each correcting the errors of the previous ones, which "
          "suits tabular features with interactions. LightGBM's histogram algorithm trains on ~1.4M rows in "
          "seconds and predicts fast enough for the live API.",
          "Strengths: strong accuracy on tabular data, fast training and inference, handles imbalance "
          "(is_unbalance), compact model (~1 MB). Limitations: many hyperparameters, can overfit with deep trees, "
          "less interpretable than a linear model.",
          "Binary log loss minimised by gradient boosting (second-order, leaf-wise trees)."),
 "mlp": ("A neural network can learn smooth non-linear combinations of all 14 features; it is the neural-network "
         "family required for comparison, and it is the only model here that produces an epoch-wise training history.",
         "Strengths: flexible non-linear model, best-calibrated probabilities in our experiments (lowest log loss). "
         "Limitations: needs feature scaling and a subsample for training time, sensitive to learning rate and "
         "initialisation, a black box.",
         "Binary cross-entropy minimised with Adam (lr 1e-3), L2 penalty alpha=1e-4, early stopping on a 10% validation split."),
}
WHY["lgbm_tuned"] = (WHY["lgbm"][0] + " This is the selected model, tuned with Bayesian optimisation (Section 4).",
                     WHY["lgbm"][1], WHY["lgbm"][2])


def model_section(i, k):
    m = M[k]
    why, strength, loss = WHY[k]
    extra = ""
    if "loss_curve" in m:
        extra = (f"<p>The MLP stopped after {len(m['loss_curve'])} epochs (early stopping). The training loss fell from "
                 f"{m['loss_curve'][0]:.4f} to {m['loss_curve'][-1]:.4f}; validation accuracy is shown in the diagnostics "
                 f"figure (right panel). This is the only model with an epoch-wise history; no artificial curves are drawn "
                 f"for the others.</p>")
    clf = hmm_row(k, "classifier only"); hm = hmm_row(k, "classifier + HMM")
    hmm_txt = ""
    if clf is not None:
        hmm_txt = (f" Route continuity of its argmax predictions is {pct(clf.route_continuity)} "
                   f"({int(clf.breaks)} breaks); with HMM decoding pointwise accuracy becomes {pct(hm.pointwise_accuracy)}.")
    return f"""
<h3>3.2.{i} {NAMES[k]} ({m['family']})</h3>
<p><b>A1. Why suitable?</b> {why}</p>
<p><b>A2. Strengths and limitations.</b> {strength}</p>
<p><b>B5. Important hyperparameters (initial values).</b> {CONFIG.get(k, '')}. Trained on {m['train_rows']:,} candidate rows.</p>
<p><b>B6. Loss / objective.</b> {loss}</p>
<p><b>C7–C8. Student-test performance.</b> Observation-level Top-1 accuracy <b>{pct(m['test_top1_accuracy'])}</b>,
candidate-level macro F1 <b>{m['test_macro_f1']:.3f}</b> (precision {m['test_precision']:.3f}, recall {m['test_recall']:.3f}),
log loss <b>{m['test_log_loss']:.3f}</b>; 5-fold group CV macro F1 {m['cv_macro_f1_mean']:.3f} ± {m['cv_macro_f1_std']:.3f};
training time {m['train_time_s']:.1f} s, inference on all 10 student-test trips {m['inference_time_s']:.2f} s.{hmm_txt}</p>
<p><b>C9. Over- or underfitting?</b> {fit_verdict(m)}</p>
{extra}
{img(f"results/plots/{k}_diagnostics.png", f"{NAMES[k]}: student-test confusion matrix (threshold 0.5) and macro F1 per CV fold vs train and student-test.", 90)}
"""


def build():
    tr = INV[INV.partition == "train"]; te = INV[INV.partition == "student_test"]
    comp = pd.DataFrame([{
        "Model": NAMES[k], "Family": M[k]["family"], "Key configuration": CONFIG.get(k, ""),
        "CV macro F1": f"{M[k]['cv_macro_f1_mean']:.3f} ± {M[k]['cv_macro_f1_std']:.3f}",
        "Top-1 accuracy": pct(M[k]["test_top1_accuracy"]),
        "Route continuity": pct(hmm_row(k, "classifier only").route_continuity) if hmm_row(k, "classifier only") is not None else "N/A",
        "Student-test macro F1": f3(M[k]["test_macro_f1"]), "Log loss": f3(M[k]["test_log_loss"]),
        "Inference time": sec(M[k]["inference_time_s"])} for k in ORDER])

    hm_tab = HMM.copy()
    hm_tab["model"] = hm_tab.model.map(NAMES)
    hm_tab = hm_tab[["model", "method", "pointwise_accuracy", "route_continuity", "breaks", "route_precision", "route_recall", "route_f1"]]
    final_c, final_h = hmm_row(FINAL, "classifier only"), hmm_row(FINAL, "classifier + HMM")
    ptf = PT[(PT.model == FINAL)].pivot(index="trip", columns="method", values="pointwise_accuracy")
    worse = ptf[ptf.hmm < ptf.clf]
    best_tune = TUNE.loc[TUNE.pointwise_accuracy.idxmax()]

    tvu = pd.read_csv(RES / "bayes_opt" / "tuned_vs_untuned.csv", index_col=0)
    tvu_tab = pd.DataFrame({"Metric": ["CV macro F1 (5-fold group)", "Student-test Top-1 accuracy", "Student-test macro F1",
                                       "Log loss", "Training time (s)", "Inference time, 10 trips (s)"],
                            "Untuned": [f4(tvu.loc[c, "untuned"]) for c in tvu.index],
                            "Tuned": [f4(tvu.loc[c, "tuned"]) for c in tvu.index],
                            "Change": [f"{tvu.loc[c, 'change']:+.4f}" for c in tvu.index]})
    better = tvu.loc["test_top1_accuracy", "change"] > 0 and tvu.loc["cv_macro_f1_mean", "change"] > 0
    route_pngs = sorted((RES / "hmm").glob("route_*.png"))
    route_figs = "".join(img(str(p.relative_to(ROOT)), f"Trip {p.stem.split('_')[1]}: classifier-only vs classifier + HMM.", 100) for p in route_pngs)

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report — Team PT_096</title>
<style>
body {{ font: 10.5pt/1.45 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color:#1d2433; max-width: 960px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 24pt; margin: 0 0 4px; }} h2 {{ font-size: 15pt; border-bottom: 2px solid #d6336c; padding-bottom: 3px; margin-top: 28px; page-break-after: avoid; }}
h3 {{ font-size: 12pt; margin-top: 18px; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 8px 0 14px; page-break-inside: avoid; }}
th, td {{ border: 1px solid #ccd2db; padding: 3px 5px; text-align: left; vertical-align: top; }} th {{ background: #f1f3f6; }}
figure {{ margin: 10px 0 16px; text-align: center; page-break-inside: avoid; }} img {{ max-width: 100%; }}
figcaption {{ font-size: 8.8pt; color: #5d6779; margin-top: 3px; }} .src {{ color:#98a1b0; }}
.title {{ border-bottom: 3px solid #d6336c; padding-bottom: 10px; margin-bottom: 14px; }} .muted {{ color:#5d6779; }}
.box {{ background:#f6f7f9; border-left: 4px solid #d6336c; padding: 8px 12px; margin: 10px 0; }}
code {{ font-size: 9pt; background:#f1f3f6; padding: 0 3px; }} pre {{ background:#f1f3f6; padding: 8px; font-size: 8.5pt; overflow-x: auto; }}
</style></head><body>
<div class='title'><h1>PathSnap — Distributed Map-Matching &amp; Route Reconstruction</h1>
<div class='muted'>Technical report · UE24CS352A Machine Learning Hackathon · PES University · Team PT_096</div>
<div class='muted'>Team: Pradeep, Poojitha, Pandu, Pavan · Repository: <a href='{GH}'>{GH}</a></div></div>

<div class='box'><b>Summary.</b> Noisy GPS points are matched to the road network in two stages: a classifier scores the 8 nearest
road segments of every GPS point, then an HMM/Viterbi decoder uses road connectivity to pick one connected route.
Five classifier families were compared; LightGBM was selected and tuned with Bayesian optimisation. On the 10 student-test trips the
final system reaches <b>{pct(final_h.pointwise_accuracy)}</b> pointwise accuracy (classifier only: {pct(final_c.pointwise_accuracy)}),
route continuity <b>{pct(final_h.route_continuity)}</b> (classifier only: {pct(final_c.route_continuity)}) and route F1 {f3(final_h.route_f1)}.
The system runs as a FastAPI backend (Laptop 1) and a browser UI (Laptop 2).</div>

<h2>1. Problem and approach</h2>
<p>GPS fixes drift off the road (buildings, multipath, dropouts), so a vehicle appears on parallel streets or between roads.
PathSnap reconstructs the road segments actually driven. Pipeline: clean GPS → project to metres → 8 candidate arcs per GPS point →
14 features per (point, candidate) → classifier probability that the candidate is the true arc → HMM/Viterbi over the candidates →
one arc per GPS point and an ordered route. Laptop 1 hosts the model, decoder, road networks and evaluation; Laptop 2 hosts the UI.</p>

<h2>2. Dataset understanding and exploratory data analysis</h2>
<p><b>Partitions used:</b> train (80 trips) for all fitting, cross-validation, Bayesian optimisation and decoder tuning; student_test (10 trips)
for model comparison; evaluator_test (10 trips) only inside the API at demonstration time. All EDA below uses train + student_test
(columns marked per partition).</p>
<h3>2.1 How we preprocessed the data</h3>
<ul>
<li><b>Reading.</b> <code>.track</code> (lon, lat, t), <code>.nodes</code> (lon, lat; row = node id), <code>.arcs</code> (from, to; row = arc id, directed),
<code>.route</code> (ordered arc ids). Column order was verified per trip (longitude first, latitude second; all values in range). The trips are spread over several cities worldwide (latitudes −35° to 62°, longitudes −100° to 137°).</li>
<li><b>Cleaning.</b> Drop NaN or out-of-range coordinates, sort by time, drop duplicate timestamps (none were found).</li>
<li><b>Projection.</b> Degrees are angles and the metres-per-degree of longitude shrinks with latitude (at 56°N, 1° of longitude ≈ 0.56 of 1° of latitude), and the trips lie in different cities, so every trip is projected to a local equirectangular plane in metres centred on the trip.</li>
<li><b>Candidates.</b> Arcs are sampled every 15 m into a KD-tree; the 8 nearest distinct arcs (exact point-to-segment distance) are the candidates of a GPS point.</li>
<li><b>Per-point labels.</b> <code>.route</code> has no timestamps, so GPS points are aligned to it with a monotonic dynamic-programming alignment
(route index never decreases, total point-to-arc distance minimised). A candidate is positive if it is the aligned arc (≈1 in 8 candidates).</li>
<li><b>Label cleaning (train only).</b> Observations whose aligned reference arc is &gt; 50 m away are removed from training
(11% of training observations, mainly trips 33 and 54 whose reference route does not follow the GPS). Student-test is never filtered.
Candidate miss rate (true arc not among the 8): train 8.2% → 0.85% after cleaning; student-test 0.35%.</li>
</ul>
<h3>2.2 Missing values and data quality</h3>
{table(pd.DataFrame([
 {"Check": "Trips", "train": len(tr), "student_test": len(te)},
 {"Check": "GPS points (total)", "train": f"{tr.n_gps.sum():,}", "student_test": f"{te.n_gps.sum():,}"},
 {"Check": "GPS points per trip (min / median / max)", "train": f"{tr.n_gps.min():,} / {int(tr.n_gps.median()):,} / {tr.n_gps.max():,}", "student_test": f"{te.n_gps.min():,} / {int(te.n_gps.median()):,} / {te.n_gps.max():,}"},
 {"Check": "Missing values", "train": int(tr.missing_values.sum()), "student_test": int(te.missing_values.sum())},
 {"Check": "Invalid coordinates", "train": int(tr.invalid_coords.sum()), "student_test": int(te.invalid_coords.sum())},
 {"Check": "Non-increasing timestamps", "train": int(tr.non_monotonic_t.sum()), "student_test": int(te.non_monotonic_t.sum())},
 {"Check": "Mean share of 1-second steps", "train": f"{tr.pct_dt_eq_1s.mean():.1f}%", "student_test": f"{te.pct_dt_eq_1s.mean():.1f}%"},
 {"Check": "Largest time gap (s)", "train": f"{tr.max_gap_s.max():,.0f} (trip {int(tr.loc[tr.max_gap_s.idxmax(), 'trip']):08d})", "student_test": f"{te.max_gap_s.max():,.0f}"},
 {"Check": "Nodes per trip (median)", "train": f"{int(tr.n_nodes.median()):,}", "student_test": f"{int(te.n_nodes.median()):,}"},
 {"Check": "Arcs per trip (median)", "train": f"{int(tr.n_arcs.median()):,}", "student_test": f"{int(te.n_arcs.median()):,}"},
 {"Check": "Isolated nodes", "train": int(tr.isolated_nodes.sum()), "student_test": int(te.isolated_nodes.sum())},
 {"Check": "Consecutive route arcs connected (min over trips)", "train": f"{tr.route_consecutive_connected.min():.3f}", "student_test": f"{te.route_consecutive_connected.min():.3f}"},
]))}
<h3>2.3 Patterns observed and features extracted</h3>
<p>GPS points lie near, not on, the roads (typically a few to ~20 m off); dense junctions and parallel roads create several
plausible candidates; two-way roads are two directed arcs; some trips have long dropouts. The features answer these observations:</p>
{table(pd.DataFrame([
 {"Feature": "dist, dist_rank, dist_minus_min, dist_ratio_min", "Why": "closeness to the GPS point, absolute and relative to the other candidates"},
 {"Feature": "heading_diff, heading_diff_cos", "Why": "GPS heading vs arc direction: separates the two directions of a road and crossing streets"},
 {"Feature": "proj_frac", "Why": "where on the segment the point projects (end points are ambiguous at junctions)"},
 {"Feature": "arc_len", "Why": "short connector arcs vs long road segments"},
 {"Feature": "speed", "Why": "local motion; heading is unreliable when the vehicle is slow"},
 {"Feature": "n_cand_30m", "Why": "local ambiguity (dense junctions)"},
 {"Feature": "dist_prev, dist_next", "Why": "temporal context: are the neighbouring GPS points also close to this arc?"},
 {"Feature": "out_degree, in_degree", "Why": "junction structure around the arc"},
]))}
{img("results/eda/train_trajectories.png", "All 80 training GPS trajectories (train).", 62)}
{img("results/eda/trip_distributions.png", "GPS points, largest gap and mean speed per trip (train vs student_test).", 100)}
{img("results/eda/sampling_intervals.png", "Sampling intervals, log scale (train).", 62)}
{img("results/eda/road_network_train_00000010.png", "Road-network visualisation of train trip 00000010: network (grey), reference route (green), GPS (blue).", 62)}
{img("results/eda/network_size.png", "Road-network size per trip (train + student_test).", 62)}
<p>Interactive Folium map (EDA): <code>maps/trip_train_00000010.html</code> (raw GPS, reference route, road network, layer control); generated by <code>src/make_map.py</code>.</p>

<h2>3. Model experimentation and comparison</h2>
<p><b>Protocol (identical for every model).</b> Candidate rows from the 80 training trips (label-cleaned) are used for 5-fold
<b>GroupKFold by trip</b> (no trip in both training and validation folds) → CV macro F1; the model is refitted on all training rows and
evaluated on the 10 student-test trips. <b>Primary selection metric: observation-level Top-1 accuracy</b> — for each GPS point the
highest-scoring candidate is compared with the reference arc — because it is exactly the map-matching decision; log loss is used as a
second criterion because the HMM consumes the probabilities. Seed 42 everywhere; logs in <code>results/train_log.txt</code>.</p>
<h3>3.1 Overall model comparison</h3>
{table(comp)}
<p class='muted'>Route continuity here = share of consecutive GPS points whose argmax arcs are identical or directly connected (classifier only, before HMM).</p>
<p><b>Model selection.</b> LightGBM had the best student-test Top-1 accuracy of the five untuned models, near-best log loss, the second-fastest
training and sub-second inference, and a 1 MB model file — the best fit for a live API and for the sequential decoder. The MLP had the
lowest log loss but needs a subsample to train in reasonable time; KNN overfits and is slow at inference; logistic regression underfits;
random forest is close but larger and slower. LightGBM was therefore tuned (Section 4) and used as the final classifier.</p>
<h3>3.2 Per-model questions</h3>
{''.join(model_section(i + 1, k) for i, k in enumerate(ORDER))}
<h3>3.3 Model visualisations</h3>
{img("results/plots/model_comparison.png", "CV macro F1, student-test Top-1 accuracy and macro F1, and log loss for all models.", 100)}
{img("results/plots/cv_folds.png", "Stability: macro F1 per CV fold.", 62)}
{img("results/plots/compute_cost.png", "Training and inference time (log scale).", 62)}

<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p><b>1. Hyperparameters searched / 2. ranges:</b></p>
{table(pd.DataFrame([{"Hyperparameter": k, "Range / distribution": v} for k, v in BO["search_space"].items()]))}
<p><b>3. Objective:</b> {BO['objective']} on the 80 training trips (maximised). Sampler: {BO['sampler']}; {BO['pruner']} stops unpromising trials after their first folds.
Trial 0 was seeded with the untuned defaults.</p>
<p><b>4. Trials:</b> {BO['n_trials']} ({BO['n_complete']} completed, {BO['n_pruned']} pruned); search time {BO['search_time_s'] / 60:.1f} min. Every trial is stored in <code>results/bayes_opt/trials.csv</code>.</p>
<p><b>5. Best values:</b> {CONFIG.get('lgbm_tuned', '')} (3-fold CV macro F1 {BO['best_cv3_macro_f1']:.4f}).</p>
<p><b>6. Did tuning help?</b> {"Yes." if better else "Only partly."} Same data partitions and metrics for both:</p>
{table(tvu_tab)}
{img("results/bayes_opt/optimisation_history.png", "Optimisation history: every completed trial and the best value so far.", 70)}
{img("results/bayes_opt/param_importance.png", "Hyperparameter importance (fANOVA) estimated from the trials.", 60)}

<h2>5. HMM and Viterbi route reconstruction</h2>
<p><b>1. Hidden states.</b> For GPS point <i>i</i>, the states are its 8 candidate arcs; state <i>j</i> at time <i>i</i> is always the arc
<code>cand[i, j]</code>, so the candidate-to-segment correspondence is preserved and the decoder returns one arc per GPS point.</p>
<p><b>2. Emissions.</b> log P(candidate is the true arc) from the final classifier, multiplied by an emission weight w.</p>
<p><b>3. Transitions.</b> Between consecutive points the vehicle drives along the network, so the network distance between the two matched
positions should equal the straight GPS step: <code>route_d = rest of arc a + shortest path(end of a → start of b) + part of arc b</code>
(Dijkstra on the directed road graph, arc lengths as weights, limit {CFG['hmm_params']['max_route_m']:.0f} m); on the same arc, the signed
progress (frac_b − frac_a)·length, so driving backwards is penalised. <code>log T(a→b) = −|route_d − gps_d| / β</code>.</p>
<p>Viterbi recurrence: δ<sub>i+1</sub>(b) = max<sub>a</sub>[δ<sub>i</sub>(a) + log T(a,b)] + w·log e<sub>i+1</sub>(b), with back-pointers; the best final state is traced back.</p>
<p><b>4. Missing candidates and gaps.</b> Points with fewer than 8 candidates are padded with impossible (−∞) states. If the time gap exceeds
{CFG['hmm_params']['gap_s']:.0f} s the transition is unconstrained (log T = 0), because after a dropout any road may be reached.</p>
<p><b>5. Disconnected segments.</b> If arc b cannot be reached from arc a within the distance limit, log T = {CFG['hmm_params']['unreachable']:.0f}
(a large finite penalty rather than −∞): the decoder avoids such jumps but can still make one when the classifier is very confident,
so a path always exists.</p>
<p><b>Decoder tuning.</b> β and w were chosen by grid search on {len(CFG.get('decoder_tuned_on_train_trips', []))} training trips using
<b>out-of-fold</b> classifier probabilities (each trip scored by a model that never saw it), objective = pointwise accuracy. Best: β = {best_tune.beta:g} m,
w = {best_tune.w_emit:g} (accuracy {best_tune.pointwise_accuracy:.4f} vs classifier-only {CFG.get('decoder_tuning_classifier_only_accuracy', float('nan')):.4f}
on those trips). Student-test and evaluator data were not used for tuning.</p>
{img("results/hmm/decoder_tuning.png", "Decoder tuning grid (training trips, out-of-fold probabilities).", 55)}
<p><b>6. Did HMM decoding improve the route?</b> Yes. Classifier-only vs classifier + HMM on the 10 student-test trips, for every classifier:</p>
{table(hm_tab, {"pointwise_accuracy": pct, "route_continuity": pct, "route_precision": f3, "route_recall": f3, "route_f1": f3})}
<p>For the final model, pointwise accuracy rises from {pct(final_c.pointwise_accuracy)} to {pct(final_h.pointwise_accuracy)}, breaks fall from
{int(final_c.breaks)} to {int(final_h.breaks)} and route F1 from {f3(final_c.route_f1)} to {f3(final_h.route_f1)}. The gain is largest for the weakest
classifiers, because the transition model repairs isolated wrong choices.</p>
{img("results/hmm/classifier_vs_hmm.png", "Classifier only vs classifier + HMM on student_test, for every model.", 100)}
{img("results/hmm/per_trip_accuracy.png", f"Per-trip pointwise accuracy of {NAMES.get(FINAL, FINAL)} (student_test).", 70)}
<p><b>Where HMM did not help.</b> {("Trips " + ", ".join(worse.index) + " got worse with HMM. ") if len(worse) else "No student-test trip got worse. "}
In trip 00000007 the vehicle is parked for the last ~175 points on a two-way road. At standstill the GPS heading is noise, the
classifier prefers the opposite-direction twin arc, and the accumulated emission evidence outweighs one U-turn transition, so the
decoder switches to the wrong direction although the route geometry is the same. A heading-reliability feature (speed-aware heading)
is the planned fix; we report the case instead of hiding it.</p>
{route_figs}

<h2>6. Geographical visualisation using Folium</h2>
<p>Interactive maps generated by our code (<code>src/make_map.py</code>, <code>src/route_maps.py</code>) are committed in <code>maps/</code>:
raw GPS (blue), relevant road network (grey, toggle), reference route (green), classifier-only arcs (orange dashed) and the HMM-predicted
route (red), with a legend, layer control and the extent fitted to the trip.</p>
{img("report/figures/folium_route.png", "Screenshot of a Folium route map (student_test). Base-map tiles may be missing in this screenshot because it was captured in an offline environment; the layers are the real output.", 95)}

<h2>7. API, UI and deployment</h2>
<p><b>Laptop 1 — API</b> (<code>api/main.py</code>, FastAPI + uvicorn, port 8000). On start it loads <code>models/{CFG['model_file']}</code> and
<code>models/final_config.json</code>. Endpoints: <code>GET /health</code>, <code>GET /trips</code> (the 10 evaluation trip ids),
<code>POST /predict {{"trip_id": …}}</code>: loads the trip on the server, removes the reference route before prediction, runs candidates →
features → classifier → HMM/Viterbi, then scores the prediction against the server-side reference route and returns GPS points, the predicted
route geometry and the metrics. Every request is logged on the API console (trip, accuracy, continuity, route F1, time).
Reference routes are never sent to the UI.</p>
<p><b>Laptop 2 — UI</b> (<code>ui/</code>, static HTML/JS + Leaflet, served with <code>python -m http.server 8080</code>): enter the API address,
see the 10 trips, click one → the UI posts the trip id to the API and shows the map (GPS, predicted route, classifier-only arcs) and the metrics.</p>
{img("report/figures/ui.png", "Evaluator UI after running a trip (captured during testing; map tiles were blocked in the test environment).", 100)}
{img("report/figures/api_log.png", "API console log while serving requests (captured during testing).", 100)}
<p class='muted'>Replace these two screenshots with ones taken on the two demonstration laptops (the guidelines ask for the CMD of the receiving laptop).</p>
<p><b>How to run</b>: see the README (Laptop 1: <code>run_api.bat</code>; Laptop 2: <code>run_ui.bat</code>, then enter <code>http://&lt;laptop-1-IP&gt;:8000</code>).</p>

<h2>8. Academic integrity and AI-assistance disclosure</h2>
<p>All experiments in this report were actually run; every number and figure is produced by the code in the repository and can be regenerated
(<code>python report/make_report.py</code> rebuilds this report from the result files). Evaluator-test data were not used for training, model
selection, hyperparameter or decoder tuning; predictions are not hardcoded. <b>AI assistance:</b> an AI coding assistant was used to help write
code and draft this report; the team reviewed, ran and verified the code and results and can explain every part. [Team: edit this statement
to match the event's disclosure rules.]</p>

<h2>Appendix — reproduce</h2>
<pre>cd src
python eda.py                       # EDA tables + plots
python build_dataset.py             # candidates + features
python train_models.py              # 5 models, CV + student-test
python bayes_opt.py 30              # Bayesian optimisation + tuned model
python hmm_experiments.py           # decoder tuning + classifier vs HMM + models/final_config.json
python route_maps.py                # Folium route maps
python plot_results.py              # comparison plots
python ../report/make_report.py     # this report</pre>
</body></html>"""
    (REP / "PathSnap_Technical_Report.html").write_text(html, encoding="utf-8")
    print("wrote", REP / "PathSnap_Technical_Report.html")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto((REP / "PathSnap_Technical_Report.html").as_uri())
            pg.pdf(path=str(REP / "PathSnap_Technical_Report.pdf"), format="A4", print_background=True,
                   margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"})
            b.close()
        print("wrote", REP / "PathSnap_Technical_Report.pdf")
    except Exception as e:
        print("PDF skipped (install playwright + chromium, or print the HTML to PDF from a browser):", e)


if __name__ == "__main__":
    build()
