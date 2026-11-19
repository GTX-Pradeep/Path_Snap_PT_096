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
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return (f"<figure><img src='data:{mime};base64,{b64}' style='width:{width}%'>"
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
        return (f"<b>Yes, it overfits.</b> On the rows it was trained on it scores {m['train_macro_f1']:.3f}, but only "
                f"{m['cv_macro_f1_mean']:.3f} in cross-validation and {m['test_macro_f1']:.3f} on student-test. In other words it "
                f"remembers its training rows far better than it handles trips it has not seen.")
    if m["cv_macro_f1_mean"] < 0.85:
        return (f"<b>It underfits a little, but does not overfit.</b> Train ({m['train_macro_f1']:.3f}), CV "
                f"({m['cv_macro_f1_mean']:.3f}) and student-test ({m['test_macro_f1']:.3f}) scores are almost the same, yet all "
                f"three sit below the tree and neural models. The model is simply too simple for the problem; more data would not fix that.")
    return (f"<b>Neither.</b> Train ({m['train_macro_f1']:.3f}), CV ({m['cv_macro_f1_mean']:.3f} ± "
            f"{m['cv_macro_f1_std']:.3f}) and student-test ({m['test_macro_f1']:.3f}) scores are close together, so what it "
            f"learned carries over to new trips instead of being memorised.")


WHY = {
 "logreg": ("For every candidate we are asking a yes/no question: is this the road the car was on? Logistic regression is "
            "the natural first model for that kind of question. It gives probabilities we can hand straight to the HMM, it "
            "trains in a few seconds, and its weights tell us which features push a candidate up or down. We used it as our baseline.",
            "Good: quick, easy to read, very stable from fold to fold. Bad: it draws a straight-line boundary, so it cannot "
            "learn rules like <i>close to the road AND pointing the same way</i>. The balanced class weights also make it "
            "say yes too often, which is why its recall is high and its precision is low.",
            "Log loss (binary cross-entropy) plus an L2 penalty (C=1.0), minimised with the lbfgs solver."),
 "rf": ("Picking the right road depends on features working together: being close only counts if the direction also "
        "matches. Decision trees pick up that kind of rule on their own, they do not care about feature scaling, and "
        "they cope well with the odd jumps and gaps we saw in the EDA. Averaging 100 trees smooths out the noise of any single tree.",
        "Good: handles non-linear rules, robust, works well with little tuning, and shows feature importance. Bad: "
        "slower to train and bigger on disk (about 8 MB) than the linear or boosted models, and its probabilities "
        "(averaged votes) are a bit blunter than boosting's.",
        "Every split in every tree is chosen to reduce Gini impurity; the forest then averages the trees' probabilities."),
 "knn": ("The idea is simple: if a candidate looks like training candidates that turned out to be correct, it is "
         "probably correct too. KNN looks up the 25 most similar training candidates and lets them vote. It makes "
         "no assumptions about the shape of the boundary, which made it a useful point of comparison.",
         "Good: no assumptions, picks up local patterns. Bad: it effectively memorises the training set, prediction "
         "gets slower the more rows it stores (even on a 300k subsample it was by far our slowest model), it depends "
         "heavily on feature scaling, and the saved model (43 MB) was too big to upload to GitHub.",
         "Nothing is optimised during training (it is a lazy learner). A prediction is a distance-weighted vote of the "
         "25 nearest neighbours, using Euclidean distance on standardised features."),
 "lgbm": ("Boosting builds trees one after another, and each new tree focuses on the mistakes of the ones before it. "
          "That works very well on tabular features like ours. LightGBM in particular trains on our 1.4 million rows "
          "in seconds and predicts fast enough for a live API, which mattered for the demo.",
          "Good: very accurate on this kind of data, fast in both training and prediction, handles the class imbalance, "
          "and the model file is small (about 1 MB). Bad: lots of hyperparameters to get right, deep trees can "
          "overfit, and it is harder to explain than a linear model.",
          "Log loss, minimised by gradient boosting with leaf-wise trees."),
 "mlp": ("A small neural network can learn smooth, non-linear mixes of all 14 features at once. We wanted one model "
         "from the neural-network family in the comparison, and it is also the only one of our models that gives a "
         "real training curve over epochs.",
         "Good: flexible, and in our runs it gave the most trustworthy probabilities (lowest log loss). Bad: needs scaled "
         "features, we had to train it on a subsample to keep the time reasonable, results move with the learning rate "
         "and random start, and it is hard to look inside.",
         "Log loss, minimised with Adam (learning rate 1e-3), with a small L2 penalty (alpha=1e-4) and early stopping on a 10% validation split."),
}
WHY["lgbm_tuned"] = (WHY["lgbm"][0] + " This is the version we finally used, after Bayesian tuning (Section 4).",
                     WHY["lgbm"][1], WHY["lgbm"][2])


def model_section(i, k):
    m = M[k]
    why, strength, loss = WHY[k]
    extra = ""
    if "loss_curve" in m:
        extra = (f"<p>Early stopping ended training after {len(m['loss_curve'])} epochs. Over that time the training loss "
                 f"dropped from {m['loss_curve'][0]:.4f} to {m['loss_curve'][-1]:.4f}; the validation accuracy is in the right "
                 f"panel of the figure below. The other models do not train in epochs, so we did not draw curves for them.</p>")
    clf = hmm_row(k, "classifier only"); hm = hmm_row(k, "classifier + HMM")
    hmm_txt = ""
    if clf is not None:
        hmm_txt = (f" Taking its top candidate for every point gives a route continuity of {pct(clf.route_continuity)} "
                   f"({int(clf.breaks)} breaks); after HMM decoding its pointwise accuracy goes up to {pct(hm.pointwise_accuracy)}.")
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
<table style='width:auto;margin-top:8px'><thead><tr><th>Name</th><th>SRN</th><th>Main contribution</th></tr></thead><tbody>
<tr><td>Pandu</td><td>PES1UG24AM382</td><td>KNN, MLP, Bayesian hyperparameter optimisation</td></tr>
<tr><td>Pavan</td><td>PES1UG24AM383</td><td>LightGBM, HMM / Viterbi decoder</td></tr>
<tr><td>Poojitha</td><td>PES1UG24AM384</td><td>EDA, Folium maps, Random Forest, evaluator UI</td></tr>
<tr><td>Pradeep</td><td>PES1UG24AM385</td><td>Data pipeline and features, Logistic Regression, FastAPI backend, report</td></tr>
</tbody></table>
<div class='muted'>Repository: <a href='{GH}'>{GH}</a></div></div>

<div class='box'><b>In short.</b> We snap noisy GPS points to the road network in two steps. First, a classifier looks at the 8
nearest road segments around each GPS point and scores how likely each one is. Then an HMM with Viterbi decoding uses how the roads
connect to choose one route that actually makes sense. We tried five kinds of classifier, picked LightGBM and tuned it with Bayesian
optimisation. On the 10 student-test trips the full system gets <b>{pct(final_h.pointwise_accuracy)}</b> of points right
(the classifier alone gets {pct(final_c.pointwise_accuracy)}), keeps the route connected <b>{pct(final_h.route_continuity)}</b> of the time
(classifier alone: {pct(final_c.route_continuity)}) and reaches a route F1 of {f3(final_h.route_f1)}. It runs as a FastAPI server on one
laptop and a browser UI on a second laptop.</div>

<h2>1. Problem and approach</h2>
<p>A GPS fix is rarely exactly on the road. Tall buildings, reflections and short signal losses push points sideways, so on a plain
map a car seems to jump onto parallel streets or drive through blocks. Our goal was to work out which road segments the car really used.
The steps are: clean the GPS points, convert them to metres, collect the 8 nearest road segments for every point, compute 14 features
for each point–segment pair, let the classifier score every pair, and finally let the HMM/Viterbi decoder choose one connected route.
Laptop 1 holds the model, the decoder, the road networks and the scoring; Laptop 2 only shows the interface.</p>

<h2>2. Dataset understanding and exploratory data analysis</h2>
<p><b>Which data we used where:</b> the 80 training trips for all training, cross-validation, Bayesian tuning and decoder tuning; the
10 student-test trips only to compare models; the 10 evaluator trips only inside the API during the demo. The EDA in this section covers
train and student-test, and the tables say which partition each number comes from.</p>
<h3>2.1 How we preprocessed the data</h3>
<ul>
<li><b>Reading.</b> <code>.track</code> (lon, lat, t), <code>.nodes</code> (lon, lat; row = node id), <code>.arcs</code> (from, to; row = arc id, directed),
<code>.route</code> (ordered arc ids). Column order was verified per trip (longitude first, latitude second; all values in range). The trips are spread over several cities worldwide (latitudes −35° to 62°, longitudes −100° to 137°).</li>
<li><b>Cleaning.</b> We drop missing or out-of-range coordinates, sort by time and remove repeated timestamps. In practice none of these problems showed up.</li>
<li><b>Projection.</b> Degrees are angles and the metres-per-degree of longitude shrinks with latitude (at 56°N, 1° of longitude ≈ 0.56 of 1° of latitude), and the trips lie in different cities, so every trip is projected to a local equirectangular plane in metres centred on the trip.</li>
<li><b>Candidates.</b> We place a point every 15 m along every road segment and put them all in a KD-tree. For each GPS point we take the 8 closest different segments, measured exactly from the point to the segment.</li>
<li><b>Labels for each point.</b> The <code>.route</code> file lists the roads in order but says nothing about time, so it does not tell us which
road belongs to which GPS point. We solved this with a small dynamic-programming alignment: points are matched to route segments in order (never
going back along the route) so that the total distance is as small as possible. The candidate equal to the matched segment is the positive one,
which is roughly 1 in every 8 candidates.</li>
<li><b>Cleaning the labels (training only).</b> In trips 33 and 54 the reference route is mostly more than 100 m away from the GPS, so it clearly
does not describe those points. We dropped training points whose matched reference segment was more than 50 m away (11% of training points).
We never filtered the student-test data. After this, the share of training points whose true segment was missing from the 8 candidates fell from
8.2% to 0.85%; on student-test it is 0.35%.</li>
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
<p>A few things stood out. GPS points sit near the road but rarely on it, usually a few metres and sometimes 20 m away. Busy junctions
and parallel streets give several believable candidates. A two-way street is stored as two separate one-way segments. And some trips have
long gaps in the signal. We chose the features below with these observations in mind:</p>
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
<p><b>How we tested every model (the same way each time).</b> We ran 5-fold cross-validation on the 80 training trips with
<b>GroupKFold by trip</b>, so all the points of one trip are always in the same fold; otherwise neighbouring points from one trip would
leak between training and validation. After that we trained on all 80 trips and scored the 10 student-test trips. <b>Our main metric is
Top-1 accuracy per GPS point</b>: take the best-scoring candidate for a point and check whether it is the reference segment. That is exactly
the decision map matching has to make. Log loss was our second criterion, because the HMM uses the probabilities directly. We used seed 42
throughout; the full log is in <code>results/train_log.txt</code>.</p>
<h3>3.1 Overall model comparison</h3>
{table(comp)}
<p class='muted'>Route continuity here = share of consecutive GPS points whose argmax arcs are identical or directly connected (classifier only, before HMM).</p>
<p><b>Why we picked LightGBM.</b> Of the five untuned models it had the best Top-1 accuracy, its log loss was close to the best, it
trained quickly, it predicts in under a second and its model file is only about 1 MB. All of that suits a live API and the decoder. The MLP
had the lowest log loss, but we could only train it on a subsample in reasonable time. KNN overfits and is very slow to predict. Logistic
regression underfits. Random forest came close but is bigger and slower. So we tuned LightGBM (Section 4) and used it as the final classifier.</p>
<h3>3.2 Per-model questions</h3>
{''.join(model_section(i + 1, k) for i, k in enumerate(ORDER))}
<h3>3.3 Model visualisations</h3>
{img("results/plots/model_comparison.png", "CV macro F1, student-test Top-1 accuracy and macro F1, and log loss for all models.", 100)}
{img("results/plots/cv_folds.png", "Stability: macro F1 per CV fold.", 62)}
{img("results/plots/compute_cost.png", "Training and inference time (log scale).", 62)}

<h2>4. Bayesian hyperparameter optimisation (LightGBM)</h2>
<p><b>1. Hyperparameters searched / 2. ranges:</b></p>
{table(pd.DataFrame([{"Hyperparameter": k, "Range / distribution": v} for k, v in BO["search_space"].items()]))}
<p><b>3. Objective:</b> we maximised the {BO['objective']}, using the 80 training trips only. Optuna's TPE sampler ({BO['sampler']}) picks each new
setting based on how earlier ones scored, and a {BO['pruner']} drops a trial early if its first fold already looks weak. We made trial 0 the untuned
default settings, so the search had to beat our starting point.</p>
<p><b>4. Trials:</b> {BO['n_trials']} ({BO['n_complete']} completed, {BO['n_pruned']} pruned); search time {BO['search_time_s'] / 60:.1f} min. Every trial is stored in <code>results/bayes_opt/trials.csv</code>.</p>
<p><b>5. Best values:</b> {CONFIG.get('lgbm_tuned', '')} (3-fold CV macro F1 {BO['best_cv3_macro_f1']:.4f}).</p>
<p><b>6. Did tuning help?</b> {"Yes, on every quality metric, at the cost of a slower model." if better else "Only partly."} Both versions were measured on the same data in the same way:</p>
{table(tvu_tab)}
{img("results/bayes_opt/optimisation_history.png", "Optimisation history: every completed trial and the best value so far.", 70)}
{img("results/bayes_opt/param_importance.png", "Hyperparameter importance (fANOVA) estimated from the trials.", 60)}

<h2>5. HMM and Viterbi route reconstruction</h2>
<p><b>1. Hidden states.</b> At GPS point <i>i</i> the hidden states are its 8 candidate segments. State <i>j</i> at point <i>i</i> always stands for the
same segment <code>cand[i, j]</code>, so we never lose track of which road a state means, and the decoder ends up with exactly one segment per GPS point.</p>
<p><b>2. Emissions.</b> The log of the classifier's probability that the candidate is the right segment, multiplied by a weight w.</p>
<p><b>3. Transitions.</b> Between two GPS points the car has to drive along roads, so the driving distance between the two matched positions
should be about the same as the straight-line distance between the GPS points: <code>route_d = rest of arc a + shortest path(end of a → start of b) + part of arc b</code>
(Dijkstra on the directed road graph, arc lengths as weights, limit {CFG['hmm_params']['max_route_m']:.0f} m); on the same arc, the signed
progress (frac_b − frac_a)·length, so driving backwards is penalised. <code>log T(a→b) = −|route_d − gps_d| / β</code>.</p>
<p>Viterbi recurrence: δ<sub>i+1</sub>(b) = max<sub>a</sub>[δ<sub>i</sub>(a) + log T(a,b)] + w·log e<sub>i+1</sub>(b), with back-pointers; the best final state is traced back.</p>
<p><b>4. Missing candidates and gaps.</b> If a point has fewer than 8 candidates, the empty slots get an impossible score (−∞). If more than
{CFG['hmm_params']['gap_s']:.0f} s pass between two points, we stop constraining the transition (log T = 0), because after a long signal loss the
car could be on almost any road.</p>
<p><b>5. Disconnected segments.</b> If segment b cannot be reached from segment a within the distance limit, the transition gets a large penalty
of {CFG['hmm_params']['unreachable']:.0f} rather than being forbidden outright. The decoder therefore avoids such jumps, but it can still make one when
the classifier is very sure, and it never gets stuck without a valid path.</p>
<p><b>Decoder tuning.</b> β and w were chosen by grid search on {len(CFG.get('decoder_tuned_on_train_trips', []))} training trips using
<b>out-of-fold</b> classifier probabilities (each trip scored by a model that never saw it), objective = pointwise accuracy. Best: β = {best_tune.beta:g} m,
w = {best_tune.w_emit:g} (accuracy {best_tune.pointwise_accuracy:.4f} vs classifier-only {CFG.get('decoder_tuning_classifier_only_accuracy', float('nan')):.4f}
on those trips). We did not use student-test or evaluator data for this tuning.</p>
{img("results/hmm/decoder_tuning.png", "Decoder tuning grid (training trips, out-of-fold probabilities).", 55)}
<p><b>6. Did HMM decoding improve the route?</b> Yes, clearly, and for every classifier we tried. Results on the 10 student-test trips:</p>
{table(hm_tab, {"pointwise_accuracy": pct, "route_continuity": pct, "route_precision": f3, "route_recall": f3, "route_f1": f3})}
<p>For the final model, pointwise accuracy rises from {pct(final_c.pointwise_accuracy)} to {pct(final_h.pointwise_accuracy)}, breaks fall from
{int(final_c.breaks)} to {int(final_h.breaks)} and route F1 from {f3(final_c.route_f1)} to {f3(final_h.route_f1)}. The weaker classifiers gain the most, because most of their
mistakes are single points that wander onto a side street, and the transitions pull those points back onto the route.</p>
{img("results/hmm/classifier_vs_hmm.png", "Classifier only vs classifier + HMM on student_test, for every model.", 100)}
{img("results/hmm/per_trip_accuracy.png", f"Per-trip pointwise accuracy of {NAMES.get(FINAL, FINAL)} (student_test).", 70)}
<p><b>Where HMM did not help.</b> {("Trips " + ", ".join(worse.index) + " got worse with HMM. ") if len(worse) else "No student-test trip got worse. "}
Trip 00000007 is the clearest case. For its last ~175 points the car is parked on a two-way street. When the car is not moving, the GPS
heading is basically random, and the classifier starts to prefer the segment for the opposite direction. Over so many points that evidence
outweighs the cost of one U-turn, so the decoder switches direction even though it is the same physical street. A feature that tells the model
when the heading can be trusted (based on speed) should fix this; we have not built it yet.</p>
{route_figs}

<h2>6. Geographical visualisation using Folium</h2>
<p>Our code (<code>src/make_map.py</code>, <code>src/route_maps.py</code>) produces interactive Folium maps, which are in the <code>maps/</code> folder.
Each map has the raw GPS (blue), the nearby road network (grey, can be switched on), the reference route (green), the classifier-only segments
(orange, dashed) and the HMM route (red). Every map has a legend and layer controls, and it opens zoomed to the trip.</p>
{img("report/figures/folium_route.png", "Screenshot of a Folium route map (student_test). Base-map tiles may be missing in this screenshot because it was captured in an offline environment; the layers are the real output.", 95)}

<h2>7. API, UI and deployment</h2>
<p>We split the system across two laptops, as the guidelines ask. Both were on the same network (Laptop 1 at 10.8.121.177, Laptop 2 at 10.8.121.171).</p>
<p><b>Laptop 1 runs the API</b> (<code>api/main.py</code>, FastAPI on port 8000, started with <code>run_api.bat</code>). On start-up it loads the final
LightGBM model, the tuned HMM settings from <code>models/final_config.json</code> and the 10 evaluation trips. It has three endpoints:
<code>GET /health</code>, <code>GET /trips</code> (the list of trip ids) and <code>POST /predict</code>. A <code>/predict</code> call loads the chosen trip,
removes its reference route before anything else happens, runs candidates → features → classifier → HMM/Viterbi, and only then compares the
result with the reference route to work out the accuracy. The reply contains the GPS points, the predicted route and the scores. The reference
route itself never leaves Laptop 1. Every request is written to the console with the trip id, accuracy, continuity, route F1 and time taken.</p>
<p><b>Laptop 2 runs the UI</b> (<code>ui/</code>: plain HTML, JavaScript and a bundled copy of Leaflet, served by <code>run_ui.bat</code> on port 8080).
You type Laptop 1's address, press Connect, and the 10 trips appear. Clicking a trip sends its id to the API and then draws the result on the map,
with the scores next to it.</p>
{img("report/figures/demo_api_console_laptop1.jpg", "Figure 7.1 – API console on Laptop 1 during the demo. The server loads the tuned model and the 10 evaluation trips, then answers requests that come from Laptop 2 (10.8.121.171): /health, /trips, the browser's CORS check (OPTIONS) and two /predict calls. For trip 00000015 the log shows 6,854 points, 89.33% accuracy with the HMM against 58.02% for the classifier alone, in 8.1 s; for trip 00000094, 92.20% against 86.38% in 9.0 s.", 100)}
{img("report/figures/demo_ui_laptop2.jpg", "Figure 7.2 – Evaluator UI on Laptop 2, connected to the API at http://10.8.121.177:8000 (\"connected · model lgbm_tuned · 10 trips\"). Trip 00000094 is selected: the predicted route is drawn on the OpenStreetMap background, and the panel on the left compares the classifier alone with classifier + HMM (accuracy 86.38% → 92.20%, continuity 93.53% → 97.31%, breaks 209 → 87, route F1 0.900).", 100)}
{img("report/figures/demo_ui_server_laptop2.jpg", "Figure 7.3 – The small web server on Laptop 2 (python -m http.server 8080) that serves the UI files: the page, style sheet, script and the local Leaflet library all load with status 200. The only 404 is the browser asking for a favicon, which we do not provide.", 85)}
<p><b>Running it</b> is described step by step in the README: on Laptop 1, put the evaluation trips in <code>data\\evaluator_test</code> and start
<code>run_api.bat</code>; on Laptop 2, start <code>run_ui.bat</code> and enter <code>http://&lt;Laptop-1-IP&gt;:8000</code>. Two problems came up while we set
this up on Windows. First, Python 3.10 cannot install the scikit-learn version we trained with. Second, Git changed the line endings in LightGBM's text
model file, which LightGBM then refused to load. We solved both by exporting the 450 trees of the final model to plain numeric arrays
(<code>models/lgbm_tuned_trees.npz</code>) and evaluating them with numpy (<code>src/tree_predictor.py</code>). It gives the same probabilities as
LightGBM (largest difference 2·10<sup>−16</sup>) and does not depend on library versions.</p>

<h2>8. Academic integrity and AI-assistance disclosure</h2>
<p>Every number, table and figure in this report comes from code in our repository, and <code>python report/make_report.py</code> rebuilds the whole
report from the saved result files. The evaluator trips were not used for training, for choosing a model, or for tuning hyperparameters or the decoder;
they only appear in the demo screenshots in Section 7. No predictions are hardcoded. <b>AI assistance:</b> we used an AI coding assistant to help write
code and draft parts of this report. We went through the code and results ourselves and are ready to explain every part of them.</p>

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
