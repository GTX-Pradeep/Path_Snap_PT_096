# PathSnap — Map-Matching & Route Reconstruction (Team PT_096)

**UE24CS352A Machine Learning Hackathon — PES University**

Noisy GPS observations are matched to the road network:

```
raw GPS ──► clean + project to metres ──► candidate road segments (8 nearest arcs per GPS point)
        ──► features per (GPS point, candidate) ──► ML classifier: P(candidate is the true arc)
        ──► HMM / Viterbi over candidates (road connectivity) ──► connected route
        ──► FastAPI backend (Laptop 1) ──► evaluator UI (Laptop 2)
```

## Team

| Member | Responsibility | Files |
|---|---|---|
| **Pradeep** | Core pipeline (data loading, candidate generation, features, training/evaluation framework), **Model 1: Logistic Regression** | `src/data_loader.py`, `src/features.py`, `src/build_dataset.py`, `src/train_models.py`, `src/plot_results.py`, `src/model_zoo/logreg.py`, `notebooks/02_*` |
| **Poojitha** | Dataset exploration (EDA), Folium maps, **Model 2: Random Forest** | `src/eda.py`, `src/make_map.py`, `src/model_zoo/rf.py`, `notebooks/01_*`, `notebooks/03_*`, `results/eda/`, `maps/` |
| **Pandu** | **Model 3: KNN**, **Model 5: MLP neural network** | `src/model_zoo/knn.py`, `src/model_zoo/mlp.py`, `notebooks/04_*`, `notebooks/06_*` |
| **Pavan** | **Model 4: LightGBM**, Bayesian optimisation, HMM/Viterbi, API + UI (upcoming) | `src/model_zoo/lgbm.py`, `notebooks/05_*` |

## Repository layout

```
src/
  data_loader.py      read .track/.nodes/.arcs/.route, clean GPS, lon/lat -> metres
  features.py         candidate generation, feature engineering, per-observation reference labels
  build_dataset.py    builds candidate tables for train and student_test
  train_models.py     GroupKFold CV + student-test evaluation for any model in model_zoo/
  plot_results.py     comparison plots + per-model diagnostics (from real results)
  eda.py              dataset inventory, quality checks, connectivity, EDA plots
  make_map.py         Folium maps (GPS, road network, reference route, predicted route)
  model_zoo/          one file per model (auto-discovered): logreg, rf, knn, lgbm, mlp
notebooks/            Kaggle notebooks (one per model + EDA), each runs the repo code end to end
models/               trained models (*.joblib)
results/              metrics_<model>.json, model_comparison.csv, train_log.txt, plots/, eda/
maps/                 Folium HTML maps
api/, ui/             FastAPI backend and evaluator UI (next phase)
```

## Dataset

| Partition | Trips | Use |
|---|---|---|
| train | 80 | fitting, 5-fold group CV, (later) Bayesian optimisation |
| student_test | 10 | development-stage model comparison |
| evaluator_test | 10 | final evaluation only: **not available to us, never used** |

Raw files per trip (whitespace-separated, no header):
* `.track`: longitude, latitude, timestamp in seconds (about 1 Hz)
* `.nodes`: longitude, latitude (row index = node id)
* `.arcs`: from_node, to_node (row index = arc id, directed)
* `.route`: chronological list of reference arc ids

**EDA findings** (`src/eda.py`, outputs in `results/eda/`):
* No missing values or invalid coordinates. Timestamps are strictly increasing.
* Median sampling interval is 1 s. Some trips have very large gaps: trip 90 has a gap of about 42 h and trip 78 one of about 14.7 h (`anomalies_top10.csv`).
* Every trip has its own road network (thousands of nodes and arcs) with no isolated nodes.
* Consecutive reference arcs are almost always connected head-to-tail.
* **Reference-route limitation:** in trips 33 and 54, the reference route lies more than 100 m from most GPS points. It does not describe where those GPS points are.

## Method

1. **Projection.** Lon/lat are angles, so degree distances are not physical. Each trip is projected to a local equirectangular plane in metres, centred on the trip, which is accurate at city scale.
2. **Candidate generation.** Every arc is sampled every 15 m and the samples go into a KD-tree. For each GPS point, the 8 nearest *distinct* arcs, by exact point-to-segment distance, are its candidates.
3. **Reference label per GPS point.** `.route` lists arcs in order with no timestamps. GPS points are aligned to it with a **monotonic dynamic-programming alignment**: the route index never moves backwards, and total point-to-arc distance is minimised. A candidate's label is 1 if it equals the aligned reference arc.
4. **Label cleaning (train only).** Training observations whose aligned reference arc is more than 50 m away are dropped, which removes about 11% of observations. This mainly covers trips 33 and 54, where route and GPS disagree. After cleaning, the candidate miss rate on train falls from 8.2% to 0.85%. **Student-test is never filtered.** Its candidate miss rate is 0.35%.
5. **Features per (GPS point, candidate arc):**

| Feature | Why |
|---|---|
| `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio_min` | closeness, absolute and relative to the other candidates |
| `heading_diff`, `heading_diff_cos` | GPS heading vs arc direction; separates the two directions of a road and crossing streets |
| `proj_frac` | where along the segment the point projects |
| `arc_len` | short connector arcs vs long road segments |
| `speed` | local motion |
| `n_cand_30m` | how ambiguous the area is (dense junctions) |
| `dist_prev`, `dist_next` | temporal context: is the neighbouring GPS point also near this arc? |
| `out_degree`, `in_degree` | junction structure |

6. **Classification.** Binary classification per candidate. Evaluation:
   * **Observation-level Top-1 accuracy**: per GPS point, take the candidate with the highest probability and compare it with the reference arc. This is the **primary metric**, because it is exactly the map-matching decision and it is what the HMM consumes.
   * Candidate-level Macro F1, precision, recall, log loss, and the confusion matrix.
   * **5-fold GroupKFold by trip**, so no trip contributes to both training and validation folds.

## Results (student_test = 10 trips; CV = 5-fold GroupKFold on 80 train trips)

| # | Model | Family | Key configuration | CV Macro F1 | Train Macro F1 | Test Top-1 Acc | Test Macro F1 | Test Precision | Test Recall | Test Log Loss | Train time | Inference time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Logistic Regression | Linear / probabilistic | C=1, L2, balanced weights, scaled | 0.826 ± 0.013 | 0.826 | **84.79%** | 0.829 | 0.583 | 0.936 | 0.228 | 7.1 s | 0.02 s |
| 2 | Random Forest | Ensemble (bagging) | 100 trees, depth 12, leaf 5, 20% bootstrap | 0.894 ± 0.020 | 0.897 | **85.49%** | 0.897 | 0.736 | 0.942 | 0.140 | 26.8 s | 0.25 s |
| 3 | KNN | Instance-based | k=25, distance-weighted, scaled, 300k rows | 0.901 ± 0.014 | 1.000 | **84.85%** | 0.907 | 0.852 | 0.826 | 0.147 | 0.9 s | 49.70 s |
| 4 | LightGBM | Boosting | 300 trees, lr 0.05, 63 leaves, is_unbalance | 0.889 ± 0.018 | 0.900 | **86.04%** | 0.890 | 0.710 | 0.955 | 0.142 | 10.8 s | 0.61 s |
| 5 | MLP | Neural network | 64-32 ReLU, Adam, early stop, 500k rows | 0.907 ± 0.017 | 0.914 | **85.65%** | 0.916 | 0.868 | 0.841 | 0.099 | 22.8 s | 0.09 s |

**Observations**
* All models land at **84.8–86.0% Top-1 accuracy**. Nearest-candidate geometry already settles most points; the remaining errors are ambiguous junctions and parallel roads, which is what the HMM stage targets.
* **LightGBM** has the best Top-1 accuracy (86.04%) and is fast to train. **MLP** has the best CV Macro F1 (0.907) and the lowest log loss (0.099), so its probabilities are the best calibrated, which matters for HMM emissions.
* **Logistic Regression** underfits: train F1 ≈ CV F1 ≈ 0.83, so the linear boundary is the limit.
* **KNN** overfits: train F1 = 1.000 (distance weighting memorises training rows) vs CV 0.901. Inference is also about 50 s for the 10 test trips, so it is unsuitable for the API.
* Tree models (RF, LightGBM) trade precision for recall (`balanced` / `is_unbalance` weighting), but the Top-1 argmax is unaffected by the threshold.
* The final model will be chosen on Top-1 accuracy plus probability quality and inference time. LightGBM and MLP are the candidates for Bayesian optimisation.
* `models/knn.joblib` (43 MB) is not committed because it is over GitHub's upload limit. It retrains in about 1 s with `python train_models.py knn`.

Plots are in `results/plots/` (`model_comparison.png`, `cv_folds.png`, `compute_cost.png`, `<model>_diagnostics.png`). Raw numbers are in `results/metrics_<model>.json` and the training log is `results/train_log.txt`.

## How to run

### Locally
```bash
pip install -r requirements.txt
```
Point the code at the dataset, either by creating `data/` (with `data/train/...` and `data/student_test/...`) or by setting:
```bash
export PATHSNAP_DATA="/path/to/Student folder"
```
```bash
cd src
python eda.py                                   # EDA tables + plots -> results/eda/
python build_dataset.py                         # candidate tables   -> results/*.parquet
python train_models.py logreg rf knn lgbm mlp   # train + evaluate   -> models/, results/
python plot_results.py                          # comparison plots   -> results/plots/
python make_map.py train 00000010               # Folium map         -> maps/
```

### On Kaggle
1. Upload the provided `Student folder` as a Kaggle Dataset.
2. Open any notebook from `notebooks/` in Kaggle.
3. Choose *Add Input*, select that dataset, and set *Internet* to **On**.
4. Run all cells. The notebook clones this repo and runs the same code.

## Reproducibility and integrity
* Random seed is 42 everywhere. Every number above comes from `results/metrics_*.json`, which `train_models.py` writes.
* The evaluator-testing data and labels are never used. No predictions are hardcoded.
* The dataset is not committed (`.gitignore`). Large intermediate parquet files are regenerated by `build_dataset.py`.

## Next phase
Bayesian hyperparameter optimisation (Optuna TPE) on the selected model, HMM/Viterbi route reconstruction (classifier-only vs classifier+HMM), FastAPI backend, evaluator UI, and the technical report.
