# PathSnap — Map-Matching & Route Reconstruction (Team PT_096)

**UE24CS352A Machine Learning Hackathon — PES University**

PathSnap snaps noisy GPS points onto the road network and rebuilds the route the vehicle actually drove.

```
raw GPS ─► clean + project to metres ─► 8 candidate road segments per GPS point ─► 14 features
        ─► LightGBM (Bayesian-tuned): P(candidate is the true road)
        ─► HMM / Viterbi over candidates (road connectivity + shortest paths) ─► one connected route
        ─► FastAPI backend (Laptop 1) ◄──► evaluator UI in the browser (Laptop 2)
```

**Final system on the 10 student-test trips:** pointwise accuracy **92.43%** (classifier only 86.37%),
route continuity **98.14%** (classifier only 93.54%), disconnected steps 1237 → 357, route F1 0.920.

📄 Technical report: [`report/PathSnap_Technical_Report.pdf`](report/PathSnap_Technical_Report.pdf)

---

## 1. Run the demo (two laptops)

| | Laptop 1 — **API** | Laptop 2 — **UI** |
|---|---|---|
| Runs | model + HMM/Viterbi + road networks + scoring (FastAPI, port 8000) | browser interface (static files, port 8080) |
| Needs | this repo, Python 3.11–3.13, `requirements.txt`, the evaluation trips | this repo (only the `ui/` folder is used), Python (or any static web server) |

### Laptop 1 — API
1. Install:
   ```bash
   pip install -r requirements.txt
   ```
2. Put the evaluation trips in `data/evaluator_test/` (one folder per trip: `<id>/<id>.track|.nodes|.arcs|.route`).
   A different location works too: `set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_test` (Windows) or `export PATHSNAP_EVAL_DIR=...`.
3. Start it: double-click **`run_api.bat`** (Windows) or run `./run_api.sh` (Mac/Linux).
   The console shows `Loaded model lgbm_tuned ...` and then logs **every request** (trip, accuracy, continuity, route F1, time).
4. Find this laptop's IP address: `ipconfig` (Windows, "IPv4 Address") or `ifconfig` / `ip a` (Mac/Linux).
   Allow Python through the Windows firewall when asked (private network).
5. Check it from a browser: `http://<laptop-1-IP>:8000/health` and `http://<laptop-1-IP>:8000/docs` (interactive API docs).

### Laptop 2 — UI
1. Start it: double-click **`run_ui.bat`** (or `./run_ui.sh`), then open `http://localhost:8080`.
2. Type `http://<laptop-1-IP>:8000` in the **API** box and click **Connect**. The 10 evaluation trips appear.
3. Click a trip. The UI sends its id to the API (`POST /predict`). The API runs the whole pipeline and returns the predicted
   route and metrics. The map shows raw GPS (blue), the predicted route (red) and classifier-only arcs (orange, dashed). Each
   layer can be toggled.
4. Show the evaluator the API console on Laptop 1: each click appears there as a log line.

Both laptops must be on the same network (e.g. one phone hotspot). The map background needs internet (OpenStreetMap tiles);
without it, the routes still draw on a blank background. Leaflet is bundled in `ui/vendor/`.
Single-laptop test: start the API, then open `http://localhost:8000/ui/`.

### API reference
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, loaded model, decoder parameters, number of trips |
| GET | `/trips` | – | `{"trips": [{"trip_id", "n_gps"}, ...]}`, no reference data |
| POST | `/predict` | `{"trip_id": "00000059"}` | GPS points, predicted route geometry + arc ids, classifier-only arcs, metrics (HMM and classifier only), timing |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000059\"}"
```
**Ground truth stays on the server.** `/predict` removes the `.route` before the trip enters the pipeline; afterwards it is used
only to compute the accuracy, and it is never returned to the UI.

---

## 2. Results

### 2.1 Models (CV = 5-fold GroupKFold by trip on 80 train trips; test = 10 student-test trips)
| Model | Family | CV macro F1 | Test macro F1 | Test log loss | Top-1 accuracy (classifier) | Route continuity (classifier) | Accuracy with HMM | Continuity with HMM | Train time | Inference (10 trips) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | 0.826 ± 0.013 | 0.829 | 0.228 | 84.79% | 92.15% | **90.29%** | 98.11% | 7.5 s | 0.02 s |
| Random Forest | Ensemble (bagging) | 0.894 ± 0.020 | 0.897 | 0.140 | 85.49% | 92.45% | **92.28%** | 98.09% | 28.5 s | 0.20 s |
| KNN | Instance-based | 0.901 ± 0.014 | 0.907 | 0.147 | 84.85% | 91.94% | **90.76%** | 98.04% | 0.6 s | 54.44 s |
| LightGBM | Boosting | 0.889 ± 0.018 | 0.890 | 0.142 | 86.04% | 93.46% | **92.41%** | 98.11% | 10.3 s | 0.54 s |
| MLP | Neural network | 0.907 ± 0.017 | 0.916 | 0.099 | 85.65% | 93.15% | **92.37%** | 98.10% | 21.2 s | 0.05 s |
| **LightGBM (tuned, final)** | Boosting (Bayesian-tuned) | 0.913 ± 0.018 | 0.918 | 0.092 | 86.37% | 93.54% | **92.43%** | 98.14% | 18.3 s | 1.63 s |

* **Primary metric: observation-level Top-1 accuracy.** For each GPS point, the highest-scoring candidate is compared with the reference arc.
  This is the map-matching decision itself. Log loss is the second criterion, because the HMM uses the probabilities.
* **Why LightGBM:** best Top-1 accuracy of the untuned models, fast training, sub-second inference and a 1 MB model file, which suits the API and the decoder.
  Logistic Regression underfits, KNN overfits (train F1 1.000) and is slow, and the MLP and Random Forest are close but slower.

### 2.2 Bayesian optimisation (Optuna TPE, LightGBM)
30 trials (23 complete, 7 pruned by the median pruner) took 15 minutes. Objective: mean macro F1, 3-fold GroupKFold by trip.
Best: `n_estimators=450`, `learning_rate=0.01414`, `num_leaves=237`, `min_child_samples=32`, `subsample=0.7158`, `colsample_bytree=0.7388`, `reg_lambda=0.003947`, `reg_alpha=1.705`, `is_unbalance=False`.

| | Untuned | Tuned |
|---|---|---|
| CV macro F1 | 0.8887 | **0.9132** |
| Student-test Top-1 accuracy | 86.04% | **86.37%** |
| Student-test macro F1 | 0.8897 | **0.9185** |
| Log loss | 0.1419 | **0.0918** |
| Inference time (10 trips) | 0.61 s | 1.63 s |

### 2.3 HMM / Viterbi
* **States:** the 8 candidate arcs of each GPS point. **Emission:** w · log P(classifier).
* **Transition:** `−|network distance − GPS distance| / β`. Network distance comes from Dijkstra on the directed road graph. On the same arc, progress is signed, so driving backwards is penalised.
* **Unreachable arcs:** a finite penalty of -30, so a path always exists.
* **GPS gaps:** if a gap is longer than 30 s, the transition is unconstrained.
* **Tuned decoder settings:** β = 1, w = 2. They were chosen by grid search on 20 *training* trips using out-of-fold probabilities.
* **Result:** the HMM improves every classifier by about 6 points of pointwise accuracy and cuts disconnected steps by about 70%.
* **Failure case:** trip 00000007 ends with the car parked on a two-way road. At standstill the GPS heading is noise, so the decoder flips to the opposite-direction arc. This is analysed in the report.

Plots: `results/plots/`, `results/bayes_opt/`, `results/hmm/`. Interactive maps: `maps/`.

---

## 3. Team and contributions
| Member | Phase 1 (Review 1) | Phase 2 (final) |
|---|---|---|
| **Pradeep** | core pipeline: data loading, candidates, features, training framework; Model 1 Logistic Regression | prediction pipeline, **FastAPI backend**, deployment scripts, README, technical report |
| **Poojitha** | EDA, Folium maps; Model 2 Random Forest | **evaluator UI**, Folium route maps (classifier vs HMM) |
| **Pandu** | Model 3 KNN, Model 5 MLP | **Bayesian hyperparameter optimisation** (Optuna), tuned-vs-untuned comparison |
| **Pavan** | Model 4 LightGBM | **HMM / Viterbi decoder**, decoder tuning, classifier-vs-HMM experiments |

## 4. Repository layout
```
api/main.py              FastAPI backend (Laptop 1)
ui/                      evaluator UI: index.html, app.js, style.css, vendor/leaflet (Laptop 2)
src/
  data_loader.py         read .track/.nodes/.arcs/.route, clean GPS, lon/lat -> metres
  features.py            candidate generation, 14 features, monotonic DP reference labels
  build_dataset.py       candidate tables for train and student_test
  model_zoo/             one file per model: logreg, rf, knn, lgbm, mlp (auto-discovered)
  train_models.py        GroupKFold CV + student-test evaluation, saves models/metrics
  plot_results.py        comparison plots + per-model diagnostics
  bayes_opt.py           Optuna TPE search on LightGBM + tuned-vs-untuned
  hmm.py                 HMM states/emissions/transitions + Viterbi + route metrics
  hmm_experiments.py     out-of-fold decoder tuning, classifier vs HMM, writes models/final_config.json
  pipeline.py            end-to-end inference used by the API (+ server-side scoring)
  eda.py, make_map.py    EDA tables/plots, Folium EDA map
  route_maps.py          Folium maps of predicted routes (+ PNGs)
models/                  lgbm_tuned.joblib (final), other models, final_config.json
notebooks/               executed notebooks (*_executed.ipynb) for EDA, each model, Bayesian opt, HMM
results/                 metrics_*.json, model_comparison.csv, train_log.txt, eda/, plots/, bayes_opt/, hmm/
maps/                    Folium HTML maps
report/                  technical report (PDF + HTML) and its generator
run_api.bat|.sh, run_ui.bat|.sh
```

## 5. Reproduce everything
Put the provided data in `data/train/` and `data/student_test/`, or set `PATHSNAP_DATA="/path/to/Student folder"`.
```bash
cd src
python eda.py                  # EDA tables + plots           -> results/eda/
python build_dataset.py        # candidates + features         -> results/*.parquet
python train_models.py         # 5 models: CV + student-test   -> models/, results/metrics_*.json
python bayes_opt.py 30         # Bayesian optimisation         -> results/bayes_opt/, models/lgbm_tuned.joblib
python hmm_experiments.py      # decoder tuning + comparison   -> results/hmm/, models/final_config.json
python plot_results.py         # comparison plots              -> results/plots/
python route_maps.py           # Folium route maps             -> maps/
python ../report/make_report.py  # technical report (HTML + PDF) from the result files
```
Seed 42 everywhere. If a saved model fails to load because of a different scikit-learn or LightGBM version, install the
pinned versions from `requirements.txt`, or retrain with the commands above.

## 6. Data and integrity rules followed
* Train (80 trips): fitting, cross-validation, Bayesian optimisation and decoder tuning. Student-test (10 trips): model comparison only.
* Evaluator-test (10 trips): used only by the API at demonstration time. It was never used for training, model selection or any tuning.
* GroupKFold by trip, so no trip appears in both training and validation.
* Training observations whose reference arc is more than 50 m away are dropped (trips 33 and 54 have reference routes that don't follow the GPS). Student-test is never filtered.
* No hardcoded predictions. The reference route is removed before prediction and is never sent to the UI.
* The dataset is not committed (`.gitignore`). Large intermediate parquet files are rebuilt by `build_dataset.py`.
  `models/knn.joblib` (43 MB) is regenerated by `python train_models.py knn`.
