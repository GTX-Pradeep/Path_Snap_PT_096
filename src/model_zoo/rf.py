"""Model 2 — Random Forest (ensemble / bagging family).
Depth and leaf-size limits plus 20% bootstrap samples per tree keep it from memorising
individual trips and keep training time manageable on ~1.4M candidate rows."""
from sklearn.ensemble import RandomForestClassifier

SPEC = dict(
    name="rf", family="Ensemble (bagging)", max_train_rows=None,
    make=lambda: RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=5, max_samples=0.2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42),
)
