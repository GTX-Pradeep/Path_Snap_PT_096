"""Model 1 — Logistic Regression (linear / probabilistic family).
Standardised features + L2-regularised logistic regression, balanced class weights
(only ~1 in 8 candidates is the true arc)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEC = dict(
    name="logreg", family="Linear / probabilistic", max_train_rows=None,
    make=lambda: make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced")),
)
