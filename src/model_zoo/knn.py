"""Model 3 — Distance-weighted K-Nearest Neighbours (instance-based family).
Features are standardised (KNN is distance-based). Fitted on a 300k-row random subsample,
because prediction cost grows with the number of stored training rows."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEC = dict(
    name="knn", family="Instance-based", max_train_rows=300_000,
    make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=25, weights="distance", n_jobs=-1)),
)
