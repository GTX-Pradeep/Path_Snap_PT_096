"""Model 5 — Multi-layer perceptron (neural-network family).
Two hidden layers (64, 32), ReLU, Adam, early stopping on a 10% validation split.
Produces an epoch-wise loss curve. Fitted on a 500k-row subsample for training time."""
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPEC = dict(
    name="mlp", family="Neural network", max_train_rows=500_000,
    make=lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
        max_iter=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
        random_state=42)),
)
