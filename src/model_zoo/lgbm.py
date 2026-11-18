"""Model 4 — LightGBM gradient boosting (boosting family).
Trees are added sequentially to correct earlier errors; histogram-based, so it is fast on
~1.4M rows. is_unbalance handles the 1:7 positive:negative candidate ratio."""
from lightgbm import LGBMClassifier

DEFAULTS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)


def make_lgbm(**overrides):
    return LGBMClassifier(**{**DEFAULTS, **overrides})


SPEC = dict(name="lgbm", family="Boosting", max_train_rows=None, make=make_lgbm)
