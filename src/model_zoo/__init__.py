"""One file per model. Each module defines SPEC = dict(name, family, make, max_train_rows).
train_models.py discovers every module here automatically."""
import importlib
import pkgutil


def discover():
    specs = {}
    for m in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{m.name}")
        if hasattr(mod, "SPEC"):
            specs[mod.SPEC["name"]] = mod.SPEC
    return specs
