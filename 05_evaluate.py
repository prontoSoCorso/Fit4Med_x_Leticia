"""Step 5: evaluate the models with leave-one-patient-out; each model is one file in models/.

Reads:  data/windows/ (windows.csv, features.csv, windows.npy) and data/embeddings/<model>.npy
Writes: results/predictions/<model>.csv  (one row per patient: true and predicted label)
Usage:  python 05_evaluate.py              -> every model in MODELS_ENABLED (config.py)
        python 05_evaluate.py svr knn      -> only these models
"""
import importlib
import sys

import numpy as np
import pandas as pd

import config
from pipeline.evaluation import evaluate
from pipeline.metrics import scores_with_ci


def metadata_matrix(info):
    """Metadata columns that exist; text columns (hemisphere) become 0/1 columns. None if there are none."""
    columns = [c for c in config.METADATA if c in info]
    return pd.get_dummies(info[columns], dtype=float).to_numpy() if columns else None


def model_input(kind, name, info):
    """The data a model asks for with its INPUT (one row per window)."""
    if kind in ("features", "features+metadata"):
        features = pd.read_csv(config.WINDOWS_DIR / "features.csv").to_numpy()
        if kind == "features":
            return features
        metadata = metadata_matrix(info)
        return None if metadata is None else np.hstack([features, metadata])
    if kind == "metadata":
        return metadata_matrix(info)
    if kind == "windows":
        return np.load(config.WINDOWS_DIR / "windows.npy")
    if kind == "embeddings":
        path = config.EMBEDDINGS_DIR / f"{name}.npy"
        if not path.exists():
            sys.exit(f"{path} not found: run 04_embed_foundation.py {name} first.")
        return np.load(path)
    sys.exit(f"models/{name}.py has an unknown INPUT '{kind}'.")


def main():
    info_path = config.WINDOWS_DIR / "windows.csv"
    if not info_path.exists():
        sys.exit(f"{info_path} not found: run 03_make_windows.py first.")
    info = pd.read_csv(info_path)
    y, groups = info["label"].to_numpy(dtype=float), info["subject"].to_numpy()
    out_dir = config.RESULTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in sys.argv[1:] or config.MODELS_ENABLED:
        model = importlib.import_module(f"models.{name}")  # the file models/<name>.py
        X = model_input(model.INPUT, name, info)
        if X is None:
            print(f"{name}: skipped, labels.csv has none of the metadata columns {config.METADATA}")
            continue
        print(f"{name}: leave-one-patient-out on {len(set(groups))} patients ", end="")
        predictions = evaluate(model, X, y, groups)
        predictions.to_csv(out_dir / f"{name}.csv", index=False)
        s = scores_with_ci(predictions)
        print(f"{name}: MAE {s['MAE']:.1f} [{s['MAE_low']:.1f}-{s['MAE_high']:.1f}]  "
              f"R2 {s['R2']:.2f}  Spearman {s['Spearman']:.2f}")


if __name__ == "__main__":
    main()
