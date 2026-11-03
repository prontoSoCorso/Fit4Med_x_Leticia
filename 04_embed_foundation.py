"""Step 4: describe every window with the frozen foundation models (Chronos-2, MOMENT, MOIRAI).

An embedding is a list of numbers with which a large pretrained model describes a window. The models
stay frozen (they never learn from our data), so this runs once for all windows without leakage;
05_evaluate.py then trains a Ridge regression on the embeddings, patient fold by patient fold.
Reads:  data/windows/windows.npy
Writes: data/embeddings/<model>.npy  (one row of numbers per window)
Usage:  python 04_embed_foundation.py           -> every model in MODELS_ENABLED that uses embeddings
        python 04_embed_foundation.py moment    -> only this one
"""
import importlib
import sys
import time

import numpy as np

import config


def main():
    windows_path = config.WINDOWS_DIR / "windows.npy"
    if not windows_path.exists():
        sys.exit(f"{windows_path} not found: run 03_make_windows.py first.")
    windows = np.load(windows_path)
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    models = {name: importlib.import_module(f"models.{name}") for name in sys.argv[1:] or config.MODELS_ENABLED}
    for name, model in models.items():
        if model.INPUT != "embeddings":
            continue
        start = time.time()
        embeddings = model.embed(windows)
        np.save(config.EMBEDDINGS_DIR / f"{name}.npy", embeddings.astype(np.float32))
        print(f"{name}: {embeddings.shape[1]} numbers per window, {time.time() - start:.0f} s -> {config.EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
