"""Chronos-2 (Amazon, slide 10), frozen: each window -> embedding, then Ridge regression.

Chronos-2 was pretrained on huge collections of time series; it reads all channels of a window
together. Its official embed() gives, per channel, one vector per patch of 16 frames: we average
the patches and put the channels side by side.
"""
import numpy as np
from chronos import Chronos2Pipeline

import config
from pipeline.evaluation import fit_ridge
from pipeline.torch_training import DEVICE

INPUT = "embeddings"


def embed(windows):
    pipeline = Chronos2Pipeline.from_pretrained(config.FOUNDATION_CHECKPOINTS["chronos2"], device_map=DEVICE)
    per_window, _ = pipeline.embed(windows.astype(np.float32))  # each: channels x (patches + 2 special) x size
    return np.stack([e[:, :-2].mean(dim=1).reshape(-1).float().numpy() for e in per_window])  # drop the 2 special tokens


def fit(X, y, groups):
    return fit_ridge(X, y, groups)


def predict(model, X):
    return model.predict(X)
