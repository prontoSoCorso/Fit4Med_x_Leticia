"""MiniRocket + Ridge (slides 8-9): thousands of random filters, then a linear regression.

MiniRocket slides about 10,000 small fixed filters along each window and keeps, for each one, the
share of positive values: about 10,000 numbers per window, and the filters are never trained.
A Ridge regression (linear, small weights) then maps those numbers to the label.
"""
import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from pipeline.evaluation import tune

INPUT = "windows"
GRID = {"ridge__alpha": np.logspace(-1, 3, 5)}  # 0.1 ... 1000: 10,000 features need a strong penalty


def fit(X, y, groups):
    rocket = MiniRocket(random_state=config.SEED, n_jobs=-1).fit(X)  # learns value ranges from training windows only
    ridge = tune(make_pipeline(StandardScaler(), Ridge()), GRID, rocket.transform(X), y, groups)
    return rocket, ridge


def predict(model, X):
    rocket, ridge = model
    return ridge.predict(rocket.transform(X))
