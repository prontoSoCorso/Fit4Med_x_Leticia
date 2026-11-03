"""MiniRocket + Ridge (slides 8-9): thousands of random filters, then a linear regression.

MiniRocket slides about 10,000 small fixed filters along each window and keeps, for each one, the
share of positive values: about 10,000 numbers per window, and the filters are never trained.
A Ridge regression (linear, small weights) then maps those numbers to the label.
"""
from aeon.transformations.collection.convolution_based import MiniRocket

import config
from pipeline.evaluation import fit_ridge

INPUT = "windows"


def fit(X, y, groups):
    rocket = MiniRocket(random_state=config.SEED, n_jobs=-1).fit(X)  # learns value ranges from training windows only
    return rocket, fit_ridge(rocket.transform(X), y, groups)


def predict(model, X):
    rocket, ridge = model
    return ridge.predict(rocket.transform(X))
