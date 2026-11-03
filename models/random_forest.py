"""Random Forest on the hand-made features (roadmap, month 2).

Many decision trees, each trained on a random part of the data; the prediction is their average.
Trees do not need standardised features.
"""
from sklearn.ensemble import RandomForestRegressor

import config
from pipeline.evaluation import tune

INPUT = "features"
GRID = {"max_depth": [None, 5]}  # None = each tree grows until its leaves are pure


def fit(X, y, groups):
    forest = RandomForestRegressor(n_estimators=200, random_state=config.SEED, n_jobs=-1)
    return tune(forest, GRID, X, y, groups)


def predict(model, X):
    return model.predict(X)
