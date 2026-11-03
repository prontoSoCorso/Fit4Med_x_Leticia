"""Baseline: Ridge regression on the metadata only (age, BMI, hemisphere; slide 6).

If the IMU models cannot beat it, the movement signals add nothing beyond the metadata.
Ridge = linear regression that keeps the weights small; alpha says how strongly.
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.evaluation import tune

INPUT = "metadata"
GRID = {"ridge__alpha": [0.1, 1, 10, 100]}


def fit(X, y, groups):
    # Missing values -> median of the training patients; then every column gets mean 0 and spread 1
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    return tune(pipeline, GRID, X, y, groups)


def predict(model, X):
    return model.predict(X)
