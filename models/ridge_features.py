"""Ridge regression on the hand-made features: the 'IMU only' arm of the late-fusion comparison (slide 13).

It uses the same head as late_fusion.py, so the two differ only by the metadata.
"""
from pipeline.evaluation import fit_ridge

INPUT = "features"


def fit(X, y, groups):
    return fit_ridge(X, y, groups)


def predict(model, X):
    return model.predict(X)
