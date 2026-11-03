"""Late fusion (slide 13): IMU features and clinical metadata side by side, then one small Ridge head.

Each source is prepared on its own (hand-made features from the movement; age, BMI and hemisphere
as numbers, hemisphere as 0/1) and they meet only at the end. Missing metadata (e.g. BMI) are filled
with the median of the training patients. Fusion is interesting only if it beats both
metadata_only.py and ridge_features.py (same head, IMU only): the three ablations of slide 13.
"""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.evaluation import RIDGE_ALPHAS, tune

INPUT = "features+metadata"


def fit(X, y, groups):
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    return tune(pipeline, RIDGE_ALPHAS, X, y, groups)


def predict(model, X):
    return model.predict(X)
