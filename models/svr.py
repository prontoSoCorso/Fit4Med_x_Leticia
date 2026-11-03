"""Support Vector Regression (SVR) on the hand-made features (slide 7).

SVR fits a smooth curve through the data; C sets how hard it tries to fit every training window and
gamma how wiggly the curve may be (RBF kernel). Features are standardised first (mean 0, spread 1).
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pipeline.evaluation import tune

INPUT = "features"
GRID = {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.001]}


def fit(X, y, groups):
    return tune(make_pipeline(StandardScaler(), SVR(kernel="rbf")), GRID, X, y, groups)


def predict(model, X):
    return model.predict(X)
