"""k-nearest neighbours on the hand-made features (slide 7).

Predicts the mean label of the k training windows most similar to the new one. Features are
standardised first, because distances need comparable units; k stays small.
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.evaluation import tune

INPUT = "features"
GRID = {"kneighborsregressor__n_neighbors": [3, 5, 10]}


def fit(X, y, groups):
    return tune(make_pipeline(StandardScaler(), KNeighborsRegressor()), GRID, X, y, groups)


def predict(model, X):
    return model.predict(X)
