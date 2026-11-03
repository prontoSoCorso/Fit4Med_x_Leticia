"""Baseline: always predict the mean label of the training patients. Every model must beat it."""
import numpy as np

INPUT = "features"  # not really used: this baseline never looks at the data


def fit(X, y, groups):
    label_per_patient = dict(zip(groups, y))  # one label per patient, not one per window
    return np.mean(list(label_per_patient.values()))


def predict(model, X):
    return np.full(len(X), model)
