"""The one evaluation used for every model: leave-one-patient-out (LOPO).

Each patient in turn is the test patient: the model is trained on all the other patients and then
predicts the windows of the test patient; their mean is the patient's prediction. So a test patient
never helps training, not even through another window, repetition or speed of the same patient.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, LeaveOneGroupOut

import config


def tune(estimator, grid, X, y, groups):
    """Choose the hyperparameters (settings fixed before training, e.g. k in kNN) on training patients only.

    Inner cross-validation by patient: the training patients are split into 5 groups; each setting is
    trained on 4 groups and scored on the 5th, and the setting with the lowest mean error wins.
    """
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    search = GridSearchCV(estimator, grid, cv=folds, scoring="neg_mean_absolute_error")
    return search.fit(X, y, groups=groups).best_estimator_


def split_validation(groups):
    """Rows for training and for validation (early stopping): about 20% of the training patients,
    never a part of one patient's windows."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    return next(splitter.split(groups, groups=groups))


def evaluate(model, X, y, groups):
    """Leave-one-patient-out predictions: one row per patient with the true and the predicted label.

    `model` is a file from models/ with fit(X, y, groups) and predict(fitted, X).
    """
    results = []
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        fitted = model.fit(X[train], y[train], groups[train])
        results.append({"subject": groups[test][0], "y_true": y[test][0],
                        "y_pred": float(np.mean(model.predict(fitted, X[test])))})
        print(".", end="", flush=True)  # one dot per patient, to show progress
    print()
    return pd.DataFrame(results)
