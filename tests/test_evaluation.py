"""The evaluation never trains on the test patient, and the scores behave as expected."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline.evaluation import evaluate, split_validation
from pipeline.metrics import scores_with_ci


def test_no_leakage_between_patients():
    groups = np.repeat(["A", "B", "C", "D", "E"], 4)  # 5 patients, 4 windows each
    X = np.repeat(np.arange(5), 4).reshape(-1, 1)       # each window carries its patient's number
    y = np.arange(20, dtype=float)
    # A spy model: it remembers which patients it saw in training and predicts 1 if the test patient is one of them
    spy = SimpleNamespace(fit=lambda X, y, groups: set(X[:, 0]),
                          predict=lambda seen, X: np.array([float(x in seen) for x in X[:, 0]]))
    predictions = evaluate(spy, X, y, groups)
    assert len(predictions) == 5 and (predictions["y_pred"] == 0).all()
    # Early stopping: validation patients are whole patients, never also in training
    fit_rows, validation_rows = split_validation(groups)
    assert len(validation_rows) and not set(groups[fit_rows]) & set(groups[validation_rows])


def test_scores_and_intervals():
    perfect = pd.DataFrame({"y_true": np.arange(10.0), "y_pred": np.arange(10.0)})
    s = scores_with_ci(perfect)
    assert np.allclose([s["MAE"], s["R2"], s["Spearman"]], [0, 1, 1])
    tiny = 1e-9  # computer rounding
    assert all(s[f"{k}_low"] - tiny <= s[k] <= s[f"{k}_high"] + tiny for k in ["MAE", "RMSE", "Spearman", "R2"])
