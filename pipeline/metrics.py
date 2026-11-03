"""How good are the patient-level predictions? MAE, RMSE, Spearman and R², each with a 95% confidence interval.

MAE: mean absolute error, in mBI points (lower is better). RMSE: like MAE but big errors count more.
Spearman: do the predictions put the patients in the right order? (1 = perfect order).
R²: share of the differences between patients that the model explains (1 = all, 0 = like the mean, < 0 = worse).
95% confidence interval: the patients are resampled with replacement many times (bootstrap) and
the score is recomputed each time; the middle 95% of those scores is the interval.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import config


def scores(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "Spearman": spearmanr(y_true, y_pred).statistic, "R2": r2_score(y_true, y_pred)}


def scores_with_ci(predictions):
    """Scores of one model plus <score>_low and <score>_high (95% bootstrap interval over patients)."""
    y_true, y_pred = predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()
    rng = np.random.default_rng(config.SEED)
    resamples = [rng.integers(0, len(y_true), len(y_true)) for _ in range(config.N_BOOTSTRAP)]
    boot = [scores(y_true[rows], y_pred[rows]) for rows in resamples]
    out = {}
    for name, value in scores(y_true, y_pred).items():
        low, high = np.nanpercentile([b[name] for b in boot], [2.5, 97.5])
        out |= {name: value, f"{name}_low": low, f"{name}_high": high}
    return out
