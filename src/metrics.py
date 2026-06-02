"""Evaluation metrics for RUL prediction."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 asymmetric scoring function (lower is better).
    Penalizes late predictions more than early ones.
    """
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.sum(score))
