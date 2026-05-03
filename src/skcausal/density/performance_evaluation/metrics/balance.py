import numpy as np
import pandas as pd
import polars as pl

from skcausal.density.performance_evaluation.metrics.base import BaseDensityMetric

__all__ = ["AverageAbsoluteWeightedCorrelationMetric"]


class AverageAbsoluteWeightedCorrelationMetric(BaseDensityMetric):
    """Average absolute weighted correlation between covariates and treatment."""

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        super().__init__()

    def _evaluate(self, density_estimator, X, t):
        X_array = _to_2d_float_array(X, name="X")
        t_array = _to_2d_float_array(t, name="t")

        if X_array.shape[1] == 0:
            raise ValueError("X must contain at least one covariate column.")
        if t_array.shape[1] != 1:
            raise ValueError(
                "AverageAbsoluteWeightedCorrelationMetric requires a single treatment column."
            )
        if X_array.shape[0] != t_array.shape[0]:
            raise ValueError("X and t must have the same number of rows.")

        density = np.asarray(
            density_estimator.predict_density(X, t), dtype=float
        ).reshape(-1)
        if density.shape[0] != X_array.shape[0]:
            raise ValueError(
                "predict_density(X, t) must return one density value per row."
            )

        weights = 1.0 / np.clip(density, self.epsilon, None)
        treatment = t_array[:, 0]

        absolute_correlations = np.array(
            [
                abs(_weighted_correlation(X_array[:, column], treatment, weights))
                for column in range(X_array.shape[1])
            ],
            dtype=float,
        )

        finite_correlations = absolute_correlations[np.isfinite(absolute_correlations)]
        if finite_correlations.size == 0:
            return np.nan

        return finite_correlations.mean()


def _weighted_correlation(x, t, w):
    mask = np.isfinite(x) & np.isfinite(t) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return np.nan

    x = x[mask]
    t = t[mask]
    w = w[mask]

    weight_sum = w.sum()
    if weight_sum <= 0:
        return np.nan

    x_centered = x - np.sum(w * x) / weight_sum
    t_centered = t - np.sum(w * t) / weight_sum

    numerator = np.sum(w * x_centered * t_centered)
    denominator = np.sqrt(np.sum(w * x_centered**2) * np.sum(w * t_centered**2))

    if denominator <= 0:
        return np.nan

    return numerator / denominator


def _to_2d_float_array(data, *, name):
    if isinstance(data, pl.DataFrame):
        array = data.to_numpy()
    elif isinstance(data, pl.Series):
        array = data.to_numpy().reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        array = data.to_numpy()
    elif isinstance(data, pd.Series):
        array = data.to_numpy().reshape(-1, 1)
    else:
        array = np.asarray(data)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, received shape {array.shape}.")

    try:
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values.") from exc
