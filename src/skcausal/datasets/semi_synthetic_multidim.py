"""Multi-dimensional semi-synthetic DGP driven by a real-outcome prognostic score."""

import numpy as np
import polars as pl
from scipy.special import log_ndtr

_LOG_2PI = np.log(2.0 * np.pi)


def _per_component(name, value, n_components, *, default=None):
    """Broadcast a scalar to every component or validate a length-k sequence."""

    if value is None:
        value = default
    if isinstance(value, str) or np.ndim(value) == 0:
        return [value] * n_components
    values = list(value)
    if len(values) != n_components:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length "
            f"n_treatments={n_components}."
        )
    return values


def _validate_loadings(value, n_components):
    loadings = np.asarray(
        _per_component("confounding_loadings", value, n_components, default=1.0),
        dtype=float,
    )
    if loadings.shape != (n_components,) or not np.isfinite(loadings).all():
        raise ValueError("confounding_loadings must be finite.")
    return loadings


def _validate_levels(value, n_components):
    levels = _per_component("n_levels", value, n_components)
    for entry in levels:
        if entry is None:
            continue
        if isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise TypeError("n_levels entries must be None or integers.")
        if entry < 2:
            raise ValueError("n_levels entries must be at least 2.")
    return [None if entry is None else int(entry) for entry in levels]


def _validate_targets(value, n_components):
    targets = _per_component("target_columns", value, n_components)
    for entry in targets:
        if entry is not None and not isinstance(entry, str):
            raise TypeError("target_columns entries must be None or column names.")
    named = [entry for entry in targets if entry is not None]
    if len(set(named)) != len(named):
        raise ValueError("target_columns must not repeat a column.")
    return targets


def _exchangeable_correlation(n_components, correlation):
    matrix = np.full((n_components, n_components), float(correlation))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _interval_log_mass(lower, upper):
    """``log(Phi(upper) - Phi(lower))`` evaluated on the side with small mass.

    Any pair with ``lower >= upper`` (including equal or equal-infinite
    bounds) has zero mass and returns ``-inf``.
    """

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        left = log_ndtr(upper) + np.log1p(-np.exp(log_ndtr(lower) - log_ndtr(upper)))
        right = log_ndtr(-lower) + np.log1p(
            -np.exp(log_ndtr(-upper) - log_ndtr(-lower))
        )
        selected = np.where(upper <= 0.0, left, right)
    return np.where(lower < upper, selected, -np.inf)


def _gaussian_logpdf(residual, cholesky):
    """Log-density of the trailing axis of ``residual`` under ``N(0, L L')``."""

    residual = np.asarray(residual, dtype=float)
    dimension = residual.shape[-1]
    if dimension == 0:
        return np.zeros(residual.shape[:-1])
    whitened = residual @ np.linalg.inv(cholesky).T
    log_determinant = float(np.log(np.diag(cholesky)).sum())
    return (
        -0.5 * np.sum(whitened**2, axis=-1)
        - log_determinant
        - 0.5 * dimension * _LOG_2PI
    )


def _conditional_gaussian(covariance, continuous, categorical):
    """Regression matrix and covariance of the categorical block given the continuous block."""

    continuous = np.asarray(continuous, dtype=int)
    categorical = np.asarray(categorical, dtype=int)
    if categorical.size == 0:
        return np.zeros((0, continuous.size)), np.zeros((0, 0))
    cov_dd = covariance[np.ix_(categorical, categorical)]
    if continuous.size == 0:
        return np.zeros((categorical.size, 0)), cov_dd
    cov_cc = covariance[np.ix_(continuous, continuous)]
    cov_dc = covariance[np.ix_(categorical, continuous)]
    regression = np.linalg.solve(cov_cc, cov_dc.T).T
    return regression, cov_dd - regression @ cov_dc.T


def _numeric_column(frame, name, label):
    series = frame.get_column(name)
    if not series.dtype.is_numeric():
        raise ValueError(f"The {label} must be numeric.")
    values = series.cast(pl.Float64).to_numpy().astype(float)
    finite = values[np.isfinite(values)]
    if np.unique(finite).size < 2:
        raise ValueError(f"The {label} must have at least two distinct finite values.")
    return values
