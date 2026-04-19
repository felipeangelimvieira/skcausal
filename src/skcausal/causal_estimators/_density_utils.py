import numpy as np


def coerce_density_array(density) -> np.ndarray:
    """Return density outputs as a 2D float array."""

    density_array = np.asarray(density, dtype=float)
    if density_array.ndim == 1:
        density_array = density_array.reshape(-1, 1)
    return density_array


def predict_density_array(density_estimator, X, t) -> np.ndarray:
    """Predict density values, defaulting to ones when no estimator is provided."""

    if density_estimator is None:
        return np.ones((len(X), 1), dtype=float)

    return coerce_density_array(density_estimator.predict_density(X, t))


def predict_inverse_density_weight(
    density_estimator, X, t, eps: float = 1e-8
) -> np.ndarray:
    """Convert density predictions into inverse-density sample weights."""

    density = predict_density_array(density_estimator, X, t)
    return 1.0 / np.clip(density, eps, None)


def is_stabilized_density(density_estimator) -> bool:
    """Return whether an estimator predicts a stabilized density ratio."""

    return (
        density_estimator is not None
        and density_estimator.get_tag("density_kind", "conditional") == "stabilized"
    )


def binary_conditional_probabilities(
    density_estimator,
    density_for_false,
    density_for_true,
    *,
    marginal_false: float,
    marginal_true: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert binary density outputs into conditional treatment probabilities."""

    p0_raw = coerce_density_array(density_for_false).reshape(-1)
    p1_raw = coerce_density_array(density_for_true).reshape(-1)

    if is_stabilized_density(density_estimator):
        p0_raw = marginal_false * p0_raw
        p1_raw = marginal_true * p1_raw

    denom = np.clip(p0_raw + p1_raw, eps, None)
    return p0_raw / denom, p1_raw / denom
