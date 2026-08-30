import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic import (
    _calibrate_confounding_strength,
    _calibrate_softmax_intercepts,
    _fit_baseline_surface,
    _fit_causal_score_map,
    _log_mixture,
    _normal_logpdf,
    _normalized_log_weights,
    _softmax,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def test_score_map_is_frozen_finite_and_handles_unknown_categories():
    X = ToyRealDataset(include_missing=True).load()[0]
    score_map, scores = _fit_causal_score_map(X, np.random.default_rng(4))
    repeated = score_map.transform(X)
    new_X = X.with_columns(pl.lit("west").alias("region"))
    unseen = score_map.transform(new_X)

    for name in ("confounders", "prognostic", "treatment_only", "modifiers"):
        np.testing.assert_allclose(getattr(scores, name), getattr(repeated, name))
        assert np.isfinite(getattr(unseen, name)).all()
    assert scores.confounders.shape == (X.height, 3)


def test_baseline_is_standardized_and_exposes_targeted_selection_signal():
    X = ToyRealDataset().load()[0]
    _, scores = _fit_causal_score_map(X, np.random.default_rng(8))
    surface = _fit_baseline_surface(scores, np.random.default_rng(9))
    mean, signal = surface.evaluate(scores)

    np.testing.assert_allclose(mean.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(mean.std(ddof=0), 1.0, atol=1e-12)
    assert mean.shape == signal.shape == (X.height,)


def test_score_map_rejects_changed_schema_and_all_missing_numeric_column():
    X = ToyRealDataset().load()[0]
    score_map, _ = _fit_causal_score_map(X, np.random.default_rng(2))

    with pytest.raises(ValueError, match="columns"):
        score_map.transform(X.drop("income"))
    with pytest.raises(ValueError, match="finite observed value"):
        _fit_causal_score_map(
            X.with_columns(pl.lit(None, dtype=pl.Float64).alias("income")),
            np.random.default_rng(2),
        )


def test_score_map_uses_a_nonlinear_term_in_each_output_score():
    X = ToyRealDataset().load()[0]
    score_map, _ = _fit_causal_score_map(X, np.random.default_rng(61984))
    n_encoded = score_map.encoder.transform(X.to_pandas()).shape[1]

    for projection in score_map.projections.values():
        assert (projection[n_encoded:] != 0).any(axis=0).all()


def test_score_map_supports_pandas_and_numpy_covariates():
    X = ToyRealDataset().load()[0]
    pandas_X = X.to_pandas()
    pandas_map, pandas_scores = _fit_causal_score_map(
        pandas_X, np.random.default_rng(12)
    )
    repeated = pandas_map.transform(pandas_X)

    np.testing.assert_allclose(pandas_scores.confounders, repeated.confounders)

    numpy_X = pandas_X[["age", "income"]].to_numpy()
    numpy_map, numpy_scores = _fit_causal_score_map(numpy_X, np.random.default_rng(13))
    np.testing.assert_allclose(
        numpy_scores.prognostic,
        numpy_map.transform(numpy_X).prognostic,
    )
    with pytest.raises(ValueError, match="width"):
        numpy_map.transform(numpy_X[:, :1])


def test_softmax_intercepts_match_requested_empirical_marginal():
    logits = np.array(
        [[-2.0, 0.0, 1.0], [-1.0, 0.5, 0.0], [2.0, -1.0, 0.5], [1.0, 1.5, -2.0]]
    )
    target = np.array([0.2, 0.3, 0.5])
    intercepts = _calibrate_softmax_intercepts(logits, target)
    probabilities = _softmax(logits + intercepts)

    np.testing.assert_allclose(probabilities.mean(axis=0), target, atol=1e-9)


def test_log_mixture_and_normalized_weights_are_stable():
    x = np.array([-20.0, 0.0, 20.0])
    left = _normal_logpdf(x, np.zeros(3), 1.0)
    right = _normal_logpdf(x, np.ones(3), 2.0)
    mixed = _log_mixture(left, right, weight=0.25)
    weights = _normalized_log_weights(mixed)

    np.testing.assert_allclose(
        np.exp(mixed), 0.75 * np.exp(left) + 0.25 * np.exp(right)
    )
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_strength_calibration_finds_crossing_and_rejects_unreachable_target():
    strength, achieved = _calibrate_confounding_strength(
        metric=lambda value: value**2,
        target=0.49,
        initial_strength=0.5,
    )
    np.testing.assert_allclose(strength, 0.7, atol=1e-6)
    np.testing.assert_allclose(achieved, 0.49, atol=1e-6)

    with pytest.raises(ValueError, match="attainable"):
        _calibrate_confounding_strength(
            metric=lambda value: min(value, 0.2),
            target=0.5,
            initial_strength=1.0,
        )


def test_strength_calibration_accepts_crossing_at_final_allowed_expansion():
    strength, achieved = _calibrate_confounding_strength(
        metric=lambda value: np.log2(value + 1.0),
        target=64.0,
        initial_strength=1.0,
    )

    np.testing.assert_allclose(strength, 2.0**64 - 1.0, atol=1e-6)
    np.testing.assert_allclose(achieved, 64.0, atol=1e-6)


def test_strength_calibration_bisects_a_large_valid_initial_bracket():
    strength, achieved = _calibrate_confounding_strength(
        metric=lambda value: value,
        target=0.5,
        initial_strength=1e60,
    )

    np.testing.assert_allclose(strength, 0.5, atol=1e-6)
    np.testing.assert_allclose(achieved, 0.5, atol=1e-6)
