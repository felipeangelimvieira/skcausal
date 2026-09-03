import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr

from skcausal.datasets.semi_synthetic import (
    _bisect_marginal_quantiles,
    _calibrate_confounding_strength,
    _calibrate_softmax_intercepts,
    _fit_baseline_surface,
    _fit_causal_score_map,
    _gaussian_mixture_marginal_cdf,
    _log_mixture,
    _normal_logpdf,
    _normalized_log_weights,
    _ridge_projections,
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


def test_normalized_log_weights_rejects_when_no_log_weight_is_finite():
    with pytest.raises(ValueError, match="at least one finite log weight"):
        _normalized_log_weights(np.array([-np.inf, -np.inf]))


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


def test_strength_calibration_returns_smallest_crossing_of_nonmonotone_metric():
    # Rises to 1 at strength 1, dips to 0 at strength 2, then rises again.
    def metric(value):
        return abs(np.sin(0.5 * np.pi * value)) * min(value, 1.0)

    strength, achieved = _calibrate_confounding_strength(
        metric=metric, target=0.5, initial_strength=1.0
    )

    np.testing.assert_allclose(achieved, 0.5, atol=1e-8)
    assert strength < 1.0
    np.testing.assert_allclose(metric(strength), 0.5, atol=1e-8)


def test_strength_calibration_caps_the_search_and_reports_attainable_range():
    with pytest.raises(ValueError, match="attainable.*strengths up to"):
        _calibrate_confounding_strength(
            metric=lambda value: np.log2(value + 1.0),
            target=64.0,
            initial_strength=1.0,
        )


def test_strength_calibration_rejects_targets_below_the_zero_strength_floor():
    with pytest.raises(ValueError, match="below the bias floor"):
        _calibrate_confounding_strength(
            metric=lambda value: 0.2 + value, target=0.1, initial_strength=1.0
        )

    strength, achieved = _calibrate_confounding_strength(
        metric=lambda value: 0.2 + value, target=0.2, initial_strength=1.0
    )
    assert strength == 0.0
    assert achieved == 0.2


def test_strength_calibration_bisects_a_large_valid_initial_bracket():
    strength, achieved = _calibrate_confounding_strength(
        metric=lambda value: value,
        target=0.5,
        initial_strength=1e60,
    )

    np.testing.assert_allclose(strength, 0.5, atol=1e-6)
    np.testing.assert_allclose(achieved, 0.5, atol=1e-6)


def test_treatment_only_scores_are_orthogonal_to_outcome_score_polynomials():
    rng = np.random.default_rng(3)
    X = pl.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.normal(size=400),
            "c": rng.choice(["u", "v", "w"], size=400),
        }
    )
    score_map, scores = _fit_causal_score_map(X, np.random.default_rng(4))
    outcome_scores = np.column_stack(
        (scores.confounders, scores.prognostic, scores.modifiers)
    )
    products = np.column_stack(
        [
            outcome_scores[:, i] * outcome_scores[:, j]
            for i in range(outcome_scores.shape[1])
            for j in range(i, outcome_scores.shape[1])
        ]
    )
    design = np.column_stack((np.ones(X.height), outcome_scores, products))

    assert score_map.residualization_order == 2
    np.testing.assert_allclose(design.T @ scores.treatment_only, 0.0, atol=1e-8)
    np.testing.assert_allclose(scores.treatment_only.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scores.treatment_only.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        score_map.transform(X).treatment_only, scores.treatment_only
    )


def test_score_map_encodes_boolean_columns_with_missing_values():
    X = pl.DataFrame(
        {
            "a": np.linspace(-1.0, 1.0, 12),
            "flag": [True, False, None] * 4,
        }
    )
    score_map, scores = _fit_causal_score_map(X, np.random.default_rng(5))

    assert np.isfinite(scores.confounders).all()
    assert score_map.categorical_columns == ("flag",)
    np.testing.assert_allclose(score_map.transform(X).confounders, scores.confounders)


def test_softmax_intercepts_converge_for_saturated_logits():
    rng = np.random.default_rng(0)
    logits = 40.0 * rng.normal(size=(300, 3))
    target = np.array([0.5, 0.3, 0.2])
    intercepts = _calibrate_softmax_intercepts(logits, target)

    np.testing.assert_allclose(
        _softmax(logits + intercepts).mean(axis=0), target, atol=1e-9
    )


def test_ridge_projections_recover_linear_targets_and_skip_missing_rows():
    rng = np.random.default_rng(3)
    library = rng.normal(size=(200, 6))
    library[:, 5] = 2.0  # zero-variance column must get a zero coefficient
    targets = np.column_stack(
        (
            3.0 * library[:, 0] - 1.5 * library[:, 2] + 0.5,
            library[:, 1] + rng.normal(scale=0.1, size=200),
        )
    )
    targets[:10, 1] = np.nan

    projection, r2 = _ridge_projections(library, targets)

    assert projection.shape == (6, 2)
    assert projection[5].tolist() == [0.0, 0.0]
    assert r2.shape == (2,)
    assert r2[0] > 0.999
    assert 0.9 < r2[1] <= 1.0
    fitted = library @ projection[:, 0]
    fitted = fitted - fitted.mean()
    np.testing.assert_allclose(fitted, targets[:, 0] - targets[:, 0].mean(), atol=0.15)


def test_ridge_projections_require_two_finite_targets():
    library = np.random.default_rng(0).normal(size=(5, 2))
    with pytest.raises(ValueError, match="at least two finite"):
        _ridge_projections(library, np.array([1.0, np.nan, np.nan, np.nan, np.nan]))


def test_score_map_accepts_fitted_confounders():
    X = ToyRealDataset(include_missing=True).load()[0]
    targets = np.column_stack(
        (
            np.linspace(-1.0, 1.0, X.height),
            X.get_column("income").to_numpy().astype(float),
        )
    )
    score_map, scores = _fit_causal_score_map(
        X, np.random.default_rng(4), fitted_confounders=targets
    )

    assert scores.confounders.shape == (X.height, 2)
    np.testing.assert_allclose(scores.confounders.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(scores.confounders.std(axis=0), 1.0, atol=1e-10)
    assert score_map.fitted_confounder_r2.shape == (2,)
    assert np.all(
        (score_map.fitted_confounder_r2 >= 0.0)
        & (score_map.fitted_confounder_r2 <= 1.0)
    )
    np.testing.assert_allclose(score_map.transform(X).confounders, scores.confounders)
    assert scores.treatment_only.shape == (X.height, 2)


def test_baseline_surface_accepts_fixed_confounder_coefficients():
    X = ToyRealDataset().load()[0]
    _, scores = _fit_causal_score_map(X, np.random.default_rng(8))
    surface = _fit_baseline_surface(
        scores, np.random.default_rng(9), confounder_coefficients=[1.0, 0.5, -0.25]
    )
    np.testing.assert_allclose(surface.confounder_coefficients, [1.0, 0.5, -0.25])
    mean, _ = surface.evaluate(scores)
    np.testing.assert_allclose(mean.std(ddof=0), 1.0, atol=1e-12)

    with pytest.raises(ValueError, match="confounder_coefficients"):
        _fit_baseline_surface(
            scores, np.random.default_rng(9), confounder_coefficients=[1.0, 0.5]
        )


def test_gaussian_mixture_marginal_cdf_and_quantiles_agree_with_closed_form():
    means = np.array([-1.0, 0.5, 2.0])
    scale, weight = 1.5, 0.25
    latent = np.array([-3.0, 0.0, 1.0, 4.0])

    expected = (1.0 - weight) * ndtr((latent[:, None] - means[None, :]) / scale).mean(
        axis=1
    ) + weight * ndtr(latent / scale)
    np.testing.assert_allclose(
        _gaussian_mixture_marginal_cdf(latent, means, scale, weight), expected
    )

    probabilities = np.array([0.1, 0.5, 0.9])
    quantiles = _bisect_marginal_quantiles(
        lambda value: _gaussian_mixture_marginal_cdf(value, means, scale, weight),
        probabilities,
        -20.0,
        20.0,
    )
    np.testing.assert_allclose(
        _gaussian_mixture_marginal_cdf(quantiles, means, scale, weight),
        probabilities,
        atol=1e-9,
    )
    assert np.all(np.diff(quantiles) > 0.0)
