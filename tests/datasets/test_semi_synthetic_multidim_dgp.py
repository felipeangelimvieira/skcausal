import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal

from skcausal.datasets.kang_schafer import KangSchaferContinuous
from skcausal.datasets.semi_synthetic_multidim import (
    MultidimSemiSyntheticDataset,
    _conditional_gaussian,
    _exchangeable_correlation,
    _gaussian_logpdf,
    _interval_log_mass,
    _numeric_column,
    _per_component,
    _validate_levels,
    _validate_loadings,
    _validate_targets,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def test_per_component_broadcasts_scalars_and_checks_lengths():
    assert _per_component("x", 2.0, 3) == [2.0, 2.0, 2.0]
    assert _per_component("x", None, 2, default=1.0) == [1.0, 1.0]
    assert _per_component("x", "income", 2) == ["income", "income"]
    assert _per_component("x", [None, "a"], 2) == [None, "a"]
    with pytest.raises(ValueError, match="length n_treatments=2"):
        _per_component("x", [1.0, 2.0, 3.0], 2)


def test_component_validators():
    np.testing.assert_allclose(_validate_loadings(None, 2), [1.0, 1.0])
    np.testing.assert_allclose(_validate_loadings([1.0, 0.0], 2), [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        _validate_loadings([1.0, np.inf], 2)

    assert _validate_levels(None, 2) == [None, None]
    assert _validate_levels([None, 3], 2) == [None, 3]
    with pytest.raises(ValueError, match="at least 2"):
        _validate_levels([1, None], 2)
    with pytest.raises(TypeError, match="integers"):
        _validate_levels([True, None], 2)

    assert _validate_targets(None, 2) == [None, None]
    assert _validate_targets([None, "income"], 2) == [None, "income"]
    with pytest.raises(ValueError, match="repeat"):
        _validate_targets(["age", "age"], 2)
    with pytest.raises(TypeError, match="column names"):
        _validate_targets([1, None], 2)


def test_exchangeable_correlation_is_symmetric_with_unit_diagonal():
    matrix = _exchangeable_correlation(3, 0.4)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix[np.triu_indices(3, 1)], 0.4)
    np.linalg.cholesky(matrix)


def test_interval_log_mass_matches_direct_formula_and_tails():
    lower = np.array([-np.inf, -1.0, 0.5, -np.inf, 30.0, -0.2])
    upper = np.array([-0.5, 0.25, np.inf, np.inf, np.inf, 0.2])
    with np.errstate(divide="ignore"):
        expected = np.log(ndtr(upper) - ndtr(lower))
    expected[4] = np.log(ndtr(-30.0))
    np.testing.assert_allclose(_interval_log_mass(lower, upper), expected, rtol=1e-12)
    # Deep tail below the double range of ndtr differences.
    assert np.isfinite(_interval_log_mass(np.array([-np.inf]), np.array([-40.0])))[0]


def test_interval_log_mass_returns_negative_infinity_for_zero_mass_bounds():
    lower = np.array([-np.inf, np.inf, 1.0, -np.inf, 0.5])
    upper = np.array([-np.inf, np.inf, -1.0, -np.inf, 0.5])
    result = _interval_log_mass(lower, upper)
    assert np.all(result == -np.inf)
    assert not np.any(np.isnan(result))


def test_gaussian_logpdf_matches_scipy_and_handles_empty_dimension():
    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    cholesky = np.linalg.cholesky(covariance)
    residual = np.random.default_rng(1).normal(size=(4, 3, 2))
    expected = (
        multivariate_normal(mean=np.zeros(2), cov=covariance)
        .logpdf(residual.reshape(-1, 2))
        .reshape(4, 3)
    )
    np.testing.assert_allclose(_gaussian_logpdf(residual, cholesky), expected)
    assert _gaussian_logpdf(np.zeros((5, 0)), np.zeros((0, 0))).shape == (5,)
    assert np.all(_gaussian_logpdf(np.zeros((5, 0)), np.zeros((0, 0))) == 0.0)


def test_conditional_gaussian_matches_textbook_formula():
    covariance = _exchangeable_correlation(3, 0.3) * 4.0
    continuous = np.array([0, 2])
    categorical = np.array([1])
    regression, conditional = _conditional_gaussian(covariance, continuous, categorical)

    cov_cc = covariance[np.ix_(continuous, continuous)]
    cov_dc = covariance[np.ix_(categorical, continuous)]
    expected_regression = cov_dc @ np.linalg.inv(cov_cc)
    np.testing.assert_allclose(regression, expected_regression)
    np.testing.assert_allclose(
        conditional,
        covariance[np.ix_(categorical, categorical)] - expected_regression @ cov_dc.T,
    )

    regression, conditional = _conditional_gaussian(
        covariance, np.array([0, 1, 2]), np.array([], dtype=int)
    )
    assert regression.shape == (0, 3) and conditional.shape == (0, 0)
    regression, conditional = _conditional_gaussian(
        covariance, np.array([], dtype=int), np.array([1])
    )
    assert regression.shape == (1, 0)
    np.testing.assert_allclose(conditional, [[4.0]])


def test_numeric_column_extracts_floats_and_rejects_bad_columns():
    X = ToyRealDataset(include_missing=True).load()[0]
    values = _numeric_column(X, "age", "target column 'age'")
    assert values.shape == (8,)
    assert np.isnan(values[2])
    with pytest.raises(ValueError, match="must be numeric"):
        _numeric_column(X, "region", "target column 'region'")
    with pytest.raises(ValueError, match="two distinct finite"):
        _numeric_column(
            X.with_columns(pl.lit(1.0).alias("age")), "age", "target column 'age'"
        )


def _two_continuous(**overrides):
    parameters = {
        "n_treatments": 2,
        "treatment_correlation": 0.6,
        "randomized_weight": 0.3,
        "random_state": 5,
    }
    parameters.update(overrides)
    return MultidimSemiSyntheticDataset(ToyRealDataset(), **parameters)


def test_multidim_releases_named_continuous_columns_and_density_integrates():
    dataset = _two_continuous()
    X, treatment, outcome = dataset.load()

    assert list(X.columns) == ["age", "income", "region"]
    assert list(treatment.columns) == ["t_0", "t_1"]
    assert dataset.column_types == {"t_0": "continuous", "t_1": "continuous"}
    assert outcome.shape == (8, 1)
    assert 0.0 <= dataset.prognostic_fit_r2_ <= 1.0

    axis = np.linspace(-12.0, 12.0, 241)
    lattice = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
    repeated_X = X.to_numpy()[np.zeros(lattice.shape[0], dtype=int)]
    density = np.exp(dataset.log_prob(repeated_X, lattice)[:, 0]).reshape(241, 241)
    integral = np.trapezoid(np.trapezoid(density, axis, axis=1), axis)
    np.testing.assert_allclose(integral, 1.0, atol=2e-3)


def test_multidim_honors_a_nonunit_treatment_noise_scale():
    dataset = _two_continuous(
        treatment_noise_scale=2.0, treatment_correlation=0.4, randomized_weight=0.3
    )
    X, _, _ = dataset.load()
    np.testing.assert_allclose(np.diag(dataset.noise_covariance_), 4.0)

    axis = np.linspace(-24.0, 24.0, 481)
    lattice = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
    repeated_X = X.to_numpy()[np.zeros(lattice.shape[0], dtype=int)]
    density = np.exp(dataset.log_prob(repeated_X, lattice)[:, 0]).reshape(481, 481)
    integral = np.trapezoid(np.trapezoid(density, axis, axis=1), axis)
    np.testing.assert_allclose(integral, 1.0, atol=2e-3)

    at_reference = dataset.predict_y(X, np.zeros((X.height, 2)))[:, 0]
    np.testing.assert_allclose(at_reference, dataset.source_mu0_)

    # The outcome basis reads latent / sigma, so doubling sigma and the treatment
    # together must leave the dose-response untouched. Every structural draw is
    # seeded identically, so the two datasets differ only in sigma.
    unit_scale = _two_continuous(
        treatment_noise_scale=1.0, treatment_correlation=0.4, randomized_weight=0.3
    )
    treatment = np.tile([0.7, -1.3], (X.height, 1))
    np.testing.assert_allclose(
        dataset.predict_y(X, 2.0 * treatment), unit_scale.predict_y(X, treatment)
    )

    central = ndtri(np.array([0.01, 0.99])) * 2.0
    grid = dataset.get_grid(9)
    assert grid.shape == (9, 2)
    for name in ("t_0", "t_1"):
        column = grid.get_column(name)
        assert column.min() <= central[0] and column.max() >= central[1]


def test_multidim_log_prob_is_the_exact_gaussian_mixture():
    dataset = _two_continuous()
    X, treatment, _ = dataset.load()
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    values = treatment.to_numpy().astype(float)
    weight = dataset.randomized_weight
    expected = np.log(
        (1.0 - weight)
        * np.array(
            [
                multivariate_normal(mean=means[i], cov=dataset.noise_covariance_).pdf(
                    values[i]
                )
                for i in range(X.height)
            ]
        )
        + weight
        * multivariate_normal(mean=np.zeros(2), cov=dataset.noise_covariance_).pdf(
            values
        )
    )
    np.testing.assert_allclose(
        dataset.log_prob(X, treatment)[:, 0], expected, rtol=1e-10
    )
    assert np.isneginf(dataset.log_prob(X.head(1), np.array([[np.inf, 0.0]]))[0, 0])


def test_multidim_predict_y_matches_noise_free_outcome_and_reference():
    dataset = _two_continuous(outcome_noise_scale=0.0)
    X, treatment, outcome = dataset.load()
    np.testing.assert_allclose(dataset.predict_y(X, treatment), outcome.to_numpy())

    at_reference = dataset.predict_y(X, np.zeros((X.height, 2)))[:, 0]
    np.testing.assert_allclose(at_reference, dataset.source_mu0_)
    assert len(dataset.interaction_coefficients_) == 1
    assert dataset.interaction_coefficients_[(0, 1)].shape == (1, 1)


def test_multidim_zero_loading_component_is_independent_of_x():
    dataset = _two_continuous(confounding_loadings=[1.0, 0.0])
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    np.testing.assert_allclose(means[:, 1], 0.0)
    assert np.std(means[:, 0]) > 0.0


def test_multidim_sampled_correlation_tracks_rho():
    dataset = MultidimSemiSyntheticDataset(
        KangSchaferContinuous(n=4000, random_state=1),
        n_treatments=2,
        treatment_correlation=0.6,
        confounding_strength=0.0,
        randomized_weight=0.0,
        random_state=2,
    )
    _, treatment, _ = dataset.load()
    correlation = np.corrcoef(treatment.to_numpy().astype(float).T)[0, 1]
    np.testing.assert_allclose(correlation, 0.6, atol=0.05)


def test_multidim_zero_bias_when_randomized_or_unconfounded():
    randomized = _two_continuous(randomized_weight=1.0)
    assert randomized.confounding_bias_ == 0.0
    unconfounded = _two_continuous(confounding_strength=0.0)
    assert unconfounded.confounding_bias_ == 0.0
    assert unconfounded.confounding_bias_ratio_ == 0.0


def test_multidim_bias_increases_with_strength_and_calibrates():
    # The oracle bias is an RMS of a signed pointwise distortion whose baseline
    # and effect-modifier channels anti-correlate at large strengths, so the
    # metric is only monotone over the operating range probed here.
    source = KangSchaferContinuous(n=300, random_state=2)
    biases = [
        MultidimSemiSyntheticDataset(
            source, n_treatments=2, confounding_strength=strength, random_state=3
        ).confounding_bias_ratio_
        for strength in (0.25, 0.5, 1.0)
    ]
    assert biases[0] < biases[1] < biases[2]

    calibrated = MultidimSemiSyntheticDataset(
        source, n_treatments=2, target_confounding_bias=0.5, random_state=3
    )
    np.testing.assert_allclose(calibrated.confounding_bias_ratio_, 0.5, atol=2e-3)
    assert calibrated.confounding_strength_ > 0.0


def test_multidim_grid_is_a_lattice_and_predict_curve_runs():
    dataset = _two_continuous()
    X, _, _ = dataset.load()
    grid = dataset.get_grid(25)
    assert grid.shape == (25, 2)
    assert grid.get_column("t_0").n_unique() == 5
    curve = np.asarray(dataset.predict(X, grid))
    assert curve.shape[0] == 25
    assert np.isfinite(curve).all()


def test_multidim_sample_is_reproducible_and_load_is_unchanged():
    dataset = _two_continuous()
    X, treatment, outcome = dataset.load()
    X_a, t_a, y_a = dataset.sample(random_state=11)
    _X_b, t_b, y_b = dataset.sample(random_state=11)
    assert t_a.equals(t_b) and y_a.equals(y_b) and X_a.equals(X)
    assert not t_a.equals(treatment)
    _X_again, treatment_again, outcome_again = dataset.load()
    assert treatment_again.equals(treatment) and outcome_again.equals(outcome)


def test_multidim_rejects_invalid_parameters():
    source = ToyRealDataset()
    with pytest.raises(ValueError, match="at least 1"):
        MultidimSemiSyntheticDataset(source, n_treatments=0)
    with pytest.raises(TypeError, match="integer"):
        MultidimSemiSyntheticDataset(source, n_treatments=2.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MultidimSemiSyntheticDataset(source, treatment_correlation=1.0)
    with pytest.raises(ValueError, match="positive"):
        MultidimSemiSyntheticDataset(source, treatment_noise_scale=0.0)
    with pytest.raises(ValueError, match="treatment_interaction_scale"):
        MultidimSemiSyntheticDataset(source, treatment_interaction_scale=-1.0)
    with pytest.raises(ValueError, match="length n_treatments=2"):
        MultidimSemiSyntheticDataset(source, confounding_loadings=[1.0, 2.0, 3.0])


def test_multidim_requires_a_single_numeric_source_outcome():
    class TwoOutcomeDataset(ToyRealDataset):
        def _load(self):
            X, t, y = super()._load()
            return X, t, y.with_columns(pl.lit(1.0).alias("extra"))

    with pytest.raises(ValueError, match="exactly one outcome column"):
        MultidimSemiSyntheticDataset(TwoOutcomeDataset(), n_treatments=1)
