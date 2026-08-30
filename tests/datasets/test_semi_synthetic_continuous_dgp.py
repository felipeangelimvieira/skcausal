import numpy as np
import pytest

from skcausal.datasets.semi_synthetic_continuous import (
    ContinuousSemiSyntheticDataset,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def _integrated_density(dataset, X_row, grid):
    repeated_X = X_row[np.zeros(grid.size, dtype=int)]
    treatment = grid.reshape(-1, 1)
    density = np.exp(dataset.log_prob(repeated_X, treatment)[:, 0])
    return np.trapezoid(density, grid)


def test_real_continuous_density_integrates_and_outcome_oracle_matches_noise_free_y():
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(),
        treatment_domain="real",
        outcome_noise_scale=0.0,
        random_state=5,
    )
    X, treatment, outcome = dataset.load()
    grid = np.linspace(-10.0, 10.0, 20001)

    np.testing.assert_allclose(
        _integrated_density(dataset, X.to_numpy()[:1], grid), 1.0, atol=2e-3
    )
    np.testing.assert_allclose(dataset.predict_y(X, treatment), outcome.to_numpy())
    assert dataset.get_grid(11).shape == (11, 1)


def test_positive_and_bounded_domains_have_exact_support_and_normalized_density():
    positive = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), treatment_domain="positive", random_state=6
    )
    bounded = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), treatment_domain=(-2.0, 3.0), random_state=7
    )
    X_positive, treatment_positive, _ = positive.load()
    X_bounded, treatment_bounded, _ = bounded.load()

    assert (treatment_positive.get_column("t") > 0).all()
    assert (treatment_bounded.get_column("t") > -2.0).all()
    assert (treatment_bounded.get_column("t") < 3.0).all()
    assert np.isneginf(positive.log_prob(X_positive.head(1), np.array([[0.0]]))[0, 0])
    assert np.isneginf(bounded.log_prob(X_bounded.head(1), np.array([[3.0]]))[0, 0])

    positive_grid = np.geomspace(1e-5, 100.0, 30000)
    bounded_grid = np.linspace(-2.0 + 1e-6, 3.0 - 1e-6, 30000)
    np.testing.assert_allclose(
        _integrated_density(positive, X_positive.to_numpy()[:1], positive_grid),
        1.0,
        atol=4e-3,
    )
    np.testing.assert_allclose(
        _integrated_density(bounded, X_bounded.to_numpy()[:1], bounded_grid),
        1.0,
        atol=4e-3,
    )


def test_bounded_log_density_matches_exact_latent_mixture_and_jacobian():
    lower, upper = -2.0, 3.0
    randomized_weight = 0.25
    noise_scale = 1.3
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(),
        treatment_domain=(lower, upper),
        randomized_weight=randomized_weight,
        treatment_noise_scale=noise_scale,
        random_state=17,
    )
    X, _, _ = dataset.load()
    treatment_values = np.full(X.height, 1.0)
    unit = (treatment_values - lower) / (upper - lower)
    latent = np.log(unit) - np.log1p(-unit)
    confounded_location = (
        dataset.confounding_strength * dataset.source_confounding_signal_
        + dataset.treatment_only_strength
        * (dataset.source_scores_.treatment_only @ dataset.treatment_only_coefficients_)
    )
    log_normalizer = np.log(noise_scale) + 0.5 * np.log(2.0 * np.pi)
    confounded_log_density = (
        -0.5 * ((latent - confounded_location) / noise_scale) ** 2 - log_normalizer
    )
    randomized_log_density = -0.5 * (latent / noise_scale) ** 2 - log_normalizer
    latent_log_density = np.logaddexp(
        np.log1p(-randomized_weight) + confounded_log_density,
        np.log(randomized_weight) + randomized_log_density,
    )
    inverse_log_jacobian = -np.log(upper - lower) - np.log(unit) - np.log1p(-unit)

    np.testing.assert_allclose(
        dataset.log_prob(X, treatment_values.reshape(-1, 1))[:, 0],
        latent_log_density + inverse_log_jacobian,
    )


def test_continuous_randomized_assignment_density_is_independent_of_x():
    dataset = ContinuousSemiSyntheticDataset(
        ToyRealDataset(), randomized_weight=1.0, random_state=10
    )
    X, _, _ = dataset.load()
    treatment = np.full((X.height, 1), 0.25)
    log_density = dataset.log_prob(X, treatment)

    np.testing.assert_allclose(
        log_density, np.repeat(log_density[:1], X.height, axis=0)
    )


def test_continuous_preserves_x_and_freezes_structure_across_resampling():
    source = ToyRealDataset(include_missing=True)
    dataset = ContinuousSemiSyntheticDataset(source, random_state=23)
    X, _, _ = dataset.load()
    coefficients_before = dataset.outcome_modifier_coefficients_.copy()
    fixed_treatment = np.full((X.height, 1), 0.5)
    mean_before = dataset.predict_y(X, fixed_treatment)

    first_X, first_treatment, _ = dataset.sample(random_state=31)
    second_X, second_treatment, _ = dataset.sample(random_state=32)

    assert X.equals(source.load()[0])
    assert first_X.equals(X)
    assert second_X.equals(X)
    assert not first_treatment.equals(second_treatment)
    np.testing.assert_array_equal(
        dataset.outcome_modifier_coefficients_, coefficients_before
    )
    np.testing.assert_allclose(dataset.predict_y(X, fixed_treatment), mean_before)


def test_continuous_oracles_accept_all_backends():
    dataset = ContinuousSemiSyntheticDataset(ToyRealDataset(), random_state=29)
    X, treatment, _ = dataset.load()

    polars_mean = dataset.predict_y(X, treatment)
    pandas_mean = dataset.predict_y(X.to_pandas(), treatment.to_pandas())
    numpy_mean = dataset.predict_y(X.to_numpy(), treatment.to_numpy())
    polars_density = dataset.log_prob(X, treatment)
    pandas_density = dataset.log_prob(X.to_pandas(), treatment.to_pandas())
    numpy_density = dataset.log_prob(X.to_numpy(), treatment.to_numpy())

    np.testing.assert_allclose(polars_mean, pandas_mean)
    np.testing.assert_allclose(polars_mean, numpy_mean)
    np.testing.assert_allclose(polars_density, pandas_density)
    np.testing.assert_allclose(polars_density, numpy_density)


def test_continuous_dgp_rejects_invalid_domains_and_noise():
    with pytest.raises(ValueError, match="treatment_domain"):
        ContinuousSemiSyntheticDataset(ToyRealDataset(), treatment_domain=(2.0, -1.0))
    with pytest.raises(ValueError, match="treatment_noise_scale"):
        ContinuousSemiSyntheticDataset(ToyRealDataset(), treatment_noise_scale=0.0)
