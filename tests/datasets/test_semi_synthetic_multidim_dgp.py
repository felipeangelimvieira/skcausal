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
    _validate_source_list,
)
from tests.datasets._semi_synthetic_test_utils import (
    SecondToyRealDataset,
    ToyRealDataset,
)


def _toy_sources():
    return [ToyRealDataset(), SecondToyRealDataset()]


def _kang_schafer_sources(n, seeds=(1, 2)):
    return [KangSchaferContinuous(n=n, random_state=seed) for seed in seeds]


def test_per_component_broadcasts_scalars_and_checks_lengths():
    assert _per_component("x", 2.0, 3) == [2.0, 2.0, 2.0]
    assert _per_component("x", None, 2, default=1.0) == [1.0, 1.0]
    assert _per_component("x", "income", 2) == ["income", "income"]
    assert _per_component("x", [None, "a"], 2) == [None, "a"]
    with pytest.raises(ValueError, match="one entry per source"):
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

    single = ToyRealDataset()
    assert _validate_source_list(single) == [single]
    assert _validate_source_list((single,)) == [single]
    with pytest.raises(TypeError, match="non-empty sequence"):
        _validate_source_list([])
    with pytest.raises(TypeError, match="non-empty sequence"):
        _validate_source_list("dataset")
    with pytest.raises(TypeError, match="non-empty sequence"):
        _validate_source_list([single, object()])


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
        "treatment_correlation": 0.6,
        "randomized_weight": 0.3,
        "random_state": 5,
    }
    parameters.update(overrides)
    return MultidimSemiSyntheticDataset(_toy_sources(), **parameters)


def _standardized_treatment(dataset, latent):
    """Map standardized latent coordinates to released continuous values."""

    return dataset.released_offsets_ + dataset.released_scales_ * np.asarray(latent)


def test_multidim_releases_named_continuous_columns_and_density_integrates():
    dataset = _two_continuous()
    X, treatment, outcome = dataset.load()

    assert list(X.columns) == [
        "d0_age",
        "d0_income",
        "d0_region",
        "d1_height",
        "d1_score",
        "d1_group",
    ]
    assert list(treatment.columns) == ["t_0", "t_1"]
    assert dataset.n_treatments_ == 2
    assert dataset.column_types == {"t_0": "continuous", "t_1": "continuous"}
    assert outcome.shape == (8, 1)
    assert dataset.prognostic_fit_r2_.shape == (2,)
    assert np.all(
        (dataset.prognostic_fit_r2_ >= 0.0) & (dataset.prognostic_fit_r2_ <= 1.0)
    )

    axis = np.linspace(-12.0, 12.0, 241)
    lattice = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
    repeated_X = X.to_numpy()[np.zeros(lattice.shape[0], dtype=int)]
    released = _standardized_treatment(dataset, lattice)
    density = np.exp(dataset.log_prob(repeated_X, released)[:, 0]).reshape(241, 241)
    density = density * np.prod(dataset.released_scales_)
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
    released = _standardized_treatment(dataset, lattice)
    density = np.exp(dataset.log_prob(repeated_X, released)[:, 0]).reshape(481, 481)
    density = density * np.prod(dataset.released_scales_)
    integral = np.trapezoid(np.trapezoid(density, axis, axis=1), axis)
    np.testing.assert_allclose(integral, 1.0, atol=2e-3)

    reference = np.tile(dataset.released_offsets_, (X.height, 1))
    at_reference = dataset.predict_y(X, reference)[:, 0]
    np.testing.assert_allclose(at_reference, dataset.source_mu0_)

    # The outcome basis reads latent / sigma, so doubling sigma and the latent
    # treatment together must leave the dose-response untouched. Every
    # structural draw is seeded identically, so the two datasets differ only in
    # sigma.
    unit_scale = _two_continuous(
        treatment_noise_scale=1.0, treatment_correlation=0.4, randomized_weight=0.3
    )
    latent = np.tile([0.7, -1.3], (X.height, 1))
    np.testing.assert_allclose(
        dataset.predict_y(X, _standardized_treatment(dataset, 2.0 * latent)),
        unit_scale.predict_y(X, _standardized_treatment(unit_scale, latent)),
    )

    central = _standardized_treatment(dataset, ndtri(np.array([[0.01], [0.99]])) * 2.0)
    grid = dataset.get_grid(9)
    assert grid.shape == (9, 2)
    for j, name in enumerate(("t_0", "t_1")):
        column = grid.get_column(name)
        assert column.min() <= central[0, j] and column.max() >= central[1, j]


def test_multidim_log_prob_is_the_exact_gaussian_mixture():
    dataset = _two_continuous()
    X, treatment, _ = dataset.load()
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    values = (
        treatment.to_numpy().astype(float) - dataset.released_offsets_
    ) / dataset.released_scales_
    weight = dataset.randomized_weight
    expected = -np.log(dataset.released_scales_).sum() + np.log(
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

    reference = np.tile(dataset.released_offsets_, (X.height, 1))
    at_reference = dataset.predict_y(X, reference)[:, 0]
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
        _kang_schafer_sources(4000),
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
    sources = _kang_schafer_sources(300, seeds=(2, 3))
    biases = [
        MultidimSemiSyntheticDataset(
            sources, confounding_strength=strength, random_state=3
        ).confounding_bias_ratio_
        for strength in (0.25, 0.5, 1.0)
    ]
    assert biases[0] < biases[1] < biases[2]

    calibrated = MultidimSemiSyntheticDataset(
        sources, target_confounding_bias=0.5, random_state=3
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
    sources = _toy_sources()
    with pytest.raises(TypeError, match="non-empty sequence"):
        MultidimSemiSyntheticDataset([])
    with pytest.raises(TypeError, match="non-empty sequence"):
        MultidimSemiSyntheticDataset([ToyRealDataset(), 3])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        MultidimSemiSyntheticDataset(sources, treatment_correlation=1.0)
    with pytest.raises(ValueError, match="positive"):
        MultidimSemiSyntheticDataset(sources, treatment_noise_scale=0.0)
    with pytest.raises(ValueError, match="treatment_interaction_scale"):
        MultidimSemiSyntheticDataset(sources, treatment_interaction_scale=-1.0)
    with pytest.raises(ValueError, match="expected length 2"):
        MultidimSemiSyntheticDataset(sources, confounding_loadings=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected length 2"):
        MultidimSemiSyntheticDataset(sources, n_levels=[None, 3, 2])


def test_multidim_requires_a_single_numeric_source_outcome():
    class TwoOutcomeDataset(ToyRealDataset):
        def _load(self):
            X, t, y = super()._load()
            return X, t, y.with_columns(pl.lit(1.0).alias("extra"))

    with pytest.raises(ValueError, match="Source dataset 1 must provide exactly one"):
        MultidimSemiSyntheticDataset([ToyRealDataset(), TwoOutcomeDataset()])


def test_sources_are_aligned_prefixed_and_released_on_their_outcome_scale():
    sources = _toy_sources()
    dataset = MultidimSemiSyntheticDataset(sources, random_state=3)
    X, treatment, _ = dataset.load()
    X_first, _, y_first = sources[0].load()
    X_second, _, y_second = sources[1].load()

    assert dataset.n == 8
    assert dataset.source_columns_ == [
        ["d0_age", "d0_income", "d0_region"],
        ["d1_height", "d1_score", "d1_group"],
    ]
    first_rows, second_rows = dataset.source_row_indices_
    np.testing.assert_array_equal(first_rows, np.arange(8))
    assert second_rows.shape == (8,)
    assert np.all(np.diff(second_rows) > 0) and second_rows.max() < 10
    np.testing.assert_array_equal(
        X.select(["d1_height", "d1_score"]).to_numpy(),
        X_second.select(["height", "score"]).to_numpy()[second_rows],
    )
    np.testing.assert_array_equal(
        X.select(["d0_age", "d0_income"]).to_numpy(),
        X_first.select(["age", "income"]).to_numpy(),
    )
    assert list(X_first.columns) == ["age", "income", "region"]

    y0 = y_first.get_column("source_y").to_numpy().astype(float)
    y1 = y_second.get_column("other_y").to_numpy().astype(float)[second_rows]
    np.testing.assert_allclose(dataset.released_offsets_, [y0.mean(), y1.mean()])
    np.testing.assert_allclose(
        dataset.released_scales_, [y0.std(ddof=0), y1.std(ddof=0)]
    )
    np.testing.assert_allclose(dataset.baseline_surface_.confounder_coefficients, 1.0)

    # Oracles accept the concatenated schema, including NumPy of that width.
    assert dataset.log_prob(X.to_numpy(), treatment).shape == (8, 1)
    with pytest.raises(ValueError):
        dataset.log_prob(X_first, treatment)

    # Released value = mean + sd * latent, so log_prob at the outcome means
    # equals the latent density at zero minus the log of both scales.
    at_mean = np.tile(dataset.released_offsets_, (8, 1))
    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    latent_density = dataset._log_density(
        np.zeros((8, 2)), np.zeros((8, 0), dtype=int), means, dataset.cut_points_
    )
    np.testing.assert_allclose(
        dataset.log_prob(X, at_mean)[:, 0],
        latent_density - np.log(dataset.released_scales_).sum(),
    )


def test_subsampling_is_seeded_and_reproducible():
    same = [
        MultidimSemiSyntheticDataset(
            _toy_sources(), random_state=3
        ).source_row_indices_[1]
        for _ in range(2)
    ]
    np.testing.assert_array_equal(same[0], same[1])
    other = MultidimSemiSyntheticDataset(
        _toy_sources(), random_state=4
    ).source_row_indices_[1]
    assert not np.array_equal(same[0], other)


def test_prognostic_score_of_a_source_ignores_the_other_source():
    dataset = MultidimSemiSyntheticDataset(_toy_sources(), random_state=6)
    X, _, _ = dataset.load()
    permuted = np.random.default_rng(0).permutation(X.height)
    shuffled = X.with_columns(
        [
            X.get_column(name)[permuted].alias(name)
            for name in dataset.source_columns_[1]
        ]
    )
    original = dataset.score_map_.transform(X).confounders
    altered = dataset.score_map_.transform(shuffled).confounders
    np.testing.assert_allclose(altered[:, 0], original[:, 0])
    assert not np.allclose(altered[:, 1], original[:, 1])

    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    assert np.std(means[:, 0]) > 0.0 and np.std(means[:, 1]) > 0.0
    assert dataset.score_map_.fitted_confounder_r2.shape == (2,)


def test_single_dataset_is_one_source():
    dataset = MultidimSemiSyntheticDataset(ToyRealDataset(), n_levels=2, random_state=1)
    X, treatment, _ = dataset.load()
    assert list(X.columns) == ["d0_age", "d0_income", "d0_region"]
    assert list(treatment.columns) == ["t_0"]
    assert dataset.released_scales_[0] == 1.0
    assert treatment.get_column("t_0").cast(pl.Utf8).is_in(["0", "1"]).all()
    assert len(dataset.real_datasets_) == 1


def _mixed(**overrides):
    parameters = {
        "n_levels": [None, 3],
        "treatment_correlation": 0.6,
        "randomized_weight": 0.3,
        "random_state": 9,
    }
    parameters.update(overrides)
    return MultidimSemiSyntheticDataset(_toy_sources(), **parameters)


def test_mixed_masses_sum_to_the_continuous_marginal_and_match_brute_force():
    dataset = _mixed()
    X, treatment, _ = dataset.load()
    assert dataset.column_types == {"t_0": "continuous", "t_1": "categorical"}
    assert treatment.get_column("t_1").cast(pl.Utf8).is_in(["0", "1", "2"]).all()
    assert dataset.cut_points_[1].shape == (2,)

    row = X.head(1)
    means = dataset._latent_means(
        dataset.source_scores_, dataset.confounding_strength_
    )[0]
    weight = dataset.randomized_weight
    covariance = dataset.noise_covariance_
    value = 0.7
    released = dataset.released_offsets_[0] + dataset.released_scales_[0] * value
    masses = (
        np.array(
            [
                np.exp(
                    dataset.log_prob(
                        row, pl.DataFrame({"t_0": [released], "t_1": [level]})
                    )[0, 0]
                )
                for level in ("0", "1", "2")
            ]
        )
        * dataset.released_scales_[0]
    )
    marginal = (1.0 - weight) * multivariate_normal(
        mean=means[:1], cov=covariance[:1, :1]
    ).pdf(value) + weight * multivariate_normal(mean=[0.0], cov=covariance[:1, :1]).pdf(
        value
    )
    np.testing.assert_allclose(masses.sum(), marginal, rtol=1e-10)

    bounds = np.concatenate(([-12.0], dataset.cut_points_[1], [12.0]))
    for level in range(3):
        z1 = np.linspace(bounds[level], bounds[level + 1], 20001)
        points = np.column_stack((np.full(z1.size, value), z1))
        joint = (1.0 - weight) * multivariate_normal(mean=means, cov=covariance).pdf(
            points
        ) + weight * multivariate_normal(mean=np.zeros(2), cov=covariance).pdf(points)
        np.testing.assert_allclose(masses[level], np.trapezoid(joint, z1), rtol=1e-4)


def test_categorical_levels_are_balanced_and_grid_crosses_levels():
    dataset = MultidimSemiSyntheticDataset(
        _kang_schafer_sources(4000),
        n_levels=[None, 4],
        confounding_strength=0.0,
        randomized_weight=0.0,
        random_state=2,
    )
    _, treatment, _ = dataset.load()
    counts = treatment.get_column("t_1").cast(pl.Utf8).value_counts().sort("t_1")
    assert counts.get_column("t_1").to_list() == ["0", "1", "2", "3"]
    assert np.all(
        np.abs(counts.get_column("count").to_numpy().astype(np.int64) - 1000) < 120
    )

    grid = dataset.get_grid(10)
    assert grid.shape == (40, 2)
    assert grid.get_column("t_0").n_unique() == 10
    assert grid.get_column("t_1").cast(pl.Utf8).n_unique() == 4


def test_categorical_only_multidim_has_level_lattice_and_exact_calibration():
    dataset = MultidimSemiSyntheticDataset(
        _kang_schafer_sources(300, seeds=(2, 3)),
        n_levels=[2, 3],
        target_confounding_bias=0.3,
        random_state=4,
    )
    X, _, _ = dataset.load()
    grid = dataset.get_grid(100)
    assert grid.shape == (6, 2)

    repeated_X = X.head(1).to_numpy()[np.zeros(6, dtype=int)]
    masses = np.exp(dataset.log_prob(repeated_X, grid)[:, 0])
    np.testing.assert_allclose(masses.sum(), 1.0, rtol=1e-10)
    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.3, atol=2e-3)
    assert dataset.interaction_coefficients_[(0, 1)].shape == (1, 2)


def test_unknown_level_is_unsupported_and_predict_y_rejects_it():
    dataset = _mixed()
    X, _, _ = dataset.load()
    bad = pl.DataFrame({"t_0": [dataset.released_offsets_[0]], "t_1": ["7"]})
    assert np.isneginf(dataset.log_prob(X.head(1), bad)[0, 0])
    with pytest.raises(ValueError, match="known categorical levels"):
        dataset.predict_y(X.head(1), bad)


def test_malformed_continuous_treatment_raises_value_error():
    dataset = _mixed()
    X, _, _ = dataset.load()
    malformed = pl.DataFrame({"t_0": ["not-a-number"], "t_1": ["0"]})
    with pytest.raises(ValueError, match="must be numeric"):
        dataset.log_prob(X.head(1), malformed)


def test_correlated_noise_with_two_categorical_components_is_rejected():
    with pytest.raises(ValueError, match="two or more components are categorical"):
        MultidimSemiSyntheticDataset(
            _toy_sources(), n_levels=[2, 3], treatment_correlation=0.2
        )
    MultidimSemiSyntheticDataset(
        _toy_sources(), n_levels=[2, 3], treatment_correlation=0.0
    )


def test_predict_curve_accepts_mixed_grid_rows():
    dataset = MultidimSemiSyntheticDataset(
        _toy_sources(), n_levels=[None, 3], random_state=9
    )
    X, _, _ = dataset.load()
    grid = dataset.get_grid(4)
    assert grid.height == 12

    actual = np.asarray(dataset.predict(X, grid), dtype=float).ravel()

    assert actual.shape == (12,)
    assert np.isfinite(actual).all()
    expected = np.array(
        [
            dataset.predict_y(
                X,
                grid.slice(row_index, 1).select(pl.all().repeat_by(X.height).explode()),
            ).mean()
            for row_index in range(grid.height)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_multidim_rejects_non_numeric_source_outcome():
    class StringOutcomeDataset(ToyRealDataset):
        def _load(self):
            X, t, y = super()._load()
            return X, t, y.with_columns(pl.lit("a").alias("source_y"))

    with pytest.raises(ValueError, match="must be numeric"):
        MultidimSemiSyntheticDataset(StringOutcomeDataset())


def test_single_component_continuous_and_categorical_normalize():
    continuous = MultidimSemiSyntheticDataset(ToyRealDataset(), random_state=2)
    X, _, _ = continuous.load()
    grid = np.linspace(-12.0, 12.0, 4001)
    repeated_X = X.head(1).to_numpy()[np.zeros(grid.size, dtype=int)]
    released = _standardized_treatment(continuous, grid.reshape(-1, 1))
    density = np.exp(continuous.log_prob(repeated_X, released)[:, 0])
    integral = np.trapezoid(density * continuous.released_scales_[0], grid)
    np.testing.assert_allclose(integral, 1.0, atol=2e-3)

    categorical = MultidimSemiSyntheticDataset(
        ToyRealDataset(), n_levels=4, random_state=2
    )
    grid4 = categorical.get_grid(50)
    assert grid4.shape == (4, 1)
    repeated_row = X.head(1).to_numpy()[np.zeros(4, dtype=int)]
    masses = np.exp(categorical.log_prob(repeated_row, grid4)[:, 0])
    np.testing.assert_allclose(masses.sum(), 1.0, atol=1e-10, rtol=0.0)
    assert categorical.interaction_coefficients_ == {}


def test_treatment_only_strength_adds_a_bias_floor_and_still_calibrates():
    sources = _kang_schafer_sources(300, seeds=(2, 3))
    floor = MultidimSemiSyntheticDataset(
        sources,
        confounding_strength=0.0,
        treatment_only_strength=1.0,
        random_state=3,
    )
    assert floor.confounding_bias_ratio_ > 0.0
    assert floor.confounding_strength_ == 0.0

    target = floor.confounding_bias_ratio_ + 0.1
    calibrated = MultidimSemiSyntheticDataset(
        sources,
        treatment_only_strength=1.0,
        target_confounding_bias=target,
        random_state=3,
    )
    np.testing.assert_allclose(calibrated.confounding_bias_ratio_, target, atol=2e-3)
    assert calibrated.confounding_strength_ > 0.0


def test_confounded_categorical_sampler_matches_log_prob():
    dataset = MultidimSemiSyntheticDataset(
        _kang_schafer_sources(2000),
        n_levels=[None, 3],
        treatment_correlation=0.5,
        confounding_strength=1.0,
        randomized_weight=0.2,
        random_state=5,
    )
    _, treatment, _ = dataset.load()

    def level_frequencies(t):
        counts = t.get_column("t_1").cast(pl.Utf8).value_counts().sort("t_1")
        return dict(
            zip(
                counts.get_column("t_1").to_list(),
                (counts.get_column("count") / t.height).to_list(),
            )
        )

    replications = [
        treatment,
        dataset.sample(random_state=1)[1],
        dataset.sample(random_state=2)[1],
    ]
    for frame in replications:
        frequencies = level_frequencies(frame)
        for level in ("0", "1", "2"):
            assert abs(frequencies.get(level, 0.0) - 1.0 / 3.0) < 0.05

    means = dataset._latent_means(dataset.source_scores_, dataset.confounding_strength_)
    correlation = np.corrcoef(means[:, 0], means[:, 1])[0, 1]
    if correlation > 0.2:
        t0 = treatment.get_column("t_0").to_numpy().astype(float)
        t1 = treatment.get_column("t_1").cast(pl.Utf8).to_numpy()
        median = np.median(t0)
        frequency_above = np.mean(t1[t0 > median] == "2")
        frequency_below = np.mean(t1[t0 <= median] == "2")
        assert frequency_above > frequency_below
    # else: the latent means of t_0 and t_1 are not positively correlated
    # enough at this random_state to expect the directional relationship, so
    # the conditional-frequency check is skipped.
