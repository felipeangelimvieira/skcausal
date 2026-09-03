import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic_categorical import (
    CategoricalSemiSyntheticDataset,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def _all_level_probabilities(dataset, X):
    columns = []
    for level in dataset.treatment_levels_:
        treatment = pl.DataFrame({"t": [level] * X.height})
        columns.append(np.exp(dataset.log_prob(X, treatment)[:, 0]))
    return np.column_stack(columns)


def test_categorical_dgp_preserves_X_and_has_exact_calibrated_probabilities():
    target = np.array([0.2, 0.3, 0.5])
    source = ToyRealDataset(include_missing=True)
    dataset = CategoricalSemiSyntheticDataset(
        source,
        n_treatments=3,
        target_probabilities=target,
        randomized_weight=0.15,
        random_state=7,
    )
    X, treatment, outcome = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    assert X.equals(source.load()[0])
    assert treatment.schema["t"] == pl.Categorical
    assert (
        treatment.get_column("t").cast(pl.Utf8).is_in(dataset.treatment_levels_).all()
    )
    assert outcome.schema["y"] == pl.Float64
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(probabilities.mean(axis=0), target, atol=1e-9)


def test_categorical_randomized_component_is_independent_of_X():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(),
        n_treatments=3,
        target_probabilities=[0.25, 0.5, 0.25],
        randomized_weight=1.0,
        random_state=4,
    )
    X, _, _ = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    np.testing.assert_allclose(
        probabilities,
        np.tile([0.25, 0.5, 0.25], (X.height, 1)),
        atol=1e-12,
    )


def test_categorical_target_bias_calibrates_strength_without_changing_marginal():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(),
        n_treatments=3,
        target_probabilities=[0.3, 0.4, 0.3],
        confounding_strength=1.0,
        target_confounding_bias=0.35,
        treatment_only_strength=0.0,
        randomized_weight=0.1,
        random_state=21,
    )
    X, _, _ = dataset.load()
    probabilities = _all_level_probabilities(dataset, X)

    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.35, atol=2e-3)
    np.testing.assert_allclose(probabilities.mean(axis=0), [0.3, 0.4, 0.3], atol=1e-9)
    assert dataset.confounding_strength_ >= 0.0


def test_fully_randomized_categorical_assignment_has_zero_oracle_bias():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), randomized_weight=1.0, random_state=8
    )

    np.testing.assert_allclose(dataset.confounding_bias_ratio_, 0.0, atol=1e-12)


def test_categorical_target_bias_must_be_nonnegative():
    with pytest.raises(ValueError, match="target_confounding_bias"):
        CategoricalSemiSyntheticDataset(ToyRealDataset(), target_confounding_bias=-0.1)


def test_fully_randomized_categorical_assignment_rejects_positive_target_bias():
    with pytest.raises(ValueError, match="unattainable.*fully randomized"):
        CategoricalSemiSyntheticDataset(
            ToyRealDataset(),
            randomized_weight=1.0,
            target_confounding_bias=0.1,
        )


def test_categorical_outcome_oracle_accepts_all_backends():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), outcome_noise_scale=0.0, random_state=12
    )
    X, treatment, outcome = dataset.load()

    polars_mean = dataset.predict_y(X, treatment)
    pandas_mean = dataset.predict_y(X.to_pandas(), treatment.to_pandas())
    numpy_mean = dataset.predict_y(X.to_numpy(), treatment.to_numpy())

    np.testing.assert_allclose(polars_mean, outcome.to_numpy())
    np.testing.assert_allclose(polars_mean, pandas_mean)
    np.testing.assert_allclose(polars_mean, numpy_mean)
    assert dataset.get_levels().shape == (2, 1)


def test_categorical_dgp_validates_configuration_and_unknown_levels():
    with pytest.raises(ValueError, match="at least 2"):
        CategoricalSemiSyntheticDataset(ToyRealDataset(), n_treatments=1)
    with pytest.raises(ValueError, match="sum to one"):
        CategoricalSemiSyntheticDataset(
            ToyRealDataset(), n_treatments=2, target_probabilities=[0.8, 0.8]
        )

    dataset = CategoricalSemiSyntheticDataset(ToyRealDataset(), random_state=2)
    X, _, _ = dataset.load()
    unknown = pl.DataFrame({"t": ["unsupported"] * X.height})
    assert np.isneginf(dataset.log_prob(X, unknown)).all()


@pytest.mark.parametrize(
    "parameter",
    [
        "confounding_strength",
        "target_confounding_bias",
        "treatment_only_strength",
        "randomized_weight",
        "outcome_effect_scale",
        "effect_heterogeneity_scale",
        "outcome_noise_scale",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_categorical_dgp_rejects_nonfinite_scalar_configuration(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        CategoricalSemiSyntheticDataset(
            ToyRealDataset(),
            **{parameter: value},
        )


def test_categorical_inverse_cdf_closes_a_rounded_probability_undershoot(
    monkeypatch,
):
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), n_treatments=3, random_state=2
    )
    X, _, _ = dataset.load()
    undershooting_probabilities = np.tile(
        [0.1, 0.2, 0.6999999999999997],
        (X.height, 1),
    )

    monkeypatch.setattr(
        dataset,
        "_probabilities",
        lambda covariates: undershooting_probabilities,
    )

    class MaximumUniformGenerator:
        @staticmethod
        def random(size):
            return np.full(size, np.nextafter(1.0, 0.0))

    treatment = dataset._sample_treatment(X, MaximumUniformGenerator())

    assert treatment.get_column("t").to_list() == ["2"] * X.height


def test_categorical_accepts_numpy_integer_treatment_counts():
    dataset = CategoricalSemiSyntheticDataset(
        ToyRealDataset(), n_treatments=np.int64(3), random_state=2
    )

    assert dataset.treatment_levels_ == ["0", "1", "2"]
    assert dataset.get_levels().shape == (3, 1)


def test_categorical_bias_is_monotone_in_strength_with_small_floor():
    from skcausal.datasets.kang_schafer import KangSchaferContinuous

    real = KangSchaferContinuous(n=400, random_state=1)
    ratios = [
        CategoricalSemiSyntheticDataset(
            real,
            n_treatments=3,
            confounding_strength=strength,
            treatment_only_strength=1.0,
            random_state=5,
        ).confounding_bias_ratio_
        for strength in (0.0, 0.5, 1.0, 2.0, 4.0)
    ]

    assert ratios[0] < 0.1
    assert np.all(np.diff(ratios) > 0.0)


def test_categorical_default_configuration_calibrates_and_reaches_zero():
    from skcausal.datasets.kang_schafer import KangSchaferContinuous

    real = KangSchaferContinuous(n=200, random_state=1)
    for target in (0.0, 0.2, 0.5):
        dataset = CategoricalSemiSyntheticDataset(
            real, target_confounding_bias=target, random_state=3
        )
        np.testing.assert_allclose(dataset.confounding_bias_ratio_, target, atol=1e-6)
        X, _, _ = dataset.load()
        np.testing.assert_allclose(
            _all_level_probabilities(dataset, X).mean(axis=0), 0.5, atol=1e-9
        )
    with pytest.raises(ValueError, match="attainable"):
        CategoricalSemiSyntheticDataset(
            real, target_confounding_bias=50.0, random_state=3
        )


def test_categorical_rejects_nonfinite_sampled_outcomes():
    dataset = CategoricalSemiSyntheticDataset(ToyRealDataset(), random_state=3)
    dataset.outcome_noise_scale = np.finfo(float).max

    with pytest.raises(ValueError, match="Sampled outcomes must be finite"):
        dataset.sample(random_state=0)


def test_categorical_handles_boolean_covariates_with_missing_values():
    class BooleanDataset(ToyRealDataset):
        def _load(self):
            X, t, y = super()._load()
            return (
                X.with_columns(
                    pl.Series("flag", [True, False, None] * 2 + [True, False])
                ),
                t,
                y,
            )

    dataset = CategoricalSemiSyntheticDataset(BooleanDataset(), random_state=3)
    X, treatment, outcome = dataset.load()

    assert X.get_column("flag").null_count() == 2
    assert np.isfinite(outcome.to_numpy()).all()
    assert np.isfinite(dataset.log_prob(X, treatment)).all()
