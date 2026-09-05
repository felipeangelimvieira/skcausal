import numpy as np
import polars as pl
import pytest
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skcausal.datasets.base import BaseDataset
from skcausal.datasets.semi_synthetic_multidim import (
    MultidimSemiSyntheticDataset,
    _coupled_orders,
    _exchangeable_correlation,
)
from tests.datasets._semi_synthetic_test_utils import (
    SecondToyRealDataset,
    ToyRealDataset,
)


def _mean_regressor():
    return make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=object),
            ),
            remainder=StandardScaler(),
        ),
        LinearRegression(),
    )


def _stochastic_regressor():
    return make_pipeline(
        make_column_transformer(
            (
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=object),
            ),
            remainder=StandardScaler(),
        ),
        RandomForestRegressor(n_estimators=8),
    )


def _dataset(**overrides):
    params = {
        "real_datasets": [ToyRealDataset(), SecondToyRealDataset()],
        "regressors": _mean_regressor(),
        "outcome_noise_scale": 0.0,
        "random_state": 13,
    }
    params.update(overrides)
    return MultidimSemiSyntheticDataset(**params)


def test_exchangeable_correlation_validates_the_positive_definite_range():
    np.testing.assert_allclose(
        _exchangeable_correlation(3, -0.49),
        [[1.0, -0.49, -0.49], [-0.49, 1.0, -0.49], [-0.49, -0.49, 1.0]],
    )
    with pytest.raises(ValueError, match="-0.5 < treatment_correlation < 1"):
        _exchangeable_correlation(3, -0.5)
    with pytest.raises(ValueError, match="must be zero"):
        _exchangeable_correlation(1, 0.1)


@pytest.mark.parametrize("rho, direction", [(0.9, 1), (-0.9, -1)])
def test_coupled_orders_are_permutations_with_the_requested_rank_direction(
    rho, direction
):
    targets = [np.linspace(-2.0, 2.0, 2000), np.linspace(7.0, -1.0, 2000)]
    orders, latent = _coupled_orders(targets, rho, np.random.default_rng(9))
    coupled = np.column_stack(
        [target[order] for target, order in zip(targets, orders)]
    )

    for order in orders:
        np.testing.assert_array_equal(np.sort(order), np.arange(2000))
    assert latent.shape == (2000, 2)
    achieved = np.corrcoef(rankdata(coupled, axis=0), rowvar=False)[0, 1]
    assert direction * achieved > 0.8


def test_coupled_orders_break_target_ties_reproducibly():
    targets = [np.repeat([0.0, 1.0], 10), np.arange(20.0)]
    first, _ = _coupled_orders(targets, 0.2, np.random.default_rng(5))
    second, _ = _coupled_orders(targets, 0.2, np.random.default_rng(5))
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_source_outcomes_become_fixed_treatments_without_breaking_rows():
    sources = [ToyRealDataset(), SecondToyRealDataset()]
    dataset = _dataset(real_datasets=sources)
    X, treatment, outcome = dataset.load()

    assert treatment.columns == ["t_0", "t_1"]
    assert X.columns == [
        "d0_age",
        "d0_income",
        "d0_region",
        "d1_height",
        "d1_score",
        "d1_group",
    ]
    for source_index, source in enumerate(sources):
        source_X, _, source_y = source.load()
        rows = dataset.source_row_indices_[source_index]
        expected_y = source_y.to_numpy().reshape(-1)[rows]
        np.testing.assert_allclose(treatment[f"t_{source_index}"], expected_y)
        for original in source_X.columns:
            np.testing.assert_array_equal(
                X[f"d{source_index}_{original}"].to_numpy(),
                source_X[original].to_numpy()[rows],
            )
    np.testing.assert_allclose(dataset.predict_y(X, treatment), outcome)
    assert all(left.equals(right) for left, right in zip(dataset.load(), dataset.load()))


def test_copula_diagnostics_and_default_confounding_baseline_are_exposed():
    dataset = _dataset(treatment_correlation=0.7)
    X, treatment, _ = dataset.load()

    np.testing.assert_allclose(dataset.confounding_weights_, [0.5, 0.5])
    raw = dataset.source_scores_ @ np.sqrt(dataset.confounding_weights_)
    expected = (raw - raw.mean()) / raw.std(ddof=0)
    np.testing.assert_allclose(dataset.source_mu0_, expected)
    np.testing.assert_allclose(dataset.treatment_correlation_matrix_, [[1, 0.7], [0.7, 1]])
    np.testing.assert_allclose(
        dataset.achieved_pearson_correlation_,
        np.corrcoef(treatment.to_numpy(), rowvar=False),
    )
    assert dataset.achieved_spearman_correlation_.shape == (2, 2)
    np.testing.assert_allclose(
        dataset.predict_y(X, treatment).mean(), dataset.source_mu0_.mean(), atol=1e-12
    )


def test_dirichlet_weights_and_all_outputs_are_seeded_once_at_construction():
    first = _dataset(confounding_concentration=0.4, outcome_noise_scale=0.3)
    again = _dataset(confounding_concentration=0.4, outcome_noise_scale=0.3)
    other = _dataset(
        confounding_concentration=0.4, outcome_noise_scale=0.3, random_state=14
    )

    assert np.all(first.confounding_weights_ > 0.0)
    np.testing.assert_allclose(first.confounding_weights_.sum(), 1.0)
    np.testing.assert_allclose(first.confounding_weights_, again.confounding_weights_)
    assert all(left.equals(right) for left, right in zip(first.load(), again.load()))
    assert any(not left.equals(right) for left, right in zip(first.load(), other.load()))


def test_nested_stochastic_regressor_is_seeded_at_construction():
    first = _dataset(regressors=_stochastic_regressor(), outcome_noise_scale=0.3)
    again = _dataset(regressors=_stochastic_regressor(), outcome_noise_scale=0.3)

    np.testing.assert_allclose(first.source_scores_, again.source_scores_)
    assert all(left.equals(right) for left, right in zip(first.load(), again.load()))


def test_one_regressor_per_source_is_cloned_and_predict_is_available():
    configured = [_mean_regressor(), _mean_regressor()]
    dataset = _dataset(regressors=configured)
    X, _, _ = dataset.load()

    assert all(
        fitted is not source for fitted, source in zip(dataset.regressors_, configured)
    )
    assert dataset.predict(X, dataset.get_grid(9).head(3)).shape == (3,)


def test_additive_spline_response_is_centered_and_supports_counterfactuals():
    dataset = _dataset(treatment_effect_scale=2.0)
    X, treatment, _ = dataset.load()
    observed = dataset._evaluate_treatment_effect(treatment)
    shifted = treatment.with_columns((pl.col("t_0") + 0.25).alias("t_0"))

    assert observed.mean() == pytest.approx(0.0, abs=1e-12)
    assert not np.allclose(dataset.predict_y(X, treatment), dataset.predict_y(X, shifted))
    grid = dataset.get_grid(100)
    assert grid.columns == ["t_0", "t_1"]
    assert grid.shape == (100, 2)


class BadRegressor(BaseEstimator):
    def __init__(self, kind="nan"):
        self.kind = kind

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.kind == "nan":
            return np.full(len(X), np.nan)
        return np.zeros((len(X), 2))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"real_datasets": []}, "non-empty sequence"),
        ({"real_datasets": ToyRealDataset()}, "non-empty sequence"),
        ({"regressors": []}, "one estimator per source"),
        ({"regressors": [object(), object()]}, "cloneable sklearn estimator"),
        ({"treatment_correlation": 1.0}, "treatment_correlation"),
        ({"treatment_correlation": -1.0}, "treatment_correlation"),
        ({"confounding_concentration": 0.0}, "confounding_concentration"),
        ({"confounding_concentration": np.inf}, "confounding_concentration"),
        ({"n_spline_knots": 1}, "n_spline_knots"),
        ({"spline_degree": -1}, "spline_degree"),
        ({"treatment_effect_scale": -1.0}, "treatment_effect_scale"),
        ({"outcome_noise_scale": -1.0}, "outcome_noise_scale"),
    ],
)
def test_multidim_rejects_invalid_configuration(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _dataset(**overrides)


def test_multidim_rejects_constant_mean_predictions():
    with pytest.raises(ValueError, match="nonconstant finite predictions"):
        _dataset(regressors=DummyRegressor(strategy="mean"))


@pytest.mark.parametrize("kind", ["nan", "two_columns"])
def test_multidim_rejects_non_scalar_or_nonfinite_predictions(kind):
    with pytest.raises(ValueError, match="finite predictions"):
        _dataset(regressors=BadRegressor(kind))


def test_multidim_disables_resampling_and_treatment_density():
    dataset = _dataset()
    X, treatment, _ = dataset.load()
    with pytest.raises(NotImplementedError, match="fixed at construction"):
        dataset.sample(4)
    with pytest.raises(NotImplementedError, match="does not define"):
        dataset.log_prob(X, treatment)


class InvalidOutcomeSource(BaseDataset):
    def __init__(self, kind):
        self.kind = kind
        super().__init__()

    def _load(self):
        outcomes = {
            "wide": pl.DataFrame(
                {"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]}
            ),
            "string": pl.DataFrame({"y": ["a", "b", "c", "d"]}),
            "nan": pl.DataFrame({"y": [0.0, 1.0, np.nan, 3.0]}),
            "constant": pl.DataFrame({"y": [1.0, 1.0, 1.0, 1.0]}),
            "short": pl.DataFrame({"y": [0.0, 1.0, 2.0]}),
        }
        return (
            pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            pl.DataFrame({"unused_t": [0.0, 0.0, 0.0, 0.0]}),
            outcomes[self.kind],
        )


@pytest.mark.parametrize(
    "kind, message",
    [
        ("wide", "exactly one outcome"),
        ("string", "must be numeric"),
        ("nan", "must be finite"),
        ("constant", "at least two distinct"),
        ("short", "one outcome per covariate row"),
    ],
)
def test_multidim_rejects_invalid_source_outcomes(kind, message):
    with pytest.raises(ValueError, match=message):
        _dataset(real_datasets=[InvalidOutcomeSource(kind)])


def test_single_source_requires_zero_correlation_and_has_matrix_diagnostics():
    single = _dataset(real_datasets=[ToyRealDataset()])
    assert single.achieved_pearson_correlation_.shape == (1, 1)
    assert single.achieved_spearman_correlation_.shape == (1, 1)
    with pytest.raises(ValueError, match="must be zero"):
        _dataset(real_datasets=[ToyRealDataset()], treatment_correlation=0.1)
