import numpy as np
import polars as pl
import pytest

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.benchmarking import (
    AverageResponseMetric,
    MAE,
    RMSE,
    evaluate_one,
)
from skcausal.causal_estimators.benchmarking.evaluate import (
    evaluate_multiple_dataset_seeds,
)
from skcausal.datasets.base import BaseSyntheticDataset


class _LinearSyntheticDataset(BaseSyntheticDataset):
    def __init__(self, treatment_values=None, random_state: int = 0, n: int = 8):
        self.treatment_values = treatment_values
        self.random_state = random_state
        self.n = n

        if treatment_values is None:
            rng = np.random.default_rng(random_state)
            treatment_values = rng.uniform(0.0, 1.0, size=(n, 1))

        treatment_values = np.asarray(treatment_values, dtype=float)
        if treatment_values.ndim == 1:
            treatment_values = treatment_values.reshape(-1, 1)

        self._treatment_values = treatment_values
        self.column_types = {
            column: "continuous"
            for column in [f"t{i}" for i in range(self._treatment_values.shape[1])]
        }

        super().__init__(n=self._treatment_values.shape[0], random_state=random_state)
        self.prepare()

    def _get_covariates(self):
        return np.arange(self.n, dtype=float).reshape(-1, 1)

    def _get_treatments(self, covariates):
        return self._treatment_values

    def _predict_y(self, covariates, treatments):
        treatments = np.asarray(treatments, dtype=float)
        if treatments.ndim == 1:
            treatments = treatments.reshape(-1, 1)

        return 1.0 + treatments.sum(axis=1, keepdims=True)

    def _inject_outcome_noise(self, expected_outcomes, covariates, treatments):
        return expected_outcomes


class _OffsetAverageResponseEstimator(BaseAverageCausalResponseEstimator):
    _tags = {
        "backend": "polars",
        "capability:t_type": ["continuous"],
        "capability:multidimensional_treatment": True,
    }

    def __init__(self, offset: float = 0.25):
        self.offset = offset
        super().__init__()

    def _fit(self, X, t, y):
        self.was_fit_ = True
        return self

    def _predict(self, t):
        values = t.to_numpy() if isinstance(t, pl.DataFrame) else np.asarray(t)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        return 1.0 + values.sum(axis=1) + self.offset


class _ObservedTreatmentMeanMetric(AverageResponseMetric):
    def _evaluate(self, dataset, estimator):
        _, treatments, _ = dataset.load()
        return float(np.asarray(treatments.to_numpy(), dtype=float).mean())


def _build_dataset_and_estimator():
    dataset = _LinearSyntheticDataset(np.linspace(0.0, 1.0, 8))
    estimator = _OffsetAverageResponseEstimator(offset=0.25)
    return dataset, estimator


def _build_seeded_dataset_and_estimator():
    dataset = _LinearSyntheticDataset(random_state=0, n=8)
    estimator = _OffsetAverageResponseEstimator(offset=0.25)
    return dataset, estimator


def test_evaluate_one_returns_one_row_with_metric_columns():
    dataset, estimator = _build_dataset_and_estimator()
    metrics = [
        MAE(n_treatments=4, random_state=0),
        RMSE(n_treatments=4, random_state=0),
    ]

    results = evaluate_one(dataset, estimator, metrics)

    assert results.columns.tolist() == [
        "MAE(n_treatments=4, random_state=0)",
        "RMSE(n_treatments=4, random_state=0)",
    ]
    assert len(results) == 1
    np.testing.assert_allclose(results.loc[0, metrics[0].__repr__()], 0.25)
    np.testing.assert_allclose(results.loc[0, metrics[1].__repr__()], 0.25)


def test_evaluate_one_returns_fitted_clone_when_requested():
    dataset, estimator = _build_dataset_and_estimator()

    results = evaluate_one(
        dataset,
        estimator,
        MAE(n_treatments=4, random_state=0),
        return_fitted=True,
    )

    fitted_model = results.loc[0, "fitted_model"]

    assert "fitted_model" in results.columns
    assert isinstance(fitted_model, _OffsetAverageResponseEstimator)
    assert fitted_model is not estimator
    assert fitted_model.was_fit_ is True
    assert not hasattr(estimator, "was_fit_")


def test_evaluate_one_rejects_invalid_metric_inputs():
    dataset, estimator = _build_dataset_and_estimator()

    with pytest.raises(ValueError, match="at least one"):
        evaluate_one(dataset, estimator, [])

    with pytest.raises(TypeError, match="AverageResponseMetric"):
        evaluate_one(dataset, estimator, [object()])


def test_evaluate_one_rejects_duplicate_metric_names():
    dataset, estimator = _build_dataset_and_estimator()
    metric = MAE(n_treatments=4, random_state=0)

    with pytest.raises(ValueError, match="distinct string representations"):
        evaluate_one(dataset, estimator, [metric, metric.clone()])


def test_evaluate_one_rejects_invalid_dataset_and_estimator_types():
    dataset, estimator = _build_dataset_and_estimator()

    with pytest.raises(TypeError, match="BaseSyntheticDataset"):
        evaluate_one(object(), estimator, MAE(n_treatments=4, random_state=0))

    with pytest.raises(TypeError, match="BaseAverageCausalResponseEstimator"):
        evaluate_one(dataset, object(), MAE(n_treatments=4, random_state=0))


def test_evaluate_multiple_dataset_seeds_returns_one_row_per_seed():
    dataset, estimator = _build_seeded_dataset_and_estimator()
    metrics = [MAE(n_treatments=4, random_state=0)]

    results = evaluate_multiple_dataset_seeds(
        dataset,
        estimator,
        metrics=metrics,
        random_states=[0, 1, 2],
    )

    assert results.columns.tolist() == [
        "MAE(n_treatments=4, random_state=0)",
        "dataset_seed",
    ]
    assert results["dataset_seed"].tolist() == [0, 1, 2]
    assert len(results) == 3


def test_evaluate_multiple_dataset_seeds_returns_fitted_models_when_requested():
    dataset, estimator = _build_seeded_dataset_and_estimator()

    results = evaluate_multiple_dataset_seeds(
        dataset,
        estimator,
        metrics=MAE(n_treatments=4, random_state=0),
        random_states=[3, 4],
        return_fitted=True,
    )

    assert "fitted_model" in results.columns
    assert results["dataset_seed"].tolist() == [3, 4]
    assert all(
        isinstance(model, _OffsetAverageResponseEstimator)
        for model in results["fitted_model"]
    )
    assert all(model is not estimator for model in results["fitted_model"])


def test_evaluate_multiple_dataset_seeds_rebuilds_dataset_for_each_seed():
    dataset, estimator = _build_seeded_dataset_and_estimator()

    results = evaluate_multiple_dataset_seeds(
        dataset,
        estimator,
        metrics=_ObservedTreatmentMeanMetric(),
        random_states=[0, 1, 2],
    )

    metric_column = str(_ObservedTreatmentMeanMetric())

    assert results["dataset_seed"].tolist() == [0, 1, 2]
    assert results[metric_column].nunique() > 1
