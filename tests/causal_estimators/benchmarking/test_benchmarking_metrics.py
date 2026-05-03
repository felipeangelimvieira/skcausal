from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest
from skbase.testing.test_all_objects import BaseFixtureGenerator, QuickTester

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.benchmarking import AverageResponseMetric, MAE, RMSE
from skcausal.datasets.base import BaseSyntheticDataset

CURRENT_METRIC_TAGS = {"object_type", "metric_name"}


class _LinearSyntheticDataset(BaseSyntheticDataset):
    def __init__(self, treatment_values):
        treatment_values = np.asarray(treatment_values, dtype=float)
        if treatment_values.ndim == 1:
            treatment_values = treatment_values.reshape(-1, 1)

        self._treatment_values = treatment_values
        self.column_types = {
            column: "continuous"
            for column in [f"t{i}" for i in range(self._treatment_values.shape[1])]
        }

        super().__init__(n=self._treatment_values.shape[0], random_state=0)
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
        return self

    def _predict(self, t):
        self.predict_t_ = t.clone() if isinstance(t, pl.DataFrame) else np.array(t)

        values = t.to_numpy() if isinstance(t, pl.DataFrame) else np.asarray(t)
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        return 1.0 + values.sum(axis=1) + self.offset


@dataclass
class _MetricScenario:
    dataset: BaseSyntheticDataset
    estimator: BaseAverageCausalResponseEstimator


def _build_metric_scenario(treatment_values):
    dataset = _LinearSyntheticDataset(treatment_values)
    X, t, y = dataset.load()
    estimator = _OffsetAverageResponseEstimator().fit(X, t, y)
    return _MetricScenario(dataset=dataset, estimator=estimator)


class TestAllBenchmarkingMetrics(QuickTester, BaseFixtureGenerator):
    package_name = "skcausal.causal_estimators.benchmarking"
    valid_tags = sorted(CURRENT_METRIC_TAGS)
    object_type_filter = AverageResponseMetric
    fixture_sequence = ["object_class", "object_instance", "scenario"]

    def _generate_object_class(self, test_name, **kwargs):
        object_classes_to_test = []
        object_names = []

        for metric_class in self._all_objects():
            if metric_class is AverageResponseMetric:
                continue

            object_classes_to_test.append(metric_class)
            object_names.append(metric_class.__name__)

        return object_classes_to_test, object_names

    def _generate_object_instance(self, test_name, **kwargs):
        metric_instances = [
            MAE(n_treatments=4, random_state=0),
            RMSE(n_treatments=4, random_state=0),
        ]
        metric_names = [type(metric).__name__ for metric in metric_instances]
        return metric_instances, metric_names

    def _generate_scenario(self, test_name, **kwargs):
        scenarios = [
            _build_metric_scenario(np.linspace(0.0, 1.0, 8)),
            _build_metric_scenario(
                np.column_stack(
                    [
                        np.linspace(0.0, 1.0, 8),
                        np.linspace(-1.0, 1.0, 8),
                    ]
                )
            ),
        ]
        scenario_names = [
            "single_continuous_treatment",
            "two_continuous_treatments",
        ]
        return scenarios, scenario_names

    def test_evaluate_returns_finite_nonnegative_score(self, object_instance, scenario):
        value = float(object_instance.evaluate(scenario.dataset, scenario.estimator))

        assert isinstance(value, (int, float))
        assert np.isfinite(value)
        assert value >= 0.0


def test_rmse_computes_root_mean_squared_error():
    metric = RMSE(n_treatments=4)

    value = metric._compute_metric(
        np.array([1.0, 3.0, 5.0]),
        np.array([2.0, 1.0, 6.0]),
    )

    assert metric.get_tag("metric_name") == "RMSE"
    np.testing.assert_allclose(value, np.sqrt(2.0))
