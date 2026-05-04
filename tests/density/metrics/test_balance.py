"""Tests for weighted-correlation balance metrics on known toy constructions."""

import numpy as np
import polars as pl
import pytest

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.performance_evaluation.metrics.balance import (
    AverageAbsoluteWeightedCorrelationMetric,
)


class ConstantDensityEstimator(BaseDensityEstimator):
    """Simple baseline estimator that assigns the same density to every row."""

    def __init__(self, density_value=1.0):
        self.density_value = density_value
        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        return np.full((len(X), 1), self.density_value, dtype=float)


class GroundTruthBinaryDensityEstimator(BaseDensityEstimator):
    """Exact binary-treatment density used to verify the balance metric."""

    def __init__(self, density_kind="conditional"):
        self.density_kind = density_kind
        super().__init__()
        self.set_tags(density_kind=self.density_kind)

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        x1 = X["x1"].to_numpy()
        x2 = X["x2"].to_numpy()
        treatment = t["t"].to_numpy()

        propensity_treated = 0.25 + 0.25 * x1 + 0.25 * x2
        conditional_density = np.where(
            treatment == 1.0,
            propensity_treated,
            1.0 - propensity_treated,
        )

        if self.density_kind == "conditional":
            density = conditional_density
        elif self.density_kind == "stabilized":
            density = conditional_density / 0.5
        else:
            raise ValueError(f"Unexpected density_kind: {self.density_kind!r}")

        return density.reshape(-1, 1)


class GroundTruthContinuousDensityEstimator(BaseDensityEstimator):
    """Exact discrete-support continuous-treatment density for balance checks."""

    _treatment_support = (0.5, 1.0, 2.0, 4.0)
    _cell_counts = {
        (0.0, 0.0): (4, 2, 1, 1),
        (0.0, 1.0): (1, 4, 2, 1),
        (1.0, 0.0): (1, 1, 4, 2),
        (1.0, 1.0): (2, 1, 1, 4),
    }

    def __init__(self, density_kind="conditional"):
        self.density_kind = density_kind
        super().__init__()
        self.set_tags(density_kind=self.density_kind)

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        densities = []

        for x1_value, x2_value, treatment_value in zip(
            X["x1"].to_numpy(),
            X["x2"].to_numpy(),
            t["t"].to_numpy(),
        ):
            counts = self._cell_counts[(float(x1_value), float(x2_value))]
            count_lookup = dict(zip(self._treatment_support, counts))
            conditional_density = count_lookup[float(treatment_value)] / sum(counts)

            if self.density_kind == "conditional":
                density = conditional_density
            elif self.density_kind == "stabilized":
                density = conditional_density / 0.25
            else:
                raise ValueError(f"Unexpected density_kind: {self.density_kind!r}")

            densities.append(density)

        return np.asarray(densities, dtype=float).reshape(-1, 1)


def _make_known_binary_balance_dataset():
    """Build a confounded binary-treatment sample with analytically known weights."""

    x1 = []
    x2 = []
    treatment = []

    for x1_value in [0.0, 1.0]:
        for x2_value in [0.0, 1.0]:
            propensity_treated = 0.25 + 0.25 * x1_value + 0.25 * x2_value
            n_cell = 4
            n_treated = int(propensity_treated * n_cell)
            n_control = n_cell - n_treated

            x1.extend([x1_value] * n_cell)
            x2.extend([x2_value] * n_cell)
            treatment.extend([1.0] * n_treated)
            treatment.extend([0.0] * n_control)

    X = pl.DataFrame({"x1": x1, "x2": x2})
    t = pl.DataFrame({"t": treatment})
    return X, t


def _make_known_continuous_balance_dataset():
    """Build a confounded numeric-treatment sample with known balancing weights."""

    x1 = []
    x2 = []
    treatment = []

    treatment_support = GroundTruthContinuousDensityEstimator._treatment_support
    cell_counts = GroundTruthContinuousDensityEstimator._cell_counts

    for (x1_value, x2_value), counts in cell_counts.items():
        for treatment_value, count in zip(treatment_support, counts):
            x1.extend([x1_value] * count)
            x2.extend([x2_value] * count)
            treatment.extend([treatment_value] * count)

    X = pl.DataFrame({"x1": x1, "x2": x2})
    t = pl.DataFrame({"t": treatment})
    return X, t


def test_balance_metric_exposes_default_name():
    """The metric should default to its class name in evaluation outputs."""

    metric = AverageAbsoluteWeightedCorrelationMetric()

    assert metric.name == "AverageAbsoluteWeightedCorrelationMetric"


@pytest.mark.parametrize("density_kind", ["conditional", "stabilized"])
def test_balance_metric_goes_to_zero_for_ground_truth_binary_treatment(density_kind):
    """Exact binary-treatment weights should remove all linear X-T association."""

    X, t = _make_known_binary_balance_dataset()
    estimator = GroundTruthBinaryDensityEstimator(density_kind=density_kind).fit(X, t)
    metric = AverageAbsoluteWeightedCorrelationMetric()

    score = metric.evaluate(estimator, X, t)

    assert score == pytest.approx(0.0, abs=1e-12)


def test_balance_metric_detects_residual_dependence_for_binary_treatment():
    """A naive unweighted baseline should balance worse than the exact binary model."""

    X, t = _make_known_binary_balance_dataset()
    naive_estimator = ConstantDensityEstimator(density_value=1.0).fit(X, t)
    ground_truth = GroundTruthBinaryDensityEstimator(density_kind="conditional").fit(
        X, t
    )
    metric = AverageAbsoluteWeightedCorrelationMetric()

    naive_score = metric.evaluate(naive_estimator, X, t)
    balanced_score = metric.evaluate(ground_truth, X, t)

    assert naive_score > 0.0
    assert balanced_score < naive_score


@pytest.mark.parametrize("density_kind", ["conditional", "stabilized"])
def test_balance_metric_goes_to_zero_for_ground_truth_continuous_treatment(
    density_kind,
):
    """Exact numeric-treatment weights should drive the average weighted correlation to zero."""

    X, t = _make_known_continuous_balance_dataset()
    estimator = GroundTruthContinuousDensityEstimator(density_kind=density_kind).fit(
        X, t
    )
    metric = AverageAbsoluteWeightedCorrelationMetric()

    score = metric.evaluate(estimator, X, t)

    assert score == pytest.approx(0.0, abs=1e-12)


def test_balance_metric_detects_residual_dependence_for_continuous_treatment():
    """A constant-density baseline should fail to balance the numeric treatment."""

    X, t = _make_known_continuous_balance_dataset()
    naive_estimator = ConstantDensityEstimator(density_value=1.0).fit(X, t)
    ground_truth = GroundTruthContinuousDensityEstimator(
        density_kind="conditional"
    ).fit(X, t)
    metric = AverageAbsoluteWeightedCorrelationMetric()

    naive_score = metric.evaluate(naive_estimator, X, t)
    balanced_score = metric.evaluate(ground_truth, X, t)

    assert naive_score > 0.0
    assert balanced_score < naive_score
