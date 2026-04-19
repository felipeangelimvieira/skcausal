import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.performance_evaluation.evaluate import evaluate
from skcausal.density.performance_evaluation.metrics.likelihood import (
    LogLikelihoodMetric,
)


class ConstantDensityEstimator(BaseDensityEstimator):
    def __init__(self, density_value=0.5):
        self.density_value = density_value
        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        return np.full((len(X), 1), self.density_value, dtype=float)


def test_log_likelihood_metric_matches_expected_constant_density_value():
    X = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
    t = pl.DataFrame({"t": [0.1, 0.2, 0.3]})
    estimator = ConstantDensityEstimator(density_value=0.5).fit(X, t)

    metric = LogLikelihoodMetric()
    score = metric.evaluate(estimator, X, t)

    assert score == np.log(0.5)


def test_density_metric_exposes_default_name():
    metric = LogLikelihoodMetric()

    assert metric.name == "LogLikelihoodMetric"


def test_evaluate_returns_dataframe_with_expected_columns():
    X = pl.DataFrame({"x": [float(i) for i in range(20)]})
    t = pl.DataFrame({"t": [float(i) * 0.1 for i in range(20)]})
    estimator = ConstantDensityEstimator(density_value=0.25)

    cv = KFold(n_splits=3)
    results = evaluate(estimator=estimator, cv=cv, X=X, t=t)

    assert "test_LogLikelihoodMetric" in results.columns
    assert "fit_time" in results.columns
    assert "score_time" in results.columns
    assert "len_t_train" in results.columns
    assert len(results) == 3


def test_evaluate_return_data():
    X = pl.DataFrame({"x": [float(i) for i in range(20)]})
    t = pl.DataFrame({"t": [float(i) * 0.1 for i in range(20)]})
    estimator = ConstantDensityEstimator(density_value=0.25)

    cv = KFold(n_splits=3)
    results = evaluate(estimator=estimator, cv=cv, X=X, t=t, return_data=True)

    assert "t_train" in results.columns
    assert "t_test" in results.columns
