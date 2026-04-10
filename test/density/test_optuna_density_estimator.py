import numpy as np
import pandas as pd
import polars as pl
import pytest
import time

optuna = pytest.importorskip("optuna")
pytest.importorskip("skpro")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from skpro.regression.residual import ResidualDouble

from skcausal.density.optuna import OptunaSearchDensityEstimator
from skcausal.density.performance_evaluation.metrics.likelihood import (
    LogLikelihoodMetric,
)
from skcausal.density.skpro import SkproDensityEstimator


class SlowLogLikelihoodMetric(LogLikelihoodMetric):
    def __init__(self, sleep_seconds=0.02, epsilon=1e-12):
        self.sleep_seconds = sleep_seconds
        super().__init__(epsilon=epsilon)

    def _evaluate(self, density_estimator, X, t):
        time.sleep(self.sleep_seconds)
        return super()._evaluate(density_estimator, X, t)


def _make_synthetic_dataset(n_samples=300, random_state=42):
    rng = np.random.default_rng(random_state)
    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    scale = 0.25 + 0.35 * np.abs(x1)
    noise = rng.normal(scale=scale, size=n_samples)
    t = 1.2 * x0 - 0.6 * x1 + noise
    X = pd.DataFrame({"x0": x0, "x1": x1})
    y = pd.DataFrame({"t": t})
    return pl.from_pandas(X), pl.from_pandas(y)


def _make_residual_double(n_estimators, random_state=0):
    return ResidualDouble(
        estimator=LinearRegression(),
        estimator_resid=RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        ),
    )


def test_optuna_density_estimator_fits_and_refits_best_model():
    X, t = _make_synthetic_dataset()
    tuner = OptunaSearchDensityEstimator(
        estimator=SkproDensityEstimator(estimator=_make_residual_double(5, 0)),
        metric=LogLikelihoodMetric(),
        param_distributions={
            "estimator": [
                _make_residual_double(5, 0),
                _make_residual_double(20, 0),
            ]
        },
        n_trials=2,
        cv=KFold(n_splits=3, shuffle=True, random_state=0),
        random_state=0,
        refit=True,
        error_score="raise",
    )

    tuner.fit(X, t)
    density = tuner.predict_density(X, t)

    assert density.shape == (len(X), 1)
    assert np.isfinite(density).all()
    assert (density >= 0).all()
    assert len(tuner.cv_results_) == 2
    assert np.isfinite(tuner.best_score_)
    assert tuner.best_score_ == pytest.approx(tuner.cv_results_["mean_score"].min())
    assert tuner.best_estimator_ is not None


def test_optuna_density_estimator_supports_callable_search_space():
    X, t = _make_synthetic_dataset()

    def _sample_estimator(trial):
        n_estimators = trial.suggest_int("rf_n_estimators", 5, 10)
        return _make_residual_double(n_estimators, 0)

    tuner = OptunaSearchDensityEstimator(
        estimator=SkproDensityEstimator(estimator=_make_residual_double(5, 0)),
        metric=LogLikelihoodMetric(),
        param_distributions={"estimator": _sample_estimator},
        n_trials=2,
        cv=KFold(n_splits=2, shuffle=True, random_state=0),
        random_state=0,
        refit=True,
        error_score="raise",
    )

    tuner.fit(X, t)

    assert len(tuner.cv_results_) == 2
    assert "rf_n_estimators" in tuner.study_.best_params


def test_optuna_density_estimator_respects_max_duration():
    X, t = _make_synthetic_dataset()
    tuner = OptunaSearchDensityEstimator(
        estimator=SkproDensityEstimator(estimator=_make_residual_double(5, 0)),
        metric=SlowLogLikelihoodMetric(sleep_seconds=0.02),
        param_distributions={
            "estimator": [
                _make_residual_double(5, 0),
                _make_residual_double(10, 0),
                _make_residual_double(15, 0),
            ]
        },
        n_trials=10,
        cv=KFold(n_splits=2, shuffle=True, random_state=0),
        random_state=0,
        refit=True,
        error_score="raise",
        max_duration=1e-4,
    )

    tuner.fit(X, t)

    assert 1 <= len(tuner.cv_results_) < 10
