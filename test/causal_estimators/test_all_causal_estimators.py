import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.direct_method import DirectRegressor
from skcausal.causal_estimators.continuous.doubly_robust import (
    DoublyRobustPseudoOutcome,
)
from skcausal.causal_estimators.binary.doubly_robust import (
    BinaryDoublyRobust,
)
from skcausal.causal_estimators.continuous import (
    PropensityWeightingContinuous,
)
from skcausal.causal_estimators.binary import (
    BinaryPropensityWeighting,
)
from skcausal.density.base import BaseDensityEstimator


class DummyDensityEstimator(BaseDensityEstimator):
    _tags = {
        "X_inner_mtype": np.ndarray,
        "t_inner_mtype": pl.DataFrame,
        "density_kind": "stabilized",
    }

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, t):
        return self

    def _predict_density(self, X, t):
        out = np.ones((len(X), 1), dtype=float)
        rng = np.random.default_rng(self.random_state)
        noise = rng.normal(loc=0.0, scale=0.001, size=out.shape)
        return np.clip(out + noise, 1e-3, None)


class ContinuousScenario:
    def __init__(self, n_samples: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)

        X = rng.normal(size=(n_samples, 4)).astype(np.float32)
        beta = np.array([1.0, -0.5, 0.25, 0.75], dtype=np.float32)
        noise = rng.normal(scale=0.1, size=n_samples).astype(np.float32)
        y = X @ beta + noise

        t_values = (
            X @ np.array([0.3, -0.1, 0.2, 0.4], dtype=np.float32)
            + rng.normal(scale=0.5, size=n_samples)
        ).astype(np.float32)

        self.X_polars = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.X_numpy = X
        self.y_numpy = y

        self.t_polars = pl.DataFrame({"t": t_values})
        self.t_numpy = t_values.reshape(-1, 1)

        grid_values = np.linspace(t_values.min(), t_values.max(), 5).astype(np.float32)
        self.t_grid_polars = pl.DataFrame({"t": grid_values})
        self.t_grid_numpy = grid_values.reshape(-1, 1)


class BinaryScenario:
    def __init__(self, n_samples: int = 64, seed: int = 1):
        rng = np.random.default_rng(seed)

        X = rng.normal(size=(n_samples, 3)).astype(np.float32)
        beta = np.array([0.5, 0.2, -0.1], dtype=np.float32)
        noise = rng.normal(scale=0.1, size=n_samples).astype(np.float32)
        y = X @ beta + noise

        t_values = rng.random(n_samples) > 0.5

        self.X_polars = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.X_numpy = X
        self.y_numpy = y

        self.t_polars = pl.DataFrame({"t": t_values})
        self.t_numpy = t_values.reshape(-1, 1)

        self.t_grid_polars = pl.DataFrame({"t": [False, True]})
        self.t_grid_numpy = np.array([[False], [True]])


SCENARIOS = {
    "continuous": ContinuousScenario,
    "binary": BinaryScenario,
}


ESTIMATOR_CONFIGS = [
    (
        DirectRegressor,
        {
            "outcome_regressor": DecisionTreeRegressor(max_depth=3, random_state=0),
            "sample_weight_regressor": DummyDensityEstimator(),
        },
        "continuous",
    ),
    (
        BinaryPropensityWeighting,
        {"treatment_regressor": DummyDensityEstimator()},
        "binary",
    ),
    (
        PropensityWeightingContinuous,
        {"treatment_regressor": DummyDensityEstimator(), "random_state": 0},
        "continuous",
    ),
    (
        BinaryDoublyRobust,
        {
            "treatment_regressor": DummyDensityEstimator(),
            "outcome_regressor": LinearRegression(),
        },
        "binary",
    ),
    (
        DoublyRobustPseudoOutcome,
        {
            "treatment_regressor": DummyDensityEstimator(),
            "outcome_regressor": LinearRegression(),
            "pseudo_outcome_regressor": LinearRegression(),
        },
        "binary",
    ),
]


@pytest.mark.parametrize("estimator_cls, init_params, scenario_key", ESTIMATOR_CONFIGS)
def test_causal_estimators_support_configured_mtypes(
    estimator_cls, init_params, scenario_key
):
    scenario = SCENARIOS[scenario_key]()

    estimator: BaseAverageCausalResponseEstimator = estimator_cls(**init_params)

    estimator.fit(scenario.X_polars, scenario.y_numpy, scenario.t_polars)

    adrf = estimator.predict(scenario.X_polars, scenario.t_grid_polars)
    adrf_array = np.atleast_1d(np.asarray(adrf, dtype=float))
    assert adrf_array.shape[0] == scenario.t_grid_polars.height
    assert np.all(np.isfinite(adrf_array))
