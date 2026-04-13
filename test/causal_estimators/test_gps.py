import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from skcausal.causal_estimators.gps import GPS, GPSOut
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from skcausal.weight_estimators.dummy import DummyWeightEstimator


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
        self.y_numpy = y
        self.t_polars = pl.DataFrame({"t": t_values})

        grid_values = np.linspace(t_values.min(), t_values.max(), 5).astype(np.float32)
        self.t_grid_polars = pl.DataFrame({"t": grid_values})


class TrainSizeWeightEstimator(BaseBalancingWeightRegressor):
    def __init__(self, scale: float = 0.01, random_state: int = 17):
        self.scale = scale
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, t):
        self.train_size_ = len(X)
        return self

    def _predict_sample_weight(self, X, t):
        base = np.full((len(X), 1), float(self.train_size_))
        return base + np.asarray(X[:, :1], dtype=float) * self.scale


class RecordingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int = 23):
        self.random_state = random_state

    def fit(self, X, y):
        self.fit_X_ = np.asarray(X, dtype=float)
        self.fit_y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        self.last_predict_X_ = np.asarray(X, dtype=float)
        return np.zeros(self.last_predict_X_.shape[0], dtype=float)


GPS_ESTIMATOR_CONFIGS = [
    (
        GPS,
        {
            "treatment_regressor": None,
            "outcome_regressor": DecisionTreeRegressor(max_depth=3, random_state=0),
            "random_state": 0,
        },
    ),
    (
        GPSOut,
        {
            "density_regressor": DummyWeightEstimator(),
            "outcome_regressor": DecisionTreeRegressor(max_depth=3, random_state=0),
            "random_state": 0,
        },
    ),
]


@pytest.mark.parametrize("estimator_cls, init_params", GPS_ESTIMATOR_CONFIGS)
def test_gps_estimators_support_configured_mtypes(estimator_cls, init_params):
    scenario = ContinuousScenario()

    estimator = estimator_cls(**init_params)

    estimator.fit(scenario.X_polars, scenario.y_numpy, scenario.t_polars)

    adrf = estimator.predict_adrf(scenario.X_polars, scenario.t_grid_polars)
    adrf_array = np.atleast_1d(np.asarray(adrf, dtype=float))
    assert adrf_array.shape[0] == scenario.t_grid_polars.height
    assert np.all(np.isfinite(adrf_array))


def test_gps_out_uses_oof_gps_for_training_and_full_fit_for_prediction():
    n_samples = 12
    X = np.arange(n_samples * 2, dtype=np.float32).reshape(n_samples, 2)
    y = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    t = pl.DataFrame({"t": np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)})

    density_regressor = TrainSizeWeightEstimator(scale=0.01, random_state=7)
    outcome_regressor = RecordingRegressor(random_state=11)
    estimator = GPSOut(
        density_regressor=density_regressor,
        outcome_regressor=outcome_regressor,
        cv=3,
        random_state=0,
    )

    estimator.fit(X, y, t)

    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    expected_oof_indices = []
    expected_fit_X = []
    t_numpy = t.to_numpy().astype(np.float32)

    for train_idx, test_idx in splitter.split(X):
        expected_oof_indices.extend(test_idx.tolist())
        weights = train_idx.shape[0] + X[test_idx, :1] * density_regressor.scale
        expected_fit_X.append(
            np.concatenate(((weights + 1e-8) ** -1, t_numpy[test_idx]), axis=1)
        )

    expected_oof_indices = np.asarray(expected_oof_indices, dtype=int)
    expected_fit_X = np.concatenate(expected_fit_X, axis=0)

    np.testing.assert_array_equal(estimator.oof_test_indices_, expected_oof_indices)
    np.testing.assert_allclose(estimator.outcome_regressor_.fit_X_, expected_fit_X)
    np.testing.assert_allclose(
        estimator.outcome_regressor_.fit_y_, y[expected_oof_indices]
    )
    assert estimator.density_regressor_.train_size_ == n_samples
    assert estimator.density_regressor_.random_state == 7
    assert estimator.outcome_regressor_.random_state == 11
    assert estimator.treatment_regressor_ is estimator.density_regressor_

    t_grid = pl.DataFrame({"t": np.array([-0.5, 0.5], dtype=np.float32)})
    estimator.predict_adrf(X, t_grid)

    expected_predict_gps = (n_samples + X[:, :1] * density_regressor.scale + 1e-8) ** -1
    np.testing.assert_allclose(
        estimator.outcome_regressor_.last_predict_X_[:, :1], expected_predict_gps
    )
