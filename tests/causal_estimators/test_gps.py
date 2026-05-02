import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from skcausal.causal_estimators.gps import GPS
from skcausal.density.base import BaseDensityEstimator


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
        self.y_polars = pl.DataFrame({"y": y})
        self.t_polars = pl.DataFrame({"t": t_values})

        grid_values = np.linspace(t_values.min(), t_values.max(), 5).astype(np.float32)
        self.t_grid_polars = pl.DataFrame({"t": grid_values})
        self.X_grid_polars = self.X_polars.head(len(grid_values))


class DummyDensityEstimator(BaseDensityEstimator):
    _tags = {
        "backend": "polars",
        "density_kind": "conditional",
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


class TrainSizeDensityEstimator(BaseDensityEstimator):
    _tags = {
        "backend": "polars",
        "density_kind": "conditional",
    }

    def __init__(self, scale: float = 0.01, random_state: int = 17):
        self.scale = scale
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, t):
        self.train_size_ = len(X)
        return self

    def _predict_density(self, X, t):
        X_array = X.to_numpy() if isinstance(X, pl.DataFrame) else np.asarray(X)
        base = np.full((len(X), 1), float(self.train_size_))
        return base + np.asarray(X_array[:, :1], dtype=float) * self.scale


class RecordingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state: int = 23):
        self.random_state = random_state

    def fit(self, X, y):
        self.fit_X_ = np.asarray(X, dtype=float)
        self.fit_y_ = np.asarray(y, dtype=float)
        self.predict_shapes_ = []
        return self

    def predict(self, X):
        self.last_predict_X_ = np.asarray(X, dtype=float)
        self.predict_shapes_.append(self.last_predict_X_.shape)
        return np.zeros(self.last_predict_X_.shape[0], dtype=float)


GPS_ESTIMATOR_CONFIGS = [
    (
        GPS,
        {
            "density_regressor": DummyDensityEstimator(random_state=0),
            "outcome_regressor": DecisionTreeRegressor(max_depth=3, random_state=0),
            "random_state": 0,
        },
    ),
    (
        GPS,
        {
            "density_regressor": DummyDensityEstimator(),
            "outcome_regressor": DecisionTreeRegressor(max_depth=3, random_state=0),
            "random_state": 0,
        },
    ),
]


@pytest.mark.parametrize("estimator_cls, init_params", GPS_ESTIMATOR_CONFIGS)
def test_gps_estimators_support_configured_mtypes(estimator_cls, init_params):
    scenario = ContinuousScenario()

    estimator = estimator_cls(**init_params)

    estimator.fit(scenario.X_polars, scenario.t_polars, scenario.y_polars)

    adrf = estimator.predict(scenario.t_grid_polars)
    adrf_array = np.atleast_1d(np.asarray(adrf, dtype=float))
    assert adrf_array.shape[0] == scenario.t_grid_polars.height
    assert np.all(np.isfinite(adrf_array))


def test_gps_predict_supports_categorical_polars_treatments_with_user_pipeline():
    rng = np.random.default_rng(7)
    n_samples = 24

    X = pl.DataFrame(
        rng.normal(size=(n_samples, 2)).astype(np.float32),
        schema=["x0", "x1"],
    )
    treatment_labels = np.where(
        X["x0"].to_numpy() > 0.5,
        "treated",
        np.where(X["x0"].to_numpy() < -0.5, "placebo", "control"),
    )
    t = pl.DataFrame({"treatment": treatment_labels}).with_columns(
        pl.col("treatment").cast(pl.Categorical)
    )
    y = pl.DataFrame(
        {
            "y": (
                0.8 * X["x0"].to_numpy()
                + np.where(treatment_labels == "treated", 1.0, 0.0)
                - np.where(treatment_labels == "placebo", 0.5, 0.0)
            ).astype(np.float32)
        }
    )
    t_grid = pl.DataFrame(
        {"treatment": ["control", "placebo", "treated"]}
    ).with_columns(pl.col("treatment").cast(pl.Categorical))

    outcome_regressor = make_pipeline(
        ColumnTransformer(
            transformers=[
                (
                    "encode_categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    make_column_selector(
                        dtype_include=["category", "object", "string"]
                    ),
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ),
        DecisionTreeRegressor(max_depth=3, random_state=0),
    )

    estimator = GPS(
        density_regressor=DummyDensityEstimator(random_state=3),
        outcome_regressor=outcome_regressor,
        cv=2,
        random_state=0,
    )

    estimator.fit(X, t, y)

    assert list(estimator.oof_treatment_gps_.columns) == ["gps", "treatment"]

    adrf = estimator.predict(t_grid)

    adrf_array = np.asarray(adrf, dtype=float).reshape(-1)
    assert adrf_array.shape == (t_grid.height,)
    assert np.all(np.isfinite(adrf_array))


def test_gps_out_uses_oof_gps_for_training_and_full_fit_for_prediction():
    n_samples = 12
    X = np.arange(n_samples * 2, dtype=np.float32).reshape(n_samples, 2)
    y = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    t = pl.DataFrame({"t": np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)})

    density_regressor = TrainSizeDensityEstimator(scale=0.01, random_state=7)
    outcome_regressor = RecordingRegressor(random_state=11)
    estimator = GPS(
        density_regressor=density_regressor,
        outcome_regressor=outcome_regressor,
        cv=3,
        random_state=0,
    )

    X_frame = pl.DataFrame(X, schema=["x0", "x1"])
    y_frame = pl.DataFrame({"y": y})

    estimator.fit(X_frame, t, y_frame)

    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    expected_oof_indices = []
    expected_fit_X = []
    t_numpy = t.to_numpy().astype(np.float32)

    for train_idx, test_idx in splitter.split(X):
        expected_oof_indices.extend(test_idx.tolist())
        densities = train_idx.shape[0] + X[test_idx, :1] * density_regressor.scale
        expected_fit_X.append(np.concatenate((densities, t_numpy[test_idx]), axis=1))

    expected_oof_indices = np.asarray(expected_oof_indices, dtype=int)
    expected_fit_X = np.concatenate(expected_fit_X, axis=0)

    np.testing.assert_array_equal(estimator.oof_test_indices_, expected_oof_indices)
    np.testing.assert_allclose(estimator.outcome_regressor_.fit_X_, expected_fit_X)
    np.testing.assert_allclose(
        estimator.outcome_regressor_.fit_y_, y[expected_oof_indices].reshape(-1, 1)
    )
    assert estimator.density_regressor_.train_size_ == n_samples
    assert estimator.density_regressor_.random_state == 7
    assert estimator.outcome_regressor_.random_state == 11
    assert estimator.treatment_regressor_ is estimator.density_regressor_

    t_grid = pl.DataFrame({"t": np.array([-0.5, 0.5], dtype=np.float32)})
    estimator.predict(t_grid)

    expected_predict_gps = n_samples + X[:, :1] * density_regressor.scale
    np.testing.assert_allclose(
        estimator.outcome_regressor_.last_predict_X_[:, :1],
        expected_predict_gps,
    )


def test_gps_can_subsample_fit_time_x_at_curve_prediction_time():
    n_samples = 12
    X = np.arange(n_samples * 2, dtype=np.float32).reshape(n_samples, 2)
    y = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    t = pl.DataFrame({"t": np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)})

    density_regressor = TrainSizeDensityEstimator(scale=0.01, random_state=7)
    outcome_regressor = RecordingRegressor(random_state=11)
    estimator = GPS(
        density_regressor=density_regressor,
        outcome_regressor=outcome_regressor,
        cv=0,
        max_samples_predict=3,
        random_state=0,
    )

    X_frame = pl.DataFrame(X, schema=["x0", "x1"])
    y_frame = pl.DataFrame({"y": y})
    t_grid = pl.DataFrame({"t": np.array([-0.5, 0.5], dtype=np.float32)})

    estimator.fit(X_frame, t, y_frame)
    estimator.predict(t_grid)

    selected = np.sort(
        np.random.default_rng(0).choice(n_samples, size=3, replace=False)
    )
    expected_predict_gps = n_samples + X[selected, :1] * density_regressor.scale

    assert estimator.outcome_regressor_.predict_shapes_ == [(3, 2), (3, 2)]
    np.testing.assert_allclose(
        estimator.outcome_regressor_.last_predict_X_[:, :1],
        expected_predict_gps,
    )


def test_gps_rejects_invalid_max_samples_predict():
    with pytest.raises(
        TypeError, match="max_samples_predict must be an integer or None"
    ):
        GPS(
            density_regressor=DummyDensityEstimator(),
            outcome_regressor=DecisionTreeRegressor(max_depth=3, random_state=0),
            max_samples_predict="3",
        )

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        GPS(
            density_regressor=DummyDensityEstimator(),
            outcome_regressor=DecisionTreeRegressor(max_depth=3, random_state=0),
            max_samples_predict=0,
        )


def test_gps_preserves_public_max_samples_predict_argument():
    max_samples_predict = np.int64(3)

    estimator = GPS(
        density_regressor=DummyDensityEstimator(),
        outcome_regressor=DecisionTreeRegressor(max_depth=3, random_state=0),
        max_samples_predict=max_samples_predict,
    )

    assert estimator.max_samples_predict is max_samples_predict
    assert estimator._max_samples_predict == 3
