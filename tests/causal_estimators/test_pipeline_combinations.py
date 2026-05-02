import copy

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin

from skcausal.causal_estimators.gps import GPS
from skcausal.causal_estimators.pipeline import Pipeline as AverageResponsePipeline
from skcausal.causal_estimators.pseudo_outcome import DoublyRobustPseudoOutcome
from skcausal.density.base import BaseDensityEstimator
from skcausal.density.pipeline import Pipeline as DensityPipeline
from skcausal.transformations.base import BaseTransformation


class ShiftAndRenamePandasTransformation(BaseTransformation):
    _tags = {"backend": "pandas"}

    def __init__(self, offset: float, prefix: str):
        self.offset = offset
        self.prefix = prefix
        super().__init__()

    def _fit(self, X):
        self.fit_columns_ = tuple(X.columns)
        return self

    def _transform(self, X):
        return X.add(self.offset).rename(
            columns=lambda column: f"{self.prefix}{column}"
        )


class ShiftAndRenamePolarsTransformation(BaseTransformation):
    _tags = {"backend": "polars"}

    def __init__(self, offset: float, prefix: str):
        self.offset = offset
        self.prefix = prefix
        super().__init__()

    def _fit(self, X):
        self.fit_columns_ = tuple(X.columns)
        return self

    def _transform(self, X):
        return X.select(
            [
                (pl.col(column) + self.offset).alias(f"{self.prefix}{column}")
                for column in X.columns
            ]
        )


class RecordingMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, prediction_value: float = 0.0):
        self.prediction_value = prediction_value

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()

        array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def fit(self, X, y):
        self.fit_X_ = self._to_numpy(X)
        self.fit_y_ = self._to_numpy(y)
        return self

    def predict(self, X):
        self.predict_X_ = self._to_numpy(X)
        return np.full(self.predict_X_.shape[0], self.prediction_value, dtype=float)


class RecordingStabilizedDensityEstimator(BaseDensityEstimator):
    _tags = {
        "backend": "pandas",
        "density_kind": "stabilized",
    }

    def _fit(self, X, t):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_t_ = copy.deepcopy(t)
        return self

    def _predict_density(self, X, t):
        self.predict_X_ = copy.deepcopy(X)
        self.predict_t_ = copy.deepcopy(t)
        return np.linspace(0.5, 1.5, len(X), dtype=float).reshape(-1, 1)


class RecordingConditionalDensityEstimator(BaseDensityEstimator):
    _tags = {
        "backend": "polars",
        "density_kind": "conditional",
        "capability:multidimensional_treatment": True,
    }

    def _fit(self, X, t):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_t_ = copy.deepcopy(t)
        return self

    def _predict_density(self, X, t):
        self.predict_X_ = copy.deepcopy(X)
        self.predict_t_ = copy.deepcopy(t)
        return (0.5 + 0.05 * np.arange(len(X), dtype=float)).reshape(-1, 1)


def test_density_pipeline_can_be_used_inside_doubly_robust_pseudo_outcome():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0, 7.0]})
    t = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"y": [10.0, 11.0, 12.0, 13.0]})
    density_pipeline = DensityPipeline(
        steps=[
            (
                "transform_X",
                ShiftAndRenamePandasTransformation(offset=5.0, prefix="x_"),
                "X",
            ),
            (
                "transform_t",
                ShiftAndRenamePandasTransformation(offset=-1.0, prefix="t_"),
                "t",
            ),
            ("density", RecordingStabilizedDensityEstimator()),
        ]
    )
    estimator = DoublyRobustPseudoOutcome(
        density_estimator=density_pipeline,
        outcome_regressor=RecordingMeanRegressor(prediction_value=2.0),
        pseudo_outcome_regressor=RecordingMeanRegressor(prediction_value=3.0),
        cv=0,
    )

    estimator.fit(X, t, y)

    expected_fit_X = pd.DataFrame(
        {"x_x": [5.0, 6.0, 7.0, 8.0], "x_z": [9.0, 10.0, 11.0, 12.0]}
    )
    expected_fit_t = pd.DataFrame({"t_t": [0.0, 1.0, 2.0, 3.0]})

    assert estimator.density_estimator_.get_tag("density_kind") == "stabilized"
    assert estimator.density_estimator_.density_estimator_.fit_X_.equals(expected_fit_X)
    assert estimator.density_estimator_.density_estimator_.fit_t_.equals(expected_fit_t)
    assert estimator.density_estimator_.density_estimator_.predict_X_.equals(
        expected_fit_X
    )
    assert estimator.density_estimator_.density_estimator_.predict_t_.equals(
        expected_fit_t
    )

    prediction = estimator.predict(pd.DataFrame({"t": [2.5, 4.5]}))

    assert prediction.shape == (2, 1)
    np.testing.assert_allclose(prediction, np.array([[3.0], [3.0]]))


def test_causal_and_density_pipelines_can_be_nested_together():
    X = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0, 7.0]})
    t = pl.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]})
    y = pl.DataFrame({"y": [10.0, 11.0, 12.0, 13.0]})
    density_pipeline = DensityPipeline(
        steps=[
            (
                "inner_X",
                ShiftAndRenamePolarsTransformation(offset=1.0, prefix="inner_"),
                "X",
            ),
            (
                "inner_t",
                ShiftAndRenamePolarsTransformation(offset=-1.0, prefix="inner_"),
                "t",
            ),
            ("density", RecordingConditionalDensityEstimator()),
        ]
    )
    estimator = AverageResponsePipeline(
        steps=[
            (
                "outer_X",
                ShiftAndRenamePolarsTransformation(offset=10.0, prefix="outer_"),
                "X",
            ),
            (
                "outer_t",
                ShiftAndRenamePolarsTransformation(offset=2.0, prefix="outer_"),
                "t",
            ),
            (
                "outer_y",
                ShiftAndRenamePolarsTransformation(offset=5.0, prefix="outer_"),
                "y",
            ),
            (
                "gps",
                GPS(
                    density_regressor=density_pipeline,
                    outcome_regressor=RecordingMeanRegressor(prediction_value=4.0),
                    cv=2,
                    random_state=0,
                ),
            ),
        ]
    )

    estimator.fit(X, t, y)

    nested_density = estimator.estimator_.density_regressor_.density_estimator_
    expected_fit_X = pl.DataFrame(
        {
            "inner_outer_x": [11.0, 12.0, 13.0, 14.0],
            "inner_outer_z": [15.0, 16.0, 17.0, 18.0],
        }
    )
    expected_fit_t = pl.DataFrame({"inner_outer_t": [2.0, 3.0, 4.0, 5.0]})
    transformed_y = np.array([[15.0], [16.0], [17.0], [18.0]])

    assert nested_density.fit_X_.equals(expected_fit_X)
    assert nested_density.fit_t_.equals(expected_fit_t)
    np.testing.assert_allclose(
        estimator.estimator_.outcome_regressor_.fit_y_,
        transformed_y[estimator.estimator_.oof_test_indices_],
    )

    response = estimator.predict(pl.DataFrame({"t": [10.0, 20.0]}))

    expected_predict_X = pl.DataFrame(
        {
            "inner_outer_x": [11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0],
            "inner_outer_z": [15.0, 16.0, 17.0, 18.0, 15.0, 16.0, 17.0, 18.0],
        }
    )
    expected_predict_t = pl.DataFrame(
        {"inner_outer_t": [11.0, 11.0, 11.0, 11.0, 21.0, 21.0, 21.0, 21.0]}
    )

    assert response.shape == (2, 1)
    np.testing.assert_allclose(response, np.array([[4.0], [4.0]]))
    assert nested_density.predict_X_.equals(expected_predict_X)
    assert nested_density.predict_t_.equals(expected_predict_t)
