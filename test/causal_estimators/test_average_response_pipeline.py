import copy

import numpy as np
import polars as pl
import pytest

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.pipeline import Pipeline
from skcausal.transformations.base import BaseTransformation


class ShiftAndRenameTransformation(BaseTransformation):
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


class RecordingAverageResponseEstimator(BaseAverageCausalResponseEstimator):
    _tags = {
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": True,
    }

    def __init__(self, prediction_value: float = 1.0):
        self.prediction_value = prediction_value
        super().__init__()

    def _fit(self, X, t, y):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_t_ = copy.deepcopy(t)
        self.fit_y_ = copy.deepcopy(y)
        return self

    def _predict(self, X, t):
        self.predict_X_ = copy.deepcopy(X)
        self.predict_t_ = copy.deepcopy(t)
        return np.full(len(t), self.prediction_value, dtype=float)


def test_pipeline_is_a_base_average_causal_response_estimator():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("estimator", RecordingAverageResponseEstimator()),
        ]
    )

    assert isinstance(pipeline, BaseAverageCausalResponseEstimator)


def test_pipeline_applies_transformations_to_requested_arguments():
    X = pl.DataFrame({"x": [0.0, 1.0], "z": [2.0, 3.0]})
    t = pl.DataFrame({"t": [4.0, 5.0]})
    y = pl.DataFrame({"y": [10.0, 11.0]})
    pipeline = Pipeline(
        steps=[
            (
                "transform_X",
                ShiftAndRenameTransformation(offset=10.0, prefix="x_"),
                "X",
            ),
            (
                "transform_t",
                ShiftAndRenameTransformation(offset=-2.0, prefix="t_"),
                "t",
            ),
            (
                "transform_y",
                ShiftAndRenameTransformation(offset=5.0, prefix="y_"),
                "y",
            ),
            ("estimator", RecordingAverageResponseEstimator(prediction_value=3.0)),
        ]
    )

    pipeline.fit(X, t, y)

    expected_fit_X = pl.DataFrame({"x_x": [10.0, 11.0], "x_z": [12.0, 13.0]})
    expected_fit_t = pl.DataFrame({"t_t": [2.0, 3.0]})
    expected_fit_y = pl.DataFrame({"y_y": [15.0, 16.0]})

    assert pipeline.estimator_.fit_X_.equals(expected_fit_X)
    assert pipeline.estimator_.fit_t_.equals(expected_fit_t)
    assert pipeline.estimator_.fit_y_.equals(expected_fit_y)

    response = pipeline.predict(X, t)

    assert response.shape == (2,)
    np.testing.assert_allclose(response, np.array([3.0, 3.0]))
    assert pipeline.estimator_.predict_X_.equals(expected_fit_X)
    assert pipeline.estimator_.predict_t_.equals(expected_fit_t)


def test_pipeline_rejects_missing_final_average_response_estimator():
    with pytest.raises(
        TypeError,
        match="last pipeline step must be a BaseAverageCausalResponseEstimator",
    ):
        Pipeline(
            steps=[
                (
                    "transform_X",
                    ShiftAndRenameTransformation(offset=1.0, prefix="x_"),
                    "X",
                )
            ]
        )


def test_pipeline_rejects_invalid_apply_to():
    with pytest.raises(ValueError, match="apply_to must be either 'X', 't', or 'y'"):
        Pipeline(
            steps=[
                (
                    "transform_X",
                    ShiftAndRenameTransformation(offset=1.0, prefix="x_"),
                    "invalid",
                ),
                ("estimator", RecordingAverageResponseEstimator()),
            ]
        )


def test_pipeline_rejects_average_response_estimator_before_final_step():
    with pytest.raises(
        ValueError,
        match="BaseAverageCausalResponseEstimator step must be the final",
    ):
        Pipeline(
            steps=[
                ("estimator", RecordingAverageResponseEstimator()),
                (
                    "transform_y",
                    ShiftAndRenameTransformation(offset=1.0, prefix="y_"),
                    "y",
                ),
            ]
        )


def test_pipeline_meta_object_mixin_exposes_nested_step_params():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("estimator", RecordingAverageResponseEstimator(prediction_value=2.0)),
        ]
    )

    params = pipeline.get_params()

    assert params["transform_X__offset"] == 1.0
    assert params["estimator__prediction_value"] == 2.0

    pipeline.set_params(transform_X__offset=2.0, estimator__prediction_value=4.0)

    assert pipeline.get_params()["transform_X__offset"] == 2.0
    assert pipeline.get_params()["estimator__prediction_value"] == 4.0
    assert pipeline.steps[0][2] == "X"


def test_pipeline_meta_object_mixin_replaces_step_without_losing_apply_to():
    pipeline = Pipeline(
        steps=[
            ("transform_y", ShiftAndRenameTransformation(offset=1.0, prefix="y_"), "y"),
            ("estimator", RecordingAverageResponseEstimator()),
        ]
    )

    replacement = ShiftAndRenameTransformation(offset=3.0, prefix="new_")
    pipeline.set_params(transform_y=replacement)

    assert pipeline.steps[0][1] is replacement
    assert pipeline.steps[0][2] == "y"


def test_pipeline_get_test_params_are_instantiable():
    test_params = Pipeline.get_test_params()
    if isinstance(test_params, dict):
        test_params = [test_params]

    assert test_params

    for params in test_params:
        pipeline = Pipeline(**params)
        assert isinstance(pipeline, Pipeline)


def test_pipeline_dunder_concat_preserves_apply_to_step_metadata():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("estimator", RecordingAverageResponseEstimator()),
        ]
    )

    concatenated = pipeline._dunder_concat(
        ("transform_y", ShiftAndRenameTransformation(offset=-2.0, prefix="y_"), "y"),
        attr_name="steps",
        concat_order="right",
    )

    assert isinstance(concatenated, Pipeline)
    assert concatenated.steps[0][0] == "transform_y"
    assert concatenated.steps[0][2] == "y"
    assert concatenated.steps[1][0] == "transform_X"
    assert concatenated.steps[1][2] == "X"
    assert concatenated.steps[-1][0] == "estimator"

    X = pl.DataFrame({"x": [0.0, 1.0], "z": [2.0, 3.0]})
    t = pl.DataFrame({"t": [4.0, 5.0]})
    y = pl.DataFrame({"y": [10.0, 11.0]})
    concatenated.fit(X, t, y)

    expected_fit_X = pl.DataFrame({"x_x": [1.0, 2.0], "x_z": [3.0, 4.0]})
    expected_fit_y = pl.DataFrame({"y_y": [8.0, 9.0]})

    assert concatenated.estimator_.fit_X_.equals(expected_fit_X)
    assert concatenated.estimator_.fit_y_.equals(expected_fit_y)
