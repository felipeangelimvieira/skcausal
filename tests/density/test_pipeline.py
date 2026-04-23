import copy

import numpy as np
import polars as pl
import pytest

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.naive import NaiveDensityEstimator
from skcausal.density.pipeline import Pipeline
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


class RecordingDensityEstimator(BaseDensityEstimator):
    _tags = {"capability:multidimensional_treatment": True}

    def _fit(self, X, t):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_t_ = copy.deepcopy(t)
        return self

    def _predict_density(self, X, t):
        self.predict_X_ = copy.deepcopy(X)
        self.predict_t_ = copy.deepcopy(t)
        return np.ones((len(X), 1), dtype=float)


def test_pipeline_is_a_base_density_estimator():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("density", RecordingDensityEstimator()),
        ]
    )

    assert isinstance(pipeline, BaseDensityEstimator)


def test_pipeline_applies_transformations_to_requested_arguments():
    X = pl.DataFrame({"x": [0.0, 1.0], "z": [2.0, 3.0]})
    t = pl.DataFrame({"t": [4.0, 5.0]})
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
            ("density", RecordingDensityEstimator()),
        ]
    )

    pipeline.fit(X, t)

    expected_fit_X = pl.DataFrame({"x_x": [10.0, 11.0], "x_z": [12.0, 13.0]})
    expected_fit_t = pl.DataFrame({"t_t": [2.0, 3.0]})

    assert pipeline.density_estimator_.fit_X_.equals(expected_fit_X)
    assert pipeline.density_estimator_.fit_t_.equals(expected_fit_t)

    density = pipeline.predict_density(X, t)

    assert density.shape == (2, 1)
    assert pipeline.density_estimator_.predict_X_.equals(expected_fit_X)
    assert pipeline.density_estimator_.predict_t_.equals(expected_fit_t)


def test_pipeline_rejects_missing_final_density_estimator():
    with pytest.raises(
        TypeError, match="last pipeline step must be a BaseDensityEstimator"
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
    with pytest.raises(ValueError, match="apply_to must be either 'X' or 't'"):
        Pipeline(
            steps=[
                (
                    "transform_X",
                    ShiftAndRenameTransformation(offset=1.0, prefix="x_"),
                    "y",
                ),
                ("density", RecordingDensityEstimator()),
            ]
        )


def test_pipeline_rejects_density_estimator_before_final_step():
    with pytest.raises(ValueError, match="BaseDensityEstimator step must be the final"):
        Pipeline(
            steps=[
                ("density", RecordingDensityEstimator()),
                (
                    "transform_t",
                    ShiftAndRenameTransformation(offset=1.0, prefix="t_"),
                    "t",
                ),
            ]
        )


def test_pipeline_meta_object_mixin_exposes_nested_step_params():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("density", NaiveDensityEstimator(epsilon=1e-4)),
        ]
    )

    params = pipeline.get_params()

    assert params["transform_X__offset"] == 1.0
    assert params["density__epsilon"] == 1e-4

    pipeline.set_params(transform_X__offset=2.0, density__epsilon=1e-3)

    assert pipeline.get_params()["transform_X__offset"] == 2.0
    assert pipeline.get_params()["density__epsilon"] == 1e-3
    assert pipeline.steps[0][2] == "X"


def test_pipeline_meta_object_mixin_replaces_step_without_losing_apply_to():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("density", RecordingDensityEstimator()),
        ]
    )

    replacement = ShiftAndRenameTransformation(offset=3.0, prefix="new_")
    pipeline.set_params(transform_X=replacement)

    assert pipeline.steps[0][1] is replacement
    assert pipeline.steps[0][2] == "X"


def test_pipeline_meta_object_mixin_round_trips_get_set_params_without_losing_apply_to():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("density", NaiveDensityEstimator(epsilon=1e-4)),
        ]
    )

    params = pipeline.get_params()
    pipeline.set_params(**params)

    assert pipeline.steps[0][2] == "X"
    assert pipeline.get_params()["transform_X__offset"] == 1.0
    assert pipeline.get_params()["density__epsilon"] == 1e-4


def test_pipeline_meta_object_mixin_replaces_full_steps_without_losing_apply_to():
    pipeline = Pipeline(
        steps=[
            ("transform_X", ShiftAndRenameTransformation(offset=1.0, prefix="x_"), "X"),
            ("density", NaiveDensityEstimator(epsilon=1e-4)),
        ]
    )

    replacement_steps = [
        ("transform_t", ShiftAndRenameTransformation(offset=-2.0, prefix="t_"), "t"),
        ("density", NaiveDensityEstimator(epsilon=1e-3)),
    ]

    pipeline.set_params(steps=replacement_steps)

    assert pipeline.steps[0][0] == "transform_t"
    assert pipeline.steps[0][2] == "t"
    assert pipeline.get_params()["transform_t__offset"] == -2.0
    assert pipeline.get_params()["density__epsilon"] == 1e-3


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
            ("density", RecordingDensityEstimator()),
        ]
    )

    concatenated = pipeline._dunder_concat(
        ("transform_t", ShiftAndRenameTransformation(offset=-2.0, prefix="t_"), "t"),
        attr_name="steps",
        concat_order="right",
    )

    assert isinstance(concatenated, Pipeline)
    assert concatenated.steps[0][0] == "transform_t"
    assert concatenated.steps[0][2] == "t"
    assert concatenated.steps[1][0] == "transform_X"
    assert concatenated.steps[1][2] == "X"
    assert concatenated.steps[-1][0] == "density"

    X = pl.DataFrame({"x": [0.0, 1.0], "z": [2.0, 3.0]})
    t = pl.DataFrame({"t": [4.0, 5.0]})
    concatenated.fit(X, t)

    expected_fit_X = pl.DataFrame({"x_x": [1.0, 2.0], "x_z": [3.0, 4.0]})
    expected_fit_t = pl.DataFrame({"t_t": [2.0, 3.0]})

    assert concatenated.density_estimator_.fit_X_.equals(expected_fit_X)
    assert concatenated.density_estimator_.fit_t_.equals(expected_fit_t)
