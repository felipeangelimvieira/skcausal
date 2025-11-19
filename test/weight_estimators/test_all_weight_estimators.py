from skbase.testing.test_all_objects import TestAllObjects
import pandas as pd
import pytest
import polars as pl
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from sklearn.datasets import make_regression, make_classification
import numpy as np

DTYPE_ITERABLES = (list, tuple, set)

RAND_SEED = 42


class ContinuousTreatmentScenario:
    """A simple scenario object for demonstration."""

    def __init__(self):

        X, t = make_regression(
            n_samples=1000, n_features=5, n_informative=2, random_state=RAND_SEED
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        t = pl.DataFrame(t, schema=["t"])
        self.t = t
        self.X = X
        self.t_dtype = t.schema["t"]


class BooleanTreatmentScenario:
    """Scenario with boolean treatment."""

    def __init__(self):
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=2,
            random_state=RAND_SEED,
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        t_series = pl.Series("t", y.astype(bool)).cast(pl.Boolean)
        t = t_series.to_frame()
        self.t = t
        self.X = X
        self.t_dtype = t.schema["t"]


class IntegerTreatmentScenario:
    """Scenario with integer treatment."""

    def __init__(self):
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=RAND_SEED,
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        rng = np.random.default_rng(RAND_SEED)
        t_values = rng.integers(0, 3, size=y.shape[0], endpoint=False).astype(np.int32)
        t_series = pl.Series("t", t_values, dtype=pl.Int32)
        t = t_series.to_frame()
        self.t = t
        self.X = X
        self.t_dtype = t.schema["t"]


class Float32TreatmentScenario:
    """Scenario with float32 treatment."""

    def __init__(self):
        X, t = make_regression(
            n_samples=1000,
            n_features=5,
            n_informative=2,
            random_state=RAND_SEED + 1,
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        t_series = pl.Series("t", t.astype(np.float32), dtype=pl.Float32)
        t = t_series.to_frame()
        self.t = t
        self.X = X
        self.t_dtype = t.schema["t"]


class EnumTreatmentScenario:
    """Scenario with enum treatment."""

    def __init__(self):
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=RAND_SEED + 2,
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        categories = ["control", "treated", "placebo"]
        rng = np.random.default_rng(RAND_SEED)
        enum_values = rng.choice(categories, size=y.shape[0])
        enum_dtype = pl.Enum(categories)
        t_series = pl.Series("t", enum_values).cast(enum_dtype)
        t = t_series.to_frame()
        self.t = t
        self.X = X
        self.t_dtype = t.schema["t"]


def _flatten_supported_dtypes(dtypes):
    flat = []
    for dtype in dtypes:
        if isinstance(dtype, DTYPE_ITERABLES):
            flat.extend(_flatten_supported_dtypes(dtype))
        else:
            flat.append(dtype)
    return flat


class TestAllBalancingWeightRegressor(TestAllObjects):

    package_name = "skcausal.weight_estimators"
    valid_tags = [
        "t_inner_mtype",
        "X_inner_mtype",
        "one_hot_encode_enum_columns",
        "capability:supports_multidimensional_treatment",
        "supported_t_dtypes",
    ]

    object_type_filter = BaseBalancingWeightRegressor

    fixture_sequence = ["object_class", "object_instance", "scenario"]

    def _generate_scenario(self, test_name, **kwargs):
        """
        Generates scenarios for testing.

        It can optionally use kwargs to access earlier fixtures, e.g.,
        object_instance = kwargs.get("object_instance")
        to create object-specific scenarios.
        """

        scenarios = [
            ContinuousTreatmentScenario(),
            BooleanTreatmentScenario(),
            IntegerTreatmentScenario(),
            Float32TreatmentScenario(),
            EnumTreatmentScenario(),
        ]

        scenario_names = [
            "continuous_treatment",
            "boolean_treatment",
            "integer_treatment",
            "float32_treatment",
            "enum_treatment",
        ]

        return scenarios, scenario_names

    def test_fit_predict_sample_weight(self, object_instance, scenario):
        """Test that fit and predict_sample_weight run without errors and return expected shapes."""

        X = scenario.X
        t = scenario.t
        scenario_dtype = scenario.t_dtype

        supported_dtypes = object_instance.get_tag("supported_t_dtypes")
        flat_supported_dtypes = _flatten_supported_dtypes(supported_dtypes)

        if scenario_dtype not in flat_supported_dtypes:
            pytest.skip(
                "Estimator does not declare support for treatment dtype"
                f" {scenario_dtype}"
            )

        # Fit the model
        object_instance.fit(X, t)

        # Predict sample weights
        sample_weights = object_instance.predict_sample_weight(X, t)

        # Check that the output is a numpy array with the correct shape
        assert isinstance(sample_weights, np.ndarray), "Output is not a numpy array"
        assert sample_weights.shape[0] == X.shape[0], "Output shape is incorrect"
        assert (
            sample_weights.ndim == 2 and sample_weights.shape[1] == 1
        ), "Output shape is incorrect"
        assert np.isfinite(sample_weights).all(), "Output contains NaN or Inf values"
