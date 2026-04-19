from skbase.testing.test_all_objects import TestAllObjects
import polars as pl
from skbase.utils.dependencies import _check_soft_dependencies
from skcausal.density.base import BaseDensityEstimator
from skcausal.density.naive import NaiveDensityEstimator
from skcausal.density.optuna import OptunaSearchDensityEstimator
from skcausal.density.performance_evaluation.metrics.likelihood import (
    LogLikelihoodMetric,
)
from skcausal.datatypes import collect_column_types
from sklearn.datasets import make_regression, make_classification
from skcausal.testing._filter_scenarios import run_test_if, object_not_instance_of
import numpy as np

RAND_SEED = 42
INFORMATIVE_SCENARIO_NAMES = {
    "continuous_treatment",
    "boolean_treatment",
    "float32_treatment",
    "two_continuous_treatments",
    "continuous_binary_treatments",
    "two_binary_treatments",
}


class ContinuousTreatmentScenario:
    """Scenario with continuous (float64) treatment."""

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


class TwoContinuousTreatmentScenario:
    """Scenario with two continuous treatment columns."""

    def __init__(self):
        X, t = make_regression(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_targets=2,
            random_state=RAND_SEED + 3,
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        t = pl.DataFrame(t, schema=["t0", "t1"])
        self.t = t
        self.X = X
        self.t_dtypes = tuple(t.dtypes)


class ContinuousBinaryTreatmentScenario:
    """Scenario with one continuous and one binary treatment column."""

    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 4)
        X = rng.normal(size=(1000, 5))

        t_continuous = (
            1.3 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.4, size=X.shape[0])
        )
        t_binary = (
            0.8 * X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.3, size=X.shape[0]) > 0
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        t = pl.DataFrame(
            {
                "t_continuous": t_continuous.astype(np.float64),
                "t_binary": t_binary.astype(bool),
            }
        ).with_columns(pl.col("t_binary").cast(pl.Boolean))
        self.t = t
        self.X = X
        self.t_dtypes = tuple(t.dtypes)


class TwoBinaryTreatmentScenario:
    """Scenario with two binary treatment columns."""

    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 5)
        X = rng.normal(size=(1000, 5))

        t_binary_0 = (
            1.1 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.3, size=X.shape[0]) > 0
        )
        t_binary_1 = (
            -0.9 * X[:, 2] + 0.6 * X[:, 3] + rng.normal(scale=0.3, size=X.shape[0]) > 0
        )

        X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        t = pl.DataFrame(
            {
                "t0": t_binary_0.astype(bool),
                "t1": t_binary_1.astype(bool),
            }
        ).with_columns(pl.all().cast(pl.Boolean))
        self.t = t
        self.X = X
        self.t_dtypes = tuple(t.dtypes)


def _get_single_treatment_scenarios():
    """Return default single-treatment scenarios used by object tests."""
    return [
        ContinuousTreatmentScenario(),
        BooleanTreatmentScenario(),
        IntegerTreatmentScenario(),
        Float32TreatmentScenario(),
        EnumTreatmentScenario(),
    ], [
        "continuous_treatment",
        "boolean_treatment",
        "integer_treatment",
        "float32_treatment",
        "enum_treatment",
    ]


def _get_multidimensional_treatment_scenarios():
    """Return multidimensional treatment scenarios for filter regression tests."""
    return [
        TwoContinuousTreatmentScenario(),
        ContinuousBinaryTreatmentScenario(),
        TwoBinaryTreatmentScenario(),
    ], [
        "two_continuous_treatments",
        "continuous_binary_treatments",
        "two_binary_treatments",
    ]


def _get_all_treatment_scenarios():
    """Return all treatment scenarios used by the object test matrix."""
    single_scenarios, single_names = _get_single_treatment_scenarios()
    multidimensional_scenarios, multidimensional_names = (
        _get_multidimensional_treatment_scenarios()
    )
    return (
        single_scenarios + multidimensional_scenarios,
        single_names + multidimensional_names,
    )


def _filter_supported_scenarios(object_instance, scenarios, scenario_names):
    """Filter scenarios by dimensionality and treatment type support."""
    if object_instance is None:
        return scenarios, scenario_names

    capability_t_type = set(object_instance.get_tag("capability:t_type", []))
    supports_multidimensional_treatment = object_instance.get_tag(
        "capability:multidimensional_treatment", False
    )

    filtered_scenarios = []
    filtered_scenario_names = []
    for scenario, name in zip(scenarios, scenario_names):
        if scenario.t.shape[1] > 1 and not supports_multidimensional_treatment:
            continue

        scenario_t_types = set(collect_column_types(scenario.t).values())
        if not scenario_t_types.issubset(capability_t_type):
            continue

        filtered_scenarios.append(scenario)
        filtered_scenario_names.append(name)

    return filtered_scenarios, filtered_scenario_names


def _filter_informative_scenarios(scenarios, scenario_names):
    filtered_scenarios = []
    filtered_scenario_names = []

    for scenario, name in zip(scenarios, scenario_names):
        if name not in INFORMATIVE_SCENARIO_NAMES:
            continue

        filtered_scenarios.append(scenario)
        filtered_scenario_names.append(name)

    return filtered_scenarios, filtered_scenario_names


def _split_holdout(frame, n_train):
    return frame.slice(0, n_train), frame.slice(n_train, len(frame) - n_train)


class TestAllDensityEstimators(TestAllObjects):
    """Test class for all density estimators using skbase TestAllObjects."""

    package_name = "skcausal.density"
    valid_tags = [
        "backend",
        "capability:t_type",
        "capability:multidimensional_treatment",
        "density_kind",
        "soft_dependencies",
    ]

    object_type_filter = BaseDensityEstimator

    fixture_sequence = ["object_class", "object_instance", "scenario"]

    def _generate_object_class(self, test_name, **kwargs):
        object_classes_to_test = []
        object_names = []

        for est in self._all_objects():
            if self.is_excluded(test_name, est):
                continue

            soft_dependencies = est.get_class_tag("soft_dependencies", [])
            if soft_dependencies and not _check_soft_dependencies(
                *soft_dependencies, severity="none"
            ):
                continue

            object_classes_to_test.append(est)
            object_names.append(est.__name__)

        return object_classes_to_test, object_names

    def _generate_scenario(self, test_name, **kwargs):
        """
        Generates scenarios for testing.

        Filters scenarios based on the object_instance's supported dtypes,
        so tests are not even generated for unsupported dtypes.
        """
        object_instance = kwargs.get("object_instance")

        scenarios, scenario_names = _get_all_treatment_scenarios()
        scenarios, scenario_names = _filter_supported_scenarios(
            object_instance, scenarios, scenario_names
        )

        if test_name == "test_estimators_outperform_naive_density":
            scenarios, scenario_names = _filter_informative_scenarios(
                scenarios, scenario_names
            )

        test_fn = getattr(self, test_name)
        if hasattr(test_fn, "_scenario_filter"):
            filter_fn = test_fn._scenario_filter
            mask = filter_fn(object_instance, scenarios)
            scenarios = [s for s, m in zip(scenarios, mask) if m]
            scenario_names = [n for n, m in zip(scenario_names, mask) if m]

        return scenarios, scenario_names

    def test_fit_predict_density(self, object_instance, scenario):
        """Test that fit and predict_density run without errors and return expected shapes."""

        X = scenario.X
        t = scenario.t

        # Fit the model
        object_instance.fit(X, t)

        # Predict density
        density = object_instance.predict_density(X, t)

        # Check that the output is a numpy array with the correct shape
        assert isinstance(density, np.ndarray), "Output is not a numpy array"
        assert (
            density.shape[0] == X.shape[0]
        ), f"Output shape {density.shape[0]} is incorrect, expected {X.shape[0]}"
        assert np.isfinite(density).all(), "Output contains NaN or Inf values"
        assert (density >= 0).all(), "Density values must be non-negative"

    @run_test_if(
        object_not_instance_of(OptunaSearchDensityEstimator, NaiveDensityEstimator)
    )
    def test_estimators_outperform_naive_density(self, object_instance, scenario):
        n_train = int(0.7 * len(scenario.X))
        X_train, X_test = _split_holdout(scenario.X, n_train)
        t_train, t_test = _split_holdout(scenario.t, n_train)

        density_kind = object_instance.get_tag("density_kind")
        naive_estimator = NaiveDensityEstimator(density_kind=density_kind).fit(
            X_train, t_train
        )
        object_instance.fit(X_train, t_train)

        metric = LogLikelihoodMetric()
        naive_score = float(metric.evaluate(naive_estimator, X_test, t_test))
        estimator_score = float(metric.evaluate(object_instance, X_test, t_test))

        assert estimator_score > naive_score + 1e-6, (
            f"{object_instance.__class__.__name__} should outperform the naive "
            f"treatment prior on scenario {scenario.__class__.__name__}. "
            f"Observed estimator_score={estimator_score:.4f}, "
            f"naive_score={naive_score:.4f}."
        )


def test_scenarios_are_filtered_by_capability_and_t_type():
    class _ContinuousOnlyDensity(BaseDensityEstimator):
        _tags = {
            "capability:t_type": ["continuous"],
            "capability:multidimensional_treatment": True,
        }

        def _fit(self, X, t):
            return self

        def _predict_density(self, X, t):
            return np.ones((len(X), 1), dtype=float)

    class _CategoricalOnlyDensity(BaseDensityEstimator):
        _tags = {
            "capability:t_type": ["categorical"],
            "capability:multidimensional_treatment": True,
        }

        def _fit(self, X, t):
            return self

        def _predict_density(self, X, t):
            return np.ones((len(X), 1), dtype=float)

    class _AllTypesSingleTreatmentDensity(BaseDensityEstimator):
        _tags = {
            "capability:multidimensional_treatment": False,
        }

        def _fit(self, X, t):
            return self

        def _predict_density(self, X, t):
            return np.ones((len(X), 1), dtype=float)

    class _AllTypesMultidimensionalDensity(BaseDensityEstimator):
        _tags = {
            "capability:multidimensional_treatment": True,
        }

        def _fit(self, X, t):
            return self

        def _predict_density(self, X, t):
            return np.ones((len(X), 1), dtype=float)

    all_scenarios, all_names = _get_all_treatment_scenarios()

    # Continuous-only: passes continuous scenarios, rejects categorical
    _, continuous_only_names = _filter_supported_scenarios(
        _ContinuousOnlyDensity(), all_scenarios, all_names
    )
    assert "continuous_treatment" in continuous_only_names
    assert "integer_treatment" in continuous_only_names
    assert "float32_treatment" in continuous_only_names
    assert "enum_treatment" not in continuous_only_names
    assert "two_continuous_treatments" in continuous_only_names

    # Categorical-only: passes categorical scenarios, rejects continuous
    _, categorical_only_names = _filter_supported_scenarios(
        _CategoricalOnlyDensity(), all_scenarios, all_names
    )
    assert "enum_treatment" in categorical_only_names
    assert "continuous_treatment" not in categorical_only_names
    assert "integer_treatment" not in categorical_only_names
    assert "float32_treatment" not in categorical_only_names
    assert "two_continuous_treatments" not in categorical_only_names

    # Single treatment: rejects all multidimensional scenarios
    _, single_treatment_names = _filter_supported_scenarios(
        _AllTypesSingleTreatmentDensity(), all_scenarios, all_names
    )
    assert "two_continuous_treatments" not in single_treatment_names
    assert "continuous_binary_treatments" not in single_treatment_names
    assert "two_binary_treatments" not in single_treatment_names

    # Multidimensional: passes all multidimensional scenarios
    _, multidimensional_names = _filter_supported_scenarios(
        _AllTypesMultidimensionalDensity(), all_scenarios, all_names
    )
    assert "two_continuous_treatments" in multidimensional_names
    assert "continuous_binary_treatments" in multidimensional_names
    assert "two_binary_treatments" in multidimensional_names

    tester = TestAllDensityEstimators()
    _, default_object_test_names = tester._generate_scenario(
        "test_fit_predict_density",
        object_instance=_AllTypesMultidimensionalDensity(),
    )
    assert "two_continuous_treatments" in default_object_test_names
    assert "continuous_binary_treatments" in default_object_test_names
    assert "two_binary_treatments" in default_object_test_names
