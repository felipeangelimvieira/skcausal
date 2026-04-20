import inspect
import numpy as np
import pandas as pd
import polars as pl
from skbase.testing.test_all_objects import BaseFixtureGenerator, QuickTester
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KernelDensity

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.categorical import (
    CategoricalDoublyRobust,
    BinaryPropensityWeighting,
)
from skcausal.causal_estimators.direct_method import DirectRegressor
from skcausal.causal_estimators.gps import GPS
from skcausal.causal_estimators.ignore_covariates import DirectNoCovariates
from skcausal.causal_estimators.pseudo_outcome import DoublyRobustPseudoOutcome
from skcausal.datatypes import collect_column_types
from skcausal.density.naive import NaiveDensityEstimator
from skcausal.density.stabilized_from_conditional import (
    KernelMarginalAndConditional,
)

RAND_SEED = 42
CURRENT_BASE_TAGS = {
    "backend",
    "capability:t_type",
    "capability:multidimensional_treatment",
    "one_hot_encode_enum_columns",
}


class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def fit(self, X, y, sample_weight=None):
        if isinstance(y, pd.DataFrame):
            values = y.to_numpy(dtype=float)
        elif isinstance(y, pd.Series):
            values = y.to_numpy(dtype=float)
        else:
            values = np.asarray(y, dtype=float)

        self.constant_ = float(values.reshape(-1).mean()) + float(self.offset)
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_, dtype=float)


class ContinuousTreatmentScenario:
    def __init__(self):
        rng = np.random.default_rng(RAND_SEED)
        X = rng.normal(size=(96, 4))
        t = 0.7 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.4, size=96)
        y = 0.6 * X[:, 0] + 0.4 * t + rng.normal(scale=0.2, size=96)

        self.X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.t = pl.DataFrame({"t": t.astype(np.float64)})
        self.y = pl.DataFrame({"y": y.astype(np.float64)})


class BooleanTreatmentScenario:
    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 1)
        X = rng.normal(size=(96, 4))
        logits = 0.8 * X[:, 0] - 0.6 * X[:, 2] + rng.normal(scale=0.3, size=96)
        t = logits > 0.0
        y = 0.5 * X[:, 1] + 1.0 * t.astype(float) + rng.normal(scale=0.2, size=96)

        self.X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.t = pl.DataFrame({"t": t}).with_columns(pl.col("t").cast(pl.Boolean))
        self.y = pl.DataFrame({"y": y.astype(np.float64)})


class EnumTreatmentScenario:
    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 2)
        X = rng.normal(size=(96, 4))
        categories = ["control", "treated", "placebo"]
        t_values = rng.choice(categories, size=96)
        treatment_effect = {
            "control": 0.0,
            "treated": 1.0,
            "placebo": -0.4,
        }
        y = np.array(
            [0.4 * X[i, 0] + treatment_effect[t_values[i]] for i in range(96)],
            dtype=float,
        )
        y += rng.normal(scale=0.2, size=96)

        self.X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.t = pl.Series("t", t_values).cast(pl.Enum(categories)).to_frame()
        self.y = pl.DataFrame({"y": y.astype(np.float64)})


class TwoBinaryTreatmentScenario:
    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 5)
        X = rng.normal(size=(96, 4))
        t0 = 0.7 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.3, size=96) > 0.0
        t1 = -0.4 * X[:, 2] + 0.6 * X[:, 3] + rng.normal(scale=0.3, size=96) > 0.0
        y = (
            0.4 * X[:, 0]
            + 0.8 * t0.astype(float)
            - 0.5 * t1.astype(float)
            + 0.6 * (t0 & t1).astype(float)
        )
        y += rng.normal(scale=0.2, size=96)

        self.X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.t = pl.DataFrame({"t0": t0, "t1": t1}).with_columns(
            pl.all().cast(pl.Boolean)
        )
        self.y = pl.DataFrame({"y": y.astype(np.float64)})


class TwoContinuousTreatmentScenario:
    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 3)
        X = rng.normal(size=(96, 4))
        t0 = 0.6 * X[:, 0] + rng.normal(scale=0.3, size=96)
        t1 = -0.5 * X[:, 1] + rng.normal(scale=0.3, size=96)
        y = 0.5 * X[:, 0] + 0.3 * t0 - 0.7 * t1 + rng.normal(scale=0.2, size=96)

        self.X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.t = pl.DataFrame(
            {"t0": t0.astype(np.float64), "t1": t1.astype(np.float64)}
        )
        self.y = pl.DataFrame({"y": y.astype(np.float64)})


class ContinuousBinaryTreatmentScenario:
    def __init__(self):
        rng = np.random.default_rng(RAND_SEED + 4)
        X = rng.normal(size=(96, 4))
        t0 = 0.9 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.3, size=96)
        t1 = 0.5 * X[:, 2] + rng.normal(scale=0.3, size=96) > 0.0
        y = 0.3 * X[:, 0] + 0.5 * t0 + 0.9 * t1.astype(float)
        y += rng.normal(scale=0.2, size=96)

        self.X = pl.DataFrame(X, schema=[f"X{i}" for i in range(X.shape[1])])
        self.t = pl.DataFrame({"t0": t0.astype(np.float64), "t1": t1}).with_columns(
            pl.col("t1").cast(pl.Boolean)
        )
        self.y = pl.DataFrame({"y": y.astype(np.float64)})


def _get_all_treatment_scenarios():
    return [
        ContinuousTreatmentScenario(),
        BooleanTreatmentScenario(),
        EnumTreatmentScenario(),
        TwoBinaryTreatmentScenario(),
        TwoContinuousTreatmentScenario(),
        ContinuousBinaryTreatmentScenario(),
    ], [
        "continuous_treatment",
        "boolean_treatment",
        "enum_treatment",
        "two_binary_treatments",
        "two_continuous_treatments",
        "continuous_binary_treatments",
    ]


def _filter_supported_scenarios(object_instance, scenarios, scenario_names):
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


def _has_current_base_tags(estimator_class) -> bool:
    local_tags = getattr(estimator_class, "_tags", {})
    return "backend" in local_tags and set(local_tags).issubset(CURRENT_BASE_TAGS)


def _build_test_instance(estimator_class):
    if estimator_class is BinaryPropensityWeighting:
        return BinaryPropensityWeighting(treatment_regressor=NaiveDensityEstimator())
    if estimator_class is CategoricalDoublyRobust:
        return CategoricalDoublyRobust(
            treatment_regressor=NaiveDensityEstimator(),
            outcome_regressor=MeanRegressor(),
        )
    if estimator_class is DirectRegressor:
        return DirectRegressor(outcome_regressor=MeanRegressor())
    if estimator_class is DirectNoCovariates:
        return DirectNoCovariates(outcome_regressor=MeanRegressor(), random_state=0)
    if estimator_class is GPS:
        return GPS(
            density_regressor=NaiveDensityEstimator(),
            outcome_regressor=MeanRegressor(),
            cv=2,
            random_state=0,
        )
    if estimator_class is DoublyRobustPseudoOutcome:
        return DoublyRobustPseudoOutcome(
            density_estimator=KernelMarginalAndConditional(
                conditional_density_estimator=NaiveDensityEstimator(),
                kernel=KernelDensity(bandwidth=0.5),
            ),
            outcome_regressor=MeanRegressor(),
            pseudo_outcome_regressor=MeanRegressor(offset=0.1),
            n_pseudo_samples=24,
            random_state=0,
        )
    raise ValueError(
        f"No test instance factory registered for {estimator_class.__name__}."
    )


class TestAllAverageCausalResponseEstimators(QuickTester, BaseFixtureGenerator):
    package_name = "skcausal.causal_estimators"
    valid_tags = sorted(CURRENT_BASE_TAGS)
    object_type_filter = BaseAverageCausalResponseEstimator
    fixture_sequence = ["object_class", "object_instance", "scenario"]

    def _generate_object_class(self, test_name, **kwargs):
        object_classes_to_test = []
        object_names = []

        for estimator_class in self._all_objects():
            if estimator_class is BaseAverageCausalResponseEstimator:
                continue
            if not _has_current_base_tags(estimator_class):
                continue

            object_classes_to_test.append(estimator_class)
            object_names.append(estimator_class.__name__)

        return object_classes_to_test, object_names

    def _generate_object_instance(self, test_name, **kwargs):
        object_class = kwargs.get("object_class")

        if object_class is None:
            object_classes, _ = self._generate_object_class(test_name=test_name)
        else:
            object_classes = [object_class]

        object_instances = []
        object_names = []
        for estimator_class in object_classes:
            object_instances.append(_build_test_instance(estimator_class))
            object_names.append(estimator_class.__name__)

        return object_instances, object_names

    def _generate_scenario(self, test_name, **kwargs):
        object_instance = kwargs.get("object_instance")
        scenarios, scenario_names = _get_all_treatment_scenarios()
        return _filter_supported_scenarios(object_instance, scenarios, scenario_names)

    def test_object_class_tags_match_current_base_contract(self, object_class):
        local_tags = getattr(object_class, "_tags", {})
        assert "backend" in local_tags
        assert set(local_tags).issubset(CURRENT_BASE_TAGS)

    def test_internal_method_signatures_follow_base_contract(self, object_class):
        fit_parameters = list(inspect.signature(object_class._fit).parameters)
        predict_parameters = list(inspect.signature(object_class._predict).parameters)

        assert fit_parameters == ["self", "X", "t", "y"]
        assert predict_parameters == ["self", "X", "t"]

    def test_fit_predict_average_response(self, object_instance, scenario):
        object_instance.fit(scenario.X, scenario.t, scenario.y)
        response = np.asarray(
            object_instance.predict(scenario.X, scenario.t), dtype=float
        )

        assert response.shape[0] == scenario.t.shape[0]
        assert np.isfinite(response).all()


def test_current_estimators_are_included_in_object_matrix():
    tester = TestAllAverageCausalResponseEstimators()
    _, object_names = tester._generate_object_class("test_fit_predict_average_response")

    assert "BinaryDoublyRobust" in object_names
    assert "BinaryPropensityWeighting" in object_names
    assert "DirectRegressor" in object_names
    assert "DirectNoCovariates" in object_names
    assert "DoublyRobustPseudoOutcome" in object_names
    assert "GPS" in object_names


def test_categorical_estimators_receive_boolean_and_enum_scenarios():
    tester = TestAllAverageCausalResponseEstimators()

    _, doubly_robust_scenarios = tester._generate_scenario(
        "test_fit_predict_average_response",
        object_instance=_build_test_instance(CategoricalDoublyRobust),
    )
    _, weighting_scenarios = tester._generate_scenario(
        "test_fit_predict_average_response",
        object_instance=_build_test_instance(BinaryPropensityWeighting),
    )

    assert doubly_robust_scenarios == [
        "boolean_treatment",
        "enum_treatment",
        "two_binary_treatments",
    ]
    assert weighting_scenarios == [
        "boolean_treatment",
        "enum_treatment",
        "two_binary_treatments",
    ]


def test_scenarios_are_filtered_by_capability_and_t_type():
    class _ContinuousOnlyEstimator(BaseAverageCausalResponseEstimator):
        _tags = {
            "backend": "polars",
            "capability:t_type": ["continuous"],
            "capability:multidimensional_treatment": True,
        }

        def _fit(self, X, t, y):
            return self

        def _predict(self, X, t):
            return np.ones(len(t), dtype=float)

    class _CategoricalOnlyEstimator(BaseAverageCausalResponseEstimator):
        _tags = {
            "backend": "polars",
            "capability:t_type": ["categorical"],
            "capability:multidimensional_treatment": True,
        }

        def _fit(self, X, t, y):
            return self

        def _predict(self, X, t):
            return np.ones(len(t), dtype=float)

    class _AllTypesSingleTreatmentEstimator(BaseAverageCausalResponseEstimator):
        _tags = {
            "backend": "polars",
            "capability:t_type": ["continuous", "categorical"],
            "capability:multidimensional_treatment": False,
        }

        def _fit(self, X, t, y):
            return self

        def _predict(self, X, t):
            return np.ones(len(t), dtype=float)

    class _AllTypesMultidimensionalEstimator(BaseAverageCausalResponseEstimator):
        _tags = {
            "backend": "polars",
            "capability:t_type": ["continuous", "categorical"],
            "capability:multidimensional_treatment": True,
        }

        def _fit(self, X, t, y):
            return self

        def _predict(self, X, t):
            return np.ones(len(t), dtype=float)

    all_scenarios, all_names = _get_all_treatment_scenarios()

    _, continuous_only_names = _filter_supported_scenarios(
        _ContinuousOnlyEstimator(), all_scenarios, all_names
    )
    assert "continuous_treatment" in continuous_only_names
    assert "boolean_treatment" not in continuous_only_names
    assert "enum_treatment" not in continuous_only_names
    assert "two_continuous_treatments" in continuous_only_names
    assert "continuous_binary_treatments" not in continuous_only_names

    _, categorical_only_names = _filter_supported_scenarios(
        _CategoricalOnlyEstimator(), all_scenarios, all_names
    )
    assert "boolean_treatment" in categorical_only_names
    assert "enum_treatment" in categorical_only_names
    assert "two_binary_treatments" in categorical_only_names
    assert "continuous_treatment" not in categorical_only_names
    assert "two_continuous_treatments" not in categorical_only_names

    _, single_treatment_names = _filter_supported_scenarios(
        _AllTypesSingleTreatmentEstimator(), all_scenarios, all_names
    )
    assert "two_continuous_treatments" not in single_treatment_names
    assert "continuous_binary_treatments" not in single_treatment_names

    _, multidimensional_names = _filter_supported_scenarios(
        _AllTypesMultidimensionalEstimator(), all_scenarios, all_names
    )
    assert "two_binary_treatments" in multidimensional_names
    assert "two_continuous_treatments" in multidimensional_names
    assert "continuous_binary_treatments" in multidimensional_names
