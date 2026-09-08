import inspect

import pandas as pd
from skbase.testing.test_all_objects import BaseFixtureGenerator, QuickTester

from skcausal import config
from skcausal.datasets.base import (
    BaseDataset,
    BaseKnownResponseDataset,
    BaseTabularDataset,
)
from skcausal.datatypes import collect_column_types

# The GasDrift benchmarks fetch their source from OpenML when constructed, so
# they cannot take part in an offline sweep. Their shape is the same as the
# vendored-source classes the sweep does cover; the fetch itself has an opt-in
# test in tests/datasets/semisynthetic/test_model_induced_benchmarks.py.
NETWORK_DATASETS = ["ModelInducedGasDriftContinuous2D", "ModelInducedGasDriftMixed"]


def _only_subclasses_of(base, object_classes, _names):
    """Narrow a swept class list to the subclasses of ``base``."""

    kept = [cls for cls in object_classes if issubclass(cls, base)]
    return kept, [cls.__name__ for cls in kept]


class TestAllTabularDatasets(QuickTester, BaseFixtureGenerator):
    """The contract BaseTabularDataset owns, swept over every dataset.

    Causal datasets and supervised sources disagree about treatment and so
    about what ``load`` returns. What they agree on is that a table of
    covariates and an outcome can be obtained from them, which is what this
    sweep pins — for both families, in one place.
    """

    package_name = "skcausal.datasets"
    valid_tags = ["object_type"]
    # OpenMLDataset fetches over the network on load, so it cannot take part in
    # an offline sweep. Its behaviour is inherited from BaseSupervisedDataset
    # and covered offline through SupervisedFrame; the fetch itself has an
    # opt-in test in tests/datasets/real/test_openml.py.
    exclude_objects = ["OpenMLDataset", *NETWORK_DATASETS]

    def _generate_object_class(self, test_name, **kwargs):
        return _only_subclasses_of(
            BaseTabularDataset,
            *super()._generate_object_class(test_name=test_name, **kwargs),
        )

    def test_covariates_and_outcome_are_two_aligned_frames(self, object_instance):
        covariates, outcome = object_instance.load_covariates_and_outcome()

        assert covariates.shape[0] == outcome.shape[0] > 0
        assert covariates.shape[1] > 0
        assert outcome.shape[1] >= 1

    def test_task_tag_is_declared(self, object_instance):
        """Every dataset names its supervised task; the None default is a stub."""

        assert object_instance.get_tag("task") in ("regression", "classification")

    def test_covariates_and_outcome_agree_with_load(self, object_instance):
        """The pair accessor must not be a second, drifting implementation."""

        covariates, outcome = object_instance.load_covariates_and_outcome()
        loaded = object_instance.load()

        assert loaded[0].equals(covariates)
        assert loaded[-1].equals(outcome)

    def test_covariates_and_outcome_respect_configured_default_backend(
        self, object_instance, monkeypatch
    ):
        monkeypatch.setattr(config, "default_backend", "pandas")

        covariates, outcome = object_instance.load_covariates_and_outcome()

        assert isinstance(covariates, pd.DataFrame)
        assert isinstance(outcome, pd.DataFrame)


class TestAllDatasets(QuickTester, BaseFixtureGenerator):
    """The causal-triple contract, swept over every dataset that claims it.

    Supervised sources are deliberately absent: they are not causal datasets
    and release ``(covariates, outcome)``. Their contract lives in
    tests/datasets/real/test_supervised.py.
    """

    package_name = "skcausal.datasets"
    valid_tags = ["object_type"]
    exclude_objects = NETWORK_DATASETS

    def _generate_object_class(self, test_name, **kwargs):
        return _only_subclasses_of(
            BaseDataset, *super()._generate_object_class(test_name=test_name, **kwargs)
        )

    def test_load_returns_three_dataframes(self, object_instance):
        out = object_instance.load()
        assert len(out) == 3

    def test_load_respects_configured_default_backend(
        self, object_instance, monkeypatch
    ):
        monkeypatch.setattr(config, "default_backend", "pandas")

        covariates, treatments, outcomes = object_instance.load()

        assert isinstance(covariates, pd.DataFrame)
        assert isinstance(treatments, pd.DataFrame)
        assert isinstance(outcomes, pd.DataFrame)

    def test_treatments_follow_declared_column_types(self, object_instance):
        _, treatments, _ = object_instance.load()

        assert collect_column_types(treatments) == object_instance.column_types


class TestAllKnownResponseDatasets(QuickTester, BaseFixtureGenerator):
    """The contract a benchmarkable dataset owns, whether or not it takes ``n``."""

    package_name = "skcausal.datasets"
    valid_tags = ["object_type"]
    exclude_objects = NETWORK_DATASETS

    def _generate_object_class(self, test_name, **kwargs):
        return _only_subclasses_of(
            BaseKnownResponseDataset,
            *super()._generate_object_class(test_name=test_name, **kwargs),
        )

    def test_has_random_state_parameter(self, object_class):
        signature = inspect.signature(object_class.__init__)

        assert "random_state" in signature.parameters
        assert "seed" not in signature.parameters

    def test_is_deterministic_for_random_state(self, object_instance):
        cloned_instance = object_instance.clone()

        for first_frame, second_frame in zip(
            object_instance.load(), cloned_instance.load()
        ):
            assert first_frame.equals(second_frame)

    def test_changing_random_state_changes_treatments_or_outcomes(
        self, object_instance
    ):
        varied_params = object_instance.get_params(deep=False).copy()
        varied_params["random_state"] = int(varied_params["random_state"]) + 1
        varied_instance = type(object_instance)(**varied_params)

        _, treatments, outcomes = object_instance.load()
        _, varied_treatments, varied_outcomes = varied_instance.load()

        assert not treatments.equals(varied_treatments) or not outcomes.equals(
            varied_outcomes
        )
