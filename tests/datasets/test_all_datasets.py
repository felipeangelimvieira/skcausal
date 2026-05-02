import inspect

from skbase.testing.test_all_objects import QuickTester, BaseFixtureGenerator

from skcausal.datasets.base import BaseSyntheticDataset


class TestAllDatasets(QuickTester, BaseFixtureGenerator):

    package_name = "skcausal.datasets"
    valid_tags = ["object_type"]

    def test_load_returns_three_dataframes(self, object_instance):
        out = object_instance.load()
        assert len(out) == 3


class TestAllSyntheticDatasets(QuickTester, BaseFixtureGenerator):

    package_name = "skcausal.datasets"
    valid_tags = ["object_type"]

    def _generate_object_class(self, test_name, **kwargs):
        object_classes, _ = super()._generate_object_class(
            test_name=test_name,
            **kwargs,
        )
        synthetic_classes = [
            object_class
            for object_class in object_classes
            if issubclass(object_class, BaseSyntheticDataset)
        ]
        synthetic_names = [object_class.__name__ for object_class in synthetic_classes]
        return synthetic_classes, synthetic_names

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
