from skbase.testing.test_all_objects import QuickTester, BaseFixtureGenerator


class TestAllDatasets(QuickTester, BaseFixtureGenerator):

    package_name = "skcausal.datasets"
    valid_tags = ["object_type"]

    def test_load_returns_three_dataframes(self, object_instance):
        out = object_instance.load()
        assert len(out) == 3
