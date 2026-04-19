"""Decorates test for filtering pairs of instances, test and scenarios"""

from typing import Callable

__all__ = [
    "run_test_if",
    "has_tag",
    "object_not_instance_of",
]


def run_test_if(*conditions: Callable[[object, object], bool]):
    """
    Add _scenario_filter lambda to the test function.

    Parameters
    ----------
    *conditions : Callable[[object, object], bool]
        A variable number of conditions, each being a function that takes an
        object instance and a scenario and returns a boolean indicating
        whether the test should be run for that pair.

    Returns
    -------
    Callable
        A decorator that adds a _scenario_filter attribute to the test function,
        which can be used to filter scenarios based on the provided conditions.
    """

    def wrapped(test_fn):
        _scenario_filter = lambda object_instance, scenarios: [
            all(condition(object_instance, scenario) for condition in conditions)
            for scenario in scenarios
        ]
        test_fn._scenario_filter = _scenario_filter
        return test_fn

    return wrapped


def has_tag(tags):
    def condition(object_instance, scenario):
        return all(
            object_instance.get_tag(tag_key) == tag_value
            for tag_key, tag_value in tags.items()
        )

    return condition


def object_not_instance_of(*excluded_classes):
    def condition(object_instance, scenario):
        return not isinstance(object_instance, excluded_classes)

    return condition
