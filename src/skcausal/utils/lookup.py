from skbase.lookup import all_objects
from skcausal.density.base import BaseDensityEstimator
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.datasets.base import BaseDataset
from typing import Optional

__all__ = [
    "all_datasets",
    "all_density_estimators",
    "all_causal_average_response_estimators",
]


def all_datasets(
    filter_tags=None,
    exclude_objects=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
    path: Optional[str] = None,
    modules_to_ignore=None,
    class_lookup=None,
):
    """Return a dictionary with all datasets."""
    return all_objects(
        BaseDataset,
        filter_tags,
        exclude_objects,
        return_names,
        as_dataframe,
        return_tags,
        suppress_import_stdout,
        package_name="skcausal.datasets",
        path=path,
        modules_to_ignore=modules_to_ignore,
        class_lookup=class_lookup,
    )


def all_density_estimators(
    filter_tags=None,
    exclude_objects=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
    path: Optional[str] = None,
    modules_to_ignore=None,
    class_lookup=None,
):
    """Return a dictionary with all density estimators."""
    return all_objects(
        BaseDensityEstimator,
        filter_tags,
        exclude_objects,
        return_names,
        as_dataframe,
        return_tags,
        suppress_import_stdout,
        package_name="skcausal.density",
        path=path,
        modules_to_ignore=modules_to_ignore,
        class_lookup=class_lookup,
    )


def all_causal_average_response_estimators(
    filter_tags=None,
    exclude_objects=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
    path: Optional[str] = None,
    modules_to_ignore=None,
    class_lookup=None,
):
    """Return a dictionary with all average causal response estimators."""
    return all_objects(
        BaseAverageCausalResponseEstimator,
        filter_tags,
        exclude_objects,
        return_names,
        as_dataframe,
        return_tags,
        suppress_import_stdout,
        package_name="skcausal.causal_estimators",
        path=path,
        modules_to_ignore=modules_to_ignore,
        class_lookup=class_lookup,
    )
