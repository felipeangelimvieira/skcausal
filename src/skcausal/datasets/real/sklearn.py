"""Load a supervised table that scikit-learn already ships.

These are the datasets every benchmark reaches for first — wine, diabetes,
breast cancer — and scikit-learn vendors them, so naming one costs no download
and no dependency. That is the point of this source over
:class:`~skcausal.datasets.real.openml.OpenMLDataset`: a semi-synthetic dataset
built on it stays reproducible and offline.

This module owns one decision only, which loader to call and what kind of
target it returns. The mapping onto the skcausal triple lives in
:mod:`skcausal.datasets.real.supervised` and knows nothing about scikit-learn's
dataset module.
"""

from sklearn import datasets as sklearn_datasets

from skcausal.datasets.real.supervised import (
    BaseSupervisedDataset,
    task_from_target_kind,
)

__all__ = ["SklearnDataset"]

# name -> (loader attribute, target kind, downloads)
_DATASETS = {
    "breast_cancer": ("load_breast_cancer", "categorical", False),
    "california_housing": ("fetch_california_housing", "numeric", True),
    "diabetes": ("load_diabetes", "numeric", False),
    "digits": ("load_digits", "categorical", False),
    "iris": ("load_iris", "categorical", False),
    "wine": ("load_wine", "categorical", False),
}


class SklearnDataset(BaseSupervisedDataset):
    """A supervised benchmark that comes with scikit-learn.

    Construction performs no I/O; the loader runs on :meth:`load`. Every
    supported name but ``"california_housing"`` is vendored inside
    scikit-learn and therefore offline. ``"california_housing"`` is fetched
    over the network on first use and cached on disk by scikit-learn, so keep
    it out of anything that must run offline.

    A classification dataset's integer class codes are released as labels
    rather than numbers, since a class code carries no order for a dose
    response to follow.

    Parameters
    ----------
    name : {"breast_cancer", "california_housing", "diabetes", "digits", \
"iris", "wine"}
        Which scikit-learn dataset to expose. Required: a source that does not
        name its dataset means nothing, and a default would be omitted from the
        object's repr, hiding which data an example actually used.

    Examples
    --------
    >>> from skcausal.datasets import SklearnDataset
    >>> covariates, outcome = SklearnDataset(
    ...     name="diabetes"
    ... ).load_covariates_and_outcome()
    >>> covariates.shape
    (442, 10)
    """

    def __init__(self, name: str):
        if name not in _DATASETS:
            supported = ", ".join(sorted(_DATASETS))
            raise ValueError(
                f"Unknown scikit-learn dataset {name!r}. Supported names are: "
                f"{supported}."
            )
        self.name = name
        super().__init__()
        self.set_tags(task=task_from_target_kind(_DATASETS[name][1]))

    @classmethod
    def available_names(cls) -> list[str]:
        """Return the dataset names this source accepts."""

        return sorted(_DATASETS)

    def _fetch_supervised(self):
        loader_name, target_kind, _ = _DATASETS[self.name]
        loader = getattr(sklearn_datasets, loader_name)
        features, target = loader(return_X_y=True, as_frame=True)
        return features, target, target_kind

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        # Only offline datasets belong here: the object sweep instantiates
        # these and must not reach the network.
        return [{"name": "wine"}, {"name": "diabetes"}]
