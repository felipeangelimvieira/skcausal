"""Load a supervised table from OpenML.

OpenML hosts the standard classification and regression benchmarks, the UCI
classics among them, so it covers the usual sources without vendoring data
files or adding a dependency: scikit-learn already ships the client.

This module owns one decision only, which dataset to load and which of its
columns is the target. The mapping onto the skcausal triple lives in
:mod:`skcausal.datasets.real.supervised` and knows nothing about OpenML.
"""

import pandas as pd
from sklearn.datasets import fetch_openml

from skcausal.datasets.real.supervised import _TARGET_KINDS as _SOURCE_TARGET_KINDS
from skcausal.datasets.real.supervised import (
    BaseSupervisedDataset,
    task_from_target_kind,
)

__all__ = ["OpenMLDataset"]

_TARGET_KINDS = ("auto", *_SOURCE_TARGET_KINDS)


def _infer_target_kind(target):
    """Return the measurement kind of a fetched target column."""

    # fetch_openml(as_frame=True) returns nominal targets as pandas categorical
    # dtype and leaves genuinely numeric ones as numbers, so the dtype is the
    # dataset's own declaration rather than a guess about its values. An
    # integer-coded classification target is the case this cannot see, which is
    # why target_kind can be set explicitly.
    if isinstance(target.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_numeric_dtype(target) and not pd.api.types.is_bool_dtype(target):
        return "numeric"
    return "categorical"


class OpenMLDataset(BaseSupervisedDataset):
    """A standard supervised benchmark fetched from OpenML.

    The first load downloads the dataset and caches it on disk; later loads
    read the cache. Construction performs no I/O, so building one of these is
    cheap and the fetch happens on :meth:`load`.

    Parameters
    ----------
    name : str or None, default=None
        OpenML dataset name, for example ``"adult"`` or ``"wine-quality-red"``.
        Give either ``name`` or ``data_id``.
    version : int or str, default="active"
        Dataset version. Pin an integer for a reproducible benchmark; the
        default follows whichever version OpenML currently considers active,
        which can change under you.
    data_id : int or None, default=None
        OpenML dataset id. Takes precedence over ``name`` and identifies a
        dataset exactly.
    target : str or None, default=None
        Column to expose as the outcome. When ``None``, the dataset's own
        declared default target is used. Point this at a numeric column when
        you need a numeric outcome from a dataset whose default target is a
        class label.
    target_kind : {"auto", "numeric", "categorical"}, default="auto"
        Measurement kind of the target. ``"auto"`` reads it from the fetched
        dtype, which is correct unless a classification target is stored as
        integer codes; set it explicitly in that case.
    data_home : str or None, default=None
        Directory for the on-disk cache. ``None`` uses scikit-learn's default.

    Examples
    --------
    >>> from skcausal.datasets import OpenMLDataset  # doctest: +SKIP
    >>> covariates, treatments, outcomes = OpenMLDataset(
    ...     name="wine-quality-red", version=1
    ... ).load()  # doctest: +SKIP
    """

    def __init__(
        self,
        name=None,
        version="active",
        data_id=None,
        target=None,
        target_kind="auto",
        data_home=None,
    ):
        if name is None and data_id is None:
            raise ValueError("Give either name or data_id to identify the dataset.")
        if target_kind not in _TARGET_KINDS:
            valid = ", ".join(_TARGET_KINDS)
            raise ValueError(
                f"target_kind must be one of {{{valid}}}. Got {target_kind!r}."
            )

        self.name = name
        self.version = version
        self.data_id = data_id
        self.target = target
        self.target_kind = target_kind
        self.data_home = data_home
        super().__init__()
        # "auto" leaves the task unknown until the fetch, so no tag is claimed.
        self.set_tags(task=task_from_target_kind(target_kind))

    def _fetch_supervised(self):
        bunch = self._fetch()

        features = bunch.data
        target = bunch.target
        if isinstance(target, pd.DataFrame):
            if target.shape[1] != 1:
                raise ValueError(
                    f"{self._describe()} exposes {target.shape[1]} target columns; "
                    "select one with the target parameter."
                )
            target = target.iloc[:, 0]

        kind = (
            _infer_target_kind(target)
            if self.target_kind == "auto"
            else self.target_kind
        )
        return features, target, kind

    def _fetch(self):
        try:
            return fetch_openml(
                name=self.name,
                version=self.version,
                data_id=self.data_id,
                target_column=(
                    "default-target" if self.target is None else self.target
                ),
                data_home=self.data_home,
                as_frame=True,
            )
        except Exception as error:
            # Anything from a wrong name to a dead network arrives here as a
            # provider-specific exception. Consumers of a dataset only know the
            # skcausal contract, so re-raise in its terms while chaining the
            # original for anyone debugging the fetch.
            raise ValueError(
                f"Could not load {self._describe()} from OpenML: {error}"
            ) from error

    def _describe(self):
        if self.data_id is not None:
            return f"dataset id {self.data_id}"
        return f"dataset {self.name!r} (version {self.version})"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        # Deliberately not registered for the shared offline dataset contract
        # test; see the registry note in the module docstring of
        # skcausal.datasets.real.supervised.
        return [{"name": "wine-quality-red", "version": 1}]
