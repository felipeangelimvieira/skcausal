"""Expose a standard supervised table as a skcausal dataset.

A classification or regression benchmark (UCI and friends) is a table of
features plus one target column. It has no treatment, and these classes do not
invent one: they release ``(covariates, outcome)``, not the causal triple.

Validating that table is what :class:`BaseSupervisedDataset` owns, and it is
all it owns. Where the table is loaded from, and which of its columns is the
target, belong to the subclass that implements ``_fetch_supervised``. The split
is deliberate: the loading source is expected to change (a vendored CSV, an
OpenML fetch, a parquet file, a frame already in memory) while the validation
does not. Nothing in this module may know which source is in use.
"""

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseTabularDataset

__all__ = ["BaseSupervisedDataset", "SupervisedFrame"]

_DEFAULT_TARGET_NAME = "target"
_TARGET_KINDS = ("numeric", "categorical")
_TASK_BY_TARGET_KIND = {"numeric": "regression", "categorical": "classification"}


def task_from_target_kind(target_kind):
    """Return the ``"task"`` tag value for a declared target kind.

    ``None`` for a kind that is not known before loading, such as OpenML's
    ``"auto"``: an unknown task is not a mismatch, so consumers let it through.
    """

    return _TASK_BY_TARGET_KIND.get(target_kind)


def _as_target_series(target, name):
    """Return ``target`` as a named Polars series."""

    if isinstance(target, pl.Series):
        return target if name is None else target.rename(name)

    if name is None:
        # A pandas Series carries its own name; anything else gets the default.
        name = getattr(target, "name", None)
        if name is None:
            name = _DEFAULT_TARGET_NAME
    values = target.to_numpy() if hasattr(target, "to_numpy") else np.asarray(target)
    if values.ndim != 1:
        # polars would accept a 2-D array as a nested dtype rather than raising.
        raise ValueError(
            "The supervised target must be one-dimensional, but it has shape "
            f"{values.shape}."
        )
    return pl.Series(str(name), values)


class BaseSupervisedDataset(BaseTabularDataset):
    """Release a supervised table as ``(covariates, outcome)``.

    Features become the covariates and the target becomes the single outcome
    column. A supervised benchmark has no treatment, so none is released — the
    semi-synthetic datasets that consume these sources generate their own.

    Subclasses implement ``_fetch_supervised`` and own the loading decision.
    They must not rely on anything this class does beyond that hook.

    Notes
    -----
    The fetched table crosses a trust boundary: it comes from outside the
    package and its shape is not under our control. It is therefore validated
    here rather than downstream, where a malformed table would surface as a
    confusing failure inside a data-generating process, or not at all.
    """

    def _fetch_supervised(self):
        """Return the supervised table this dataset exposes.

        Returns
        -------
        features : DataFrame
            One row per sample, at least one column. Unique column names are
            enforced by the Polars conversion, not re-checked here.
        target : Series or array-like
            One value per feature row, in the same row order.
        target_kind : {"numeric", "categorical"}
            Measurement kind of the target. ``"numeric"`` covers regression
            targets, ``"categorical"`` covers classification labels.
        """

        raise NotImplementedError(
            "Supervised dataset classes should implement _fetch_supervised."
        )

    def load(self):
        """Return this table's ``(covariates, outcome)`` in the configured backend."""

        covariates, outcome = self._load()
        return (
            self._coerce_backend_frame(covariates),
            self._coerce_backend_frame(outcome),
        )

    def load_covariates_and_outcome(self):
        # A supervised table has no treatment to drop.
        return self.load()

    def _load(self):
        features, target, target_kind = self._fetch_supervised()

        if target_kind not in _TARGET_KINDS:
            valid = ", ".join(_TARGET_KINDS)
            raise ValueError(
                f"target_kind must be one of {{{valid}}}. Got {target_kind!r}."
            )

        covariates = self._coerce_backend_frame(features, backend="polars")
        if covariates.height == 0 or covariates.width == 0:
            raise ValueError(
                "A supervised source must provide at least one feature column and "
                "one row."
            )
        outcome = _as_target_series(target, None)
        if outcome.len() != covariates.height:
            raise ValueError(
                "The supervised target must have one value per feature row: got "
                f"{outcome.len()} target values for {covariates.height} rows."
            )
        if outcome.name in covariates.columns:
            raise ValueError(
                f"The target column {outcome.name!r} must not also appear among the "
                "features."
            )
        if outcome.null_count():
            raise ValueError("The supervised target must not contain missing values.")

        outcome = self._check_target_kind(outcome, target_kind)
        return covariates, outcome.to_frame()

    def _check_target_kind(self, outcome, target_kind):
        """Validate the target against its declared kind and normalize its dtype."""

        if target_kind == "numeric":
            if outcome.dtype.is_numeric():
                outcome = outcome.cast(pl.Float64)
            else:
                # Declaring the kind is the source's job, and a source may hold
                # genuinely numeric values in a label dtype: OpenML publishes
                # ordinal scores such as wine quality as a nominal class. Honour
                # the declaration by coercing, and fail loudly when the values
                # are not numbers after all.
                try:
                    outcome = outcome.cast(pl.Float64, strict=True)
                except Exception as error:
                    raise ValueError(
                        f"A target declared numeric must hold numbers, but "
                        f"{outcome.name!r} is {outcome.dtype} and could not be "
                        f"read as numeric: {error}"
                    ) from error
            if not np.isfinite(outcome.to_numpy()).all():
                raise ValueError("A numeric supervised target must be finite.")
        else:
            outcome = outcome.cast(pl.Utf8)

        if outcome.n_unique() < 2:
            raise ValueError(
                "The supervised target must have at least two distinct values; a "
                "constant target carries no signal for a semi-synthetic source."
            )
        return outcome


class SupervisedFrame(BaseSupervisedDataset):
    """Expose a supervised table that is already in memory.

    This is the source to use when the table is at hand: your own dataframe, a
    file you loaded yourself, or a small fixture. It performs no I/O, which
    also makes it the offline stand-in for sources that fetch.

    Parameters
    ----------
    features : DataFrame or mapping of str to sequence
        Feature table, or anything ``polars.DataFrame`` accepts.
    target : Series or array-like
        One target value per feature row, in the same row order.
    target_kind : {"numeric", "categorical"}, default="numeric"
        Measurement kind of the target. Use ``"categorical"`` for
        classification labels, so that consumers treat them as levels rather
        than as numbers.
    target_name : str or None, default=None
        Name for the released outcome column. When ``None``, the target's own
        name is used if it has one, and ``"target"`` otherwise.

    Examples
    --------
    >>> from skcausal.datasets import SupervisedFrame
    >>> dataset = SupervisedFrame(
    ...     features={"age": [21.0, 35.0, 48.0], "city": ["n", "s", "n"]},
    ...     target=[1.5, 2.5, 3.5],
    ... )
    >>> covariates, outcomes = dataset.load()
    >>> covariates.columns
    ['age', 'city']
    >>> outcomes.columns
    ['target']
    """

    def __init__(self, features, target, target_kind="numeric", target_name=None):
        self.features = features
        self.target = target
        self.target_kind = target_kind
        self.target_name = target_name
        super().__init__()
        self.set_tags(task=task_from_target_kind(target_kind))

    def _fetch_supervised(self):
        features = self.features
        if not hasattr(features, "columns") and not isinstance(features, np.ndarray):
            features = pl.DataFrame(features)
        target = _as_target_series(self.target, self.target_name)
        return features, target, self.target_kind

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        rows = 24
        index = np.arange(rows, dtype=float)
        features = {
            "x_continuous": np.sin(index),
            "x_linear": index / rows,
            "x_categorical": ["a", "b", "c"] * (rows // 3),
        }
        return [
            {
                "features": features,
                "target": index / rows + np.cos(index) / 4.0,
                "target_kind": "numeric",
                "target_name": "source_y",
            },
            {
                "features": features,
                "target": ["low", "high"] * (rows // 2),
                "target_kind": "categorical",
                "target_name": "source_label",
            },
        ]
