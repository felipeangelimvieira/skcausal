"""Extension template for supervised table sources.

Purpose of this template
------------------------
Use this file as a starting point when adding a source under
``skcausal.datasets.real``: a plain classification or regression table
(features plus one target column) with no treatment. These are the sources the
semi-synthetic datasets consume — see ``model_induced_dataset.py`` — and they
can be benchmarked on their own as ``(covariates, outcome)``.

If your table already carries a treatment, it is a causal dataset: use
``observational_dataset.py`` instead.

How to use this template
------------------------
1. Copy the file to a module with a descriptive name.
2. Rename ``MySupervisedDataset``.
3. Update the module and class docstrings, and cite the data source.
4. Implement ``_fetch_supervised``.
5. Declare the ``task`` tag, or set it in ``__init__`` when the target kind is
   a constructor argument.
6. Add ``get_test_params`` so the source can be instantiated in local tests.

Repo-specific contract notes
----------------------------
* ``_fetch_supervised`` is the only hook: it owns *where* the table comes from
  and nothing else. Validation of the table lives in ``BaseSupervisedDataset``
  and must not be duplicated here.
* The target kind you declare drives the released dtype: ``"numeric"`` is cast
  to floats, ``"categorical"`` to string labels, so that class codes are not
  mistaken for a dose.
* The base class rejects a target with missing values, with fewer than two
  distinct values, with the wrong length, or whose name collides with a
  feature column. Let it: those errors are already clear.
* Keep ``__init__`` free of I/O so that constructing and cloning a source stays
  cheap; fetch on ``load``.
* When the target kind is only known after loading, tag ``task=None`` and let
  the load-time validation do the work.
"""

from __future__ import annotations

from skcausal.datasets.real.supervised import (
    BaseSupervisedDataset,
    task_from_target_kind,
)

__all__ = ["MySupervisedDataset"]


class MySupervisedDataset(BaseSupervisedDataset):
    """Template for a supervised table released as ``(covariates, outcome)``.

    Rename this class and replace the TODO markers with source-specific logic.

    Parameters
    ----------
    target_kind : {"numeric", "categorical"}, default="numeric"
            Measurement kind of the target. Drop this parameter and hard-code the
            kind when your source only ever serves one.
    """

    def __init__(self, target_kind: str = "numeric"):
        self.target_kind = target_kind
        super().__init__()
        # A fixed-kind source can instead declare _tags = {"task": "regression"}.
        self.set_tags(task=task_from_target_kind(target_kind))

    def _fetch_supervised(self):
        """Return ``(features, target, target_kind)``.

        Returns
        -------
        features : DataFrame
                One row per sample, at least one column.
        target : Series or array-like
                One value per feature row, in the same row order. A named series
                names the released outcome column; anything else is named
                ``"target"``.
        target_kind : {"numeric", "categorical"}
                Measurement kind of the target.
        """

        # TODO: load the table here — read the file, call the vendor loader,
        # hit the API — and return it as-is. Do not validate or clean it; the
        # base class does that.
        raise NotImplementedError(
            "Replace the template _fetch_supervised implementation with "
            "source-specific logic."
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return lightweight constructor arguments for local checks.

        The object sweep instantiates these and loads them, so only return
        parameters that stay offline.
        """

        return [{"target_kind": "numeric"}, {"target_kind": "categorical"}]
