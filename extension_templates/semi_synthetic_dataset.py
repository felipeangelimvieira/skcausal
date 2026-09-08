"""Extension template for semi-synthetic datasets built on a supervised table.

Purpose of this template
------------------------
Use this file as a starting point when adding a dataset under
``skcausal.datasets.semisynthetic``: real covariates from a supervised source,
with a treatment and a response surface you induce from a model fitted on that
source. The covariates keep the correlation structure of real data while the
response stays known, so estimators can still be scored against truth.

If you generate the covariates yourself, use ``synthetic_dataset.py``. If the
treatment is already in the table, use ``observational_dataset.py``.

How to use this template
------------------------
1. Copy the file to a module with a descriptive name.
2. Rename ``MySemiSyntheticDataset``.
3. Update the module and class docstrings, and write down the data-generating
   process — a reader has to be able to tell what truth is being scored.
4. Set ``column_types`` and the ``task``/``source_task`` tags.
5. Implement ``_prepare_target``, ``_get_treatments``, and ``_predict_y``.
6. Expose a treatment grid: ``get_grid`` for a continuous treatment, or a
   helper such as ``get_levels`` for a categorical one.
7. Add ``get_test_params`` so the dataset can be instantiated in local tests.

Repo-specific contract notes
----------------------------
* The source is any ``BaseSupervisedDataset``; the base class refuses one whose
  ``task`` tag disagrees with your ``source_task`` tag, and refuses a causal
  dataset outright.
* These datasets subclass ``BaseKnownResponseDataset``, not
  ``BaseSyntheticDataset``: the true response surface is known, so they are
  benchmarkable, but the sample size comes from the source and is not a
  constructor argument.
* ``_prepare`` runs the whole DGP once, so call it at the end of ``__init__``,
  after every attribute it reads has been assigned. It takes no ``n``.
* ``_get_covariates`` is already implemented: it loads the source, standardizes
  the covariate matrix, and sets ``self.n``, ``self._n_features``, and
  ``self.covariate_columns_``. Your hooks receive that standardized matrix.
* ``self.outcome_noise_scale`` must exist before ``_prepare`` runs: the
  inherited noise model reads it.
* Draw every random effect from ``self._rng``, so that ``random_state``
  reproduces the dataset exactly.
* Fit wrapped scikit-learn estimators through ``self._clone_estimator``, which
  clones them and seeds an unset ``random_state``.
* ``_predict_y`` must accept counterfactual treatments, not only the realized
  ones, since ``predict_curve`` evaluates it over a grid.
* The module that defines ``BaseSemiSyntheticDataset`` already provides ``_as_matrix``, ``_as_1d``,
  ``_standardize``, ``_fit_random_spline``/``_evaluate_random_spline`` (a random
  centered curve over a continuous variable) and ``_random_level_effects`` (its
  categorical counterpart). Reuse them rather than re-rolling an effect.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from skcausal.datasets.semisynthetic._model_induced_base import (
    BaseSemiSyntheticDataset,
    _as_1d,
    _as_matrix,
    _standardize,
)

__all__ = ["MySemiSyntheticDataset"]


class MySemiSyntheticDataset(BaseSemiSyntheticDataset):
    """Template for a model-induced semi-synthetic dataset.

    Rename this class and replace the TODO markers with dataset-specific logic.

    Parameters
    ----------
    model : sklearn estimator
            Model whose fit on the source problem induces the treatment.
    dataset : BaseSupervisedDataset
            Source of the supervised sample, for example ``SklearnDataset``,
            ``SupervisedFrame`` or ``OpenMLDataset``.
    random_state : int, default=42
            Seed for the derived data-generating process.
    treatment_effect_scale : float, default=1.0
            Size of the induced treatment effect.
    outcome_noise_scale : float, default=1.0
            Standard deviation of the additive outcome noise.
    """

    # "task" is the outcome this dataset poses, "source_task" the one it needs
    # from its source. They differ whenever a classification source is used to
    # build a continuous outcome.
    _tags = {"task": "regression", "source_task": "regression"}
    column_types = {"t": "continuous"}

    def __init__(
        self,
        model,
        dataset,
        random_state: int = 42,
        treatment_effect_scale: float = 1.0,
        outcome_noise_scale: float = 1.0,
    ):
        self.model = model
        self.treatment_effect_scale = treatment_effect_scale
        self.outcome_noise_scale = outcome_noise_scale

        super().__init__(dataset=dataset, random_state=random_state)
        self._prepare()

    def _prepare_target(self, outcome_frame) -> np.ndarray:
        """Return the source outcome as a 1-D array, and store what you need.

        Parameters
        ----------
        outcome_frame : pl.DataFrame
                The source's single outcome column.

        Returns
        -------
        np.ndarray
                The raw target, used only to check that the source released one
                value per covariate row.
        """

        target = _as_1d(
            outcome_frame, name=f"{type(self).__name__} target", dtype=float
        )
        # TODO: store the form your DGP actually consumes — a standardized
        # regression target, a label vector, a class-probability matrix.
        self.normalized_target_ = _standardize(target)
        return target

    def _get_treatments(self, covariates: np.ndarray) -> np.ndarray:
        """Return one treatment row per covariate row.

        Parameters
        ----------
        covariates : np.ndarray
                Standardized covariate matrix from the source.

        Returns
        -------
        np.ndarray
                Treatment assignments, shaped ``(n_samples, 1)`` for a
                single-column treatment.
        """

        covariate_array = _as_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )

        # TODO: fit the model on the source problem and turn its output into a
        # treatment. Confounding is the point: the treatment has to depend on
        # the same covariates the outcome does.
        # TODO: draw the random treatment effect here, once, while the realized
        # treatments are in hand, and store it in attributes ending with ``_``
        # so that _predict_y can evaluate it later.
        _ = covariate_array
        raise NotImplementedError(
            "Replace the template _get_treatments implementation with "
            "dataset-specific logic."
        )

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        """Return the noiseless structural response ``E[Y | X, T]``.

        Parameters
        ----------
        covariates : np.ndarray or pl.DataFrame
                Covariate rows.
        treatments : np.ndarray or pl.DataFrame
                Treatment rows, possibly counterfactual: ``predict_curve`` calls
                this with grid values that were never realized.

        Returns
        -------
        np.ndarray
                One mean outcome per row, shaped ``(n_samples, 1)``.
        """

        # TODO: combine the model's baseline prediction at ``covariates`` with
        # the stored treatment effect evaluated at ``treatments``.
        raise NotImplementedError(
            "Replace the template _predict_y implementation with dataset-specific "
            "logic."
        )

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        """Return a treatment grid spanning the realized treatment range.

        Swap this for a helper such as ``get_levels`` when the induced treatment
        is categorical.
        """

        lower, upper = self.treatment_range_
        grid = (
            np.repeat(lower, n)
            if np.isclose(lower, upper)
            else np.linspace(lower, upper, n)
        )
        return self._coerce_treatment_frame(pl.DataFrame({"t": grid}))

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return lightweight constructor arguments for local checks.

        Name an offline source: the object sweep instantiates and loads these,
        and must not reach the network.
        """

        from skcausal.datasets.real.sklearn import SklearnDataset

        return [
            {
                "model": LinearRegression(),
                "dataset": SklearnDataset(name="diabetes"),
                "random_state": 0,
            },
        ]
