"""Extension template for density estimators.

Purpose of this template
------------------------
Use this file as a starting point when adding a new estimator under
``skcausal.density``.

How to use this template
------------------------
1. Copy the file to a module with a descriptive name.
2. Rename ``MyDensityEstimator``.
3. Update the module and class docstrings.
4. Set the estimator tags so they match the treatment types, backend, and
   density quantity you actually support.
5. Implement the mandatory private hooks ``_fit`` and ``_predict_density``.
6. Add ``get_test_params`` so the estimator can be instantiated by automated
   checks and local tests.

Repo-specific contract notes
----------------------------
* ``BaseDensityEstimator.fit`` converts ``X`` and ``t`` to the backend declared
  in ``_tags`` before calling ``_fit``.
* ``predict_density`` must return an out-of-sample score for every input row.
* The estimator may return either the conditional density ``p(t | x)`` or a
  stabilized ratio ``p(t | x) / p(t)``; declare which one you return through
  the ``density_kind`` tag.
* ``_predict_density`` may return shape ``(n_samples,)`` or ``(n_samples, 1)``;
  the base class coerces both to a 2D float array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skcausal.density.base import BaseDensityEstimator

__all__ = ["MyDensityEstimator"]


class MyDensityEstimator(BaseDensityEstimator):
    """Template for conditional or stabilized treatment density estimators.

    Rename this class and replace the TODO markers with estimator-specific
    logic.

    Parameters
    ----------
    some_hyperparameter : float, default=1.0
            Example hyperparameter showing how constructor arguments are exposed to
            ``get_params``/``set_params``.
    """

    _tags = {
        "backend": "pandas",
        "capability:t_type": ["continuous"],
        "capability:multidimensional_treatment": False,
        "density_kind": "conditional",
        "soft_dependencies": [],
    }

    def __init__(self, some_hyperparameter: float = 1.0):
        self.some_hyperparameter = some_hyperparameter
        super().__init__()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        """Fit the estimator on backend-native covariates and treatments.

        Parameters
        ----------
        X : pd.DataFrame
                Covariate matrix already converted to the backend declared in
                ``_tags``.
        t : pd.DataFrame
                Treatment table already checked against the estimator's treatment
                capability tags.

        Returns
        -------
        self
                Fitted estimator.
        """

        # TODO: validate hyperparameters or preprocess X/t here.
        # TODO: clone and fit any wrapped models here, and store fitted objects
        # in attributes ending with ``_``.
        raise NotImplementedError(
            "Replace the template _fit implementation with estimator-specific logic."
        )

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        """Return one density-like score per row.

        Parameters
        ----------
        X : pd.DataFrame
                Covariate matrix already converted to the estimator backend.
        t : pd.DataFrame
                Treatment rows already converted to the estimator backend.

        Returns
        -------
        np.ndarray
                One density or stabilized-ratio value per input row.
        """

        # TODO: compute one out-of-sample score for each row in ``X`` and ``t``.
        raise NotImplementedError(
            "Replace the template _predict_density implementation with estimator-specific logic."
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return lightweight constructor arguments for local checks."""

        return [
            {"some_hyperparameter": 1.0},
            {"some_hyperparameter": 2.0},
        ]
