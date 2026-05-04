"""Extension template for average causal response estimators.

Purpose of this template
------------------------
Use this file as a starting point when adding a new estimator under
``skcausal.causal_estimators``.

How to use this template
------------------------
1. Copy the file to a module with a descriptive name.
2. Rename ``MyAverageCausalResponseEstimator``.
3. Update the module and class docstrings.
4. Set the estimator tags so they match the treatment types and backend you
   actually support.
5. Implement the mandatory private hooks ``_fit`` and ``_predict``.
6. Add ``get_test_params`` so the estimator can be instantiated by automated
   checks and local tests.

Repo-specific contract notes
----------------------------
* ``BaseAverageCausalResponseEstimator.fit`` converts ``X``, ``t``, and ``y``
  to the backend declared in ``_tags`` before calling ``_fit``.
* ``predict`` only accepts treatment values. Average-response estimators are
  expected to average over the covariate sample stored during ``fit``.
* Treatment schema checks happen in the base class, so ``_predict`` receives
  backend-native treatment rows that match the fit-time treatment metadata.
* If your estimator depends on scikit-learn regressors, categorical treatment
  encoding should usually live inside the supplied model or pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator

__all__ = ["MyAverageCausalResponseEstimator"]


class MyAverageCausalResponseEstimator(BaseAverageCausalResponseEstimator):
    """Template for average causal response estimators.

    Rename this class and replace the TODO markers with estimator-specific
    logic. The public ``fit`` and ``predict`` methods are inherited from
    ``BaseAverageCausalResponseEstimator``.

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
    }

    def __init__(self, some_hyperparameter: float = 1.0):
        self.some_hyperparameter = some_hyperparameter
        super().__init__()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame, y: pd.DataFrame):
        """Fit the estimator on backend-native inputs.

        Parameters
        ----------
        X : pd.DataFrame
                Covariate matrix already converted to the backend declared in
                ``_tags``.
        t : pd.DataFrame
                Treatment table already checked against the estimator's treatment
                capability tags.
        y : pd.DataFrame
                Outcome table already converted to the estimator backend.

        Returns
        -------
        self
                Fitted estimator.
        """

        # TODO: validate hyperparameters or training data here.
        # TODO: clone and fit any nuisance components here, and store fitted
        # objects in attributes ending with ``_``.
        # TODO: if your estimator needs transformed features, build them from
        # the backend-native ``X`` and ``t`` received here.
        raise NotImplementedError(
            "Replace the template _fit implementation with estimator-specific logic."
        )

    def _predict(self, t: pd.DataFrame) -> np.ndarray:
        """Return one average response per treatment row.

        Parameters
        ----------
        t : pd.DataFrame
                Treatment rows already converted to the estimator backend.

        Returns
        -------
        np.ndarray
                Array with one prediction per row in ``t``. Returning shape
                ``(n_t,)`` or ``(n_t, 1)`` is fine; the base class will coerce the
                output to a 2D column vector.
        """

        fit_X = self._get_fit_X()

        # TODO: use ``fit_X`` and ``t`` to compute one mean response per
        # requested treatment row. Most estimators repeat each treatment row
        # across the fitted covariate sample, evaluate an individual response,
        # and then average over rows.
        _ = fit_X
        raise NotImplementedError(
            "Replace the template _predict implementation with estimator-specific logic."
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Return lightweight constructor arguments for local checks.

        Replace this with parameters that instantiate a fast, representative
        version of your estimator.
        """

        return [
            {"some_hyperparameter": 1.0},
            {"some_hyperparameter": 2.0},
        ]
