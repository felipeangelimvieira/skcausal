from typing import Any
import numpy as np
import polars as pl
from skbase.base import BaseEstimator as _BaseEstimator

from skbase.utils.dependencies import _check_soft_dependencies
from skcausal.datatypes import (
    convert,
    collect_column_types,
)
from skcausal.base.mixin import TreatmentCheckMixin


class BaseDensityEstimator(TreatmentCheckMixin, _BaseEstimator):
    """
    Base class for density estimators.

    Tags
    ----
    t_inner_mtype : str
        The inner mtype for the treatment data. This is the mtype that the
        estimator's inner methods accept as input. If the input to the public
        methods is not in this mtype, it will be converted.
    X_inner_mtype : str
        The inner mtype for the covariate data. This is the mtype that the
        estimator's inner methods accept as input. If the input to the public
        methods is not in this mtype, it will be converted.
    supported_t_dtypes : list of dtypes
        The dtypes supported for the treatment data. The estimator should be able
        to handle treatment data with these dtypes without errors. If the input
        treatment data has a different dtype, an error may be raised.
    capability:multidimensional_treatment : bool
        Whether the estimator can handle multidimensional treatment data, i.e.,
        treatment data with more than one column. If False, ``fit`` rejects
        treatment data with more than one column.
    density_kind : {"conditional", "stabilized"}
        The kind of density-like quantity returned by the estimator.
        ``"conditional"`` corresponds to the conditional treatment density
        :math:`p(t \mid x)`, while ``"stabilized"`` corresponds to the
        stabilized ratio :math:`p(t \mid x) / p(t)`.

    """

    _tags = {
        "object_type": ["density_estimator"],
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": False,
        "density_kind": "conditional",
        "soft_dependencies": [],
    }

    def __init__(self):

        _check_soft_dependencies(*self.get_tag("soft_dependencies", []))
        super().__init__()

    def fit(self, X: pl.DataFrame, t: pl.DataFrame):
        """
        Fit the density estimator to the data.

        Parameters
        ----------
        X : pl.DataFrame,
            The covariate data.
        t : pl.DataFrame,
            The treatment data.
        """

        X, t = self._check_and_transform(X, t, is_fit=True)

        self._fit(X=X, t=t)
        return self

    def _fit(self, X: np.ndarray, t: np.ndarray):
        """
        Fit the density estimator to the data.

        Abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        """
        Predict the density of the treatment given the covariates.

        Parameters
        ----------
        X : pl.DataFrame,
            The covariate data.
        t : pl.DataFrame,
            The treatment data.

        Returns
        -------
        np.ndarray
            The predicted density values.
        """
        X, t = self._check_and_transform(X, t, is_fit=False)
        density = self._predict_density(X=X, t=t)
        return self._coerce_density_output(density)

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict the density of the treatment given the covariates.

        Abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    @staticmethod
    def _coerce_density_output(density) -> np.ndarray:
        """Return density outputs as a 2D float array."""
        density_array = np.asarray(density, dtype=float)

        if density_array.ndim == 1:
            density_array = density_array.reshape(-1, 1)

        if density_array.ndim != 2:
            raise ValueError(
                "Expected density output to be 1D or 2D, but received array with "
                f"shape {density_array.shape}."
            )

        return density_array

    def _check_and_transform_X(self, X: pl.DataFrame, is_fit=False):
        X = convert(X, self.get_tag("backend"))
        return X

    def _check_and_transform(self, X, t, is_fit=False):
        X = self._check_and_transform_X(X, is_fit=is_fit)
        t = self._check_and_transform_t(t, is_fit=is_fit)
        return X, t
