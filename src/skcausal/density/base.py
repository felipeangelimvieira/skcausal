import numpy as np
import polars as pl
from skbase.base import BaseEstimator as _BaseEstimator

from skcausal.utils.polars import ALL_DTYPES, assert_schema_equal
from skcausal.utils.mtype import convert_mtype
from skbase.utils.dependencies import _check_soft_dependencies


class BaseDensityEstimator(_BaseEstimator):
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
        "t_inner_mtype": pl.DataFrame,
        "X_inner_mtype": pl.DataFrame,
        "supported_t_dtypes": ALL_DTYPES,
        "capability:multidimensional_treatment": False,
        "density_kind": "conditional",
        "soft_dependencies": [],
    }

    def __init__(self):
        self._t_schema = None
        self._t_preprocessed_schema = None

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

        self._t_schema = t.schema
        t = self._check_and_transform_treatment(t, is_fit=True)
        X = self._check_and_transform_X(X)
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
        t = self._check_and_transform_treatment(t, is_fit=False)
        X = self._check_and_transform_X(X)
        return self._predict_density(X=X, t=t)

    def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict the density of the treatment given the covariates.

        Abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _check_and_transform_treatment(self, t: pl.DataFrame, *, is_fit: bool = False):
        """
        Preprocess the treatment dataframe.

        This method is called during fit and predict times.
        Does the following checks and actions:

        * Check that treatment dtypes are supported.
        * Convert pl.Enum columns to 1-hot encoded columns if
            self.get_tag("preprocess_enum_columns", False) is True.
        * Asserts that the schema (after preprocessing)
            is the same as the first time that the method was called.
        * Convert to the inner mtype.
        """

        self._check_treatment_dimensionality(t)
        self._check_treatment_dtypes(t)

        if is_fit:
            self._t_preprocessed_schema = t.schema
        else:
            assert_schema_equal(t.schema, self._t_preprocessed_schema)

        inner_mtype = self.get_tag("t_inner_mtype", "np.ndarray")
        if inner_mtype == "np.ndarray":
            t = convert_mtype(t, np.ndarray)
        else:
            t = convert_mtype(t, inner_mtype)
        return t

    def _check_and_transform_X(self, X: pl.DataFrame):
        """
        Preprocess the covariate dataframe.

        Converts to the inner mtype specified by the tag "X_inner_mtype".
        """
        inner_mtype = self.get_tag("X_inner_mtype", "np.ndarray")
        if inner_mtype == "np.ndarray":
            X = convert_mtype(X, np.ndarray)
        else:
            X = convert_mtype(X, inner_mtype)
        return X

    def _check_treatment_dtypes(self, t: pl.DataFrame):
        """
        Check that treatment column dtypes are supported.

        Raises ValueError if any column has an unsupported dtype.
        """
        for col, dtype in zip(t.columns, t.dtypes):
            if dtype not in self.get_tag("supported_t_dtypes"):
                raise ValueError(
                    f"Column {col} has dtype {dtype}, which is not in the supported "
                    f"dtypes: {self.get_tag('supported_t_dtypes')}"
                )

    def _check_treatment_dimensionality(self, t: pl.DataFrame):
        """Check whether the estimator supports multidimensional treatment."""
        if self.get_tag("capability:multidimensional_treatment", False):
            return

        if t.shape[1] > 1:
            raise ValueError(
                f"{self.__class__.__name__} does not support multidimensional "
                f"treatment data, but received {t.shape[1]} treatment columns."
            )
