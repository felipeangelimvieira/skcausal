import numpy as np
import polars as pl
from skbase.base import BaseEstimator as _BaseEstimator

from skcausal.utils.polars import ALL_DTYPES, assert_schema_equal, to_dummies
from skcausal.utils.mtype import convert_mtype


class BaseBalancingWeightRegressor(_BaseEstimator):
    """
    Base class for balancing weight regressors.

    Balancing weight regressors are estimators that receive the covariates and
    the treatment as input, and return a weight for each sample. The weight is equal
    to the balancing factor, i.e., some value of the form Q(T) / P(T|X), where
    Q and P are probability measures.

    Main methods
    ------------
    fit(X, t)
        Fit the estimator to the data.

    predict_sample_weight(X, t)
        Predict the sample weights for the data.

    Additional tags
    ---------------
    utility:scale_treatment : {"minmax", None}
        Optional preprocessing applied to continuous treatment columns before
        delegating to inner ``fit``/``predict`` implementations.

    """

    _tags = {
        "t_inner_mtype": pl.DataFrame,
        "X_inner_mtype": pl.DataFrame,
        "one_hot_encode_enum_columns": False,
        "capability:supports_multidimensional_treatment": False,
        "supported_t_dtypes": ALL_DTYPES,
        "balancing_weight_type": None,
        "utility:scale_treatment": None,
    }

    def __init__(self):
        self._t_schema = None
        self._t_preprocessed_schema = None
        self._t_scaling_params = {}
        super().__init__()

    def fit(self, X: pl.DataFrame, t: pl.DataFrame):
        """
        Fit the estimator to the data.

        Abstract method that must be implemented by subclasses.
        """

        self._t_schema = t.schema
        self._t_preprocessed_schema = None
        self._t_scaling_params = {}

        t = self._check_and_transform_treatment(t, is_fit=True)
        X = self._check_and_transform_X(X)

        self._fit(X=X, t=t)

    def predict_sample_weight(self, X: pl.DataFrame, t: pl.DataFrame):
        """
        Predict the probability of each class for each sample.
        :param X: Feature data.
        """
        t = self._check_and_transform_treatment(t, is_fit=False)
        X = self._check_and_transform_X(X)
        return self._predict_sample_weight(X=X, t=t)

    def _fit(self, X: np.ndarray, t: np.ndarray):
        """
        Fit the estimator to the data.

        Abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _predict_sample_weight(self, X: np.ndarray, t: np.ndarray):
        """
        Predict the probability of each class for each sample.
        :param X: Feature data.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _check_and_transform_treatment(self, t: pl.DataFrame, *, is_fit: bool = False):
        """
        Preprocess the treatment dataframe.

        This method is called during fit and predict times.
        Does the following checks and actions:

        * Convert pl.Enum columns to 1-hot encoded columns if
            self.get_tag("one_hot_encode_enum_columns", False) is True.
        * Asserts that the schema (after preprocessing)
            is the same as the first time that the method was called.


        """
        # Reorder t columns according to self._t_schema
        self._check_treatment_dtypes(t)

        if self.get_tag("one_hot_encode_enum_columns", False):
            for col, dtype in zip(t.columns, t.dtypes):
                if dtype == pl.Enum:
                    t = to_dummies(t, col)

        t = self._apply_treatment_scaling(t, is_fit=is_fit)

        if self._t_preprocessed_schema is None:
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
        inner_mtype = self.get_tag("X_inner_mtype", "np.ndarray")
        if inner_mtype == "np.ndarray":
            X = convert_mtype(X, np.ndarray)
        else:
            X = convert_mtype(X, inner_mtype)
        return X

    def _check_treatment_dtypes(self, t: pl.DataFrame):
        for col, dtype in zip(t.columns, t.dtypes):
            if dtype not in self.get_tag("supported_t_dtypes"):
                raise ValueError(
                    f"Column {col} has dtype {dtype}, which is not in the supported"
                    f"dtypes: {self.get_tag('supported_t_dtypes')}"
                )

    def _transform_enum_columns(self, t: pl.DataFrame):
        for col, dtype in zip(t.columns, t.dtypes):
            if dtype == pl.Enum:
                t = to_dummies(t, col)
        return t

    def _apply_treatment_scaling(self, t: pl.DataFrame, *, is_fit: bool):
        scale_strategy = self.get_tag("utility:scale_treatment", None)
        if scale_strategy is None:
            return t

        if scale_strategy != "minmax":
            raise ValueError(
                "Unsupported value for tag 'utility:scale_treatment': "
                f"{scale_strategy}. Supported values are None and 'minmax'."
            )

        if is_fit:
            float_columns = [
                col
                for col, dtype in zip(t.columns, t.dtypes)
                if dtype in (pl.Float32, pl.Float64)
            ]

            if not float_columns:
                return t

            # Collect scaling parameters using the training data only.
            for col in float_columns:
                col_min = t[col].min()
                col_max = t[col].max()
                self._t_scaling_params[col] = (col_min, col_max)
        else:
            float_columns = [col for col in self._t_scaling_params if col in t.columns]

            if not float_columns:
                return t

            new_float_columns = [
                col
                for col, dtype in zip(t.columns, t.dtypes)
                if dtype in (pl.Float32, pl.Float64)
                and col not in self._t_scaling_params
            ]
            if new_float_columns:
                raise ValueError(
                    "Encountered float treatment columns without fitted scaling"
                    " parameters: "
                    + ", ".join(new_float_columns)
                    + ". Make sure to call fit before predict when using treatment"
                    " scaling, and avoid changing treatment dtypes between fit and"
                    " predict."
                )

            missing_params = [
                col for col in float_columns if col not in self._t_scaling_params
            ]
            if missing_params:
                raise ValueError(
                    "Missing scaling parameters for columns: "
                    + ", ".join(missing_params)
                    + ". Make sure to call fit before predict when using treatment scaling."
                )

        scaled_columns = []
        for col in float_columns:
            col_min, col_max = self._t_scaling_params[col]
            denom = (
                col_max - col_min
                if col_max is not None and col_min is not None
                else None
            )
            if denom is None or denom == 0:
                scaled_columns.append(pl.lit(0.0).alias(col))
            else:
                scaled_columns.append(((pl.col(col) - col_min) / denom).alias(col))

        if scaled_columns:
            t = t.with_columns(scaled_columns)

        return t
