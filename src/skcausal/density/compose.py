import copy

import numpy as np
import polars as pl

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.sklearn import SklearnCategoricalDensity
from skcausal.density.skpro import SkproDensityEstimator
from skcausal.utils.polars import ALL_DTYPES, FLOAT_DTYPES


__all__ = ["CompositeFactorizedDensityEstimator"]


class CompositeFactorizedDensityEstimator(BaseDensityEstimator):
    """Compose independent per-dimension density estimators.

    Float treatment columns are modeled with a skpro probabilistic regressor,
    wrapped through ``SkproDensityEstimator``. Discrete treatment columns are
    modeled with a sklearn-style classifier wrapped through
    ``SklearnCategoricalDensity``.

    The joint density is computed under an independence assumption across
    treatment dimensions, i.e. as the product of per-column densities or class
    probabilities.

    Parameters
    ----------
    continuous_estimator : object
            skpro probabilistic regressor used for float treatment columns.
    classifier : object
            sklearn-style classifier used for discrete treatment columns.
    autoregressive : bool, default=False
            If True, columns are sorted by (task, name) — continuous first,
            then discrete — and each fitted estimator's predictions are
            appended to the feature matrix for the next estimator.  This
            relaxes the independence assumption by decomposing the joint
            density autoregressively, letting each model condition on all
            previously modeled treatments.
    """

    _tags = {
        "supported_t_dtypes": ALL_DTYPES,
        "capability:multidimensional_treatment": True,
        "density_kind": "conditional",
        "soft_dependencies": ["skpro"],
    }

    def __init__(self, continuous_estimator, classifier, autoregressive=False):
        self.continuous_estimator = continuous_estimator
        self.classifier = classifier
        self.autoregressive = autoregressive

        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        self.column_estimators_ = {}
        self.column_kinds_ = {}
        self.ordered_columns_ = self._order_columns(t)

        X_running = X

        for column in self.ordered_columns_:
            dtype = t[column].dtype
            if dtype in FLOAT_DTYPES:
                estimator = SkproDensityEstimator(
                    copy.deepcopy(self.continuous_estimator)
                )
                self.column_kinds_[column] = "continuous"
            else:
                estimator = SklearnCategoricalDensity(copy.deepcopy(self.classifier))
                self.column_kinds_[column] = "categorical"

            estimator.fit(X_running, t.select(column))
            self.column_estimators_[column] = estimator

            if self.autoregressive:
                X_running = self._append_predictions(X_running, column)

        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        joint_density = np.ones((len(X), 1), dtype=float)
        X_running = X

        for column in self.ordered_columns_:
            estimator = self.column_estimators_[column]
            column_frame = t.select(column)

            column_density = estimator.predict_density(X_running, column_frame)

            joint_density *= column_density

            if self.autoregressive:
                X_running = self._append_predictions(X_running, column)

        return joint_density

    def _append_predictions(self, X_running: pl.DataFrame, column: str) -> pl.DataFrame:
        """Append fitted estimator predictions as new columns."""
        estimator = self.column_estimators_[column]
        X_pd = SklearnCategoricalDensity._to_pandas(X_running)

        if self.column_kinds_[column] == "continuous":
            preds = estimator.estimator_.predict(X_pd)
            if hasattr(preds, "to_numpy"):
                preds = preds.to_numpy()
            preds = np.asarray(preds).ravel()
            return X_running.with_columns(pl.Series(f"_pred_{column}", preds))

        probas = estimator.classifier_.predict_proba(X_pd)
        new_cols = [
            pl.Series(f"_proba_{column}_{i}", probas[:, i])
            for i in range(probas.shape[1])
        ]
        return X_running.with_columns(new_cols)

    @staticmethod
    def _order_columns(t: pl.DataFrame) -> list:
        """Return columns sorted by (task, name): continuous first, then discrete."""
        continuous = sorted(c for c, d in zip(t.columns, t.dtypes) if d in FLOAT_DTYPES)
        discrete = sorted(
            c for c, d in zip(t.columns, t.dtypes) if d not in FLOAT_DTYPES
        )
        return continuous + discrete

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from skpro.regression.residual import ResidualDouble

        return [
            {
                "continuous_estimator": ResidualDouble(
                    estimator=LinearRegression(),
                    estimator_resid=RandomForestRegressor(
                        n_estimators=10, random_state=0
                    ),
                ),
                "classifier": LogisticRegression(max_iter=1000),
            }
        ]
