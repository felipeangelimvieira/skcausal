import copy

import numpy as np
import pandas as pd
import polars as pl

from skcausal.density.base import BaseDensityEstimator
from skcausal.utils.polars import INTEGER_DTYPES


__all__ = ["SklearnCategoricalDensity"]


class SklearnCategoricalDensity(BaseDensityEstimator):
    """
    Density estimator adapter for sklearn-style classifiers.

    Parameters
    ----------
    classifier : object
        A fitted-compatible classifier implementing ``fit(X, y)`` and
        ``predict_proba(X)``.
    """

    _tags = {
        "supported_t_dtypes": [
            pl.Boolean,
            pl.Enum,
            pl.Utf8,
            pl.String,
            pl.Categorical,
            *INTEGER_DTYPES,
        ],
        "density_kind": "conditional",
    }

    def __init__(self, classifier):
        self.classifier = classifier

        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        self.classifier_ = copy.deepcopy(self.classifier)
        self.classifier_.fit(self._to_pandas(X), self._series_to_labels(t.to_series(0)))
        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        labels = self._series_to_labels(t.to_series(0))

        probabilities = self._coerce_probability_output(
            self.classifier_.predict_proba(self._to_pandas(X)),
            classes=self.classifier_.classes_,
            n_rows=len(labels),
        )
        return self._select_observed_class_probability(
            self.classifier_.classes_, probabilities, labels
        )

    @staticmethod
    def _to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(frame.to_dict(as_series=False))

    @classmethod
    def _coerce_probability_output(
        cls,
        probabilities,
        *,
        classes,
        n_rows: int,
    ) -> np.ndarray:
        probability_array = np.asarray(probabilities, dtype=float)
        n_classes = len(classes)

        if probability_array.ndim == 1:
            if probability_array.shape[0] == n_rows and n_classes == 2:
                return cls._expand_binary_probability_output(probability_array)

            if probability_array.shape[0] == n_rows * n_classes:
                probability_array = probability_array.reshape(n_rows, n_classes)
            else:
                raise ValueError(
                    "Expected predict_proba output to be compatible with "
                    f"{n_rows} rows and {n_classes} classes, but received "
                    f"shape {probability_array.shape}."
                )
        elif probability_array.ndim == 2:
            if probability_array.shape[0] != n_rows:
                raise ValueError(
                    "Expected predict_proba output to contain "
                    f"{n_rows} rows, but received shape {probability_array.shape}."
                )

            if probability_array.shape[1] == 1 and n_classes == 2:
                return cls._expand_binary_probability_output(probability_array[:, 0])

            if probability_array.shape[1] != n_classes:
                raise ValueError(
                    "Expected predict_proba output to contain one probability "
                    "column per class, but received shape "
                    f"{probability_array.shape} for {n_classes} classes."
                )
        else:
            raise ValueError(
                "Expected predict_proba output to be 1D or 2D, but received array "
                f"with shape {probability_array.shape}."
            )

        return probability_array

    @staticmethod
    def _expand_binary_probability_output(positive_class_probability) -> np.ndarray:
        positive_class_probability = np.asarray(
            positive_class_probability,
            dtype=float,
        ).reshape(-1)
        return np.column_stack(
            [1.0 - positive_class_probability, positive_class_probability]
        )

    @classmethod
    def _select_observed_class_probability(cls, classes, probabilities, labels):
        class_to_index = {
            cls._normalize_scalar(label): idx for idx, label in enumerate(classes)
        }

        indices = []
        for label in labels:
            if label not in class_to_index:
                raise ValueError(
                    "Observed treatment label was not seen during classifier fit: "
                    f"{label!r}."
                )
            indices.append(class_to_index[label])

        row_index = np.arange(len(labels))
        return probabilities[row_index, np.asarray(indices)].reshape(-1, 1)

    @classmethod
    def _series_to_labels(cls, series: pl.Series) -> list:
        return [cls._normalize_scalar(value) for value in series.to_list()]

    @staticmethod
    def _normalize_scalar(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.linear_model import LogisticRegression

        return [{"classifier": LogisticRegression(max_iter=1000)}]
