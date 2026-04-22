import copy

import numpy as np
import pandas as pd

from skcausal.density.base import BaseDensityEstimator

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
        "backend": "pandas",
        "capability:t_type": ["categorical"],
        "density_kind": "conditional",
        "soft_dependencies": ["sklearn"],
    }

    def __init__(self, classifier):
        self.classifier = classifier

        super().__init__()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        """Fit the wrapped classifier on the single casted treatment column."""
        self.classifier_ = copy.deepcopy(self.classifier)
        self.classifier_.fit(X, self._series_to_labels(self._first_series(t)))
        return self

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        """Return the fitted probability assigned to each observed treatment label."""
        labels = self._series_to_labels(self._first_series(t))

        probabilities = self._coerce_probability_output(
            self.classifier_.predict_proba(X),
            classes=self.classifier_.classes_,
            n_rows=len(labels),
        )
        return self._select_observed_class_probability(
            self.classifier_.classes_, probabilities, labels
        )

    @staticmethod
    def _first_series(frame: pd.DataFrame) -> pd.Series:
        """Extract the single treatment column after base-class casting to pandas."""
        return frame.iloc[:, 0]

    @classmethod
    def _coerce_probability_output(
        cls,
        probabilities,
        *,
        classes,
        n_rows: int,
    ) -> np.ndarray:
        """Normalize classifier probability outputs to a dense 2D array.

        sklearn-compatible classifiers are expected to return one probability
        column per class, but some binary wrappers emit only the positive-class
        probability or flatten the output. This helper reshapes those variants
        into a consistent ``(n_rows, n_classes)`` array before the observed
        treatment column is selected.
        """
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
        """Expand binary positive-class probabilities into two class columns."""
        positive_class_probability = np.asarray(
            positive_class_probability,
            dtype=float,
        ).reshape(-1)
        return np.column_stack(
            [1.0 - positive_class_probability, positive_class_probability]
        )

    @classmethod
    def _select_observed_class_probability(cls, classes, probabilities, labels):
        """Pick the probability associated with each observed treatment label.

        ``classifier.classes_`` and the labels extracted from ``t`` may contain
        scalar wrapper types such as numpy scalars or pandas categorical values.
        Both sides are normalized first so that semantically equivalent labels
        match reliably during lookup.
        """
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
    def _series_to_labels(cls, series: pd.Series) -> list:
        """Convert a pandas Series into normalized scalar labels.

        This keeps the labels passed to ``classifier.fit`` and the labels used
        later for probability lookup in the same canonical Python form.
        Without this normalization, comparisons against ``classifier.classes_``
        can fail due to numpy, pandas, or categorical scalar wrappers.
        """
        return [cls._normalize_scalar(value) for value in series.to_list()]

    @staticmethod
    def _normalize_scalar(value):
        """Unwrap scalar wrapper types while leaving plain Python values alone."""
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
