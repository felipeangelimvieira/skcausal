from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skcausal.datasets.base import BaseSyntheticDataset
from skcausal.datasets.semi_synthetic_regressor import (
    _as_feature_matrix,
    _standardize_matrix,
)

__all__ = ["SemiSyntheticClassifier"]


def _default_test_dataset() -> tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=2000,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=17,
    )


def _as_label_array(values, *, name: str):
    if isinstance(values, pl.DataFrame):
        if values.width != 1:
            raise ValueError(f"{name} must have exactly one column.")
        values = values.to_series(0).to_numpy()
    elif isinstance(values, pl.Series):
        values = values.to_numpy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"{name} must have exactly one column.")
        values = values.to_numpy()
    elif isinstance(values, pd.Series):
        values = values.to_numpy()

    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _as_string_labels(values) -> np.ndarray:
    return np.asarray(values, dtype=object).astype(str)


def _as_treatment_labels(values, *, levels: list[str] | None, name: str) -> np.ndarray:
    labels = _as_string_labels(_as_label_array(values, name=name))
    if levels is None:
        return labels

    unknown = sorted(set(labels) - set(levels))
    if unknown:
        supported = ", ".join(levels)
        unexpected = ", ".join(unknown)
        raise ValueError(
            f"{name} received unknown treatment levels: {unexpected}. "
            f"Supported levels are: {supported}."
        )

    return labels


class SemiSyntheticClassifier(BaseSyntheticDataset):
    r"""Semi-synthetic categorical-treatment dataset from a classification task.

    The dataset starts from a supervised classification sample ``(X, y)``
    returned by ``_load_dataset`` or by the optional ``load_dataset`` callable.
    The covariates are standardized, a clone of the supplied scikit-learn
    classifier is fit on the normalized features, and its fitted class
    predictions become the observed treatment labels. The structural response
    combines the fitted class probability for a requested treatment level with a
    random centered treatment-specific shift.

    If the raw classification sample is
    :math:`(X_i^{\mathrm{raw}}, y_i^{\mathrm{raw}})` for
    :math:`i = 1, \ldots, n`, the dataset first standardizes each feature
    column:

    .. math::

        X_{ij} =
        \frac{X_{ij}^{\mathrm{raw}} - \mu_j}{s_j},

    where zero empirical standard deviations are replaced by 1 in the code.

    A cloned classifier is then fit on the normalized covariates and its fitted
    predictions define the realized treatment labels:

    .. math::

        \hat{c} = \operatorname{fit}(\text{classifier}, X, y),
        \qquad
        A_i = \hat{c}(X_i).

    Let :math:`\hat{p}_a(x)` denote the fitted class probability assigned by
    ``predict_proba`` to treatment level :math:`a`. For the distinct realized
    treatment levels :math:`a_1, \ldots, a_K`, the implementation samples
    random offsets

    .. math::

        \beta_k \stackrel{\mathrm{iid}}{\sim} \mathcal{N}\!\left(0, \frac{1}{K}\right),
        \qquad k = 1, \ldots, K,

    and centers them over the observed treatment sample:

    .. math::

        g(a) = \lambda
        \left(
            \beta(a)
            - \frac{1}{n} \sum_{i=1}^n \beta(A_i)
        \right),

    where :math:`\lambda =` ``treatment_effect_scale``.

    The noiseless response surface exposed by :meth:`predict_y` is therefore

    .. math::

        \mu(x, a) = \hat{p}_a(x) + g(a).

    The observed outcomes returned by :meth:`load` are sampled at the realized
    treatments with additive Gaussian noise:

    .. math::

        Y_i = \mu(X_i, A_i) + \varepsilon_i,
        \qquad
        \varepsilon_i \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2),

    where :math:`\sigma =` ``outcome_noise_scale``. The supplied classifier
    must implement both ``predict`` and ``predict_proba``.
    """

    TREATMENT_SCHEMA = pl.Schema({"t": pl.Utf8})

    def __init__(
        self,
        classifier,
        load_dataset: Callable[[], tuple[object, object]] | None = None,
        random_state: int = 42,
        treatment_effect_scale: float = 1.0,
        outcome_noise_scale: float = 1.0,
    ):
        self.classifier = classifier
        self.load_dataset = load_dataset
        self.treatment_effect_scale = treatment_effect_scale
        self.outcome_noise_scale = outcome_noise_scale

        super().__init__(n=0, random_state=random_state)

        self.covariate_columns_ = None
        self.classification_target_ = None
        self.classifier_ = None
        self.treatment_levels_ = None
        self.treatment_effects_ = None
        self._class_index_ = None
        self._n_features = None

        self._prepare()

    def _load_dataset(self):
        if self.load_dataset is None:
            raise NotImplementedError(
                "SemiSyntheticClassifier requires a load_dataset callable or a "
                "subclass override of _load_dataset()."
            )
        return self.load_dataset()

    def _get_covariates(self) -> np.ndarray:
        loaded = self._load_dataset()
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("_load_dataset must return exactly (X, y).")

        covariates, target = loaded
        covariate_array, columns = _as_feature_matrix(
            covariates,
            expected_width=None,
            name=f"{type(self).__name__} covariates",
        )
        target_array = _as_label_array(target, name=f"{type(self).__name__} target")

        if covariate_array.shape[0] != target_array.shape[0]:
            raise ValueError(
                "_load_dataset must return covariates and targets with the same "
                "number of rows."
            )

        normalized_covariates, _, _ = _standardize_matrix(covariate_array)

        self.n = covariate_array.shape[0]
        self._n_features = covariate_array.shape[1]
        self.covariate_columns_ = columns or [
            f"x{i}" for i in range(covariate_array.shape[1])
        ]
        self.classification_target_ = target_array

        return normalized_covariates

    def _clone_classifier(self):
        estimator = clone(self.classifier)
        params = estimator.get_params(deep=False)
        if "random_state" in params and params["random_state"] is None:
            estimator.set_params(random_state=self.random_state)
        return estimator

    def _predict_proba_matrix(self, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self.classifier_, "predict_proba"):
            raise ValueError(
                "SemiSyntheticClassifier requires a classifier implementing "
                "predict_proba()."
            )

        probabilities = np.asarray(
            self.classifier_.predict_proba(covariates), dtype=float
        )
        if probabilities.ndim == 1:
            if len(self.treatment_levels_) != 2:
                raise ValueError(
                    "predict_proba must return one probability per class for "
                    "multiclass problems."
                )
            positive_probability = probabilities.reshape(-1)
            probabilities = np.column_stack(
                [1.0 - positive_probability, positive_probability]
            )
        elif probabilities.ndim == 2 and probabilities.shape[1] == 1:
            if len(self.treatment_levels_) != 2:
                raise ValueError(
                    "predict_proba must return one probability per class for "
                    "multiclass problems."
                )
            positive_probability = probabilities[:, 0]
            probabilities = np.column_stack(
                [1.0 - positive_probability, positive_probability]
            )

        if probabilities.ndim != 2 or probabilities.shape[0] != covariates.shape[0]:
            raise ValueError(
                "predict_proba must return an array with shape "
                "(n_samples, n_classes)."
            )

        if probabilities.shape[1] != len(self.treatment_levels_):
            raise ValueError(
                "predict_proba returned a different number of classes than those "
                "observed during fit."
            )

        return probabilities

    def _fit_treatment_effect(self, treatments) -> None:
        labels = _as_treatment_labels(
            treatments,
            levels=self.treatment_levels_,
            name=f"{type(self).__name__} treatments",
        )
        n_levels = len(self.treatment_levels_)
        raw_effects = self._rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, n_levels)),
            size=n_levels,
        )
        raw_effect_lookup = {
            label: effect for label, effect in zip(self.treatment_levels_, raw_effects)
        }
        observed_effects = np.array(
            [raw_effect_lookup[label] for label in labels],
            dtype=float,
        )
        offset = float(observed_effects.mean()) if observed_effects.size else 0.0
        self.treatment_effects_ = {
            label: float(self.treatment_effect_scale * (effect - offset))
            for label, effect in raw_effect_lookup.items()
        }

    def _evaluate_treatment_effect(self, treatments) -> np.ndarray:
        labels = _as_treatment_labels(
            treatments,
            levels=self.treatment_levels_,
            name=f"{type(self).__name__} treatments",
        )
        return np.array(
            [self.treatment_effects_[label] for label in labels], dtype=float
        )

    def _observed_class_probability(
        self, covariates: np.ndarray, treatments
    ) -> np.ndarray:
        labels = _as_treatment_labels(
            treatments,
            levels=self.treatment_levels_,
            name=f"{type(self).__name__} treatments",
        )
        probabilities = self._predict_proba_matrix(covariates)
        class_indices = np.array(
            [self._class_index_[label] for label in labels], dtype=int
        )
        return probabilities[np.arange(covariates.shape[0]), class_indices]

    def _get_treatments(self, covariates) -> np.ndarray:
        covariate_array, _ = _as_feature_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )
        if self.classification_target_ is None:
            raise RuntimeError("Covariates must be prepared before treatments.")

        self.classifier_ = self._clone_classifier()
        self.classifier_.fit(covariate_array, self.classification_target_)

        if not hasattr(self.classifier_, "classes_"):
            raise ValueError(
                "Classifier must expose classes_ after fitting "
                "SemiSyntheticClassifier."
            )

        self.treatment_levels_ = _as_string_labels(self.classifier_.classes_).tolist()
        self._class_index_ = {
            label: index for index, label in enumerate(self.treatment_levels_)
        }

        treatments = _as_string_labels(self.classifier_.predict(covariate_array))
        self._fit_treatment_effect(treatments)
        return treatments.reshape(-1, 1)

    def _inject_outcome_noise(self, expected_outcomes, covariates, treatments):
        expected = np.asarray(expected_outcomes, dtype=float).reshape(-1, 1)

        return expected + self._rng.normal(
            loc=0.0,
            scale=self.outcome_noise_scale,
            size=expected.shape[0],
        ).reshape(-1, 1)

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_array, _ = _as_feature_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )
        treatment_labels = _as_treatment_labels(
            treatments,
            levels=self.treatment_levels_,
            name=f"{type(self).__name__} treatments",
        )

        baseline = self._observed_class_probability(covariate_array, treatment_labels)
        effect = self._evaluate_treatment_effect(treatment_labels)
        return (baseline + effect).reshape(-1, 1)

    def _covariate_frame(self, covariates: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                column: covariates[:, idx]
                for idx, column in enumerate(self.covariate_columns_)
            }
        )

    def _treatment_frame(self, treatments: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {"t": _as_string_labels(treatments).reshape(-1)},
            schema=self.TREATMENT_SCHEMA,
        ).with_columns(pl.col("t").cast(pl.Categorical))

    def _outcome_frame(self, outcomes: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"y": np.asarray(outcomes, dtype=float).reshape(-1)})

    def _prepare(self, n: int = None):
        if n is not None:
            raise ValueError(
                "SemiSyntheticClassifier derives its sample size from the loaded "
                "dataset and does not support prepare(n=...)."
            )

        covariates = self._get_covariates()
        treatments = self._get_treatments(covariates)
        outcomes = self._get_outcomes(covariates, treatments)

        self._covariates = self._covariate_frame(covariates)
        self._treatments = self._treatment_frame(treatments)
        self._outcomes = self._outcome_frame(outcomes)
        self.n = self._covariates.height

        return self

    def get_levels(self) -> pl.DataFrame:
        if self.treatment_levels_ is None:
            raise RuntimeError(
                "The dataset must be prepared before requesting treatment levels."
            )

        return pl.DataFrame(
            {"t": self.treatment_levels_},
            schema=self.TREATMENT_SCHEMA,
        ).with_columns(pl.col("t").cast(pl.Categorical))

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        from functools import partial
        from sklearn.datasets import load_wine

        _load = partial(load_wine, return_X_y=True, as_frame=True)

        return [
            {
                "classifier": LogisticRegression(max_iter=2000),
                "load_dataset": _load,
                "treatment_effect_scale": 2.0,
                "random_state": 0,
            },
            {
                "classifier": LogisticRegression(max_iter=2000),
                "load_dataset": _default_test_dataset,
                "random_state": 1,
                "treatment_effect_scale": 0.5,
            },
        ]
