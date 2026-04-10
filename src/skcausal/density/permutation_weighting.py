import copy

import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.preprocessing import SplineTransformer

from skcausal.density.base import BaseDensityEstimator
from skcausal.utils.polars import ALL_DTYPES, convert_categorical_to_dummies
from skcausal.utils.sklearn import _resolve_sample_weight_fit_args

__all__ = ["PermutationWeighting"]


class PermutationWeighting(BaseDensityEstimator):
    """Estimate a stabilized treatment density ratio via permutation weighting.

    The estimator constructs a synthetic sample by pairing observed covariates
    with independently permuted treatments, so the synthetic distribution
    approximates ``p(x) p(t)`` while the observed sample follows ``p(x, t)``.
    A classifier trained to discriminate observed from permuted pairs yields
    posterior odds proportional to the stabilized density ratio
    ``p(t | x) / p(t)``.

    Parameters
    ----------
    classifier : ClassifierMixin
        Classifier following the scikit-learn API and exposing
        :meth:`predict_proba`.
    treatment_transformation : TransformerMixin, optional
        Transformer applied to the treatment column(s) before concatenation with
        the features. Pass the string ``"spline"`` to use a spline basis with
        10 knots and degree 2. Defaults to ``None``.
    n_datasets : int, optional
        Multiplier controlling how many synthetic samples are generated relative
        to the original sample size. Defaults to ``1``.
    max_trials : int, optional
        Maximum number of synthetic datasets to fit when running in convergence
        or ensemble mode. Defaults to ``5``.
    random_state : int, optional
        Seed forwarded to the internal random number generator. Defaults to ``0``.
    fit_mode : {"ensemble", "convergence"}, optional
        Controls how the classifier is trained on the synthetic data. Defaults
        to ``"ensemble"``.
    drop_not_converged : bool, optional
        Whether to discard ensemble members that fail the convergence heuristic.
        Defaults to ``False``.
    mean_type : {"arithmetic", "geometric"}, optional
        Aggregation used when ``fit_mode="ensemble"``. Defaults to
        ``"arithmetic"``.
    epsilon : float, optional
        Lower bound used to stabilize the denominator in the posterior odds.
        Defaults to ``1e-12``.
    """

    _tags = {
        "t_inner_mtype": pl.DataFrame,
        "X_inner_mtype": pl.DataFrame,
        "supported_t_dtypes": ALL_DTYPES,
        "capability:multidimensional_treatment": False,
        "density_kind": "stabilized",
    }

    def __init__(
        self,
        classifier: ClassifierMixin,
        treatment_transformation: TransformerMixin = None,
        n_datasets: int = 1,
        max_trials: int = 5,
        random_state: int = 0,
        fit_mode: str = "ensemble",
        drop_not_converged: bool = False,
        mean_type: str = "arithmetic",
        epsilon: float = 1e-12,
    ):
        self.classifier = classifier
        self.treatment_transformation = treatment_transformation
        self.n_datasets = n_datasets
        self.max_trials = max_trials
        self.random_state = random_state
        self.fit_mode = fit_mode
        self.drop_not_converged = drop_not_converged
        self.mean_type = mean_type
        self.epsilon = epsilon

        super().__init__()

    def _fit(self, X: pl.DataFrame, t: pl.DataFrame):
        self._validate_hyperparameters()
        self._rng = np.random.default_rng(self.random_state)
        self._treatment_transformation_ = self._clone_treatment_transformation()
        self.classifier_ = None
        self.classifiers_ = None

        treatment_features = self._prepare_treatment_features(
            t,
            fit_transformer=True,
        )

        if self.fit_mode == "ensemble":
            self._fit_ensemble(X, treatment_features)
        elif self.fit_mode == "convergence":
            self._fit_until_convergence(X, treatment_features)
        else:
            raise ValueError(f"Invalid fit_mode: {self.fit_mode}")

        return self

    def _predict_density(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        Xt = self._make_feature_data(X, t)
        probabilities = self._predict_with_classifiers(Xt)
        density_ratio = probabilities[:, 1] / np.clip(
            probabilities[:, 0], self.epsilon, None
        )
        return density_ratio.reshape(-1, 1)

    def _fit_until_convergence(self, X: pl.DataFrame, treatment_features: np.ndarray):
        for _ in range(self.max_trials):
            classifier, probabilities = self._fit_trial_classifier(
                X, treatment_features
            )
            if probabilities[:, 1].std() > 1e-6:
                self.classifier_ = classifier
                return

        raise ValueError(
            "Standard deviation of densities is too low. Check your model."
        )

    def _fit_ensemble(self, X: pl.DataFrame, treatment_features: np.ndarray):
        self.classifiers_ = []

        for trial in range(self.max_trials):
            classifier, probabilities = self._fit_trial_classifier(
                X, treatment_features
            )
            converged = probabilities[:, 1].std() > 1e-6

            if self.drop_not_converged and not converged:
                print(f"Trial {trial} did not converge, skipping.")
                continue

            self.classifiers_.append(classifier)

        if not self.classifiers_:
            raise ValueError(
                "No ensemble members converged; consider relaxing constraints."
            )

        self.classifier_ = None

    def _fit_trial_classifier(
        self,
        X: pl.DataFrame,
        treatment_features: np.ndarray,
    ):
        X_array = self._to_numpy_2d(X)
        X_classif, y_classif, sample_weights = self._make_classification_dataset(
            X_array,
            treatment_features,
        )

        classifier = copy.deepcopy(self.classifier)
        fit_kwargs = _resolve_sample_weight_fit_args(classifier, sample_weights)
        classifier.fit(X_classif, y_classif, **fit_kwargs)

        probabilities = self._predict_classifier_probabilities(classifier, X_classif)
        return classifier, probabilities

    def _make_classification_dataset(
        self,
        X_array: np.ndarray,
        treatment_features: np.ndarray,
    ):
        dataset_size = self._get_synthetic_dataset_size(len(X_array))
        Xt_observed = np.concatenate([X_array, treatment_features], axis=1)

        repeated_idx = np.arange(dataset_size) % len(X_array)
        X_permuted = X_array[repeated_idx]
        treatment_permuted = treatment_features[
            self._sample_balanced_permutation_indices(len(X_array), dataset_size)
        ]
        Xt_permuted = np.concatenate([X_permuted, treatment_permuted], axis=1)

        X_classif = np.concatenate([Xt_permuted, Xt_observed], axis=0).astype(float)
        y_classif = np.concatenate(
            [
                np.zeros(Xt_permuted.shape[0], dtype=int),
                np.ones(Xt_observed.shape[0], dtype=int),
            ]
        )

        sample_weights = np.ones_like(y_classif, dtype=float)
        sample_weights[y_classif == 0] = (
            0.5 * len(y_classif) / max((y_classif == 0).sum(), 1)
        )
        sample_weights[y_classif == 1] = (
            0.5 * len(y_classif) / max((y_classif == 1).sum(), 1)
        )

        return X_classif, y_classif, sample_weights

    def _sample_balanced_permutation_indices(self, n_samples: int, dataset_size: int):
        n_full_repetitions, remainder = divmod(dataset_size, n_samples)

        indices = [self._rng.permutation(n_samples) for _ in range(n_full_repetitions)]
        if remainder:
            indices.append(self._rng.permutation(n_samples)[:remainder])

        if not indices:
            return np.array([], dtype=int)

        return np.concatenate(indices)

    def _make_feature_data(self, X: pl.DataFrame, t: pl.DataFrame) -> np.ndarray:
        X_array = self._to_numpy_2d(X)
        treatment_features = self._prepare_treatment_features(
            t,
            fit_transformer=False,
        )
        return np.concatenate([X_array, treatment_features], axis=1).astype(float)

    def _prepare_treatment_features(
        self,
        t: pl.DataFrame,
        *,
        fit_transformer: bool,
    ) -> np.ndarray:
        treatment = convert_categorical_to_dummies(t)
        treatment_array = np.asarray(treatment.to_numpy(), dtype=float)
        if treatment_array.ndim == 1:
            treatment_array = treatment_array.reshape(-1, 1)

        if self._treatment_transformation_ is None:
            return treatment_array

        if fit_transformer:
            return self._treatment_transformation_.fit_transform(treatment_array)

        return self._treatment_transformation_.transform(treatment_array)

    def _predict_with_classifiers(self, Xt: np.ndarray) -> np.ndarray:
        if self.classifiers_:
            return self._aggregate_classifier_predictions(Xt)

        if self.classifier_ is None:
            raise ValueError("PermutationWeighting must be fitted before predicting.")

        return self._predict_classifier_probabilities(self.classifier_, Xt)

    def _aggregate_classifier_predictions(self, Xt: np.ndarray) -> np.ndarray:
        probabilities = [
            self._predict_classifier_probabilities(classifier, Xt)
            for classifier in self.classifiers_
        ]

        if not probabilities:
            raise ValueError("No classifiers available for aggregation.")

        probabilities = np.stack(probabilities, axis=0)
        if self.mean_type == "geometric":
            return np.prod(probabilities, axis=0) ** (1 / probabilities.shape[0])
        if self.mean_type == "arithmetic":
            return np.mean(probabilities, axis=0)

        raise ValueError(
            f"Unsupported mean_type '{self.mean_type}'. Use 'arithmetic' or 'geometric'."
        )

    def _predict_classifier_probabilities(
        self, classifier, Xt: np.ndarray
    ) -> np.ndarray:
        probabilities = classifier.predict_proba(Xt)
        return self._coerce_binary_probability_output(
            probabilities,
            classes=getattr(classifier, "classes_", None),
            n_rows=Xt.shape[0],
        )

    @staticmethod
    def _to_numpy_2d(frame: pl.DataFrame) -> np.ndarray:
        array = np.asarray(frame.to_numpy())
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def _clone_treatment_transformation(self):
        if self.treatment_transformation is None:
            return None
        if self.treatment_transformation == "spline":
            return SplineTransformer(n_knots=10, degree=2)
        return copy.deepcopy(self.treatment_transformation)

    def _validate_hyperparameters(self):
        if self.max_trials < 1:
            raise ValueError(f"max_trials must be at least 1, got {self.max_trials}.")
        if self.n_datasets <= 0:
            raise ValueError("n_datasets must be positive, got " f"{self.n_datasets}.")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}.")

    def _get_synthetic_dataset_size(self, n_samples: int) -> int:
        dataset_size = int(n_samples * self.n_datasets)
        if dataset_size < 1:
            raise ValueError("n_datasets produced an empty synthetic dataset.")
        return dataset_size

    @classmethod
    def _coerce_binary_probability_output(cls, probabilities, *, classes, n_rows: int):
        probability_array = np.asarray(probabilities, dtype=float)

        if probability_array.ndim == 1:
            if probability_array.shape[0] != n_rows:
                raise ValueError(
                    "Expected predict_proba output to contain one value per row, "
                    f"but received shape {probability_array.shape}."
                )
            probability_array = np.column_stack(
                [1.0 - probability_array, probability_array]
            )
        elif probability_array.ndim == 2:
            if probability_array.shape[0] != n_rows:
                raise ValueError(
                    "Expected predict_proba output to contain "
                    f"{n_rows} rows, but received shape {probability_array.shape}."
                )
            if probability_array.shape[1] == 1:
                probability_array = np.column_stack(
                    [1.0 - probability_array[:, 0], probability_array[:, 0]]
                )
            elif probability_array.shape[1] != 2:
                raise ValueError(
                    "PermutationWeighting expects binary predict_proba output, "
                    f"but received shape {probability_array.shape}."
                )
        else:
            raise ValueError(
                "Expected predict_proba output to be 1D or 2D, but received array "
                f"with shape {probability_array.shape}."
            )

        if classes is None:
            return probability_array

        normalized_classes = [cls._normalize_scalar(value) for value in classes]
        if len(normalized_classes) != 2:
            raise ValueError(
                "PermutationWeighting expects a binary classifier with exactly two classes."
            )

        try:
            zero_index = normalized_classes.index(0)
            one_index = normalized_classes.index(1)
        except ValueError as exc:
            raise ValueError(
                "PermutationWeighting expects classifier classes to be labeled 0 and 1."
            ) from exc

        return probability_array[:, [zero_index, one_index]]

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

        return [
            {
                "classifier": LogisticRegression(max_iter=1000),
                "n_datasets": 1,
                "max_trials": 2,
                "random_state": 0,
            }
        ]
