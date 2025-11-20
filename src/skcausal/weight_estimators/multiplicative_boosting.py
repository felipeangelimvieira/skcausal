import copy
import numpy as np
from typing import Callable
import polars as pl
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from skcausal.utils.polars import ALL_DTYPES
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from skcausal.weight_estimators.synthetic_weight import (
    make_probabilistic_classification_dataset,
    domain_size,
)
from skcausal.utils.sklearn import _resolve_sample_weight_fit_args

__all__ = [
    "DiscriminativeWeightBoosting",
]


class DiscriminativeWeightBoosting(BaseBalancingWeightRegressor):
    """Boosted classifier-based balancing weights via density-ratio factorization.

    Learns 1 / rho(x,t) multiplicatively from a sequence of synthetic
    classification problems distinguishing:
        - class 0: reweighted observational distribution P_s^{(k-1)}
        - class 1: synthetic target distribution P_t (controlled by `method`)

    At prediction time, returns balancing weights
        w_i ≈ rho(x_i, t_i) = P_t(x_i, t_i) / P_s(x_i, t_i),
    which specialize to (stabilized) inverse propensity weights for suitable `method`.
    """

    _tags = {
        "t_inner_mtype": pl.DataFrame,
        "one_hot_encode_enum_columns": False,
        "supported_t_dtypes": ALL_DTYPES,
        # conceptually can be more general, but keep for compatibility
        "balancing_weight_type": "propensity_score",
    }

    def __init__(
        self,
        classifier: ClassifierMixin,
        n_boosting_iter: int = 100,
        complexity_factor: float = 1,
        method: str = "uniform",
        treatment_transformation: TransformerMixin = None,
        random_state: int = 0,
        n_datasets: int = 1,
    ):
        self.classifier = classifier
        self.n_boosting_iter = n_boosting_iter
        self.complexity_factor = complexity_factor
        self.treatment_transformation = treatment_transformation
        self.random_state = random_state
        self.method = method
        self.n_datasets = n_datasets

        super().__init__()

        # RNG
        self._rng = np.random.default_rng(self.random_state)

        # optional spline transform
        self._treatment_transformation = self.treatment_transformation
        if self.treatment_transformation == "spline":
            self._treatment_transformation = SplineTransformer(n_knots=10, degree=2)

        self.num_synthetic_dataset_generations = n_datasets

        # boosting learning rate exponent
        self._learning_rate_power = self.complexity_factor / self.n_boosting_iter

        if self.method.startswith("balanced"):
            self.set_tags(balancing_weight_type="stabilized")

    # ------------------------------------------------------------------
    # core: fit boosting sequence
    # ------------------------------------------------------------------
    def _fit(self, X, t):
        # assume X, t are polars DataFrames as per tags
        self.domain_size_ = domain_size(t)

        # fit treatment transformer on original t
        if self._treatment_transformation is not None:
            t_array = t.to_numpy()
            if t_array.ndim == 1:
                t_array = t_array.reshape(-1, 1)
            self._treatment_transformation.fit(t_array)

        self.classifiers_ = []
        self._n_training_samples_ = X.shape[0]

        for _ in range(self.n_boosting_iter):
            # Synthetic classification problem for current iteration:
            # y=0: observational; y=1: synthetic target P_t (depends on `method`).
            X_classif, y_classif, _ = make_probabilistic_classification_dataset(
                rng=self._rng,
                X=X,
                t=t,
                method=self.method,
                dataset_size=int(X.shape[0] * self.num_synthetic_dataset_generations),
                feature_transformation=self._make_feature_data,
            )

            n_tot = X_classif.shape[0]
            idx_0 = y_classif == 0
            idx_1 = y_classif == 1

            # ----- compute 1 / rho_{k-1}(z) on THIS pooled dataset -----
            inv_rho_prev = np.ones(n_tot, dtype=float)
            if self.classifiers_:
                inv_rho_prev = self._predict_all_classifiers(X_classif)

            # P_s^{(k-1)}(z) ∝ P_s(z) * rho_{k-1}(z)
            # since inv_rho_prev ≈ 1 / rho_{k-1}, we use weights ∝ 1 / inv_rho_prev
            sample_weight = np.zeros(n_tot, dtype=float)

            if idx_0.any():
                w0 = 1.0 / np.clip(inv_rho_prev[idx_0], 1e-12, None)
                # normalize: half of total weight mass to class 0
                w0 *= (n_tot / 2.0) / idx_0.sum()  # w0.sum()
                sample_weight[idx_0] = w0

            if idx_1.any():
                # class 1: synthetic target P_t, uniform within; half mass
                w1 = np.ones(idx_1.sum(), dtype=float)
                w1 *= (n_tot / 2.0) / w1.sum()
                sample_weight[idx_1] = w1

            # ----- fit new classifier under these weights -----
            classif = copy.deepcopy(self.classifier)
            fit_kwargs = _resolve_sample_weight_fit_args(classif, sample_weight)
            if not fit_kwargs:
                raise TypeError(
                    "Classifier does not expose a sample_weight-compatible fit signature"
                )
            classif.fit(X_classif, y_classif, **fit_kwargs)
            self.classifiers_.append(classif)

        self.n_estimators_trained_ = len(self.classifiers_)
        return self

    # ------------------------------------------------------------------
    # helper: product of per-round ratios = estimate of 1 / rho(z)
    # ------------------------------------------------------------------
    def _predict_all_classifiers(self, Xt):
        """Compute inv_rho(z) ≈ prod_k ratio_k(z)^{lr} on given design points."""
        p = np.ones(Xt.shape[0], dtype=float)
        for classif in self.classifiers_:
            ratio = self._classifier_ratio(classif, Xt)
            p *= ratio**self._learning_rate_power
        return p

    def _classifier_ratio(self, classifier, Xt):
        """Return P(C=0|z) / P(C=1|z), clipped for stability."""
        probas = classifier.predict_proba(Xt)
        # class 0 = (reweighted) source, class 1 = target
        num = probas[:, 0]
        den = np.clip(probas[:, 1], 1e-9, None)
        ratio = num / den
        return np.clip(ratio, 1e-6, 1e6)

    # ------------------------------------------------------------------
    # public: balancing weights for observational sample
    # ------------------------------------------------------------------
    def _predict_sample_weight(self, X, t):
        """Return balancing weights w_i ≈ rho(x_i, t_i) for the chosen target."""
        if not hasattr(self, "classifiers_") or not self.classifiers_:
            raise NotFittedError(
                "DiscriminativeWeightBoosting instance is not fitted yet."
            )

        # Build a fresh pooled dataset (same construction as in training).
        dataset_size = int(X.shape[0] * self.num_synthetic_dataset_generations)
        Xt_aug, y_aug, _ = make_probabilistic_classification_dataset(
            rng=self._rng,
            X=X,
            t=t,
            method=self.method,
            dataset_size=dataset_size,
            feature_transformation=self._make_feature_data,
        )

        n_tot = Xt_aug.shape[0]
        batch_size = 10_000

        # compute inv_rho(z) on the pooled grid (batched if needed)
        if n_tot <= batch_size:
            inv_rho = self._predict_all_classifiers(Xt_aug)
        else:
            parts = []
            for start in range(0, n_tot, batch_size):
                end = min(start + batch_size, n_tot)
                parts.append(self._predict_all_classifiers(Xt_aug[start:end]))
            inv_rho = np.concatenate(parts, axis=0)

        # y_aug == 1: synthetic target P_t samples
        mask_target = y_aug == 1
        if not mask_target.any():
            raise ValueError("No target-class samples in augmented dataset.")
        inv_rho_target = inv_rho[mask_target]

        # Z ≈ E_{P_t}[1 / rho(Z)] ; should be close to 1 if model is correct

        Z_x = inv_rho_target.reshape(
            (mask_target.sum() // X.shape[0], X.shape[0])
        ).mean(axis=0)

        inv_rho_target /= np.tile(Z_x, self.num_synthetic_dataset_generations)

        # y_aug == 0: original observational samples, in original order by construction
        mask_source = y_aug == 0
        inv_rho_source = inv_rho[mask_source]
        inv_rho_source /= Z_x
        if inv_rho_source.shape[0] != X.shape[0]:
            raise ValueError(
                "Mismatch between observational samples and augmented labeling."
            )

        # rho = 1 / inv_rho; weights w_i = rho_i / Z
        rho = 1.0 / np.clip(inv_rho_source, 1e-12, None)

        w = rho.reshape(-1, 1)

        if self.method == "uniform":
            w *= self.domain_size_

        if not np.all(np.isfinite(w)):
            raise ValueError(
                "Weights contain NaN or Inf. Check model fit or clipping thresholds."
            )

        return w

    # ------------------------------------------------------------------
    # feature builder: [X, transformed(t)]
    # ------------------------------------------------------------------
    def _make_feature_data(self, X, t):
        # X, t can be polars or numpy/pandas; convert to numpy arrays
        if hasattr(X, "to_numpy"):
            X_array = X.to_numpy()
        else:
            X_array = np.asarray(X)

        t_array = t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)
        if t_array.ndim == 1:
            t_array = t_array.reshape(-1, 1)

        if self._treatment_transformation is not None:
            t_array = self._treatment_transformation.transform(t_array)

        Xt = np.concatenate([X_array, t_array], axis=1)
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.linear_model import LogisticRegression

        return [{"classifier": LogisticRegression()}]
