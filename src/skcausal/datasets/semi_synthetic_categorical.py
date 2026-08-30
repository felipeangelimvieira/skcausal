from typing import ClassVar

import numpy as np
import polars as pl

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _calibrate_confounding_strength,
    _calibrate_softmax_intercepts,
    _fit_baseline_surface,
    _fit_causal_score_map,
    _softmax,
)

__all__ = ["CategoricalSemiSyntheticDataset"]


class CategoricalSemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types: ClassVar = {"t": "categorical"}

    def __init__(
        self,
        real_dataset,
        n_treatments=2,
        target_probabilities=None,
        confounding_strength=1.0,
        target_confounding_bias=None,
        treatment_only_strength=1.0,
        randomized_weight=0.1,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(n_treatments, bool) or not isinstance(n_treatments, int):
            raise TypeError("n_treatments must be an integer.")
        if n_treatments < 2:
            raise ValueError("n_treatments must be at least 2.")
        probabilities = (
            np.full(n_treatments, 1.0 / n_treatments)
            if target_probabilities is None
            else np.asarray(target_probabilities, dtype=float)
        )
        if (
            probabilities.shape != (n_treatments,)
            or np.any(probabilities <= 0.0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "target_probabilities must contain one positive value per treatment "
                "and sum to one."
            )
        for name, value in {
            "confounding_strength": confounding_strength,
            "treatment_only_strength": treatment_only_strength,
            "outcome_effect_scale": outcome_effect_scale,
            "effect_heterogeneity_scale": effect_heterogeneity_scale,
            "outcome_noise_scale": outcome_noise_scale,
        }.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if target_confounding_bias is not None and (
            not np.isfinite(float(target_confounding_bias))
            or float(target_confounding_bias) < 0.0
        ):
            raise ValueError("target_confounding_bias must be nonnegative.")
        if not 0.0 <= float(randomized_weight) <= 1.0:
            raise ValueError("randomized_weight must be between 0 and 1.")
        self.n_treatments = n_treatments
        self.target_probabilities = target_probabilities
        self.confounding_strength = confounding_strength
        self.target_confounding_bias = target_confounding_bias
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.outcome_noise_scale = outcome_noise_scale
        self._validated_target_probabilities = probabilities
        super().__init__(real_dataset=real_dataset, random_state=random_state)

    def _prepare_dgp(self, X, rng):
        self.score_map_, self.source_scores_ = _fit_causal_score_map(X, rng)
        self.baseline_surface_ = _fit_baseline_surface(self.source_scores_, rng)
        self.source_mu0_, self.source_confounding_signal_ = (
            self.baseline_surface_.evaluate(self.source_scores_)
        )
        self.baseline_standard_deviation_ = float(np.std(self.source_mu0_, ddof=0))

        self.treatment_levels_ = [str(index) for index in range(self.n_treatments)]
        self.target_probabilities_ = self._validated_target_probabilities.copy()

        self.confounding_loadings_ = np.linspace(-1.0, 1.0, self.n_treatments)
        self.confounding_loadings_ -= self.confounding_loadings_.mean()

        self.treatment_only_coefficients_ = rng.normal(
            scale=1.0 / np.sqrt(2.0), size=(2, self.n_treatments)
        )
        self.treatment_only_coefficients_ -= self.treatment_only_coefficients_.mean(
            axis=1, keepdims=True
        )

        self.outcome_main_effects_ = rng.normal(
            scale=self.outcome_effect_scale, size=self.n_treatments - 1
        )
        self.outcome_modifier_coefficients_ = rng.normal(
            scale=self.effect_heterogeneity_scale,
            size=(self.n_treatments - 1, 2),
        )

        self._source_category_outcome_means = np.column_stack(
            [
                self.source_mu0_,
                self.source_mu0_[:, None]
                + self.outcome_main_effects_[None, :]
                + self.source_scores_.modifiers @ self.outcome_modifier_coefficients_.T,
            ]
        )

        if self.baseline_standard_deviation_ == 0.0:
            raise ValueError(
                "Categorical confounding bias cannot be normalized because the "
                "baseline standard deviation is zero."
            )
        if (
            self.randomized_weight == 1.0
            and self.target_confounding_bias is not None
            and float(self.target_confounding_bias) > 0.0
        ):
            raise ValueError(
                "A positive target_confounding_bias is unattainable for a fully "
                "randomized assignment."
            )

        if self.target_confounding_bias is None:
            self.confounding_strength_ = self.confounding_strength
        elif self.randomized_weight == 1.0:
            self.confounding_strength_ = 0.0
        else:
            self.confounding_strength_, _ = _calibrate_confounding_strength(
                lambda strength: self._bias_at_strength(X, strength)[1],
                float(self.target_confounding_bias),
                self.confounding_strength,
            )

        _, self.assignment_intercepts_ = self._probabilities_at_strength(
            X, self.confounding_strength_
        )
        self.confounding_bias_, self.confounding_bias_ratio_ = self._bias_at_strength(
            X, self.confounding_strength_
        )

    def _check_and_transform_X(self, covariates):
        if isinstance(covariates, np.ndarray):
            array = np.asarray(covariates)
            if array.ndim != 2 or array.shape[1] != len(self.score_map_.columns):
                raise ValueError(
                    "NumPy covariates must match the frozen covariate width."
                )
            covariates = pl.DataFrame(
                {
                    column: array[:, index]
                    for index, column in enumerate(self.score_map_.columns)
                }
            )
        return super()._check_and_transform_X(covariates)

    def _confounded_logits(self, X, strength):
        scores = self.score_map_.transform(X)
        _, signal = self.baseline_surface_.evaluate(scores)
        treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
        return (
            strength * signal[:, None] * self.confounding_loadings_[None, :]
            + self.treatment_only_strength * treatment_only
        )

    def _probabilities_at_strength(self, X, strength):
        logits = self._confounded_logits(X, strength)
        intercepts = _calibrate_softmax_intercepts(logits, self.target_probabilities_)
        confounded = _softmax(logits + intercepts)
        probabilities = (
            1.0 - self.randomized_weight
        ) * confounded + self.randomized_weight * self.target_probabilities_
        return probabilities, intercepts

    def _bias_at_strength(self, X, strength):
        if self.randomized_weight == 1.0:
            return 0.0, 0.0
        probabilities, _ = self._probabilities_at_strength(X, strength)
        causal_means = self._source_category_outcome_means.mean(axis=0)
        observational_means = (probabilities * self._source_category_outcome_means).sum(
            axis=0
        ) / probabilities.sum(axis=0)
        distortions = observational_means - causal_means
        bias = float(np.ptp(distortions))
        return bias, bias / self.baseline_standard_deviation_

    def _probabilities(self, X, strength=None):
        if strength is not None:
            probabilities, _ = self._probabilities_at_strength(X, strength)
            return probabilities
        confounded = _softmax(
            self._confounded_logits(X, self.confounding_strength_)
            + self.assignment_intercepts_
        )
        return (
            1.0 - self.randomized_weight
        ) * confounded + self.randomized_weight * self.target_probabilities_

    def _sample_treatment(self, X, rng):
        probabilities = self._probabilities(X)
        uniforms = rng.random(probabilities.shape[0])
        indices = (uniforms[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
        levels = np.asarray(self.treatment_levels_, dtype=str)[indices]
        return pl.DataFrame({"t": levels})

    def _treatment_indices(self, treatment, *, allow_unknown):
        labels = treatment.get_column("t").cast(pl.Utf8).to_numpy()
        level_to_index = {
            level: index for index, level in enumerate(self.treatment_levels_)
        }
        indices = np.array(
            [level_to_index.get(str(label), -1) for label in labels], dtype=int
        )
        if not allow_unknown and np.any(indices < 0):
            unsupported = sorted(set(labels[indices < 0].tolist()))
            raise ValueError(f"Unsupported treatment levels: {unsupported}.")
        return indices

    def _log_prob(self, X, treatment):
        probabilities = self._probabilities(X)
        indices = self._treatment_indices(treatment, allow_unknown=True)
        log_probabilities = np.full(indices.shape, -np.inf, dtype=float)
        valid = indices >= 0
        with np.errstate(divide="ignore"):
            log_probabilities[valid] = np.log(
                probabilities[np.arange(indices.size)[valid], indices[valid]]
            )
        return log_probabilities

    def _treatment_basis(self, treatment):
        indices = self._treatment_indices(treatment, allow_unknown=False)
        return (indices[:, None] == np.arange(1, self.n_treatments)[None, :]).astype(
            float
        )

    def _predict_y(self, X, treatment):
        scores = self.score_map_.transform(X)
        mu0, _ = self.baseline_surface_.evaluate(scores)
        basis = self._treatment_basis(treatment)
        modifier_effects = basis @ self.outcome_modifier_coefficients_
        interaction = np.einsum("ij,ij->i", modifier_effects, scores.modifiers)
        return mu0 + basis @ self.outcome_main_effects_ + interaction

    def _sample_outcome(self, mean, X, treatment, rng):
        mean = np.asarray(mean, dtype=float)
        return mean + rng.normal(scale=self.outcome_noise_scale, size=mean.shape)

    def _get_grid(self, n):
        return self.get_levels()

    def get_levels(self):
        return self._coerce_treatment_frame(pl.DataFrame({"t": self.treatment_levels_}))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.datasets.kang_schafer import KangSchaferContinuous

        return [
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=3),
                "random_state": 7,
            }
        ]
