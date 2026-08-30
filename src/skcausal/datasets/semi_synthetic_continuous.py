from statistics import NormalDist
from typing import ClassVar

import numpy as np
import polars as pl

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _fit_baseline_surface,
    _fit_causal_score_map,
    _log_mixture,
    _normal_logpdf,
)

__all__ = ["ContinuousSemiSyntheticDataset"]


class ContinuousSemiSyntheticDataset(BaseSemiSyntheticDataset):
    column_types: ClassVar = {"t": "continuous"}

    def __init__(
        self,
        real_dataset,
        treatment_domain="real",
        confounding_strength=1.0,
        treatment_only_strength=1.0,
        randomized_weight=0.1,
        treatment_noise_scale=1.0,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(treatment_domain, str):
            valid_domain = treatment_domain in {"real", "positive"}
        else:
            domain_array = np.asarray(treatment_domain)
            valid_domain = (
                isinstance(treatment_domain, tuple)
                and len(treatment_domain) == 2
                and np.issubdtype(domain_array.dtype, np.number)
                and np.isfinite(domain_array).all()
                and float(treatment_domain[0]) < float(treatment_domain[1])
            )
        if not valid_domain:
            raise ValueError(
                "treatment_domain must be 'real', 'positive', or a finite "
                "(lower, upper) tuple."
            )
        if not np.isfinite(float(treatment_noise_scale)) or (
            float(treatment_noise_scale) <= 0.0
        ):
            raise ValueError("treatment_noise_scale must be positive.")
        for name, value in {
            "confounding_strength": confounding_strength,
            "treatment_only_strength": treatment_only_strength,
            "outcome_effect_scale": outcome_effect_scale,
            "effect_heterogeneity_scale": effect_heterogeneity_scale,
            "outcome_noise_scale": outcome_noise_scale,
        }.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if not np.isfinite(float(randomized_weight)) or not (
            0.0 <= float(randomized_weight) <= 1.0
        ):
            raise ValueError("randomized_weight must be between 0 and 1.")

        self.treatment_domain = treatment_domain
        self.confounding_strength = confounding_strength
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.treatment_noise_scale = treatment_noise_scale
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.outcome_noise_scale = outcome_noise_scale
        super().__init__(real_dataset=real_dataset, random_state=random_state)

    def _prepare_dgp(self, X, rng):
        self.score_map_, self.source_scores_ = _fit_causal_score_map(X, rng)
        self.baseline_surface_ = _fit_baseline_surface(self.source_scores_, rng)
        self.source_mu0_, self.source_confounding_signal_ = (
            self.baseline_surface_.evaluate(self.source_scores_)
        )
        self.baseline_standard_deviation_ = float(np.std(self.source_mu0_, ddof=0))

        self.treatment_only_coefficients_ = rng.normal(scale=1.0 / np.sqrt(2.0), size=2)
        self.outcome_main_effects_ = rng.normal(scale=self.outcome_effect_scale, size=5)
        self.outcome_modifier_coefficients_ = rng.normal(
            scale=self.effect_heterogeneity_scale, size=(5, 2)
        )
        self.confounding_strength_ = self.confounding_strength

        standard_normal = NormalDist()
        self.latent_grid_bounds_ = np.array(
            [standard_normal.inv_cdf(0.01), standard_normal.inv_cdf(0.99)]
        ) * float(self.treatment_noise_scale)
        self.hinge_knots_ = np.array(
            [standard_normal.inv_cdf(1.0 / 3.0), standard_normal.inv_cdf(2.0 / 3.0)]
        ) * float(self.treatment_noise_scale)
        self.latent_basis_grid_ = np.linspace(*self.latent_grid_bounds_, 1001)
        self.raw_treatment_basis_ = self._raw_treatment_basis(self.latent_basis_grid_)
        self.treatment_basis_means_ = self.raw_treatment_basis_.mean(axis=0)
        self.treatment_basis_scales_ = self.raw_treatment_basis_.std(axis=0, ddof=0)
        self.treatment_basis_scales_[self.treatment_basis_scales_ == 0.0] = 1.0
        self.reference_treatment_basis_ = self._standardized_treatment_basis(
            np.zeros(1)
        )[0]

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

    def _forward_transform(self, latent):
        latent = np.asarray(latent, dtype=float)
        if self.treatment_domain == "real":
            return latent.copy()
        if self.treatment_domain == "positive":
            return np.exp(latent)

        lower, upper = map(float, self.treatment_domain)
        unit = np.empty_like(latent)
        nonnegative = latent >= 0.0
        unit[nonnegative] = 1.0 / (1.0 + np.exp(-latent[nonnegative]))
        exponentiated = np.exp(latent[~nonnegative])
        unit[~nonnegative] = exponentiated / (1.0 + exponentiated)
        return lower + (upper - lower) * unit

    def _inverse_transform(self, treatment):
        treatment = np.asarray(treatment, dtype=float)
        if self.treatment_domain == "real":
            return treatment.copy()
        if self.treatment_domain == "positive":
            return np.log(treatment)

        lower, upper = map(float, self.treatment_domain)
        unit = (treatment - lower) / (upper - lower)
        return np.log(unit) - np.log1p(-unit)

    def _log_abs_inverse_jacobian(self, treatment):
        treatment = np.asarray(treatment, dtype=float)
        if self.treatment_domain == "real":
            return np.zeros_like(treatment)
        if self.treatment_domain == "positive":
            return -np.log(treatment)

        lower, upper = map(float, self.treatment_domain)
        unit = (treatment - lower) / (upper - lower)
        return -np.log(upper - lower) - np.log(unit) - np.log1p(-unit)

    def _valid_treatment(self, treatment):
        treatment = np.asarray(treatment, dtype=float)
        if self.treatment_domain == "real":
            return np.isfinite(treatment)
        if self.treatment_domain == "positive":
            return np.isfinite(treatment) & (treatment > 0.0)
        lower, upper = map(float, self.treatment_domain)
        return np.isfinite(treatment) & (treatment > lower) & (treatment < upper)

    def _latent_mean(self, X, strength=None):
        strength = self.confounding_strength_ if strength is None else strength
        scores = self.score_map_.transform(X)
        _, signal = self.baseline_surface_.evaluate(scores)
        treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
        return strength * signal + self.treatment_only_strength * treatment_only

    def _sample_treatment(self, X, rng):
        randomized = rng.random(X.height) < self.randomized_weight
        location = np.where(randomized, 0.0, self._latent_mean(X))
        latent = location + rng.normal(scale=self.treatment_noise_scale, size=X.height)
        return pl.DataFrame({"t": self._forward_transform(latent)})

    def _log_prob(self, X, treatment):
        values = treatment.get_column("t").to_numpy().astype(float, copy=False)
        valid = self._valid_treatment(values)
        log_density = np.full(values.shape, -np.inf, dtype=float)
        if not valid.any():
            return log_density

        latent = self._inverse_transform(values[valid])
        confounded = _normal_logpdf(
            latent,
            self._latent_mean(X)[valid],
            self.treatment_noise_scale,
        )
        randomized = _normal_logpdf(latent, 0.0, self.treatment_noise_scale)
        log_density[valid] = _log_mixture(
            confounded,
            randomized,
            self.randomized_weight,
        ) + self._log_abs_inverse_jacobian(values[valid])
        return log_density

    def _raw_treatment_basis(self, latent):
        latent = np.asarray(latent, dtype=float).reshape(-1)
        return np.column_stack(
            (
                latent,
                latent**2,
                np.sin(latent),
                np.maximum(latent - self.hinge_knots_[0], 0.0),
                np.maximum(latent - self.hinge_knots_[1], 0.0),
            )
        )

    def _standardized_treatment_basis(self, latent):
        return (
            self._raw_treatment_basis(latent) - self.treatment_basis_means_
        ) / self.treatment_basis_scales_

    def _treatment_basis(self, treatment):
        values = treatment.get_column("t").to_numpy().astype(float, copy=False)
        if not self._valid_treatment(values).all():
            raise ValueError("Treatment values must lie within treatment_domain.")
        latent = self._inverse_transform(values)
        return (
            self._standardized_treatment_basis(latent) - self.reference_treatment_basis_
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
        latent = np.linspace(*self.latent_grid_bounds_, n)
        return pl.DataFrame({"t": self._forward_transform(latent)})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.datasets.kang_schafer import KangSchaferContinuous

        return [
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=4),
                "random_state": 7,
            },
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=4),
                "treatment_domain": (-2.0, 3.0),
                "random_state": 8,
            },
        ]
