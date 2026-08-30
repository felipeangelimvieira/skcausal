import math
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

_SMALLEST_POSITIVE_FLOAT = np.nextafter(0.0, 1.0)
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


def _log_positive_difference(upper, lower):
    upper, lower = np.broadcast_arrays(
        np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)
    )
    with np.errstate(over="ignore"):
        difference = upper - lower
    result = np.log(difference)
    overflow = np.isposinf(difference)
    if overflow.any():
        result = np.asarray(result)
        result[overflow] = np.logaddexp(
            np.log(upper[overflow]), np.log(-lower[overflow])
        )
    return result


def _standard_normal_logcdf(value):
    value = float(value)
    if value == -np.inf:
        return -np.inf
    if value == np.inf:
        return 0.0
    if value > 0.0:
        log_survival = _standard_normal_logcdf(-value)
        return float(np.log1p(-np.exp(log_survival)))
    if value > -8.0:
        return math.log(0.5 * math.erfc(-value / math.sqrt(2.0)))

    inverse_square = 1.0 / (value * value)
    series = 1.0
    term = 1.0
    previous = np.inf
    for index in range(1, 100):
        term *= -(2 * index - 1) * inverse_square
        if abs(term) >= previous:
            break
        series += term
        if abs(term) <= np.finfo(float).eps * abs(series):
            break
        previous = abs(term)
    return -0.5 * value * value - math.log(-value) - _HALF_LOG_2PI + math.log(series)


def _log_difference(log_larger, log_smaller):
    if log_smaller == -np.inf:
        return log_larger
    difference = log_smaller - log_larger
    if difference >= 0.0:
        return -np.inf
    return log_larger + float(np.log(-np.expm1(difference)))


def _standard_normal_log_interval(lower, upper):
    lower = float(lower)
    upper = float(upper)
    if not lower < upper:
        return -np.inf
    if lower == -np.inf:
        return _standard_normal_logcdf(upper)
    if upper == np.inf:
        return _standard_normal_logcdf(-lower)
    if upper <= 0.0:
        return _log_difference(
            _standard_normal_logcdf(upper), _standard_normal_logcdf(lower)
        )
    if lower >= 0.0:
        return _log_difference(
            _standard_normal_logcdf(-lower), _standard_normal_logcdf(-upper)
        )

    excluded_probability = np.exp(_standard_normal_logcdf(lower)) + np.exp(
        _standard_normal_logcdf(-upper)
    )
    return float(np.log1p(-excluded_probability))


def _truncated_normal_logpdf(value, location, scale, lower, upper):
    value, location = np.broadcast_arrays(
        np.asarray(value, dtype=float), np.asarray(location, dtype=float)
    )
    normalizers = np.array(
        [
            _standard_normal_log_interval(
                (lower - row_location) / scale,
                (upper - row_location) / scale,
            )
            for row_location in location.flat
        ]
    ).reshape(location.shape)
    return _normal_logpdf(value, location, scale) - normalizers


def _sample_truncated_normal(location, scale, lower, upper, rng):
    location = np.asarray(location, dtype=float)
    uniforms = np.maximum(rng.random(location.size), _SMALLEST_POSITIVE_FLOAT)
    samples = np.empty(location.size, dtype=float)
    for index, (row_location, uniform) in enumerate(zip(location.flat, uniforms)):
        standardized_lower = (lower - row_location) / scale
        standardized_upper = (upper - row_location) / scale
        log_normalizer = _standard_normal_log_interval(
            standardized_lower, standardized_upper
        )
        target_log_mass = math.log(uniform) + log_normalizer
        left = lower
        right = upper
        for _ in range(100):
            middle = 0.5 * left + 0.5 * right
            standardized_middle = (middle - row_location) / scale
            log_mass = _standard_normal_log_interval(
                standardized_lower, standardized_middle
            )
            if log_mass < target_log_mass:
                left = middle
            else:
                right = middle
        samples[index] = 0.5 * left + 0.5 * right

    return samples.reshape(location.shape)


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
        self.latent_support_bounds_ = self._latent_support_bounds()

        standard_normal = NormalDist()
        central_bounds = np.array(
            [standard_normal.inv_cdf(0.01), standard_normal.inv_cdf(0.99)]
        ) * float(self.treatment_noise_scale)
        self.latent_grid_bounds_ = np.array(
            [
                max(central_bounds[0], self.latent_support_bounds_[0]),
                min(central_bounds[1], self.latent_support_bounds_[1]),
            ]
        )
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
        return (1.0 - unit) * lower + unit * upper

    def _latent_support_bounds(self):
        if self.treatment_domain == "real":
            return np.array([-np.inf, np.inf])
        if self.treatment_domain == "positive":
            lower = float(np.log(_SMALLEST_POSITIVE_FLOAT))
            upper = float(np.log(np.finfo(float).max))
            while not np.isfinite(np.exp(upper)):
                upper = np.nextafter(upper, -np.inf)
            return np.array([lower, upper])

        domain_lower, domain_upper = map(float, self.treatment_domain)
        treatment_lower = np.nextafter(domain_lower, domain_upper)
        treatment_upper = np.nextafter(domain_upper, domain_lower)
        if treatment_lower > treatment_upper:
            raise ValueError(
                "treatment_domain must contain a representable interior value."
            )
        lower = float(self._inverse_transform(np.array([treatment_lower]))[0])
        upper = float(self._inverse_transform(np.array([treatment_upper]))[0])
        if self._forward_transform(np.array([lower]))[0] <= domain_lower:
            unsafe = lower
            safe = 0.0
            for _ in range(100):
                middle = 0.5 * unsafe + 0.5 * safe
                if self._forward_transform(np.array([middle]))[0] > domain_lower:
                    safe = middle
                else:
                    unsafe = middle
            lower = safe
        if self._forward_transform(np.array([upper]))[0] >= domain_upper:
            safe = 0.0
            unsafe = upper
            for _ in range(100):
                middle = 0.5 * safe + 0.5 * unsafe
                if self._forward_transform(np.array([middle]))[0] < domain_upper:
                    safe = middle
                else:
                    unsafe = middle
            upper = safe
        return np.array([lower, upper])

    def _inverse_transform(self, treatment):
        treatment = np.asarray(treatment, dtype=float)
        if self.treatment_domain == "real":
            return treatment.copy()
        if self.treatment_domain == "positive":
            return np.log(treatment)

        lower, upper = map(float, self.treatment_domain)
        return _log_positive_difference(treatment, lower) - _log_positive_difference(
            upper, treatment
        )

    def _log_abs_inverse_jacobian(self, treatment):
        treatment = np.asarray(treatment, dtype=float)
        if self.treatment_domain == "real":
            return np.zeros_like(treatment)
        if self.treatment_domain == "positive":
            return -np.log(treatment)

        lower, upper = map(float, self.treatment_domain)
        return (
            _log_positive_difference(upper, lower)
            - _log_positive_difference(treatment, lower)
            - _log_positive_difference(upper, treatment)
        )

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
        if self.treatment_domain == "real":
            latent = location + rng.normal(
                scale=self.treatment_noise_scale, size=X.height
            )
        else:
            latent = _sample_truncated_normal(
                location,
                self.treatment_noise_scale,
                *self.latent_support_bounds_,
                rng,
            )
        return pl.DataFrame({"t": self._forward_transform(latent)})

    def _log_prob(self, X, treatment):
        values = treatment.get_column("t").to_numpy().astype(float, copy=False)
        valid = self._valid_treatment(values)
        log_density = np.full(values.shape, -np.inf, dtype=float)
        if not valid.any():
            return log_density

        valid_indices = np.flatnonzero(valid)
        latent = self._inverse_transform(values[valid_indices])
        if self.treatment_domain != "real":
            lower, upper = self.latent_support_bounds_
            supported = (latent >= lower) & (latent <= upper)
            valid_indices = valid_indices[supported]
            latent = latent[supported]
            if valid_indices.size == 0:
                return log_density

        if self.treatment_domain == "real":
            confounded = _normal_logpdf(
                latent,
                self._latent_mean(X)[valid_indices],
                self.treatment_noise_scale,
            )
            randomized = _normal_logpdf(latent, 0.0, self.treatment_noise_scale)
        else:
            confounded = _truncated_normal_logpdf(
                latent,
                self._latent_mean(X)[valid_indices],
                self.treatment_noise_scale,
                lower,
                upper,
            )
            randomized = _truncated_normal_logpdf(
                latent,
                0.0,
                self.treatment_noise_scale,
                lower,
                upper,
            )
        log_density[valid_indices] = _log_mixture(
            confounded,
            randomized,
            self.randomized_weight,
        ) + self._log_abs_inverse_jacobian(values[valid_indices])
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
