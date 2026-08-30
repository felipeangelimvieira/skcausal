import math
from statistics import NormalDist
from typing import ClassVar

import numpy as np
import polars as pl

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _calibrate_confounding_strength,
    _fit_baseline_surface,
    _fit_causal_score_map,
    _log_mixture,
    _normal_logpdf,
    _normalized_log_weights,
)

__all__ = ["ContinuousSemiSyntheticDataset"]

_SMALLEST_POSITIVE_FLOAT = np.nextafter(0.0, 1.0)
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


def _require_finite(values, message):
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError(message)


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

    width = upper - lower
    midpoint = 0.5 * lower + 0.5 * upper
    if width * max(1.0, abs(midpoint)) <= np.sqrt(np.finfo(float).eps):
        return math.log(width) - 0.5 * midpoint * midpoint - _HALF_LOG_2PI

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
    r"""Semi-synthetic DGP with one scalar continuous treatment.

    The assignment is a mixture of a covariate-dependent latent Gaussian and a
    randomized latent Gaussian. For ``treatment_domain="real"``, the latent
    Gaussians are unconditioned and the released treatment equals the latent
    value. For ``"positive"`` and finite ``(lower, upper)`` domains, each latent
    Gaussian component is instead conditioned on the finite interval whose
    endpoints round-trip through the configured transform to distinct,
    representable, strictly interior floating-point treatments. Sampling and
    :meth:`log_prob` use this same conditioned law exactly; it is deliberately
    not an untruncated real-number Gaussian followed by clipping.

    Parameters
    ----------
    real_dataset : BaseDataset
        Dataset supplying the covariate rows retained by this DGP.
    treatment_domain : {"real", "positive"} or tuple of float, default="real"
        Released treatment support. A tuple must contain finite ``lower < upper``
        bounds and at least two distinct representable interior float values.
        Positive and bounded domains use the conditioned latent law described
        above because more extreme latent floats cannot be transformed and
        inverted representably.
    confounding_strength : float, default=1.0
        Nonnegative scale of the outcome-predictive assignment signal.
    target_confounding_bias : float or None, default=None
        Optional nonnegative normalized oracle-bias calibration target.
    treatment_only_strength : float, default=1.0
        Nonnegative scale of treatment-only predictors.
    randomized_weight : float, default=0.1
        Weight in ``[0, 1]`` of the covariate-independent mixture component.
    treatment_noise_scale : float, default=1.0
        Strictly positive latent Gaussian standard deviation.
    outcome_effect_scale : float, default=1.0
        Nonnegative scale of treatment main-effect coefficients.
    effect_heterogeneity_scale : float, default=0.5
        Nonnegative scale of treatment-by-covariate coefficients.
    outcome_noise_scale : float, default=1.0
        Nonnegative Gaussian outcome-noise standard deviation.
    random_state : int, default=42
        Seed controlling frozen structure and the initial sample.
    """

    column_types: ClassVar = {"t": "continuous"}

    def __init__(
        self,
        real_dataset,
        treatment_domain="real",
        confounding_strength=1.0,
        target_confounding_bias=None,
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
        if target_confounding_bias is not None and (
            not np.isfinite(float(target_confounding_bias))
            or float(target_confounding_bias) < 0.0
        ):
            raise ValueError("target_confounding_bias must be nonnegative.")
        if not np.isfinite(float(randomized_weight)) or not (
            0.0 <= float(randomized_weight) <= 1.0
        ):
            raise ValueError("randomized_weight must be between 0 and 1.")

        self.treatment_domain = treatment_domain
        self.confounding_strength = confounding_strength
        self.target_confounding_bias = target_confounding_bias
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
        _require_finite(
            self.source_mu0_, "Derived continuous baseline means must be finite."
        )
        _require_finite(
            self.source_confounding_signal_,
            "Derived continuous confounding signals must be finite.",
        )
        self.baseline_standard_deviation_ = float(np.std(self.source_mu0_, ddof=0))
        if (
            not np.isfinite(self.baseline_standard_deviation_)
            or self.baseline_standard_deviation_ <= 0.0
        ):
            raise ValueError(
                "Continuous confounding bias cannot be normalized because the "
                "baseline standard deviation must be finite and positive."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            self.treatment_only_coefficients_ = rng.normal(
                scale=1.0 / np.sqrt(2.0), size=2
            )
            self.outcome_main_effects_ = rng.normal(
                scale=self.outcome_effect_scale, size=5
            )
            self.outcome_modifier_coefficients_ = rng.normal(
                scale=self.effect_heterogeneity_scale, size=(5, 2)
            )
        _require_finite(
            self.treatment_only_coefficients_,
            "Derived treatment-only coefficients must be finite.",
        )
        _require_finite(
            self.outcome_main_effects_,
            "Derived outcome main-effect coefficients must be finite.",
        )
        _require_finite(
            self.outcome_modifier_coefficients_,
            "Derived outcome modifier coefficients must be finite.",
        )
        self.confounding_strength_ = self.confounding_strength
        self.latent_support_bounds_ = self._latent_support_bounds()

        standard_normal = NormalDist()
        with np.errstate(over="ignore", invalid="ignore"):
            central_bounds = np.array(
                [standard_normal.inv_cdf(0.01), standard_normal.inv_cdf(0.99)]
            ) * float(self.treatment_noise_scale)
        _require_finite(
            central_bounds, "Derived central latent grid bounds must be finite."
        )
        self.latent_grid_bounds_ = np.array(
            [
                max(central_bounds[0], self.latent_support_bounds_[0]),
                min(central_bounds[1], self.latent_support_bounds_[1]),
            ]
        )
        if not self.latent_grid_bounds_[0] < self.latent_grid_bounds_[1]:
            raise ValueError(
                "Derived latent grid bounds must be finite and strictly ordered."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            self.hinge_knots_ = np.array(
                [
                    standard_normal.inv_cdf(1.0 / 3.0),
                    standard_normal.inv_cdf(2.0 / 3.0),
                ]
            ) * float(self.treatment_noise_scale)
        _require_finite(
            self.hinge_knots_, "Derived treatment hinge knots must be finite."
        )
        self.latent_basis_grid_ = np.linspace(*self.latent_grid_bounds_, 1001)
        self.raw_treatment_basis_ = self._raw_treatment_basis(self.latent_basis_grid_)
        self.treatment_basis_means_ = self.raw_treatment_basis_.mean(axis=0)
        self.treatment_basis_scales_ = self.raw_treatment_basis_.std(axis=0, ddof=0)
        self.treatment_basis_scales_[self.treatment_basis_scales_ == 0.0] = 1.0
        _require_finite(
            self.treatment_basis_means_, "Derived treatment basis means must be finite."
        )
        _require_finite(
            self.treatment_basis_scales_,
            "Derived treatment basis scales must be finite.",
        )
        self.reference_treatment_basis_ = self._standardized_treatment_basis(
            np.zeros(1)
        )[0]

        self._bias_grid_ = self._get_grid(31)
        self._source_grid_outcome_means_ = np.vstack(
            [
                self._predict_y(
                    X,
                    pl.DataFrame(
                        {
                            "t": np.full(
                                X.height,
                                intervention,
                            )
                        }
                    ),
                )
                for intervention in self._bias_grid_.get_column("t")
            ]
        )
        _require_finite(
            self._source_grid_outcome_means_,
            "Oracle outcome means on the continuous bias grid must be finite.",
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
        self.confounding_bias_, self.confounding_bias_ratio_ = self._bias_at_strength(
            X, self.confounding_strength_
        )
        _require_finite(
            [
                self.confounding_strength_,
                self.confounding_bias_,
                self.confounding_bias_ratio_,
            ],
            "Stored continuous confounding-bias attributes must be finite.",
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
            with np.errstate(over="ignore"):
                while not np.isfinite(np.exp(upper)):
                    upper = np.nextafter(upper, -np.inf)
            return np.array([lower, upper])

        domain_lower, domain_upper = map(float, self.treatment_domain)
        treatment_lower = np.nextafter(domain_lower, domain_upper)
        treatment_upper = np.nextafter(domain_upper, domain_lower)
        if treatment_lower >= treatment_upper:
            raise ValueError(
                "treatment_domain must contain at least two distinct representable "
                "interior treatment values."
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
        latent_bounds = np.array([lower, upper])
        treatment_bounds = self._forward_transform(latent_bounds)
        round_tripped_bounds = self._inverse_transform(treatment_bounds)
        round_tripped_treatments = self._forward_transform(round_tripped_bounds)
        if (
            not np.isfinite(latent_bounds).all()
            or not lower < upper
            or not np.isfinite(treatment_bounds).all()
            or not (
                domain_lower < treatment_bounds[0] < treatment_bounds[1] < domain_upper
            )
            or not np.isfinite(round_tripped_bounds).all()
            or not round_tripped_bounds[0] < round_tripped_bounds[1]
            or not np.array_equal(round_tripped_treatments, treatment_bounds)
        ):
            raise ValueError(
                "treatment_domain must yield strictly ordered latent bounds that "
                "round-trip to distinct representable interior treatments."
            )
        return round_tripped_bounds

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
        with np.errstate(over="ignore", invalid="ignore"):
            mean = strength * signal + self.treatment_only_strength * treatment_only
        _require_finite(mean, "Derived latent treatment means must be finite.")
        return mean

    def _sample_treatment(self, X, rng):
        randomized = rng.random(X.height) < self.randomized_weight
        location = np.where(randomized, 0.0, self._latent_mean(X))
        if self.treatment_domain == "real":
            with np.errstate(over="ignore", invalid="ignore"):
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
        _require_finite(latent, "Sampled latent treatments must be finite.")
        treatment = self._forward_transform(latent)
        if not self._valid_treatment(treatment).all():
            raise ValueError(
                "Sampled treatments must be finite and lie within treatment_domain."
            )
        return pl.DataFrame({"t": treatment})

    def _log_prob_at_strength(self, X, treatment, strength):
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
                self._latent_mean(X, strength)[valid_indices],
                self.treatment_noise_scale,
            )
            randomized = _normal_logpdf(latent, 0.0, self.treatment_noise_scale)
        else:
            confounded = _truncated_normal_logpdf(
                latent,
                self._latent_mean(X, strength)[valid_indices],
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

    def _log_prob(self, X, treatment):
        return self._log_prob_at_strength(X, treatment, self.confounding_strength_)

    def _bias_at_strength(self, X, strength):
        if self.randomized_weight == 1.0 or (
            strength == 0.0 and self.treatment_only_strength == 0.0
        ):
            return 0.0, 0.0

        distortions = []
        for intervention, outcome_means in zip(
            self._bias_grid_.get_column("t"), self._source_grid_outcome_means_
        ):
            treatment = pl.DataFrame({"t": np.full(X.height, intervention)})
            log_density = self._log_prob_at_strength(X, treatment, strength)
            weights = _normalized_log_weights(log_density)
            with np.errstate(over="ignore", invalid="ignore"):
                distortion = weights @ outcome_means - outcome_means.mean()
            _require_finite(
                distortion, "Continuous confounding distortions must be finite."
            )
            distortions.append(distortion)
        with np.errstate(over="ignore", invalid="ignore"):
            bias = float(np.sqrt(np.mean(np.square(distortions))))
            ratio = bias / self.baseline_standard_deviation_
        _require_finite(bias, "Continuous confounding bias must be finite.")
        _require_finite(ratio, "Continuous confounding bias ratio must be finite.")
        return bias, ratio

    def _raw_treatment_basis(self, latent):
        latent = np.asarray(latent, dtype=float).reshape(-1)
        with np.errstate(over="ignore", invalid="ignore"):
            basis = np.column_stack(
                (
                    latent,
                    latent**2,
                    np.sin(latent),
                    np.maximum(latent - self.hinge_knots_[0], 0.0),
                    np.maximum(latent - self.hinge_knots_[1], 0.0),
                )
            )
        _require_finite(basis, "Derived treatment basis must be finite.")
        return basis

    def _standardized_treatment_basis(self, latent):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            basis = (
                self._raw_treatment_basis(latent) - self.treatment_basis_means_
            ) / self.treatment_basis_scales_
        _require_finite(basis, "Derived treatment basis must be finite.")
        return basis

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
        with np.errstate(over="ignore", invalid="ignore"):
            modifier_effects = basis @ self.outcome_modifier_coefficients_
            interaction = np.einsum("ij,ij->i", modifier_effects, scores.modifiers)
            mean = mu0 + basis @ self.outcome_main_effects_ + interaction
        _require_finite(mean, "Oracle outcome means must be finite.")
        return mean

    def _sample_outcome(self, mean, X, treatment, rng):
        mean = np.asarray(mean, dtype=float)
        _require_finite(mean, "Outcome means passed to the sampler must be finite.")
        with np.errstate(over="ignore", invalid="ignore"):
            outcome = mean + rng.normal(scale=self.outcome_noise_scale, size=mean.shape)
        _require_finite(outcome, "Sampled outcomes must be finite.")
        return outcome

    def _get_grid(self, n):
        latent = np.linspace(*self.latent_grid_bounds_, n)
        treatment = self._forward_transform(latent)
        if not self._valid_treatment(treatment).all():
            raise ValueError(
                "Derived treatment grid must be finite and lie within treatment_domain."
            )
        return pl.DataFrame({"t": treatment})

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
