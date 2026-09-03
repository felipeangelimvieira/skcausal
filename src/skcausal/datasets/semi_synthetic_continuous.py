from typing import ClassVar

import numpy as np
import polars as pl
from scipy.special import expit, log_ndtr, logsumexp, ndtr, ndtri, ndtri_exp

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _log_mixture,
    _normal_logpdf,
    _validate_shared_dgp_parameters,
)

__all__ = ["ContinuousSemiSyntheticDataset"]

_SMALLEST_POSITIVE_FLOAT = np.nextafter(0.0, 1.0)
_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)
_NARROW_INTERVAL = np.sqrt(np.finfo(float).eps)
_QUADRATURE_TAIL = 6.0
_QUADRATURE_POINTS_PER_SCALE = 6
_QUADRATURE_MIN_POINTS = 65
_QUADRATURE_MAX_POINTS = 8193
_QUADRATURE_CHUNK_ELEMENTS = 2**21


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


def _is_narrow_interval(lower, upper):
    midpoint = 0.5 * lower + 0.5 * upper
    return (upper - lower) * np.maximum(1.0, np.abs(midpoint)) <= _NARROW_INTERVAL


def _log_interval_mass(lower, upper):
    """Return ``log(Phi(upper) - Phi(lower))`` stably for standardized bounds.

    Tails are evaluated on the side where the mass is small, and intervals
    narrower than the floating-point resolution use the midpoint density.
    """

    lower, upper = np.broadcast_arrays(
        np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )
    result = np.full(lower.shape, -np.inf, dtype=float)
    ordered = lower < upper
    if not ordered.any():
        return result

    lower = lower[ordered]
    upper = upper[ordered]
    narrow = np.isfinite(lower) & np.isfinite(upper) & _is_narrow_interval(lower, upper)
    flip = lower > 0.0
    high = np.where(flip, log_ndtr(-lower), log_ndtr(upper))
    low = np.where(flip, log_ndtr(-upper), log_ndtr(lower))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = high + np.log(-np.expm1(low - high))
    values = np.where(np.isneginf(low), high, values)
    midpoint = 0.5 * lower + 0.5 * upper
    with np.errstate(divide="ignore", invalid="ignore"):
        narrow_values = np.log(upper - lower) - 0.5 * midpoint**2 - _HALF_LOG_2PI
    result[ordered] = np.where(narrow, narrow_values, values)
    return result


def _interval_cdf(value, lower, upper, log_mass):
    """Return the CDF of a standard normal conditioned on ``[lower, upper]``."""

    value, lower, upper, log_mass = np.broadcast_arrays(
        *(np.asarray(item, dtype=float) for item in (value, lower, upper, log_mass))
    )
    clipped = np.clip(value, lower, upper)
    flip = lower > 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        forward = (ndtr(clipped) - ndtr(lower)) * np.exp(-log_mass)
        backward = 1.0 - (ndtr(-clipped) - ndtr(-upper)) * np.exp(-log_mass)
        uniform = (clipped - lower) / (upper - lower)
    narrow = np.isfinite(lower) & np.isfinite(upper) & _is_narrow_interval(lower, upper)
    result = np.where(narrow, uniform, np.where(flip, backward, forward))
    return np.clip(result, 0.0, 1.0)


def _sample_truncated_normal(location, scale, lower, upper, rng):
    """Draw exactly from a normal conditioned on ``[lower, upper]``."""

    location = np.asarray(location, dtype=float)
    uniforms = np.clip(rng.random(location.shape), _SMALLEST_POSITIVE_FLOAT, 1.0)
    standardized_lower = (lower - location) / scale
    standardized_upper = (upper - location) / scale
    log_mass = _log_interval_mass(standardized_lower, standardized_upper)
    flip = standardized_lower > 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        forward = ndtri_exp(
            np.logaddexp(log_ndtr(standardized_lower), np.log(uniforms) + log_mass)
        )
        backward = -ndtri_exp(
            np.logaddexp(log_ndtr(-standardized_upper), np.log1p(-uniforms) + log_mass)
        )
        uniform = standardized_lower + uniforms * (
            standardized_upper - standardized_lower
        )
    narrow = _is_narrow_interval(standardized_lower, standardized_upper)
    standardized = np.where(narrow, uniform, np.where(flip, backward, forward))
    standardized = np.clip(standardized, standardized_lower, standardized_upper)
    return location + scale * standardized


class ContinuousSemiSyntheticDataset(BaseSemiSyntheticDataset):
    r"""Semi-synthetic DGP with one scalar continuous treatment.

    The assignment is a mixture of a covariate-dependent latent Gaussian and a
    randomized latent Gaussian. The covariate-dependent latent mean is
    ``confounding_strength * s(x) + treatment_only_strength * z(x)``, where
    ``s`` is the standardized confounder contribution to the baseline outcome
    and ``z`` is a treatment-only score residualized against the outcome
    scores. For ``treatment_domain="real"``, the latent Gaussians are
    unconditioned and the released treatment equals the latent value. For
    ``"positive"`` and finite ``(lower, upper)`` domains, each latent Gaussian
    component is instead conditioned on the finite interval whose endpoints
    round-trip through the configured transform to distinct, representable,
    strictly interior floating-point treatments. Sampling and
    :meth:`log_prob` use this same conditioned law exactly.

    The oracle confounding bias is the root-mean-square gap between the naive
    conditional mean ``E[Y | A = a]`` and the causal mean ``E[Y(a)]``, averaged
    over the marginal treatment distribution, in units of the baseline-outcome
    standard deviation. ``get_grid`` spans the central 98% of that marginal.

    Parameters
    ----------
    real_dataset : BaseDataset
        Dataset supplying the covariate rows retained by this DGP.
    treatment_domain : {"real", "positive"} or tuple of float, default="real"
        Released treatment support. A tuple must contain finite ``lower < upper``
        bounds and at least two distinct representable interior float values.
    confounding_strength : float, default=1.0
        Nonnegative scale of the outcome-predictive assignment signal.
    target_confounding_bias : float or None, default=None
        Optional nonnegative normalized oracle-bias calibration target. The
        smallest strength reaching the target is frozen.
    treatment_only_strength : float, default=0.0
        Nonnegative scale of the residualized treatment-only predictors. These
        scores add assignment predictability that is orthogonal, to second
        order, to the outcome surface; because real covariates are dependent, a
        small residual confounding floor that grows roughly quadratically with
        this value remains at ``confounding_strength=0``.
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
        treatment_only_strength=0.0,
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
        _validate_shared_dgp_parameters(
            confounding_strength=confounding_strength,
            target_confounding_bias=target_confounding_bias,
            treatment_only_strength=treatment_only_strength,
            randomized_weight=randomized_weight,
            outcome_effect_scale=outcome_effect_scale,
            effect_heterogeneity_scale=effect_heterogeneity_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

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

    # -- structural preparation ------------------------------------------------

    def _prepare_dgp(self, X, rng):
        self._fit_causal_structure(X, rng)
        self.treatment_only_coefficients_ = rng.normal(scale=1.0 / np.sqrt(2.0), size=2)
        self.outcome_main_effects_ = rng.normal(scale=self.outcome_effect_scale, size=5)
        self.outcome_modifier_coefficients_ = rng.normal(
            scale=self.effect_heterogeneity_scale, size=(5, 2)
        )
        self.latent_support_bounds_ = self._latent_support_bounds()

        scale = float(self.treatment_noise_scale)
        central = ndtri(np.array([0.01, 0.99])) * scale
        self.latent_basis_bounds_ = np.array(
            [
                max(central[0], self.latent_support_bounds_[0]),
                min(central[1], self.latent_support_bounds_[1]),
            ]
        )
        if not np.isfinite(self.latent_basis_bounds_).all() or not (
            self.latent_basis_bounds_[0] < self.latent_basis_bounds_[1]
        ):
            raise ValueError(
                "Derived latent basis bounds must be finite and strictly ordered."
            )
        self.hinge_knots_ = ndtri(np.array([1.0 / 3.0, 2.0 / 3.0]))
        basis_grid = np.linspace(*self.latent_basis_bounds_, 1001)
        raw_basis = self._raw_treatment_basis(basis_grid)
        self.treatment_basis_means_ = raw_basis.mean(axis=0)
        self.treatment_basis_scales_ = raw_basis.std(axis=0, ddof=0)
        self.treatment_basis_scales_[self.treatment_basis_scales_ == 0.0] = 1.0
        self.reference_treatment_basis_ = self._standardized_treatment_basis(
            np.zeros(1)
        )[0]

        self._resolve_confounding_strength(
            self._bias_at_strength,
            confounding_strength=self.confounding_strength,
            target_confounding_bias=self.target_confounding_bias,
            randomized_weight=self.randomized_weight,
        )
        quantiles = self._latent_marginal_quantiles(
            np.array([0.01, 0.99]),
            self._latent_means(self.source_scores_, self.confounding_strength_),
        )
        self.latent_grid_bounds_ = np.array(
            [
                min(quantiles[0], self.latent_basis_bounds_[0]),
                max(quantiles[1], self.latent_basis_bounds_[1]),
            ]
        )
        if not np.isfinite(self.latent_grid_bounds_).all() or not (
            self.latent_grid_bounds_[0] < self.latent_grid_bounds_[1]
        ):
            raise ValueError(
                "Derived latent grid bounds must be finite and strictly ordered."
            )

    # -- treatment transforms --------------------------------------------------

    def _forward_transform(self, latent):
        latent = np.asarray(latent, dtype=float)
        if self.treatment_domain == "real":
            return latent.copy()
        if self.treatment_domain == "positive":
            return np.exp(latent)
        lower, upper = map(float, self.treatment_domain)
        unit = expit(latent)
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

    # -- assignment law --------------------------------------------------------

    def _latent_means(self, scores, strength):
        _, signal = self.baseline_surface_.evaluate(scores)
        treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
        with np.errstate(over="ignore", invalid="ignore"):
            mean = strength * signal + self.treatment_only_strength * treatment_only
        if not np.isfinite(mean).all():
            raise ValueError("Derived latent treatment means must be finite.")
        return mean

    def _latent_mean(self, X, strength=None):
        strength = self.confounding_strength_ if strength is None else strength
        return self._latent_means(self._scores(X), strength)

    def _component_log_masses(self, means):
        """Log normalizers of the conditioned confounded and randomized laws."""

        if self.treatment_domain == "real":
            return np.zeros_like(means), 0.0
        lower, upper = self.latent_support_bounds_
        scale = self.treatment_noise_scale
        confounded = _log_interval_mass(
            (lower - means) / scale, (upper - means) / scale
        )
        randomized = float(_log_interval_mass(lower / scale, upper / scale))
        return confounded, randomized

    def _latent_log_density(self, latent, means):
        """Mixture log-density of latent values broadcast against row means."""

        scale = self.treatment_noise_scale
        confounded_mass, randomized_mass = self._component_log_masses(means)
        confounded = _normal_logpdf(latent, means, scale) - confounded_mass
        randomized = _normal_logpdf(latent, 0.0, scale) - randomized_mass
        return _log_mixture(confounded, randomized, self.randomized_weight)

    def _sample_treatment(self, X, rng):
        means = self._latent_mean(X)
        randomized = rng.random(means.shape) < self.randomized_weight
        location = np.where(randomized, 0.0, means)
        if self.treatment_domain == "real":
            latent = location + rng.normal(
                scale=self.treatment_noise_scale, size=location.shape
            )
        else:
            latent = _sample_truncated_normal(
                location,
                self.treatment_noise_scale,
                *self.latent_support_bounds_,
                rng,
            )
        treatment = self._forward_transform(latent)
        if not self._valid_treatment(treatment).all():
            raise ValueError(
                "Sampled treatments must be finite and lie within treatment_domain."
            )
        return pl.DataFrame({"t": treatment})

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

        means = self._latent_mean(X)[valid_indices]
        log_density[valid_indices] = self._latent_log_density(
            latent, means
        ) + self._log_abs_inverse_jacobian(values[valid_indices])
        return log_density

    # -- marginal treatment distribution ---------------------------------------

    def _latent_marginal_cdf(self, latent, means):
        scale = self.treatment_noise_scale
        latent = np.asarray(latent, dtype=float)[:, None]
        if self.treatment_domain == "real":
            confounded = ndtr((latent - means[None, :]) / scale)
            randomized = ndtr(latent / scale)
        else:
            lower, upper = self.latent_support_bounds_
            confounded_mass, randomized_mass = self._component_log_masses(means)
            confounded = _interval_cdf(
                (latent - means[None, :]) / scale,
                ((lower - means) / scale)[None, :],
                ((upper - means) / scale)[None, :],
                confounded_mass[None, :],
            )
            randomized = _interval_cdf(
                latent / scale, lower / scale, upper / scale, randomized_mass
            )
        mixture = _log_mixture_probability(
            confounded.mean(axis=1), randomized[:, 0], self.randomized_weight
        )
        return mixture

    def _latent_marginal_range(self, means):
        scale = self.treatment_noise_scale
        lower = min(float(means.min()), 0.0) - _QUADRATURE_TAIL * scale
        upper = max(float(means.max()), 0.0) + _QUADRATURE_TAIL * scale
        return np.array(
            [
                max(lower, self.latent_support_bounds_[0]),
                min(upper, self.latent_support_bounds_[1]),
            ]
        )

    def _latent_marginal_quantiles(self, probabilities, means):
        lower, upper = self._latent_marginal_range(means)
        low = np.full(probabilities.shape, lower)
        high = np.full(probabilities.shape, upper)
        for _ in range(200):
            middle = 0.5 * low + 0.5 * high
            below = self._latent_marginal_cdf(middle, means) < probabilities
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
            if np.all(high - low <= 1e-12 * np.maximum(1.0, np.abs(middle))):
                break
        return 0.5 * low + 0.5 * high

    def _latent_quadrature(self, means):
        """Latent grid and normalized marginal-density weights."""

        lower, upper = self._latent_marginal_range(means)
        spacing = self.treatment_noise_scale / _QUADRATURE_POINTS_PER_SCALE
        n_points = int(
            np.clip(
                np.ceil((upper - lower) / spacing) + 1,
                _QUADRATURE_MIN_POINTS,
                _QUADRATURE_MAX_POINTS,
            )
        )
        grid = np.linspace(lower, upper, n_points)
        trapezoid = np.full(n_points, grid[1] - grid[0])
        trapezoid[[0, -1]] *= 0.5
        return grid, trapezoid

    def _bias_at_strength(self, strength):
        if self.randomized_weight == 1.0 or (
            strength == 0.0 and self.treatment_only_strength == 0.0
        ):
            return 0.0, 0.0

        scores = self.source_scores_
        means = self._latent_means(scores, strength)
        grid, trapezoid = self._latent_quadrature(means)
        basis = (
            self._standardized_treatment_basis(grid) - self.reference_treatment_basis_
        )
        modifier_effects = basis @ self.outcome_modifier_coefficients_
        mu0 = self.source_mu0_
        modifiers = scores.modifiers
        n_rows = means.size

        density = np.empty(grid.size)
        weighted_mu0 = np.empty(grid.size)
        weighted_modifiers = np.empty((grid.size, modifiers.shape[1]))
        chunk = max(1, _QUADRATURE_CHUNK_ELEMENTS // n_rows)
        for start in range(0, grid.size, chunk):
            stop = min(start + chunk, grid.size)
            log_density = self._latent_log_density(
                grid[start:stop, None], means[None, :]
            )
            log_total = logsumexp(log_density, axis=1)
            supported = np.isfinite(log_total)
            weights = np.zeros_like(log_density)
            weights[supported] = np.exp(
                log_density[supported] - log_total[supported, None]
            )
            density[start:stop] = np.where(
                supported, np.exp(log_total - np.log(n_rows)), 0.0
            )
            weighted_mu0[start:stop] = np.where(supported, weights @ mu0, mu0.mean())
            weighted_modifiers[start:stop] = np.where(
                supported[:, None], weights @ modifiers, modifiers.mean(axis=0)
            )

        distortion = (weighted_mu0 - mu0.mean()) + np.einsum(
            "ij,ij->i", modifier_effects, weighted_modifiers - modifiers.mean(axis=0)
        )
        quadrature = density * trapezoid
        total = quadrature.sum()
        with np.errstate(over="ignore", invalid="ignore"):
            bias = float(np.sqrt(np.sum(quadrature * distortion**2) / total))
        if not np.isfinite(total) or total <= 0.0 or not np.isfinite(bias):
            raise ValueError(
                "Continuous confounding bias is not representable at this "
                "configuration: the latent assignment law has no finite log "
                "weight or the naive-curve distortion overflows."
            )
        return bias, bias / self.baseline_standard_deviation_

    # -- outcome surface -------------------------------------------------------

    def _raw_treatment_basis(self, latent):
        """Bounded-growth basis in standardized latent units.

        The terms are linear, a saturating curvature ``1 - exp(-v^2 / 2)``, a
        sinusoid, and two hinges at the tertiles of the randomized latent law,
        so the dose-response grows at most linearly in the latent treatment.
        """

        standardized = np.asarray(latent, dtype=float).reshape(-1) / float(
            self.treatment_noise_scale
        )
        with np.errstate(over="ignore", invalid="ignore"):
            basis = np.column_stack(
                (
                    standardized,
                    -np.expm1(-0.5 * standardized**2),
                    np.sin(standardized),
                    np.maximum(standardized - self.hinge_knots_[0], 0.0),
                    np.maximum(standardized - self.hinge_knots_[1], 0.0),
                )
            )
        if not np.isfinite(basis).all():
            raise ValueError("Derived treatment basis must be finite.")
        return basis

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
        return self._outcome_mean(
            self._scores(X),
            self._treatment_basis(treatment),
            self.outcome_main_effects_,
            self.outcome_modifier_coefficients_,
        )

    def _sample_outcome(self, mean, X, treatment, rng):
        return self._gaussian_outcome(mean, self.outcome_noise_scale, rng)

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


def _log_mixture_probability(confounded, randomized, weight):
    if weight == 0.0:
        return confounded
    if weight == 1.0:
        return randomized
    return (1.0 - weight) * confounded + weight * randomized
