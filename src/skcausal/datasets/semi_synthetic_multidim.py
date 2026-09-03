"""Multi-dimensional semi-synthetic DGP driven by a real-outcome prognostic score."""

import itertools

import numpy as np
import polars as pl
from scipy.special import log_ndtr, logsumexp, ndtri

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _bisect_marginal_quantiles,
    _gaussian_mixture_marginal_cdf,
    _log_mixture,
    _validate_nonnegative,
    _validate_shared_dgp_parameters,
)

__all__ = ["MultidimSemiSyntheticDataset"]

_LOG_2PI = np.log(2.0 * np.pi)
_BIAS_DESIGN_POINTS = 1024
_BIAS_CHUNK_ELEMENTS = 2**21
_MARGINAL_TAIL = 6.0
_BASIS_GRID_POINTS = 1001


def _per_component(name, value, n_components, *, default=None):
    """Broadcast a scalar to every component or validate a length-k sequence."""

    if value is None:
        value = default
    if isinstance(value, str) or np.ndim(value) == 0:
        return [value] * n_components
    values = list(value)
    if len(values) != n_components:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length "
            f"n_treatments={n_components}."
        )
    return values


def _validate_loadings(value, n_components):
    loadings = np.asarray(
        _per_component("confounding_loadings", value, n_components, default=1.0),
        dtype=float,
    )
    if loadings.shape != (n_components,) or not np.isfinite(loadings).all():
        raise ValueError("confounding_loadings must be finite.")
    return loadings


def _validate_levels(value, n_components):
    levels = _per_component("n_levels", value, n_components)
    for entry in levels:
        if entry is None:
            continue
        if isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise TypeError("n_levels entries must be None or integers.")
        if entry < 2:
            raise ValueError("n_levels entries must be at least 2.")
    return [None if entry is None else int(entry) for entry in levels]


def _validate_targets(value, n_components):
    targets = _per_component("target_columns", value, n_components)
    for entry in targets:
        if entry is not None and not isinstance(entry, str):
            raise TypeError("target_columns entries must be None or column names.")
    named = [entry for entry in targets if entry is not None]
    if len(set(named)) != len(named):
        raise ValueError("target_columns must not repeat a column.")
    return targets


def _exchangeable_correlation(n_components, correlation):
    matrix = np.full((n_components, n_components), float(correlation))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _interval_log_mass(lower, upper):
    """``log(Phi(upper) - Phi(lower))`` evaluated on the side with small mass.

    Any pair with ``lower >= upper`` (including equal or equal-infinite
    bounds) has zero mass and returns ``-inf``.
    """

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        left = log_ndtr(upper) + np.log1p(-np.exp(log_ndtr(lower) - log_ndtr(upper)))
        right = log_ndtr(-lower) + np.log1p(
            -np.exp(log_ndtr(-upper) - log_ndtr(-lower))
        )
        selected = np.where(upper <= 0.0, left, right)
    return np.where(lower < upper, selected, -np.inf)


def _gaussian_logpdf(residual, cholesky):
    """Log-density of the trailing axis of ``residual`` under ``N(0, L L')``."""

    residual = np.asarray(residual, dtype=float)
    dimension = residual.shape[-1]
    if dimension == 0:
        return np.zeros(residual.shape[:-1])
    whitened = residual @ np.linalg.inv(cholesky).T
    log_determinant = float(np.log(np.diag(cholesky)).sum())
    return (
        -0.5 * np.sum(whitened**2, axis=-1)
        - log_determinant
        - 0.5 * dimension * _LOG_2PI
    )


def _conditional_gaussian(covariance, continuous, categorical):
    """Regression matrix and covariance of the categorical block given the continuous block."""

    continuous = np.asarray(continuous, dtype=int)
    categorical = np.asarray(categorical, dtype=int)
    if categorical.size == 0:
        return np.zeros((0, continuous.size)), np.zeros((0, 0))
    cov_dd = covariance[np.ix_(categorical, categorical)]
    if continuous.size == 0:
        return np.zeros((categorical.size, 0)), cov_dd
    cov_cc = covariance[np.ix_(continuous, continuous)]
    cov_dc = covariance[np.ix_(categorical, continuous)]
    regression = np.linalg.solve(cov_cc, cov_dc.T).T
    return regression, cov_dd - regression @ cov_dc.T


def _numeric_column(frame, name, label):
    series = frame.get_column(name)
    if not series.dtype.is_numeric():
        raise ValueError(f"The {label} must be numeric.")
    values = series.cast(pl.Float64).to_numpy().astype(float)
    finite = values[np.isfinite(values)]
    if np.unique(finite).size < 2:
        raise ValueError(f"The {label} must have at least two distinct finite values.")
    return values


class MultidimSemiSyntheticDataset(BaseSemiSyntheticDataset):
    r"""Semi-synthetic DGP with a vector of continuous and categorical treatments.

    A ridge prognostic score fitted on the real outcome ``y`` is the
    confounding backbone: it enters the baseline outcome with unit weight and
    the latent mean of component ``j`` with loading ``confounding_loadings[j]``
    times ``confounding_strength``. Components share exchangeable latent noise
    with correlation ``treatment_correlation``. A component may additionally
    mimic a numeric covariate named in ``target_columns``; that column is
    removed from the released ``X`` and its ridge prediction enters both the
    baseline outcome and that component's assignment. A component with
    ``n_levels`` set is released as ordered levels ``"0"`` to ``"K-1"`` by
    thresholding its latent at equal-mass marginal quantiles.

    Parameters
    ----------
    real_dataset : BaseDataset
        Source of covariates and of the outcome used for the prognostic score.
    n_treatments : int, default=2
        Number of treatment components, released as ``t_0`` to ``t_{k-1}``.
    confounding_loadings : float or sequence, default=None
        Per-component loading on the prognostic score; ``None`` means ``1.0``.
    n_levels : int, sequence, or None, default=None
        Per-component level count for discretized components; ``None`` keeps
        a component continuous.
    target_columns : str, sequence, or None, default=None
        Per-component numeric covariate to mimic; ``None`` mimics nothing.
    treatment_correlation : float, default=0.0
        Exchangeable latent noise correlation in ``[0, 1)``. Must be zero when
        two or more components are categorical.
    confounding_strength : float, default=1.0
        Nonnegative scale of the outcome-predictive assignment signal.
    target_confounding_bias : float or None, default=None
        Optional nonnegative normalized oracle-bias calibration target. The
        bias metric need not be monotone in ``confounding_strength``, so the
        frozen strength is the smallest crossing bracketed by the search
        ladder; a target that falls inside a local bump may therefore resolve
        to a larger strength than the first crossing of the underlying curve.
        The metric itself is a fixed 1024-point Monte Carlo estimate under the
        marginal treatment law, so its resolution is of order ``1/sqrt(1024)``
        relative.
    treatment_only_strength : float, default=0.0
        Nonnegative scale of the residualized treatment-only predictors.
    randomized_weight : float, default=0.1
        Weight in ``[0, 1]`` of the covariate-independent mixture component.
    treatment_noise_scale : float, default=1.0
        Strictly positive latent noise standard deviation shared by components.
    outcome_effect_scale : float, default=1.0
        Nonnegative scale of treatment main-effect coefficients.
    effect_heterogeneity_scale : float, default=0.5
        Nonnegative scale of treatment-by-covariate coefficients.
    treatment_interaction_scale : float, default=0.5
        Nonnegative scale of cross-component effect coefficients.
    outcome_noise_scale : float, default=1.0
        Nonnegative Gaussian outcome-noise standard deviation.
    random_state : int, default=42
        Seed controlling frozen structure and the initial sample.
    """

    def __init__(
        self,
        real_dataset,
        n_treatments=2,
        confounding_loadings=None,
        n_levels=None,
        target_columns=None,
        treatment_correlation=0.0,
        confounding_strength=1.0,
        target_confounding_bias=None,
        treatment_only_strength=0.0,
        randomized_weight=0.1,
        treatment_noise_scale=1.0,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        treatment_interaction_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(n_treatments, bool) or not isinstance(
            n_treatments, (int, np.integer)
        ):
            raise TypeError("n_treatments must be an integer.")
        if n_treatments < 1:
            raise ValueError("n_treatments must be at least 1.")
        loadings = _validate_loadings(confounding_loadings, n_treatments)
        levels = _validate_levels(n_levels, n_treatments)
        targets = _validate_targets(target_columns, n_treatments)
        correlation = float(treatment_correlation)
        if not np.isfinite(correlation) or not 0.0 <= correlation < 1.0:
            raise ValueError("treatment_correlation must lie in [0, 1).")
        if correlation > 0.0 and sum(entry is not None for entry in levels) >= 2:
            raise ValueError(
                "treatment_correlation must be zero when two or more components "
                "are categorical."
            )
        if not np.isfinite(float(treatment_noise_scale)) or (
            float(treatment_noise_scale) <= 0.0
        ):
            raise ValueError("treatment_noise_scale must be positive.")
        _validate_nonnegative(
            "treatment_interaction_scale", treatment_interaction_scale
        )
        _validate_shared_dgp_parameters(
            confounding_strength=confounding_strength,
            target_confounding_bias=target_confounding_bias,
            treatment_only_strength=treatment_only_strength,
            randomized_weight=randomized_weight,
            outcome_effect_scale=outcome_effect_scale,
            effect_heterogeneity_scale=effect_heterogeneity_scale,
            outcome_noise_scale=outcome_noise_scale,
        )
        self.n_treatments = n_treatments
        self.confounding_loadings = confounding_loadings
        self.n_levels = n_levels
        self.target_columns = target_columns
        self.treatment_correlation = treatment_correlation
        self.confounding_strength = confounding_strength
        self.target_confounding_bias = target_confounding_bias
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.treatment_noise_scale = treatment_noise_scale
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.treatment_interaction_scale = treatment_interaction_scale
        self.outcome_noise_scale = outcome_noise_scale
        self._validated_loadings = loadings
        self._validated_levels = levels
        self._validated_targets = targets
        super().__init__(real_dataset=real_dataset, random_state=random_state)

    # -- source split ----------------------------------------------------------

    def _split_source(self, X, treatment, outcome):
        outcome = self._coerce_backend_frame(outcome, backend="polars")
        if outcome.width != 1:
            raise ValueError("real_dataset must provide exactly one outcome column.")
        self._source_outcome = _numeric_column(
            outcome, outcome.columns[0], "source outcome"
        )
        feature_names = [name for name in self._validated_targets if name is not None]
        self._source_features = {}
        for name in feature_names:
            if name not in X.columns:
                raise ValueError(
                    f"target column {name!r} is not a covariate of real_dataset."
                )
            self._source_features[name] = _numeric_column(
                X, name, f"target column {name!r}"
            )
        released = X.drop(feature_names)
        if released.width == 0:
            raise ValueError(
                "At least one covariate must remain after removing target_columns."
            )
        return released

    # -- structural preparation ------------------------------------------------

    def _prepare_dgp(self, X, rng):
        n_components = int(self.n_treatments)
        self.treatment_columns_ = [f"t_{index}" for index in range(n_components)]
        self.n_levels_ = list(self._validated_levels)
        self.target_columns_ = list(self._validated_targets)
        self.confounding_loadings_ = self._validated_loadings.copy()
        self.continuous_indices_ = np.array(
            [j for j, levels in enumerate(self.n_levels_) if levels is None], dtype=int
        )
        self.categorical_indices_ = np.array(
            [j for j, levels in enumerate(self.n_levels_) if levels is not None],
            dtype=int,
        )
        self.column_types = {
            name: "continuous" if levels is None else "categorical"
            for name, levels in zip(self.treatment_columns_, self.n_levels_)
        }
        self.treatment_levels_ = {
            int(j): [str(level) for level in range(self.n_levels_[j])]
            for j in self.categorical_indices_
        }

        self.feature_names_ = [
            name for name in self.target_columns_ if name is not None
        ]
        self.feature_coefficients_ = rng.normal(size=len(self.feature_names_))
        fitted = np.column_stack(
            [self._source_outcome]
            + [self._source_features[name] for name in self.feature_names_]
        )
        scores = self._fit_causal_structure(
            X,
            rng,
            fitted_confounders=fitted,
            confounder_coefficients=np.concatenate(([1.0], self.feature_coefficients_)),
        )
        r2 = self.score_map_.fitted_confounder_r2
        self.prognostic_fit_r2_ = float(r2[0])
        self.feature_fit_r2_ = dict(zip(self.feature_names_, r2[1:].tolist()))
        self.feature_assignment_matrix_ = np.zeros(
            (len(self.feature_names_), n_components)
        )
        self.released_offsets_ = np.zeros(n_components)
        self.released_scales_ = np.ones(n_components)
        for j, name in enumerate(self.target_columns_):
            if name is None:
                continue
            position = self.feature_names_.index(name)
            self.feature_assignment_matrix_[position, j] = self.feature_coefficients_[
                position
            ]
            if self.n_levels_[j] is None:
                values = self._source_features[name]
                finite = values[np.isfinite(values)]
                self.released_offsets_[j] = float(finite.mean())
                self.released_scales_[j] = float(finite.std(ddof=0))

        self.treatment_only_coefficients_ = rng.normal(
            scale=1.0 / np.sqrt(2.0), size=(2, n_components)
        )
        self.noise_correlation_matrix_ = _exchangeable_correlation(
            n_components, self.treatment_correlation
        )
        self.noise_covariance_ = (
            float(self.treatment_noise_scale) ** 2 * self.noise_correlation_matrix_
        )
        self._noise_cholesky = np.linalg.cholesky(self.noise_covariance_)
        continuous, categorical = self.continuous_indices_, self.categorical_indices_
        self._continuous_cholesky = (
            np.linalg.cholesky(self.noise_covariance_[np.ix_(continuous, continuous)])
            if continuous.size
            else np.zeros((0, 0))
        )
        self._conditional_regression, conditional_covariance = _conditional_gaussian(
            self.noise_covariance_, continuous, categorical
        )
        self._conditional_scales = np.sqrt(np.diag(conditional_covariance))

        self.latent_basis_bounds_ = ndtri(np.array([0.01, 0.99]))
        self.hinge_knots_ = ndtri(np.array([1.0 / 3.0, 2.0 / 3.0]))
        raw = self._raw_continuous_basis(
            np.linspace(*self.latent_basis_bounds_, _BASIS_GRID_POINTS)
        )
        self.basis_means_ = raw.mean(axis=0)
        self.basis_scales_ = raw.std(axis=0, ddof=0)
        self.basis_scales_[self.basis_scales_ == 0.0] = 1.0
        self.reference_basis_ = (
            (self._raw_continuous_basis(np.zeros(1)) - self.basis_means_)
            / self.basis_scales_
        )[0]
        self.basis_sizes_ = [
            5 if levels is None else levels - 1 for levels in self.n_levels_
        ]
        self.interaction_sizes_ = [
            1 if levels is None else levels - 1 for levels in self.n_levels_
        ]
        total = int(sum(self.basis_sizes_))
        self.outcome_main_effects_ = rng.normal(
            scale=self.outcome_effect_scale, size=total
        )
        self.outcome_modifier_coefficients_ = rng.normal(
            scale=self.effect_heterogeneity_scale, size=(total, 2)
        )
        self.interaction_coefficients_ = {
            (j, l): rng.normal(
                scale=self.treatment_interaction_scale,
                size=(self.interaction_sizes_[j], self.interaction_sizes_[l]),
            )
            for j in range(n_components)
            for l in range(j + 1, n_components)
        }

        self._bias_rows = rng.integers(self.n, size=_BIAS_DESIGN_POINTS)
        self._bias_noise = (
            rng.standard_normal((_BIAS_DESIGN_POINTS, n_components))
            @ self._noise_cholesky.T
        )
        self._bias_uniforms = rng.random(_BIAS_DESIGN_POINTS)
        combinations = list(
            itertools.product(*[range(self.n_levels_[j]) for j in categorical])
        )
        self._level_combinations = np.array(combinations, dtype=int).reshape(
            len(combinations), categorical.size
        )

        self._resolve_confounding_strength(
            self._bias_at_strength,
            confounding_strength=self.confounding_strength,
            target_confounding_bias=self.target_confounding_bias,
            randomized_weight=self.randomized_weight,
        )
        means = self._latent_means(scores, self.confounding_strength_)
        self.cut_points_ = self._cut_points(means)
        self.latent_grid_bounds_ = self._latent_grid_bounds(means)

    # -- assignment law --------------------------------------------------------

    def _latent_means(self, scores, strength):
        confounders = scores.confounders
        signal = confounders[:, :1] * self.confounding_loadings_[None, :]
        if self.feature_names_:
            signal = signal + confounders[:, 1:] @ self.feature_assignment_matrix_
        treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
        means = strength * signal + self.treatment_only_strength * treatment_only
        if not np.isfinite(means).all():
            raise ValueError("Derived latent treatment means must be finite.")
        return means

    def _marginal_quantiles(self, column, probabilities):
        scale = float(self.treatment_noise_scale)
        lower = min(float(column.min()), 0.0) - _MARGINAL_TAIL * scale
        upper = max(float(column.max()), 0.0) + _MARGINAL_TAIL * scale
        return _bisect_marginal_quantiles(
            lambda value: _gaussian_mixture_marginal_cdf(
                value, column, scale, self.randomized_weight
            ),
            probabilities,
            lower,
            upper,
        )

    def _cut_points(self, means):
        return {
            int(j): self._marginal_quantiles(
                means[:, j], np.arange(1, self.n_levels_[j]) / self.n_levels_[j]
            )
            for j in self.categorical_indices_
        }

    def _log_density(self, latent_continuous, levels, means, cut_points):
        """Joint mixture log-density of continuous latents and categorical levels.

        Leading dimensions of ``latent_continuous`` ``(..., k_C)``, ``levels``
        ``(..., k_D)`` and ``means`` ``(..., k)`` broadcast against each other.
        """

        continuous, categorical = self.continuous_indices_, self.categorical_indices_

        def component(location):
            residual = latent_continuous - location[..., continuous]
            value = _gaussian_logpdf(residual, self._continuous_cholesky)
            if categorical.size:
                conditional = (
                    location[..., categorical]
                    + residual @ self._conditional_regression.T
                )
                for position, j in enumerate(categorical):
                    bounds = np.concatenate(([-np.inf], cut_points[int(j)], [np.inf]))
                    index = levels[..., position]
                    scale = self._conditional_scales[position]
                    value = value + _interval_log_mass(
                        (bounds[index] - conditional[..., position]) / scale,
                        (bounds[index + 1] - conditional[..., position]) / scale,
                    )
            return value

        # The randomized component has no row dependence, so evaluate it once per
        # design point and let ``_log_mixture`` broadcast it against the confounded
        # term rather than materializing an identical value for every row.
        randomized_location = np.zeros((1,) * (means.ndim - 1) + (means.shape[-1],))
        return _log_mixture(
            component(means), component(randomized_location), self.randomized_weight
        )

    def _release(self, latent):
        columns = {}
        for j, name in enumerate(self.treatment_columns_):
            if self.n_levels_[j] is None:
                columns[name] = (
                    self.released_offsets_[j] + self.released_scales_[j] * latent[:, j]
                )
            else:
                index = np.searchsorted(self.cut_points_[j], latent[:, j], side="left")
                columns[name] = np.asarray(self.treatment_levels_[j], dtype=str)[index]
        return pl.DataFrame(columns)

    def _sample_treatment(self, X, rng):
        means = self._latent_means(self._scores(X), self.confounding_strength_)
        randomized = rng.random(means.shape[0]) < self.randomized_weight
        location = np.where(randomized[:, None], 0.0, means)
        latent = location + rng.standard_normal(means.shape) @ self._noise_cholesky.T
        return self._release(latent)

    def _parse_treatment(self, treatment, *, allow_unknown):
        """Return standardized continuous latents, level indices, and a validity mask."""

        n_rows = treatment.height
        continuous = np.empty((n_rows, self.continuous_indices_.size))
        valid = np.ones(n_rows, dtype=bool)
        for position, j in enumerate(self.continuous_indices_):
            values = (
                treatment.get_column(self.treatment_columns_[j])
                .cast(pl.Float64)
                .to_numpy()
                .astype(float)
            )
            valid &= np.isfinite(values)
            continuous[:, position] = (
                values - self.released_offsets_[j]
            ) / self.released_scales_[j]
        levels = np.empty((n_rows, self.categorical_indices_.size), dtype=int)
        for position, j in enumerate(self.categorical_indices_):
            labels = treatment.get_column(self.treatment_columns_[j]).cast(pl.Utf8)
            index = (
                labels.replace_strict(
                    {
                        level: index
                        for index, level in enumerate(self.treatment_levels_[j])
                    },
                    default=-1,
                    return_dtype=pl.Int64,
                )
                .fill_null(-1)
                .to_numpy()
            )
            levels[:, position] = index
            valid &= index >= 0
        if not allow_unknown and not valid.all():
            raise ValueError(
                "Treatment values must be finite continuous values and known "
                "categorical levels."
            )
        return continuous, levels, valid

    def _log_prob(self, X, treatment):
        continuous, levels, valid = self._parse_treatment(treatment, allow_unknown=True)
        log_density = np.full(treatment.height, -np.inf)
        if not valid.any():
            return log_density
        means = self._latent_means(self._scores(X), self.confounding_strength_)
        jacobian = -float(np.log(self.released_scales_[self.continuous_indices_]).sum())
        log_density[valid] = (
            self._log_density(
                continuous[valid], levels[valid], means[valid], self.cut_points_
            )
            + jacobian
        )
        return log_density

    # -- outcome surface -------------------------------------------------------

    def _raw_continuous_basis(self, standardized):
        v = np.asarray(standardized, dtype=float).reshape(-1)
        return np.column_stack(
            (
                v,
                -np.expm1(-0.5 * v**2),
                np.sin(v),
                np.maximum(v - self.hinge_knots_[0], 0.0),
                np.maximum(v - self.hinge_knots_[1], 0.0),
            )
        )

    def _component_bases(self, latent_continuous, levels):
        """Stacked outcome basis ``(n, P)`` and per-component cross-term features."""

        scale = float(self.treatment_noise_scale)
        bases, features = [], []
        continuous_position = categorical_position = 0
        for j in range(int(self.n_treatments)):
            if self.n_levels_[j] is None:
                v = latent_continuous[:, continuous_position] / scale
                continuous_position += 1
                bases.append(
                    (self._raw_continuous_basis(v) - self.basis_means_)
                    / self.basis_scales_
                    - self.reference_basis_
                )
                features.append(np.tanh(v)[:, None])
            else:
                index = levels[:, categorical_position]
                categorical_position += 1
                indicators = (
                    index[:, None] == np.arange(1, self.n_levels_[j])[None, :]
                ).astype(float)
                bases.append(indicators)
                features.append(indicators)
        return np.column_stack(bases), features

    def _interaction_effects(self, features, n_rows):
        total = np.zeros(n_rows)
        for (j, l), coefficients in self.interaction_coefficients_.items():
            total = total + np.einsum(
                "ip,pq,iq->i", features[j], coefficients, features[l]
            )
        return total

    def _predict_y(self, X, treatment):
        continuous, levels, _ = self._parse_treatment(treatment, allow_unknown=False)
        basis, features = self._component_bases(continuous, levels)
        mean = self._outcome_mean(
            self._scores(X),
            basis,
            self.outcome_main_effects_,
            self.outcome_modifier_coefficients_,
        )
        return mean + self._interaction_effects(features, continuous.shape[0])

    def _sample_outcome(self, mean, X, treatment, rng):
        return self._gaussian_outcome(mean, self.outcome_noise_scale, rng)

    # -- confounding bias ------------------------------------------------------

    def _bias_at_strength(self, strength):
        if self.randomized_weight == 1.0 or (
            strength == 0.0 and self.treatment_only_strength == 0.0
        ):
            return 0.0, 0.0

        scores = self.source_scores_
        means = self._latent_means(scores, strength)
        cut_points = self._cut_points(means)
        location = np.where(
            self._bias_uniforms[:, None] < self.randomized_weight,
            0.0,
            means[self._bias_rows],
        )
        latent = (location + self._bias_noise)[:, self.continuous_indices_]
        if self.continuous_indices_.size == 0:
            # With no continuous components every design point has an identical
            # (empty) continuous latent, so one point stands in for all 1024.
            latent = latent[:1]
        mu0 = self.source_mu0_
        modifiers = scores.modifiers
        n_rows = means.shape[0]
        n_points = latent.shape[0]
        combinations = self._level_combinations
        marginal = np.empty((n_points, len(combinations)))
        distortion = np.empty((n_points, len(combinations)))
        chunk = max(1, _BIAS_CHUNK_ELEMENTS // n_rows)
        for combo_index, combination in enumerate(combinations):
            levels = np.broadcast_to(combination, (n_points, combination.size))
            basis, _ = self._component_bases(latent, levels)
            modifier_effects = basis @ self.outcome_modifier_coefficients_
            for start in range(0, n_points, chunk):
                stop = min(start + chunk, n_points)
                log_density = self._log_density(
                    latent[start:stop, None, :],
                    levels[start:stop, None, :],
                    means[None, :, :],
                    cut_points,
                )
                log_total = logsumexp(log_density, axis=1)
                supported = np.isfinite(log_total)
                weights = np.zeros_like(log_density)
                weights[supported] = np.exp(
                    log_density[supported] - log_total[supported, None]
                )
                marginal[start:stop, combo_index] = np.where(
                    supported, np.exp(log_total - np.log(n_rows)), 0.0
                )
                weighted_mu0 = np.where(supported, weights @ mu0, mu0.mean())
                weighted_modifiers = np.where(
                    supported[:, None], weights @ modifiers, modifiers.mean(axis=0)
                )
                distortion[start:stop, combo_index] = (
                    weighted_mu0 - mu0.mean()
                ) + np.einsum(
                    "ij,ij->i",
                    modifier_effects[start:stop],
                    weighted_modifiers - modifiers.mean(axis=0),
                )
        total = marginal.sum(axis=1, keepdims=True)
        if not np.isfinite(total).all() or (total <= 0.0).any():
            raise ValueError(
                "Multidimensional confounding bias is not representable at this "
                "configuration: a design point has no finite assignment density."
            )
        probabilities = marginal / total
        bias = float(np.sqrt(np.mean(np.sum(probabilities * distortion**2, axis=1))))
        if not np.isfinite(bias):
            raise ValueError("Multidimensional confounding bias must be finite.")
        return bias, bias / self.baseline_standard_deviation_

    # -- grid ------------------------------------------------------------------

    def _latent_grid_bounds(self, means):
        central = self.latent_basis_bounds_ * float(self.treatment_noise_scale)
        bounds = np.empty((self.continuous_indices_.size, 2))
        for position, j in enumerate(self.continuous_indices_):
            quantiles = self._marginal_quantiles(means[:, j], np.array([0.01, 0.99]))
            bounds[position] = (
                min(quantiles[0], central[0]),
                max(quantiles[1], central[1]),
            )
        return bounds

    def _get_grid(self, n):
        n_continuous = self.continuous_indices_.size
        points = max(2, round(n ** (1.0 / n_continuous))) if n_continuous else 0
        grid = None
        for j, name in enumerate(self.treatment_columns_):
            if self.n_levels_[j] is None:
                position = int(np.flatnonzero(self.continuous_indices_ == j)[0])
                latent = np.linspace(*self.latent_grid_bounds_[position], points)
                frame = pl.DataFrame(
                    {
                        name: self.released_offsets_[j]
                        + self.released_scales_[j] * latent
                    }
                )
            else:
                frame = pl.DataFrame({name: self.treatment_levels_[j]})
            grid = frame if grid is None else grid.join(frame, how="cross")
        return grid

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
                "n_treatments": 3,
                "confounding_loadings": [1.0, 0.5, 0.0],
                "n_levels": [None, None, 3],
                "target_columns": [None, "x2", None],
                "treatment_correlation": 0.4,
                "random_state": 8,
            },
        ]
