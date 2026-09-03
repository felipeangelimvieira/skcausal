from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skcausal.datasets.base import BaseDataset, BaseSyntheticDataset

__all__ = ["BaseSemiSyntheticDataset"]

_LOG_2PI = np.log(2.0 * np.pi)
_MISSING_CATEGORY = "__missing__"
_ROLE_DIMENSIONS = {
    "confounders": 3,
    "prognostic": 2,
    "treatment_only": 2,
    "modifiers": 2,
}
_OUTCOME_ROLES = ("confounders", "prognostic", "modifiers")
_STRENGTH_LADDER = 0.05 * 2.0 ** np.arange(11)


@dataclass(frozen=True)
class _CausalScores:
    confounders: np.ndarray
    prognostic: np.ndarray
    treatment_only: np.ndarray
    modifiers: np.ndarray


@dataclass
class _CausalScoreMap:
    """Frozen map from raw covariates to standardized causal role scores.

    The treatment-only scores are residualized against a polynomial design of
    the outcome-relevant scores so that, to the order of that design, they
    predict assignment without predicting potential outcomes.
    """

    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    encoder: ColumnTransformer
    pair_indices: np.ndarray
    projections: dict[str, np.ndarray]
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    residualization_order: int
    residualization: np.ndarray
    fitted_confounder_r2: np.ndarray | None = None

    def transform(self, X):
        frame = _prepare_covariate_frame(X, self.columns, self.categorical_columns)
        encoded = np.asarray(self.encoder.transform(frame), dtype=float)
        library = _nonlinear_library(encoded, self.pair_indices)
        raw = {name: library @ self.projections[name] for name in self.projections}
        return self._scores_from_raw(raw)

    def _scores_from_raw(self, raw):
        standardized = {
            name: (raw[name] - self.means[name]) / self.scales[name]
            for name in _OUTCOME_ROLES
        }
        design = _residual_design(
            np.column_stack([standardized[name] for name in _OUTCOME_ROLES]),
            self.residualization_order,
        )
        treatment_only = raw["treatment_only"] - design @ self.residualization
        standardized["treatment_only"] = (
            treatment_only - self.means["treatment_only"]
        ) / self.scales["treatment_only"]
        return _CausalScores(**standardized)


@dataclass(frozen=True)
class _BaselineSurface:
    confounder_coefficients: np.ndarray
    prognostic_coefficients: np.ndarray
    mean: float
    scale: float
    signal_mean: float
    signal_scale: float

    def evaluate(self, scores):
        confounding = scores.confounders @ self.confounder_coefficients
        prognostic = scores.prognostic @ self.prognostic_coefficients
        mu0 = (confounding + prognostic - self.mean) / self.scale
        signal = (confounding - self.signal_mean) / self.signal_scale
        return mu0, signal


def _covariates_to_pandas(X, columns=None):
    if isinstance(X, np.ndarray):
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("NumPy covariates must be a two-dimensional array.")
        if columns is None:
            columns = tuple(f"x{index}" for index in range(array.shape[1]))
        elif array.shape[1] != len(columns):
            raise ValueError("NumPy covariates must match the frozen column width.")
        return pd.DataFrame(array, columns=columns)

    if isinstance(X, pl.DataFrame):
        frame = X.to_pandas()
    elif isinstance(X, pd.DataFrame):
        frame = X
    else:
        raise TypeError(
            "Covariates must be a NumPy array, pandas DataFrame, or Polars DataFrame."
        )

    if columns is not None and tuple(frame.columns) != tuple(columns):
        raise ValueError("Covariate columns must match the frozen columns exactly.")
    return frame


def _prepare_covariate_frame(X, columns, categorical_columns):
    frame = _covariates_to_pandas(X, columns)
    if not categorical_columns:
        return frame
    frame = frame.copy()
    for name in categorical_columns:
        values = frame[name].astype(object)
        observed = values.notna()
        frame[name] = values.where(~observed, values[observed].map(str)).where(
            observed, np.nan
        )
    return frame


def _nonlinear_library(encoded, pair_indices):
    encoded = np.asarray(encoded, dtype=float)
    pair_indices = np.asarray(pair_indices, dtype=int)
    if pair_indices.size:
        products = encoded[:, pair_indices[:, 0]] * encoded[:, pair_indices[:, 1]]
    else:
        products = np.empty((encoded.shape[0], 0), dtype=float)
    return np.column_stack(
        (
            encoded,
            encoded**2,
            np.sin(encoded),
            (encoded > 0).astype(float),
            products,
        )
    )


def _residual_design(scores, order):
    scores = np.asarray(scores, dtype=float)
    columns = [np.ones((scores.shape[0], 1))]
    if order >= 1:
        columns.append(scores)
    if order >= 2:
        left, right = np.triu_indices(scores.shape[1])
        columns.append(scores[:, left] * scores[:, right])
    return np.column_stack(columns)


def _residualization_order(n_rows, n_scores):
    for order in (2, 1):
        n_columns = _residual_design(np.zeros((1, n_scores)), order).shape[1]
        if n_rows >= 2 * n_columns:
            return order
    return 0


_RIDGE_PENALTY_FRACTION = 0.01


def _ridge_projections(library, targets):
    """Ridge-fit each target column on the nonlinear library.

    Returns a ``(p, d)`` projection that maps the raw library to a prediction
    equal to the centered ridge fit up to an additive constant, and the ``(d,)``
    in-sample R² of each fit. Rows whose target is not finite are excluded from
    the fit. The penalty is ``0.01 * n_observed`` on library columns
    standardized to unit variance, so shrinkage is scale-free; zero-variance
    columns receive a zero coefficient.
    """

    library = np.asarray(library, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    projection = np.zeros((library.shape[1], targets.shape[1]))
    r2 = np.zeros(targets.shape[1])
    for index in range(targets.shape[1]):
        observed = np.isfinite(targets[:, index])
        if observed.sum() < 2:
            raise ValueError(
                "Fitted confounder targets need at least two finite values."
            )
        design = library[observed]
        center = design.mean(axis=0)
        scale = design.std(axis=0)
        active = scale > 0.0
        standardized = (design[:, active] - center[active]) / scale[active]
        target = targets[observed, index]
        target = target - target.mean()
        penalty = _RIDGE_PENALTY_FRACTION * observed.sum()
        gram = standardized.T @ standardized + penalty * np.eye(int(active.sum()))
        weights = np.linalg.solve(gram, standardized.T @ target)
        projection[active, index] = weights / scale[active]
        residual = target - standardized @ weights
        total = float(target @ target)
        r2[index] = 1.0 - float(residual @ residual) / total if total > 0.0 else 0.0
    return projection, r2


def _fit_causal_score_map(X, rng, fitted_confounders=None):
    frame = _covariates_to_pandas(X)
    columns = tuple(frame.columns)
    numeric_columns = [
        name
        for name in columns
        if pd.api.types.is_numeric_dtype(frame[name])
        and not pd.api.types.is_bool_dtype(frame[name])
    ]
    categorical_columns = tuple(name for name in columns if name not in numeric_columns)

    for name in numeric_columns:
        observed = frame[name].dropna().to_numpy(dtype=float)
        if observed.size == 0 or not np.isfinite(observed).all():
            raise ValueError(
                f"Numeric column {name!r} must contain a finite observed value."
            )
    frame = _prepare_covariate_frame(frame, columns, categorical_columns)

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=_MISSING_CATEGORY),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                list(categorical_columns),
            )
        )
    if not transformers:
        raise ValueError("Covariates must include at least one column.")

    encoder = ColumnTransformer(transformers)
    encoded = np.asarray(encoder.fit_transform(frame), dtype=float)
    n_encoded = encoded.shape[1]
    candidates = np.array(
        [
            (left, right)
            for left in range(n_encoded)
            for right in range(left + 1, n_encoded)
        ],
        dtype=int,
    ).reshape(-1, 2)
    n_pairs = min(16, len(candidates))
    if n_pairs:
        pair_indices = candidates[
            rng.choice(len(candidates), size=n_pairs, replace=False)
        ]
    else:
        pair_indices = np.empty((0, 2), dtype=int)
    library = _nonlinear_library(encoded, pair_indices)

    projections = {}
    raw = {}
    fitted_r2 = None
    for name, dimension in _ROLE_DIMENSIONS.items():
        if name == "confounders" and fitted_confounders is not None:
            projection, fitted_r2 = _ridge_projections(library, fitted_confounders)
            projections[name] = projection
            raw[name] = library @ projection
            continue
        projection = np.zeros((library.shape[1], dimension))
        for index in range(dimension):
            n_terms = min(5, library.shape[1])
            nonlinear = rng.integers(n_encoded, library.shape[1])
            remaining = np.delete(np.arange(library.shape[1]), nonlinear)
            selected = np.concatenate(
                (
                    np.array([nonlinear]),
                    rng.choice(remaining, size=n_terms - 1, replace=False),
                )
            )
            projection[selected, index] = rng.normal(size=selected.size)
        projections[name] = projection
        raw[name] = library @ projection

    means = {}
    scales = {}
    for name in _OUTCOME_ROLES:
        means[name], scales[name] = _standardization(raw[name])
    standardized = np.column_stack(
        [(raw[name] - means[name]) / scales[name] for name in _OUTCOME_ROLES]
    )
    order = _residualization_order(standardized.shape[0], standardized.shape[1])
    design = _residual_design(standardized, order)
    residualization = np.linalg.lstsq(design, raw["treatment_only"], rcond=None)[0]
    residual = raw["treatment_only"] - design @ residualization
    means["treatment_only"], scales["treatment_only"] = _standardization(residual)

    score_map = _CausalScoreMap(
        columns=columns,
        categorical_columns=categorical_columns,
        encoder=encoder,
        pair_indices=pair_indices,
        projections=projections,
        means=means,
        scales=scales,
        residualization_order=order,
        residualization=residualization,
        fitted_confounder_r2=fitted_r2,
    )
    return score_map, score_map._scores_from_raw(raw)


def _standardization(values):
    scale = values.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return values.mean(axis=0), scale


def _fit_baseline_surface(scores, rng, confounder_coefficients=None):
    n_confounders = scores.confounders.shape[1]
    prognostic_scale = 1.0 / np.sqrt(scores.prognostic.shape[1])
    if confounder_coefficients is None:
        confounder_coefficients = rng.normal(
            scale=1.0 / np.sqrt(n_confounders), size=n_confounders
        )
    else:
        confounder_coefficients = np.asarray(confounder_coefficients, dtype=float)
        if (
            confounder_coefficients.shape != (n_confounders,)
            or not np.isfinite(confounder_coefficients).all()
        ):
            raise ValueError(
                "confounder_coefficients must be a finite vector with one entry "
                "per confounder score."
            )
    prognostic_coefficients = rng.normal(
        scale=prognostic_scale, size=scores.prognostic.shape[1]
    )
    confounding = scores.confounders @ confounder_coefficients
    total = confounding + scores.prognostic @ prognostic_coefficients
    scale = total.std()
    signal_scale = confounding.std()
    return _BaselineSurface(
        confounder_coefficients=confounder_coefficients,
        prognostic_coefficients=prognostic_coefficients,
        mean=float(total.mean()),
        scale=float(scale if scale != 0 else 1.0),
        signal_mean=float(confounding.mean()),
        signal_scale=float(signal_scale if signal_scale != 0 else 1.0),
    )


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _calibrate_softmax_intercepts(logits, target):
    """Find intercepts whose softmax has the requested empirical marginal.

    The intercepts minimize the convex function
    ``mean_i logsumexp(logits_i + c) - c @ target`` over ``c`` with the last
    intercept fixed at zero; its gradient is exactly the marginal residual.
    Damped Newton steps with an Armijo line search converge from any start.
    """

    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=float)
    if logits.ndim != 2 or target.shape != (logits.shape[1],):
        raise ValueError("logits and target must have compatible category dimensions.")
    if not np.isfinite(logits).all():
        raise ValueError("Softmax intercept calibration requires finite logits.")

    n_rows, n_categories = logits.shape
    intercepts = np.zeros(n_categories, dtype=float)

    def objective(candidate):
        return float(logsumexp(logits + candidate, axis=1).mean() - candidate @ target)

    value = objective(intercepts)
    for _ in range(200):
        probabilities = _softmax(logits + intercepts)
        gradient = probabilities.mean(axis=0)[:-1] - target[:-1]
        if np.max(np.abs(gradient)) < 1e-10:
            return intercepts

        active = probabilities[:, :-1]
        hessian = np.diag(active.mean(axis=0)) - active.T @ active / n_rows
        ridge = 1e-12 * max(1.0, float(np.trace(hessian)))
        try:
            step = np.linalg.solve(
                hessian + ridge * np.eye(n_categories - 1), -gradient
            )
        except np.linalg.LinAlgError:
            step = -gradient
        if not np.isfinite(step).all() or gradient @ step >= 0.0:
            step = -gradient

        slope = float(gradient @ step)
        damping = 1.0
        for _ in range(60):
            candidate = intercepts.copy()
            candidate[:-1] += damping * step
            candidate_value = objective(candidate)
            if np.isfinite(candidate_value) and (
                candidate_value <= value + 1e-4 * damping * slope
            ):
                intercepts, value = candidate, candidate_value
                break
            damping *= 0.5
        else:
            raise ValueError(
                "Softmax intercept calibration could not reduce its objective."
            )

    raise ValueError(
        "Softmax intercept calibration did not converge after 200 iterations."
    )


def _normal_logpdf(value, location, scale):
    with np.errstate(over="ignore", invalid="ignore"):
        return -0.5 * ((value - location) / scale) ** 2 - np.log(scale) - 0.5 * _LOG_2PI


def _log_mixture(left, right, weight):
    if weight == 0.0:
        return left
    if weight == 1.0:
        return right
    return np.logaddexp(np.log1p(-weight) + left, np.log(weight) + right)


def _normalized_log_weights(log_values):
    log_values = np.asarray(log_values, dtype=float)
    if (
        log_values.size == 0
        or not np.isfinite(log_values).any()
        or np.isnan(log_values).any()
        or np.isposinf(log_values).any()
    ):
        raise ValueError(
            "Normalized log weights require at least one finite log weight and "
            "cannot contain NaN or positive infinity."
        )
    shifted = log_values - np.max(log_values)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Normalized log weights must have a finite positive sum.")
    return weights / total


def _log_mixture_probability(confounded, randomized, weight):
    if weight == 0.0:
        return confounded
    if weight == 1.0:
        return randomized
    return (1.0 - weight) * confounded + weight * randomized


def _gaussian_mixture_marginal_cdf(latent, means, scale, weight):
    """CDF of the row-averaged latent mixture at each ``latent`` value.

    The marginal is ``(1 - weight) * mean_i N(means_i, scale^2)`` plus
    ``weight * N(0, scale^2)``.
    """

    latent = np.asarray(latent, dtype=float)
    means = np.asarray(means, dtype=float)
    confounded = ndtr((latent[:, None] - means[None, :]) / scale).mean(axis=1)
    randomized = ndtr(latent / scale)
    return _log_mixture_probability(confounded, randomized, weight)


def _bisect_marginal_quantiles(cdf, probabilities, lower, upper):
    """Vectorized bisection of a monotone ``cdf`` on ``[lower, upper]``."""

    probabilities = np.asarray(probabilities, dtype=float)
    low = np.full(probabilities.shape, float(lower))
    high = np.full(probabilities.shape, float(upper))
    for _ in range(200):
        middle = 0.5 * low + 0.5 * high
        below = cdf(middle) < probabilities
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
        if np.all(high - low <= 1e-12 * np.maximum(1.0, np.abs(middle))):
            break
    return 0.5 * low + 0.5 * high


def _finite_metric(metric, strength):
    value = float(metric(strength))
    if not np.isfinite(value):
        raise ValueError(
            f"Confounding-strength metric must be finite at strength {strength}."
        )
    return value


def _calibrate_confounding_strength(metric, target, initial_strength):
    """Return the strength of the smallest crossing bracketed by the ladder.

    The metric is evaluated at zero and then along a geometric ladder of
    strengths (with ``initial_strength`` inserted) until it first reaches the
    target; the crossing is then refined by Brent's method. The search does not
    assume monotonicity, but it is not exhaustive either: a target reached and
    then left inside a bump that lies strictly between two consecutive ladder
    strengths is invisible to the ladder scan, so the returned strength is the
    smallest crossing the ladder actually brackets, not necessarily the
    smallest strength at which ``metric`` reaches ``target``.
    """

    target = float(target)
    floor = _finite_metric(metric, 0.0)
    if target <= floor:
        if target == floor:
            return 0.0, floor
        raise ValueError(
            f"target_confounding_bias={target} is below the bias floor {floor:.6g} "
            "already present at confounding_strength=0. Treatment-only predictors "
            "and effect modifiers can induce this floor; reduce "
            "treatment_only_strength or request a larger target."
        )

    ladder = set(_STRENGTH_LADDER.tolist())
    if initial_strength is not None and float(initial_strength) > 0.0:
        ladder.add(float(initial_strength))
    lower_strength = 0.0
    largest_metric = floor
    for strength in sorted(ladder):
        value = _finite_metric(metric, strength)
        largest_metric = max(largest_metric, value)
        if value >= target:
            if value == target:
                return strength, value
            upper_strength = strength
            break
        lower_strength = strength
    else:
        raise ValueError(
            f"target_confounding_bias={target} is outside the attainable metric "
            f"interval [{floor:.6g}, {largest_metric:.6g}] for strengths up to "
            f"{max(ladder):g}."
        )

    root = brentq(
        lambda strength: _finite_metric(metric, strength) - target,
        lower_strength,
        upper_strength,
        xtol=1e-10,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=200,
    )
    return float(root), _finite_metric(metric, root)


def _validate_nonnegative(name, value):
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_unit_interval(name, value):
    if not np.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be finite and between 0 and 1.")


def _validate_shared_dgp_parameters(
    *,
    confounding_strength,
    target_confounding_bias,
    treatment_only_strength,
    randomized_weight,
    outcome_effect_scale,
    effect_heterogeneity_scale,
    outcome_noise_scale,
):
    for name, value in {
        "confounding_strength": confounding_strength,
        "treatment_only_strength": treatment_only_strength,
        "outcome_effect_scale": outcome_effect_scale,
        "effect_heterogeneity_scale": effect_heterogeneity_scale,
        "outcome_noise_scale": outcome_noise_scale,
    }.items():
        _validate_nonnegative(name, value)
    if target_confounding_bias is not None:
        _validate_nonnegative("target_confounding_bias", target_confounding_bias)
    _validate_unit_interval("randomized_weight", randomized_weight)


class BaseSemiSyntheticDataset(BaseSyntheticDataset, ABC):
    """Abstract semi-synthetic DGP over the covariates of a real dataset.

    Subclasses implement the causal mathematics through the abstract hooks.
    The base class owns the lifecycle, backend coercion, random-stream
    separation, and the public :meth:`sample` and :meth:`log_prob` wrappers.
    It also provides shared machinery for the built-in score-based DGPs: the
    frozen causal score map, the standardized baseline outcome surface, the
    Gaussian outcome model, and the confounding-strength resolution.
    """

    outcome_columns = ("y",)

    def __init__(self, real_dataset: BaseDataset, random_state: int = 42):
        if not isinstance(real_dataset, BaseDataset):
            raise TypeError("real_dataset must be an instance of BaseDataset.")
        self.real_dataset = real_dataset
        super().__init__(n=0, random_state=random_state)
        self._prepare()

    def _prepare(self, n=None):
        if n is not None:
            raise ValueError(
                "BaseSemiSyntheticDataset derives its sample size from real_dataset "
                "and does not accept a sample size."
            )
        self.real_dataset_ = self.real_dataset.clone()
        X, treatment, outcome = self.real_dataset_.load()
        X = self._coerce_backend_frame(X, backend="polars")
        if X.height == 0 or len(set(X.columns)) != X.width:
            raise ValueError(
                "real_dataset must provide nonempty covariates with unique columns."
            )
        X = self._coerce_backend_frame(
            self._split_source(X, treatment, outcome), backend="polars"
        )
        if X.width == 0:
            raise ValueError(
                "At least one covariate column must remain after the source split."
            )
        self.n = X.height
        self._covariates = X
        structural_seed, sample_seed = np.random.SeedSequence(self.random_state).spawn(
            2
        )
        self._prepare_dgp(X, np.random.default_rng(structural_seed))
        treatment, outcome = self._draw_sample(X, sample_seed)
        self._treatments = treatment
        self._outcomes = outcome
        return self

    def _draw_sample(self, X, seed):
        treatment_seed, outcome_seed = seed.spawn(2)
        treatment = self._to_polars(
            self._sample_treatment(X, np.random.default_rng(treatment_seed))
        )
        mean = self.predict_y(X, treatment)
        outcome = self._coerce_outcome_frame(
            self._sample_outcome(
                mean, X, treatment, np.random.default_rng(outcome_seed)
            )
        )
        return treatment, outcome

    def _coerce_outcome_frame(self, values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape != (self.n, len(self.outcome_columns)):
            raise ValueError(
                "Outcome samples must match source rows and outcome columns."
            )
        if not np.isfinite(array).all():
            raise ValueError("Sampled outcomes must be finite.")
        return pl.DataFrame(
            {name: array[:, index] for index, name in enumerate(self.outcome_columns)}
        )

    def _split_source(self, X, treatment, outcome):
        """Return the covariates to release; subclasses may keep source columns.

        Called once during preparation with the loaded Polars covariates and
        the raw source treatment and outcome frames. The default releases
        ``X`` unchanged.
        """

        return X

    def _check_and_transform_X(self, covariates):
        source = getattr(self, "_covariates", None)
        if isinstance(covariates, np.ndarray) and source is not None:
            array = np.asarray(covariates)
            if array.ndim == 2 and array.shape[1] == source.width:
                covariates = pl.DataFrame(
                    {
                        column: array[:, index]
                        for index, column in enumerate(source.columns)
                    }
                )
        return self._coerce_backend_frame(covariates, backend="polars")

    def _check_and_transform_t(self, treatments):
        return self._coerce_treatment_frame(treatments, backend="polars")

    def sample(self, random_state):
        treatment, outcome = self._draw_sample(
            self._covariates, np.random.SeedSequence(random_state)
        )
        return (
            self._coerce_backend_frame(self._covariates),
            self._coerce_backend_frame(
                treatment, column_types=self._get_treatment_column_types()
            ),
            self._coerce_backend_frame(outcome),
        )

    def log_prob(self, X, treatment):
        X = self._check_and_transform_X(X)
        treatment = self._check_and_transform_t(treatment)
        if self._get_n_samples(X) != self._get_n_samples(treatment):
            raise ValueError(
                "log_prob requires X and treatment to have the same number of rows."
            )
        values = np.asarray(self._log_prob(X, treatment), dtype=float).reshape(-1, 1)
        if values.shape[0] != self._get_n_samples(X) or np.isnan(values).any():
            raise ValueError("_log_prob must return one non-NaN value per row.")
        return values

    def get_grid(self, n=100):
        return self._coerce_treatment_frame(self._get_grid(n))

    # -- shared machinery for score-based DGPs ---------------------------------

    def _fit_causal_structure(
        self, X, rng, *, fitted_confounders=None, confounder_coefficients=None
    ):
        """Freeze the score map and standardized baseline outcome surface."""

        self.score_map_, self.source_scores_ = _fit_causal_score_map(
            X, rng, fitted_confounders=fitted_confounders
        )
        self.baseline_surface_ = _fit_baseline_surface(
            self.source_scores_, rng, confounder_coefficients=confounder_coefficients
        )
        self.source_mu0_, self.source_confounding_signal_ = (
            self.baseline_surface_.evaluate(self.source_scores_)
        )
        self.baseline_standard_deviation_ = float(np.std(self.source_mu0_, ddof=0))
        if (
            not np.isfinite(self.baseline_standard_deviation_)
            or self.baseline_standard_deviation_ <= 0.0
        ):
            raise ValueError(
                "Confounding bias cannot be normalized because the baseline "
                "standard deviation must be finite and positive."
            )
        return self.source_scores_

    def _scores(self, X):
        """Return causal scores for ``X``, reusing the frozen source scores."""

        source = self._covariates
        if X is source or (
            isinstance(X, pl.DataFrame)
            and X.shape == source.shape
            and X.columns == source.columns
            and X.equals(source)
        ):
            return self.source_scores_
        return self.score_map_.transform(X)

    def _resolve_confounding_strength(
        self,
        metric,
        *,
        confounding_strength,
        target_confounding_bias,
        randomized_weight,
    ):
        """Store the frozen strength and the oracle bias it induces.

        ``metric(strength)`` must return ``(bias, bias_ratio)``; calibration
        targets the ratio.
        """

        if randomized_weight == 1.0:
            if target_confounding_bias is not None and target_confounding_bias > 0.0:
                raise ValueError(
                    "A positive target_confounding_bias is unattainable for a fully "
                    "randomized assignment."
                )
            strength = (
                0.0 if target_confounding_bias is not None else confounding_strength
            )
        elif target_confounding_bias is None:
            strength = float(confounding_strength)
        else:
            strength, _ = _calibrate_confounding_strength(
                lambda value: metric(value)[1],
                float(target_confounding_bias),
                float(confounding_strength),
            )
        self.confounding_strength_ = float(strength)
        self.confounding_bias_, self.confounding_bias_ratio_ = (
            float(value) for value in metric(self.confounding_strength_)
        )
        if not np.isfinite(
            [
                self.confounding_strength_,
                self.confounding_bias_,
                self.confounding_bias_ratio_,
            ]
        ).all():
            raise ValueError("Stored confounding-bias attributes must be finite.")

    def _outcome_mean(self, scores, basis, main_effects, modifier_coefficients):
        """Evaluate ``mu0(x) + b(a)'beta + b(a)'Gamma q_M(x)`` row by row."""

        mu0, _ = self.baseline_surface_.evaluate(scores)
        modifier_effects = basis @ modifier_coefficients
        interaction = np.einsum("ij,ij->i", modifier_effects, scores.modifiers)
        mean = mu0 + basis @ main_effects + interaction
        if not np.isfinite(mean).all():
            raise ValueError("Oracle outcome means must be finite.")
        return mean

    @staticmethod
    def _gaussian_outcome(mean, scale, rng):
        mean = np.asarray(mean, dtype=float)
        return mean + rng.normal(scale=scale, size=mean.shape)

    # -- abstract hooks --------------------------------------------------------

    @abstractmethod
    def _prepare_dgp(self, X, rng):
        raise NotImplementedError

    @abstractmethod
    def _sample_treatment(self, X, rng):
        raise NotImplementedError

    @abstractmethod
    def _log_prob(self, X, treatment):
        raise NotImplementedError

    @abstractmethod
    def _predict_y(self, X, treatment):
        raise NotImplementedError

    @abstractmethod
    def _sample_outcome(self, mean, X, treatment, rng):
        raise NotImplementedError

    @abstractmethod
    def _get_grid(self, n):
        raise NotImplementedError
