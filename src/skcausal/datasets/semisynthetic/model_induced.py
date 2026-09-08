from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import ndtr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

from skcausal.datasets.semisynthetic._model_induced_base import (
    BaseSemiSyntheticDataset,
    _as_1d,
    _as_matrix,
    _evaluate_random_spline,
    _fit_random_spline,
    _random_level_effects,
)

__all__ = ["ModelInducedConfounding"]

_PROBABILITY_CLIP = 1e-6
_TREATMENT_TYPES = ("continuous", "categorical")


def _correlation_matrix(value, size: int) -> np.ndarray:
    """Validate ``value`` as a ``size`` x ``size`` correlation matrix.

    A scalar fills every off-diagonal entry. The Cholesky factorization is the
    positive-definiteness check, and the factor is what the sampler needs.
    """

    if np.ndim(value) == 0:
        rho = float(value)
        if not -1.0 < rho < 1.0:
            raise ValueError("treatment_correlation must satisfy -1 < rho < 1.")
        matrix = np.full((size, size), rho)
        np.fill_diagonal(matrix, 1.0)
    else:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (size, size):
            raise ValueError(
                "treatment_correlation must be a scalar or a matrix with shape "
                f"({size}, {size}), one row per treatment coordinate."
            )
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1.0):
            raise ValueError(
                "treatment_correlation must be symmetric with a unit diagonal."
            )
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("treatment_correlation must be positive definite.") from None
    return matrix


class ModelInducedConfounding(BaseSemiSyntheticDataset):
    r"""Semi-synthetic treatment dataset induced by a model on a classification task.

    One classifier fitted on a real classification sample ``(X, y)`` supplies
    every treatment coordinate. ``treatment_types`` declares the coordinates,
    each ``"continuous"`` or ``"categorical"``, and a Gaussian copula over their
    assignment latents with correlation matrix ``treatment_correlation`` dials
    how much they depend on each other *beyond* the covariate channel they
    already share through the fitted model. A single coordinate is allowed:
    ``treatment_types=("continuous",)`` gives a one-dimensional continuous
    treatment, ``("categorical",)`` a one-dimensional categorical one, and the
    copula then has nothing to couple.

    The covariates are standardized and a cloned classifier is fit on them.
    Write :math:`\hat{p}_a(x)` for the fitted probability of level :math:`a`,
    and let :math:`a_1 < \cdots < a_C` be the levels in ``classes_`` order.
    The standardized log-odds of class :math:`c` is

    .. math::

        \ell_c(x) = \log \frac{\hat{p}_{a_c}(x)}{1 - \hat{p}_{a_c}(x)},
        \qquad
        h_c(x) = \frac{\ell_c(x) - \bar{\ell}_c}{s_{\ell_c}},

    with the probabilities clipped to :math:`[10^{-6}, 1 - 10^{-6}]` before the
    logit, and :math:`\bar{\ell}_c, s_{\ell_c}` the mean and standard deviation
    over the realized sample (stored so that :meth:`predict_y` reproduces
    :math:`h_c` exactly on new covariates).

    For each row a vector of :math:`K` standard normals is drawn with
    correlation matrix :math:`R =` ``treatment_correlation``,

    .. math::

        (Z_{i,1}, \ldots, Z_{i,K}) \stackrel{\mathrm{iid}}{\sim}
        \mathcal{N}(0, R),

    and coordinate :math:`k` is

    .. math::

        T_{i,k} = h_{c(k)}(X_i) + \sigma_T Z_{i,k}
        \quad\text{(continuous)},
        \qquad
        T_{i,k} = F^{-1}\!\left(1 - \Phi(Z_{i,k}) \;\middle|\; X_i\right)
        \quad\text{(categorical)},

    where :math:`\sigma_T =` ``treatment_noise_scale``, :math:`F^{-1}(\cdot
    \mid x)` is the inverse CDF of the fitted class distribution
    :math:`\hat{p}(\cdot \mid x)` over the levels in ``classes_`` order, and
    :math:`c(k)` is the class whose log-odds drive the :math:`j`-th continuous
    coordinate: :math:`c = j \bmod C`, so the first continuous coordinate uses
    the reference class and later ones move to the next classes, cycling if
    there are more continuous coordinates than classes. Every categorical
    coordinate samples from the same fitted class distribution, each with its
    own latent.

    Three consequences are worth stating, because they are the point of the
    construction:

    * Since :math:`\Phi(Z_{i,k})` is marginally uniform for any :math:`R`, the
      law of a categorical coordinate given :math:`X_i` is *exactly*
      :math:`\hat{p}(\cdot \mid X_i)`. It has a known propensity, and
      ``treatment_correlation`` does not perturb it.
    * The reference level occupies the *top* of the unit interval, through the
      :math:`1 - \Phi` inversion. A positive entry between a categorical
      coordinate and the reference-class continuous coordinate then reinforces
      the association the covariates already induce between them — a high
      :math:`h_1(x)` both raises that coordinate and makes the reference level
      more likely — rather than opposing it.
    * :math:`\sigma_T` is what gives a continuous coordinate any overlap at
      all. At :math:`\sigma_T = 0` it is a deterministic function of the
      covariates, which is a total positivity violation; it also makes every
      entry of :math:`R` touching a continuous coordinate inert, since there is
      then no latent left in it for the others to correlate with.

    The response surface is additive across the coordinates,

    .. math::

        \mu(x, t) = h_1(x) + \sum_{k=1}^{K} g_k(t_k),

    with :math:`g_k` a random centered B-spline for a continuous coordinate and
    a random centered offset per level for a categorical one, all drawn from
    the seeded generator in coordinate order and scaled by
    ``treatment_effect_scale``. There is no interaction term: each coordinate
    has a marginal dose-response that does not depend on the others. Observed outcomes add Gaussian noise
    with standard deviation ``outcome_noise_scale``.

    The dependence actually achieved is reported, not requested:
    ``achieved_mutual_information_`` is a :math:`K \times K` matrix of
    estimated pairwise mutual information between the realized coordinates in
    nats, which includes the part induced through the covariates. Its diagonal
    is zero. There is no solver that hits a requested value.

    Parameters
    ----------
    classifier : estimator
        Cloneable scikit-learn classifier exposing ``predict_proba`` and
        ``classes_``.
    dataset : BaseTabularDataset
        Source of the supervised classification sample.
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_types : sequence of str, default=("continuous", "categorical")
        Kind of each treatment coordinate, at least one. The released
        treatment columns are ``t_0, ..., t_{K-1}`` in this order.
    treatment_correlation : float or array-like, default=0.0
        Correlation matrix :math:`R` of the assignment latents. A scalar
        :math:`\rho \in (-1, 1)` fills every off-diagonal entry; a matrix must
        be :math:`K \times K`, symmetric, with unit diagonal and positive
        definite. Zero leaves the coordinates dependent only through the
        covariates.
    treatment_noise_scale : float, default=2.0
        Standard deviation :math:`\sigma_T \ge 0` of the assignment noise added
        to each continuous coordinate, in units of the standardized score.
        Since :math:`h_c` is standardized, the covariate share of
        :math:`\mathrm{Var}(T_k)` is exactly :math:`1/(1 + \sigma_T^2)`: 20% at
        the default, 50% at 1.0, 80% at 0.5. Below 1.0 the assignment is close
        enough to deterministic that no estimator recovers the curve.
    n_spline_knots : int, default=6
        Knots of the random spline in each continuous coordinate.
    spline_degree : int, default=3
        Degree of those splines.
    treatment_effect_scale : float, default=5.0
        Multiplier on every random treatment effect. The random spline spans
        about 0.28 at 1.0, against a unit-variance confounder and unit-variance
        outcome noise, which leaves no causal signal to find; the default puts
        the curve a few outcome standard deviations wide.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    _tags = {"task": "regression", "source_task": "classification"}
    column_types = {"t_0": "continuous", "t_1": "categorical"}

    def __init__(
        self,
        classifier,
        dataset,
        random_state: int = 42,
        treatment_types=_TREATMENT_TYPES,
        treatment_correlation=0.0,
        treatment_noise_scale: float = 2.0,
        n_spline_knots: int = 6,
        spline_degree: int = 3,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        types = list(treatment_types)
        if not types or any(kind not in _TREATMENT_TYPES for kind in types):
            raise ValueError(
                "treatment_types must list at least one coordinate, each "
                "'continuous' or 'categorical'."
            )
        if float(treatment_noise_scale) < 0.0:
            raise ValueError("treatment_noise_scale must be non-negative.")
        if n_spline_knots < 2:
            raise ValueError("n_spline_knots must be at least 2.")
        if spline_degree < 0:
            raise ValueError("spline_degree must be non-negative.")

        self.classifier = classifier
        self.treatment_types = treatment_types
        self.treatment_correlation = treatment_correlation
        self.treatment_noise_scale = treatment_noise_scale
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree
        self.treatment_effect_scale = treatment_effect_scale
        self.outcome_noise_scale = outcome_noise_scale

        self.column_types = {f"t_{k}": kind for k, kind in enumerate(types)}
        self.correlation_matrix_ = _correlation_matrix(
            treatment_correlation, len(types)
        )

        super().__init__(dataset=dataset, random_state=random_state)
        self._prepare()

    def _prepare_target(self, outcome_frame):
        self.classification_target_ = _as_1d(
            outcome_frame, name=f"{type(self).__name__} target"
        )
        return self.classification_target_

    # -- coordinate bookkeeping ----------------------------------------------

    @property
    def _columns(self) -> list[str]:
        return list(self.column_types)

    def _continuous_class(self, coordinate: int) -> int:
        """Return the class whose log-odds drive a continuous coordinate."""

        kinds = list(self.column_types.values())[:coordinate]
        return kinds.count("continuous") % len(self.treatment_levels_)

    # -- the fitted scores ---------------------------------------------------

    def _predict_proba(self, covariates: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(
            self.classifier_.predict_proba(covariates), dtype=float
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != len(
            self.treatment_levels_
        ):
            raise ValueError(
                "predict_proba must return an array with shape "
                "(n_samples, n_classes) matching the levels seen during fit."
            )
        return probabilities

    def _logits(self, covariates: np.ndarray) -> np.ndarray:
        """Return the raw per-class log-odds, before standardization."""

        clipped = np.clip(
            self._predict_proba(covariates), _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP
        )
        return np.log(clipped) - np.log1p(-clipped)

    def _scores(self, covariates: np.ndarray) -> np.ndarray:
        """Return ``h_c(x)`` for every class, as columns."""

        return (self._logits(covariates) - self.score_offsets_) / self.score_scales_

    def _sample_levels(self, probabilities: np.ndarray, uniforms: np.ndarray):
        """Draw one level per row from the fitted class distribution.

        The reference level sits at the top of the unit interval so that a
        positive coupling with the reference-class continuous coordinate
        reinforces, rather than opposes, the association the covariates
        already induce.
        """

        cumulative = np.cumsum(probabilities, axis=1)
        thresholds = (1.0 - uniforms).reshape(-1, 1)
        indices = np.clip(
            (thresholds > cumulative).sum(axis=1), 0, len(self.treatment_levels_) - 1
        )
        return np.asarray(self.treatment_levels_, dtype=object)[indices].astype(str)

    def _get_treatments(self, covariates) -> pl.DataFrame:
        covariate_array = _as_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )

        self.classifier_ = self._clone_estimator(self.classifier)
        self.classifier_.fit(covariate_array, self.classification_target_)
        if not hasattr(self.classifier_, "classes_"):
            raise ValueError(
                f"Classifier must expose classes_ after fitting {type(self).__name__}."
            )
        self.treatment_levels_ = (
            np.asarray(self.classifier_.classes_, dtype=object).astype(str).tolist()
        )
        if len(self.treatment_levels_) < 2:
            raise ValueError(
                f"{type(self).__name__} requires a source with at least two "
                "classes, since one class leaves no categorical treatment to vary."
            )
        self.reference_level_ = self.treatment_levels_[0]

        logits = self._logits(covariate_array)
        self.score_offsets_ = logits.mean(axis=0)
        scales = logits.std(axis=0, ddof=0)
        self.score_scales_ = np.where(scales > 0.0, scales, 1.0)
        scores = self._scores(covariate_array)
        probabilities = self._predict_proba(covariate_array)

        latents = (
            self._rng.standard_normal(
                (covariate_array.shape[0], len(self.column_types))
            )
            @ np.linalg.cholesky(self.correlation_matrix_).T
        )

        coordinates = {}
        for k, (column, kind) in enumerate(self.column_types.items()):
            if kind == "continuous":
                coordinates[column] = (
                    scores[:, self._continuous_class(k)]
                    + float(self.treatment_noise_scale) * latents[:, k]
                )
            else:
                coordinates[column] = self._sample_levels(
                    probabilities, ndtr(latents[:, k])
                )

        self._fit_treatment_effects(coordinates)
        self.achieved_mutual_information_ = self._estimate_mutual_information(
            coordinates
        )
        return pl.DataFrame(coordinates)

    def _estimate_mutual_information(self, coordinates: dict) -> np.ndarray:
        def codes(values):
            return np.unique(np.asarray(values, dtype=str), return_inverse=True)[1]

        columns = self._columns
        matrix = np.zeros((len(columns), len(columns)))
        for i, a in enumerate(columns):
            for j in range(i + 1, len(columns)):
                b = columns[j]
                kinds = (self.column_types[a], self.column_types[b])
                x, y = coordinates[a], coordinates[b]
                if kinds == ("continuous", "continuous"):
                    value = mutual_info_regression(
                        x.reshape(-1, 1), y, random_state=self.random_state
                    )[0]
                elif kinds == ("categorical", "categorical"):
                    value = mutual_info_score(codes(x), codes(y))
                else:
                    continuous, labels = (x, y) if kinds[0] == "continuous" else (y, x)
                    label_codes = codes(labels)
                    value = (
                        0.0
                        if label_codes.max() == 0
                        else mutual_info_classif(
                            continuous.reshape(-1, 1),
                            label_codes,
                            random_state=self.random_state,
                        )[0]
                    )
                matrix[i, j] = matrix[j, i] = float(value)
        return matrix

    # -- the response surface ------------------------------------------------

    def _fit_treatment_effects(self, coordinates: dict) -> None:
        self.spline_effects_ = {}
        self.level_effects_ = {}
        self.treatment_range_ = {}
        for column, kind in self.column_types.items():
            values = coordinates[column]
            if kind == "continuous":
                self.spline_effects_[column] = _fit_random_spline(
                    values,
                    rng=self._rng,
                    n_knots=self.n_spline_knots,
                    degree=self.spline_degree,
                )
                self.treatment_range_[column] = (
                    float(np.min(values)),
                    float(np.max(values)),
                )
            else:
                self.level_effects_[column] = _random_level_effects(
                    values, self.treatment_levels_, rng=self._rng
                )

    def _split_treatments(self, treatments) -> dict:
        """Return one array per coordinate from any accepted treatment container."""

        columns = self._columns
        if isinstance(treatments, np.ndarray):
            if treatments.ndim != 2 or treatments.shape[1] != len(columns):
                raise ValueError(
                    f"{type(self).__name__} expects treatments with shape "
                    f"(n_samples, {len(columns)}), one column per coordinate."
                )
            raw = {column: treatments[:, k] for k, column in enumerate(columns)}
        else:
            frame = self._coerce_backend_frame(treatments, backend="polars")
            missing = set(columns) - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{type(self).__name__} treatments must include the columns "
                    f"{', '.join(columns)}; missing {sorted(missing)}."
                )
            raw = {column: frame.get_column(column).to_numpy() for column in columns}

        coordinates = {}
        for column, kind in self.column_types.items():
            if kind == "continuous":
                coordinates[column] = np.asarray(raw[column], dtype=float)
                continue
            levels = np.asarray(raw[column]).astype(str)
            unknown = sorted(set(levels) - set(self.treatment_levels_))
            if unknown:
                raise ValueError(
                    f"{type(self).__name__} received unknown treatment levels in "
                    f"{column}: {', '.join(unknown)}. Supported levels are: "
                    f"{', '.join(self.treatment_levels_)}."
                )
            coordinates[column] = levels
        return coordinates

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_array = _as_matrix(
            covariates,
            expected_width=self._n_features,
            name=f"{type(self).__name__} covariates",
        )
        coordinates = self._split_treatments(treatments)

        total = self._scores(covariate_array)[:, 0]
        for column, kind in self.column_types.items():
            if kind == "continuous":
                transformer, coefficients, offset = self.spline_effects_[column]
                effect = _evaluate_random_spline(
                    coordinates[column],
                    transformer=transformer,
                    coefficients=coefficients,
                    offset=offset,
                )
            else:
                lookup = self.level_effects_[column]
                effect = np.array(
                    [lookup[level] for level in coordinates[column]], dtype=float
                )
            total = total + self.treatment_effect_scale * effect
        return total.reshape(-1, 1)

    # -- released frames -----------------------------------------------------

    def _treatment_frame(self, treatments) -> pl.DataFrame:
        # The base builds a single "t" column; this dataset already carries one
        # column per coordinate with the dtypes column_types declares.
        return self._to_polars(treatments)

    def get_levels(self) -> pl.DataFrame:
        categorical = [
            c for c, kind in self.column_types.items() if kind == "categorical"
        ]
        return self._coerce_backend_frame(
            pl.DataFrame({column: self.treatment_levels_ for column in categorical}),
            column_types={column: "categorical" for column in categorical},
        )

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        """Cross every coordinate: ``n`` points per continuous one, all levels per
        categorical one. The height is ``n ** n_continuous * n_levels ** n_categorical``.
        """

        grid = None
        for column, kind in self.column_types.items():
            if kind == "continuous":
                lower, upper = self.treatment_range_[column]
                values = (
                    np.repeat(lower, n)
                    if np.isclose(lower, upper)
                    else np.linspace(lower, upper, n)
                )
                axis = pl.DataFrame({column: values})
            else:
                axis = pl.DataFrame({column: self.treatment_levels_})
            grid = axis if grid is None else grid.join(axis, how="cross")
        return self._coerce_treatment_frame(grid)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        from skcausal.datasets.real.sklearn import SklearnDataset

        # Wine is very nearly separable. An unregularized fit would make the
        # categorical coordinate a deterministic function of the covariates,
        # leaving no overlap and nothing for treatment_correlation to move.
        return [
            {
                "classifier": LogisticRegression(max_iter=2000, C=0.01),
                "dataset": SklearnDataset(name="digits"),
                "treatment_correlation": 0.5,
                "treatment_effect_scale": 2.0,
                "random_state": 0,
            },
        ]
