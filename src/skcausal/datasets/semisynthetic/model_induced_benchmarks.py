"""Named benchmarks built on :class:`ModelInducedConfounding`.

Each class here fixes a source table, a classifier and a treatment shape, so a
benchmark can be named rather than reassembled.

Most are backed by the UCI classics scikit-learn vendors, which keeps them
offline and reproducible. The ``GasDrift`` pair is the exception: 1797 rows is
the largest vendored classification table, so a benchmark of the size an
estimator comparison usually wants has to be fetched. Those two download from
OpenML on construction, cache on disk afterwards, and are excluded from the
offline object sweep for that reason.

The fixed ``C`` is not decoration. These tables are close to separable, and an
unregularized fit drives the induced propensity to 0 and 1: the treatment then
becomes a deterministic function of the covariates, which is a positivity
violation rather than a benchmark. Each class regularizes hard enough to keep
overlap, and its docstring reports what that buys.

``treatment_noise_scale=2.0`` and ``treatment_effect_scale=5.0`` are the
defaults everywhere a continuous coordinate appears, and both were measured
rather than picked. The score :math:`h_c(x)` driving a continuous coordinate is
standardized, so the covariate share of :math:`\\mathrm{Var}(T)` is exactly
:math:`1/(1 + \\sigma_T^2)`: 80% at :math:`\\sigma_T = 0.5`, 50% at 1.0, 20% at
2.0. At the first two, assignment is close enough to deterministic that a
correctly specified estimator cannot recover the curve — on digits at
:math:`\\sigma_T = 0.5` every estimator scored *worse* than a flat line at
``mean(y)`` (DR 0.78 RMSE against a true curve spanning 0.28). At 2.0 the
usual ordering appears. ``treatment_effect_scale=1.0`` is the other half of
that: the random spline spans about 0.28 while the confounder and the outcome
noise each have unit standard deviation, so there is no causal signal to find.
At 5.0 the curve spans roughly three times the outcome standard deviation.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from skcausal.datasets.real.openml import OpenMLDataset
from skcausal.datasets.real.sklearn import SklearnDataset
from skcausal.datasets.semisynthetic.model_induced import ModelInducedConfounding

__all__ = [
    "ModelInducedBreastCancerBinary",
    "ModelInducedDigitsContinuous",
    "ModelInducedDigitsContinuous2D",
    "ModelInducedDigitsMixed",
    "ModelInducedGasDriftContinuous2D",
    "ModelInducedGasDriftMixed",
]

# UCI Gas Sensor Array Drift, pinned: 13910 rows, 128 numeric features, six
# classes with shares from 0.12 to 0.22. Version 1 is pinned rather than
# "active" so the benchmark cannot change under a caller when OpenML promotes a
# new version.
_GAS_DRIFT = {"name": "gas-drift", "version": 1}


def _classifier(regularization: float) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=regularization)


class ModelInducedBreastCancerBinary(ModelInducedConfounding):
    """Binary treatment induced on scikit-learn's breast-cancer table.

    569 rows and 30 real-valued covariates. The treatment is drawn from the
    fitted class distribution, so its propensity is exactly
    :math:`\\hat{p}(\\cdot \\mid x)` and the response surface is known: the
    reference benchmark for a binary-treatment estimator.

    ``C=0.003`` is what makes it usable. An unregularized logistic fit on this
    table is near-perfect, and 74% of rows would sit above 0.99 on their
    predicted class; at this regularization 92% of the propensities fall in
    :math:`[0.05, 0.95]` and the smallest is 3e-4.

    This is the one benchmark that already discriminated at
    ``treatment_effect_scale=1.0``, because a known propensity leaves the
    adjustment nothing to get wrong: DirectRegressor scored 0.19 RMSE and the
    doubly robust pseudo-outcome 0.20, against 0.56 for a flat line. The
    default is 5.0 anyway, to keep it on the same scale as its siblings.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_effect_scale : float, default=5.0
        Multiplier on the random per-level effect.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    column_types = {"t_0": "categorical"}

    def __init__(
        self,
        random_state: int = 42,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        super().__init__(
            classifier=_classifier(0.003),
            dataset=SklearnDataset(name="breast_cancer"),
            random_state=random_state,
            treatment_types=("categorical",),
            treatment_effect_scale=treatment_effect_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [{"random_state": 0, "treatment_effect_scale": 2.0}]


class ModelInducedDigitsContinuous(ModelInducedConfounding):
    """Continuous treatment induced on scikit-learn's digits table.

    1797 rows and 64 pixel covariates, the largest table scikit-learn vendors,
    which is what a dose-response curve wants. The treatment is the
    standardized log-odds of the reference class plus Gaussian assignment
    noise; ``treatment_noise_scale`` is the only thing giving it overlap, since
    at zero it is a deterministic function of the covariates.

    This is the benchmark the module-level defaults were measured on. At the
    former ``treatment_noise_scale=0.5`` the covariates owned 80% of
    :math:`\\mathrm{Var}(T)` and nothing worked: RMSE 0.99 naive, 0.72 direct,
    0.78 doubly robust, against 0.07 for a flat line at ``mean(y)`` on a true
    curve spanning 0.28. At 2.0 with ``treatment_effect_scale=5.0`` the
    ordering is naive 0.65, direct 0.17, doubly robust 0.12.

    Three pixel columns are constant and are released as such.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_noise_scale : float, default=2.0
        Standard deviation of the assignment noise, in units of the
        standardized score. Lower it for a harder positivity problem; below
        1.0 the benchmark stops discriminating between estimators.
    treatment_effect_scale : float, default=5.0
        Multiplier on the random spline effect.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    column_types = {"t_0": "continuous"}

    def __init__(
        self,
        random_state: int = 42,
        treatment_noise_scale: float = 2.0,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        super().__init__(
            classifier=_classifier(0.01),
            dataset=SklearnDataset(name="digits"),
            random_state=random_state,
            treatment_types=("continuous",),
            treatment_noise_scale=treatment_noise_scale,
            treatment_effect_scale=treatment_effect_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [{"random_state": 0, "treatment_effect_scale": 2.0}]


class ModelInducedDigitsMixed(ModelInducedConfounding):
    """Mixed continuous-plus-categorical treatment on the digits table.

    ``t_0`` is the standardized log-odds of the reference class, ``t_1`` a draw
    from the fitted ten-class distribution, so the categorical coordinate has an
    exactly known propensity whatever ``treatment_correlation`` is set to. The
    two already share the covariate channel; :math:`\\rho` adds dependence
    beyond it and raises the achieved mutual information monotonically
    (0.05, 0.09, 0.20 nats at :math:`\\rho` = 0, 0.5, 0.9 under the defaults).

    Lowering ``treatment_noise_scale`` to 1.0 raises those figures (0.17, 0.21,
    0.32) but not :math:`\\rho`'s share of them: at 2.0 the covariates own only
    20% of :math:`\\mathrm{Var}(T_0)`, so less of the dependence arrives through
    the shared covariate channel and more of what remains is :math:`\\rho`'s own
    doing. Under the defaults the benchmark separates estimators, at RMSE 0.88
    naive, 0.69 direct and 0.59 doubly robust against 2.24 for a flat line at
    ``mean(y)``.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_correlation : float or array-like, default=0.0
        Correlation :math:`\\rho \\in (-1, 1)` of the two assignment latents:
        how much the coordinates depend on each other *given* the covariates.
    treatment_noise_scale : float, default=2.0
        Standard deviation of the assignment noise on ``t_0``. At zero the
        continuous coordinate is deterministic and :math:`\\rho` goes inert.
    treatment_effect_scale : float, default=5.0
        Multiplier on both random treatment effects.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    column_types = {"t_0": "continuous", "t_1": "categorical"}

    def __init__(
        self,
        random_state: int = 42,
        treatment_correlation=0.0,
        treatment_noise_scale: float = 2.0,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        super().__init__(
            classifier=_classifier(0.01),
            dataset=SklearnDataset(name="digits"),
            random_state=random_state,
            treatment_types=("continuous", "categorical"),
            treatment_correlation=treatment_correlation,
            treatment_noise_scale=treatment_noise_scale,
            treatment_effect_scale=treatment_effect_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [{"random_state": 0, "treatment_correlation": 0.5}]


class ModelInducedDigitsContinuous2D(ModelInducedConfounding):
    """Two continuous treatment coordinates on the digits table.

    ``t_0`` and ``t_1`` are the standardized log-odds of the first two classes
    plus correlated assignment noise. Ten classes are what make this shape
    possible: on a binary source the second log-odds is the negative of the
    first, which is not a second coordinate.

    The assignment noise defaults to 2.0, which is also what makes
    :math:`\\rho` a clean dependence dial. At 0.5 the two log-odds are
    anti-correlated through the covariates (:math:`\\mathrm{corr} = -0.35`) and
    a positive :math:`\\rho` cancels that before it adds; at 1.0 the realized
    correlation moves with the knob but never reaches zero at
    :math:`\\rho = 0` (-0.25, 0.01, 0.23); at 2.0 the covariate channel is
    diluted enough that the knob reads almost directly (-0.03, +0.35, +0.64 at
    :math:`\\rho` = 0, 0.5, 0.9). Lower it if you want stronger confounding and
    do not need the knob, at the cost of separation between estimators. Under
    the defaults the benchmark separates them, at RMSE 0.64 naive, 0.41 direct
    and 0.25 doubly robust against 1.31 for a flat line at ``mean(y)``.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_correlation : float or array-like, default=0.0
        Correlation :math:`\\rho \\in (-1, 1)` of the two assignment latents.
    treatment_noise_scale : float, default=2.0
        Standard deviation of the assignment noise on both coordinates.
    treatment_effect_scale : float, default=5.0
        Multiplier on both random spline effects.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    column_types = {"t_0": "continuous", "t_1": "continuous"}

    def __init__(
        self,
        random_state: int = 42,
        treatment_correlation=0.0,
        treatment_noise_scale: float = 2.0,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        super().__init__(
            classifier=_classifier(0.01),
            dataset=SklearnDataset(name="digits"),
            random_state=random_state,
            treatment_types=("continuous", "continuous"),
            treatment_correlation=treatment_correlation,
            treatment_noise_scale=treatment_noise_scale,
            treatment_effect_scale=treatment_effect_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [{"random_state": 0, "treatment_correlation": 0.5}]


class ModelInducedGasDriftMixed(ModelInducedConfounding):
    """Mixed treatment on UCI's gas-sensor drift table, at 13910 rows.

    The large sibling of :class:`ModelInducedDigitsMixed`: 13910 rows and 128
    numeric sensor features, which is the sample size an estimator comparison
    usually wants and eight times what the vendored tables offer. ``t_0`` is the
    standardized log-odds of the reference class, ``t_1`` a draw from the fitted
    six-class distribution, so the categorical coordinate has an exactly known
    propensity whatever :math:`\\rho` is set to.

    Six classes rather than digits' ten leave each level enough of the sample to
    matter, and :math:`\\rho` dials dependence monotonically: the achieved mutual
    information is 0.08, 0.12 and 0.20 nats at :math:`\\rho` = 0, 0.5, 0.9. Those
    figures were 0.36, 0.41 and 0.53 at the former
    ``treatment_noise_scale=0.5``, where most of the dependence arrived through
    the shared covariate channel rather than through the knob — and where the
    assignment was too close to deterministic to benchmark against.

    Constructing this fetches the table from OpenML the first time and reads
    scikit-learn's on-disk cache afterwards. It is not offline.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_correlation : float or array-like, default=0.0
        Correlation :math:`\\rho \\in (-1, 1)` of the two assignment latents:
        how much the coordinates depend on each other *given* the covariates.
    treatment_noise_scale : float, default=2.0
        Standard deviation of the assignment noise on ``t_0``. At zero the
        continuous coordinate is deterministic and :math:`\\rho` goes inert.
    treatment_effect_scale : float, default=5.0
        Multiplier on both random treatment effects.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    column_types = {"t_0": "continuous", "t_1": "categorical"}

    def __init__(
        self,
        random_state: int = 42,
        treatment_correlation=0.0,
        treatment_noise_scale: float = 2.0,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        super().__init__(
            classifier=_classifier(0.003),
            dataset=OpenMLDataset(**_GAS_DRIFT),
            random_state=random_state,
            treatment_types=("continuous", "categorical"),
            treatment_correlation=treatment_correlation,
            treatment_noise_scale=treatment_noise_scale,
            treatment_effect_scale=treatment_effect_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [{"random_state": 0, "treatment_correlation": 0.5}]


class ModelInducedGasDriftContinuous2D(ModelInducedConfounding):
    """Two continuous treatment coordinates on the gas-drift table, at 13910 rows.

    The large sibling of :class:`ModelInducedDigitsContinuous2D`. ``t_0`` and
    ``t_1`` are the standardized log-odds of the first two of six classes plus
    correlated assignment noise.

    As there, the assignment noise defaults to 2.0: at 0.5 the two log-odds are
    anti-correlated through the covariates and :math:`\\rho` only cancels that
    (corr -0.25 to -0.07 over the whole range), at 1.0 the realized correlation
    tracks the knob without reaching zero (-0.16, 0.09, 0.29), and at 2.0 it
    reads almost directly off it (-0.05, +0.33, +0.65 at :math:`\\rho` = 0, 0.5,
    0.9). Lower it if you want stronger confounding and do not need the knob —
    below 1.0 the assignment is too close to deterministic for an estimator to
    recover the curve at all.

    Constructing this fetches the table from OpenML the first time and reads
    scikit-learn's on-disk cache afterwards. It is not offline.

    Parameters
    ----------
    random_state : int, default=42
        Seed for the derived data-generating process.
    treatment_correlation : float or array-like, default=0.0
        Correlation :math:`\\rho \\in (-1, 1)` of the two assignment latents.
    treatment_noise_scale : float, default=2.0
        Standard deviation of the assignment noise on both coordinates.
    treatment_effect_scale : float, default=5.0
        Multiplier on both random spline effects.
    outcome_noise_scale : float, default=1.0
        Standard deviation of the additive outcome noise.
    """

    column_types = {"t_0": "continuous", "t_1": "continuous"}

    def __init__(
        self,
        random_state: int = 42,
        treatment_correlation=0.0,
        treatment_noise_scale: float = 2.0,
        treatment_effect_scale: float = 5.0,
        outcome_noise_scale: float = 1.0,
    ):
        super().__init__(
            classifier=_classifier(0.003),
            dataset=OpenMLDataset(**_GAS_DRIFT),
            random_state=random_state,
            treatment_types=("continuous", "continuous"),
            treatment_correlation=treatment_correlation,
            treatment_noise_scale=treatment_noise_scale,
            treatment_effect_scale=treatment_effect_scale,
            outcome_noise_scale=outcome_noise_scale,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [{"random_state": 0, "treatment_correlation": 0.5}]
