import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["ExampleCategorical"]


def _as_covariate_frame(covariates) -> pl.DataFrame:
    if isinstance(covariates, pl.DataFrame):
        return covariates

    array = np.asarray(covariates, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "ExampleCategorical expects covariates with shape (n_samples, 2)."
        )

    return pl.DataFrame({"x0": array[:, 0], "x1": array[:, 1]})


def _as_treatment_labels(treatments) -> np.ndarray:
    if isinstance(treatments, pl.DataFrame):
        if treatments.width != 1:
            raise ValueError("ExampleCategorical expects a single treatment column.")
        return treatments.get_column("treatment").cast(pl.Utf8).to_numpy()

    array = np.asarray(treatments)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError(
                "ExampleCategorical expects treatments with shape (n_samples, 1)."
            )
        array = array[:, 0]

    if array.ndim != 1:
        raise ValueError("ExampleCategorical expects one treatment label per sample.")

    return array.astype(str)


class ExampleCategorical(BaseSyntheticDataset):
    r"""Three-level categorical treatment dataset with observed confounding.

    The observed covariates satisfy

    .. math::

        X_0, X_1 \stackrel{\mathrm{iid}}{\sim} \mathcal{N}(0, 1).

    A latent treatment score is generated as

    .. math::

        S = X_0 + w_1 X_1 + \varepsilon_T,
        \qquad
        \varepsilon_T \sim \mathcal{N}(0, \sigma_T^2),

    where :math:`w_1 =` ``score_x1_weight`` and
    :math:`\sigma_T =` ``treatment_noise``. The observed treatment is then the
    three-level threshold rule

    .. math::

        A =
        \begin{cases}
        \mathrm{treated}, & S > \tau, \\
        \mathrm{placebo}, & S < -\tau, \\
        \mathrm{control}, & \mathrm{otherwise},
        \end{cases}

    with :math:`\tau =` ``treatment_threshold``.

    The noiseless response surface is

    .. math::

        m(X, A) = \beta X_0 + \alpha(A),

    where :math:`\beta =` ``covariate_effect`` and :math:`\alpha(A)` is the
    category-specific shift given by ``control_effect``, ``placebo_effect``, and
    ``treated_effect``. Observed outcomes satisfy

    .. math::

        Y \mid X, A \sim \mathcal{N}(m(X, A), \sigma_Y^2),

    with :math:`\sigma_Y =` ``outcome_noise``.
    """

    column_types = {"treatment": "categorical"}

    def __init__(
        self,
        n=500,
        treatment_noise=0.4,
        outcome_noise=0.2,
        covariate_effect=0.8,
        score_x1_weight=-0.5,
        treatment_threshold=0.6,
        control_effect=0.0,
        placebo_effect=-0.5,
        treated_effect=1.2,
        random_state=0,
    ):
        self.treatment_noise = treatment_noise
        self.outcome_noise = outcome_noise
        self.covariate_effect = covariate_effect
        self.score_x1_weight = score_x1_weight
        self.treatment_threshold = treatment_threshold
        self.control_effect = control_effect
        self.placebo_effect = placebo_effect
        self.treated_effect = treated_effect
        self.random_state = random_state

        super().__init__(n=n, random_state=random_state)
        self._prepare(self.n)

    @property
    def treatment_effects(self) -> dict[str, float]:
        return {
            "control": float(self.control_effect),
            "placebo": float(self.placebo_effect),
            "treated": float(self.treated_effect),
        }

    def get_levels(self) -> pl.DataFrame:
        return self._coerce_treatment_frame(
            pl.DataFrame({"treatment": ["control", "placebo", "treated"]})
        )

    def _prepare(self, n: int = None):
        if n is not None:
            self.n = n

        covariates = _as_covariate_frame(self._get_covariates())
        treatments = self._to_polars(self._get_treatments(covariates))
        outcomes = self._get_outcomes(covariates, treatments)

        self._covariates = covariates
        self._treatments = treatments
        if isinstance(outcomes, pl.DataFrame):
            self._outcomes = outcomes
        else:
            self._outcomes = pl.DataFrame({"y": np.asarray(outcomes, dtype=float)})

        return self

    def _get_covariates(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "x0": self._rng.normal(size=self.n),
                "x1": self._rng.normal(size=self.n),
            }
        )

    def _get_treatment_score(self, covariates) -> np.ndarray:
        covariate_frame = _as_covariate_frame(covariates)
        return (
            covariate_frame["x0"].to_numpy()
            + self.score_x1_weight * covariate_frame["x1"].to_numpy()
            + self._rng.normal(scale=self.treatment_noise, size=covariate_frame.height)
        )

    def _get_treatments(self, covariates) -> pl.DataFrame:
        score = self._get_treatment_score(covariates)
        labels = np.where(
            score > self.treatment_threshold,
            "treated",
            np.where(score < -self.treatment_threshold, "placebo", "control"),
        )
        return self._to_polars(pl.DataFrame({"treatment": labels}))

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_frame = _as_covariate_frame(covariates)
        labels = _as_treatment_labels(treatments)
        treatment_effect = np.array(
            [self.treatment_effects[label] for label in labels], dtype=float
        )
        return (
            self.covariate_effect * covariate_frame["x0"].to_numpy() + treatment_effect
        )

    def _get_outcomes(self, covariates, treatments) -> pl.DataFrame:
        expected_outcomes = self.predict_y(covariates=covariates, treatments=treatments)
        noisy_outcomes = expected_outcomes + self._rng.normal(
            scale=self.outcome_noise,
            size=expected_outcomes.shape,
        )
        return pl.DataFrame({"y": np.asarray(noisy_outcomes, dtype=float).reshape(-1)})

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {"n": 64, "random_state": 7},
            {
                "n": 32,
                "treatment_noise": 0.3,
                "outcome_noise": 0.1,
                "random_state": 11,
            },
        ]
