from importlib import resources
from pathlib import Path

import numpy as np
import polars as pl

from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["IHDPContinuous"]

_COVARIATE_COLUMNS = [f"x{i}" for i in range(1, 26)]
_EXPECTED_RAW_SHAPE = (747, 30)
_RAW_COVARIATE_START = 5
_S_DIS_1 = np.array([4, 7, 8, 9, 10, 11, 12, 13, 14, 15]) - 1
_S_DIS_2 = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25]) - 1
_EPSILON = 1e-8


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_values))


def _as_2d_float_array(values, *, expected_width: int, name: str) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        values = values.to_numpy()
    elif hasattr(values, "to_numpy"):
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if expected_width == 1:
            array = array.reshape(-1, 1)
        elif array.size == expected_width:
            array = array.reshape(1, -1)
        else:
            raise ValueError(
                f"{name} expects rows with exactly {expected_width} values."
            )

    if array.ndim != 2 or array.shape[1] != expected_width:
        raise ValueError(
            f"{name} expects inputs with shape (n_samples, {expected_width})."
        )

    return array


def _as_treatment_array(values, *, name: str) -> np.ndarray:
    return _as_2d_float_array(values, expected_width=1, name=name)[:, 0]


def _stabilize_denominator(values, *, epsilon: float = _EPSILON) -> np.ndarray:
    denominator = np.asarray(values, dtype=float).copy()
    small = np.abs(denominator) < epsilon
    denominator[small] = np.where(denominator[small] < 0.0, -epsilon, epsilon)
    return denominator


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / span


def _standardize_covariates(covariates: np.ndarray) -> np.ndarray:
    means = covariates.mean(axis=0)
    scales = covariates.std(axis=0, ddof=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return (covariates - means) / scales


def _default_source_path() -> Path:
    return Path(resources.files("skcausal.datasets").joinpath("raw/ihdp.csv"))


def _load_ihdp_covariates(source_path: str | Path | None) -> tuple[np.ndarray, Path]:
    path = _default_source_path() if source_path is None else Path(source_path)
    raw = np.loadtxt(path, delimiter=",", dtype=float)

    if raw.shape != _EXPECTED_RAW_SHAPE:
        raise ValueError(
            "IHDP raw data must contain exactly 747 rows and 30 columns: "
            "the original binary treatment column, four outcome columns, and "
            "25 covariates."
        )

    treatment_values = np.unique(raw[:, 0])
    if not np.all(np.isin(treatment_values, [0.0, 1.0])):
        raise ValueError(
            "IHDP raw data must use a binary first column for the original "
            "treatment indicator."
        )

    covariates = raw[:, _RAW_COVARIATE_START:]
    if covariates.shape[1] != len(_COVARIATE_COLUMNS):
        raise ValueError("IHDP raw data must expose exactly 25 covariate columns.")

    return covariates, path


class IHDPContinuous(BaseSyntheticDataset):
    r"""Semi-synthetic IHDP benchmark with generated continuous treatment.

    This dataset reads the standard Hill IHDP benchmark CSV with shape
    :math:`747 \times 30`, discards the original treatment and outcome columns,
    standardizes the 25 covariates, and then generates a continuous treatment and
    outcome according to the ADRF benchmark paper. The treatment mechanism is
    driven by the continuous features and the ``S_{\mathrm{dis},2}`` block, while
    the structural outcome depends on the continuous features and the
    ``S_{\mathrm{dis},1}`` block.

    The raw file is validated on load to ensure it still matches the expected
    IHDP layout before the semi-synthetic responses are generated. This class
    exposes the full 747-row semi-synthetic dataset and leaves train/test
    splitting to downstream split objects.
    """

    column_types = {"t": "continuous"}

    def __init__(
        self,
        random_state: int = 42,
        source_path: str | Path | None = None,
        minmax_normalize_t: bool = True,
        treatment_noise_variance: float = 0.25,
        outcome_noise_variance: float = 0.25,
    ):
        if treatment_noise_variance < 0.0 or outcome_noise_variance < 0.0:
            raise ValueError("Noise variances must be non-negative.")

        self.source_path = source_path
        self.minmax_normalize_t = minmax_normalize_t
        self.treatment_noise_variance = treatment_noise_variance
        self.outcome_noise_variance = outcome_noise_variance
        self.source_path_ = None
        self.c1_ = None
        self.c2_ = None

        super().__init__(n=_EXPECTED_RAW_SHAPE[0], random_state=random_state)
        self._prepare()

    def _covariate_frame(self, covariates: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame(
            {
                column: covariates[:, idx]
                for idx, column in enumerate(_COVARIATE_COLUMNS)
            }
        )

    def _treatment_frame(self, treatments: np.ndarray) -> pl.DataFrame:
        return self._to_polars(
            pl.DataFrame({"t": np.asarray(treatments, dtype=float).reshape(-1)})
        )

    def _outcome_frame(self, outcomes: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({"y": np.asarray(outcomes, dtype=float).reshape(-1)})

    def _get_covariates(self) -> np.ndarray:
        covariates, _ = _load_ihdp_covariates(self.source_path)
        return _standardize_covariates(covariates)

    def _treatment_latent_score(self, covariates) -> np.ndarray:
        covariate_array = _as_2d_float_array(
            covariates,
            expected_width=len(_COVARIATE_COLUMNS),
            name=type(self).__name__,
        )
        x1, x2, x3, _, x5, x6 = covariate_array[:, :6].T
        c2 = self.c2_
        if c2 is None:
            c2 = covariate_array[:, _S_DIS_2].mean(axis=1).mean()

        return (
            2.0 * x1 / _stabilize_denominator(1.0 + x2)
            + 2.0
            * np.maximum.reduce([x3, x5, x6])
            / _stabilize_denominator(0.2 + np.minimum.reduce([x3, x5, x6]))
            + 2.0
            * np.tanh(
                5.0 * ((covariate_array[:, _S_DIS_2] - c2).sum(axis=1) / len(_S_DIS_2))
            )
            - 4.0
        )

    def _get_treatments(self, covariates) -> pl.DataFrame:
        covariate_array = _as_2d_float_array(
            covariates,
            expected_width=len(_COVARIATE_COLUMNS),
            name=type(self).__name__,
        )
        latent_score = self._treatment_latent_score(covariate_array)
        latent_score = latent_score + self._rng.normal(
            scale=np.sqrt(self.treatment_noise_variance),
            size=covariate_array.shape[0],
        )
        treatments = _sigmoid(latent_score)
        if self.minmax_normalize_t:
            treatments = _minmax_normalize(treatments)
        return self._treatment_frame(treatments)

    def _predict_y(self, covariates, treatments) -> np.ndarray:
        covariate_array = _as_2d_float_array(
            covariates,
            expected_width=len(_COVARIATE_COLUMNS),
            name=type(self).__name__,
        )
        treatment_array = _as_treatment_array(
            treatments,
            name=f"{type(self).__name__} treatments",
        )

        x1, x2, x3, _, x5, x6 = covariate_array[:, :6].T
        c1 = self.c1_
        if c1 is None:
            c1 = covariate_array[:, _S_DIS_1].mean(axis=1).mean()

        return (
            np.sin(3.0 * np.pi * treatment_array)
            / _stabilize_denominator(1.2 - treatment_array)
            * (
                np.tanh(
                    5.0
                    * ((covariate_array[:, _S_DIS_1] - c1).sum(axis=1) / len(_S_DIS_1))
                )
                + np.exp(0.2 * (x1 - x6))
                / _stabilize_denominator(0.5 + 5.0 * np.minimum.reduce([x2, x3, x5]))
            )
        )

    def _get_outcomes(self, covariates, treatments) -> pl.DataFrame:
        mean_outcomes = self._predict_y(covariates, treatments)
        outcomes = mean_outcomes + self._rng.normal(
            scale=np.sqrt(self.outcome_noise_variance),
            size=mean_outcomes.shape[0],
        )
        return self._outcome_frame(outcomes)

    def _prepare(self, n: int = None):
        covariates, source_path = _load_ihdp_covariates(self.source_path)
        covariates = _standardize_covariates(covariates)
        self.source_path_ = source_path
        self.c1_ = covariates[:, _S_DIS_1].mean(axis=1).mean()
        self.c2_ = covariates[:, _S_DIS_2].mean(axis=1).mean()

        covariate_frame = self._covariate_frame(covariates)
        treatment_frame = self._get_treatments(covariates)
        outcome_frame = self._get_outcomes(covariates, treatment_frame)

        self._covariates = covariate_frame
        self._treatments = treatment_frame
        self._outcomes = outcome_frame
        self.n = self._covariates.height

        return self

    def get_grid(self, n: int = 100) -> pl.DataFrame:
        return self._coerce_treatment_frame(
            pl.DataFrame({"t": np.linspace(0.0, 1.0, n)})
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"random_state": 7}]
