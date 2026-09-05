"""Multi-dimensional semi-synthetic DGP with fixed source-outcome treatments."""

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.preprocessing import SplineTransformer

from skcausal.datasets.base import BaseDataset
from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _validate_source_datasets,
)

__all__ = ["MultidimSemiSyntheticDataset"]


def _source_prefix(index):
    return f"d{index}_"


def _exchangeable_correlation(n_components, correlation):
    rho = float(correlation)
    if not np.isfinite(rho):
        raise ValueError("treatment_correlation must be finite.")
    if n_components == 1:
        if rho != 0.0:
            raise ValueError(
                "treatment_correlation must be zero with one source dataset."
            )
        return np.ones((1, 1))
    lower = -1.0 / (n_components - 1)
    if not lower < rho < 1.0:
        raise ValueError(
            "treatment_correlation must satisfy "
            f"{lower:g} < treatment_correlation < 1 for {n_components} sources."
        )
    matrix = np.full((n_components, n_components), rho)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _coupled_orders(targets, correlation, rng):
    arrays = [np.asarray(target, dtype=float).reshape(-1) for target in targets]
    if not arrays or len({array.size for array in arrays}) != 1:
        raise ValueError("Coupling requires equally sized source targets.")
    covariance = _exchangeable_correlation(len(arrays), correlation)
    latent = rng.multivariate_normal(
        np.zeros(len(arrays)), covariance, size=arrays[0].size
    ).reshape(arrays[0].size, len(arrays))
    orders = []
    for index, target in enumerate(arrays):
        latent_order = np.argsort(latent[:, index], kind="stable")
        source_order = np.lexsort((rng.random(target.size), target))
        order = np.empty(target.size, dtype=int)
        order[latent_order] = source_order
        orders.append(order)
    return orders, latent


def _numeric_column(frame, name, label):
    series = frame.get_column(name)
    if not series.dtype.is_numeric():
        raise ValueError(f"The {label} must be numeric.")
    values = series.cast(pl.Float64).to_numpy().astype(float)
    if not np.isfinite(values).all():
        raise ValueError(f"The {label} must be finite.")
    if np.unique(values).size < 2:
        raise ValueError(f"The {label} must have at least two distinct values.")
    return values


def _correlation_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(values, rowvar=False)


class MultidimSemiSyntheticDataset(BaseSemiSyntheticDataset):
    """A fixed multi-source treatment dataset with regression-score confounding."""

    def __init__(
        self,
        real_datasets,
        regressors,
        treatment_correlation=0.0,
        confounding_concentration=None,
        n_spline_knots=6,
        spline_degree=3,
        treatment_effect_scale=1.0,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(real_datasets, BaseDataset):
            raise TypeError("real_datasets must be a non-empty sequence of datasets.")
        datasets = _validate_source_datasets(real_datasets, name="real_datasets")
        _exchangeable_correlation(len(datasets), treatment_correlation)
        if confounding_concentration is not None and (
            not np.isfinite(float(confounding_concentration))
            or float(confounding_concentration) <= 0.0
        ):
            raise ValueError("confounding_concentration must be positive and finite.")
        if not isinstance(n_spline_knots, (int, np.integer)) or n_spline_knots < 2:
            raise ValueError("n_spline_knots must be an integer at least 2.")
        if not isinstance(spline_degree, (int, np.integer)) or spline_degree < 0:
            raise ValueError("spline_degree must be a non-negative integer.")
        for name, value in {
            "treatment_effect_scale": treatment_effect_scale,
            "outcome_noise_scale": outcome_noise_scale,
        }.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        self.real_datasets = real_datasets
        self.regressors = regressors
        self.treatment_correlation = treatment_correlation
        self.confounding_concentration = confounding_concentration
        self.n_spline_knots = n_spline_knots
        self.spline_degree = spline_degree
        self.treatment_effect_scale = treatment_effect_scale
        self.outcome_noise_scale = outcome_noise_scale
        super().__init__(real_dataset=datasets, random_state=random_state)

    def _combine_sources(self, sources, rng):
        n_rows = min(X.height for X, _, _ in sources)
        selected_X, selected_y, selected_rows = [], [], []
        for index, (X, _, outcome) in enumerate(sources):
            outcome = self._coerce_backend_frame(outcome, backend="polars")
            if outcome.width != 1:
                raise ValueError(
                    f"Source dataset {index} must provide exactly one outcome column."
                )
            if outcome.height != X.height:
                raise ValueError(
                    f"Source dataset {index} must provide one outcome per covariate row."
                )
            y = _numeric_column(
                outcome, outcome.columns[0], f"outcome of source dataset {index}"
            )
            rows = (
                np.arange(X.height)
                if X.height == n_rows
                else np.sort(rng.choice(X.height, n_rows, replace=False))
            )
            if np.unique(y[rows]).size < 2:
                raise ValueError(
                    f"Selected outcome of source dataset {index} must have at least "
                    "two distinct values."
                )
            selected_X.append(X[rows])
            selected_y.append(y[rows])
            selected_rows.append(rows)
        orders, self.copula_latent_ = _coupled_orders(
            selected_y, self.treatment_correlation, rng
        )
        self.source_row_indices_ = [
            rows[order] for rows, order in zip(selected_rows, orders)
        ]
        self.source_feature_names_ = [list(frame.columns) for frame in selected_X]
        self.source_columns_ = [
            [f"d{j}_{name}" for name in names]
            for j, names in enumerate(self.source_feature_names_)
        ]
        frames = [
            frame[order].rename(dict(zip(frame.columns, self.source_columns_[j])))
            for j, (frame, order) in enumerate(zip(selected_X, orders))
        ]
        coupled_y = np.column_stack(
            [values[order] for values, order in zip(selected_y, orders)]
        )
        self.treatment_columns_ = [f"t_{j}" for j in range(len(sources))]
        self.column_types = {name: "continuous" for name in self.treatment_columns_}
        self._fixed_treatments = pl.DataFrame(
            dict(zip(self.treatment_columns_, coupled_y.T))
        )
        self.treatment_correlation_matrix_ = _exchangeable_correlation(
            len(sources), self.treatment_correlation
        )
        self.achieved_pearson_correlation_ = _correlation_matrix(coupled_y)
        self.achieved_spearman_correlation_ = _correlation_matrix(
            rankdata(coupled_y, axis=0)
        )
        return pl.concat(frames, how="horizontal_extend")

    def _regressor_list(self):
        if hasattr(self.regressors, "fit") and hasattr(self.regressors, "predict"):
            configured = [self.regressors] * len(self.source_columns_)
        else:
            try:
                configured = list(self.regressors)
            except TypeError as error:
                raise TypeError(
                    "regressors must be one estimator or one estimator per source."
                ) from error
            if len(configured) != len(self.source_columns_):
                raise ValueError("regressors must contain one estimator per source.")
        try:
            return [clone(regressor) for regressor in configured]
        except TypeError as error:
            raise TypeError(
                "Every regressor must be a cloneable sklearn estimator."
            ) from error

    def _source_frames(self, X):
        return [
            X.select(prefixed)
            .rename(dict(zip(prefixed, original)))
            .to_pandas()
            for prefixed, original in zip(
                self.source_columns_, self.source_feature_names_
            )
        ]

    def _score_matrix(self, X):
        columns = []
        for regressor, frame, center, scale in zip(
            self.regressors_,
            self._source_frames(X),
            self.score_means_,
            self.score_scales_,
        ):
            prediction = np.asarray(regressor.predict(frame), dtype=float).reshape(-1)
            if prediction.size != X.height or not np.isfinite(prediction).all():
                raise ValueError("Each regressor must predict one finite value per row.")
            columns.append((prediction - center) / scale)
        return np.column_stack(columns)

    def _prepare_dgp(self, X, rng):
        self.regressors_ = self._regressor_list()
        self.target_means_ = self._fixed_treatments.to_numpy().mean(axis=0)
        self.target_scales_ = self._fixed_treatments.to_numpy().std(axis=0, ddof=0)
        frames = self._source_frames(X)
        fitted = []
        for index, (regressor, frame) in enumerate(zip(self.regressors_, frames)):
            normalized_y = (
                self._fixed_treatments
                .get_column(self.treatment_columns_[index])
                .to_numpy()
                - self.target_means_[index]
            ) / self.target_scales_[index]
            random_states = {
                name: int(rng.integers(np.iinfo(np.int32).max))
                for name, value in regressor.get_params(deep=True).items()
                if name.endswith("random_state") and value is None
            }
            if random_states:
                regressor.set_params(**random_states)
            regressor.fit(frame, normalized_y)
            prediction = np.asarray(regressor.predict(frame), dtype=float).reshape(-1)
            if (
                prediction.size != self.n
                or not np.isfinite(prediction).all()
                or prediction.std(ddof=0) <= 0.0
            ):
                raise ValueError(
                    "Each fitted regressor must produce nonconstant finite predictions."
                )
            fitted.append(prediction)
        fitted = np.column_stack(fitted)
        self.score_means_ = fitted.mean(axis=0)
        self.score_scales_ = fitted.std(axis=0, ddof=0)
        self.source_scores_ = (fitted - self.score_means_) / self.score_scales_
        if self.confounding_concentration is None:
            self.confounding_weights_ = np.full(
                len(self.regressors_), 1.0 / len(self.regressors_)
            )
        else:
            self.confounding_weights_ = rng.dirichlet(
                np.full(len(self.regressors_), float(self.confounding_concentration))
            )
        raw = self.source_scores_ @ np.sqrt(self.confounding_weights_)
        self.baseline_mean_ = float(raw.mean())
        self.baseline_scale_ = float(raw.std(ddof=0))
        if not np.isfinite(self.baseline_scale_) or self.baseline_scale_ <= 0.0:
            raise ValueError("The weighted confounding baseline must be nonconstant.")
        self.source_mu0_ = (raw - self.baseline_mean_) / self.baseline_scale_
        self._fit_treatment_effect(rng)

    def _baseline(self, X):
        raw = self._score_matrix(X) @ np.sqrt(self.confounding_weights_)
        return (raw - self.baseline_mean_) / self.baseline_scale_

    def _fit_treatment_effect(self, rng):
        observed = self._fixed_treatments.to_numpy()
        self.spline_transformers_ = []
        self.spline_coefficients_ = []
        raw_effect = np.zeros(self.n)
        for values in observed.T:
            unique_count = np.unique(np.round(values, 12)).size
            transformer = SplineTransformer(
                n_knots=min(self.n_spline_knots, max(2, unique_count)),
                degree=self.spline_degree,
                include_bias=False,
                extrapolation="continue",
            )
            basis = transformer.fit_transform(values.reshape(-1, 1))
            coefficients = rng.normal(
                scale=1.0
                / np.sqrt(len(self.treatment_columns_) * basis.shape[1]),
                size=basis.shape[1],
            )
            self.spline_transformers_.append(transformer)
            self.spline_coefficients_.append(coefficients)
            raw_effect += basis @ coefficients
        self.treatment_effect_offset_ = float(raw_effect.mean())
        self.treatment_ranges_ = np.column_stack(
            (observed.min(axis=0), observed.max(axis=0))
        )

    def _evaluate_treatment_effect(self, treatment):
        values = treatment.select(self.treatment_columns_).to_numpy()
        raw = sum(
            transformer.transform(values[:, [index]]) @ coefficients
            for index, (transformer, coefficients) in enumerate(
                zip(self.spline_transformers_, self.spline_coefficients_)
            )
        )
        return self.treatment_effect_scale * (raw - self.treatment_effect_offset_)

    def _sample_treatment(self, X, rng):
        return self._fixed_treatments

    def sample(self, random_state):
        raise NotImplementedError(
            "MultidimSemiSyntheticDataset is fixed at construction; create a new "
            "instance with another random_state instead."
        )

    def log_prob(self, X, treatment):
        raise NotImplementedError(
            "Mean regressors do not define p(T | X), so this dataset does not "
            "define log_prob."
        )

    def _log_prob(self, X, treatment):
        raise NotImplementedError

    def _predict_y(self, X, treatment):
        return self._baseline(X) + self._evaluate_treatment_effect(treatment)

    def _sample_outcome(self, mean, X, treatment, rng):
        return np.asarray(mean).reshape(-1) + rng.normal(
            scale=self.outcome_noise_scale, size=X.height
        )

    def _get_grid(self, n):
        points = max(2, round(n ** (1.0 / len(self.treatment_columns_))))
        grid = None
        for name, (lower, upper) in zip(
            self.treatment_columns_, self.treatment_ranges_
        ):
            component = pl.DataFrame({name: np.linspace(lower, upper, points)})
            grid = component if grid is None else grid.join(component, how="cross")
        return grid

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sklearn.linear_model import LinearRegression

        from skcausal.datasets.kang_schafer import KangSchaferContinuous

        return [
            {
                "real_datasets": [
                    KangSchaferContinuous(n=64, random_state=4),
                    KangSchaferContinuous(n=80, random_state=5),
                ],
                "regressors": LinearRegression(),
                "treatment_correlation": 0.4,
                "random_state": 7,
            }
        ]
