import numpy as np

from skcausal.datatypes import collect_column_types
from skcausal.density.base import BaseDensityEstimator

__all__ = ["NaiveDensityEstimator"]


class NaiveDensityEstimator(BaseDensityEstimator):
    """Density estimator based on the marginal treatment distribution.

    Continuous treatment columns are modeled with a univariate Gaussian fitted
    to the observed treatment values. Categorical treatment columns are modeled
    with Laplace-smoothed empirical frequencies. Joint densities are computed as
    the product of per-column marginals.
    """

    _tags = {
        "capability:multidimensional_treatment": True,
        "density_kind": "conditional",
    }

    def __init__(self, density_kind="conditional", epsilon=1e-12):
        self.density_kind = density_kind
        self.epsilon = epsilon
        super().__init__()

        if self.density_kind not in ["conditional", "stabilized"]:
            raise ValueError(
                f"Invalid density_kind '{self.density_kind}'. "
                "Expected 'conditional' or 'stabilized'."
            )

        self.set_tags(density_kind=self.density_kind)

    def _fit(self, X, t):
        self.column_types_ = collect_column_types(t)
        self.column_params_ = {}

        for column, column_type in self.column_types_.items():
            values = np.asarray(t[column].to_numpy())

            if self.density_kind == "conditional":
                if column_type == "continuous":

                    def fn(params, values):
                        centered = (values.astype(float) - params["mean"]) / params[
                            "std"
                        ]
                        column_density = np.exp(-0.5 * centered**2) / (
                            params["std"] * np.sqrt(2.0 * np.pi)
                        )
                        return column_density

                    self.column_params_[column] = {
                        "mean": float(values.astype(float).mean()),
                        "std": max(
                            float(values.astype(float).std(ddof=0)), self.epsilon
                        ),
                        "fn": fn,
                    }

                elif column_type == "categorical":

                    def fn(params, values):
                        return np.array(
                            [
                                params["counts"].get(value, 1.0) / params["total"]
                                for value in values
                            ],
                            dtype=float,
                        )

                    unique_values = np.unique(values)
                    counts = {value: 1 for value in unique_values}
                    for value in values:
                        counts[value] += 1

                    self.column_params_[column] = {
                        "counts": counts,
                        "total": float(len(values) + len(unique_values)),
                        "fn": fn,
                    }

            if self.density_kind == "stabilized":
                if column_type == "continuous":

                    def fn(params, values):
                        volume = params["max"] - params["min"]
                        value_in_range = (values.astype(float) >= params["min"]) & (
                            values.astype(float) <= params["max"]
                        )
                        stabilized = value_in_range.astype(float) / volume
                        return stabilized

                    self.column_params_[column] = {
                        "min": float(values.astype(float).min()),
                        "max": float(values.astype(float).max()),
                        "fn": fn,
                    }

                elif column_type == "categorical":

                    def fn(params, values):
                        return np.ones(len(values), dtype=float) / params["volume"]

                    unique_values = np.unique(values)
                    self.column_params_[column] = {
                        "volume": len(unique_values),
                        "fn": fn,
                    }

        return self

    def _predict_density(self, X, t):
        density = np.ones((len(t), 1), dtype=float)

        for column, column_type in self.column_types_.items():
            values = np.asarray(t[column].to_numpy())
            params = self.column_params_[column]

            column_density = params["fn"](params, values)

            density *= column_density.reshape(-1, 1)

        return density
