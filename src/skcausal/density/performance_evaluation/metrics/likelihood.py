import numpy as np

from skcausal.density.performance_evaluation.metrics.base import BaseDensityMetric


class LogLikelihoodMetric(BaseDensityMetric):
    """Average log-likelihood under the estimated density."""

    _tags = {
        "lower_is_better": False,
    }

    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        super().__init__()

    def _evaluate(self, density_estimator, X, t):
        density = density_estimator.predict_density(X, t)
        density = np.asarray(density, dtype=float)
        density = np.clip(density, self.epsilon, None)
        return np.log(density).mean()
