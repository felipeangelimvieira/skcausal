from skbase.base import BaseObject

__all__ = ["BaseDensityMetric"]


class BaseDensityMetric(BaseObject):
    """Base class for density estimator evaluation metrics."""

    _tags = {
        "lower_is_better": True,
    }

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Human-readable metric name used in evaluation outputs."""
        return getattr(self, "_name", type(self).__name__)

    def evaluate(self, density_estimator, X, t):
        """Evaluate the density estimator on the supplied data."""
        return self._evaluate(density_estimator, X, t)

    def _evaluate(self, density_estimator, X, t):
        """Concrete metric implementation."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __call__(self, density_estimator, X, t):
        return self.evaluate(density_estimator, X, t)
