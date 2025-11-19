from skbase.base import BaseObject


class BaseMetric(BaseObject): ...


class BaseBalancingWeightMetric(BaseMetric):
    """
    Base class for balancing weight metrics.

    Balancing weight metrics are used to evaluate the performance of balancing
    weight estimators.

    Main methods
    ------------
    evaluate(weight_estimator, X, t)
        Evaluate the performance of the balancing weight estimator on the data.

    """

    _tags = {}

    def __init__(self):
        super().__init__()

    def evaluate(self, weight_estimator, X, t):
        """
        Evaluate the performance of the balancing weight estimator on the data.

        Abstract method that must be implemented by subclasses.
        """
        return self._evaluate(weight_estimator, X, t)

    def _evaluate(self, weight_estimator, X, t):
        """
        Evaluate the performance of the balancing weight estimator on the data.

        Abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __call__(self, weight_estimator, X, t):
        return self.evaluate(weight_estimator, X, t)
