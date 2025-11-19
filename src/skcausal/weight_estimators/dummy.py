from skcausal.weight_estimators.base import BaseBalancingWeightRegressor

import numpy as np


class DummyWeightEstimator(BaseBalancingWeightRegressor):
    """A weight estimator that does nothing (returns 1)."""

    _tags = {
        "balancing_weight_type": "stabilized",
    }

    def __init__(self, random_state=0):
        self.random_state = random_state

        super(DummyWeightEstimator, self).__init__()

    def _fit(self, X, t):
        pass

    def _predict_sample_weight(self, X, t):
        out = np.ones(len(X)).reshape(-1, 1)

        # Add random noise
        rng = np.random.default_rng(self.random_state)
        noise = rng.normal(loc=0.0, scale=0.001, size=out.shape)
        out += noise
        return out
