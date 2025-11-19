"""
Evaluate.

"""

from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
import numpy as np
from skcausal.weight_estimators.performance_evaluation.base import (
    BaseBalancingWeightMetric,
)
import polars as pl
import pandas as pd
from skcausal.weight_estimators.synthetic_weight import sample_synthetic_random


class LogLikelihoodMetric(BaseBalancingWeightMetric):
    """
    Log-Likelihood Metric for evaluating balancing weight estimators.


    """

    def __init__(self, n_samples=10, random_state=0):
        self.n_samples = n_samples
        self.random_state = random_state
        super().__init__()

    def _evaluate(self, weight_estimator, X, t):
        """
        Evaluate the performance of the balancing weight estimator on the data.
        """
        balancing_weight_type = weight_estimator.get_tag("balancing_weight_type")
        if balancing_weight_type is None:
            raise ValueError(
                "The balancing weight estimator must have the 'balancing_weight_type' tag set."
            )
        if balancing_weight_type == "stabilized":
            return relative_log_likelihood(weight_estimator, X, t, self.random_state)
        if balancing_weight_type == "propensity_score":
            w = weight_estimator.predict_sample_weight(X, t)
            log_likelihood = -np.log(1 / w).mean()
            return log_likelihood
        raise ValueError(f"Unsupported balancing weight type: {balancing_weight_type}")


class ClassificationMetric(BaseBalancingWeightMetric):
    """
    Classification Metric for evaluating balancing weight estimators.


    """

    def __init__(self, random_state=0, classification_loss="log_loss"):
        self.random_state = random_state
        self.classification_loss = classification_loss
        super().__init__()

    def _evaluate(self, weight_estimator, X, t):
        """
        Evaluate the performance of the balancing weight estimator on the data.
        """
        rng = np.random.default_rng(self.random_state)
        balancing_weight_type = weight_estimator.get_tag("balancing_weight_type")
        assert balancing_weight_type in ["propensity_score", "stabilized"], (
            "The balancing weight estimator must have the 'balancing_weight_type' tag set to "
            "'propensity_score' or 'stabilized'."
        )
        method = "balanced" if balancing_weight_type == "stabilized" else "uniform"
        X_synth, t_synth = sample_synthetic_random(
            rng, X=X, t=t, method=method, dataset_size=X.shape[0]
        )

        # Weight is p_synthetic / p_real
        # r_real is the inverse: p_real / p_synthetic
        r_real = weight_estimator.predict_sample_weight(X, t) ** (-1)
        r_synth = weight_estimator.predict_sample_weight(X_synth, t_synth) ** (-1)

        # The probability P(real) / (P(real) + P(synthetic)) = 1 / (1 + r) * r
        p_real = r_real / (1 + r_real)
        p_synth = r_synth / (1 + r_synth)
        import sklearn.metrics as _metrics

        loss_func = getattr(_metrics, self.classification_loss)
        y_true = np.concatenate([np.ones(p_real.shape[0]), np.zeros(p_synth.shape[0])])
        y_pred = np.concatenate([p_real, p_synth])
        if "score" in self.classification_loss:
            # For score functions, higher is better, so we return negative
            return -loss_func(y_true, y_pred)
        return loss_func(y_true, y_pred)


def relative_log_likelihood(
    weight_estimator: BaseBalancingWeightRegressor, X, t, random_state=0
):
    """
    Compute the relative log-likelihood of the treatment
    given the covariates, weighted by the balancing weights.

    Formula:

    w = P(T) / P(T|X)
    r = w^-1

    P(T|X) = P(T) * r / Z(r), where Z(r) = E_{T ~ P(T)}[r]

    So

    $$
    NegLogLikelihood = -\mathbb{E}_{(X,T) \sim P(T,X)}[\log P(T|X)]
     = -\mathbb{E}_{(X,T) \sim P(T,X)}[\log P(T) + \log r - \log Z(r)]
     = -\mathbb{E}_{(X,T) \sim P(T,X)}[\log r - \log Z(r)]
    $$

    """
    # Randomly sample multiple t for same X to estimate Z(r)

    n_samples = 10
    rng = np.random.default_rng(random_state)

    X_repeated = _repeat(X, n_samples)
    t_repeated = _repeat(t, n_samples)
    idx_perm = rng.permutation(np.arange(t_repeated.shape[0]))
    t_repeated = t_repeated[idx_perm]
    r_monte_carlo = weight_estimator.predict_sample_weight(X_repeated, t_repeated) ** -1

    r = weight_estimator.predict_sample_weight(X, t) ** -1

    Z = r_monte_carlo.reshape(-1, X.shape[0]).transpose((1, 0)).mean(axis=1)
    loss = -np.log(r) + np.log(Z)
    return loss.mean()


def _repeat(X, n: int):
    if isinstance(X, np.ndarray):
        return np.repeat(X, n, axis=0)
    elif isinstance(X, pl.DataFrame):
        return pl.concat([X] * n, how="vertical")
    elif isinstance(X, pl.Series):
        return pl.concat([X] * n, how="vertical")
    elif isinstance(X, pd.DataFrame):
        return pd.concat([X] * n, axis=0, ignore_index=True)
    elif isinstance(X, pd.Series):
        return pd.concat([X] * n, axis=0, ignore_index=True)
    else:
        raise ValueError(f"Unsupported data type: {type(X)}")
