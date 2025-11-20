import copy
from typing import List, Callable
import numpy as np
from tqdm import tqdm
from skcausal.utils.polars import ALL_DTYPES, convert_categorical_to_dummies
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from skcausal.weight_estimators.performance_evaluation.base import (
    BaseBalancingWeightMetric,
)
import pandas as pd
import optuna
from sklearn.model_selection import KFold, BaseCrossValidator


class OptunaSearchWeightRegressor(BaseBalancingWeightRegressor):
    """
    Optuna-based hyperparameter tuner for balancing weight regressors.

    Parameters
    ----------
    weight_estimator_class : BaseBalancingWeightRegressor
        The balancing weight regressor class to be tuned.
    param_distributions : dict
        A dictionary where keys are hyperparameter names (str) and values are
        callables that take a random state and return a sampled hyperparameter value.
    n_trials : int
        Number of hyperparameter trials to perform.
    metric : Callable
        A callable that takes (weight_estimator, X, t) and returns a float metric
        to be maximized.
    random_state : int
        Random state for reproducibility.

    """

    _tags = {
        "t_inner_mtype": pd.DataFrame,
        "one_hot_encode_enum_columns": False,
        "supported_t_dtypes": ALL_DTYPES,
        "balancing_weight_type": None,
    }

    def __init__(
        self,
        weight_estimator: BaseBalancingWeightRegressor,
        metric: BaseBalancingWeightMetric,
        param_distributions: dict,
        n_evals: int,
        cv: BaseCrossValidator = None,
        random_state=0,
        sampler: str = "TPESampler",
    ):
        self.weight_estimator = weight_estimator
        self.param_distributions = param_distributions
        self.n_evals = n_evals
        self.metric = metric
        self.random_state = random_state
        self.sampler = sampler
        self.cv = cv
        super().__init__()

        self._cv = cv
        if cv is None:
            self._cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

    def _tune(self, X, t):
        self.sampler_ = getattr(optuna.samplers, self.sampler)(seed=self.random_state)
        all_results = []
        self.study_ = optuna.create_study(direction="minimize", sampler=self.sampler_)

        pbar = tqdm(range(self.n_evals))
        for _ in pbar:
            trial = self.study_.ask(self.param_distributions)
            params = {
                name: trial.params[name] for name, v in self.param_distributions.items()
            }

            estimator_ = self.weight_estimator.clone().set_params(**params)
            status = optuna.trial.TrialState.COMPLETE
            try:
                scores = evaluate_cv(estimator_, X, t, self._cv, self.metric)
                score = np.mean(scores)
                all_results.append(
                    {
                        "params": params,
                        "score": score,
                    }
                )
            except Exception as e:
                status = optuna.trial.TrialState.FAIL
                all_results.append(
                    {
                        "params": params,
                        "score": None,
                    }
                )

            self.study_.tell(trial, score, state=status)

            if score is not None and np.isfinite(score):
                # Update progress bar with the current best trial score and its number
                if self.study_.best_trial:
                    pbar.set_postfix(
                        {
                            "best_score": self.study_.best_trial.value,
                            "best_trial": self.study_.best_trial.number,
                            "last_score": score,
                        }
                    )

        self.trials_dataframe_ = self.study_.trials_dataframe()

        self.best_params_ = self.study_.best_params
        self.best_trial_ = self.study_.best_trial

        self.best_weight_estimator_ = self.weight_estimator.clone().set_params(
            **self.best_params_
        )
        self.all_results_ = pd.DataFrame(all_results)

    def _fit(self, X, t):
        self._tune(X, t)
        self.fitted_model_ = self.best_weight_estimator_.clone()
        self.fitted_model_.fit(X, t)
        return self

    def _predict_sample_weight(self, X, t):
        return self.fitted_model_.predict_sample_weight(X, t)

    @classmethod
    def get_test_params(cls, parameter_set="default"):

        from skcausal.weight_estimators.dummy import DummyWeightEstimator
        from skcausal.weight_estimators.performance_evaluation.evaluate import (
            ClassificationMetric,
        )

        return [
            {
                "weight_estimator": DummyWeightEstimator(),
                "metric": ClassificationMetric(),
                "param_distributions": {
                    "random_state": optuna.distributions.IntUniformDistribution(0, 10),
                },
                "n_evals": 2,
                "random_state": 0,
            }
        ]


def evaluate_cv(estimator, X, t, cv, metric):
    scores = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        t_train, t_test = t[train_idx], t[test_idx]

        estimator_clone = estimator.clone()
        estimator_clone.fit(X_train, t_train)
        score = metric(estimator_clone, X_test, t_test)
        scores.append(score)
    return scores
