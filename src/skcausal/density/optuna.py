import math
import time

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, check_cv

from skcausal.density.base import BaseDensityEstimator
from skcausal.density.performance_evaluation.metrics.likelihood import (
    LogLikelihoodMetric,
)
from skcausal.density.performance_evaluation.metrics import BaseDensityMetric

__all__ = ["OptunaSearchDensityEstimator"]


class OptunaSearchDensityEstimator(BaseDensityEstimator):
    """Optuna-based hyperparameter tuner for density estimators.

    This estimator wraps another density estimator and performs hyperparameter
    search with Optuna over cross-validation splits. Candidate parameter
    settings are evaluated with a density metric, by default
    ``LogLikelihoodMetric``.

    Parameters
    ----------
    estimator : BaseDensityEstimator
        Base density estimator to tune.
    metric : BaseDensityMetric or None, default=None
        Metric used to score candidate estimators. If ``None``,
        ``LogLikelihoodMetric()`` is used.
    param_distributions : dict, list of dict, or None, default=None
        Search space specification.

        Supported forms for each parameter value are:

        * ``callable(trial) -> value`` for fully custom Optuna sampling
        * ``list``/``tuple``/1D ``numpy.ndarray`` for categorical sampling
        * Optuna distribution instances

        If a top-level list of dicts is passed, one dict is selected per trial,
        enabling mixture-of-spaces search.
    n_trials : int, default=50
        Maximum number of Optuna trials to execute.
    cv : cross-validation splitter or None, default=None
        Cross-validation splitter used to evaluate candidates. If ``None``, a
        3-fold shuffled ``KFold`` is used.
    refit : bool, default=True
        If ``True``, the best estimator is refit on the full dataset after the
        search. Predictions require ``refit=True``.
    random_state : int, default=0
        Random seed used for the default CV splitter and Optuna sampler.
    sampler : str, Optuna sampler, or None, default="TPESampler"
        Sampler used by Optuna. If a string is provided, it must match a class
        name in ``optuna.samplers``.
    error_score : {"raise", float}, default=np.nan
        Value assigned to failed trials. If set to ``"raise"``, exceptions from
        candidate evaluation are propagated. If a non-finite numeric value is
        used, failed trials are marked as failed in Optuna instead of completed.
    max_duration : float or None, default=None
        Maximum search duration in minutes. Once the elapsed wall-clock time
        reaches this budget, no new trials are started and the best completed
        trial found so far is returned.

    Attributes
    ----------
    study_ : optuna.Study
        Underlying Optuna study.
    cv_results_ : pandas.DataFrame
        Per-trial results including sampled parameters, fold scores, mean score,
        and rank.
    trials_dataframe_ : pandas.DataFrame
        Optuna's trial summary dataframe.
    best_index_ : int
        Row index of the best trial in ``cv_results_``.
    best_score_ : float
        Best cross-validated score.
    best_params_ : dict
        Best parameter setting.
    best_estimator_ : BaseDensityEstimator
        Refit best estimator when ``refit=True``.
    """

    _tags = {
        "soft_dependencies": ["optuna"],
        "density_kind": "conditional",
    }

    def __init__(
        self,
        estimator: BaseDensityEstimator,
        metric: BaseDensityMetric,
        param_distributions=None,
        n_trials=50,
        cv=None,
        refit=True,
        random_state=0,
        sampler="TPESampler",
        error_score=np.nan,
        max_duration=None,
    ):
        self.estimator = estimator
        self.metric = metric
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.cv = cv
        self.refit = refit
        self.random_state = random_state
        self.sampler = sampler
        self.error_score = error_score
        self.max_duration = max_duration

        super().__init__()

        self.set_tags(
            supported_t_dtypes=estimator.get_tag(
                "supported_t_dtypes", self.get_tag("supported_t_dtypes")
            ),
            **{
                "capability:multidimensional_treatment": estimator.get_tag(
                    "capability:multidimensional_treatment",
                    self.get_tag("capability:multidimensional_treatment"),
                ),
                "density_kind": estimator.get_tag(
                    "density_kind", self.get_tag("density_kind")
                ),
            },
        )

        if cv is None:
            self._cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        else:
            self._cv = check_cv(cv)

    def _get_timeout_seconds(self):
        if self.max_duration is None:
            return None

        try:
            timeout_seconds = float(self.max_duration) * 60.0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_duration must be convertible to float minutes."
            ) from exc

        if timeout_seconds <= 0:
            raise ValueError(f"max_duration must be positive, got {self.max_duration}.")

        return timeout_seconds

    def _resolve_search_space(self, trial):
        # soft_dependencies tag is checked by BaseDensityEstimator.__init__,
        # so optuna is guaranteed to be available here.
        if self.param_distributions is None:
            raise ValueError("param_distributions must be provided.")

        if isinstance(self.param_distributions, (list, tuple)):
            if len(self.param_distributions) == 0:
                raise ValueError("param_distributions list cannot be empty.")
            idx = trial.suggest_int(
                "__optuna_space_index__", 0, len(self.param_distributions) - 1
            )
            space = self.param_distributions[idx]
        else:
            space = self.param_distributions

        if not hasattr(space, "items"):
            raise ValueError(
                "param_distributions must be a dict or a list/tuple of dicts."
            )

        params = {}
        for name, spec in space.items():
            if callable(spec):
                params[name] = spec(trial)
                continue

            if isinstance(spec, (list, tuple, np.ndarray)):
                if len(spec) == 0:
                    raise ValueError(
                        f"Parameter values for '{name}' need a non-empty sequence."
                    )
                if isinstance(spec, np.ndarray) and spec.ndim > 1:
                    raise ValueError(
                        f"Parameter array for '{name}' should be one-dimensional."
                    )
                params[name] = trial.suggest_categorical(name, list(spec))
                continue

            if "optuna" in type(spec).__module__:
                params[name] = trial._suggest(name, spec)
                continue

            raise ValueError(
                f"Unsupported specification for parameter '{name}': {type(spec)}."
            )

        return params

    def _fit(self, X, t):
        import optuna

        metric = self.metric if self.metric is not None else LogLikelihoodMetric()
        lower_is_better = metric.get_tag("lower_is_better", True, raise_error=False)
        direction = "minimize" if lower_is_better else "maximize"

        if self.sampler is None:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        elif isinstance(self.sampler, str):
            sampler = getattr(optuna.samplers, self.sampler)(seed=self.random_state)
        else:
            sampler = self.sampler

        self.study_ = optuna.create_study(direction=direction, sampler=sampler)

        trial_rows = []
        timeout_seconds = self._get_timeout_seconds()
        search_start_time = time.perf_counter()

        for _ in range(self.n_trials):
            if timeout_seconds is not None:
                elapsed = time.perf_counter() - search_start_time
                if elapsed >= timeout_seconds:
                    break

            trial = self.study_.ask()
            params = self._resolve_search_space(trial)

            try:
                fold_scores = self._evaluate_params(params, X, t, metric)
                mean_score = float(np.mean(fold_scores))
                status = optuna.trial.TrialState.COMPLETE
                trial_value = mean_score
                error_message = None
            except Exception as exc:
                if self.error_score == "raise":
                    raise
                fold_scores = []
                mean_score = float(self.error_score)
                if np.isfinite(mean_score):
                    status = optuna.trial.TrialState.COMPLETE
                    trial_value = mean_score
                else:
                    status = optuna.trial.TrialState.FAIL
                    trial_value = None
                error_message = repr(exc)

            self.study_.tell(trial, trial_value, state=status)
            trial_rows.append(
                {
                    "number": trial.number,
                    "params": params,
                    "mean_score": mean_score,
                    "fold_scores": fold_scores,
                    "error": error_message,
                }
            )

        self.cv_results_ = self._build_results_frame(trial_rows, lower_is_better)
        self.trials_dataframe_ = self.study_.trials_dataframe()

        if len(self.cv_results_) == 0:
            raise RuntimeError("No Optuna trials were executed.")

        rank_column = "rank_score"
        best_row_idx = int(self.cv_results_[rank_column].argmin())
        best_row = self.cv_results_.iloc[best_row_idx]

        self.best_index_ = best_row_idx
        self.best_score_ = float(best_row["mean_score"])
        self.best_params_ = best_row["params"]

        self.best_estimator_ = self.estimator.clone().set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, t)

        return self

    def _predict_density(self, X, t):
        if not self.refit:
            raise RuntimeError(
                "OptunaSearchDensityEstimator requires refit=True for predict_density."
            )
        return self.best_estimator_.predict_density(X, t)

    def _evaluate_params(self, params, X, t, metric):
        # Clone once and reuse across folds. BaseDensityEstimator.fit()
        # fully resets state, so refitting the same instance is safe.
        estimator = self.estimator.clone().set_params(**params)
        scores = []
        for train_idx, test_idx in self._cv.split(X):
            X_train = _take_rows(X, train_idx)
            X_test = _take_rows(X, test_idx)
            t_train = _take_rows(t, train_idx)
            t_test = _take_rows(t, test_idx)
            estimator.fit(X_train, t_train)
            scores.append(metric(estimator, X_test, t_test))
        return scores

    @staticmethod
    def _build_results_frame(trial_rows, lower_is_better):
        import pandas as pd

        results = pd.DataFrame(trial_rows)
        results["rank_score"] = results["mean_score"].rank(
            ascending=lower_is_better,
            method="dense",
        )
        return results

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        class _GaussianToyDensityEstimator(BaseDensityEstimator):
            _tags = {
                "t_inner_mtype": np.ndarray,
                "X_inner_mtype": np.ndarray,
                "supported_t_dtypes": [pl.Float32, pl.Float64],
                "density_kind": "conditional",
            }

            def __init__(self, mean_weight=1.0, bias=0.0, scale=1.0):
                self.mean_weight = mean_weight
                self.bias = bias
                self.scale = scale
                super().__init__()

            def _fit(self, X: np.ndarray, t: np.ndarray):
                return self

            def _predict_density(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
                scale = max(float(self.scale), 1e-6)
                mean = self.bias + self.mean_weight * X[:, [0]]
                z = (t - mean) / scale
                normalizer = scale * math.sqrt(2.0 * math.pi)
                return np.exp(-0.5 * z**2) / normalizer

        return [
            {
                "estimator": _GaussianToyDensityEstimator(),
                "metric": LogLikelihoodMetric(),
                "param_distributions": {
                    "mean_weight": [0.5, 1.0],
                    "scale": [0.5, 1.0],
                },
                "n_trials": 2,
                "random_state": 0,
            }
        ]


def _take_rows(X, indices):
    if isinstance(X, np.ndarray):
        return X[indices]
    if isinstance(X, pl.DataFrame):
        return X[indices]
    if isinstance(X, pl.Series):
        return X[indices]

    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    if isinstance(X, pd.Series):
        return X.iloc[indices]

    raise ValueError(f"Unsupported data type: {type(X)}")
