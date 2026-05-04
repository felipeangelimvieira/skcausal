"""Utility for benchmarking one average-response estimator on a dataset."""

import pandas as pd

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.causal_estimators.benchmarking.metrics import AverageResponseMetric
from skcausal.datasets.base import BaseSyntheticDataset

__all__ = ["evaluate_one", "evaluate_multiple_dataset_seeds"]


def evaluate_one(
    dataset: BaseSyntheticDataset,
    estimator: BaseAverageCausalResponseEstimator,
    metrics,
    return_fitted: bool = False,
):
    """Fit one estimator on one synthetic dataset and score it with metrics.

    Parameters
    ----------
    dataset : BaseSyntheticDataset
            Synthetic dataset used for fitting and ground-truth evaluation.
    estimator : BaseAverageCausalResponseEstimator
            Estimator to fit and benchmark.
    metrics : AverageResponseMetric or list of AverageResponseMetric
            Metric or metrics used to evaluate the fitted estimator.
    return_fitted : bool, default=False
            If True, includes the fitted estimator clone in the result.

    Returns
    -------
    pd.DataFrame
            One-row dataframe with one column per metric and, when requested,
            ``fitted_model``.
    """
    _validate_dataset_and_estimator(dataset, estimator)
    metrics = _check_metrics(metrics)

    X, t, y = dataset.load()
    fitted_estimator = estimator.clone()
    fitted_estimator.fit(X, t, y)

    result = {
        _get_metric_result_key(metric): [metric.evaluate(dataset, fitted_estimator)]
        for metric in metrics
    }

    if return_fitted:
        result["fitted_model"] = [fitted_estimator]

    return pd.DataFrame(result)


def _check_metrics(metrics) -> list[AverageResponseMetric]:
    """Validate and coerce metric input to a non-empty metric list."""
    if isinstance(metrics, AverageResponseMetric):
        metrics = [metrics]
    elif metrics is None:
        raise ValueError("metrics must contain at least one AverageResponseMetric.")
    else:
        metrics = list(metrics)

    if len(metrics) == 0:
        raise ValueError("metrics must contain at least one AverageResponseMetric.")

    for metric in metrics:
        if not isinstance(metric, AverageResponseMetric):
            raise TypeError(
                "metrics must contain only AverageResponseMetric instances. "
                f"Got {type(metric).__name__}."
            )

    metric_names = [_get_metric_result_key(metric) for metric in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metrics must have distinct string representations.")

    return metrics


def _get_metric_result_key(metric: AverageResponseMetric) -> str:
    """Return the output column name for a metric."""
    return str(metric)


def _validate_dataset_and_estimator(dataset, estimator) -> None:
    """Validate the public evaluate contract."""
    if not isinstance(dataset, BaseSyntheticDataset):
        raise TypeError(
            "dataset must be an instance of BaseSyntheticDataset. "
            f"Got {type(dataset).__name__}."
        )
    if not isinstance(estimator, BaseAverageCausalResponseEstimator):
        raise TypeError(
            "estimator must be an instance of BaseAverageCausalResponseEstimator. "
            f"Got {type(estimator).__name__}."
        )


def evaluate_multiple_dataset_seeds(
    dataset,
    estimator,
    metrics,
    random_states,
    return_fitted=False,
):
    """Evaluate one estimator on multiple dataset seeds."""
    results = []

    for seed in random_states:
        dataset_with_seed = dataset.clone().set_params(random_state=seed)
        result = evaluate_one(
            dataset_with_seed,
            estimator,
            metrics,
            return_fitted=return_fitted,
        )
        result["dataset_seed"] = seed
        results.append(result)

    return pd.concat(results, ignore_index=True)
