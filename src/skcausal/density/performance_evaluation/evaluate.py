"""Utility for benchmark evaluation of density estimators."""

import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

from skcausal.density.performance_evaluation.metrics.likelihood import (
    LogLikelihoodMetric,
)


def _split(X, t, train, test):
    """Split X and t according to train/test indices."""
    results = dict()
    results["X_train"] = _take_rows(X, train)
    results["X_test"] = _take_rows(X, test)
    results["t_train"] = _take_rows(t, train)
    results["t_test"] = _take_rows(t, test)
    return results


def evaluate(
    estimator,
    cv,
    X,
    t,
    scoring=None,
    return_data=False,
    error_score=np.nan,
):
    r"""Evaluate density estimator using cross-validation folds.

    All-in-one statistical performance benchmarking utility for density
    estimators which runs a simple cross-validation experiment and returns
    a summary pd.DataFrame.

    The experiment run is the following:

    Denote by :math:`X_{train, 1}, X_{test, 1}, \dots, X_{train, K}, X_{test, K}`
    the train/test folds produced by the generator ``cv.split(X)``
    Denote by :math:`t_{train, 1}, t_{test, 1}, \dots, t_{train, K}, t_{test, K}`
    the corresponding train/test folds for the treatment.

    0. For ``i = 1`` to ``cv.get_n_splits(X)`` do:
    1. ``fit`` the ``estimator`` to :math:`X_{train, i}`, :math:`t_{train, i}`
    2. Compute ``scoring`` on the fitted estimator using
       :math:`X_{test, i}`, :math:`t_{test, i}`.

    Results returned in this function's return are:

    * results of ``scoring`` calculations, from 2, in the `i`-th loop
    * runtimes for fitting and scoring, from 1, 2 in the `i`-th loop
    * :math:`t_{train, i}`, :math:`t_{test, i}` (optional)

    Parameters
    ----------
    estimator : BaseDensityEstimator
        Density estimator to benchmark.
    cv : sklearn splitter
        Determines split of ``X`` and ``t`` into test and train folds.
    X : data container
        Covariate data.
    t : data container
        Treatment data. Must be same length as ``X``.
    scoring : BaseDensityMetric, list of BaseDensityMetric, or None
        Metric(s) used to evaluate the density estimator. If ``None``,
        ``LogLikelihoodMetric()`` is used.
    return_data : bool, default=False
        If True, returns additional columns containing t_train and t_test
        for each fold.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs during estimator
        fitting. If set to "raise", the exception is raised. If a numeric
        value is given, a warning is raised.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with information regarding each fold.
        Row index is the fold index.
        Columns are as follows:

        - test_{scoring.name}: (float) Model performance score.
          If ``scoring`` is a list,
          then there is a column with name ``test_{scoring.name}`` for each scorer.
        - fit_time: (float) Time in seconds for ``fit`` on train fold.
        - score_time: (float) Time in seconds to score from fitted estimator.
        - len_t_train: (int) Length of t_train.
        - t_train: only present if ``return_data=True``.
        - t_test: only present if ``return_data=True``.
    """
    scoring = _check_scores(scoring)

    _evaluate_fold_kwargs = {
        "estimator": estimator,
        "scoring": scoring,
        "return_data": return_data,
        "error_score": error_score,
    }

    def gen_X_t_train_test(X, t, cv):
        """Generate joint splits of X, t as per cv.

        Yields
        ------
        dict with keys X_train, X_test, t_train, t_test.
        """
        for train, test in cv.split(X):
            yield _split(X, t, train, test)

    results = [
        _evaluate_fold(split_data, _evaluate_fold_kwargs)
        for split_data in gen_X_t_train_test(X, t, cv)
    ]

    results = pd.concat(results)
    results = results.reset_index(drop=True)

    return results


def _evaluate_fold(x, meta):
    """Evaluate a single CV fold."""
    X_train = x["X_train"]
    X_test = x["X_test"]
    t_train = x["t_train"]
    t_test = x["t_test"]

    estimator = meta["estimator"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
    error_score = meta["error_score"]

    # Set default result values in case estimator fitting fails
    fit_time = np.nan
    score_time = np.nan

    temp_result = dict()

    try:
        estimator = estimator.clone()

        # fit
        start_fit = time.perf_counter()
        estimator.fit(X_train, t_train)
        fit_time = time.perf_counter() - start_fit

        # score
        start_score = time.perf_counter()
        for metric in scoring:
            result_key = f"test_{metric.name}"
            score = metric(estimator, X_test, t_test)
            temp_result[result_key] = [score]
        score_time = time.perf_counter() - start_score

    except Exception as e:
        if error_score == "raise":
            raise e
        else:
            warnings.warn(
                f"In evaluate, fitting of estimator {type(estimator).__name__} "
                f"failed, you can set error_score='raise' in evaluate to see "
                f"the exception message. Fit failed for "
                f"len(t_train)={len(t_train)}. "
                f"The score will be set to {error_score}. "
                f"Failed estimator with parameters: {estimator}.",
                stacklevel=2,
            )
            for metric in scoring:
                result_key = f"test_{metric.name}"
                temp_result[result_key] = [error_score]

    # format results data frame and return
    temp_result["fit_time"] = [fit_time]
    temp_result["score_time"] = [score_time]
    temp_result["len_t_train"] = [len(t_train)]

    if return_data:
        temp_result["t_train"] = [t_train]
        temp_result["t_test"] = [t_test]

    result = pd.DataFrame(temp_result)
    result = result.astype({"len_t_train": int})

    column_order = _get_column_order_and_datatype(scoring, return_data)
    result = result.reindex(columns=column_order.keys())

    return result


def _get_column_order_and_datatype(scoring, return_data=False):
    """Get the ordered column name and input datatype of results."""
    metrics_metadata = {}
    for metric in scoring:
        result_key = f"test_{metric.name}"
        metrics_metadata[result_key] = "float"

    fit_metadata = {
        "fit_time": "float",
        "score_time": "float",
        "len_t_train": "int",
    }

    metrics_metadata.update(fit_metadata)

    if return_data:
        metrics_metadata["t_train"] = "object"
        metrics_metadata["t_test"] = "object"

    return metrics_metadata


def _check_scores(scoring):
    """Validate and coerce scoring to a list of metrics.

    Parameters
    ----------
    scoring : BaseDensityMetric, list of BaseDensityMetric, or None

    Returns
    -------
    list of BaseDensityMetric
    """
    if scoring is None:
        return [LogLikelihoodMetric()]
    if not isinstance(scoring, list):
        return [scoring]
    return list(scoring)


def _take_rows(X, indices):
    """Index rows from a data container by integer indices."""
    if isinstance(X, np.ndarray):
        return X[indices]
    if isinstance(X, pl.DataFrame):
        return X[indices]
    if isinstance(X, pl.Series):
        return X[indices]

    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    if isinstance(X, pd.Series):
        return X.iloc[indices]

    raise TypeError(f"Unsupported data type: {type(X)}")
