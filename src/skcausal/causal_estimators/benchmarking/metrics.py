from skbase.base import BaseObject
from skcausal.datasets.base import BaseSyntheticDataset
from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.utils.treatment_grid import sample_treatment_rows
import numpy as np

__all__ = ["AverageResponseMetric", "MAE", "RMSE"]


class AverageResponseMetric(BaseObject):
    """Base class for metrics that evaluate the average response function.

    Subclasses must override `_evaluate` method.
    """

    _tags = {
        "object_type": ["average_response_metric"],
    }

    def __init__(self):
        super().__init__()

    def evaluate(
        self,
        dataset: BaseSyntheticDataset,
        estimator: BaseAverageCausalResponseEstimator,
    ):
        """Evaluate the metric on the given dataset and estimator.

        Parameters
        ----------
        dataset : BaseSyntheticDataset
            The dataset to evaluate on.
        estimator : BaseAverageCausalResponseEstimator
            The estimator to evaluate.

        Returns
        -------
        float
            The value of the metric.
        """
        if not isinstance(dataset, BaseSyntheticDataset):
            raise TypeError(
                "dataset must be an instance of BaseSyntheticDataset. "
                f"Got {type(dataset).__name__}."
            )
        if not isinstance(estimator, BaseAverageCausalResponseEstimator):
            raise TypeError(
                "estimator must be an instance of "
                "BaseAverageCausalResponseEstimator. "
                f"Got {type(estimator).__name__}."
            )

        return float(self._evaluate(dataset, estimator))

    def _evaluate(
        self,
        dataset: BaseSyntheticDataset,
        estimator: BaseAverageCausalResponseEstimator,
    ):
        """Evaluate the metric on the given dataset and estimator.

        Subclasses must override this method.

        Parameters
        ----------
        dataset : BaseSyntheticDataset
            The dataset to evaluate on.
        estimator : BaseAverageCausalResponseEstimator
            The estimator to evaluate.

        Returns
        -------
        float
            The value of the metric.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class _BaseSubsampledAverageResponseMetric(AverageResponseMetric):
    """Base class for average response metrics that require subsampling the treatment space.

    This is useful for metrics that require integration over the treatment space, such as MISE.
    """

    def __init__(self, n_treatments: int, random_state: int = 42):
        self.n_treatments = n_treatments
        self.random_state = random_state
        super().__init__()

    def _subsample_treatments(self, treatments):
        """Subsample treatment rows to a fixed number of evaluation points.

        Parameters
        ----------
        treatments : dataframe-like or np.ndarray
            Observed treatment rows.

        Returns
        -------
        dataframe-like or np.ndarray
            The subsampled treatment rows.
        """
        if hasattr(treatments, "height"):
            return sample_treatment_rows(
                treatments,
                n_rows=self.n_treatments,
                unique_rows=False,
                random_state=self.random_state,
            )

        treatments = np.asarray(treatments)
        if len(treatments) <= self.n_treatments:
            return treatments

        rng = np.random.default_rng(self.random_state)
        subsampled_indices = rng.choice(
            len(treatments), size=self.n_treatments, replace=False
        )
        return treatments[subsampled_indices]

    def _coerce_curve_values(self, values, *, n_rows: int, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            raise ValueError(f"{name} must return at least one prediction.")
        if values.shape[0] != n_rows:
            raise ValueError(
                f"{name} must return one value per requested treatment row."
            )
        return values.reshape(n_rows, -1)

    def _evaluate(self, dataset, estimator):

        X, t, _ = dataset.load()

        t_grid = self._subsample_treatments(t)
        n_rows = t_grid.height if hasattr(t_grid, "height") else len(t_grid)

        y_pred = self._coerce_curve_values(
            estimator.predict(t_grid),
            n_rows=n_rows,
            name="estimator.predict",
        )
        y_true = self._coerce_curve_values(
            dataset.predict(covariates=X, treatment_list=t_grid),
            n_rows=n_rows,
            name="dataset.predict",
        )
        return self._compute_metric(y_true, y_pred)

    def _compute_metric(self, y_true, y_pred) -> float:
        """Compute the metric given the true and predicted average responses.

        Subclasses must override this method.

        Parameters
        ----------
        y_true : np.ndarray
            The true average response values.
        y_pred : np.ndarray
            The predicted average response values.

        Returns
        -------
        float
            The value of the metric.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class MAE(_BaseSubsampledAverageResponseMetric):
    """Mean Absolute Error (MAE) metric for evaluating the average response function.

    This metric computes the mean absolute error between the true average response
    function and the estimated average response function.
    """

    _tags = {
        "metric_name": "MAE",
    }

    def __init__(self, n_treatments: int, random_state: int = 42):
        super().__init__(n_treatments=n_treatments, random_state=random_state)

    def _compute_metric(self, y_true, y_pred):
        """Compute the MAE metric.

        Parameters
        ----------
        y_true : np.ndarray
            The true average response values.
        y_pred : np.ndarray
            The predicted average response values.

        Returns
        -------
        float
            The value of the MAE metric.
        """
        return np.mean(np.abs(y_true - y_pred))


class RMSE(_BaseSubsampledAverageResponseMetric):
    """Root Mean Squared Error (RMSE) metric for average response evaluation.

    This metric computes the root mean squared error between the true average
    response function and the estimated average response function.
    """

    _tags = {
        "metric_name": "RMSE",
    }

    def __init__(self, n_treatments: int, random_state: int = 42):
        super().__init__(n_treatments=n_treatments, random_state=random_state)

    def _compute_metric(self, y_true, y_pred):
        """Compute the RMSE metric.

        Parameters
        ----------
        y_true : np.ndarray
            The true average response values.
        y_pred : np.ndarray
            The predicted average response values.

        Returns
        -------
        float
            The value of the RMSE metric.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
