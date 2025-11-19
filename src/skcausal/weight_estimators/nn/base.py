import numpy as np
import polars as pl

from lightning import Trainer, seed_everything
from lightning.pytorch import callbacks, loggers
import torch
from skcausal._pytorch.utils.data import create_train_val_dataloaders_from_dataset
from pathlib import Path
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor


class BaseTorchBalancingWeightRegressor(BaseBalancingWeightRegressor):

    _tags = {
        "t_inner_mtype": np.ndarray,
        "X_inner_mtype": np.ndarray,
    }

    def __init__(
        self,
        n_epochs=500,
        early_stopping_patience=10,
        batch_size=512,
        split_ratio=0.1,
        random_state=0,
        verbose=False,
    ):

        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.verbose = verbose
        self._x_mean_ = None
        self._x_std_ = None
        super().__init__()

    def fit(self, X: pl.DataFrame, t: pl.DataFrame):
        """Fit estimator while standardizing covariates."""

        # Reset stored statistics before each new fit.
        self._x_mean_ = None
        self._x_std_ = None

        super().fit(X, t)
        return self

    def _create_dataloaders(self, X, t):

        dataset = self._create_dataset(X, t)

        train_loader, val_loader = create_train_val_dataloaders_from_dataset(
            dataset,
            split_ratio=self.split_ratio,
            batch_size=self.batch_size,
        )
        return train_loader, val_loader

    def _create_dataset(self, X, t):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _instantiate_model(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _instatiate_lightning_module(self):
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _check_and_transform_X(self, X: pl.DataFrame):
        X_arr = super()._check_and_transform_X(X)

        if not isinstance(X_arr, np.ndarray):
            raise TypeError(
                "BaseTorchBalancingWeightRegressor expects numpy arrays after"
                " conversion in _check_and_transform_X."
            )

        return X_arr

    def _transform_features(
        self, X_arr: np.ndarray, *, update_stats: bool
    ) -> np.ndarray:
        """Standardize features using stored statistics."""

        if update_stats:
            self._x_mean_ = X_arr.mean(axis=0)
            self._x_std_ = X_arr.std(axis=0)
            # Avoid division by zero for constant covariates.
            self._x_std_[self._x_std_ == 0] = 1.0

        if self._x_mean_ is None or self._x_std_ is None:
            raise ValueError(
                "Feature standardization parameters are unavailable."
                " Ensure that fit is called before predict."
            )

        if X_arr.shape[1] != self._x_mean_.shape[0]:
            raise ValueError(
                "Feature dimensionality mismatch when applying stored"
                " standardization parameters."
            )

        return (X_arr - self._x_mean_) / self._x_std_

    def _fit(self, X, t):

        # Set seeds
        seed_everything(self.random_state)

        X = self._transform_features(X, update_stats=True)

        train_loader, val_loader = self._create_dataloaders(X, t)

        self._instantiate_model(X, t)
        self._instatiate_lightning_module(X, t)
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=self.early_stopping_patience
            ),
        ]

        if self.verbose:
            callback_list.append(callbacks.RichProgressBar())

        self.trainer_: Trainer = Trainer(
            max_epochs=self.n_epochs,
            log_every_n_steps=len(train_loader),
            val_check_interval=len(train_loader),
            enable_checkpointing=False,
            callbacks=callback_list,
            # logger=loggers.CSVLogger(),
            enable_model_summary=self.verbose,
            enable_progress_bar=self.verbose,
        )

        self.trainer_.fit(
            self.lightning_module_,
            train_loader,
            val_loader,
        )

        metrics_file = Path(self.lightning_module_.logger.log_dir) / Path("metrics.csv")

        try:
            self.metrics_ = pl.read_csv(metrics_file)
        except Exception as e:
            print(f"Could not read metrics file: {metrics_file}, error: {e}")
            self.metrics_ = None

        return self

    def _predict_sample_weight(self, X: np.ndarray, t: pl.DataFrame):
        """
        Predict the probability of each class for each sample.
        :param X: Feature data.
        """

        X = self._transform_features(X, update_stats=False)

        default_dtype = torch.get_default_dtype()
        X = torch.tensor(X, dtype=default_dtype)
        t = torch.tensor(t, dtype=default_dtype)

        #        dataloader = create_dataloader(X, t, shuffle=False, batch_size=self.batch_size)

        if t.ndim == 1:
            t = t.reshape(-1, 1)

        self.model_.eval()
        with torch.no_grad():

            y_pred = self.model_(X, t)

        y_pred = y_pred.squeeze().cpu().numpy()
        # Check no nans
        if np.any(np.isnan(y_pred)):
            raise ValueError("NaN values found in predictions.")
        return y_pred.flatten()
