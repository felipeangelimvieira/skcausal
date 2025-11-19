import numpy as np
import polars as pl
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from lightning import Trainer, seed_everything
from lightning.pytorch import callbacks, loggers

from skcausal._pytorch.layers.splines import knots_for_each_treatment_dim
from skcausal._pytorch.density_ratio_net.propensity_score_heads import (
    SplineTreatmentLinearModule,
)
from skcausal._pytorch.density_ratio_net.dataset import (
    SyntheticDatasetResampleOnce,
    SyntheticDatasetShuffled,
    SyntheticDatasetResampleEachBatch,
)
from skcausal._pytorch.density_ratio_net.lightning_module import (
    _BatchLevelDirectPropensityTrainingLogic,
)


from skcausal.weight_estimators.nn.base import BaseTorchBalancingWeightRegressor

__all__ = [
    "TreatmentDensityRatioRegressor",
]

DATASET_METHODS = {
    "resample-each-batch": SyntheticDatasetResampleEachBatch,
    "resample-once": SyntheticDatasetResampleOnce,
    "resample-once-shuffled": SyntheticDatasetShuffled,
}


class TreatmentDensityRatioRegressor(BaseTorchBalancingWeightRegressor):
    _tags = {
        "t_inner_mtype": "np.ndarray",
        "one_hot_encode_enum_columns": True,
    }

    def __init__(
        self,
        loss_method="balanced",
        dataset_method="resample-each-batch",
        learning_rate=1e-3,
        n_units_hidden=50,
        n_layers_hidden=1,
        hidden_layers_activation="relu",
        last_layer_activation="softplus",
        knots=None,
        degree=2,
        split_ratio=None,
        batch_size=128,
        n_epochs=500,
        l2_penalty=0,
        l2_inv_penalty=0,
        n_repeats=1,
        random_state=0,
    ):
        self.n_units_hidden = n_units_hidden
        self.n_layers_hidden = n_layers_hidden

        hidden_units = [(n_units_hidden, hidden_layers_activation)] * n_layers_hidden

        self.last_layer_activation = last_layer_activation
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.hidden_layers_activation = hidden_layers_activation
        self.knots = knots
        self.degree = degree
        self.n_epochs = n_epochs
        self.loss_method = loss_method
        self.dataset_method = dataset_method
        self.l2_penalty = l2_penalty
        self.l2_inv_penalty = l2_inv_penalty
        self.n_repeats = n_repeats

        super().__init__(
            random_state=random_state,
            n_epochs=n_epochs,
            split_ratio=split_ratio,
            batch_size=batch_size,
            early_stopping_patience=100,
        )

        hidden_units = self.hidden_units
        if hidden_units is None:
            hidden_units = [
                (50, "relu"),
            ]

        self._hidden_units = hidden_units

    def _instatiate_lightning_module(self, X, t):
        self.lightning_module_ = _BatchLevelDirectPropensityTrainingLogic(
            model=self.model_,
            learning_rate=self.learning_rate,
            l2_penalty=self.l2_penalty,
            l2_inv_penalty=self.l2_inv_penalty,
        )

    def _instantiate_model(self, X, t):

        knots = self.knots
        if knots is None:
            knots = [0.33, 0.66]

        self.covariate_dims_ = X.shape[1]
        self.knots_ = knots_for_each_treatment_dim(self._t_preprocessed_schema, knots)
        self.degrees_ = [self.degree if len(knots) > 1 else 1 for knots in self.knots_]
        self.discrete_dims_ = []
        for i, dtype in enumerate(self._t_preprocessed_schema.dtypes()):
            if not dtype.is_numeric():
                self.discrete_dims_.append(i)

        self.model_ = SplineTreatmentLinearModule(
            input_dim=self.covariate_dims_,
            # Add last layer
            hidden_units=self._hidden_units + [(1, self.last_layer_activation)],
            knots=self.knots_,
            degrees=self.degrees_,
        )

    @property
    def _binary_dims(self):
        t_schema = (
            self._t_schema
            if self._t_preprocessed_schema is None
            else self._t_preprocessed_schema
        )
        return [i for i, dtype in enumerate(t_schema.dtypes()) if dtype == pl.Boolean]

    @property
    def _dummy_sets(self):

        t_schema = (
            self._t_schema
            if self._t_preprocessed_schema is None
            else self._t_preprocessed_schema
        )

        return _extract_dummy_sets_from_schema(t_schema)

    @property
    def mutually_exclusive_sets(self):
        t_schema = (
            self._t_schema
            if self._t_preprocessed_schema is None
            else self._t_preprocessed_schema
        )
        return _extract_mutually_exclusive_sets_from_schema(t_schema)

    def _create_dataset(self, X, t):
        DatasetClass = DATASET_METHODS[self.dataset_method]
        return DatasetClass(
            X,
            t,
            method=self.loss_method,
            binary_dims=self._binary_dims,
            dummy_sets=self._dummy_sets,
            mutually_excludent_sets=self.mutually_exclusive_sets,
            n_repeats=self.n_repeats,
        )


def _extract_dummy_sets_from_schema(t_schema: pl.Schema):

    column_names = t_schema.names()

    dummy_prefixes = set()
    for name in column_names:
        if "__dummy" in name:
            dummy_prefixes.add(name.split("__dummy")[0])

    dummy_sets = []
    for prefix in dummy_prefixes:
        dummy_set = [
            i
            for i, name in enumerate(column_names)
            if name.startswith(f"{prefix}__dummy")
        ]
        dummy_sets.append(dummy_set)
    return dummy_sets


def _extract_mutually_exclusive_sets_from_schema(t_schema: pl.Schema):

    column_names = t_schema.names()

    exclusive_prefixes = set()
    for name in column_names:
        if "__exclusive" in name:
            exclusive_prefixes.add(name.split("__exclusive")[0])

    exclusive_sets = []
    for prefix in exclusive_prefixes:
        exclusive_set = [
            i
            for i, name in enumerate(column_names)
            if name.startswith(f"{prefix}__exclusive")
        ]
        exclusive_sets.append(exclusive_set)
    return exclusive_sets
