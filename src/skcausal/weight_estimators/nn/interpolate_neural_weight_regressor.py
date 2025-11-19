import polars as pl
from skcausal._pytorch.proba_linear_interpolation._treatment_head_types import (
    Interpolate,
    LogisticHeadType,
    SoftmaxHeadType,
)
from skcausal._pytorch.proba_linear_interpolation._parametric_density_block import (
    ParametricDensityBlock,
)
from skcausal._pytorch.proba_linear_interpolation.lightning_module import (
    LinearInterpolationProbaLogic,
)
from skcausal._pytorch.utils.data import (
    ArgsDataset,
)

from skcausal.weight_estimators.nn.base import BaseTorchBalancingWeightRegressor

__all__ = [
    "InterpolateNeuralWeightRegressor",
]


class InterpolateNeuralWeightRegressor(BaseTorchBalancingWeightRegressor):

    _tags = {
        "t_inner_mtype": "np.ndarray",
        "one_hot_encode_enum_columns": False,
        "balancing_weight_type": "propensity_score",
        "utility:scale_treatment": "minmax",
    }

    def __init__(
        self,
        learning_rate=1e-3,
        n_units=100,
        n_layers=1,
        split_ratio=None,
        batch_size=128,
        n_epochs=500,
        init=0.01,
        n_bins=10,
        random_state=None,
    ):
        self.n_units = n_units
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.embedding_hidden_units = [n_units] * n_layers
        self.n_bins = n_bins
        self.init = init

        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            split_ratio=split_ratio,
            random_state=random_state,
            early_stopping_patience=10,
        )

    def _instantiate_model(self, X, t):

        self.covariate_dims_ = X.shape[1]
        self.treatment_heads_ = self._map_treat_schema_to_types()

        self.model_ = ParametricDensityBlock(
            input_size=self.covariate_dims_,
            treatment_heads=self.treatment_heads_,
            hidden_units=self.embedding_hidden_units,
        )

    def _create_dataset(self, X, t):
        return ArgsDataset(X, t)

    def _instatiate_lightning_module(self, X, t):

        self.lightning_module_ = LinearInterpolationProbaLogic(
            model=self.model_,
            learning_rate=self.learning_rate,
        )

    def _map_treat_schema_to_types(self):
        treatment_heads = []
        for dtype in self._t_schema.dtypes():

            if dtype == pl.Enum:
                treatment_heads.append(SoftmaxHeadType(n_classes=len(dtype.categories)))
            elif dtype == pl.Boolean:
                treatment_heads.append(LogisticHeadType())

            elif dtype.is_numeric():
                # treatment_heads.append(NormalHeadType())
                treatment_heads.append(Interpolate(self.n_bins))
            else:
                raise ValueError(f"Invalid treatment type: {dtype}")
        return treatment_heads
