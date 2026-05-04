from __future__ import annotations

from skcausal.datasets.meta_multidim import MetaMultidimDataset
from skcausal.datasets.synthetic2 import SyntheticDataset2

__all__ = ["Synthetic2MultidimDataset"]


class Synthetic2MultidimDataset(MetaMultidimDataset):
    r"""Meta-multidimensional wrapper specialized to :class:`SyntheticDataset2`.

    This convenience dataset keeps the mixed-treatment construction from
    :class:`~skcausal.datasets.meta_multidim.MetaMultidimDataset` while fixing
    the wrapped base generator to :class:`~skcausal.datasets.synthetic2.SyntheticDataset2`.
    """

    def __init__(
        self,
        n: int = 1000,
        outcome_noise: float = 0.5,
        treatment_noise: float = 0.5,
        random_state: int = 5,
        n_features: int = 6,
        n_categorical_treatments: int = 4,
        mutual_info: float = 1.0,
        categorical_effect_scale: float = 0.15,
        categorical_column: str | None = None,
    ):
        self.n = n
        self.outcome_noise = outcome_noise
        self.treatment_noise = treatment_noise
        self.random_state = random_state
        self.n_features = n_features

        base_dataset = SyntheticDataset2(
            n=n,
            outcome_noise=outcome_noise,
            treatment_noise=treatment_noise,
            random_state=random_state,
            n_features=n_features,
        )

        super().__init__(
            base_dataset=base_dataset,
            n_categorical_treatments=n_categorical_treatments,
            mutual_info=mutual_info,
            categorical_effect_scale=categorical_effect_scale,
            categorical_column=categorical_column,
            random_state=random_state,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        return [
            {
                "n": 2000,
                "categorical_effect_scale": 0.2,
                "mutual_info": 0.5,
                "random_state": 7,
            },
            {
                "n": 48,
                "categorical_effect_scale": 0.2,
                "n_categorical_treatments": 3,
                "mutual_info": 0.0,
                "random_state": 11,
            },
        ]
