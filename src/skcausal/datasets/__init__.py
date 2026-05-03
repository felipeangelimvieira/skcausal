from importlib import import_module

from .categorical import ExampleCategorical
from .ihdp import IHDPContinuous
from .kang_schafer import (
    KangSchaferBinary,
    KangSchaferBinaryMisspecified,
    KangSchaferContinuous,
    KangSchaferContinuousMisspecified,
)
from .meta_multidim import MetaMultidimDataset
from .nurse_staffing import NurseStaffing
from .semi_synthetic_classifier import SemiSyntheticClassifier
from .semi_synthetic_regressor import SemiSyntheticRegressor
from .synthetic_vcnet import SyntheticVCNet
from .synthetic2 import SyntheticDataset2, SyntheticDataset2Discrete
from .synthetic2_multidim import Synthetic2MultidimDataset

__all__ = [
    "ExampleCategorical",
    "IHDPContinuous",
    "KangSchaferBinary",
    "KangSchaferBinaryMisspecified",
    "KangSchaferContinuous",
    "KangSchaferContinuousMisspecified",
    "MetaMultidimDataset",
    "NurseStaffing",
    "SemiSyntheticClassifier",
    "SemiSyntheticRegressor",
    "SyntheticVCNet",
    "SyntheticDataset2",
    "SyntheticDataset2Discrete",
    "Synthetic2MultidimDataset",
]
