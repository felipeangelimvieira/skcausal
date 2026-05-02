from importlib import import_module

from .categorical import ExampleCategorical
from .ihdp import IHDPContinuous
from .kang_schafer import (
    KangSchaferBinary,
    KangSchaferBinaryMisspecified,
    KangSchaferContinuous,
    KangSchaferContinuousMisspecified,
)
from .nurse_staffing import NurseStaffing
from .semi_synthetic_classifier import SemiSyntheticClassifier
from .semi_synthetic_regressor import SemiSyntheticRegressor
from .synthetic_vcnet import SyntheticVCNet
from .synthetic2 import SyntheticDataset2, SyntheticDataset2Discrete

__all__ = [
    "ExampleCategorical",
    "IHDPContinuous",
    "KangSchaferBinary",
    "KangSchaferBinaryMisspecified",
    "KangSchaferContinuous",
    "KangSchaferContinuousMisspecified",
    "NurseStaffing",
    "SemiSyntheticClassifier",
    "SemiSyntheticRegressor",
    "SyntheticVCNet",
    "SyntheticDataset2",
    "SyntheticDataset2Discrete",
]
