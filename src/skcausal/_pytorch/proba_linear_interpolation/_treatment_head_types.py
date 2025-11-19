"""
Containers of metadata about each treatment head
"""

__all__ = [
    "TreatmentHeadType",
    "Interpolate",
    "NormalHeadType",
    "SoftmaxHeadType",
    "LogisticHeadType",
]


class TreatmentHeadType:

    name = None

    def equals(self, other: str):
        return other == self.name


class Interpolate(TreatmentHeadType):

    name = "interpolate"

    def __init__(self, n_bins):
        self.n_bins = n_bins


class NormalHeadType(TreatmentHeadType):

    name = "normal"


class SoftmaxHeadType(TreatmentHeadType):

    name = "softmax"

    def __init__(self, n_classes):
        self.n_classes = n_classes


class LogisticHeadType(TreatmentHeadType):

    name = "logistic"

    def __init__(self): ...
