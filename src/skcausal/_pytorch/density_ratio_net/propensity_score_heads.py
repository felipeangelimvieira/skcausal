from functools import reduce
from typing import List, Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from skcausal._pytorch.layers.splines import SplineBasis
from skcausal._pytorch.layers.dynamic_layer import (
    dynamic_block_from_config,
    BaseDynamicHiddenBlock,
)
from skcausal._pytorch.layers.fully_connected import (
    fc_block_from_config,
    BaseFCHiddenBlock,
)

__all__ = [
    "LinearInterpolateDensityBlock",
]


class SplineTreatmentLinearModule(nn.Module):
    """
    Expands treatment variable using spline basis functions.

    Then, passes the concatenation of the input features and the spline-expanded
    treatment through a fully connected neural network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (excluding treatment).
    hidden_units : List[Tuple[int, str]], optional
        Configuration for hidden layers in the fully connected network.
        Each tuple specifies (number of units, activation function).
        If None, defaults to [(50, 'relu')].
    knots : list, optional
        Knot positions for the spline basis functions. If None, defaults to None.
    degrees : List[int]
        Degrees of the spline basis functions for each treatment dimension.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[List[Tuple[int, str]]] = None,
        knots: Optional[list] = None,
        degrees: List[int] = None,
    ) -> None:

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.knots = knots
        self.degrees = degrees

        super().__init__()

        self.splines_ = SplineBasis(degrees=degrees, knots=knots)

        if hidden_units is None:
            self._hidden_units = [(50, "relu")]
        else:
            self._hidden_units = hidden_units

        self.embedding_fc_ = self._build_embedding_block(
            input_size=self.splines_.output_size + input_dim,
            embedding_hidden_units=self._hidden_units,
        )

    def _build_embedding_block(self, input_size, embedding_hidden_units):
        return fc_block_from_config(
            input_size=input_size, block_config=embedding_hidden_units
        )

    def forward(self, x, t):

        if t.ndim == 1:
            t = t.unsqueeze(1)
        t = self.splines_(t)
        xt = torch.cat([x, t], dim=1)
        out = self.embedding_fc_(xt)

        return out


class SplineTreatmentLinearRegressorHead(SplineTreatmentLinearModule):

    def __init__(
        self,
        input_dim: int,
        hidden_units: Optional[List[Tuple[int, str]]] = None,
        knots: Optional[list] = None,
        degrees: List[int] = None,
        dynamic_type="splines",
    ):

        super().__init__(
            input_dim=input_dim,
            hidden_units=hidden_units,
            knots=knots,
            degrees=degrees,
            dynamic_type=dynamic_type,
        )

        last_hidden_size = self._hidden_units[-1][0]

        self.linear_ = nn.Linear(last_hidden_size, 1, bias=True)

    def forward(self, x, t):
        x = super().forward(x, t)
        return self.linear_(x)
