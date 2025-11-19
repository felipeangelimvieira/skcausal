from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from skcausal._pytorch.utils.activation import parse_activation
from skcausal._pytorch.layers.splines import SplineBasis
from skcausal._pytorch.layers.multi_output_sequential import MultiOutputSequential


__all__ = [
    "DynamicFC",
    "BaseDynamicHiddenBlock",
    "dynamic_block_from_config",
]


class DynamicFC(nn.Module):
    """
    Dynamic Fully-Connected Layer for ADMIT.

    The `dynamic_type` parameter determines the type of dynamic layer to use.
    Two options are available:
    - "power": Uses a spline basis function.
    - "mlp": Uses an MLP.

    The `degree` and `knots` parameters are only used when `dynamic_type` is set to

    Parameters
    ----------
    input_size : int
        The size of the input tensor.
    output_size : int
        The size of the output tensor (covariates + treatment)
    degree : int
        The degree of the spline basis function.
    knots : list[float]
        The knots of the spline basis function.
    activation : str, optional
        The activation function to use. Default is "relu".
    use_bias : int, optional
        Whether to use a bias term. Default is 1.
    should_concat_treat_on_output : int, optional
        Whether to concatenate the treatment values to the output tensor. Default is 0.
    dynamic_type : str, optional
        The type of dynamic layer to use. Default is "power".
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        degrees: List[int],
        knots: List[List[float]],
        activation: str = "relu",
        should_concat_treat_on_output=0,
        dynamic_type="power",
        init=0.1,
    ):
        super(DynamicFC, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.degrees = degrees
        self.knots = knots
        self.concat_treat_on_output = should_concat_treat_on_output

        if dynamic_type in ["power", "splines"]:
            self.spline_basis = SplineBasis(degrees=degrees, knots=knots)
            self.d = self.spline_basis.output_size  # num of basis
        else:
            raise ValueError(f"Unknown dynamic type: {dynamic_type}")

        self.weight = nn.Parameter(
            torch.normal(0, init, size=(self.d, self.output_size, self.input_size)),
            requires_grad=True,
        )

        self.bias = nn.Parameter(
            torch.normal(0, init, (self.output_size, self.d)),
            requires_grad=True,
        )

        self.activation = parse_activation(activation)

    def forward(self, *args):
        # x: batch_size * (treatment, other feature)
        # Enforce t has at leas 2 dims (-1,1), (-1, 2) or ...
        x, t = args
        x_feature = x
        x_treat = t

        # The feature weight contains as first dimension the batch size (the samples).
        # The second dimension is the output dimension of the layer, and the last
        # layer are the weights of each component of the spline basis function
        # weight_permuted = self.weight.permute(2, 1, 0)

        x_feature_weight = self.weight @ x_feature.T  # d, outd, bs
        x_feature_weight = x_feature_weight.permute(2, 1, 0)  # bs, outd, d

        x_treat_basis = self.spline_basis(x_treat)  # bs, d

        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # Here, we multiply each basis func value fot t_i by
        # The corresponding by the "feature_weight", computed previously.
        # When we sum this last dimension, we get the evaluation of the basis function
        # For each x_i and t_i
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2)  # bs, outd

        out_bias = torch.matmul(self.bias, x_treat_basis.T).T
        out = out + out_bias

        if self.activation is not None:
            out = self.activation(out)
        if self.concat_treat_on_output:
            return (out, t)

        return out


class BaseDynamicHiddenBlock(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_config: List[Tuple[int, str]] = None,
        knots=None,
        degrees=2,
        dynamic_type="splines",
        init=0.01,
    ):

        self.input_size = input_size
        self.hidden_config = hidden_config
        self.knots = knots if knots is not None else [0.33, 0.66]
        self.degrees = degrees
        self.dynamic_type = dynamic_type

        super().__init__()

        self._hidden_block = None
        if self.hidden_config is not None:
            self._hidden_block = dynamic_block_from_config(
                input_size=input_size,
                degrees=degrees,
                knots=knots,
                dynamic_type=dynamic_type,
                hidden_layers=hidden_config,
                init=init,
            )

    def forward(self, x, t):
        if self._hidden_block is None:
            return x
        return self._hidden_block(x, t)


def dynamic_block_from_config(
    input_size,
    hidden_layers,
    degrees,
    knots,
    dynamic_type="splines",
    init=0.01,
):
    # construct the rw-weighting network

    if hidden_layers is None or len(hidden_layers) == 0:
        return None

    dynamic_layers = []

    for layer_idx, (layer_size, activation) in enumerate(hidden_layers):
        is_last_layer = layer_idx == len(hidden_layers) - 1
        dynamic_layers.append(
            DynamicFC(
                input_size=input_size,
                output_size=layer_size,
                degrees=degrees,
                knots=knots,
                activation=activation,
                should_concat_treat_on_output=not is_last_layer,
                dynamic_type=dynamic_type,
                init=init,
            )
        )

        input_size = layer_size

    return MultiOutputSequential(*dynamic_layers)
