from typing import List

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from skcausal._pytorch.utils.activation import parse_activation


__all__ = [
    "SplineBasis",
    "SplineRegress",
]


def knots_for_each_treatment_dim(t: pl.Schema, knots: list):
    if knots is None:
        knots = []

    all_knots = []
    for dtype in t.dtypes():
        if dtype.is_numeric():
            all_knots.append(knots)
        else:
            all_knots.append([])
    return all_knots


class SplineBasis(nn.Module):
    """
    Spline Basis for ADMIT.

    Receives a tensor x with shape (batch_size, 1) and returns a tensor with shape
    (batch_size, num_of_basis).

    Parameters
    ----------
    degree : int
        The degree of the spline basis function.
    knots : list[float]
        The knots of the spline basis function.
    """

    def __init__(self, degrees: List[int], knots: List[List[float]], hidden_depth=None):
        super().__init__()
        self.degrees = degrees
        self.knots = knots
        self.knots_per_treatment_dim = [len(k) for k in self.knots]
        self.hidden_depth = hidden_depth

        self.num_of_basis = [
            degree + num_knots if num_knots > 0 else degree
            for degree, num_knots in zip(degrees, self.knots_per_treatment_dim)
        ]

        self.output_size = sum(self.num_of_basis) + 1

        if self.hidden_depth is not None:
            # nn.Linear(self.output_size, self.output_size)  and relu
            self.hidden = nn.Sequential(
                nn.Linear(self.output_size, self.output_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.output_size, self.output_size),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim < 2:
            x = x.reshape(-1, 1)

        outs = torch.zeros(
            (x.shape[0], self.output_size),
            dtype=x.dtype,
            device=x.device,
        )
        outs[:, 0] = 1

        index_to_fill = 1
        for feature_idx, (degree, knots) in enumerate(zip(self.degrees, self.knots)):

            out = torch.zeros(x.shape[0], degree + len(knots), device=x.device)
            for i in range(degree):
                out[:, i] = x[:, feature_idx].flatten() ** (i + 1)

            for j, knot in enumerate(knots):
                out[:, degree + j] = (
                    self.relu(x[:, feature_idx] - knot).flatten() ** degree
                )

            outs[:, index_to_fill : index_to_fill + out.shape[1]] = out

            index_to_fill += out.shape[1]

        if self.hidden_depth is not None:
            outs = self.hidden(outs)
        return outs


class SplineRegress(nn.Module):

    def __init__(self, degrees: List[int], knots: List[List[float]], init=0.001):
        super().__init__()
        self.spline_basis = SplineBasis(degrees, knots)
        self.init = init

        self.coefficients = torch.nn.Parameter(
            torch.normal(0, init, size=(self.spline_basis.output_size, 1)),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.spline_basis(x)
        return basis @ self.coefficients
