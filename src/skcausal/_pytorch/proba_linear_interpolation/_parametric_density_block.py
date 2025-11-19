import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from skcausal._pytorch.proba_linear_interpolation._treatment_head_types import (
    TreatmentHeadType,
)
from skcausal._pytorch.layers.fully_connected import BaseFCHiddenBlock


class ParametricDensityBlock(nn.Module):
    """
    Computes P(T_i|X) for each treatment i in treatment_heads.

    Uses T_{i-1}, T_{i-2}, ... as input for T_i, i.e.
    P(T_i|X, T_{i-1}, T_{i-2}, ...)

    Param
    """

    def __init__(
        self,
        input_size: int,
        treatment_heads: List[TreatmentHeadType],
        hidden_units: List[int] = None,
        init=0.01,
    ):

        self.input_size = input_size
        self.treatment_heads = treatment_heads
        self.hidden_units = hidden_units

        if hidden_units is None:
            hidden_units = [50]

        super().__init__()

        self.proba_layers_ = nn.ParameterList()
        extra_input_size = 0
        for i, treatment in enumerate(treatment_heads):

            if treatment.equals("logistic"):
                head = LogisticHead(
                    input_size=input_size + extra_input_size,
                    hidden_units=hidden_units,
                )
                extra_input_size += 1
                self.proba_layers_.append(head)

            elif treatment.equals("interpolate"):
                head = LinearInterpolateDensityBlock(
                    input_size=input_size + extra_input_size,
                    n_bins=treatment.n_bins,
                    hidden_config=[(units, "relu") for units in hidden_units],
                )
                extra_input_size += 1
                self.proba_layers_.append(head)
            elif treatment.equals("softmax"):
                head = SoftmaxHead(
                    input_size=input_size + extra_input_size,
                    n_classes=treatment.n_classes,
                    hidden_units=hidden_units,
                )

                extra_input_size += treatment.n_classes
                self.proba_layers_.append(head)
            else:
                raise ValueError(f"treatment {treatment} not supported")

    def forward(self, x, t):
        proba = 1
        t_inputs = t[:, :0]
        for i, proba_layer in enumerate(self.proba_layers_):

            _x = torch.cat([x, t_inputs], dim=1)
            _proba = 1 / proba_layer(_x, t[:, i])
            proba *= _proba.reshape((-1, 1))

            _t = t[:, i].reshape((-1, 1))
            if isinstance(proba_layer, SoftmaxHead) and proba_layer.n_classes > 2:
                _t = F.one_hot(_t.flatten().to(int), num_classes=proba_layer.n_classes)
            t_inputs = torch.cat([t_inputs, _t], dim=1)
        return 1 / (proba + 1e-9)


class SoftmaxHead(nn.Module):

    def __init__(self, input_size, n_classes, hidden_units=None):

        self.input_size = input_size
        self.n_classes = n_classes
        super().__init__()

        self.sequential_fc = None
        if hidden_units is not None:
            layers = []
            for layer_size in hidden_units:
                layers.append(nn.Linear(input_size, layer_size))
                layers.append(nn.ReLU(inplace=True))
                input_size = layer_size
            self.sequential_fc = nn.Sequential(*layers)

        self.head_fc = nn.Linear(input_size, n_classes)

    def forward(self, x, t):
        if self.sequential_fc is not None:
            x = self.sequential_fc(x)

        logits = F.softmax(self.head_fc(x), dim=1)

        # Get index given t tensor of shape (N, 1) with values {0,1,...,n_classes}
        proba = logits[torch.arange(logits.size(0)), t.tolist()].reshape((-1, 1))
        return 1 / proba


class LogisticHead(nn.Module):

    def __init__(self, input_size, hidden_units=None):

        self.input_size = input_size
        super().__init__()

        self.sequential_fc = None
        if hidden_units is not None:
            layers = []
            for layer_size in hidden_units:
                layers.append(nn.Linear(input_size, layer_size))
                layers.append(nn.ReLU(inplace=True))
                input_size = layer_size
            self.sequential_fc = nn.Sequential(*layers)

        self.head_fc = nn.Linear(input_size, 1, bias=True)

    def forward(self, x, t):
        if self.sequential_fc is not None:
            x = self.sequential_fc(x)

        # Cast t to float
        t = t.float()

        logits = self.head_fc(x)
        proba = 1 / (1 + torch.exp(-logits.flatten()))
        proba = t * proba + (1 - t) * (1 - proba)
        return 1 / (proba + 1e-9)


class LinearInterpolateDensityBlock(BaseFCHiddenBlock):

    def __init__(
        self,
        input_size,
        n_bins: List[int],
        hidden_config: List[Tuple[int, str]] = None,
    ):

        super().__init__(input_size=input_size, hidden_config=hidden_config)

        self.n_bins = n_bins

        # if not isinstance(n_bins, list):
        #    raise ValueError("n_bins must be a list of integers")

        self.linear = nn.Linear(self.input_size_last_layer, n_bins + 1, bias=True)

    def forward(self, x, t):

        t = t.reshape((-1, 1))

        x = super().forward(x, t)
        # Compute probabilities with a linear layer followed by softmax
        probabilities = self._compute_probas(x)

        # Scale t to the range [0, n_bins]
        t_scaled = t * self.n_bins

        # Find the two nearest bins
        lower_bin = torch.floor(t_scaled).long()
        upper_bin = torch.ceil(t_scaled).long()

        # Ensure bins are within valid range
        lower_bin = torch.clamp(lower_bin, 0, self.n_bins)
        upper_bin = torch.clamp(upper_bin, 0, self.n_bins)

        # Compute weights for interpolation
        upper_weight = t_scaled - lower_bin.float()
        lower_weight = 1.0 - upper_weight

        # Interpolate between the two nearest bins
        all_idx = torch.arange(0, x.shape[0])
        total_weight = (lower_weight + upper_weight).flatten()
        output = (
            lower_weight.flatten()
            / total_weight
            * probabilities[all_idx, lower_bin.flatten()]
            + upper_weight.flatten()
            / total_weight
            * probabilities[all_idx, upper_bin.flatten()]
        )

        normalization_factor = (
            probabilities[:, 1:] * 0.5 + probabilities[:, :-1] * 0.5
        ).mean(dim=1)
        output = output / normalization_factor
        return 1 / output

    def _compute_probas(self, x):
        # Compute probabilities with a linear layer followed by softmax
        logits = self.linear(x)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
