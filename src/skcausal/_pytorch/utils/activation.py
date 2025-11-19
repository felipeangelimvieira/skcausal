import torch
import torch.nn as nn

__all__ = ["parse_activation", "IdentityActivation"]


class IdentityActivation(nn.Module):

    def __init__(self):
        super(IdentityActivation, self).__init__()

    def forward(self, x):
        return x


def parse_activation(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation is None:
        return IdentityActivation()
    else:
        raise ValueError(f"Unknown activation function: {activation}")
