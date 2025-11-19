import torch.nn as nn

__all__ = ["MultiOutputSequential"]


class MultiOutputSequential(nn.Sequential):
    def forward(self, *inputs):
        x = inputs  # Collect all inputs
        for module in self:
            if isinstance(x, tuple):  # Handle multiple outputs from previous layer
                x = module(*x)  # Pass multiple outputs as arguments
            else:
                x = module(x)  # Single input/output case
        return x
