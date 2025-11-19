from typing import List, Tuple

import torch.nn as nn

from skcausal._pytorch.utils.activation import parse_activation

__all__ = [
    "BaseFCHiddenBlock",
    "fc_block_from_config",
]


class BaseFCHiddenBlock(nn.Module):

    def __init__(self, input_size: int, hidden_config: List[Tuple[int, str]] = None):

        self.input_size = input_size
        self.hidden_config = hidden_config

        super().__init__()

        self._hidden_block = None
        if self.hidden_config is not None:
            self._hidden_block = fc_block_from_config(
                input_size=input_size, block_config=hidden_config
            )

    @property
    def input_size_last_layer(self):
        if self.hidden_config is None or len(self.hidden_config) == 0:
            return self.input_size
        return self.hidden_config[-1][0]

    def forward(self, x, t):
        if self._hidden_block is None:
            return x
        return self._hidden_block(x)


def fc_block_from_config(input_size, block_config):
    # construct the representation network
    hidden_blocks = []
    _input_size = input_size
    for layer_size, activation in block_config:
        # fc layer

        hidden_blocks.append(
            nn.Linear(in_features=_input_size, out_features=layer_size, bias=True)
        )

        hidden_blocks.append(parse_activation(activation))

        _input_size = layer_size

    return nn.Sequential(*hidden_blocks)
