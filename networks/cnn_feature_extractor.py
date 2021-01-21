from typing import (
    Callable,
    List
)

import torch.nn as nn

from .layers import (
    Conv2D,
    DarknetBlock
)
from .network_utils import layer_init


class CNNFeatureExtractor(nn.Module):
    def __init__(self, channels: List[int] = [], sizes: List[int] = [], strides: List[int] = [],
                 paddings: List[int] = [], layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        Feature extractor
        Args:
            channels: List with the number of channels for each convolution
            sizes: List with the kernel size for each convolution
            strides: List with the stride for each convolution
            padding: List with the padding for each convolution
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()

        self.blocks = nn.Sequential(*[Conv2D(channels[i], channels[i+1], sizes[i], stride=strides[i],
                                             padding=paddings[i])
                                      for i in range(0, len(channels)-1)])

        if layer_init:
            self.apply(layer_init)

    def forward(self, inputs):
        x = self.blocks(inputs)
        return x


class DarknetFeatureExtrator(nn.Module):
    def __init__(self, channels: List[int] = [], blocks: List[int] = [],
                 layer_init: Callable[[nn.Module], None] = layer_init, **kwargs):
        """
        Feature extractor
        Args:
            channels: List with the number of channels for each convolution
            sizes: List with the kernel size for each convolution
            strides: List with the stride for each convolution
            padding: List with the padding for each convolution
            layer_init: Function used to initialise the layers of the network
        """
        super().__init__()

        self.blocks = nn.Sequential(*[DarknetBlock(channels[i], channels[i+1], blocks[i])
                                      for i in range(0, len(channels)-1)])

        if layer_init:
            self.apply(layer_init)

    def forward(self, inputs):
        x = self.blocks(inputs)
        return x
