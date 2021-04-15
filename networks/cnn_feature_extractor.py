from typing import (
    Callable,
    Union
)

import torch.nn as nn

from .layers import (
    Conv2D,
    DarknetBlock
)
from .network_utils import layer_init


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 channels: list[int],
                 sizes: list[Union[int, tuple[int, int, int]]],
                 strides: list[Union[int, tuple[int, int, int]]],
                 paddings: list[Union[int, tuple[int, int, int]]],
                 layer_init: Callable[[nn.Module], None] = layer_init,
                 **kwargs):
        """ Feature extractor

        Args:
            channels (list): List with the number of channels for each convolution
            sizes (list): List with the kernel size for each convolution
            strides (list): List with the stride for each convolution
            paddings (list): List with the padding for each convolution
            layer_init (callable): Function used to initialise the layers of the network
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
    def __init__(self,
                 channels: list[int],
                 blocks: list[int],
                 layer_init: Callable[[nn.Module], None] = layer_init,
                 **kwargs):
        """ Feature extractor

        Args:
            channels (list): List with the number of channels for each convolution
            blocks (list): List the number of residual block in each darknet block
            layer_init (callable): Function used to initialise the layers of the network
        """
        super().__init__()

        self.blocks = nn.Sequential(*[DarknetBlock(channels[i], channels[i+1], blocks[i])
                                      for i in range(0, len(channels)-1)])

        if layer_init:
            self.apply(layer_init)

    def forward(self, inputs):
        x = self.blocks(inputs)
        return x
