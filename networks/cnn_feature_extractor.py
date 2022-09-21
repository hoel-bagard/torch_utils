from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from .layers import Conv2D, DarknetBlock
from .network_utils import layer_init


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 channels: list[int],
                 kernel_sizes: list[int],
                 strides: list[Union[int, tuple[int, int]]],
                 paddings: list[Union[int, tuple[int, int]]],
                 layer_init_fn: Optional[Callable[[nn.Module], None]] = layer_init,
                 **kwargs: dict[str, object]):
        """Feature extractor.

        Args:
            channels: List with the number of channels for each convolution.
            kernel_sizes: List with the kernel size for each convolution.
            strides: List with the stride for each convolution.
            paddings: List with the padding for each convolution.
            layer_init_fn: Function used to initialise the layers of the network.
        """
        super().__init__()

        self.blocks = nn.Sequential(*[Conv2D(channels[i],
                                             channels[i+1],
                                             kernel_sizes[i],
                                             stride=strides[i],
                                             padding=paddings[i],
                                             **kwargs)
                                      for i in range(0, len(channels)-1)])

        if layer_init_fn is not None:
            self.apply(layer_init_fn)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.blocks(inputs)
        return x


class DarknetFeatureExtrator(nn.Module):
    def __init__(self,
                 channels: list[int],
                 blocks: list[int],
                 layer_init_fn: Optional[Callable[[nn.Module], None]] = layer_init,
                 **kwargs: dict[str, object]):
        """Feature extractor.

        Args:
            channels: List with the number of channels for each convolution.
            blocks: List the number of residual block in each darknet block.
            layer_init_fn: Function used to initialise the layers of the network>.
        """
        super().__init__()

        self.blocks = nn.Sequential(*[DarknetBlock(channels[i], channels[i+1], blocks[i], **kwargs)
                                      for i in range(0, len(channels)-1)])

        if layer_init_fn is not None:
            self.apply(layer_init_fn)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.blocks(inputs)
        return x
