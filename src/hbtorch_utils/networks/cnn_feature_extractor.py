from collections.abc import Callable
from typing import Self

import torch
from torch import nn

from .layers import Conv2D, DarknetBlock
from .network_utils import layer_init


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self: Self,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int | tuple[int, int]],
        paddings: list[int | tuple[int, int]],
        layer_init_fn: Callable[[nn.Module], None] | None = layer_init,
        **kwargs: dict[str, object],
    ) -> None:
        """Feature extractor.

        Args:
            channels: List with the number of channels for each convolution.
            kernel_sizes: List with the kernel size for each convolution.
            strides: List with the stride for each convolution.
            paddings: List with the padding for each convolution.
            layer_init_fn: Function used to initialise the layers of the network.
            kwargs: kwargs
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

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)


class DarknetFeatureExtrator(nn.Module):
    def __init__(
        self: Self,
        channels: list[int],
        blocks: list[int],
        layer_init_fn: Callable[[nn.Module], None] | None = layer_init,
        **kwargs: dict[str, object],
    ) -> None:
        """Feature extractor.

        Args:
            channels: List with the number of channels for each convolution.
            blocks: List the number of residual block in each darknet block.
            layer_init_fn: Function used to initialise the layers of the network>.
            kwargs: kwargs
        """
        super().__init__()

        self.blocks = nn.Sequential(*[DarknetBlock(channels[i], channels[i+1], blocks[i], **kwargs)
                                      for i in range(0, len(channels)-1)])

        if layer_init_fn is not None:
            self.apply(layer_init_fn)

    def forward(self: Self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(inputs)
