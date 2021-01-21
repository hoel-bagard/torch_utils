from typing import (
    Union,
    Tuple,
    Optional
)

import torch
import torch.nn as nn


class Layer(nn.Module):
    # Default layer arguments
    ACTIVATION = torch.nn.LeakyReLU(negative_slope=0.1)

    BATCH_NORM_TRAINING = True
    BATCH_NORM_MOMENTUM = 0.01

    def __init__(self, activation):
        super().__init__()
        # Preload default
        self.activation = Layer.ACTIVATION if activation == 0 else activation

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        output = input_data
        if self.activation is not None:
            output = self.activation(output)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        return output


class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 activation=0, use_batch_norm: bool = True, **kwargs):
        super().__init__(activation)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=not use_batch_norm)
        self.batch_norm = nn.BatchNorm2d(out_channels, momentum=Layer.BATCH_NORM_MOMENTUM,
                                         track_running_stats=Layer.BATCH_NORM_TRAINING) if use_batch_norm else None

    def forward(self, x):
        return super().forward(self.conv(x))


class Conv3d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 activation=0, use_batch_norm: bool = True, **kwargs):
        super().__init__(activation)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=not self.use_batch_norm, **kwargs)
        self.batch_norm = nn.BatchNorm3d(out_channels, momentum=Layer.BATCH_NORM_MOMENTUM,
                                         track_running_stats=Layer.BATCH_NORM_TRAINING) if self.use_batch_norm else None

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return super().forward(self.conv(input_data))


class DarknetResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = Conv2D(filters, filters//2, 1)
        self.conv2 = Conv2D(filters//2, filters, 3, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = torch.add(inputs, x)
        return x


class DarknetBlock(nn.Module):
    def __init__(self, in_filters, out_filters, blocks):
        super().__init__()
        self.conv = Conv2D(in_filters, out_filters, 3, stride=2)
        self.dark_res_blocks = nn.Sequential(*[DarknetResidualBlock(out_filters) for _ in range(blocks)])

    def forward(self, inputs):
        x = self.conv(inputs)
        for dark_res_block in self.dark_res_blocks:
            x = dark_res_block(x)
        return x
