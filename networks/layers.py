from typing import (
    Union,
    Tuple
)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, momentum=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.1)

        return x


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
