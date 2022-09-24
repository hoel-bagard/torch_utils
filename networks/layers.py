from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange  # type: ignore


class Layer(nn.Module):
    # Default layer arguments
    ACTIVATION = torch.nn.LeakyReLU
    ACTIVATION_KWARGS = {"negative_slope": 0.1}

    USE_BATCH_NORM = True
    BATCH_NORM_TRAINING = True
    BATCH_NORM_MOMENTUM = 0.01

    def __init__(self, activation: Callable[[torch.Tensor], torch.Tensor] | Literal[0], use_batch_norm: Optional[bool]):
        super().__init__()
        # Preload default
        self.batch_norm: Optional[torch.nn.Module] = None
        self.activation = Layer.ACTIVATION(**Layer.ACTIVATION_KWARGS) if activation == 0 else activation
        self.use_batch_norm = Layer.USE_BATCH_NORM if use_batch_norm is None else use_batch_norm

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        output = input_data
        if self.activation is not None:
            output = self.activation(output)
        if self.use_batch_norm and self.batch_norm is not None:
            # It is assumed here that if using batch norm, then self.batch_norm has been instanciated.
            output = self.batch_norm(output)
        return output


class Conv2D(Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 activation: Callable[[torch.Tensor], torch.Tensor] | Literal[0] = 0,
                 use_batch_norm: Optional[bool] = None,
                 **kwargs: dict[str, object]):
        super().__init__(activation, use_batch_norm)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=not self.use_batch_norm, **kwargs)
        self.batch_norm = nn.BatchNorm2d(
            out_channels, momentum=Layer.BATCH_NORM_MOMENTUM,
            track_running_stats=Layer.BATCH_NORM_TRAINING) if self.use_batch_norm else None

    def forward(self, input_data: torch.Tensor):
        return super().forward(self.conv(input_data))


class Conv3D(Layer):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 activation: Callable[[torch.Tensor], torch.Tensor] | Literal[0] = 0,
                 use_batch_norm: Optional[bool] = None,
                 **kwargs: dict[str, object]):
        super().__init__(activation, use_batch_norm)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              bias=not self.use_batch_norm, **kwargs)
        self.batch_norm = nn.BatchNorm3d(
            out_channels, momentum=Layer.BATCH_NORM_MOMENTUM,
            track_running_stats=Layer.BATCH_NORM_TRAINING) if self.use_batch_norm else None

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return super().forward(self.conv(input_data))


class DarknetResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.convs = nn.Sequential(Conv2D(filters, filters//2, 1),
                                   Conv2D(filters//2, filters, 3, padding=1),)

    def forward(self, inputs: torch.Tensor):
        return torch.add(inputs, self.convs(inputs))


class DarknetBlock(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, blocks: int):
        super().__init__()
        self.conv = Conv2D(in_filters, out_filters, 3, stride=2)
        self.dark_res_blocks = nn.Sequential(*[DarknetResidualBlock(out_filters) for _ in range(blocks)])

    def forward(self, inputs: torch.Tensor):
        x = self.conv(inputs)
        for dark_res_block in self.dark_res_blocks:
            x = dark_res_block(x)
        return x


class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        if "(" in pattern or ")" in pattern:
            self.permute = False
            self.pattern = pattern
        else:
            self.permute = True
            left_expr, right_expr = pattern.split(" -> ")
            right_expr_split, left_expr_split = left_expr.split(), right_expr.split()
            self.permute_pattern = [right_expr_split.index(symbol) for symbol in left_expr_split]

    def forward(self, x: torch.Tensor):
        if self.permute:
            x = x.permute(*self.permute_pattern)
        else:
            x = rearrange(x, self.pattern)
        return x
