import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn


def layer_init(
    layer: nn.Module,
    weight_gain: int = 1,
    bias_const: float = 0,
    weights_init: str = "xavier",
    bias_init: str = "zeros",
) -> None:
    """Layer initialisation function.

    Args:
        layer: layer to be initialized.
        weight_gain: Use nn.init.calculate_gain.
        bias_const: ?
        weights_init: Can be "xavier", "orthogonal" or "uniform".
        bias_init: Can be "zeros", "uniform".
    """
    if isinstance(layer, nn.Linear):
        if weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)
    if isinstance(layer, nn.Conv2d):
        if weights_init == "xavier":
            nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        else:
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()


def xavier_init(
    gain: float = 1,
    bias: float = 0,
    distribution: Literal["uniform", "normal"] = "normal",
) -> Callable[[nn.Module], None]:
    def xavier_init_fn(module: nn.Module) -> None:
        if hasattr(module, "weight") and module.weight is not None:
            assert isinstance(module.weight, torch.Tensor)  # noqa: S101  For pyright
            if distribution == "uniform":
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            assert isinstance(module.bias, torch.Tensor)  # noqa: S101  For pyright
            nn.init.constant_(module.bias, bias)
    return xavier_init_fn


def normal_init(
    mean: float = 0,
    std: float = 1,
    bias: float = 0,
) -> Callable[[nn.Module], None]:
    def normal_init_fn(module: nn.Module) -> None:
        if hasattr(module, "weight") and module.weight is not None:
            assert isinstance(module.weight, torch.Tensor)  # noqa: S101  For pyright
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            assert isinstance(module.bias, torch.Tensor)  # noqa: S101  For pyright
            nn.init.constant_(module.bias, bias)
    return normal_init_fn


def get_cnn_output_size(
    image_sizes: tuple[int, int],
    sizes: list[int],
    strides: list[int],
    paddings: list[int],
    output_channels: int | None = None,
    *,
    dense: bool = True,
) -> int | tuple[int, int]:
    """Compute the output size of a cnn  (flattened).

    Args:
        image_sizes: Dimensions of the input image (width, height).
        sizes: List with the kernel size for each convolution
        strides: List with the stride for each convolution
        paddings: List with the padding for each convolution
        output_channels: Number of output channels of the last convolution, required if dense=True
        dense: If True, then this function returns an int (number of values) otherwise it returns [width, height]

    Returns:
        The output size of the cnn, either as an int or a (width, height) tuple.
    """
    width, height = image_sizes
    for kernel_size, stride, padding in zip(sizes, strides, paddings, strict=True):
        width = ((width - kernel_size + 2*padding) // stride) + 1

    for kernel_size, stride, padding in zip(sizes, strides, paddings, strict=True):
        height = ((height - kernel_size + 2*padding) // stride) + 1

    if dense:
        if output_channels is None:
            msg = "The output_channels argument is required in the 'dense' case."
            raise ValueError(msg)
        return width*height*output_channels
    return (width, height)
