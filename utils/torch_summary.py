"""Taken from https://github.com/sksq96/pytorch-summary."""

from collections import OrderedDict
from typing import Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn


class SummaryEntry(TypedDict):
    input_shape: list[int]
    output_shape: list[list[int]] | list[int]
    trainable: bool
    nb_params: int


def summary(model: nn.Module,
            input_shape: tuple[int, ...] | list[tuple[int, ...]],
            line_length: int = 64,
            batch_size: int = -1,
            device: Optional[torch.device] = None,
            dtypes: Optional[torch.TensorType | list[torch.TensorType]] = None) -> list[str]:
    """Make a summary of the given model.

    # TODO: Get the layers' names (they appear when simply printing a model)

    Args:
        model: The model whose summary should be created.
        input_shape: The input shape of the network.
        line_length: Number of caracters per line.
        batch_size: ?
        device: ?
        dtypes: Type of each input, should have the same shape as input_shape

    Returns:
        A list where each entry is a line of the summary.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_shape)  # type: ignore
    elif isinstance(input_shape, list) and (not isinstance(dtypes, list) or len(dtypes) != len(input_shape)):
        raise ValueError(("The number of values given for the input shapes and the types do not match:"
                          f" {input_shape=}, {dtypes=}"))

    def register_hook(module: nn.Module):
        def hook(module: nn.Module,
                 inputs: torch.Tensor,
                 output: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor):
            class_name = str(module.__class__).rsplit(".", maxsplit=1)[-1].split("'")[0]
            module_idx = len(summary_dict)

            m_key = f"{class_name}-{module_idx+1}"
            input_shape_module = [batch_size, *list(inputs[0].size())[1:]]
            if isinstance(output, (list, tuple)):
                output_shape = [[-1] + list(o.size())[1:] for o in output]
            else:
                output_shape = [batch_size, *list(output.size())[1:]]

            params: int = 0
            trainable = False
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))  # type: ignore
                trainable = bool(module.weight.requires_grad)  # Bool should do nothing, pytorch typing seems wrong.
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))  # type: ignore

            summary_dict[m_key] = SummaryEntry(input_shape=input_shape_module,
                                               output_shape=output_shape,
                                               trainable=trainable,
                                               nb_params=params)

        if (not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    # Multiple inputs to the network
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]

    # Batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)  # type: ignore
         for in_size, dtype in zip(input_shape, dtypes)]  # type: ignore

    # Create properties
    summary_dict: OrderedDict[str, SummaryEntry] = OrderedDict()
    hooks: list[torch.utils.hooks.RemovableHandle] = []  # type: ignore

    # Register hook
    model.apply(register_hook)

    # Make a forward pass
    model(*x)

    # Remove these hooks
    for h in hooks:  # type: ignore
        h.remove()

    summary_lines: list[str] = []

    summary_lines.append(line_length*"-")
    summary_lines.append(f"{'Layer (type)':>20}  {'Output Shape':>25} {'Param #':>15}")
    summary_lines.append(line_length*"=")
    total_params = 0
    total_output: int = 0
    trainable_params = 0
    for layer in summary_dict:
        # input_shape, output_shape, trainable, nb_params
        summary_lines.append(f"{layer:>20} {str(summary_dict[layer]['output_shape']):>25} "
                             f"{summary_dict[layer]['nb_params']:>15,}")
        total_params += summary_dict[layer]["nb_params"]

        total_output += np.prod(summary_dict[layer]["output_shape"])  # type: ignore
        if "trainable" in summary_dict[layer]:
            if summary_dict[layer]["trainable"]:
                trainable_params += summary_dict[layer]["nb_params"]

    # Assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_shape, ())) * batch_size * 4 / (1024 ** 2))
    total_output_size = abs(2 * total_output * 4 / (1024 ** 2))  # x2 for gradients
    total_params_size = abs(total_params * 4 / (1024 ** 2))
    total_size = total_params_size + total_output_size + total_input_size

    summary_lines.append(line_length*"=")
    summary_lines.append(f"Total params: {total_params:,}")
    summary_lines.append(f"Trainable params: {trainable_params:,}")
    summary_lines.append(f"Non-trainable params: {total_params - trainable_params:,}")
    summary_lines.append(line_length*"-")
    summary_lines.append(f"Input size (MB): {total_input_size:.2f}")
    summary_lines.append(f"Forward/backward pass size (MB): {total_output_size:.2f}")
    summary_lines.append(f"Params size (MB): {total_params_size:.2f}")
    summary_lines.append(f"Estimated Total Size (MB): {total_size:.2f}")
    summary_lines.append(line_length*"-")

    return summary_lines
