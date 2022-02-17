from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .batch_generator import BatchGenerator
from .metrics import Metrics
from .misc import clean_print


class TensorBoard(ABC):
    def __init__(self,
                 model: nn.Module,
                 tb_dir: Path,
                 metrics: Optional[Metrics] = None,
                 write_graph: bool = True,
                 input_shape: Optional[list[int]] = None):
        """Class with TensorBoard utility functions for classification-like tasks.

        Args:
            model (nn.Module): Pytorch model whose performance are to be recorded
            tb_dir (Path): Path to where the tensorboard files will be saved
            metrics (Metrics, optional): Instance of the Metrics class, used to compute classification metrics
            write_graph (bool): If True, add the network graph to the TensorBoard
            input_shape (list, optional): Shape of a sample. Must be given if adding the network graph.
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.metrics = metrics

        self.weights_warning_printed: bool = False  # Prints a warning if the network cannot give its weights

        self.train_tb_writer = SummaryWriter(tb_dir / "Train")
        self.val_tb_writer = SummaryWriter(tb_dir / "Validation")
        if write_graph:
            print("Adding network graph to TensorBoard")
            assert input_shape is not None, "You must provide the input shape to get the network graph"
            dummy_input = (torch.empty(2, *input_shape, device=self.device), )
            self.train_tb_writer.add_graph(model, dummy_input)
            self.train_tb_writer.flush()

    def close_writers(self) -> None:
        self.train_tb_writer.close()
        self.val_tb_writer.close()

    @abstractmethod
    def write_images(self,
                     epoch: int,
                     dataloader: BatchGenerator,
                     draw_fn: Callable[[Tensor, Tensor, Tensor, Optional[Any]], np.ndarray],
                     mode: str = "Train",
                     preprocess_fn: Optional[Callable[["TensorBoard", Tensor, Tensor], tuple[Tensor, Tensor]]] = None,
                     postprocess_fn: Optional[Callable[["TensorBoard", Tensor, Tensor], tuple[Tensor, Tensor]]] = None):
        """Writes images with predictions written on them to TensorBoard.

        Args:
            epoch (int): Current epoch
            dataloader (BatchGenerator): The images will be sampled from this dataset
            draw_fn (callable): Function that takes in the tensor images, labels and predictions
                                and draws on the images before returning them.
            mode (str): Either "Train" or "Validation"
            preprocess_fn (callable, optional): Function called before inference.
                                                Gets data and labels as input, expects them as outputs
            postprocess_fn (callable, optional): Function called after inference.
                                                 Gets data and predictions as input, expects them as outputs
        """
        return

    def write_metrics(self, epoch: int, mode: str = "Train") -> None:
        """Writes metrics in TensorBoard.

        Args:
            epoch (int): Current epoch
            mode (str): Either "Train" or "Validation"

        Raises:
            TypeError: No Metrics instance
        """
        if not isinstance(self.metrics, Metrics):
            raise TypeError("Trying to write metrics, but did not get a Metrics instance during initialization")

        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        metrics = self.metrics.get_metrics(mode)

        clean_print("Adding scalars to TensorBoard", end="\r")
        for scalar_metric_name, scalar_metric_value in metrics["scalars"].items():
            tb_writer.add_scalar(scalar_metric_name, scalar_metric_value, epoch)

        clean_print("Adding images to TensorBoard", end="\r")
        for img_metric_name, img_metric_value in metrics["imgs"].items():
            tb_writer.add_image(img_metric_name, img_metric_value, global_step=epoch, dataformats="HWC")

    def write_weights_grad(self, epoch: int):
        """Writes the model's weights and gradients to tensorboard, if the model can provide them.

        Args:
            epoch (int): Current epoch
        """
        try:
            for tag, (weight, grad) in self.model.get_weight_and_grads().items():  # type: ignore
                self.train_tb_writer.add_histogram(f"{tag}/weights", weight, epoch)
                self.train_tb_writer.add_histogram(f"{tag}/gradients", grad, epoch)
        except AttributeError:  # torch.nn.modules.module.ModuleAttributeError:
            if not self.weights_warning_printed:
                print(f"Warning: The model {self.model.__class__.__name__}"
                      f" does not support recording weights and gradients.")
                self.weights_warning_printed = True

    def write_loss(self, epoch: int, loss: float, mode: str = "Train"):
        """Writes loss metric in TensorBoard.

        Args:
            epoch (int): Current epoch
            loss (float): Epoch loss that will be added to the TensorBoard
            mode (str): Either "Train" or "Validation"
        """
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        tb_writer.add_scalar("Loss", loss, epoch)
        self.train_tb_writer.flush()

    def write_lr(self, epoch: int, lr: float):
        """Writes learning rate in the TensorBoard.

        Args:
            epoch (int): Current epoch
            lr (float): Learning rate for the given epoch
        """
        self.train_tb_writer.add_scalar("Learning Rate", lr, epoch)
