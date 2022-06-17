from abc import ABC
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .batch_generator import BatchGenerator
from .metrics import Metrics
from .misc import clean_print


class TensorBoard(ABC):
    def __init__(self,
                 model: nn.Module,
                 tb_dir: Path,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 logger: Logger,
                 metrics: Optional[Metrics] = None,
                 write_graph: bool = True):
        """Class with TensorBoard utility functions for classification-like tasks.

        Args:
            model (nn.Module): Pytorch model whose performance are to be recorded
            tb_dir (Path): Path to where the tensorboard files will be saved
            train_dataloader (BatchGenerator): DataLoader with a PyTorch DataLoader like interface, contains train data
            val_dataloader (BatchGenerator): DataLoader containing  validation data
            logger (Logger): Used to print things.
            metrics (Metrics, optional): Instance of the Metrics class, used to compute classification metrics
            write_graph (bool): If True, add the network graph to the TensorBoard
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tb_dir = tb_dir
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.metrics = metrics

        self.weights_warning_printed: bool = False  # Prints a warning if the network cannot give its weights

        self.train_tb_writer = SummaryWriter(tb_dir / "Train")
        self.val_tb_writer = SummaryWriter(tb_dir / "Validation")
        if write_graph:
            self.logger.info("Adding network graph to TensorBoard")
            dummy_input = (torch.empty(2, *self.train_dataloader.data_shape, device=self.device), )
            self.train_tb_writer.add_graph(model, dummy_input)
            self.train_tb_writer.flush()

    def close_writers(self) -> None:
        self.train_tb_writer.close()
        self.val_tb_writer.close()

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

    def write_config(self,
                     config: dict[str, int | float | str | bool | torch.Tensor | list[float | int] | tuple],
                     metrics: Optional[dict[str, float]] = None):
        """Writes the config (and optionally metrics) to the TensorBoard.

        Args:
            config: The config to add to the TensorBoard.
                    The config's values should be "simple" values.
                    If a tuple/list is given, it will either be split up into several values or transformed into an str.
            metrics: The metrics for this run.
        """
        fixed_config: dict[str, int | float | str | bool | torch.Tensor] = {}
        for key, value in config.items():
            if isinstance(value, (tuple, list, np.ndarray)):
                if len(value) < 3:
                    if isinstance(value, np.ndarray):
                        value = value.tolist()  # Convert from numpy types to python ones
                    for i in range(len(value)):
                        fixed_config[key + f"[{i}]"] = float(value[i])
                else:
                    fixed_config[key] = str(value)
            elif isinstance(value, int | float | str | bool | torch.Tensor):
                fixed_config[key] = value
            else:  # For things like None or other random types
                fixed_config[key] = str(value)

        config_tb_writer = SummaryWriter(self.tb_dir)
        config_tb_writer.add_hparams(fixed_config, metrics if metrics is not None else {"None": 0}, run_name="Config")
        config_tb_writer.close()


if __name__ == "__main__":
    def _main():
        import argparse
        from .logger import create_logger
        parser = argparse.ArgumentParser(description=("Script to test the Tensorboard template. "
                                                      "Run with 'python -m utils.tensorboard_template'."),
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--output_path", "-o", type=Path, default=Path("tb_temp"),
                            help="Path to where the TB file will be saved")
        parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                            help="Logger level.")
        args = parser.parse_args()

        output_path: Path = args.output_path

        logger = create_logger("Test TB", verbose_level=args.verbose_level)

        def _test_config():
            config = {"lr": 4e-3, "Batch Size": 32, "image_sizes": (224, 224)}
            metrics = {"Final/Accuracy": 0.99, "Final/counter": 3}
            tensorboard = TensorBoard(None, output_path, None, None, None, write_graph=False)
            tensorboard.write_config(config, metrics)
            logger.info(f"Tested the config writing part. TB can be found at {output_path}")
        _test_config()
    _main()
