import shutil
import time
from typing import (
    Callable,
    Optional
)

import numpy as np
import numpy.typing as npt
import torch

from .batch_generator import BatchGenerator


class Trainer:
    """Trainer class that handles training and validation epochs."""
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 use_amp: bool = False,
                 loss_names: Optional[list[str]] = None,
                 on_epoch_begin: Optional[Callable[["Trainer"], None]] = None):
        """Initialize the trainer instance.

        Args:
            model (torch.nn.Module): The PyTorch model to train
            loss_fn (torch.nn.Module): Function used to compute the loss of the model
            optimizer (torch.optim.Optimizer): Optimizer to use
            train_dataloader (BatchGenerator): DataLoader with a PyTorch DataLoader like interface, contains train data
            val_dataloader (BatchGenerator): DataLoader containing  validation data
            use_amp (bool): If True then use Mixed Precision Training.
            loss_names: List with the name for each loss component.
            on_epoch_begin (callable): function that will be called at the beginning of every epoch.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = train_dataloader.batch_size
        self.loss_names = loss_names if loss_names is not None else ["Loss"]
        self.use_amp = use_amp
        self.on_epoch_begin = on_epoch_begin
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def epoch_loop(self, train: bool = True) -> npt.NDArray[np.float32] | float:
        """Does a pass on every batch of the train or validation dataset.

        Args:
            train (bool): Whether it is a train or validation loop.
        """
        epoch_losses = np.zeros(len(self.loss_names), dtype=np.float32)
        self.model.train(train)
        data_loader = self.train_dataloader if train else self.val_dataloader
        step_time: float | None = None
        fetch_time: float | None = None
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, (inputs, labels) in enumerate(data_loader, start=1):
            data_loading_finished_time = time.perf_counter()
            self.optimizer.zero_grad()
            if self.on_epoch_begin:
                self.on_epoch_begin(self)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.model(inputs)
                losses = self.loss_fn(outputs, labels)
                total_loss: torch.Tensor = sum(losses) if isinstance(losses, tuple) else losses
                if train:
                    total_loss.backward()
                    self.optimizer.step()
            epoch_losses += [loss.item() for loss in losses] if isinstance(losses, tuple) else [losses.item()]

            previous_step_start_time = step_start_time
            if step_time and fetch_time:
                step_time = 0.9*step_time + 0.1*1000*(time.perf_counter() - step_start_time)
                fetch_time = 0.9*fetch_time + 0.1*1000*(data_loading_finished_time - previous_step_start_time)
            else:
                step_time = 1000*(time.perf_counter() - step_start_time)
                fetch_time = 1000*(data_loading_finished_time - previous_step_start_time)
            step_start_time = time.perf_counter()
            self._print(step, data_loader.steps_per_epoch, losses if isinstance(losses, tuple) else (losses,),
                        self.loss_names, step_time, fetch_time)

        epoch_losses = epoch_losses / data_loader.steps_per_epoch
        return epoch_losses if len(epoch_losses) > 1 else float(epoch_losses)  # For backward compatibility

    def train_epoch(self):
        """Performs a training epoch."""
        return self.epoch_loop()

    def val_epoch(self):
        """Performs a validation epoch."""
        with torch.no_grad():
            epoch_loss = self.epoch_loop(train=False)
        return epoch_loss

    @staticmethod
    def _print(step: int,
               max_steps: int,
               losses: tuple[torch.Tensor, ...],
               loss_names: list[str],
               step_time: float,
               fetch_time: float):
        """Prints information related to the current step.

        Args:
            step (int): Current step (within the epoch)
            max_steps (int): Number of steps in the current epoch
            losses: Tuple with the loss(es) for the current step.
            loss_names: List with the name for each loss component.
            step_time (float): Time it took to perform the whole step
            fetch_time (float): Time it took to load the data for the step
        """
        pre_string = f"{step}/{max_steps} ["
        post_string = "],  " + ",  ".join([f"{name}: {loss.item():.3e}" for name, loss in zip(loss_names, losses)])
        post_string += f"  -  Step time: {step_time:.2f}ms  -  Fetch time: {fetch_time:.2f}ms    "
        terminal_cols = shutil.get_terminal_size(fallback=(156, 38)).columns
        progress_bar_len = min(terminal_cols - len(pre_string) - len(post_string)-1, 30)
        epoch_progress = int(progress_bar_len * (step/max_steps))
        print(pre_string + f"{epoch_progress*'='}>{(progress_bar_len-epoch_progress)*'.'}" + post_string,
              end=('\r' if step < max_steps else '\n'), flush=True)
