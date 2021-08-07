import shutil
import time
from typing import (
    Callable,
    Optional
)

import torch

from .batch_generator import BatchGenerator
from .projected_gradient_descent import projected_gradient_descent


class Trainer:
    """ Trainer class that handles training and validation epochs"""
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 train_dataloader: BatchGenerator, val_dataloader: BatchGenerator,
                 on_epoch_begin: Optional[Callable[["Trainer"], None]] = None):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to train
            loss_fn (torch.nn.Module): Function used to compute the loss of the model
            optimizer (torch.optim.Optimizer): Optimizer to use
            train_dataloader (BatchGenerator): DataLoader with a PyTorch DataLoader like interface, contains train data
            val_dataloader (BatchGenerator): DataLoader containing  validation data
            on_epoch_begin (callable): function that will be called at the beginning of every epoch.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = train_dataloader.batch_size
        self.on_epoch_begin = on_epoch_begin
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def epoch_loop(self, train: bool = True):
        """ Does a pass on every batch of the train or validation dataset.

        Args:
            train (bool): Whether it is a train or validation loop.
        """
        epoch_loss = 0.0
        self.model.train(train)
        data_loader = self.train_dataloader if train else self.val_dataloader
        step_time, fetch_time = None, None
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, (inputs, labels) in enumerate(data_loader, start=1):
            data_loading_finished_time = time.perf_counter()
            self.optimizer.zero_grad()
            if self.on_epoch_begin:
                self.on_epoch_begin(self)

            if train:
                inputs = projected_gradient_descent(self.model, inputs, labels, self.loss_fn)
                # inputs = projected_gradient_descent(self.model, inputs, labels, self.loss_fn, 1, 2, 4, 4, 2)
                # self.optimizer.zero_grad()  # Not sure if this is good but...

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            if train:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()

            previous_step_start_time = step_start_time
            if step_time:
                step_time = 0.9*step_time + 0.1*1000*(time.perf_counter() - step_start_time)
                fetch_time = 0.9*fetch_time + 0.1*1000*(data_loading_finished_time - previous_step_start_time)
            else:
                step_time = 1000*(time.perf_counter() - step_start_time)
                fetch_time = 1000*(data_loading_finished_time - previous_step_start_time)
            step_start_time = time.perf_counter()
            self._print(step, data_loader.steps_per_epoch, loss, step_time, fetch_time)

        return epoch_loss / data_loader.steps_per_epoch

    def train_epoch(self):
        """Performs a training epoch"""
        return self.epoch_loop()

    def val_epoch(self):
        """Performs a validation epoch"""
        with torch.no_grad():
            epoch_loss = self.epoch_loop(train=False)
        return epoch_loss

    @staticmethod
    def _print(step: int, max_steps: int, loss: float, step_time: float, fetch_time: float):
        """ Prints information related to the current step

        Args:
            step (int): Current step (within the epoch)
            max_steps (int): Number of steps in the current epoch
            loss (float): Loss of current step
            step_time (float): Time it took to perform the whole step
            fetch_time (float): Time it took to load the data for the step
        """
        pre_string = f"{step}/{max_steps} ["
        post_string = (f"],  Loss: {loss.item():.3e}  -  Step time: {step_time:.2f}ms"
                       f"  -  Fetch time: {fetch_time:.2f}ms    ")
        terminal_cols = shutil.get_terminal_size(fallback=(156, 38)).columns
        progress_bar_len = min(terminal_cols - len(pre_string) - len(post_string)-1, 30)
        epoch_progress = int(progress_bar_len * (step/max_steps))
        print(pre_string + f"{epoch_progress*'='}>{(progress_bar_len-epoch_progress)*'.'}" + post_string,
              end=('\r' if step < max_steps else '\n'), flush=True)
