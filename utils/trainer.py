import os
import time
from typing import (
    Callable,
    Optional
)

import torch


class Trainer:
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
                 batch_size: int, optimizer: torch.optim.Optimizer,
                 on_epoch_begin: Optional[Callable[["Trainer"], None]] = None):
        """
        Trainer class that handles training and validation epochs
        Args:
            model: The PyTorch model to train
            loss_fn: Function used to compute the loss of the model
            train_dataloader: DataLoader with a PyTorch DataLoader like interface, contains train data
            val_dataloader: DataLoader with a PyTorch DataLoader like interface, contains validation data
            optimizer: Optimizer to use
            model_name: LRCN if using an LSTM (reset the state at each step)
            on_epoch_begin: function that will be called at the beginning of every epoch.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # TODO: get get batchsize from the dataloader (require a dataloader wrapper that does videos, images, etc...)
        self.batch_size = batch_size
        self.on_epoch_begin = on_epoch_begin
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO: remove hardcoded 0 ?

        self.train_steps_per_epoch = (len(train_dataloader) + (batch_size - 1)) // batch_size
        self.val_steps_per_epoch = (len(val_dataloader) + (batch_size - 1)) // batch_size

    def epoch_loop(self, train: bool = True):
        """
        Does a pass on every batch of the train or validation dataset.
        Args:
            train: Whether it is a train or validation loop.
        """
        epoch_loss = 0.0
        self.model.train(train)
        data_loader = self.train_dataloader if train else self.val_dataloader
        steps_per_epoch = self.train_steps_per_epoch if train else self.val_steps_per_epoch
        step_time, fetch_time = None, None
        step_start_time = time.perf_counter()  # Needs to be outside the loop to include dataloading
        for step, batch in enumerate(data_loader, start=1):
            data_loading_finished_time = time.perf_counter()
            self.optimizer.zero_grad()
            if self.on_epoch_begin:
                self.on_epoch_begin(self)

            inputs, labels = batch["data"], batch["label"]
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
            self._print(step, steps_per_epoch, loss, step_time, fetch_time)

        return epoch_loss / self.train_steps_per_epoch

    def train_epoch(self):
        return self.epoch_loop()

    def val_epoch(self):
        with torch.no_grad():
            epoch_loss = self.epoch_loop(train=False)
        return epoch_loss

    @staticmethod
    def _print(step, max_steps, loss, step_time, fetch_time):
        pre_string = f"{step}/{max_steps} ["
        post_string = (f"],  Loss: {loss.item():.3e}  -  Step time: {step_time:.2f}ms"
                       f"  -  Fetch time: {fetch_time:.2f}ms    ")
        terminal_cols = os.get_terminal_size().columns
        progress_bar_len = min(terminal_cols - len(pre_string) - len(post_string)-1, 30)
        epoch_progress = int(progress_bar_len * (step/max_steps))
        print(pre_string + f"{epoch_progress*'='}>{(progress_bar_len-epoch_progress)*'.'}" + post_string,
              end='\r', flush=True)
