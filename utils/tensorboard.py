import os
from typing import (
    Tuple,
    Dict,
    Optional,
    Callable
)

from einops import rearrange
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .draw import (
    draw_pred_img,
    draw_pred_video
)
from .metrics import Metrics


class TensorBoard():
    def __init__(self, model: nn.Module, metrics: Metrics, label_map: Dict[int, str], tb_dir: str,
                 sequence_length: int, gray_scale: bool, image_sizes: Tuple[int, int],
                 n_to_n: bool = False, write_graph: bool = True, max_outputs: int = 4):
        """
        Class with TensorBoard utility functions.
        Args:
            model: Model'whose performance are to be recorded
            metrics: Instance of the Metrics class, used to compute classification metrics
            label_map: Dictionary linking class index to class name
            tb_dir: Path to where the tensorboard files will be saved
            sequence_length: Number of elements in each sequence
            gray_scale: True if using gray scale
            image_sizes: Dimensions of the input images (width, height)
            n_to_n: If using videos, is it N to 1 or N to N
            max_outputs: Number of images kept and dislpayed in TensorBoard
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.metrics = metrics
        self.max_outputs = max_outputs
        self.label_map = label_map
        self.n_to_n = n_to_n

        self.train_tb_writer = SummaryWriter(os.path.join(tb_dir, "Train"))
        self.val_tb_writer = SummaryWriter(os.path.join(tb_dir, "Validation"))
        if write_graph:
            print("Adding TensorBoard to graph")
            self.train_tb_writer.add_graph(model, (torch.empty(2, sequence_length,
                                                               1 if gray_scale else 3,
                                                               image_sizes[0], image_sizes[1],
                                                               device=self.device), ))
            self.train_tb_writer.flush()

    def close_writers(self) -> None:
        self.train_tb_writer.close()
        self.val_tb_writer.close()

    def write_images(self, epoch: int, dataloader: torch.utils.data.DataLoader,
                     mode: str = "Train",  input_is_video: bool = "False",
                     preprocess_fn: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
                     postprocess_fn: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None) -> None:
        """
        Writes images with predictions written on them to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
            preprocess_fn: function called before inference. Gets data and labels as input, expects them as outputs
            postprocess: function called after inference. Gets data and predictions as input, expects them as outputs
        """
        print("Writing images" + ' ' * (os.get_terminal_size()[0] - len("Writing images")), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer

        batch = next(iter(dataloader))  # Get some data

        data, labels = batch["data"][:self.max_outputs].float(), batch["label"][:self.max_outputs]
        if preprocess_fn:
            data, labels = preprocess_fn(data, labels)

        # Get some predictions
        predictions = self.model(data.to(self.device))
        data, predictions = data[:self.max_outputs], predictions[:self.max_outputs]  # Just in case (damn LSTM)
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        if postprocess_fn:
            data, predictions = postprocess_fn(data, predictions)

        if input_is_video:
            # Keep only on frame per video (middle one)
            frame_to_keep = labels.shape[1] // 2
            imgs = data[:, frame_to_keep, :, :, :]
            if self.n_to_n:
                predictions = predictions[:, frame_to_keep]
                labels = labels[:, frame_to_keep]
        else:
            imgs = data

        # Write prediction on the images
        out_imgs = draw_pred_img(imgs, predictions, labels, self.label_map)

        # Add them to TensorBoard
        for image_index, out_img in enumerate(out_imgs):
            tb_writer.add_image(f"{mode}/prediction_{image_index}", out_img, global_step=epoch, dataformats="HWC")

    def write_videos(self, epoch: int, dataloader: torch.utils.data.DataLoader, mode: str = "Train",
                     preprocess_fn: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
                     postprocess_fn: Optional[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]] = None):
        """
        Write a video with predictions written on it to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
            preprocess_fn: function called before inference. Gets data and labels as input, expects them as outputs
            postprocess: function called after inference. Gets data and predictions as input, expects them as outputs
        """
        print("Writing videos" + ' ' * (os.get_terminal_size()[0] - len("Writing videos")), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer

        batch = next(iter(dataloader))  # Get some data

        videos, labels = batch["data"][:1].float(), batch["label"][:1]
        if preprocess_fn:
            videos, labels = preprocess_fn(videos, labels)

        # Get some predictions
        predictions = self.model(videos.to(self.device))
        videos, predictions = videos[:1], predictions[:1]  # Just in case (damn LSTM)
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        if postprocess_fn:
            videos, predictions = postprocess_fn(videos, predictions)

        # Write prediction on a video and add it to TensorBoard
        out_video = draw_pred_video(videos[0], predictions[0], labels[0], self.label_map, self.n_to_n)
        out_video = rearrange(out_video, 't h w c -> t c h w')
        out_video = np.expand_dims(out_video, 0)  # Re-add batch dimension

        tb_writer.add_video("Video", out_video, global_step=epoch, fps=16)

    def write_metrics(self, epoch: int, mode: str = "Train") -> float:
        """
        Writes accuracy metrics in TensorBoard
        Args:
            epoch: Current epoch
            mode: Either "Train" or "Validation"
        Returns:
            avg_acc: Average accuracy
        """
        print("Computing confusion matrix" + ' ' * (os.get_terminal_size()[0] - 26), end="\r", flush=True)
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        self.metrics.compute_confusion_matrix(mode=mode)

        print("Computing average accuracy" + ' ' * (os.get_terminal_size()[0] - 26), end="\r", flush=True)
        avg_acc = self.metrics.get_avg_acc()
        tb_writer.add_scalar("Average Accuracy", avg_acc, epoch)

        print("Computing per class accuracy" + ' ' * (os.get_terminal_size()[0] - 28), end="\r", flush=True)
        per_class_acc = self.metrics.get_class_accuracy()
        for key, acc in enumerate(per_class_acc):
            tb_writer.add_scalar(f"Per Class Accuracy/{self.label_map[key]}", acc, epoch)

        print("Creating confusion matrix image" + ' ' * (os.get_terminal_size()[0] - 31), end="\r", flush=True)
        confusion_matrix = self.metrics.get_confusion_matrix()
        tb_writer.add_image("Confusion Matrix", confusion_matrix, global_step=epoch, dataformats="HWC")

        return avg_acc

    def write_loss(self, epoch: int, loss: float, mode: str = "Train"):
        """
        Writes loss metric in TensorBoard
        Args:
            epoch: Current epoch
            loss: Epoch loss that will be added to the TensorBoard
            mode: Either "Train" or "Validation"
        """
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        tb_writer.add_scalar("Loss", loss, epoch)
        self.train_tb_writer.flush()

    def write_lr(self, epoch: int, lr: float):
        """
        Writes learning rate in the TensorBoard
        Args:
            epoch: Current epoch
            lr: Learning rate for the given epoch
        """
        self.train_tb_writer.add_scalar("Learning Rate", lr, epoch)
