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
from torch.nn.modules.module import ModuleAttributeError
from torch.utils.tensorboard import SummaryWriter

from .draw import (
    draw_pred_img,
    draw_pred_video,
    draw_segmentation
)
from .metrics import Metrics
from .misc import clean_print
from .batch_generator import BatchGenerator


class TensorBoard:
    def __init__(self, model: nn.Module, metrics: Metrics, label_map: Dict[int, str], tb_dir: str,
                 image_sizes: Tuple[int, int], gray_scale: bool = False,
                 color_map: Optional[list[tuple[int, int, int]]] = None,
                 n_to_n: bool = False, sequence_length: Optional[int] = None, segmentation: bool = True,
                 write_graph: bool = True, max_outputs: int = 4):
        """
        Class with TensorBoard utility functions.
        Args:
            model: Model'whose performance are to be recorded
            metrics: Instance of the Metrics class, used to compute classification metrics
            label_map: Dictionary linking class index to class name
            color_map: List linking class index to class color
            tb_dir: Path to where the tensorboard files will be saved
            gray_scale: True if using gray scale
            image_sizes: Dimensions of the input images (width, height)
            n_to_n: If using videos, is it N to 1 or N to N
            sequence_length: If using videos, Number of elements in each sequence
            segmentation: If doing segmentation
            max_outputs (int): Maximal number of images kept and displayed in TensorBoard (per function call)
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.metrics = metrics
        self.max_outputs = max_outputs
        self.label_map = label_map
        self.color_map = color_map
        self.n_to_n = n_to_n
        self.segmentation = segmentation

        self.weights_warning_printed: bool = False  # Prints a warning if the network cannot give its weights

        self.train_tb_writer = SummaryWriter(os.path.join(tb_dir, "Train"))
        self.val_tb_writer = SummaryWriter(os.path.join(tb_dir, "Validation"))
        if write_graph:
            print("Adding network graph to TensorBoard")
            if sequence_length:
                dummy_input = (torch.empty(2, sequence_length, 1 if gray_scale else 3,
                                           image_sizes[0], image_sizes[1], device=self.device), )
            else:
                dummy_input = (torch.empty(2, 1 if gray_scale else 3,
                                           image_sizes[0], image_sizes[1], device=self.device), )
            self.train_tb_writer.add_graph(model, dummy_input)
            self.train_tb_writer.flush()

    def close_writers(self) -> None:
        self.train_tb_writer.close()
        self.val_tb_writer.close()

    def write_images(self, epoch: int, dataloader: BatchGenerator,
                     draw_fn: Callable[[Tensor, Tensor], np.ndarray] = draw_pred_img,
                     mode: str = "Train", input_is_video: bool = False,
                     preprocess_fn: Optional[Callable[["TensorBoard", Tensor, Tensor], Tuple[Tensor, Tensor]]] = None,
                     postprocess_fn: Optional[Callable[["TensorBoard", Tensor, Tensor],
                                                       Tuple[Tensor, Tensor]]] = None) -> None:
        """
        Writes images with predictions written on them to TensorBoard
        Args:
            epoch (int): Current epoch
            dataloader (BatchGenerator): The images will be sampled from this dataset
            draw_fn (callable): Function that takes in the tensor images, labels and predictions
                                and draws on the images before returning them.
            mode (str): Either "Train" or "Validation"
            input_is_video (bool): If the input data is a video.
            preprocess_fn (callable, optional): function called before inference.
                                                Gets data and labels as input, expects them as outputs
            postprocess_fn (callable, optional): function called after inference.
                                                 Gets data and predictions as input, expects them as outputs
        """
        clean_print("Writing images", end="\r")
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer

        batch = dataloader.next_batch()  # Get some data
        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

        data, labels = batch[0][:self.max_outputs].float(), batch[1][:self.max_outputs]
        if preprocess_fn:
            data, labels = preprocess_fn(self, data, labels)

        # Get some predictions
        predictions = self.model(data.to(self.device))
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        if postprocess_fn:
            data, predictions = postprocess_fn(self, data, predictions)

        if input_is_video:
            # Keep only one frame per video (middle one)
            frame_to_keep = data.shape[1] // 2
            imgs = data[:, frame_to_keep, :, :, :]
            if self.n_to_n:
                predictions = predictions[:, frame_to_keep]
                labels = labels[:, frame_to_keep]
        else:
            imgs = data

        # Write prediction on the images
        out_imgs = draw_fn(imgs, predictions, labels, **{"label_map": self.label_map})

        # Add them to TensorBoard
        for image_index, out_img in enumerate(out_imgs):
            tb_writer.add_image(f"{mode}/prediction_{image_index}", out_img, global_step=epoch, dataformats="HWC")

    def write_segmentation(self, epoch: int, dataloader: BatchGenerator, mode: str = "Train",
                           preprocess_fn: Optional[Callable[["TensorBoard", Tensor, Tensor],
                                                            Tuple[Tensor, Tensor]]] = None,
                           postprocess_fn: Optional[Callable[["TensorBoard", Tensor, Tensor],
                                                             Tuple[Tensor, Tensor]]] = None) -> None:
        """
        Writes images with predicted segmentation mask next to them to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
            preprocess_fn: function called before inference. Gets data and labels as input, expects them as outputs
            postprocess: function called after inference. Gets data and predictions as input, expects them as outputs
        """
        clean_print("Writing segmentation image", end="\r")
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer

        batch = dataloader.next_batch()  # Get some data
        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

        data, labels = batch[0][:self.max_outputs].float(), batch[1][:self.max_outputs]
        if preprocess_fn:
            data, labels = preprocess_fn(self, data, labels)

        # Get some predictions
        predictions = self.model(data.to(self.device))
        if postprocess_fn:
            data, predictions = postprocess_fn(self, data, predictions)

        out_imgs = draw_segmentation(data, predictions, labels, color_map=self.color_map)

        # Add them to TensorBoard
        for image_index, img in enumerate(out_imgs):
            tb_writer.add_image(f"{mode}/segmentation_output_{image_index}", img, global_step=epoch, dataformats="HWC")

    def write_videos(self, epoch: int, dataloader: BatchGenerator, mode: str = "Train",
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
        clean_print("Writing videos", end="\r")
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer

        batch = dataloader.next_batch()  # Get some data
        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

        videos, labels = batch["data"][:1].float(), batch["label"][:1]
        if preprocess_fn:
            videos, labels = preprocess_fn(self, videos, labels)

        # Get some predictions
        predictions = self.model(videos.to(self.device))
        videos, predictions = videos[:1], predictions[:1]  # Just in case (damn LSTM)
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        if postprocess_fn:
            videos, predictions = postprocess_fn(self, videos, predictions)

        # Write prediction on a video and add it to TensorBoard
        out_video = draw_pred_video(videos[0], predictions[0], labels[0], self.label_map, self.n_to_n)
        out_video = rearrange(out_video, 't h w c -> t c h w')
        out_video = np.expand_dims(out_video, 0)  # Re-add batch dimension

        tb_writer.add_video("Video", out_video, global_step=epoch, fps=16)

    def write_metrics(self, epoch: int, mode: str = "Train", write_defect_acc: bool = False) -> float:
        """
        Writes accuracy metrics in TensorBoard (for classification like tasks)

        Args:
            epoch (int): Current epoch
            mode (str): Either "Train" or "Validation"
            write_defect_acc (bool): If doing defect detection, this expect the "good" class to be 0

        Mostly comments changes, removed the LSTM special case since the postprocess_fn can handle it.

        Returns:
            float: Average accuracy
        """
        clean_print("Computing confusion matrix", end="\r")
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        self.metrics.compute_confusion_matrix(mode=mode)

        clean_print("Computing average accuracy", end="\r")
        avg_acc = self.metrics.get_avg_acc()
        tb_writer.add_scalar("Average Accuracy", avg_acc, epoch)

        clean_print("Computing per class accuracy", end="\r")
        per_class_acc = self.metrics.get_class_accuracy()
        for key, acc in enumerate(per_class_acc):
            tb_writer.add_scalar(f"Per Class Accuracy/{self.label_map[key]}", acc, epoch)

        if self.segmentation:
            clean_print("Computing per class IOU", end="\r")
            per_class_iou = self.metrics.get_class_iou()
            for key, iou in enumerate(per_class_iou):
                tb_writer.add_scalar(f"Per Class IOU/{self.label_map[key]}", iou, epoch)

        if write_defect_acc:
            acc = self.metrics.get_group_accuracy()
            tb_writer.add_scalar("Defect accuracy", acc, epoch)

        clean_print("Creating confusion matrix image", end="\r")
        confusion_matrix = self.metrics.get_confusion_matrix()
        tb_writer.add_image("Confusion Matrix", confusion_matrix, global_step=epoch, dataformats="HWC")

        return avg_acc

    def write_weights_grad(self, epoch):
        try:
            for tag, (weight, grad) in self.model.get_weight_and_grads().items():
                self.train_tb_writer.add_histogram(f"{tag}/weights", weight, epoch)
                self.train_tb_writer.add_histogram(f"{tag}/gradients", grad, epoch)
        except ModuleAttributeError:
            if not self.weights_warning_printed:
                print(f"Warning: The model {self.model.__class__.__name__}"
                      f" does not support recording weights and gradients.")
                self.weights_warning_printed = True

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
