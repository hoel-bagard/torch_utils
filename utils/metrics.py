import itertools
from typing import (
    Dict,
    List,
    Optional
)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .batch_generator import BatchGenerator


class Metrics:
    def __init__(self, model: nn.Module, train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator, label_map: Dict[int, str], max_batches: Optional[int] = 10,
                 n_to_n: bool = False, segmentation: bool = False):
        """
        Class computing usefull metrics for classification tasks
        Args:
            model: The PyTorch model being trained
            train_dataloader: DataLoader containing train data
            val_dataloader: DataLoader containing validation data
            label_map: Dictionary linking class index to class name
            max_batches: If not None, then the metrics will be computed using at most this number of batches
            n_to_n: Used when input is a sequence. True if using one label for each element of the sequence
            segmentation: Used when doing segmentation.
        """
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.label_map = label_map
        self.n_to_n = n_to_n
        self.segmentation = segmentation
        self.nb_output_classes = len(label_map)
        self.max_batches = max_batches

    def compute_confusion_matrix(self, mode: str = "Train"):
        """
        Computes the confusion matrix. This function has to be called before using the get functions.
        Args:
            mode: Either "Train" or "Validation"
        """
        self.cm = np.zeros((self.nb_output_classes, self.nb_output_classes))
        for step, batch in enumerate(self.train_dataloader if mode == "Train" else self.val_dataloader, start=1):
            data_batch, labels_batch = batch[0].float(), batch[1]
            predictions_batch = self.model(data_batch.to(self.device))

            if not self.segmentation:
                predictions_batch = torch.argmax(predictions_batch, dim=-1).int().cpu().detach().numpy()
                for (label, pred) in zip(labels_batch, predictions_batch):
                    if self.n_to_n:
                        for (label_frame, pred_frame) in zip(label, pred):
                            self.cm[label_frame, pred_frame] += 1
                    else:
                        self.cm[label, pred] += 1
            else:
                predictions_batch = torch.argmax(predictions_batch, dim=1).int().cpu().detach().numpy()
                labels_batch = torch.argmax(labels_batch, dim=1).cpu().detach().numpy()
                for (label_pixel, pred_pixel) in zip(labels_batch.flatten(), predictions_batch.flatten()):
                    self.cm[label_pixel, pred_pixel] += 1
            if self.max_batches and step >= self.max_batches:
                break

    def get_avg_acc(self) -> float:
        """
        Uses the confusion matrix to return the average accuracy of the model
        Returns:
            avg_acc: Average accuracy
        """
        avg_acc = np.sum([self.cm[i, i] for i in range(len(self.cm))]) / np.sum(self.cm)
        return avg_acc

    def get_class_accuracy(self) -> List[float]:
        """
        Uses the confusion matrix to return the average accuracy of the model
        Returns:
            per_class_acc: An array containing the accuracy for each class
        """
        per_class_acc = [self.cm[i, i] / max(1, np.sum(self.cm[i])) for i in range(len(self.cm))]
        return per_class_acc

    def get_group_accuracy(self, classes: Optional[list[int]] = None) -> float:
        """
        Uses the confusion matrix to return the accuracy for the given classes vs the all the others.
        Args:
            classes: List with the classes that should be grouped together
        Returns:
            acc: accuracy for the given group
        """
        if not classes:
            classes = np.arange(1, self.nb_output_classes)
        correct = 0
        total = 0
        for cls in classes:
            correct += self.cm[cls, cls]
            total += np.sum(self.cm[cls])
        acc = correct / max(1, total)
        return acc

    def get_class_iou(self) -> float:
        """
        Uses the confusion matrix to return the iou for each class
        Returns:
            avg_acc: Average accuracy
        """
        intersections = [self.cm[i, i] for i in range(len(self.cm))]
        unions = [np.sum(self.cm[i, :]) + np.sum(self.cm[:, i]) - self.cm[i, i] for i in range(self.nb_output_classes)]
        per_class_iou = [intersections[i] / unions[i] for i in range(self.nb_output_classes)]
        return per_class_iou

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Returns an image containing the plotted confusion matrix.
        Taken from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12

        Returns:
            img: Image of the confusion matrix.
        """
        # Normalize the confusion matrix.
        cm = np.around(self.cm.astype("float") / self.cm.sum(axis=1)[:, np.newaxis], decimals=2)
        class_names = self.label_map.values()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label", labelpad=-5)
        plt.xlabel("Predicted label")
        fig.canvas.draw()

        # Convert matplotlib plot to normal image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close figure explicitly to avoid memory leak

        return img
