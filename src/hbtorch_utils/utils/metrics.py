from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import numpy.typing as npt
import torch

from .batch_generator import BatchGenerator


class Metrics(ABC):
    def __init__(
        self: Self,
        model: torch.nn.Module,
        train_dataloader: BatchGenerator,
        val_dataloader: BatchGenerator,
        max_batches: int | None = 10,
    ) -> None:
        """Class computing usefull metrics.

        Args:
            model: The PyTorch model being trained.
            train_dataloader: DataLoader containing train data.
            val_dataloader: DataLoader containing validation data.
            max_batches: If not None, maximum number of batches used when computing the metrics.
        """
        super().__init__()
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_batches = max_batches

    @abstractmethod
    def get_metrics(
        self: Self,
        mode: str = "Train",
        **kwargs: dict[str, object],
    ) -> dict[str, dict[str, dict[str, float] | dict[str, npt.NDArray[np.uint8]]]]:
        """Return all the metrics the class can provide.

        This Method calls the class's other method to aggregate all the metrics into a dictionnary.
        The dictionnary contains a 2 keys: "scalars" and "imgs".
        For those two keys, the associated value is a dictionnary where the key is the metrics' name,
        and the value its value.

        Examples:
            Return example:

            {
             "scalars": {
                        "Per Class Accuracy/cat": 0.75,
                        "Per Class Accuracy/dog": 0.63
                       }
             "imgs": {"Confusion Matrix": np.asarray([[255, 0], [0, 255]])}
            }

        Args:
            mode: Either "Train" or "Validation", determines which dataloader to use.
            kwargs: kwargs.

        Returns:
            dict: The dictionnary containing the metrics the class provides
        """
