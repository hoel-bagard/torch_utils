from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from .batch_generator import BatchGenerator


class Metrics(ABC):
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 max_batches: Optional[int] = 10):
        """Class computing usefull metrics.

        Args:
            model (torch.nn.Module): The PyTorch model being trained.
            train_dataloader (BatchGenerator): DataLoader containing train data.
            val_dataloader (BatchGenerator): DataLoader containing validation data.
            max_batches (int, optional): If not None, maximum number of batches used when computing the metrics.
        """
        super().__init__()
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_batches = max_batches

    @abstractmethod
    def get_metrics(self, mode: str = "Train", **kwargs) -> dict[str, dict[str, Any]]:
        """Method returning all the metrics the class can provide.

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

        Returns:
            dict: The dictionnary containing the metrics the class provides
        """
