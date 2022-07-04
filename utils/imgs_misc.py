from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange


def show_img(img: npt.NDArray[np.uint8], window_name: str = "Image") -> None:
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small^^)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def denormalize_np(img: npt.NDArray[np.uint8],
                   mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> npt.NDArray[np.uint8]:
    """Undo the normalization process on an image.

    Args:
        img (np.ndarray): The normalized image.
        mean (tuple): The mean values that were used to normalize the image.
        std (tuple): The std values that were used to normalize the image.

    Returns:
        The denormalized image.
    """
    std_array = np.asarray(std)
    mean_array = np.asarray(mean)
    img = (img * (255*std_array) + 255*mean_array).astype(np.uint8)
    return img


def denormalize_tensor(imgs: torch.Tensor,
                       mean: Optional[torch.Tensor] = None,
                       std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Undo the normalization process on a batch of images. The normalization values default to the ImageNet ones.

    Args:
        imgs (Tensor): The normalized images.
        mean (Tensor): The mean values that were used to normalize the image.
        std (Tensor): The std values that were used to normalize the image.

    Returns:
        The denormalized image.
    """
    mean = mean if mean is not None else torch.tensor((0.485, 0.456, 0.406))
    std = std if std is not None else torch.tensor((0.229, 0.224, 0.225))

    imgs = rearrange(imgs, 'b c w h -> b w h c')
    imgs = imgs * (255*std) + 255*mean
    imgs = rearrange(imgs, 'b w h c -> b c w h')
    return imgs
