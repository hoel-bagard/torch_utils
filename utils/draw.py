from typing import Callable, Optional

import cv2
import numpy as np
import torch
from einops import rearrange


def denormalize_np(img: np.ndarray,
                   mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                   std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Undo the normalization process on an image.

    Args:
        img (np.ndarray): The normalized image.
        mean (tuple): The mean values that were used to normalize the image.
        std (tuple): The std values that were used to normalize the image.

    Returns:
        The denormalized image.
    """
    std = np.asarray(std)
    mean = np.asarray(mean)
    img = img * (255*std) + 255*mean
    return img.astype(np.uint8)


def denormalize_tensor(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo the normalization process on an image. The normalization values default to the ImageNet ones.

    Args:
        img (Tensor): The normalized image.
        mean (Tensor): The mean values that were used to normalize the image.
        std (Tensor): The std values that were used to normalize the image.

    Returns:
        The denormalized image.
    """
    mean = mean if mean is not None else torch.tensor((0.485, 0.456, 0.406))
    std = std if std is not None else torch.tensor((0.229, 0.224, 0.225))

    img = img * (255*std) + 255*mean
    return img


def draw_pred_img(imgs_tensor: torch.Tensor,
                  predictions_tensor: torch.Tensor,
                  labels_tensor: torch.Tensor,
                  label_map: dict[int, str],
                  size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Draws predictions and labels on the image to help with TensorBoard visualisation.

    Args:
        imgs_tensor (torch.Tensor): Raw imgs.
        predictions_tensor (torch.Tensor): Predictions of the network, after softmax but before taking argmax
        labels_tensor (torch.Tensor): Labels corresponding to the images
        label_map (dict): Dictionary linking class index to class name
        size (tuple, optional): If given, the images will be resized to this size

    Returns:
        np.ndarray: images with information written on them
    """
    imgs: np.ndarray = imgs_tensor.cpu().detach().numpy()
    labels: np.ndarray = labels_tensor.cpu().detach().numpy()
    predictions: np.ndarray = predictions_tensor.cpu().detach().numpy()

    imgs = rearrange(imgs, 'b c w h -> b w h c')  # imgs.transpose(0, 2, 3, 1)

    out_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        nb_to_keep = 3 if len(preds) > 3 else 2  # have at most 3 classes printed
        idx = np.argpartition(preds, -nb_to_keep)[-nb_to_keep:]  # Gets indices of top predictions
        idx = idx[np.argsort(preds[idx])][::-1]
        preds = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])

        img = np.asarray(img * 255.0, dtype=np.uint8)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = cv2.UMat(img)
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_img = img.get()
        # If opencv resizes a grayscale image, it removes the channel dimension
        if out_img.ndim == 2:
            out_img = np.expand_dims(out_img, -1)
        out_imgs.append(out_img)
    return np.asarray(out_imgs)


def draw_pred_video(video_tensor: torch.Tensor,
                    label_tensor: torch.Tensor,
                    prediction_tensor: torch.Tensor,
                    label_map: dict[int, str],
                    n_to_n: bool = False,
                    size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Draws predictions and labels on the video to help with TensorBoard visualisation.

    Args:
        video (torch.Tensor): Raw video.
        label (torch.Tensor): Label corresponding to the video
        prediction (torch.Tensor): Prediction of the network, after softmax but before taking argmax
        label_map (dict): Dictionary linking class index to class name
        n_to_n (bool): True if using one label for each element of the sequence
        size (tuple, optional): If given, the images will be resized to this size

    Returns:
        np.ndarray: Videos with information written on them
    """
    video: np.ndarray = video_tensor.cpu().detach().numpy()
    labels: np.ndarray = label_tensor.cpu().detach().numpy()
    preds: np.ndarray = prediction_tensor.cpu().detach().numpy()
    if not n_to_n:
        labels = np.broadcast_to(labels, video.shape[0])
        preds = np.broadcast_to(preds, (video.shape[0], preds.shape[0]))

    video = rearrange(video, 'b c h w -> b h w c')

    new_video_list = []
    for img, preds, label in zip(video, preds, labels):
        # If there are too many classes, just print the top 3 ones
        if len(preds) > 5:
            # Gets indices of top 3 pred
            idx = np.argpartition(preds, -3)[-3:]
            idx = idx[np.argsort(preds[idx])][::-1]
            preds_text = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])
        else:
            preds_text = str([round(float(conf), 2) for conf in preds]) + f"  ==> {np.argmax(preds)}"

        img = np.asarray(img * 255.0, dtype=np.uint8)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = cv2.UMat(img)
        img = cv2.putText(img, preds_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_video_list.append(img.get())

    new_video = np.asarray(new_video_list)

    # Keep a channel dimension if in gray scale mode
    if new_video.ndim == 3:
        new_video = np.expand_dims(new_video, -1)

    return new_video


def draw_segmentation(input_imgs: torch.Tensor,
                      one_hot_masks_preds: torch.Tensor,
                      one_hot_masks_labels: torch.Tensor,
                      color_map: list[tuple[int, int, int]],
                      denormalize_img_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                      size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Recreate the segmentation masks from their one hot representations, and place them next to the original image.

    Args:
        input_imgs (torch.Tensor): Images that were fed to the network.
        one_hot_masks_labels (torch.Tensor): One hot representation of the label segmentation masks.
        one_hot_masks_preds (torch.Tensor): One hot representation of the prediction segmentation masks.
        color_map (list): List linking class index to its color
        denormalize_img_fn (callable): Function to reverse the normalization process.
        size (int, optional): If given, the images will be resized to this size

    Returns:
        np.ndarray: RGB segmentation masks and original image (in one image)
    """
    imgs = rearrange(input_imgs, "b c w h -> b w h c").cpu().detach().numpy()
    if denormalize_img_fn is None:
        imgs = np.asarray(imgs * 255.0, dtype=np.uint8)
    else:
        imgs = denormalize_img_fn(imgs).astype(np.uint8)

    one_hot_masks_preds = rearrange(one_hot_masks_preds, "b c w h -> b w h c")
    masks_preds: np.ndarray = torch.argmax(one_hot_masks_preds, dim=-1).cpu().detach().numpy()
    one_hot_masks_labels = rearrange(one_hot_masks_labels, "b c w h -> b w h c")
    masks_labels: np.ndarray = torch.argmax(one_hot_masks_labels, dim=-1).cpu().detach().numpy()

    width, height, _ = imgs[0].shape  # All images are expected to have the same shape

    # Create a blank image with some text to explain what things are
    text_img = np.full((width, height, 3), 255, dtype=np.uint8)
    text_img = cv2.putText(text_img, "Top left: input image.", (20, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Top right: label mask", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    text_img = cv2.putText(text_img, "Bottom left: predicted mask", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    out_imgs = []
    for img, pred_mask, label_mask in zip(imgs, masks_preds, masks_labels):
        # Recreate the segmentation mask from its one hot representation
        pred_mask_rgb = np.asarray(color_map[pred_mask], dtype=np.uint8)
        label_mask_rgb = np.asarray(color_map[label_mask], dtype=np.uint8)

        out_img_top = cv2.hconcat((img, label_mask_rgb))
        out_img_bot = cv2.hconcat((pred_mask_rgb, text_img))
        out_img = cv2.vconcat((out_img_top, out_img_bot))
        if size:
            out_img = cv2.resize(out_img, size, interpolation=cv2.INTER_AREA)
        out_imgs.append(out_img)

    return np.asarray(out_imgs)
