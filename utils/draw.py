from typing import Optional

import cv2
from einops import rearrange
import numpy as np
import torch


def draw_pred_img(imgs: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor,
                  label_map: dict[int, str], size: Optional[tuple[int, int]] = None, ) -> np.ndarray:
    """
    Draw predictions and labels on the image to help with TensorBoard visualisation.
    Args:
        imgs: Raw imgs.
        predictions: Predictions of the network, after softmax but before taking argmax
        labels: Labels corresponding to the images
        label_map: Dictionary linking class index to class name
        size: If given, the images will be resized to this size
    Returns: images with information written on them
    """
    imgs: np.ndarray = imgs.cpu().detach().numpy()
    labels: np.ndarray = labels.cpu().detach().numpy()
    predictions: np.ndarray = predictions.cpu().detach().numpy()

    imgs = rearrange(imgs, 'b c w h -> b w h c')  # imgs.transpose(0, 2, 3, 1)

    out_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        # Just print the top 3 classes
        # Gets indices of top 3 pred
        nb_to_keep = 3 if len(preds) > 3 else 2
        idx = np.argpartition(preds, -nb_to_keep)[-nb_to_keep:]
        idx = idx[np.argsort(preds[idx])][::-1]
        preds = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])

        img = np.asarray(img * 255.0, dtype=np.uint8)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = cv2.UMat(img)
        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        out_img = img.get()
        # If opencv resizes a grayscale image, it removes the channel dimension
        if out_img.ndim == 2:
            out_img = np.expand_dims(out_img, -1)
        out_imgs.append(out_img)
    return np.asarray(out_imgs)


def draw_pred_video(video: torch.Tensor, prediction: torch.Tensor, label: torch.Tensor,
                    label_map: dict[int, str], n_to_n: bool = False,
                    size: Optional[tuple[int, int]] = None, ) -> np.ndarray:
    """
    Draw predictions and labels on the video to help with TensorBoard visualisation.
    Args:
        video: Raw video.
        prediction: Prediction of the network, after softmax but before taking argmax
        label: Label corresponding to the video
        label_map: Dictionary linking class index to class name
        n_to_n: True if using one label for each element of the sequence
        size: If given, the images will be resized to this size
    Returns: images with information written on them
    """
    video: np.ndarray = video.cpu().detach().numpy()
    labels: np.ndarray = label.cpu().detach().numpy()
    preds: np.ndarray = prediction.cpu().detach().numpy()
    if not n_to_n:
        labels = np.broadcast_to(labels, video.shape[0])
        preds = np.broadcast_to(preds, (video.shape[0], preds.shape[0]))

    video = rearrange(video, 'b c h w -> b h w c')

    new_video = []
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
        new_video.append(img.get())

    new_video = np.asarray(new_video)

    # Keep a channel dimension if in gray scale mode
    if new_video.ndim == 3:
        new_video = np.expand_dims(new_video, -1)

    return new_video


def draw_segmentation_map(one_hot_masks: torch.Tensor, color_map: dict[int, str],
                          size: Optional[tuple[int, int]] = None, ) -> np.ndarray:
    """
    Recreate the segmentation mask from its one hot representation
    Args:
        one_hot_masks: One hot representation of the segmentation masks.
        color_map: Dictionary linking class index to its color
        size: If given, the images will be resized to this size
    Returns: RGB segmentation masks
    """
    one_hot_masks = rearrange(one_hot_masks, "b c w h -> b w h c")
    masks: np.ndarray = torch.argmax(one_hot_masks, dim=-1).cpu().detach().numpy()
    width, height, _ = one_hot_masks[0].shape  # All images are expected to have the same shape

    out_imgs = []
    for mask in masks:
        img = np.empty((width, height, 3))
        # TODO: optimize this later
        for i in range(width):
            for j in range(height):
                img[i, j] = color_map[mask[i, j]]
        out_imgs.append(img)

        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return np.asarray(out_imgs)
