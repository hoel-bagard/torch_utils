from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange


def draw_pred_img(imgs_tensor: torch.Tensor,
                  predictions_tensor: torch.Tensor,
                  labels_tensor: torch.Tensor,
                  label_map: dict[int, str],
                  size: Optional[tuple[int, int]] = None) -> npt.NDArray[np.uint8]:
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
    imgs: npt.NDArray[np.uint8] = imgs_tensor.cpu().detach().numpy()
    labels: npt.NDArray[np.int64] = labels_tensor.cpu().detach().numpy()
    predictions: npt.NDArray[np.int64] = predictions_tensor.cpu().detach().numpy()

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
                    size: Optional[tuple[int, int]] = None) -> npt.NDArray[np.uint8]:
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
    video: npt.NDArray[np.uint8] = video_tensor.cpu().detach().numpy()
    labels: npt.NDArray[np.int64] = label_tensor.cpu().detach().numpy()
    preds: npt.NDArray[np.int64] = prediction_tensor.cpu().detach().numpy()

    if not n_to_n:
        labels = np.broadcast_to(labels, video.shape[0])
        preds = np.broadcast_to(preds, (video.shape[0], preds.shape[0]))

    video = rearrange(video, 'b c h w -> b h w c')

    new_video_list = []
    for img, pred, label in zip(video, preds, labels):
        # If there are too many classes, just print the top 3 ones
        if len(pred) > 5:
            # Gets indices of top 3 pred
            idx = np.argpartition(pred, -3)[-3:]
            idx = idx[np.argsort(pred[idx])][::-1]
            preds_text = str([label_map[i] + f":  {round(float(pred[i]), 2)}" for i in idx])
        else:
            preds_text = str([round(float(conf), 2) for conf in pred]) + f"  ==> {np.argmax(pred)}"

        img = np.asarray(img * 255.0, dtype=np.uint8)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = cv2.UMat(img)
        img = cv2.putText(img, preds_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_video_list.append(img.get())

    new_video = np.asarray(new_video_list, dtype=np.uint8)

    # Keep a channel dimension if in gray scale mode
    if new_video.ndim == 3:
        new_video = np.expand_dims(new_video, -1)

    return new_video
