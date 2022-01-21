from dataclasses import asdict, fields
from shutil import get_terminal_size
from typing import (
    Any,
    Dict,
    Optional,
    Tuple
)

import cv2
import numpy as np


def clean_print(msg: str, fallback: Optional[Tuple[int, int]] = (156, 38), end='\n'):
    r"""Function that prints the given string to the console and erases any previous print made on the same line.

    Args:
        msg (str): String to print to the console
        fallback (tuple, optional): Size of the terminal to use if it cannot be determined by shutil
                                    (if using windows for example)
        end (str): What to add at the end of the print. Usually '\n' (new line), or '\r' (back to the start of the line)
    """
    print(msg + ' ' * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)


def get_config_as_dict(config) -> Dict[str, Any]:
    """Takes a config object and returns it as dictionnary.

    Args:
        config (type): A config is just a python class with some class variables.

    Returns:
        config_dict: A dictionnary mapping a (lowercase) class variable name to its value.
    """
    config_attribute_dict = vars(config)

    config_dict = {}
    for key, value in config_attribute_dict.items():
        if not key.startswith('__') and key[0].isupper():
            config_dict[key.lower()] = value

    return config_dict


def get_dataclass_as_dict(config, lower_case: bool = True) -> Dict[str, Any]:
    """Takes a dataclass instance and returns it as a dictionnary.

    Args:
        config: The dataclass instance.
        lower_case (bool): If true then the field names will all be lowercase.

    Returns:
        A dictionnary where the keys are the field names and the values are the config values.
    """
    if lower_case:
        return dict((field.name.lower(), getattr(config, field.name)) for field in fields(config))
    else:
        return asdict(config)


def show_img(img: np.ndarray, window_name: str = "Image"):
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
