from shutil import get_terminal_size
from typing import (
    Optional,
    Tuple,
    Dict,
    Any
)


def clean_print(msg: str, fallback: Optional[Tuple[int, int]] = (156, 38), end='\n'):
    """ Function that prints the given string to the console and erases any previous print made on the same line

    Args:
        msg (str): String to print to the console
        fallback (tuple, optional): Size of the terminal to use if it cannot be determined by shutil
                                    (if using windows for example)
    """
    print(msg + ' ' * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)


def get_config_as_dict(config) -> Dict[str, Any]:
    """ Takes a config object and returns it as dictionnary

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
