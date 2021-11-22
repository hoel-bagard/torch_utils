from shutil import get_terminal_size
from typing import (
    Any,
    Dict,
    Optional,
    Tuple
)


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
