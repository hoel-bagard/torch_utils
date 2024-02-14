from dataclasses import asdict, fields
from shutil import get_terminal_size
from typing import Any


def clean_print(msg: str, fallback: tuple[int, int] = (156, 38), end: str = "\n"):
    r"""Function that prints the given string to the console and erases any previous print made on the same line.

    Args:
        msg (str): String to print to the console
        fallback (tuple, optional): Size of the terminal to use if it cannot be determined by shutil
                                    (if using windows for example)
        end (str): What to add at the end of the print. Usually '\n' (new line), or '\r' (back to the start of the line)
    """
    print(msg + " " * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)


def get_config_as_dict(config: object) -> dict[str, object]:
    """Take a config object (class with class attributes) and return it as dictionary.

    Args:
        config: A config is just a python class with some class variables.

    Returns:
        config_dict: A dictionary mapping a (lowercase) class variable name to its value.
    """
    config_attribute_dict = vars(config)

    config_dict: dict[str, object] = {}
    for key, value in config_attribute_dict.items():
        if not key.startswith("__") and key[0].isupper():
            config_dict[key.lower()] = value

    return config_dict


def get_dataclass_as_dict(config: object, lower_case: bool = True) -> dict[str, Any]:
    """Takes a dataclass instance and returns it as a dictionary.

    Args:
        config: The dataclass instance.
        lower_case: If true then the field names will all be lowercase.

    Returns:
        A dictionary where the keys are the field names and the values are the config values.
    """
    if lower_case:
        return {field.name.lower(): getattr(config, field.name) for field in fields(config)}

    return asdict(config)
