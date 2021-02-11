from shutil import get_terminal_size
from typing import (
    Optional,
    Tuple
)


def clean_print(msg: str, fallback: Optional[Tuple[int, int]] = (156, 38), end='\n'):
    """
    Function that prints the given string to the console and erases any previous print made on the same line
    Args:
        msg: String to print to the console
        fallback: Size of the terminal to use if it cannot be determined by shutil (if using windows for example)
    """
    print(msg + ' ' * (get_terminal_size(fallback=fallback).columns - len(msg)), end=end, flush=True)
