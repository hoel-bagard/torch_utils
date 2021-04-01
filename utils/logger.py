from pathlib import Path
from logging import handlers, StreamHandler
import logging
import os
import sys

# Mostly taken from https://gitlab.com/corentin-pro/torch_utils/-/tree/master/utils


class ConsoleColor(object):
    """ Simple shortcut to use colors in the console """
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    RED = '\033[91m'
    ENDCOLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ColoredFormatter(logging.Formatter):
    """ Formatter adding colors to levelname """
    def format(self, record):
        levelno = record.levelno
        if logging.ERROR == levelno:
            levelname_color = ConsoleColor.RED + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.WARNING == levelno:
            levelname_color = ConsoleColor.ORANGE + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.INFO == levelno:
            levelname_color = ConsoleColor.GREEN + record.levelname + ConsoleColor.ENDCOLOR
        elif logging.DEBUG == levelno:
            levelname_color = ConsoleColor.BLUE + record.levelname + ConsoleColor.ENDCOLOR
        else:
            levelname_color = record.levelname
        record.levelname = levelname_color
        return logging.Formatter.format(self, record)


class DummyLogger():
    """ Dummy logger that just outputs the string to stdout """
    def debug(self, string, *args):
        print(string, *args)

    def info(self, string, *args):
        print(string, *args)

    def warn(self, string, *args):
        print(string, *args)

    def warning(self, string, *args):
        print(string, *args)

    def error(self, string, *args):
        print(string, *args)

    def critical(self, string, *args):
        print(string, *args)

    def fatal(self, string, *args):
        print(string, *args)


def create_logger(name: str, log_dir: Path, stdout=True):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Add a (rotating) file handler to the logging system
    file_log_handler = handlers.RotatingFileHandler(log_dir / (name + ".log"), maxBytes=500000, backupCount=2)
    file_log_handler.setFormatter(log_formatter)
    file_log_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_log_handler)

    if stdout:
        # Add an handler to the logging system (default has none) : outputing in stdout
        terminal_log_handler = StreamHandler(sys.stdout)
        terminal_log_handler.setLevel(logging.DEBUG)
        if os.name != 'nt':
            # Fancy color for non windows console
            colored_log_formatter = ColoredFormatter("%(levelname)s - %(message)s")
            terminal_log_handler.setFormatter(colored_log_formatter)
        else:
            terminal_log_handler.setFormatter(log_formatter)
        logger.addHandler(terminal_log_handler)

    return logger
