"""Logger."""

import logging
import sys


def get_logger(name: str, path: str) -> logging.Logger:

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a file handler to log to the specified log file
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)

    # Create a stream handler to log to stdout (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Define a formatter for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
