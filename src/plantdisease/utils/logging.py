from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "plantdisease"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def add_file_handler(log_file: str | Path) -> logging.FileHandler:
    """Also write logs to ``log_file``. Returns the handler so callers can remove it later."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(_FORMAT))
    get_logger().addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler | None) -> None:
    if handler is not None:
        get_logger().removeHandler(handler)
        handler.close()
