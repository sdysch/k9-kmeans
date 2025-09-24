import logging
import sys


import logging
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a logger with a single StreamHandler."""

    if fmt is None:
        fmt = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M:%S'

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    return logger
