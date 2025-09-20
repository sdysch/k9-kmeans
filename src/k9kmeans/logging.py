import logging
import sys


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # prevent adding multiple handlers if this is called multiple times
    if not logger.hasHandlers():
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


__all__ = ['setup_logger']
