"""Logging module for the DAC framework.

Provides a unified logging interface that works in both headless (library)
and GUI contexts. In headless mode, log messages go to the standard
Python logging system. In GUI mode, they can be routed to the log widget.

Usage:
    import logging
    from dac.core.logging import logger

    logger.info("Loading data...")
    logger.warning("Missing optional dependency")
    logger.error("Failed to process action")
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _create_default_handler():
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler


logger = logging.getLogger("dac")
logger.setLevel(logging.WARNING)

_default_handler = _create_default_handler()
logger.addHandler(_default_handler)

dac_logger = logger


def configure_logging(level: int | str = logging.INFO, handler: logging.Handler = None):
    """Configure the DAC logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        handler: Optional custom handler. If None, the default stderr handler
                 is used.
    """
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.setLevel(level)
    if handler is not None:
        logger.addHandler(handler)
    else:
        logger.addHandler(_create_default_handler())


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name to append to 'dac' prefix.

    Returns:
        A logger instance.
    """
    if name:
        return logging.getLogger(f"dac.{name}")
    return logger
