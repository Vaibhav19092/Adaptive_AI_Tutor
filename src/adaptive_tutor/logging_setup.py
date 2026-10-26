"""Configure Python's standard logging once for the whole app."""

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Send log messages to the terminal. Safe to call more than once."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # httpx logs every HTTP request at INFO level; that's too noisy for us.
    logging.getLogger("httpx").setLevel(logging.WARNING)