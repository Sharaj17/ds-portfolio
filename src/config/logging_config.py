# src/config/logging.py
import logging
import sys

def init_logging(level: str = "INFO") -> logging.Logger:
    """
    Minimal, production-friendly logging:
      - streams to stdout (so Docker / cloud captures it)
      - includes level, timestamp, logger name, message
    """
    root = logging.getLogger()
    if not root.handlers:  # avoid duplicate handlers in reload
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logging.getLogger("iris_api")
