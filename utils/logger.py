"""
Project logger with File + Stream handlers.

- File: writes to config.LOG_FILE (for long-term inspection and Streamlit tail)
- Stream: writes to stderr (so subprocess output shows something even when stdout is empty)
"""

import logging
import sys
from pathlib import Path
from config import LOG_FILE, LOG_DIR

# Ensure directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False

def get_logger(name: str = __name__) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)

    if not _configured:
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(fh)

        # Stream handler (stderr)
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(sh)

        _configured = True

    return logger

# Backward-compatible alias
logger = get_logger(__name__)
