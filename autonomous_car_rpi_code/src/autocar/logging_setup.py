from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import LoggingConfig

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure(cfg: LoggingConfig) -> None:
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg.file:
        handlers.append(logging.FileHandler(cfg.file))

    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else "autocar")
