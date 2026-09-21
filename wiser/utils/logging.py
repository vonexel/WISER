from __future__ import annotations

import logging
from logging import Logger
from typing import Optional
from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str | int = "INFO", *, use_rich: bool = True) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handlers: list[logging.Handler] = []
    if use_rich:
        try:
            handlers.append(RichHandler(rich_tracebacks=True, markup=False, show_path=False, show_time=True))
        except ImportError:
            handlers.append(logging.StreamHandler())
    else:
        handlers.append(logging.StreamHandler())
    fmt = "%(message)s" if use_rich else "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> Logger:
    return logging.getLogger(name if name is not None else "wiser")