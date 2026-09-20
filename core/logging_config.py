"""
Centralized logging configuration with per-run log files.

Each run gets its own log file under logs/<run_id>.log so errors
can be traced back to the exact run that produced them.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"
_INITIALIZED = False
_root_logger: Optional[logging.Logger] = None


def _ensure_logs_dir():
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_level: str = "INFO"):
    """Configure root logger with console + rotating file handler.

    Called once at startup. Subsequent calls are no-ops.
    """
    global _INITIALIZED, _root_logger
    if _INITIALIZED:
        return

    _ensure_logs_dir()
    _root_logger = logging.getLogger()
    _root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates from basicConfig
    _root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console.setFormatter(formatter)
    _root_logger.addHandler(console)

    # Global rotating file handler (captures everything across all runs)
    _ensure_logs_dir()
    global_handler = logging.handlers.RotatingFileHandler(
        _LOGS_DIR / "scholargraph.log",
        maxBytes=10_000_000,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    global_handler.setLevel(logging.DEBUG)
    global_handler.setFormatter(formatter)
    _root_logger.addHandler(global_handler)

    _INITIALIZED = True


def attach_run_log(run_id: str) -> logging.Logger:
    """Attach a per-run file handler to the root logger.

    Returns the root logger for convenience. The handler is removed when
    ``detach_run_log`` is called (or the process exits).
    """
    global _root_logger
    if _root_logger is None:
        _root_logger = logging.getLogger()

    _ensure_logs_dir()
    run_log_path = _LOGS_DIR / f"{run_id}.log"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.handlers.RotatingFileHandler(
        run_log_path,
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    handler.set_name(f"run_{run_id}")
    _root_logger.addHandler(handler)
    return _root_logger


def detach_run_log(run_id: str):
    """Remove the per-run file handler for *run_id*."""
    global _root_logger
    if _root_logger is None:
        return
    target_name = f"run_{run_id}"
    _root_logger.handlers = [
        h for h in _root_logger.handlers
        if getattr(h, "name", None) != target_name
    ]
