"""
Structured logger with trace-ID support for the Policy & Incident Copilot.

Every log record carries:
  - A UUID trace_id  (generated once per request, propagated through all agents)
  - The agent/component name
  - Severity level
  - Timestamp (UTC, ISO-8601)

Rich is used for human-friendly console output; a plain JSON handler can be
swapped in for production log aggregation (ELK, CloudWatch, etc.).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

import config.settings as settings

_console = Console(stderr=True)


def _configure_root_logger() -> None:
    """Configure the root logger once at import time."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, show_path=False)],
    )


_configure_root_logger()


class CopilotLogger:
    """
    Thin wrapper around :class:`logging.Logger` that keeps a *trace_id*
    in every message so entire multi-agent runs can be correlated.

    Usage::

        logger = CopilotLogger("router")
        logger.info("Routing request", trace_id=trace_id)
    """

    def __init__(self, name: str, trace_id: Optional[str] = None) -> None:
        self._logger = logging.getLogger(f"copilot.{name}")
        self.trace_id: str = trace_id or self.new_trace_id()

    # ── public helpers ────────────────────────────────────────────────────────

    @staticmethod
    def new_trace_id() -> str:
        """Return a new UUID4 string to identify a single request lifecycle."""
        return str(uuid.uuid4())

    def bind_trace(self, trace_id: str) -> None:
        """Attach an existing trace_id (e.g. propagated from a parent agent)."""
        self.trace_id = trace_id

    # ── logging methods ───────────────────────────────────────────────────────

    def _fmt(self, msg: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        return f"[{ts}][{self.trace_id[:8]}] {msg}"

    def debug(self, msg: str, **extra) -> None:
        self._logger.debug(self._fmt(msg), extra=extra)

    def info(self, msg: str, **extra) -> None:
        self._logger.info(self._fmt(msg), extra=extra)

    def warning(self, msg: str, **extra) -> None:
        self._logger.warning(self._fmt(msg), extra=extra)

    def error(self, msg: str, **extra) -> None:
        self._logger.error(self._fmt(msg), extra=extra)

    def step(self, agent: str, action: str, detail: str = "") -> None:
        """Convenience log for agent step events shown prominently."""
        detail_str = f" — {detail}" if detail else ""
        self._logger.info(self._fmt(f"[STEP] {agent} :: {action}{detail_str}"))
