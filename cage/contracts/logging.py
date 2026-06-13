"""Structured logging configuration for the Cage framework.

Three-layer output:
  1. Console — human-readable logs
  2. Run log  — JSONL to .cage_runs/{agent_label}/run-{timestamp}/{mode}/.cage.runlog
  3. Debug log — optional verbose file at
     .cage_runs/{agent_label}/run-{timestamp}/{mode}/.cage.debuglog

Context binding via structlog.contextvars:
  bind_run_context(run_id=..., agent_label=...)
  bind_trial_context(trial_id=..., trial_index=..., sample_id=...)
  clear_trial_context()
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
    unbind_contextvars,
)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

@dataclass
class LoggingConfig:
    """Logging configuration, sourced from project.yml + CLI flags."""

    console_level: str = "INFO"
    file_level: str = "DEBUG"
    debug_file_enabled: bool = False
    terminal_ui: bool = True
    inspect_mode: str = "on"
    console_colors: bool = True  # auto-disabled when stderr is not a TTY
    run_log_path: Path | None = None  # set by orchestrator per agent
    debug_log_path: Path | None = None  # set by orchestrator per agent


# ------------------------------------------------------------------ #
# Context binding helpers
# ------------------------------------------------------------------ #

# Keys managed at the trial level — cleared between trials
_TRIAL_CONTEXT_KEYS = ("trial_id", "trial_index", "sample_id")


def bind_run_context(**kwargs: Any) -> None:
    """Bind run-level context (run_id, agent_label) to all log messages."""
    bind_contextvars(**kwargs)


def bind_trial_context(**kwargs: Any) -> None:
    """Bind trial-level context (trial_id, trial_index, sample_id) to log messages."""
    bind_contextvars(**kwargs)


def clear_trial_context() -> None:
    """Remove trial-level context between trials. Keeps run-level context."""
    unbind_contextvars(*_TRIAL_CONTEXT_KEYS)


def clear_all_context() -> None:
    """Clear all context variables (call at experiment end)."""
    clear_contextvars()


# ------------------------------------------------------------------ #
# Processors
# ------------------------------------------------------------------ #

def _add_timestamp(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add ISO-8601 timestamp to every log entry."""
    import datetime

    event_dict["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return event_dict


def _rename_event_key(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Rename structlog's 'event' key to 'message' for JSONL consistency."""
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def _filter_cage_noise(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Drop noisy third-party log messages at DEBUG level."""
    logger_name = event_dict.get("logger_name", "")
    noisy_prefixes = ("httpx", "docker", "urllib3", "httpcore")
    if any(logger_name.startswith(p) for p in noisy_prefixes):
        if event_dict.get("level") == "debug":
            raise structlog.DropEvent
    return event_dict


def _build_shared_processors() -> list[Any]:
    """Build the shared processor chain used by all handlers."""
    return [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_timestamp,
        _filter_cage_noise,
    ]


# ------------------------------------------------------------------ #
# Setup
# ------------------------------------------------------------------ #

_setup_done = False


class ProgressLine(Protocol):
    """A single-line progress display that console logs must not clobber.

    The CLI progress reporter registers itself as the active progress line via
    :func:`register_active_progress_line`; the console handler then clears the
    line before each log record and redraws it after. Defined here (not in the
    CLI) so the console handler cooperates with the line without importing up
    into ``cage.cli``.
    """

    def clear_for_external_write(self) -> None: ...

    def redraw_after_external_write(self) -> None: ...


_active_progress_line: "ProgressLine | None" = None


def register_active_progress_line(line: ProgressLine) -> None:
    """Register the progress line console logs must yield the terminal to."""
    global _active_progress_line  # noqa: PLW0603
    _active_progress_line = line


def unregister_active_progress_line(line: ProgressLine) -> None:
    """Clear the active progress line if ``line`` is still the registered one."""
    global _active_progress_line  # noqa: PLW0603
    if _active_progress_line is line:
        _active_progress_line = None


def mark_run_stopping() -> None:
    """Flag the live progress line as interrupted (renders "⏹ stopping").

    Called from the SIGINT handler so the bar stops yanking toward a misleading
    100% as interrupted trials resolve to failed. Best-effort and duck-typed —
    only the run progress reporter honours ``_stopping``; anything else ignores
    it. Safe to call when no progress line is active.
    """
    line = _active_progress_line
    if line is None:
        return
    try:
        line._stopping = True  # type: ignore[attr-defined]
        line.redraw_after_external_write()
    except Exception:  # noqa: BLE001
        pass


def clear_active_progress_line() -> None:
    """Erase the live progress line so the next stdout write starts clean."""
    line = _active_progress_line
    if line is None:
        return
    try:
        line.clear_for_external_write()
    except Exception:  # noqa: BLE001
        pass


class _ProgressAwareConsoleHandler(logging.StreamHandler):
    """Console handler that keeps Cage's single-line progress readable."""

    def emit(self, record: logging.LogRecord) -> None:
        line = _active_progress_line
        if line is not None:
            line.clear_for_external_write()
        try:
            super().emit(record)
        finally:
            if line is not None:
                line.redraw_after_external_write()


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure structured logging for the Cage framework.

    Must be called once at startup (from cli.py). Creates a
    ProcessorFormatter bridge so existing stdlib logger calls
    are routed through the structlog pipeline.
    """
    global _setup_done  # noqa: PLW0603
    if _setup_done:
        return
    _setup_done = True

    if config is None:
        config = LoggingConfig()

    console_colors = config.console_colors and sys.stderr.isatty()
    shared_processors = _build_shared_processors()

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Clear any existing handlers from basicConfig
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)  # allow everything; handlers filter

    # --- Console handler (human-readable) ---
    console_handler = _ProgressAwareConsoleHandler(sys.stderr)
    console_handler.setLevel(
        getattr(logging, config.console_level.upper(), logging.INFO)
    )
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=console_colors),
        ],
        foreign_pre_chain=shared_processors,
    )
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)


def add_file_handlers(
    run_log_path: Path,
    debug_log_path: Path | None = None,
    file_level: str = "DEBUG",
) -> None:
    """Add file handlers for a specific agent run.

    Called by the orchestrator after creating RunStorage.
    """
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(run_log_path), encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))

    shared_processors = _build_shared_processors()
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _rename_event_key,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        foreign_pre_chain=shared_processors,
    )
    file_handler.setFormatter(file_formatter)

    # Tag our handlers so we can identify them later
    file_handler._cage_log = True  # type: ignore[attr-defined]
    file_handler._cage_log_path = str(run_log_path)  # type: ignore[attr-defined]

    root = logging.getLogger()
    root.addHandler(file_handler)

    if debug_log_path:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_handler = logging.FileHandler(str(debug_log_path), encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        debug_handler._cage_log = True  # type: ignore[attr-defined]
        debug_handler._cage_log_path = str(debug_log_path)  # type: ignore[attr-defined]
        root.addHandler(debug_handler)


def remove_file_handlers() -> None:
    """Remove all cage file handlers from the root logger.

    Called when switching between agent loops so logs go to the
    correct agent directory.
    """
    root = logging.getLogger()
    to_remove = [
        h for h in root.handlers
        if isinstance(h, logging.FileHandler) and getattr(h, "_cage_log", False)
    ]
    for handler in to_remove:
        handler.close()
        root.removeHandler(handler)


@contextmanager
def quiet_console_logging(enabled: bool) -> Iterator[None]:
    """Suppress console logs while keeping run/debug file logging intact.

    While a single-line progress UI owns the terminal, streamed console log
    records would clobber it. Temporarily raise the level of the console
    ``StreamHandler``s above CRITICAL for the duration; file handlers are left
    untouched so the run/debug logs still capture everything. Restores the
    original levels on exit.
    """
    if not enabled:
        yield
        return

    root = logging.getLogger()
    changed: list[tuple[logging.Handler, int]] = []
    silent_level = logging.CRITICAL + 1
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler) and handler.level <= logging.CRITICAL:
            changed.append((handler, handler.level))
            handler.setLevel(silent_level)
    try:
        yield
    finally:
        for handler, level in changed:
            handler.setLevel(level)


# ------------------------------------------------------------------ #
# Progress reporter — console output during cage run
# ------------------------------------------------------------------ #

class ProgressReporter:
    """Reports trial progress to the console during cage run."""

    def __init__(self, total_trials: int) -> None:
        self.total = total_trials
        self.completed = 0
        self.failed = 0

    def trial_started(self, trial_id: str, sample_id: str) -> None:
        """Print trial start (no trailing newline — completed/failed appends)."""
        idx = self.completed + self.failed + 1
        short_id = sample_id[:12] if len(sample_id) > 12 else sample_id
        sys.stderr.write(f"  [{idx}/{self.total}] {trial_id} ({short_id}...) ")

    def trial_completed(self, trial_id: str, duration_ms: int, exit_code: int) -> None:
        """Print trial completion."""
        self.completed += 1
        secs = duration_ms / 1000
        if exit_code == 0:
            sys.stderr.write(f"OK ({secs:.1f}s)\n")
        else:
            sys.stderr.write(f"exit={exit_code} ({secs:.1f}s)\n")

    def trial_failed(self, trial_id: str, error: str) -> None:
        """Print trial failure."""
        self.failed += 1
        sys.stderr.write(f"FAILED ({error[:80]})\n")

    def scoring_started(self) -> None:
        """Print scoring phase start."""
        sys.stderr.write(f"  Scoring {self.completed} completed trials...\n")

    def summary(self) -> str:
        """Return a one-line progress summary."""
        return f"{self.completed + self.failed}/{self.total} done ({self.failed} failed)"
