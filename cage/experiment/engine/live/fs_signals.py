"""Filesystem signal primitives shared by the web inspector and ``cage gc``.

Both the inspector's "is this run running?" badge and the GC's
"can I reclaim this run's docker resources?" decision are answered
from the same low-level question: **has any** ``progress.json``
**under ``.cage_runs/<rid>/trials/`` been touched recently?**

The primitives live here so the policy modules (``cage/experiment/engine/live/liveness.py``
and ``cage/web/data/__init__.py``) can share an authoritative implementation.
``cage/web/cache.py`` re-exports the names for backward compatibility with
existing consumers.

No flask, no docker — only stdlib. Safe to import from any layer.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LIVE_WINDOW_NS",
    "RunFsSignals",
    "as_int",
    "dashboard_pending_count",
    "is_recently_active",
    "safe_mtime_ns",
    "scan_run_signals",
]


def as_int(value) -> int:
    """Best-effort int conversion that returns 0 on bad input.

    Used by ``dashboard_pending_count`` (and other places that read
    integers out of loosely-typed JSON payloads).
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def dashboard_pending_count(data) -> int:
    """Count trials the dashboard marks as not-yet-done.

    Mirrors the formula used by the web inspector (and that the
    orchestrator's dashboard.json schema actually produces): each agent
    entry carries ``total / completed / failed`` integers and a
    ``trials`` list. The pending count is ``max(0, total - completed -
    failed)`` summed across agents.

    This helper is the **single source of truth** for the formula —
    both ``cage/web/data/__init__.py`` and ``cage/experiment/engine/live/liveness.py`` import
    it, so the GC's "is this run alive?" decision can never silently
    diverge from the inspector's "running" badge.
    """
    if not isinstance(data, dict):
        return 0
    agents = data.get("agents") or {}
    if not isinstance(agents, dict):
        return 0
    count = 0
    for agent_data in agents.values():
        if not isinstance(agent_data, dict):
            continue
        total = as_int(agent_data.get("total"))
        completed = as_int(agent_data.get("completed"))
        failed = as_int(agent_data.get("failed"))
        count += max(0, total - completed - failed)
    return count


# Trials whose progress.json hasn't ticked in this long are considered
# "stale" (orchestrator likely crashed) rather than "running". Public so
# ``cage gc`` callers can monkey-patch a smaller window in tests.
LIVE_WINDOW_NS = 300 * 1_000_000_000  # 5 minutes


def safe_mtime_ns(path: Path) -> int:
    """``stat().st_mtime_ns`` that returns -1 instead of raising."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


@dataclass(frozen=True)
class RunFsSignals:
    """Cheap filesystem-derived signals for one run.

    Computed by walking ``run_dir/trials/`` once with no JSON reads
    beyond ``stat``. The fields are everything downstream callers need
    to decide "is this run running?", "did anything change since last
    tick?" and "where do I find live progress files?".
    """
    completed_count: int            # trial dirs with task_output.json
    active_count: int               # trial dirs with progress.json but no task_output.json
    newest_progress_mtime_ns: int   # for change detection / "last active" badge
    progress_files: tuple[Path, ...]   # paths to read for live aggregates (active only)
    completed_dirs: tuple[Path, ...]   # trial dirs that finished (for delta API)
    active_dirs: tuple[Path, ...]      # trial dirs still in flight

    @property
    def last_active_ms(self) -> int:
        if self.newest_progress_mtime_ns <= 0:
            return 0
        return self.newest_progress_mtime_ns // 1_000_000


_SIGNALS_TTL_NS = 1_000_000_000  # 1s — concurrent polls share one scan
_signals_cache: dict[Path, tuple[int, "RunFsSignals"]] = {}
_signals_lock = threading.RLock()


def scan_run_signals(run_dir: Path) -> RunFsSignals:
    """Walk ``run_dir/trials`` once and collect cheap signals.

    Recognizes three trial-dir layouts (flat ``trials/<id>``, nested
    ``trials/<challenge>/<variant>``, and the legacy ``<mode>/trials/<id>``).
    A directory is considered a trial dir iff it contains either
    ``task_output.json`` or ``proxy/progress.json``.

    No JSON reads — only ``stat`` and ``scandir``. Designed to finish in
    a few ms even for 320-trial runs on local disk. A 1-second TTL
    deduplicates back-to-back walks from concurrent inspector tabs.
    """
    now = time.time_ns()
    with _signals_lock:
        cached = _signals_cache.get(run_dir)
        if cached and (now - cached[0]) < _SIGNALS_TTL_NS:
            return cached[1]

    result = _scan_run_signals_uncached(run_dir)
    with _signals_lock:
        _signals_cache[run_dir] = (now, result)
        # Trim if it grows unbounded.
        if len(_signals_cache) > 512:
            for k in list(_signals_cache.keys())[: len(_signals_cache) - 512]:
                _signals_cache.pop(k, None)
    return result


def _scan_run_signals_uncached(run_dir: Path) -> RunFsSignals:
    completed: list[Path] = []
    active: list[Path] = []
    progress_files: list[Path] = []
    newest_mtime = 0

    roots: list[Path] = []
    direct = run_dir / "trials"
    if direct.is_dir():
        roots.append(direct)
    else:
        # Legacy: stateless/trials/, stateful/trials/, ...
        try:
            for child in run_dir.iterdir():
                if child.is_dir() and (child / "trials").is_dir():
                    roots.append(child / "trials")
        except OSError:
            return RunFsSignals(0, 0, 0, (), (), ())

    stack = list(roots)
    while stack:
        cur = stack.pop()
        try:
            children = list(os.scandir(cur))
        except OSError:
            continue

        has_task_output = False
        progress_path: Path | None = None
        sub_dirs: list[Path] = []

        for entry in children:
            try:
                if entry.is_dir(follow_symlinks=False):
                    # Resume archives are historical snapshots of past trial
                    # attempts (see cage.artifacts.run_storage.RESUME_ARCHIVE_MARKER) — they
                    # carry the same artifact layout as live trials, but must
                    # not contribute to the run's completed/active counts.
                    if ".before_resume_" in entry.name:
                        continue
                    if entry.name == "proxy":
                        candidate = Path(entry.path) / "progress.json"
                        m = safe_mtime_ns(candidate)
                        if m >= 0:
                            progress_path = candidate
                            if m > newest_mtime:
                                newest_mtime = m
                    else:
                        sub_dirs.append(Path(entry.path))
                elif entry.name == "task_output.json":
                    has_task_output = True
            except OSError:
                continue

        if has_task_output:
            completed.append(Path(cur))
        elif progress_path is not None:
            active.append(Path(cur))
            progress_files.append(progress_path)
        else:
            # Not a trial dir itself — recurse for nested layouts.
            stack.extend(sub_dirs)

    return RunFsSignals(
        completed_count=len(completed),
        active_count=len(active),
        newest_progress_mtime_ns=newest_mtime,
        progress_files=tuple(progress_files),
        completed_dirs=tuple(completed),
        active_dirs=tuple(active),
    )


def is_recently_active(mtime_ns: int, *, window_ns: int = LIVE_WINDOW_NS) -> bool:
    """True if mtime is within the live window. Used to distinguish
    "currently running" from "orchestrator died, leaving artifacts."
    """
    if mtime_ns <= 0:
        return False
    return (time.time_ns() - mtime_ns) < window_ns
