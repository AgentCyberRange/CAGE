"""Caching primitives for the web inspector.

Two goals:
  1. Make repeat reads of the same run / trial near-free when nothing
     on disk has changed.
  2. Make the "running" detection cheap enough that the index page can
     poll once a second without ever blocking the UI.

Everything here is read-only and process-local. Writes to disk are still
owned by the orchestrator + container proxy.

Filesystem signal primitives (``scan_run_signals``, ``RunFsSignals``,
``safe_mtime_ns``, ``is_recently_active``) used to live here. They moved
to ``cage/experiment/engine/live/fs_signals.py`` so ``cage gc`` can share the same
"is this run still ticking?" logic. We re-export them here for backward
compatibility with existing inspector consumers.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, TypeVar

from cage.experiment.engine.live import fs_signals as _fs_signals

# Re-exports kept identical to the original module surface so the web package
# can migrate internally without changing its local imports in the same batch.
_LIVE_WINDOW_NS = _fs_signals.LIVE_WINDOW_NS
RunFsSignals = _fs_signals.RunFsSignals
_signals_cache = _fs_signals._signals_cache
_signals_lock = _fs_signals._signals_lock
is_recently_active = _fs_signals.is_recently_active
safe_mtime_ns = _fs_signals.safe_mtime_ns
scan_run_signals = _fs_signals.scan_run_signals

T = TypeVar("T")


def run_has_live_activity(run_dir: Path) -> bool:
    """True when the run actually has something running right now.

    A trial whose record was never finalized past ``running`` (the process was
    Ctrl+C-killed mid-flight) leaves a phantom "running" row long after the run
    is dead. The authoritative liveness signal is filesystem activity: a trial
    is genuinely live only if a progress file is currently active or the newest
    progress tick is within the live window. Mirrors the run-level liveness used
    by the index (``RunInfo.running``) so a dead run never shows "Running".
    """
    signals = scan_run_signals(run_dir)
    return signals.active_count > 0 or is_recently_active(signals.newest_progress_mtime_ns)


def dashboard_projection_is_stale(run_dir: Path) -> bool:
    """True when ``dashboard.json`` predates a resume/re-plan of the run.

    ``dashboard.json`` is a compatibility projection written only when a run
    completes, so its per-trial rows freeze at that moment. ``planned_trials.json``
    is written at run start *and* rewritten on every resume. So a planned file
    newer than the dashboard means the run was resumed/re-run after the dashboard
    snapshot — the snapshot's trial rows (status/usage) are now stale and the
    overview must fall back to the authoritative per-trial scan instead of the
    settled fast path. Cheap: two run-level file stats, no per-trial walk.
    """
    dash = safe_mtime_ns(run_dir / "dashboard.json")
    if dash <= 0:
        return False
    return safe_mtime_ns(run_dir / "planned_trials.json") > dash


class SignatureCache:
    """Thread-safe bounded LRU keyed on (path, signature).

    A cache hit requires the signature to match exactly. When the
    underlying mtime/state changes, the next ``get`` returns ``None``
    and the caller recomputes. Eviction is strict LRU.
    """

    def __init__(self, max_size: int) -> None:
        self._data: OrderedDict[Path, tuple[Any, Any]] = OrderedDict()
        self._max = max_size
        self._lock = threading.RLock()

    def get(self, key: Path, signature: Any) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            sig, value = entry
            if sig != signature:
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: Path, signature: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = (signature, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def get_or_compute(
    cache: SignatureCache,
    key: Path,
    signature: Any,
    compute: Callable[[], T],
) -> T:
    """Cache-aside helper. Computes only on miss."""
    cached = cache.get(key, signature)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    value = compute()
    cache.put(key, signature, value)
    return value


# --------------------------------------------------------------------
# Process-wide cache singletons
# --------------------------------------------------------------------

# Run-summary cache. 256 is well above the realistic number of runs
# anyone has scrolled through in a single inspector session.
run_summary_cache = SignatureCache(max_size=256)

# Run-level projection caches for runs that never persisted a projection file
# (live, crashed, or abandoned). Both the run-history reconstruction and the
# stub dashboard otherwise re-walk every trial (+ resume archive) on disk on
# *every* page load — the dominant cost of opening a large run's detail page on
# NAS-backed storage. Keyed on the cheap structural run signature, so a settled
# run is served from cache and a live run only recomputes when a trial dir is
# added or a resume happens, not on each navigation. 256 distinct runs per
# session is well past what anyone scrolls through.
run_history_cache = SignatureCache(max_size=256)
stub_dashboard_cache = SignatureCache(max_size=256)

# Aggregate tool-call distribution per run. Reads each trial's already-persisted
# tool counts (proxy/progress.json), falling back to a full proxy.jsonl parse
# only for legacy runs. Served async (never on the critical detail-page render)
# and cached on the same cheap structural signature: a settled run is computed
# once, a live run recomputes only when a trial is added or resumed.
run_tools_cache = SignatureCache(max_size=256)

# Per-trial summary cache. 4096 trials covers a 13-run × 320-trial
# binge without eviction. Each entry is ~1 KB of parsed JSON.
trial_summary_cache = SignatureCache(max_size=4096)

# Project-tree discovery cache. Keyed on the inspector's ROOT_DIR.
# Walking the tree every request is wasteful; runs don't appear in
# new locations mid-session.
discovery_cache = SignatureCache(max_size=16)
# 60s TTL — discovery only re-walks when a brand-new ``.cage_runs/``
# parent dir appears under root, which is rare; new runs land *inside*
# existing ``.cage_runs/`` and are picked up by ``scan_runs`` directly,
# not by discovery. Cheaper than the previous 10s bucket while still
# self-healing within a minute if someone clones a sibling project.
_DISCOVERY_TTL_NS = 60 * 1_000_000_000  # 60s


def discovery_signature() -> int:
    """Time-bucketed signature so the discovery cache expires every ~60s."""
    return time.time_ns() // _DISCOVERY_TTL_NS
