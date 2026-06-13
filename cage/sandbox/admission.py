"""Resource admission control — host-level back-pressure for concurrent trials.

`HostMemoryGate` is a host-wide memory watermark with hysteresis:

  - When `psutil.virtual_memory().percent` rises above ``pause_at`` (default 80%),
    every call to ``wait_until_ok()`` blocks.
  - Blocked callers only resume after usage drops back below ``resume_at``
    (default 70%). The two-watermark design prevents oscillation at the
    boundary.

The gate is intentionally **dumb**: it does not estimate per-trial cost. It
just stops feeding new trials when the host as a whole is hot, and lets the
queue drain when the host cools down. Per-challenge estimation is unreliable
in practice; this is the operational primitive we actually want.

The gate is safe to share across threads. ``wait_until_ok`` honours a
``threading.Event`` stop signal so Ctrl+C does not get swallowed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdmissionConfig:
    """Configuration for host-level admission control."""

    enabled: bool = True
    memory_pause_at: float = 0.80   # block when used_pct > pause_at
    memory_resume_at: float = 0.70  # resume when used_pct <= resume_at
    poll_seconds: float = 3.0       # how often to re-check memory while paused
    log_every_seconds: float = 30.0  # throttle "still paused" log messages


def _virtual_memory_percent() -> float:
    """Return host memory usage as a fraction in [0, 1]. Indirection so tests can patch."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed — admission gate disabled")
        return 0.0
    return psutil.virtual_memory().percent / 100.0


class HostMemoryGate:
    """Block trial admission when host memory is hot. Resume only after it cools.

    The two-watermark design (pause_at > resume_at) gives hysteresis so admissions
    don't flap on/off at the boundary.

    Thread-safe. ``wait_until_ok`` returns ``False`` if the optional ``stop_event``
    fires (e.g. on Ctrl+C); ``True`` if memory pressure cleared.
    """

    def __init__(self, config: AdmissionConfig) -> None:
        if not (0.0 < config.memory_resume_at < config.memory_pause_at <= 1.0):
            raise ValueError(
                f"invalid watermarks: resume_at={config.memory_resume_at} "
                f"must be < pause_at={config.memory_pause_at} (both in (0, 1])"
            )
        self._cfg = config
        # is the gate currently latched closed? Once latched, only `resume_at`
        # can unlatch it (hysteresis).
        self._latched = False
        self._lock = threading.Lock()

    def _read_pct(self) -> float:
        return _virtual_memory_percent()

    def _should_pass(self, pct: float) -> bool:
        """Apply hysteresis: state transition + admission decision in one place."""
        with self._lock:
            if self._latched:
                # currently paused; only release when below resume_at
                if pct <= self._cfg.memory_resume_at:
                    self._latched = False
                    return True
                return False
            # currently open; latch closed if we cross pause_at
            if pct > self._cfg.memory_pause_at:
                self._latched = True
                return False
            return True

    def wait_until_ok(self, *, stop_event: threading.Event | None = None) -> bool:
        """Block until memory pressure clears. Returns False if stop_event fired."""
        if not self._cfg.enabled:
            return True

        pct = self._read_pct()
        if self._should_pass(pct):
            return True

        # Entered paused state: log once, then poll quietly.
        logger.warning(
            "admission_gate_pause memory=%.1f%% pause_at=%.0f%% resume_at=%.0f%% — waiting",
            pct * 100, self._cfg.memory_pause_at * 100, self._cfg.memory_resume_at * 100,
        )
        last_log = time.monotonic()
        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("admission_gate_interrupted — stop_event fired while paused")
                return False
            time.sleep(self._cfg.poll_seconds)
            pct = self._read_pct()
            if self._should_pass(pct):
                logger.info("admission_gate_resume memory=%.1f%%", pct * 100)
                return True
            now = time.monotonic()
            if now - last_log >= self._cfg.log_every_seconds:
                logger.info(
                    "admission_gate_still_paused memory=%.1f%% (resume_at=%.0f%%)",
                    pct * 100, self._cfg.memory_resume_at * 100,
                )
                last_log = now

    @property
    def is_paused(self) -> bool:
        """Snapshot of current latched state. For monitoring/status only."""
        with self._lock:
            return self._latched


_NULL_GATE: HostMemoryGate | None = None


def null_gate() -> HostMemoryGate:
    """Return a never-blocking gate. Useful as a default when admission is disabled."""
    global _NULL_GATE
    if _NULL_GATE is None:
        _NULL_GATE = HostMemoryGate(AdmissionConfig(enabled=False))
    return _NULL_GATE
