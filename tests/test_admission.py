"""Tests for the host memory admission gate.

We patch the memory-percent reader directly so the tests are deterministic
and do not depend on actual host memory.
"""

from __future__ import annotations

import threading
import time

import pytest

import cage.sandbox.admission as admission
from cage.sandbox.admission import AdmissionConfig, HostMemoryGate


def _gate_with_reader(reader, **cfg_kwargs):
    """Build a gate that reads memory pct from the given callable instead of psutil."""
    cfg = AdmissionConfig(**{"poll_seconds": 0.01, "log_every_seconds": 0.01, **cfg_kwargs})
    gate = HostMemoryGate(cfg)
    gate._read_pct = reader  # type: ignore[assignment]
    return gate


def test_invalid_watermarks_rejected() -> None:
    with pytest.raises(ValueError):
        HostMemoryGate(AdmissionConfig(memory_pause_at=0.5, memory_resume_at=0.6))
    with pytest.raises(ValueError):
        HostMemoryGate(AdmissionConfig(memory_pause_at=1.5, memory_resume_at=0.7))
    with pytest.raises(ValueError):
        HostMemoryGate(AdmissionConfig(memory_pause_at=0.8, memory_resume_at=0.0))


def test_disabled_gate_is_always_open() -> None:
    cfg = AdmissionConfig(enabled=False)
    gate = HostMemoryGate(cfg)
    # Even with the reader returning 0.99 (very hot), disabled gate passes.
    gate._read_pct = lambda: 0.99  # type: ignore[assignment]
    assert gate.wait_until_ok() is True
    assert gate.is_paused is False


def test_open_when_memory_below_pause() -> None:
    gate = _gate_with_reader(lambda: 0.50)
    assert gate.wait_until_ok() is True
    assert gate.is_paused is False


def test_latches_closed_above_pause_and_unlatches_below_resume() -> None:
    # Sequence: 0.85 (latch), 0.85 (still hot), 0.65 (cool, unlatch)
    pcts = iter([0.85, 0.85, 0.65])
    gate = _gate_with_reader(lambda: next(pcts))
    t0 = time.monotonic()
    assert gate.wait_until_ok() is True
    assert time.monotonic() - t0 >= 0.01  # at least one poll cycle
    assert gate.is_paused is False


def test_hysteresis_between_resume_and_pause_stays_latched() -> None:
    """While latched, a reading between resume_at and pause_at must NOT release."""
    pcts = iter([0.85, 0.75, 0.75, 0.60])  # 0.75 is between resume(0.70) and pause(0.80)
    gate = _gate_with_reader(lambda: next(pcts))
    assert gate.wait_until_ok() is True
    assert gate.is_paused is False


def test_stop_event_interrupts_wait() -> None:
    """If memory never recovers but stop_event fires, wait_until_ok returns False."""
    gate = _gate_with_reader(lambda: 0.95)  # permanently hot
    stop = threading.Event()

    def fire_stop():
        time.sleep(0.05)
        stop.set()

    threading.Thread(target=fire_stop, daemon=True).start()
    t0 = time.monotonic()
    result = gate.wait_until_ok(stop_event=stop)
    elapsed = time.monotonic() - t0
    assert result is False
    assert elapsed < 1.0, "stop_event should interrupt promptly"


def test_concurrent_waiters_all_release_when_pressure_clears() -> None:
    """Multiple threads blocked on the gate all wake up after memory cools."""
    state = {"pct": 0.90}
    gate = _gate_with_reader(lambda: state["pct"])

    finished: list[int] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        gate.wait_until_ok()
        with lock:
            finished.append(idx)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()

    # Let them all enter paused state.
    time.sleep(0.05)
    assert len(finished) == 0

    # Cool down.
    state["pct"] = 0.50
    for t in threads:
        t.join(timeout=2.0)
    assert sorted(finished) == [0, 1, 2, 3]


def test_null_gate_returns_singleton() -> None:
    g1 = admission.null_gate()
    g2 = admission.null_gate()
    assert g1 is g2
    g1._read_pct = lambda: 0.99  # type: ignore[assignment]
    assert g1.wait_until_ok() is True  # disabled, never blocks
