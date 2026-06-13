"""Run-scoped concurrency scheduler + cooperative stop signal.

A single :class:`RunScheduler` owns every piece of shared admission/concurrency
state for one ``run_experiment`` call:

* the cooperative **stop event** — the cross-cutting cancellation token threads
  poll to bail out of gates and waits when the user presses Ctrl+C;
* the **admission gate** — a :class:`~cage.sandbox.admission.HostMemoryGate`
  that pauses new trials while host memory is saturated;
* a **two-level trial concurrency gate** — a global cap on simultaneous trials
  (``runtime.max_trials_global``) plus a per-agent cap (``AgentInstance.max_concurrent``);
* a **target-setup gate** — limits concurrent target-stack launches without
  limiting agent testing.

These used to be six module globals in ``orchestrator.py``; collapsing them into
one object means the conductor builds it once and threads the *same reference*
through trial execution and cleanup (a mutable global can't be re-exported across
modules, an object reference can). An :meth:`RunScheduler.inactive` instance with
no semaphores makes every gate a no-op, so call sites never need ``None`` guards
when no run is active (unit tests, leaf classifiers).
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

from cage.sandbox.admission import HostMemoryGate, null_gate

logger = logging.getLogger(__name__)


class RunScheduler:
    """Concurrency gates + stop signal for a single run.

    Acquisition order in :meth:`concurrency_gate` is per-agent first (so an
    agent-blocked trial does NOT hold a global slot), then global; release in
    reverse. All semaphores are optional — an :meth:`inactive` scheduler leaves
    them ``None`` so every gate is a pass-through.
    """

    def __init__(
        self,
        *,
        trial_cap: int = 0,
        agent_caps: dict[int, int] | None = None,
        target_setup_cap: int = 0,
        admission: HostMemoryGate | None = None,
    ) -> None:
        self._stop = threading.Event()
        self._admission = admission
        self._trial_sem = (
            threading.BoundedSemaphore(trial_cap) if trial_cap and trial_cap > 0 else None
        )
        self._agent_sems: dict[int, threading.BoundedSemaphore] = {
            aid: threading.BoundedSemaphore(max(1, cap))
            for aid, cap in (agent_caps or {}).items()
        }
        self._target_setup_sem = (
            threading.BoundedSemaphore(target_setup_cap)
            if target_setup_cap and target_setup_cap > 0
            else None
        )

    @classmethod
    def inactive(cls) -> "RunScheduler":
        """Return a no-op scheduler: every gate passes through, stop never set."""
        return cls()

    # -- stop signal ----------------------------------------------------------

    @property
    def stop_event(self) -> threading.Event:
        """The cooperative cancellation token (always present, never ``None``)."""
        return self._stop

    def request_stop(self) -> None:
        """Signal every thread blocked in a gate or admission wait to bail."""
        self._stop.set()

    def is_stopped(self) -> bool:
        return self._stop.is_set()

    # -- admission ------------------------------------------------------------

    @property
    def admission(self) -> HostMemoryGate:
        """The host-memory admission gate, or a null gate when inactive."""
        return self._admission or null_gate()

    # -- concurrency ----------------------------------------------------------

    @contextlib.contextmanager
    def concurrency_gate(self, agent: Any):
        """Acquire the per-agent and global trial semaphores for the block.

        ``agent`` is the :class:`AgentInstance` whose budget gates this trial.
        Both semaphores are optional — when the scheduler is inactive this is a
        no-op. Polls the stop event so a cancelled run releases blocked waiters.
        """
        agent_sem = self._agent_sems.get(id(agent)) if agent is not None else None
        trial_sem = self._trial_sem

        def _acquire(sem: threading.BoundedSemaphore | None) -> bool:
            if sem is None:
                return True
            while True:
                if sem.acquire(timeout=0.5):
                    return True
                if self._stop.is_set():
                    return False

        got_agent = _acquire(agent_sem)
        if not got_agent:
            raise RuntimeError("concurrency gate cancelled (per-agent)")
        got_trial = False
        try:
            got_trial = _acquire(trial_sem)
            if not got_trial:
                raise RuntimeError("concurrency gate cancelled (global)")
            yield
        finally:
            if got_trial and trial_sem is not None:
                try:
                    trial_sem.release()
                except ValueError:
                    pass
            if got_agent and agent_sem is not None:
                try:
                    agent_sem.release()
                except ValueError:
                    pass

    @contextlib.contextmanager
    def target_setup_gate(self, chal_id: str, trial_id: str = ""):
        """Limit concurrent target stack launches without limiting agent testing."""
        sem = self._target_setup_sem
        if sem is None:
            yield
            return

        while True:
            if sem.acquire(timeout=0.5):
                break
            if self._stop.is_set():
                raise RuntimeError("target setup gate cancelled")

        logger.info(
            "target setup gate acquired: chal_id=%s trial_id=%s",
            chal_id,
            trial_id,
        )
        try:
            yield
        finally:
            sem.release()
            logger.info(
                "target setup gate released: chal_id=%s trial_id=%s",
                chal_id,
                trial_id,
            )
