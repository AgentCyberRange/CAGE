"""Run reporter protocol.

The conductor drives a run headlessly; a *reporter* is the optional sink that
turns lifecycle callbacks into human-facing output. The protocol lives in
``contracts`` (layer 0) so the conductor (and the trial runner, and the proxy
monitor) depend *down* on this interface and never *up* on the concrete CLI
reporter. The CLI builds a concrete reporter and injects a factory into
``run_experiment``; when none is injected the run is silent.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class Reporter(Protocol):
    """Sink for run/trial lifecycle callbacks during an experiment.

    Implementations render however they like (a live terminal line, a log, a
    test spy). Every method is best-effort and side-effecting; the conductor
    treats a missing reporter (``None``) as "render nothing", so an
    implementation only needs to support the callbacks it cares about.
    """

    enabled: bool

    def live(self) -> AbstractContextManager[Reporter]:
        """Enter the reporter's active display for the duration of the run."""
        ...

    def agent_started(self, agent_label: str, total_trials: int) -> None: ...

    def trial_started(
        self,
        *,
        agent_label: str,
        trial_id: str,
        sample_id: str,
        trial_index: int,
    ) -> None: ...

    def trial_finished(
        self,
        *,
        agent_label: str,
        trial_id: str,
        status: str,
        duration_ms: int | None,
        exit_code: int,
    ) -> None: ...

    def trial_replayed(
        self,
        *,
        agent_label: str,
        trial_id: str,
        sample_id: str,
        trial_index: int,
        status: str,
    ) -> None: ...

    def record_model_request(self, event: Any) -> None: ...


# A factory the CLI injects into the conductor. It receives the run's display
# parameters and returns a concrete Reporter. Kept as a broad callable because
# the concrete builder lives in the CLI layer and owns its own keyword surface.
ReporterFactory = Any


__all__ = ["Reporter", "ReporterFactory"]
