"""Cage framework error taxonomy.

These are the typed exceptions the framework raises and classifies. They live in
``contracts`` (layer 0) because they are vocabulary shared across every layer:
runtime raises them, the conductor catches them, and
``termination_info_from_exception`` maps them onto a terminal verdict. Keeping the
taxonomy here — rather than inside a runtime module — means a lower layer can
raise a ``CageError`` without importing up into the runtime.
"""

from __future__ import annotations


class CageError(Exception):
    """Base class for all Cage framework errors."""


class CageInterrupt(CageError, KeyboardInterrupt):
    """Experiment interrupted by user (Ctrl+C / SIGINT).

    When caught by the conductor, partial results are persisted before the
    process exits.
    """


class TrialError(CageError):
    """Error during a single trial execution."""

    def __init__(self, trial_id: str, message: str) -> None:
        self.trial_id = trial_id
        super().__init__(f"Trial {trial_id}: {message}")


class TrialTimeout(TrialError):
    """Trial exceeded its allowed duration."""

    def __init__(self, trial_id: str, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        super().__init__(trial_id, f"timed out after {timeout_s:.0f}s")


class ContainerError(CageError):
    """Error in container lifecycle (start, exec, stop)."""

    def __init__(self, container_name: str, message: str) -> None:
        self.container_name = container_name
        super().__init__(f"Container {container_name}: {message}")


class ProxyError(CageError):
    """Error in the in-container proxy (start, health check, stop)."""


__all__ = [
    "CageError",
    "CageInterrupt",
    "ContainerError",
    "ProxyError",
    "TrialError",
    "TrialTimeout",
]
