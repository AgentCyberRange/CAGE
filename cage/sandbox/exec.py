"""Command execution result types shared by runtime components."""

from __future__ import annotations

from dataclasses import dataclass

from cage.contracts.execution import Timing

__all__ = ["ExecResult", "Timing"]


@dataclass(frozen=True)
class ExecResult:
    """Result of executing a command in a container or host runtime."""

    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
