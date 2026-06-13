from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


NormalizedChallenge = dict[str, Any]


@dataclass(frozen=True)
class BenchmarkSource:
    adapter_kind: str
    root: Path
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchSpec:
    mode: str
    working_directory: str
    compose_files: list[str] = field(default_factory=list)
    target_services: list[str] = field(default_factory=list)
    dependency_services: list[str] = field(default_factory=list)
    runtime_patches: dict[str, Any] = field(default_factory=dict)
    exposure_mode: str = "host_ports"


class BenchmarkAdapter(Protocol):
    adapter_kind: str

    def discover(self, source: BenchmarkSource) -> dict[str, NormalizedChallenge]:
        ...

    def build_launch_spec(self, challenge: NormalizedChallenge) -> LaunchSpec:
        ...


def derive_flag_format(flag: str) -> str:
    if not flag:
        return "{...}"
    if "{" not in flag:
        return "{...}"
    return flag.split("{", 1)[0] + "{...}"
