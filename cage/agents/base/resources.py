"""Runtime resource declarations requested by agent implementations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentContainerResources:
    """Extra Docker resources an agent needs before container start."""

    volumes: dict[str, str] = field(default_factory=dict)
    group_add: list[str] = field(default_factory=list)
    # Launch the trial container with ``--privileged``. Needed by agents that
    # run their own Docker daemon inside the container (e.g. a Docker-in-Docker
    # orchestrator that spawns worker containers). Off by default — only agents
    # that genuinely need it opt in via this flag.
    privileged: bool = False


@dataclass
class HostRunService:
    """Host-side background process an agent needs while a run is live.

    The orchestrator starts these once at run start and terminates them at run
    end, including interrupted runs. ``dedup_key`` lets multiple agents share a
    single equivalent service declaration.
    """

    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    dedup_key: str = ""
