"""Pydantic schemas for the challenge server's launch/stop API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ServiceInfo(BaseModel):
    service_name: str
    alias: str
    ip: str
    host: Optional[str] = None
    port: Optional[int] = None
    protocol: Optional[str] = None
    inner_host: Optional[str] = None
    inner_ip: Optional[str] = None
    inner_port: Optional[int] = None
    internal_port: Optional[int]
    external_host: Optional[str] = None
    external_port: Optional[int]


class EntryUrl(BaseModel):
    """A user-facing target entry point published to host network.

    Only returned for ``audience=external`` callers. ``role`` is the
    service name from the challenge's ``application_service_keys``
    (e.g. ``web``, ``entry_web``, ``ssh_jump``); ``url`` is the
    full address an external client should hit (``http://<host>:<port>``
    for HTTP, ``<host>:<port>`` otherwise).
    """
    name: str
    role: str
    url: str
    host: str
    port: int
    protocol: str = "tcp"


class LaunchResponse(BaseModel):
    status: str
    chal_id: str
    run_id: Optional[str] = None
    project_name: Optional[str] = None
    network_name: Optional[str] = None
    network_subnet: Optional[str] = None
    network_gateway: Optional[str] = None
    scoring: Dict[str, Any] = Field(default_factory=dict)
    debug: Dict[str, Any] = Field(default_factory=dict)
    services: List[ServiceInfo] = []
    entry_urls: List[EntryUrl] = Field(default_factory=list)
    # Internal ``ip:port`` of the target services ON the isolated docker
    # network — the serve-only counterpart of the host-published ``entry_urls``.
    # Together with ``network_name``/``network_subnet`` this is the network an
    # external agent attaches to (however it chooses to; the server does not
    # dictate that) to reach the targets at their real internal addresses.
    container_addr: List[str] = Field(default_factory=list)


class StopResponse(BaseModel):
    status: str
    chal_id: str
    message: str
    # Container logs captured just before purge, for audit. One entry per
    # container: ``{name, service, status, exit_code, error, logs}``. Empty
    # when the instance was already gone / not in memory.
    container_logs: List[Dict[str, Any]] = Field(default_factory=list)


class InstanceSummary(BaseModel):
    """A running target instance, projected for the management console.

    Returned by ``GET /instances``. Public-safe: carries only the operational
    facts a console needs (what is running, since when, is it healthy, where
    are its entry points) and never a flag, verify script, or scoring secret.
    ``entry_urls`` is populated for external callers exactly like
    :class:`LaunchResponse`; internal callers get the docker-internal view.
    """
    run_id: str
    chal_id: str
    benchmark: str = ""
    category: str = ""
    target_scope: str = ""
    audience: str = ""
    cage_run_id: str = ""
    lifecycle_state: str = ""
    healthy: Optional[bool] = None
    created_at: Optional[float] = None
    uptime_s: Optional[float] = None
    network_name: Optional[str] = None
    network_subnet: Optional[str] = None
    network_gateway: Optional[str] = None
    service_count: int = 0
    entry_urls: List[EntryUrl] = Field(default_factory=list)
    # Internal ``ip:port`` of the targets on the isolated docker network (see
    # ``LaunchResponse.container_addr``). This is the network output a serve-only
    # consumer needs; how the external agent connects to it is not our concern.
    container_addr: List[str] = Field(default_factory=list)


class ChallengeSummary(BaseModel):
    """Public-facing projection of a challenge, returned by ``GET /challenges``.

    Strictly excludes flag, verify scripts, ``agent_input``, and any
    ``source_fields`` content. The list endpoint is safe to expose to
    external callers carrying the bearer token.
    """
    id: str
    name: str
    benchmark: str
    category: str
    description: str
    task_profile: str
    entry_service_count: int


class SubmitResponse(BaseModel):
    """Verdict for a serve-only submission, returned by ``POST /submit/{run_id}``.

    The server gathered live evidence against the still-running instance and
    scored the submission with the challenge's benchmark scorer. ``scores`` maps
    each scorer name to ``{value, answer, explanation, metadata}``. ``run_dir``
    is the ``.cage_runs`` path the verdict was persisted to — the same tree the
    cage inspector renders, so a submission is inspectable exactly like a
    cage-run trial. ``closed`` reports whether the instance was torn down after
    scoring (``?close=true``).

    **One submission per instance.** The score is locked in on the first submit;
    a repeat call for the same ``run_id`` returns that recorded verdict unchanged
    with ``already_submitted=true`` (an agent cannot resubmit to fish for a
    pass). To make another attempt, launch a fresh instance.
    """
    status: str = "scored"
    run_id: str
    chal_id: str
    benchmark_module: str = ""
    scores: Dict[str, Any] = Field(default_factory=dict)
    run_dir: str = ""
    closed: bool = False
    already_submitted: bool = False


class PromptResponse(BaseModel):
    """Agent-facing task briefing for a launched instance (``GET /prompt/{run_id}``).

    Two forms of the benchmark's own ``build_prompt`` output — the SAME briefing a
    CAGE-managed agent receives as ``{task_instruction}``:

    - ``task_prompt`` — ready to hand to your agent as-is: the live target
      address(es) of THIS instance are filled in. This is the one you use.
    - ``task_prompt_template`` — the same briefing with the target address(es)
      shown as placeholder tokens (the un-filled template).

    ``prompt_level`` is the OPERATOR-selected hint tier
    (``cage benchmark serve --prompt-level``); an agent cannot raise it. Both
    strings are empty if the benchmark exposes no ``build_prompt``.
    """
    run_id: str
    chal_id: str
    task_profile: str = ""
    prompt_level: str = "l0"
    task_prompt: str = ""
    task_prompt_template: str = ""
