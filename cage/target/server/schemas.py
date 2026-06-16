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


class StopResponse(BaseModel):
    status: str
    chal_id: str
    message: str
    # Container logs captured just before purge, for audit. One entry per
    # container: ``{name, service, status, exit_code, error, logs}``. Empty
    # when the instance was already gone / not in memory.
    container_logs: List[Dict[str, Any]] = Field(default_factory=list)


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
