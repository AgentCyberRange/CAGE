"""Compose runtime materialization for a launched challenge.

Builds the per-trial ``docker-compose.runtime.*.yml`` (network model, subnet
allocation, label stamping) consumed by the launch workflow. Subnet/network
allocation lives in :mod:`network_alloc`; compose-file plumbing in
:mod:`compose_files`.
"""
from __future__ import annotations

import logging
import re
import yaml
from cage.target.adapters.base import LaunchSpec
from cage.target.server.server_state import TARGET_SERVER_NAMESPACE
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cage.target.server.network_alloc import (
    _allocate_project_local_subnet,
    _inject_compose_project_local_ipam,
    _remap_conflicting_local_networks,
    _stamp_cage_run_id_labels,
    _stamp_target_server_network_labels,
)
from cage.target.compose_files import (
    _absolutize_compose_paths,
    _compose_project_directory,
    _inject_external_network,
    expand_compose_env_values,
    load_compose_stack,
)

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


_NETWORK_MODE_DEFAULT = "compose_project_local"


_NETWORK_MODE_LEGACY = "shared_external"


_NETWORK_MODE_ALIASES = {
    "": _NETWORK_MODE_DEFAULT,
    "default": _NETWORK_MODE_DEFAULT,
    "compose_project_local": _NETWORK_MODE_DEFAULT,
    "project_local": _NETWORK_MODE_DEFAULT,
    "shared_external": _NETWORK_MODE_LEGACY,
    # ``alias`` is the historical name for the same behavior — keep it
    # accepted so older challenge.json files don't suddenly silently
    # switch policy when this default flip lands.
    "alias": _NETWORK_MODE_LEGACY,
}


@dataclass
class ComposeRuntimePlan:
    compose_path: Path
    config: dict[str, Any]
    services: list[dict[str, Any]]
    public_service_names: list[str]
    external_ports: dict[str, int]
    agent_network_name: str | None = None


def parse_internal_port(port_def: Any) -> int | None:
    try:
        if isinstance(port_def, int):
            return port_def
        if isinstance(port_def, str):
            return int(port_def.split(":")[-1].split("/")[0])
    except (TypeError, ValueError):
        return None
    return None


def parse_port_protocol(port_def: Any) -> str | None:
    if not isinstance(port_def, str):
        return None
    text = port_def.strip().lower()
    if "/" not in text:
        return None
    protocol = text.rsplit("/", 1)[-1]
    if protocol in {"tcp", "udp"}:
        return protocol
    return None


def _resolve_publish_bind_ip(
    *,
    audience: str,
    service_name: str,
    entry_service_keys: set[str] | None,
) -> str | None:
    """Return the host IP the published port should bind to, or ``None`` for docker's default.

    For audience=external + a configured ``application_service_keys`` allow-list,
    non-entry services (scoring sidecars, internal databases) are pinned to
    127.0.0.1 so external clients cannot reach them. Entry services and the
    legacy internal-audience path keep docker's default (0.0.0.0).
    """
    if audience != "external" or not entry_service_keys:
        return None
    if service_name in entry_service_keys:
        return None
    return "127.0.0.1"


def build_service_alias(project_name: str, service_name: str) -> str:
    return f"{project_name}_{service_name}"


def build_service_inner_host(project_name: str, service_name: str) -> str:
    # Strip the ``cage_bench_<ns>_`` prefix (or the legacy ``ctf_<ns>_`` prefix
    # from older deployments) to recover the bare challenge identifier.
    token = re.sub(r"^(?:cage_bench|ctf)_[^_]+_", "", project_name)
    if token.endswith("_runtime"):
        token = token[: -len("_runtime")]
    if not token or token == "runtime":
        return service_name
    return f"{token}_{service_name}"


def materialize_compose_runtime(
    *,
    spec: LaunchSpec,
    project_name: str,
    docker_network: str,
    host_ip: str,
    runtime_compose_path: Path,
    find_free_port_fn: Callable[[], int],
    existing_external_ports: dict[str, int] | None,
    allocate_subnet_fn: Callable[[str, str, set[str]], str] | None = None,
    cage_run_id: str | None = None,
    audience: str = "internal",
    entry_service_keys: set[str] | None = None,
    network_only: bool = False,
    challenge_id: str | None = None,
) -> ComposeRuntimePlan:
    # ``audience='external'`` + a non-empty ``entry_service_keys`` is the
    # only combination that publishes user-facing services to 0.0.0.0;
    # every other published service (e.g. scoring sidecars like
    # ``evaluator``) is bound to 127.0.0.1 so external clients carrying
    # the bearer token cannot reach the scoring channel.
    if spec.mode != "compose":
        raise ValueError(f"Cannot materialize non-compose launch spec: {spec.mode}")

    compose_env = dict(spec.runtime_patches.get("compose_env", {}) or {})
    config = load_compose_stack(spec.compose_files, compose_env=compose_env)
    config = expand_compose_env_values(config, compose_env)
    project_directory = _compose_project_directory(spec)
    _absolutize_compose_paths(config, project_directory, compose_env=compose_env)
    services_config = config.get("services", {}) or {}
    target_service_names = list(spec.target_services)
    launch_service_names = set(target_service_names) | set(spec.dependency_services)
    public_service_names: list[str] = []
    services: list[dict[str, Any]] = []
    external_ports: dict[str, int] = {}
    existing_external_ports = existing_external_ports or {}
    # Resolve network mode. Default is ``compose_project_local`` — each
    # target instance owns its own compose-created networks. Empty / unset
    # is treated as the default. Legacy alias-on-shared-bridge is opt-in
    # via ``shared_external`` (or its historical alias name ``alias``).
    raw_network_mode = str(spec.runtime_patches.get("network_mode", "") or "").strip().lower()
    network_mode = _NETWORK_MODE_ALIASES.get(raw_network_mode)
    if network_mode is None:
        # Unknown literal — fall back to the safe default rather than
        # raising, so a single typo in challenge.json doesn't kill a whole
        # batch. The warning shows up in target_server logs for review.
        logger.warning(
            "Unknown network_mode=%r; falling back to %s",
            raw_network_mode, _NETWORK_MODE_DEFAULT,
        )
        network_mode = _NETWORK_MODE_DEFAULT
    use_compose_project_local_networks = network_mode == _NETWORK_MODE_DEFAULT
    parallel_mode = str(
        spec.runtime_patches.get("parallel_mode", "")
        or ("network" if use_compose_project_local_networks else "alias")
    ).strip().lower()
    if parallel_mode not in {"network", "alias"}:
        parallel_mode = "network" if use_compose_project_local_networks else "alias"
    agent_network = str(spec.runtime_patches.get("agent_network", "") or "").strip() or None
    # Auto-pick agent_network when challenge.json doesn't specify one.
    # Rules:
    #   0 declared (user-owned, non-external) networks → inject a
    #     ``default: {}`` entry so IPAM allocation + the cage label-stamp
    #     pass can treat it like any other declared network. This matches
    #     compose's own behavior: when no ``networks:`` block is present it
    #     auto-creates ``<project>_default`` and attaches every implicit
    #     service to it. By materializing the entry up front we keep the
    #     subnet allocation deterministic and namespace-scoped, instead of
    #     letting docker pick from its global default pool (which collides
    #     under heavy concurrent compose-up).
    #   1 declared → that one (the only sensible choice).
    #   >1 declared → require explicit ``agent_network`` in challenge.json.
    #     Picking arbitrarily would silently land the agent on the wrong
    #     plane (e.g. an internal-only DB segment) for some challenges.
    if use_compose_project_local_networks and agent_network is None:
        declared_networks_dict = config.get("networks") or {}
        if not isinstance(declared_networks_dict, dict):
            declared_networks_dict = {}
        user_owned_networks = [
            name for name, cfg in declared_networks_dict.items()
            if not (isinstance(cfg, dict) and cfg.get("external"))
        ]
        if len(user_owned_networks) == 0:
            config.setdefault("networks", {})["default"] = {}
            agent_network = "default"
        elif len(user_owned_networks) == 1:
            agent_network = user_owned_networks[0]
        else:
            raise ValueError(
                "Multiple compose networks declared "
                f"({user_owned_networks!r}) but no agent_network specified "
                "in challenge.json. Add \"agent_network\": \"<name>\" so "
                "cage knows which network the agent should join."
            )
    # Whether auto-picked or explicit, make sure the agent_network exists
    # as an entry under ``config["networks"]`` so the IPAM allocation +
    # label-stamp passes treat it like any other declared network. Without
    # this, an explicit ``agent_network: default`` against a compose file
    # with no ``networks:`` block would let docker allocate a subnet from
    # its global default pool (172.17/16, 172.18/16…) — fine in isolation
    # but prone to clashing with host VPNs and not labelled for our
    # namespace-scoped orphan cleanup.
    if use_compose_project_local_networks and agent_network:
        nets = config.setdefault("networks", {})
        nets.setdefault(agent_network, {})
    agent_network_name = docker_network
    if use_compose_project_local_networks and agent_network:
        agent_network_name = f"{project_name}_{agent_network}"
    subnet_policy = {
        "pool": str(spec.runtime_patches.get("project_local_subnet_pool", "") or "").strip() or None,
        "prefix": spec.runtime_patches.get("project_local_subnet_prefix"),
    }

    filtered_services: dict[str, Any] = {}

    for service_name, service_config in services_config.items():
        if launch_service_names and service_name not in launch_service_names:
            continue

        service_config.pop("container_name", None)
        # Lint: every service with ``build:`` must declare an explicit
        # ``image:``. Cage gives every trial a unique project_name for
        # isolation, so docker compose's default ``<project>-<service>``
        # tagging would produce a fresh image tag per trial — hundreds of
        # alias-of-the-same-content tags after a passk batch. Forcing
        # benchmarks to write the tag they want makes cache reuse explicit
        # and visible (see commit history for the tag-pollution audit).
        if "build" in service_config and not str(service_config.get("image") or "").strip():
            raise ValueError(
                f"Service {service_name!r} declares 'build:' but no 'image:'. "
                f"Cage requires explicit image tags so trials share cache "
                f"instead of polluting ``docker image ls``. Add e.g. "
                f"``image: <benchmark>-<challenge>-{service_name}:latest`` "
                f"to the compose file."
            )
        if use_compose_project_local_networks and parallel_mode == "network":
            service_alias = service_name
        else:
            service_alias = build_service_alias(project_name, service_name)
        if use_compose_project_local_networks:
            service_inner_host = service_name
        else:
            service_inner_host = build_service_inner_host(project_name, service_name)
            _inject_external_network(
                service_config,
                docker_network,
                aliases=[service_inner_host, service_alias],
                project_directory=project_directory,
                compose_env=compose_env,
            )

        if service_name not in target_service_names:
            service_config.pop("ports", None)
            filtered_services[service_name] = service_config
            services.append(
                {
                    "service_name": service_name,
                    "alias": service_alias,
                    "ip": host_ip,
                    "host": host_ip,
                    "port": None,
                    "inner_host": service_inner_host,
                    "inner_ip": None,
                    "inner_port": None,
                    "internal_port": None,
                    "external_host": host_ip,
                    "external_port": None,
                }
            )
            continue

        internal_port = None
        external_port = None
        protocol = "tcp"
        inferred_ports = (spec.runtime_patches.get("target_ports", {}) or {})
        inferred_protocols = (spec.runtime_patches.get("target_port_protocols", {}) or {})
        if service_name in inferred_ports:
            internal_port = inferred_ports.get(service_name)
            protocol = str(inferred_protocols.get(service_name, "tcp") or "tcp").lower()
        else:
            original_ports = service_config.get("ports", []) or []
            if original_ports:
                internal_port = parse_internal_port(original_ports[0])
                protocol = parse_port_protocol(original_ports[0]) or "tcp"

        _entry_keys = entry_service_keys or set()
        if network_only and spec.exposure_mode == "host_ports":
            # network_only: NOTHING is host-published — every service (the
            # user-facing entry target AND scoring sidecars like ``evaluator``)
            # is reachable only over the isolated docker network at its internal
            # address (``container_addr``/``inner_ip``). An agent scanning
            # localhost or the host reaches nothing. The host-side scorer does
            # not need a host port either: the host routes directly to the
            # per-instance bridge, so it POSTs to the evaluator at its inner IP
            # (see ``_resolve_evaluator_base_url``, which prefers the inner
            # address). This keeps the isolation promise total — no part of an
            # instance leaks onto the host.
            service_config.pop("ports", None)
        elif spec.exposure_mode == "host_ports" and internal_port is not None:
            external_port = existing_external_ports.get(service_name) or find_free_port_fn()
            port_suffix = f"/{protocol}" if protocol != "tcp" else ""
            bind_ip = _resolve_publish_bind_ip(
                audience=audience,
                service_name=service_name,
                entry_service_keys=entry_service_keys,
            )
            if bind_ip:
                service_config["ports"] = [f"{bind_ip}:{external_port}:{internal_port}{port_suffix}"]
            else:
                service_config["ports"] = [f"{external_port}:{internal_port}{port_suffix}"]
            external_ports[service_name] = external_port
            public_service_names.append(service_name)
        else:
            service_config.pop("ports", None)

        filtered_services[service_name] = service_config

        services.append(
            {
                "service_name": service_name,
                "alias": service_alias,
                "ip": host_ip,
                "host": host_ip,
                "port": external_port,
                "protocol": protocol,
                "inner_host": service_inner_host,
                "inner_ip": None,
                "inner_port": internal_port,
                "internal_port": internal_port,
                "external_host": host_ip,
                "external_port": external_port,
            }
        )

    config["services"] = filtered_services
    _remap_conflicting_local_networks(
        config,
        project_name=project_name,
    )
    if use_compose_project_local_networks and parallel_mode == "network":
        _inject_compose_project_local_ipam(
            config,
            project_name=project_name,
            allocate_subnet_fn=allocate_subnet_fn
            or (
                lambda project_name, network_name, allocated_subnets: _allocate_project_local_subnet(
                    project_name,
                    network_name,
                    allocated_subnets,
                    pool_cidr=subnet_policy["pool"],
                    prefix=subnet_policy["prefix"],
                )
            ),
        )
    elif not use_compose_project_local_networks:
        config.setdefault("networks", {})
        config["networks"][docker_network] = {"external": True}

    _stamp_target_server_network_labels(config, docker_network)
    if cage_run_id:
        _stamp_cage_run_id_labels(
            config,
            cage_run_id,
            namespace=TARGET_SERVER_NAMESPACE,
        )

    runtime_compose_path.write_text(
        yaml.dump(config, default_flow_style=False, indent=2),
        encoding="utf-8",
    )

    return ComposeRuntimePlan(
        compose_path=runtime_compose_path,
        config=config,
        services=services,
        public_service_names=public_service_names,
        external_ports=external_ports,
        agent_network_name=agent_network_name,
    )

