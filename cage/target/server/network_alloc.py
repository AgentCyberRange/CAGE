"""Docker network & subnet allocation: project-local subnets, agent-home/cage-trial pools, octet slots, conflict remap and run-id labels."""
from __future__ import annotations

import hashlib
import ipaddress
import os
import threading
from typing import Any, Callable


_PROJECT_LOCAL_SUBNET_LOCK = threading.Lock()


_PROJECT_LOCAL_RESERVED_SUBNETS: set[str] = set()


_AGENT_HOME_SUBNET_LOCK = threading.Lock()


_AGENT_HOME_RESERVED_SUBNETS: set[str] = set()


_CAGE_TRIAL_SUBNET_LOCK = threading.Lock()


_CAGE_TRIAL_RESERVED_SUBNETS: set[str] = set()


def _pool_candidates_and_existing(
    pool_cidr: str, prefix: int
) -> tuple[list[ipaddress._BaseNetwork], list[ipaddress._BaseNetwork], ipaddress._BaseNetwork]:
    """Pre-compute (candidates, existing-docker-subnets, pool_obj) outside any lock.

    Both the docker-daemon RPC (`_collect_used_docker_subnets`) and the
    full enumeration of /<prefix> candidates inside the pool are expensive
    and need no shared state — pull them outside the per-pool lock so
    concurrent allocators don't serialise behind each other.
    """
    pool = ipaddress.ip_network(pool_cidr, strict=False)
    if prefix < pool.prefixlen:
        raise RuntimeError(
            f"_carve_subnet_from_pool: prefix /{prefix} broader than pool {pool_cidr}"
        )
    candidates = list(pool.subnets(new_prefix=prefix))
    if not candidates:
        raise RuntimeError(
            f"_carve_subnet_from_pool: no /{prefix} candidates inside {pool_cidr}"
        )
    return candidates, _collect_used_docker_subnets(pool), pool


def _pick_subnet_with_lock(
    *,
    candidates: list[ipaddress._BaseNetwork],
    existing_used: list[ipaddress._BaseNetwork],
    seed: str,
    extra_reserved: set[str],
    pool_cidr: str,
    prefix: int,
) -> str:
    """Pick the first non-conflicting candidate using both daemon-observed
    networks (``existing_used``) and in-memory reservations
    (``extra_reserved``). Caller MUST hold the pool's lock around this so
    ``extra_reserved`` is read+mutated atomically.

    ``seed`` is hashed into a starting offset so concurrent allocations
    from the same pool spread out instead of all racing for slot 0.

    Raises ``RuntimeError`` when no free slot exists.
    """
    used_networks: list[ipaddress._BaseNetwork] = list(existing_used)
    used_networks.extend(
        ipaddress.ip_network(cidr, strict=False) for cidr in extra_reserved
    )
    start = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % len(candidates)
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if any(candidate.overlaps(used) for used in used_networks):
            continue
        return str(candidate)
    raise RuntimeError(
        f"_carve_subnet_from_pool: pool {pool_cidr} fully occupied at /{prefix}"
    )


def _carve_subnet_from_pool(
    *,
    pool_cidr: str,
    prefix: int,
    seed: str,
    extra_reserved: set[str],
    lock: threading.Lock,
) -> str:
    """Allocate a ``/<prefix>`` from ``pool_cidr`` not conflicting with anything.

    Two conflict sources are checked:

      1. **Live docker networks** — every bridge / overlay network the
         daemon currently advertises an IPAM ``Subnet`` for, via
         ``_collect_used_docker_subnets``. Cross-process safety.

      2. **In-memory reservations** — ``extra_reserved`` (the caller-owned
         set guarded by ``lock``), for the race window between "this
         process picked the cidr" and "docker daemon registered the
         network".

    The docker-daemon list happens *outside* the lock; only the
    pick + add-to-set runs inside. This prevents concurrent passk
    allocations from serialising behind a slow daemon RPC (observed
    under heavy load).
    """
    candidates, existing_used, _ = _pool_candidates_and_existing(pool_cidr, prefix)
    with lock:
        chosen = _pick_subnet_with_lock(
            candidates=candidates,
            existing_used=existing_used,
            seed=seed,
            extra_reserved=extra_reserved,
            pool_cidr=pool_cidr,
            prefix=prefix,
        )
        extra_reserved.add(chosen)
        return chosen


def _agent_home_pool_settings() -> tuple[str, int]:
    pool = os.getenv("TARGET_SERVER_AGENT_HOME_SUBNET_POOL", "10.200.0.0/16").strip() or "10.200.0.0/16"
    prefix_raw = os.getenv("TARGET_SERVER_AGENT_HOME_SUBNET_PREFIX", "24").strip() or "24"
    try:
        prefix = int(prefix_raw)
    except (TypeError, ValueError):
        prefix = 24
    return pool, prefix


def allocate_agent_home_subnet(network_name: str) -> str:
    """Reserve a ``/24`` (default) from the agent_home pool for ``network_name``.

    The caller is expected to either pass the resulting cidr into
    ``docker network create --ipam-...`` immediately, or call
    ``release_agent_home_subnet`` to give it back.
    """
    pool, prefix = _agent_home_pool_settings()
    return _carve_subnet_from_pool(
        pool_cidr=pool,
        prefix=prefix,
        seed=f"agent_home:{network_name}",
        extra_reserved=_AGENT_HOME_RESERVED_SUBNETS,
        lock=_AGENT_HOME_SUBNET_LOCK,
    )


def release_agent_home_subnet(cidr: str | None) -> None:
    if not cidr:
        return
    try:
        normalized = str(ipaddress.ip_network(cidr, strict=False))
    except ValueError:
        return
    with _AGENT_HOME_SUBNET_LOCK:
        _AGENT_HOME_RESERVED_SUBNETS.discard(normalized)


def _cage_trial_pool_settings() -> tuple[str, int]:
    pool = os.getenv("CAGE_TRIAL_SUBNET_POOL", "10.201.0.0/16").strip() or "10.201.0.0/16"
    prefix_raw = os.getenv("CAGE_TRIAL_SUBNET_PREFIX", "26").strip() or "26"
    try:
        prefix = int(prefix_raw)
    except (TypeError, ValueError):
        prefix = 26
    return pool, prefix


def allocate_cage_trial_subnet(network_name: str) -> str:
    """Reserve a ``/26`` (default) from the cage-trial pool.

    Used by the orchestrator's ``build_agent_isolation_network`` to
    create per-trial bridges without letting docker auto-grab a ``/16``
    from its default pool.
    """
    pool, prefix = _cage_trial_pool_settings()
    return _carve_subnet_from_pool(
        pool_cidr=pool,
        prefix=prefix,
        seed=f"cage_trial:{network_name}",
        extra_reserved=_CAGE_TRIAL_RESERVED_SUBNETS,
        lock=_CAGE_TRIAL_SUBNET_LOCK,
    )


def release_cage_trial_subnet(cidr: str | None) -> None:
    if not cidr:
        return
    try:
        normalized = str(ipaddress.ip_network(cidr, strict=False))
    except ValueError:
        return
    with _CAGE_TRIAL_SUBNET_LOCK:
        _CAGE_TRIAL_RESERVED_SUBNETS.discard(normalized)


_OCTET_SLOT_LOCK = threading.Lock()


_OCTET_SLOT_RESERVATIONS: dict[str, int] = {}  # project_name → slot


def _stamp_target_server_network_labels(config: dict[str, Any], docker_network: str) -> None:
    """Tag compose-owned runtime networks with their owning server network."""
    networks = config.get("networks") or {}
    if not isinstance(networks, dict):
        return
    for net_name, net_cfg in list(networks.items()):
        if net_cfg is None:
            net_cfg = {}
            networks[net_name] = net_cfg
        if not isinstance(net_cfg, dict):
            continue
        if net_cfg.get("external"):
            continue
        _set_label_dict(net_cfg, "cage.target.network", docker_network)
        _set_label_dict(net_cfg, "cage.target.role", "runtime")


def _stamp_cage_run_id_labels(
    config: dict[str, Any],
    cage_run_id: str,
    *,
    namespace: str | None = None,
) -> None:
    """Tag every service + network + named volume in the runtime compose with cage labels.

    Stamps:
      * ``cage.run_id=<id>`` — so a whole-run teardown can find the
        resources;
      * ``cage.component=target`` — so the cage orchestrator's SIGTERM
        sweep, which targets ``cage.component=agent``, doesn't reach
        into the server's territory;
      * ``cage.target.namespace=<ns>`` (when ``namespace`` is
        provided) — required so ``cleanup_orphan_volumes`` and
        ``cage gc --namespace`` can isolate cross-namespace cleanup
        safely. The same label is already stamped on networks by
        ``_stamp_target_server_network_labels``; this function makes
        the volume side match.

    Compose accepts ``labels`` as either a dict or a ``key=value`` list
    — we normalise to dict.

    Named volumes (declared in the top-level ``volumes:`` block) get the
    same labels so ``cage gc`` / ``sweep_run`` can reclaim them by label
    after a crashed run. Anonymous volumes don't need labels — they get
    deleted alongside their owning container by ``docker rm -v``.
    External volumes (pre-existing, user-managed) are skipped — we don't
    own them.
    """
    extra: dict[str, str] = {
        "cage.run_id": cage_run_id,
        "cage.component": "target",
    }
    if namespace:
        extra["cage.target.namespace"] = namespace
    services = config.get("services") or {}
    if isinstance(services, dict):
        for service_config in services.values():
            if not isinstance(service_config, dict):
                continue
            for key, value in extra.items():
                _set_label_dict(service_config, key, value)
    networks = config.get("networks") or {}
    if isinstance(networks, dict):
        for net_name, net_cfg in list(networks.items()):
            if net_cfg is None:
                net_cfg = {}
                networks[net_name] = net_cfg
            if not isinstance(net_cfg, dict):
                continue
            if net_cfg.get("external"):
                # External networks are pre-created; we don't own their labels.
                continue
            for key, value in extra.items():
                _set_label_dict(net_cfg, key, value)
    volumes = config.get("volumes") or {}
    if isinstance(volumes, dict):
        for vol_name, vol_cfg in list(volumes.items()):
            if vol_cfg is None:
                vol_cfg = {}
                volumes[vol_name] = vol_cfg
            if not isinstance(vol_cfg, dict):
                continue
            if vol_cfg.get("external"):
                continue
            for key, value in extra.items():
                _set_label_dict(vol_cfg, key, value)


def _set_label_dict(parent: dict[str, Any], key: str, value: str) -> None:
    existing = parent.get("labels")
    if isinstance(existing, list):
        labels_dict: dict[str, str] = {}
        for item in existing:
            if isinstance(item, str) and "=" in item:
                k, _, v = item.partition("=")
                labels_dict[k] = v
        labels_dict[key] = value
        parent["labels"] = labels_dict
        return
    if not isinstance(existing, dict):
        existing = {}
    existing[key] = value
    parent["labels"] = existing


def _remap_conflicting_local_networks(
    config: dict[str, Any],
    *,
    project_name: str,
) -> None:
    networks_config = config.get("networks", {}) or {}
    if not isinstance(networks_config, dict):
        return

    remapped_networks: dict[str, tuple[ipaddress._BaseNetwork, ipaddress._BaseNetwork]] = {}
    allocated_subnets: set[str] = set()

    for network_name, raw_network_config in list(networks_config.items()):
        network_config = raw_network_config
        if network_config is None:
            network_config = {}
        if not isinstance(network_config, dict):
            continue
        if network_config.get("external"):
            continue

        existing_ipam = dict(network_config.get("ipam", {}) or {})
        existing_configs = list(existing_ipam.get("config", []) or [])
        if not existing_configs:
            continue

        subnet = str((existing_configs[0] or {}).get("subnet", "") or "").strip()
        if not subnet:
            continue

        try:
            original_network = ipaddress.ip_network(subnet, strict=False)
        except ValueError:
            continue

        conflicting_subnets = _collect_used_docker_subnets(original_network)
        if not any(original_network.overlaps(used) for used in conflicting_subnets):
            continue

        remapped_subnet = _allocate_remapped_runtime_subnet(
            project_name,
            network_name,
            allocated_subnets,
            str(original_network),
        )
        allocated_subnets.add(remapped_subnet)
        remapped_network = ipaddress.ip_network(remapped_subnet, strict=False)

        updated_ipam = dict(existing_ipam)
        updated_configs = list(existing_configs)
        updated_entry = dict(updated_configs[0] or {})
        updated_entry["subnet"] = remapped_subnet
        updated_configs[0] = updated_entry
        updated_ipam["config"] = updated_configs
        network_config["ipam"] = updated_ipam
        networks_config[network_name] = network_config
        remapped_networks[network_name] = (original_network, remapped_network)

    if not remapped_networks:
        return

    services_config = config.get("services", {}) or {}
    for service_config in services_config.values():
        service_networks = service_config.get("networks", {}) or {}
        if not isinstance(service_networks, dict):
            continue

        for network_name, network_membership in service_networks.items():
            mapping = remapped_networks.get(network_name)
            if not mapping or not isinstance(network_membership, dict):
                continue
            original_network, remapped_network = mapping
            raw_ip = str(network_membership.get("ipv4_address", "") or "").strip()
            if not raw_ip:
                continue
            try:
                original_ip = ipaddress.ip_address(raw_ip)
            except ValueError:
                continue
            if original_ip not in original_network:
                continue

            offset = int(original_ip) - int(original_network.network_address)
            remapped_ip = ipaddress.ip_address(int(remapped_network.network_address) + offset)
            network_membership["ipv4_address"] = str(remapped_ip)

    config["networks"] = networks_config


def _inject_compose_project_local_ipam(
    config: dict[str, Any],
    *,
    project_name: str,
    allocate_subnet_fn: Callable[[str, str, set[str]], str],
) -> None:
    networks_config = config.get("networks", {}) or {}
    allocated_subnets: set[str] = set()

    for network_name, raw_network_config in list(networks_config.items()):
        network_config = raw_network_config
        if network_config is None:
            network_config = {}
        if not isinstance(network_config, dict):
            continue
        if network_config.get("external"):
            networks_config[network_name] = network_config
            continue

        existing_ipam = dict(network_config.get("ipam", {}) or {})
        existing_configs = list(existing_ipam.get("config", []) or [])
        if existing_configs:
            subnet = str((existing_configs[0] or {}).get("subnet", "") or "").strip()
            if subnet:
                allocated_subnets.add(subnet)
            networks_config[network_name] = network_config
            continue

        subnet = allocate_subnet_fn(project_name, network_name, allocated_subnets)
        allocated_subnets.add(subnet)
        network_config["ipam"] = {"config": [{"subnet": subnet}]}
        networks_config[network_name] = network_config

    config["networks"] = networks_config


def _allocate_remapped_runtime_subnet(
    project_name: str,
    network_name: str,
    allocated_subnets: set[str],
    original_subnet: str,
) -> str:
    original_network = ipaddress.ip_network(original_subnet, strict=False)
    pool_cidr = os.getenv("TARGET_SERVER_RUNTIME_REMAP_SUBNET_POOL", "172.16.0.0/12").strip() or "172.16.0.0/12"
    pool = ipaddress.ip_network(pool_cidr, strict=False)
    prefix = original_network.prefixlen

    if prefix < pool.prefixlen:
        raise RuntimeError(
            f"Unable to remap {original_subnet}: pool {pool_cidr} is smaller than requested /{prefix}"
        )

    candidates = [pool] if prefix == pool.prefixlen else list(pool.subnets(new_prefix=prefix))
    if not candidates:
        raise RuntimeError(f"No candidate subnets available in {pool_cidr} with prefix /{prefix}")

    used_networks = _collect_used_docker_subnets(pool)
    used_networks.extend(ipaddress.ip_network(subnet, strict=False) for subnet in allocated_subnets)

    seed = f"{project_name}:{network_name}:{original_subnet}".encode("utf-8")
    start_index = int(hashlib.sha1(seed).hexdigest(), 16) % len(candidates)

    for offset in range(len(candidates)):
        candidate = candidates[(start_index + offset) % len(candidates)]
        if any(candidate.overlaps(used_network) for used_network in used_networks):
            continue
        return str(candidate)

    raise RuntimeError(f"Unable to allocate remapped subnet for {original_subnet} from {pool_cidr}")


def _allocate_project_local_subnet(
    project_name: str,
    network_name: str,
    allocated_subnets: set[str],
    *,
    pool_cidr: str | None = None,
    prefix: int | str | None = None,
) -> str:
    resolved_pool = str(
        pool_cidr
        or os.getenv("TARGET_SERVER_PROJECT_LOCAL_SUBNET_POOL", "172.31.0.0/16").strip()
        or "172.31.0.0/16"
    )
    # Default /24 (254 usable host IPs). The previous default of /28 was
    # too tight for compose stacks with many services — Dify ships 14
    # containers and would have eaten the whole network. The
    # ``172.31.0.0/16`` pool has 256 /24s, so even 30 concurrent trials
    # use a small slice of the pool. Override via env or per-challenge
    # ``project_local_subnet_prefix`` if a tighter allocation is needed
    # (e.g. a challenge whose private ``target_network`` uses /28 explicitly).
    resolved_prefix = int(
        prefix
        if prefix is not None
        else os.getenv("TARGET_SERVER_PROJECT_LOCAL_SUBNET_PREFIX", "24")
    )
    pool = ipaddress.ip_network(resolved_pool, strict=False)
    if resolved_prefix < pool.prefixlen:
        raise RuntimeError(
            f"Unable to allocate /{resolved_prefix} from pool {resolved_pool}: prefix is broader than pool"
        )
    candidates = list(pool.subnets(new_prefix=resolved_prefix))
    if not candidates:
        raise RuntimeError(f"No candidate subnets available in {resolved_pool} with prefix /{resolved_prefix}")

    seed = f"{project_name}:{network_name}".encode("utf-8")
    start_index = int(hashlib.sha1(seed).hexdigest(), 16) % len(candidates)

    with _PROJECT_LOCAL_SUBNET_LOCK:
        used_networks = _collect_used_docker_subnets(pool)
        # ``allocated_subnets`` carries strings copied verbatim from the
        # compose ``networks.<n>.ipam.config[0].subnet`` field. When a
        # benchmark writes that field as ``${VAR}.0/24`` (env-var to be
        # expanded by docker-compose at up time), it reaches us as the raw
        # placeholder, not a parseable CIDR. Such strings can't collide
        # with any real subnet, so skip them rather than letting
        # ``ip_network()`` raise.
        used_networks.extend(
            ipaddress.ip_network(subnet, strict=False)
            for subnet in allocated_subnets
            if "$" not in subnet
        )
        used_networks.extend(
            ipaddress.ip_network(subnet, strict=False)
            for subnet in _PROJECT_LOCAL_RESERVED_SUBNETS
        )

        for offset in range(len(candidates)):
            candidate = candidates[(start_index + offset) % len(candidates)]
            if any(candidate.overlaps(used_network) for used_network in used_networks):
                continue
            subnet = str(candidate)
            _PROJECT_LOCAL_RESERVED_SUBNETS.add(subnet)
            return subnet

    raise RuntimeError(f"Unable to allocate an available /{resolved_prefix} subnet from {resolved_pool}")


def _release_reserved_project_local_subnets(pool_cidr: str | None = None) -> None:
    with _PROJECT_LOCAL_SUBNET_LOCK:
        if pool_cidr is None:
            _PROJECT_LOCAL_RESERVED_SUBNETS.clear()
            return
        pool = ipaddress.ip_network(pool_cidr, strict=False)
        stale = [
            subnet
            for subnet in _PROJECT_LOCAL_RESERVED_SUBNETS
            if ipaddress.ip_network(subnet, strict=False).overlaps(pool)
        ]
        for subnet in stale:
            _PROJECT_LOCAL_RESERVED_SUBNETS.discard(subnet)


def release_reserved_project_local_subnet(subnet_cidr: str | None) -> None:
    if not subnet_cidr:
        return
    try:
        subnet = str(ipaddress.ip_network(subnet_cidr, strict=False))
    except ValueError:
        # ``all_subnets`` is built from the in-memory compose config BEFORE
        # ``${VAR}`` expansion, so it can carry literals like
        # ``"${RANGE1_PUBLIC_NET_PREFIX}.0/24"`` that aren't real CIDRs.
        # Those entries were never in the reservation set; letting the
        # ValueError propagate aborts the rest of cleanup_instance silently,
        # which leaks compose YAML files + leaves the instance dict
        # half-popped. Skip instead.
        return
    with _PROJECT_LOCAL_SUBNET_LOCK:
        _PROJECT_LOCAL_RESERVED_SUBNETS.discard(subnet)


def _collect_used_octet_slots(first_octet: int, slot_range: tuple[int, int]) -> set[int]:
    """Return second-octets in [slot_range] already claimed by a docker network.

    Scans every network's IPAM ``Subnet`` for ``<first_octet>.<slot>.x.x``.
    Slots outside the configured range are ignored — instances using slot
    99 don't block instances asking for [50..80].
    """
    used: set[int] = set()
    low, high = slot_range
    try:
        import subprocess

        ls = subprocess.run(
            ["docker", "network", "ls", "-q"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        ids = [line.strip() for line in (ls.stdout or "").splitlines() if line.strip()]
        if not ids:
            return used
        inspect = subprocess.run(
            ["docker", "network", "inspect", "--format",
             "{{range .IPAM.Config}}{{.Subnet}} {{end}}", *ids],
            capture_output=True, text=True, timeout=20, check=False,
        )
        for line in (inspect.stdout or "").splitlines():
            for token in line.split():
                token = token.strip()
                if not token or "/" not in token:
                    continue
                try:
                    net = ipaddress.ip_network(token, strict=False)
                except ValueError:
                    continue
                octets = str(net.network_address).split(".")
                if len(octets) != 4:
                    continue
                try:
                    first = int(octets[0])
                    second = int(octets[1])
                except ValueError:
                    continue
                if first != first_octet:
                    continue
                if low <= second <= high:
                    used.add(second)
    except (OSError, subprocess.SubprocessError):
        pass
    return used


def allocate_octet_slot(
    project_name: str,
    *,
    first_octet: int,
    slot_range: tuple[int, int],
) -> int:
    """Reserve a free second-octet from ``[slot_range]``.

    Thread-safe across the target_server process. Free := neither claimed by an
    existing Docker network nor reserved here for an in-flight launch.
    The reservation is keyed by ``project_name`` so ``release_octet_slot``
    can free it on teardown / error.
    """
    low, high = slot_range
    if low > high:
        raise ValueError(f"invalid slot range: [{low}, {high}]")

    with _OCTET_SLOT_LOCK:
        # Idempotent: a recreate of the same project should keep its slot.
        if project_name in _OCTET_SLOT_RESERVATIONS:
            return _OCTET_SLOT_RESERVATIONS[project_name]

        reserved = set(_OCTET_SLOT_RESERVATIONS.values())
        in_use = _collect_used_octet_slots(first_octet, (low, high)) | reserved
        for slot in range(low, high + 1):
            if slot in in_use:
                continue
            _OCTET_SLOT_RESERVATIONS[project_name] = slot
            return slot

    raise RuntimeError(
        f"no free octet slot in [{low}..{high}] under {first_octet}.x.x.x "
        f"(in-use slots: {sorted(in_use)})"
    )


def release_octet_slot(project_name: str | None) -> None:
    if not project_name:
        return
    with _OCTET_SLOT_LOCK:
        _OCTET_SLOT_RESERVATIONS.pop(project_name, None)


def expand_subnet_pool_vars(
    *, first_octet: int, slot: int, subnet_vars: list[str], ordinal_start: int = 1,
) -> dict[str, str]:
    """Materialize ``{var_name: "<first>.<slot>.<ordinal>"}`` for compose env.

    Mirrors upstream rangectl's ``prefix_for_slot`` expansion: var i in the
    declared list maps to ``<first>.<slot>.<ordinal_start + i>``. The
    compose then resolves ``${VAR}.20`` to ``172.51.1.20``.
    """
    out: dict[str, str] = {}
    for i, var in enumerate(subnet_vars):
        ordinal = ordinal_start + i
        if not (1 <= ordinal <= 254):
            raise ValueError(f"subnet ordinal out of range for {var!r}: {ordinal}")
        out[var] = f"{first_octet}.{slot}.{ordinal}"
    return out


def _collect_used_docker_subnets(pool: ipaddress._BaseNetwork) -> list[ipaddress._BaseNetwork]:
    try:
        import docker
    except Exception:
        return []

    try:
        client = docker.from_env()
        networks = client.networks.list()
    except Exception:
        return []

    used: list[ipaddress._BaseNetwork] = []
    for network in networks:
        attrs = getattr(network, "attrs", {}) or {}
        ipam_configs = ((attrs.get("IPAM", {}) or {}).get("Config") or [])
        for config in ipam_configs:
            subnet = str((config or {}).get("Subnet", "") or "").strip()
            if not subnet:
                continue
            try:
                parsed = ipaddress.ip_network(subnet, strict=False)
            except ValueError:
                continue
            if parsed.version != pool.version:
                continue
            if parsed.overlaps(pool):
                used.append(parsed)
    return used

