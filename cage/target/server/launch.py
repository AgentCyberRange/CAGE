"""Challenge launch: docker compose up, image build, post-start verification."""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import time
import uuid
from cage.target.adapters.source_config import build_default_registry
from cage.target.scope import normalize_target_scope, resolve_target_scope
from cage.target.server.health_probes import (
    is_instance_healthy,
    wait_for_containers_running,
    wait_for_inner_services_ready,
    wait_for_services_healthy,
)
from cage.target.server.launch_runtime import (
    ComposeRuntimePlan,
    materialize_compose_runtime,
)
from cage.target.server.network_alloc import (
    allocate_octet_slot,
    expand_subnet_pool_vars,
    release_octet_slot,
    release_reserved_project_local_subnet,
)
from cage.target.server.network_admin import (
    capture_project_logs,
    resolve_service_inner_ips,
    self_heal_docker_network,
    summarize_project_containers,
)
from cage.target.server.schemas import EntryUrl, LaunchResponse, ServiceInfo
from cage.target.server.server_state import (
    COMPOSE_UP_TIMEOUT_S,
    DOCKER_NETWORK,
    HOST_IP,
    STARTUP_POLL_INTERVAL_S,
    STARTUP_TIMEOUT_S,
    TARGET_SERVER_NAMESPACE,
    challenge_build_locks,
    find_free_port,
    get_running_instance,
    load_all_challenges,
    set_running_instance,
)
from fastapi import HTTPException
from pathlib import Path
from typing import Any, Dict, List, Optional
from cage.target.server.cleanup import (
    _cleanup_instance_impl,
    _purge_project_by_name,
    _unlink_runtime_compose,
)
from cage.target.server.network_debug import (
    _build_entry_urls,
    build_container_addrs,
    build_network_debug,
)
from cage.target.server.launch_workflow import (
    ensure_docker_cli_config_dir,
    load_env_file_vars,
)

logger = logging.getLogger(__name__)

def _entry_service_keys_from_meta(meta: Dict[str, Any]) -> set[str]:
    """Pick the user-facing entry services declared by the challenge.

    Source of truth is ``challenge.json: application_service_keys``. Order
    is preserved by callers via ``meta['source_fields']``; the set form
    here is only used for membership checks.
    """
    source_fields = meta.get("source_fields", {}) or {}
    keys = source_fields.get("application_service_keys") or []
    if isinstance(keys, str):
        keys = [keys]
    return {str(k) for k in keys if k}


def _compose_up_cmd(
    *,
    chal_path: Path,
    project_name: str,
    runtime_compose_rel: str,
) -> list[str]:
    # ``up`` is always ``--no-build``: the caller has already executed
    # ``docker compose build`` under the per-challenge build lock (see
    # ``_compose_build_locked``) if a build was needed. Re-triggering build
    # here would race classic builder's tag-create step
    # (``AlreadyExists: image <tag> already exists``) — the exact failure
    # the build lock was introduced to prevent.
    return [
        "docker", "compose",
        "--project-directory", str(chal_path),
        "-p", project_name,
        "-f", runtime_compose_rel,
        "up", "-d",
        "--no-build",
        "--force-recreate",
    ]


def _compose_build_locked(
    *,
    chal_id: str,
    chal_path: Path,
    project_name: str,
    runtime_compose_rel: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run ``docker compose build`` under a per-challenge lock.

    Classic builder (``DOCKER_BUILDKIT=0``, our default — see comment above
    on ``DOCKER_BUILDKIT``) does not atomically swap the resulting image
    tag. When N concurrent ``compose up --build`` runs all finish a fresh
    build of the same Dockerfile, the daemon races on the tag-create step
    and fails one or more of them with
    ``AlreadyExists: image "<tag>" already exists``. Per-agent launches in
    a passk batch trigger exactly this pattern, since each parallel trial
    builds the same image set.

    Serialising the build phase via ``challenge_build_locks`` is enough to
    fix this: the first trial in the batch does the cold build, subsequent
    trials wait for the lock, then enter it against a fully warm layer
    cache and the build returns in seconds (no-op tag write — daemon
    confirms the existing tag still points at the same content sha). The
    much more expensive ``up`` step (container create + healthchecks)
    still runs without the lock, preserving fan-out parallelism.

    The build subprocess shares ``COMPOSE_UP_TIMEOUT_S`` with the up step;
    we don't introduce a separate build timeout because the budget is
    already sized for the slow path.
    """
    build_cmd = [
        "docker", "compose",
        "--project-directory", str(chal_path),
        "-p", project_name,
        "-f", runtime_compose_rel,
        "build",
    ]
    with challenge_build_locks.get_lock(chal_id):
        logger.info("compose build (locked) for chal=%s project=%s", chal_id, project_name)
        return subprocess.run(
            build_cmd,
            cwd=chal_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=COMPOSE_UP_TIMEOUT_S,
        )


def _image_exists(image_name: str, env: dict[str, str]) -> bool:
    res = subprocess.run(
        ["docker", "image", "inspect", image_name],
        env=env,
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def _missing_build_images(runtime_plan: ComposeRuntimePlan, env: dict[str, str]) -> list[str]:
    """Return image tags that have a ``build:`` directive but no existing image.

    Used by ``_launch_challenge_impl`` to fail fast when a benchmark target was
    not prepared through ``cage benchmark build`` before launch.

    The presence of the explicit ``image:`` tag (required by the
    ``materialize_compose_runtime`` lint) is a high-fidelity signal that the
    benchmark build hook completed successfully at some point.
    """
    services = (runtime_plan.config.get("services") or {})
    missing: list[str] = []
    for _svc_name, svc_cfg in services.items():
        if not isinstance(svc_cfg, dict):
            continue
        if "build" not in svc_cfg:
            continue
        image = str(svc_cfg.get("image") or "").strip()
        # ``materialize_compose_runtime`` rejects build: without image:; this
        # function runs on already-materialized plans, so the field is
        # guaranteed non-empty.
        if not _image_exists(image, env=env):
            missing.append(image)
    return missing


def _missing_build_images_detail(
    *,
    chal_id: str,
    meta: dict[str, Any],
    missing_images: list[str],
) -> str:
    images = ", ".join(missing_images) if missing_images else "unknown"
    return (
        "Required target image(s) are missing and target launch does not build images. "
        f"Missing images: {images}. "
        f"Run: {_benchmark_build_command_hint(chal_id, meta)}"
    )


def _benchmark_build_command_hint(chal_id: str, meta: dict[str, Any]) -> str:
    source_fields = meta.get("source_fields", {}) or {}
    benchmark = (
        os.getenv("CAGE_BENCHMARK_ID", "").strip()
        or str(meta.get("benchmark_id") or source_fields.get("benchmark_id") or "").strip()
        or str(
            meta.get("benchmark")
            or meta.get("benchmark_name")
            or source_fields.get("benchmark")
            or source_fields.get("benchmark_name")
            or ""
        ).strip()
    )
    sample_id = str(
        source_fields.get("challenge_id")
        or source_fields.get("sample_id")
        or meta.get("id")
        or chal_id
    ).strip() or chal_id
    if benchmark:
        return f"cage benchmark build {benchmark} --sample {sample_id}"
    return f"cage benchmark build <benchmark> --sample {sample_id}"


def _reused_launch_response(
    chal_id: str,
    instance: dict,
    *,
    audience: str = "internal",
) -> LaunchResponse:
    debug = instance.get("debug", {}) or {}
    network_debug = debug.get("network", {}) or {}
    services = instance["services"]
    entry_keys = set(instance.get("entry_service_keys") or [])
    entry_urls: List[EntryUrl] = []
    if audience == "external":
        entry_urls = _build_entry_urls(
            services=services,
            entry_service_keys=entry_keys,
            host_ip=HOST_IP,
        )
    return LaunchResponse(
        status="reused",
        chal_id=chal_id,
        run_id=instance.get("run_id"),
        project_name=instance["project_name"],
        network_name=instance.get("network_name"),
        network_subnet=instance.get("network_subnet") or network_debug.get("subnet"),
        network_gateway=instance.get("network_gateway") or network_debug.get("gateway"),
        scoring=instance.get("scoring", {}),
        debug=debug,
        services=services,
        entry_urls=entry_urls,
        container_addr=build_container_addrs(services, entry_keys),
    )


def resolve_parallel_mode(meta: Dict[str, Any], requested_parallel_mode: Optional[str]) -> str:
    """Normalise/validate a requested parallel_mode, with universal default.

    ``network`` is the only sane default now that the framework defaults to
    ``compose_project_local`` network_mode — per-trial compose stacks each
    own their own bridge, so the IPAM-injection path in
    ``launch_runtime.materialize_compose_runtime`` (gated on
    ``parallel_mode == "network"``) must fire to keep subnets out of
    docker's global default /16 pool. Benchmarks that genuinely need the
    legacy alias-on-shared-bridge wiring (``network_mode:
    shared_external``) must opt in by setting ``parallel_mode: alias`` in
    their challenge.json / runtime_args.
    """
    requested = str(requested_parallel_mode or "").strip().lower()
    if requested:
        if requested not in {"network", "alias"}:
            raise HTTPException(status_code=400, detail=f"Unsupported parallel_mode: {requested_parallel_mode}")
        return requested
    return "network"


def resolve_server_target_scope(meta: Dict[str, Any], requested_target_scope: Optional[str]) -> str:
    requested = normalize_target_scope(requested_target_scope)
    if requested:
        return requested
    return resolve_target_scope(chal_data=meta, runtime_args={})


def _launch_challenge_impl(
    chal_id: str,
    force_recreate: bool,
    parallel_mode: Optional[str] = None,
    target_scope: Optional[str] = None,
    cage_run_id: Optional[str] = None,
    audience: str = "internal",
    network_only: bool = False,
    prompt_level: str = "l0",
) -> LaunchResponse:
    """Launch (or reuse, or rebuild) the runtime instance for a challenge.

    - Existing healthy instance + not forced → reuse.
    - Existing instance unhealthy or force_recreate → cleanup and rebuild (port reuse).
    - First launch → allocate fresh ports.

    ``audience='external'`` pins non-entry services (scoring sidecars,
    internal databases) to 127.0.0.1 so external clients holding the
    bearer token cannot reach them. If a per_challenge instance was
    first launched with a different audience, it is force-recreated:
    swapping bind IPs without recreate would leave dangling 0.0.0.0
    publishes that defeat the isolation.
    """
    challenges = load_all_challenges()
    if chal_id not in challenges:
        raise HTTPException(status_code=404, detail="Challenge not found")

    meta = challenges[chal_id]
    effective_parallel_mode = resolve_parallel_mode(meta, parallel_mode)
    effective_target_scope = resolve_server_target_scope(meta, target_scope)
    allow_parallel_runs = effective_target_scope == "per_agent"
    entry_service_keys = _entry_service_keys_from_meta(meta)

    existing_instance = None if allow_parallel_runs else get_running_instance(chal_id)
    should_recreate = False
    reason = ""

    if existing_instance:
        cached_audience = str(existing_instance.get("audience") or "internal").lower()
        if force_recreate:
            should_recreate = True
            reason = "force_recreate=True"
        elif not is_instance_healthy(chal_id):
            should_recreate = True
            reason = "instance unhealthy (port(s) closed)"
        elif cached_audience != audience:
            should_recreate = True
            reason = f"audience changed: {cached_audience} -> {audience}"
        else:
            logger.info(f"Reusing healthy instance for {chal_id}")
            return _reused_launch_response(chal_id, existing_instance, audience=audience)

    if should_recreate:
        logger.info(f"Recreating {chal_id} because: {reason}. Cleaning up old instance...")
        _cleanup_instance_impl(chal_id)

    adapter = build_default_registry().get(meta["adapter_kind"])
    launch_spec = adapter.build_launch_spec(meta)
    launch_spec.runtime_patches["parallel_mode"] = effective_parallel_mode
    launch_spec.runtime_patches["target_scope"] = effective_target_scope
    chal_path = Path(launch_spec.working_directory)

    if launch_spec.mode == "static":
        return LaunchResponse(status="static", chal_id=chal_id, run_id=chal_id)

    safe_id = chal_id.replace('-', '_').lower()
    run_id = chal_id
    if allow_parallel_runs:
        run_id = f"{safe_id}_{uuid.uuid4().hex[:8]}"
        project_name = f"cage_bench_{TARGET_SERVER_NAMESPACE}_{safe_id}_{run_id.split('_')[-1]}_runtime"
        runtime_compose_filename = f"docker-compose.runtime.{TARGET_SERVER_NAMESPACE}.{run_id.split('_')[-1]}.yml"
    else:
        project_name = f"cage_bench_{TARGET_SERVER_NAMESPACE}_{safe_id}_runtime"
        runtime_compose_filename = f"docker-compose.runtime.{TARGET_SERVER_NAMESPACE}.yml"
    # Park materialized compose files under ``<chal_dir>/.cage_runtime/`` so
    # the challenge dir stays clean (matches the ``.cage_runs/`` convention).
    # We pass ``--project-directory <chal_dir>`` below to keep compose's
    # notion of "project root" (used for env_file / .env resolution) at the
    # challenge dir, regardless of where the materialized file lives.
    runtime_compose_dir = chal_path / ".cage_runtime"
    runtime_compose_dir.mkdir(parents=True, exist_ok=True)
    runtime_compose_path = runtime_compose_dir / runtime_compose_filename
    runtime_compose_rel = runtime_compose_path.relative_to(chal_path).as_posix()

    # Everything from compose materialization to readiness verification
    # acquires real resources: the materialized compose file, reserved
    # project-local subnets, an octet slot, then running containers. Each
    # is registered on the stack as it is acquired, so ANY exception (the
    # named HTTP error paths and unexpected ones alike) unwinds exactly
    # what exists so far — containers are purged by project_name before
    # subnets release because ``set_running_instance`` hasn't run yet and
    # ``cleanup_instance`` couldn't find the instance. ``pop_all()``
    # commits on success.
    with contextlib.ExitStack() as failure_cleanup:
        saved_existing_instance = existing_instance if should_recreate else None
        runtime_plan = materialize_compose_runtime(
            spec=launch_spec,
            project_name=project_name,
            docker_network=DOCKER_NETWORK,
            host_ip=HOST_IP,
            runtime_compose_path=runtime_compose_path,
            find_free_port_fn=find_free_port,
            existing_external_ports=(saved_existing_instance or {}).get("external_ports"),
            cage_run_id=cage_run_id or None,
            audience=audience,
            entry_service_keys=entry_service_keys,
            network_only=network_only,
            challenge_id=chal_id,
        )
        failure_cleanup.callback(_unlink_runtime_compose, runtime_compose_path)
        public_service_names = runtime_plan.public_service_names
        final_services = [ServiceInfo(**item) for item in runtime_plan.services]
        scoring = dict((meta.get("source_fields", {}) or {}).get("runtime_scoring", {}) or {})
        runtime_network_name = runtime_plan.agent_network_name or DOCKER_NETWORK

        # Collect project-local subnets BEFORE container start so error paths can release them
        # even when the instance is not yet registered.
        allocated_subnets: list[str] = []
        _runtime_networks = (runtime_plan.config.get("networks", {}) or {})
        for _net_name, _net_cfg in _runtime_networks.items():
            if not isinstance(_net_cfg, dict):
                continue
            if _net_cfg.get("external"):
                continue
            _ipam = _net_cfg.get("ipam", {}) or {}
            _ipam_cfgs = _ipam.get("config", []) or []
            for _ipam_entry in _ipam_cfgs:
                _subnet = str((_ipam_entry or {}).get("subnet", "") or "").strip()
                if _subnet:
                    allocated_subnets.append(_subnet)

        def _release_allocated_subnets_on_error():
            for _subnet in allocated_subnets:
                try:
                    release_reserved_project_local_subnet(_subnet)
                except Exception as _exc:
                    logger.warning(f"Failed to release subnet {_subnet} on error path: {_exc}")

        failure_cleanup.callback(_release_allocated_subnets_on_error)

        logger.info(f"Launching {project_name} (recreate={should_recreate})...")
        env = os.environ.copy()
        # BuildKit/bake can hang on some challenge contexts in this environment; default to classic builder.
        env["DOCKER_BUILDKIT"] = os.getenv("TARGET_SERVER_DOCKER_BUILDKIT", "0")
        env.update(launch_spec.runtime_patches.get("compose_env", {}) or {})

        # ``subnet_pool``: borrowed from upstream rangectl. challenge.json declares
        #     {first_octet, second_octet_range:[low,high], subnet_vars:[...],
        #      ordinal_start:1}
        # The hub reserves a free second-octet under a process-wide lock and
        # exports per-trial subnet prefix env vars, e.g.
        #     RANGE1_PUBLIC_NET_PREFIX=172.51.1
        # The compose then resolves ``${RANGE1_PUBLIC_NET_PREFIX}.20`` per service.
        # This lets N parallel trials of the same challenge each own a non-
        # overlapping /24 family without rewriting the compose.
        allocated_slot: int | None = None
        pool_spec = launch_spec.runtime_patches.get("subnet_pool")
        if isinstance(pool_spec, dict) and pool_spec.get("subnet_vars"):
            first_octet = int(pool_spec.get("first_octet", 172))
            rng = pool_spec.get("second_octet_range") or [50, 80]
            slot_range = (int(rng[0]), int(rng[1]))
            ordinal_start = int(pool_spec.get("ordinal_start", 1))
            subnet_vars = [str(v) for v in pool_spec.get("subnet_vars", [])]
            try:
                allocated_slot = allocate_octet_slot(
                    project_name,
                    first_octet=first_octet,
                    slot_range=slot_range,
                )
            except RuntimeError as exc:
                raise HTTPException(500, detail=f"subnet_pool exhausted: {exc}")
            expanded = expand_subnet_pool_vars(
                first_octet=first_octet,
                slot=allocated_slot,
                subnet_vars=subnet_vars,
                ordinal_start=ordinal_start,
            )
            env.update(expanded)
            logger.info(
                "subnet_pool: project=%s slot=%d vars=%s",
                project_name, allocated_slot, expanded,
            )

        env.update(load_env_file_vars(launch_spec.runtime_patches.get("env_file")))
        ensure_docker_cli_config_dir(env)

        def _release_slot_on_error() -> None:
            if allocated_slot is not None:
                release_octet_slot(project_name)

        failure_cleanup.callback(_release_slot_on_error)

        # Self-heal DOCKER_NETWORK before compose-up. The compose file declares it
        # as ``external: True``; if the docker daemon dropped it (race under heavy
        # concurrent compose, OS reboot, or external rm), every subsequent launch
        # would otherwise fail with "external network not found" until the server
        # is restarted. Cheap idempotent check, runs before every up.
        try:
            self_heal_docker_network()
        except Exception as exc:
            logger.warning("self_heal_docker_network failed (continuing): %s", exc)

        missing_images = _missing_build_images(runtime_plan, env)
        if missing_images:
            detail = _missing_build_images_detail(
                chal_id=chal_id,
                meta=meta,
                missing_images=missing_images,
            )
            logger.error("target launch rejected for chal=%s: %s", chal_id, detail)
            raise HTTPException(500, detail=detail)
        cmd = _compose_up_cmd(
            chal_path=chal_path,
            project_name=project_name,
            runtime_compose_rel=runtime_compose_rel,
        )
        failure_cleanup.callback(_purge_project_by_name, project_name)
        try:
            res = subprocess.run(cmd, cwd=chal_path, env=env, capture_output=True, text=True, timeout=COMPOSE_UP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.error(f"Compose up timed out after {COMPOSE_UP_TIMEOUT_S}s for {project_name}")
            raise HTTPException(500, detail=f"Docker compose up timed out after {COMPOSE_UP_TIMEOUT_S}s")
        if res.returncode != 0:
            logger.error(f"Up failed:\n{res.stderr}")
            raise HTTPException(500, detail=f"Docker up failed: {res.stderr}")

        # docker compose can return 0 even if containers exit immediately — verify before reporting success.
        try:
            containers_err = wait_for_containers_running(
                project_name=project_name,
                required_services=public_service_names if public_service_names else None,
                timeout_s=STARTUP_TIMEOUT_S,
                poll_interval_s=STARTUP_POLL_INTERVAL_S,
            )
            if containers_err:
                raise RuntimeError(containers_err)

            health_err = wait_for_services_healthy(
                project_name=project_name,
                required_services=public_service_names if public_service_names else None,
                timeout_s=STARTUP_TIMEOUT_S,
                poll_interval_s=STARTUP_POLL_INTERVAL_S,
            )
            if health_err:
                raise RuntimeError(health_err)

            service_inner_ips = resolve_service_inner_ips(project_name, runtime_network_name)
            for service in final_services:
                service.inner_ip = service_inner_ips.get(service.service_name)
                if (
                    str(meta.get("category", "") or "").lower() == "web"
                    and service.inner_port is not None
                    and str(service.protocol or "tcp").lower() == "tcp"
                ):
                    service.protocol = "http"

            inner_err = wait_for_inner_services_ready(
                services=final_services,
                network_name=runtime_network_name,
                timeout_s=STARTUP_TIMEOUT_S,
                poll_interval_s=STARTUP_POLL_INTERVAL_S,
            )
            if inner_err:
                raise RuntimeError(inner_err)
        except Exception as e:
            # Capture full container logs BEFORE the failure_cleanup ExitStack
            # purges the project on the way out (LIFO ``_purge_project_by_name``).
            # Without this the real crash cause (e.g. a JVM cgroup-v2 NPE that
            # kills the container on boot) is destroyed with the containers and
            # only the symptom ("service X unreachable") survives. ``error`` is
            # kept as a concise one-liner; the verbose per-service network dump
            # is dropped — the orchestrator persists ``containers`` (the logs)
            # as a trial artifact, so it must not be smashed into a truncated
            # error string.
            details = {
                "error": str(e),
                "project_name": project_name,
                "containers": capture_project_logs(project_name),
            }
            logger.error(f"[LaunchVerify] Instance {chal_id} failed to become ready: {e}")
            raise HTTPException(status_code=500, detail=details)

        failure_cleanup.pop_all()

    debug = build_network_debug(
        project_name=project_name,
        network_name=runtime_network_name,
        parallel_mode=effective_parallel_mode,
    )
    debug["target_scope"] = effective_target_scope
    network_debug = debug.get("network", {}) or {}
    network_subnet = network_debug.get("subnet")
    network_gateway = network_debug.get("gateway")

    all_subnets = []
    runtime_networks = (runtime_plan.config.get("networks", {}) or {})
    for _net_name, _net_cfg in runtime_networks.items():
        if not isinstance(_net_cfg, dict):
            continue
        if _net_cfg.get("external"):
            continue
        _ipam = _net_cfg.get("ipam", {}) or {}
        _ipam_cfgs = _ipam.get("config", []) or []
        for _ipam_entry in _ipam_cfgs:
            _subnet = str((_ipam_entry or {}).get("subnet", "") or "").strip()
            if _subnet:
                all_subnets.append(_subnet)

    set_running_instance(run_id, {
        "chal_id": chal_id,
        "run_id": run_id,
        "project_name": project_name,
        "network_name": runtime_network_name,
        "network_subnet": network_subnet,
        "network_gateway": network_gateway,
        "scoring": scoring,
        "debug": debug,
        "compose_path": runtime_compose_path,
        "services": final_services,
        "public_services": public_service_names,
        "external_ports": runtime_plan.external_ports,
        "lifecycle_state": "running",
        "target_scope": effective_target_scope,
        "all_subnets": all_subnets,
        "cage_run_id": cage_run_id or "",
        "octet_slot": allocated_slot,
        "audience": audience,
        # Hint tier bound to THIS instance at launch (GET /prompt reads it). Set
        # per-launch (?prompt_level=), so different instances on one server can
        # run at different tiers without restarting serve.
        "prompt_level": prompt_level,
        "entry_service_keys": sorted(entry_service_keys),
        # Wall-clock launch time (epoch seconds). Read only by the management
        # console's ``GET /instances`` to show per-instance uptime; nothing in
        # the trial path depends on it.
        "created_at": time.time(),
    })

    entry_urls: List[EntryUrl] = []
    if audience == "external":
        entry_urls = _build_entry_urls(
            services=final_services,
            entry_service_keys=entry_service_keys,
            host_ip=HOST_IP,
        )

    status = "launched" if not should_recreate else "recreated"
    return LaunchResponse(
        status=status,
        chal_id=chal_id,
        run_id=run_id,
        project_name=project_name,
        network_name=runtime_network_name,
        network_subnet=network_subnet,
        network_gateway=network_gateway,
        scoring=scoring,
        debug=debug,
        services=final_services,
        entry_urls=entry_urls,
        container_addr=build_container_addrs(final_services, entry_service_keys),
    )
