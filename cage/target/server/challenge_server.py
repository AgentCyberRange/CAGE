#!/usr/bin/env python3
"""FastAPI entry point for the challenge server.

Routes, lifecycle handlers, and the background health monitor live here.
Lower-level functionality is split across:
  - server_state: env config, docker client, port pool, instance registry, challenge cache
  - schemas: Pydantic request/response models
  - network_admin: docker network/container helpers
  - health_probes: container/port/HTTP readiness checks
  - launch_workflow: launch and cleanup orchestration
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Annotated, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request

try:
    from cage.target.server.path_bootstrap import ensure_repo_root_on_sys_path
except ModuleNotFoundError:
    from path_bootstrap import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path(__file__)

# --- Re-exports for backward compatibility ---
# Tests and external callers patch / import these symbols from challenge_server.
# Keeping them here avoids breaking external mock.patch.object(challenge_server, "X") use.
from cage.target.server import server_state as _server_state  # noqa: F401
from cage.target.server.health_probes import (  # noqa: F401
    is_instance_healthy,
    is_port_open,
    probe_inner_service,
    wait_for_containers_running,
    wait_for_http_ready,
    wait_for_inner_services_ready,
    wait_for_ports_open,
    wait_for_services_healthy,
)
from cage.target.server.cleanup import (  # noqa: F401
    _cleanup_instance_impl,
    cleanup_instance,
)
from cage.target.server.launch import (  # noqa: F401
    _launch_challenge_impl,
    _reused_launch_response,
    resolve_parallel_mode,
    resolve_server_target_scope,
)
from cage.target.server.launch_workflow import (  # noqa: F401
    ensure_docker_cli_config_dir,
    load_env_file_vars,
    parse_internal_port,
    pydantic_to_dict,
)
from cage.target.server.network_debug import build_network_debug  # noqa: F401
from cage.target.server.network_admin import (  # noqa: F401
    _has_active_endpoints_error,
    cleanup_orphan_networks,
    cleanup_orphan_volumes,
    ensure_docker_network,
    list_project_containers,
    remove_network_with_retry,
    resolve_service_inner_ips,
    summarize_project_containers,
)
from cage.target.server.network_admin import (
    remove_own_docker_network as _remove_own_docker_network,
)
from cage.target.server.schemas import (  # noqa: F401
    ChallengeSummary,
    EntryUrl,
    LaunchResponse,
    ServiceInfo,
    StopResponse,
)
from cage.target.server.server_state import (  # noqa: F401
    BASE_DIR,
    BENCHMARK_ROOT,
    COMPOSE_UP_TIMEOUT_S,
    DOCKER_NETWORK,
    HEALTH_POLL_INTERVAL_S,
    HEALTH_TIMEOUT_S,
    HOST_IP,
    INSTANCE_HEALTH_TIMEOUT_S,
    NETWORK_REMOVE_RETRY_INTERVAL_S,
    NETWORK_REMOVE_RETRY_TIMEOUT_S,
    PORT_OPEN_STABILITY_CHECKS,
    STARTUP_POLL_INTERVAL_S,
    STARTUP_TIMEOUT_S,
    TARGET_SERVER_NAMESPACE,
    challenge_locks,
    find_free_port,
    get_docker_client,
    get_running_instance,
    invalidate_challenge_cache,
    load_all_challenges,
    load_benchmark_sources,
    pop_running_instance,
    recovery_coordinator,
    release_allocated_port,
    running_instances,
    running_instances_lock,
    set_running_instance,
    snapshot_running_instance_ids,
    update_running_instance,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="target_server server")


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def resolve_audience(request: Request) -> str:
    """Decide whether the caller is internal or external.

    Rules:
    - ``Authorization: Bearer <token>`` with a value matching the configured
      ``TARGET_SERVER_EXTERNAL_TOKEN`` → ``external``.
    - Bearer present but mismatched → 401.
    - No bearer + caller from 127.0.0.1/::1/localhost → ``internal``.
    - No bearer + non-loopback caller while a token IS configured → 401.
    - No bearer + token not configured (legacy mode) → ``internal`` (back-compat).
    """
    token_config = _server_state.EXTERNAL_TOKEN
    auth_header = request.headers.get("Authorization", "")
    bearer = ""
    if auth_header.startswith("Bearer "):
        bearer = auth_header[len("Bearer "):].strip()

    if bearer:
        if not token_config:
            # Token-not-configured server should not silently accept any bearer.
            raise HTTPException(
                status_code=401,
                detail="bearer auth not configured on this server",
            )
        if bearer != token_config:
            raise HTTPException(status_code=401, detail="invalid bearer token")
        return "external"

    client_host = request.client.host if request.client else "127.0.0.1"
    if client_host in _LOOPBACK_HOSTS:
        return "internal"

    if token_config:
        raise HTTPException(
            status_code=401,
            detail="bearer token required for non-loopback callers",
        )
    return "internal"


async def monitor_instances():
    """Background task: every minute, check every running instance and auto-restart unhealthy ones."""
    logger.info("Health monitor started.")
    while True:
        try:
            await asyncio.sleep(60)
            current_ids = snapshot_running_instance_ids()
            if not current_ids:
                continue

            logger.info(f"[Monitor] Scanning {len(current_ids)} instances for health...")

            for instance_key in current_ids:
                info = get_running_instance(instance_key)
                if info is None:
                    continue
                lifecycle_state = str(info.get("lifecycle_state", "") or "").strip().lower()
                if lifecycle_state in {"stopping", "cleanup", "restarting"}:
                    logger.info(
                        "[Monitor] Skipping instance %s because lifecycle_state=%s",
                        instance_key,
                        lifecycle_state,
                    )
                    continue
                # per_agent instances are owned by individual workers; skip auto-restart.
                target_scope = str(info.get("target_scope", "") or "").strip().lower()
                if target_scope == "per_agent":
                    continue
                chal_id = str(info.get("chal_id") or instance_key)

                if not is_instance_healthy(instance_key):
                    logger.warning(
                        f"[Monitor] 🚨 Instance {instance_key} (challenge {chal_id}) found UNHEALTHY. Initiating auto-restart..."
                    )
                    try:
                        update_running_instance(instance_key, lifecycle_state="restarting")
                        # Background monitor calls launch_challenge as a plain python
                        # function (not via FastAPI), so the Depends() default for
                        # ``audience`` isn't resolved. Recover whatever audience the
                        # instance was originally launched with so bind-IPs survive
                        # the restart.
                        restart_audience = str(info.get("audience") or "internal").lower()
                        if instance_key != chal_id:
                            await asyncio.to_thread(cleanup_instance, chal_id, instance_key)
                            await asyncio.to_thread(
                                launch_challenge,
                                chal_id=chal_id,
                                force_recreate=False,
                                audience=restart_audience,
                            )
                        else:
                            await asyncio.to_thread(
                                launch_challenge,
                                chal_id=chal_id,
                                force_recreate=True,
                                audience=restart_audience,
                            )
                        logger.info(f"[Monitor] ✅ Instance {instance_key} successfully restarted.")
                    except Exception as e:
                        logger.error(f"[Monitor] ❌ Failed to auto-restart {instance_key}: {e}")

        except asyncio.CancelledError:
            logger.info("[Monitor] Task cancelled, stopping.")
            break
        except Exception as e:
            logger.error(f"[Monitor] Unexpected error in monitor loop: {e}")
            await asyncio.sleep(5)


def _startup_gc_namespace_scoped() -> None:
    """Reclaim docker resources from non-running runs in this namespace.

    Runs label-sweep + liveness GC at server startup so that a server
    restart after SIGKILL / OOM / host reboot doesn't leave the host
    accumulating crashed-run containers / networks / named volumes.

    Strictly namespace-scoped (only touches resources whose
    ``cage.target.namespace`` label matches this server's
    namespace). Composes cleanly with the cross-namespace peer
    reclamation that lives in ``cleanup_orphan_networks`` on main —
    the two operate on disjoint label sets.

    Skipped (with a warning) when:
      * ``CAGE_STARTUP_GC=0`` (operator opt-out)
      * No ``.cage_runs/`` root is reachable from the server's working
        directory — without artifacts we can't distinguish alive from
        dead, and reclaiming everything would be catastrophic. The
        user can still invoke ``cage gc --apply --root <path>``
        explicitly when they want.
    """
    raw = os.getenv("CAGE_STARTUP_GC", "1").strip().lower()
    # Treat empty string as "use default" (on), not as opt-out — empty
    # env vars usually mean "the user exported without a value", not "I
    # explicitly want the feature off".
    if raw in {"0", "false", "no"}:
        logger.info("startup-gc: disabled via CAGE_STARTUP_GC=%r", raw)
        return

    from cage.gc.runner import default_cage_runs_roots, gc_all

    roots = default_cage_runs_roots()
    if not roots:
        logger.info(
            "startup-gc: no .cage_runs/ root discoverable from cwd=%s — skipping; "
            "set CAGE_RUNS_ROOT env or run `cage gc --apply --root <path>` "
            "manually to clean up after crashes",
            os.getcwd(),
        )
        return
    try:
        report = gc_all(namespace=TARGET_SERVER_NAMESPACE, apply=True, search_roots=roots)
    except Exception as exc:
        logger.warning("startup-gc failed: %s", exc)
        return
    summary = report.summary()
    if summary["removed"]["containers"] or summary["removed"]["networks"] or summary["removed"]["volumes"]:
        logger.info(
            "startup-gc: reclaimed containers=%d networks=%d volumes=%d "
            "(alive=%d dead=%d orphan=%d)",
            summary["removed"]["containers"],
            summary["removed"]["networks"],
            summary["removed"]["volumes"],
            summary["alive"],
            summary["dead"],
            summary["orphan"],
        )
    else:
        logger.info(
            "startup-gc: nothing to reclaim "
            "(alive=%d dead=%d orphan=%d)",
            summary["alive"],
            summary["dead"],
            summary["orphan"],
        )


@app.on_event("startup")
async def startup_event():
    # Order matters:
    #   1. cleanup_orphan_networks — namespace-scoped network sweep
    #      (uses ``cage.target.namespace`` label).
    #   2. cleanup_orphan_volumes — namespace-scoped volume sweep
    #      (now requires the namespace label to be present; legacy
    #      unlabelled volumes are left alone).
    #   3. ensure_docker_network — claims exclusive namespace ownership.
    #      If a peer server is alive on the same namespace, this fails
    #      and we abort before doing the heavier liveness-based GC,
    #      because we'd otherwise risk reclaiming the live peer's
    #      runs.
    #   4. _startup_gc_namespace_scoped — liveness-based reclamation
    #      keyed on .cage_runs/<rid>/. Only runs after this server has
    #      asserted ownership of the namespace, so a live peer cannot
    #      have its runs misclassified as dead.
    await asyncio.to_thread(cleanup_orphan_networks)
    await asyncio.to_thread(cleanup_orphan_volumes)
    ensure_docker_network()
    await asyncio.to_thread(_startup_gc_namespace_scoped)
    asyncio.create_task(monitor_instances())


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down server, cleaning up all containers...")
    chal_ids = snapshot_running_instance_ids()
    if chal_ids:
        tasks = [asyncio.to_thread(cleanup_instance, cid) for cid in chal_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        with running_instances_lock:
            running_instances.clear()

    await asyncio.to_thread(_remove_own_docker_network)


@app.delete("/launch/{chal_id}", response_model=StopResponse)
def stop_challenge(
    chal_id: str,
    run_id: Annotated[str, Query(description="Specific runtime instance id to stop. REQUIRED — DELETE always operates on exactly one instance.")],
):
    """Stop and clean up exactly one running challenge target.

    ``run_id`` is mandatory. A DELETE that didn't carry it used to fall
    through to ``cleanup_instance(chal_id, run_id=None)`` which, combined
    with the registry's old chal_id-fallback lookup, silently destroyed
    *some other agent's* per_agent instance. Both halves of that bug have
    been removed: registry helpers are now strict-key-only, and this
    endpoint refuses calls that don't name a specific instance.
    """
    if not run_id:
        raise HTTPException(
            status_code=400,
            detail="run_id query parameter is required (one DELETE = one target)",
        )
    if get_running_instance(run_id) is None:
        # Instance not in registry. Don't try to guess — could be already
        # cleaned up by a prior DELETE, or never registered (launch error
        # path took ``_purge_project_by_name`` before ``set_running_instance``).
        # Either way, nothing else to do.
        return StopResponse(
            status="stopped",
            chal_id=chal_id,
            message=f"Instance {run_id} not in memory; no action taken.",
        )

    update_running_instance(run_id, lifecycle_state="stopping")
    captured_logs = cleanup_instance(chal_id, run_id=run_id)
    return StopResponse(
        status="stopped",
        chal_id=chal_id,
        message="Instance stopped and removed.",
        container_logs=captured_logs or [],
    )


# DELETE /run/{cage_run_id} removed: batch teardown was the wrong shape.
# A single endpoint that killed every instance for a run amplified cleanup
# bugs (one stale call → N orphaned agents) and hid which trial owned what.
# Run-end teardown now happens client-side by iterating ``_runtime_cache``
# and issuing per-instance ``DELETE /launch?run_id=<id>``. Belt-and-
# suspenders label-sweep for crashed processes lives in
# ``cage.target.local_cleanup.sweep_run`` (orchestrator-side).


@app.get("/launch/{chal_id}", response_model=LaunchResponse)
def launch_challenge(
    chal_id: str,
    force_recreate: Annotated[bool, Query(description="If true, shutdown existing instance and create a fresh one.")] = False,
    parallel_mode: Annotated[Optional[str], Query(description="Parallelization strategy: network or alias.")] = None,
    target_scope: Annotated[Optional[str], Query(description="Target allocation scope: per_challenge or per_agent.")] = None,
    cage_run_id: Annotated[Optional[str], Query(description="Cage-side run id. Stamped as cage.run_id label on every container/network.")] = None,
    audience: str = Depends(resolve_audience),
):
    # Resolve target_scope early to decide locking strategy. For per_agent each launch
    # gets a unique run_id, so the per-chal lock would needlessly serialize concurrent
    # samples and cause multi-minute startup delays under batch concurrency.
    effective_scope = "per_challenge"
    try:
        challenges = load_all_challenges()
        if chal_id in challenges:
            effective_scope = resolve_server_target_scope(challenges[chal_id], target_scope)
    except Exception:
        pass

    if effective_scope == "per_agent":
        return _launch_challenge_impl(
            chal_id=chal_id,
            force_recreate=force_recreate,
            parallel_mode=parallel_mode,
            target_scope=target_scope,
            cage_run_id=cage_run_id,
            audience=audience,
        )

    lock = challenge_locks.get_lock(chal_id)

    if force_recreate:
        def is_healthy() -> bool:
            instance = get_running_instance(chal_id)
            return instance is not None and is_instance_healthy(chal_id)

        def recover_action() -> LaunchResponse:
            with lock:
                return _launch_challenge_impl(
                    chal_id=chal_id,
                    force_recreate=True,
                    parallel_mode=parallel_mode,
                    target_scope=target_scope,
                    cage_run_id=cage_run_id,
                    audience=audience,
                )

        result = recovery_coordinator.run_serialized_recovery(
            runtime_key=chal_id,
            is_healthy=is_healthy,
            recover_action=recover_action,
        )
        if result == "reused_recent":
            existing_instance = get_running_instance(chal_id)
            if existing_instance and is_instance_healthy(chal_id):
                logger.info("Reusing recently recovered instance for %s", chal_id)
                return _reused_launch_response(chal_id, existing_instance, audience=audience)
            with lock:
                return _launch_challenge_impl(
                    chal_id=chal_id,
                    force_recreate=True,
                    parallel_mode=parallel_mode,
                    target_scope=target_scope,
                    cage_run_id=cage_run_id,
                    audience=audience,
                )
        return result

    with lock:
        return _launch_challenge_impl(
            chal_id=chal_id,
            force_recreate=force_recreate,
            parallel_mode=parallel_mode,
            target_scope=target_scope,
            cage_run_id=cage_run_id,
            audience=audience,
        )


@app.get("/challenges", response_model=list[ChallengeSummary])
def list_challenges(
    benchmark: Annotated[Optional[str], Query(description="Filter by benchmark name (exact match).")] = None,
    category: Annotated[Optional[str], Query(description="Filter by category (exact match).")] = None,
    _audience: str = Depends(resolve_audience),
):
    """List challenges with public-safe metadata only.

    Strictly excludes flag, verify scripts, ``agent_input``, and any
    ``source_fields`` content. ``entry_service_count`` is the number of
    user-facing entry services declared via ``application_service_keys``
    (1 for web challenges, 2 for post-exploitation with web+jump host, etc.).
    Authentication is the same as ``/launch``: loopback OR bearer token.
    """
    try:
        challenges = load_all_challenges()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load challenges: {exc}")

    benchmark_filter = (benchmark or "").strip() or None
    category_filter = (category or "").strip() or None

    results: list[ChallengeSummary] = []
    for chal_id, meta in challenges.items():
        bench = str(meta.get("benchmark") or meta.get("benchmark_name") or "")
        cat = str(meta.get("category") or "unknown")
        if benchmark_filter and bench != benchmark_filter:
            continue
        if category_filter and cat != category_filter:
            continue
        source_fields = meta.get("source_fields", {}) or {}
        entry_keys = source_fields.get("application_service_keys") or []
        if isinstance(entry_keys, str):
            entry_keys = [entry_keys]
        results.append(ChallengeSummary(
            id=str(chal_id),
            name=str(meta.get("name") or chal_id),
            benchmark=bench,
            category=cat,
            description=str(meta.get("description") or meta.get("task") or ""),
            task_profile=str(meta.get("task_profile") or ""),
            entry_service_count=len([k for k in entry_keys if k]),
        ))
    results.sort(key=lambda item: (item.benchmark, item.id))
    return results


if __name__ == "__main__":
    # Default to loopback so the server is not exposed to other hosts unless
    # the operator explicitly opts in by passing 0.0.0.0 (or another address).
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    uvicorn.run(app, host=host, port=port)
