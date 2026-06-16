"""Challenge instance teardown: docker compose down, project purge, state cleanup."""
from __future__ import annotations

import logging
from cage.target.server.network_alloc import (
    release_octet_slot,
    release_reserved_project_local_subnet,
)
from cage.target.server.network_admin import (
    capture_project_logs,
    remove_network_with_retry,
)
from cage.target.server.server_state import (
    DOCKER_NETWORK,
    TARGET_SERVER_NAMESPACE,
    challenge_locks,
    get_docker_client,
    get_running_instance,
    pop_running_instance,
    release_allocated_port,
    update_running_instance,
)
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _purge_project_by_name(project_name: str) -> None:
    """Force-purge a compose project's containers / networks / volumes by name.

    Used by the launch error paths BEFORE ``set_running_instance`` has been
    called: at that point ``_cleanup_instance_impl``'s registry lookup misses,
    and its fallback ``project_name`` formula (chal_id + ``_runtime``) is
    wrong because the real project name carries a per-launch hex suffix when
    ``allow_parallel_runs`` is on. Calling this with the actual project_name
    we just generated correctly targets the failed stack's resources.
    """
    if not project_name:
        return
    _purge_compose_project_resources(project_name, log_prefix="[purge]")


def _purge_compose_project_resources(project_name: str, *, log_prefix: str = "") -> None:
    """Kill and remove a compose project's containers, networks, and volumes.

    The shared docker network (``DOCKER_NETWORK``) is never touched. Every
    step is best-effort: one stuck resource must not stop the rest of the
    purge. Both teardown faces use this — the launch-error purge (by the
    just-generated project name) and the registry-driven instance cleanup.
    """

    pre = f"{log_prefix} " if log_prefix else ""
    client = get_docker_client()
    label_filter = {"label": f"com.docker.compose.project={project_name}"}
    try:
        for container in client.containers.list(all=True, filters=label_filter):
            try:
                try:
                    container.kill()
                except Exception:
                    pass
                container.remove(force=True, v=True)
                logger.info(f"{pre}Removed container: {container.name}")
            except Exception as e:
                logger.warning(f"{pre}Failed to remove container {container.name}: {e}")
    except Exception as e:
        logger.error(f"{pre}Error listing/removing containers for {project_name}: {e}")
    try:
        for network in client.networks.list(filters=label_filter):
            if network.name == DOCKER_NETWORK:
                continue
            try:
                remove_network_with_retry(network)
            except Exception as e:
                logger.warning(f"{pre}Failed to remove network {network.name}: {e}")
    except Exception as e:
        logger.error(f"{pre}Error cleaning networks for {project_name}: {e}")
    try:
        for volume in client.volumes.list(filters=label_filter):
            try:
                volume.remove(force=True)
                logger.info(f"{pre}Removed volume: {volume.name}")
            except Exception as e:
                logger.warning(f"{pre}Failed to remove volume {volume.name}: {e}")
    except Exception as e:
        logger.error(f"{pre}Error cleaning volumes for {project_name}: {e}")


def _unlink_runtime_compose(compose_path: Optional[Path]) -> None:
    if compose_path is None:
        return
    try:
        compose_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to remove runtime compose {compose_path}: {e}")
    parent = compose_path.parent
    if parent.name == ".cage_runtime":
        try:
            if not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def _cleanup_instance_impl(chal_id: str, run_id: Optional[str] = None) -> list[dict]:
    """Force-clean a project's containers, networks, volumes, and tracked allocations.

    Returns the captured container logs (one entry per container) so the
    teardown endpoint can ship them back to the orchestrator for audit. Logs
    are grabbed BEFORE the purge — once containers are removed they're gone.
    """
    instance_key = run_id or chal_id
    update_running_instance(instance_key, lifecycle_state="cleanup")
    existing_instance = get_running_instance(instance_key)
    safe_id = chal_id.replace('-', '_').lower()
    project_name = (
        existing_instance.get("project_name")
        if existing_instance is not None and existing_instance.get("project_name")
        else f"cage_bench_{TARGET_SERVER_NAMESPACE}_{safe_id}_runtime"
    )

    logger.info(f"Cleaning up project: {project_name} ...")

    captured_logs: list[dict] = []
    try:
        captured_logs = capture_project_logs(project_name)
    except Exception as exc:
        logger.warning(f"Failed to capture target logs for {project_name}: {exc}")

    _purge_compose_project_resources(project_name)

    instance = pop_running_instance(instance_key)
    if instance is not None:
        release_reserved_project_local_subnet(instance.get("network_subnet"))
        for extra_subnet in (instance.get("all_subnets") or []):
            release_reserved_project_local_subnet(extra_subnet)
        # ``project_name`` is the slot reservation key (see allocate_octet_slot).
        release_octet_slot(instance.get("project_name"))
        for port in (instance.get("external_ports") or {}).values():
            if port is not None:
                release_allocated_port(int(port))
        compose_path = instance.get("compose_path")
        if compose_path:
            _unlink_runtime_compose(Path(compose_path))
        logger.info(f"Instance {instance_key} removed from memory.")

    return captured_logs


def cleanup_instance(chal_id: str, run_id: Optional[str] = None) -> list[dict]:
    with challenge_locks.get_lock(chal_id):
        return _cleanup_instance_impl(chal_id, run_id=run_id)
