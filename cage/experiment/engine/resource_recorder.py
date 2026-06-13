"""Runtime resource recorders — write ResourceLedger entries for trial resources.

When a trial brings up a runtime resource (an agent container, an agent isolation
network, an in-container proxy process, or a target_server runtime), it appends a
canonical record to ``resources.jsonl`` so inspect, replay, and GC can explain
what was created and whether cleanup succeeded. These helpers translate the
runtime objects into ledger fields and persist them through
:class:`~cage.artifacts.trial_session.TrialRuntimeSession`, always keyed by the
canonical (``TrialRecord``) trial id rather than the legacy ``sample/pass_n``
shape.

This is distinct from ``cage/artifacts/resources.py`` (the ledger reader/writer
primitives): these are the *runtime* recorders that build the records.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from cage.agents.base import AgentInstance
from cage.artifacts.run_storage import RunStorage
from cage.artifacts.trial_session import TrialRuntimeSession
from cage.artifacts.canonical_marks import plan_trial_id
from cage.experiment.model import Trial
from cage.proxy.host import ContainerProxyInstance
from cage.sandbox.containers import Container
from cage.target.provisioning import AgentIsolationNetwork

logger = logging.getLogger(__name__)


def _record_agent_container_resource(
    *,
    storage: RunStorage,
    run_id: str,
    agent: AgentInstance,
    trial: Trial,
    container: Container,
    status: str,
    cleanup_error: str | None = None,
) -> None:
    """Best-effort ledger record for an orchestrator-owned agent container.

    The legacy scheduler identifies a trial as ``sample/pass_n``. Canonical
    resource records should instead point at the ``TrialRecord`` id, which also
    includes the agent/model subject. Keeping that conversion here makes
    ``resources.jsonl`` usable by inspect, replay, and future GC code without
    teaching those consumers legacy trial id shapes.
    """

    try:
        TrialRuntimeSession(
            run_dir=storage.run_dir,
            trial_id=plan_trial_id(agent, trial),
        ).record_resource(
            run_id=run_id,
            resource_id=f"docker_container:{container.name}",
            kind="docker_container",
            provider="docker",
            external_id=container.name,
            status=status,
            cleanup_action=f"docker rm -f {container.name}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata={
                "image": str(getattr(container, "image", "") or ""),
                "labels": dict(getattr(container, "labels", {}) or {}),
            },
            cleanup_error=cleanup_error,
        )
    except Exception as exc:
        logger.warning(
            "resource ledger update failed for container %s: %s",
            getattr(container, "name", "(unknown)"),
            exc,
        )

def _record_agent_isolation_network_resource(
    *,
    storage: RunStorage,
    run_id: str,
    agent: AgentInstance,
    trial: Trial,
    isolation: AgentIsolationNetwork,
    status: str,
    cleanup_error: str | None = None,
) -> None:
    """Best-effort ledger record for a trial-local agent isolation bridge.

    Agent isolation bridges are Cage-owned Docker networks created outside the
    target server so agents can reach only public target services. They are
    trial-local runtime resources, so their lifecycle belongs in the same
    canonical ``resources.jsonl`` ledger as agent containers. The record uses
    the canonical trial id to keep cleanup/debug views aligned with
    ``TrialRecord``.
    """

    try:
        TrialRuntimeSession(
            run_dir=storage.run_dir,
            trial_id=plan_trial_id(agent, trial),
        ).record_resource(
            run_id=run_id,
            resource_id=f"docker_network:{isolation.name}",
            kind="docker_network",
            provider="docker",
            external_id=isolation.name,
            status=status,
            cleanup_action=f"docker network rm {isolation.name}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata={
                "subnet": isolation.subnet or "",
                "connected_targets": list(isolation.connected_targets),
            },
            cleanup_error=cleanup_error,
        )
    except Exception as exc:
        logger.warning(
            "resource ledger update failed for isolation network %s: %s",
            getattr(isolation, "name", "(unknown)"),
            exc,
        )

def _teardown_agent_isolation_network(
    *,
    storage: RunStorage,
    run_id: str,
    agent: AgentInstance,
    trial: Trial,
    isolation: AgentIsolationNetwork,
) -> None:
    """Tear down one isolation bridge and persist its cleanup outcome.

    Both legacy execution paths create ``AgentIsolationNetwork`` handles, so
    cleanup policy must live in one place: call Docker teardown, translate the
    daemon result into ``released`` or ``cleanup_failed``, and append the
    canonical ResourceLedger entry. Exceptions are logged and recorded instead
    of escaping because network cleanup should not prevent later target/client
    teardown from running.
    """

    isolation_released = False
    cleanup_error = None
    try:
        isolation_released = isolation.teardown()
    except Exception as exc:
        cleanup_error = str(exc)
        logger.warning("Agent isolation teardown failed: %s", exc)
    _record_agent_isolation_network_resource(
        storage=storage,
        run_id=run_id,
        agent=agent,
        trial=trial,
        isolation=isolation,
        status="released" if isolation_released else "cleanup_failed",
        cleanup_error=(
            cleanup_error
            if cleanup_error is not None
            else (
                None
                if isolation_released
                else "docker network rm did not confirm removal"
            )
        ),
    )

def _record_container_proxy_resource(
    *,
    storage: RunStorage,
    run_id: str,
    agent: AgentInstance,
    trial: Trial,
    proxy: ContainerProxyInstance,
    status: str,
    cleanup_error: str | None = None,
) -> None:
    """Best-effort ledger record for a trial-local container proxy process.

    The proxy is not a Docker container, network, or volume; it is a process
    launched inside the agent container. Recording it still matters because it
    owns a local port, runtime config path, log directory, and cleanup action.
    Metadata comes from ``ContainerProxyInstance.resource_metadata()`` so the
    ledger never serializes upstream API keys, auth headers, or proxy config
    contents.
    """

    canonical_trial_id = plan_trial_id(agent, trial)
    container_name = str(getattr(proxy.container, "name", "") or "")
    external_id = (
        f"{container_name}:{proxy.pid}"
        if container_name and proxy.pid
        else str(proxy.pid or container_name)
    )
    cleanup_action = (
        f"docker exec {container_name} kill -TERM {proxy.pid}"
        if container_name and proxy.pid
        else f"kill -TERM {proxy.pid}"
    )

    try:
        TrialRuntimeSession(
            run_dir=storage.run_dir,
            trial_id=canonical_trial_id,
        ).record_resource(
            run_id=run_id,
            resource_id=f"container_proxy:{canonical_trial_id}",
            kind="container_proxy",
            provider="docker_exec",
            external_id=external_id,
            status=status,
            cleanup_action=cleanup_action,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata=proxy.resource_metadata(),
            cleanup_error=cleanup_error,
        )
    except Exception as exc:
        logger.warning(
            "resource ledger update failed for container proxy %s: %s",
            getattr(proxy, "pid", "(unknown)"),
            exc,
        )

def _stop_container_proxy_resource(
    *,
    storage: RunStorage,
    run_id: str,
    agent: AgentInstance,
    trial: Trial,
    proxy: ContainerProxyInstance,
    artifact_dir: Path,
) -> None:
    """Stop a container proxy and persist the cleanup outcome.

    This wrapper preserves the old runtime behavior: a ``proxy.stop`` exception
    still propagates to the caller. The difference is that the failure is first
    recorded as ``cleanup_failed`` in the canonical ResourceLedger so inspect
    and GC can explain partial cleanup state.
    """

    try:
        proxy.stop(artifact_dir=artifact_dir)
    except Exception as exc:
        _record_container_proxy_resource(
            storage=storage,
            run_id=run_id,
            agent=agent,
            trial=trial,
            proxy=proxy,
            status="cleanup_failed",
            cleanup_error=str(exc),
        )
        raise
    _record_container_proxy_resource(
        storage=storage,
        run_id=run_id,
        agent=agent,
        trial=trial,
        proxy=proxy,
        status="released",
    )

def _target_runtime_resource_metadata(
    *,
    chal_id: str,
    target_data: dict[str, Any],
) -> dict[str, object]:
    """Build ResourceLedger-safe metadata for one target_server runtime.

    Target launch metadata may contain prompt-facing connection strings,
    scoring config, debug service/container maps, and other benchmark-specific
    details. Cleanup/debug ledgers only need stable identifiers: challenge id,
    target_server run/project/network identifiers, and service names. Keeping a
    positive allowlist here prevents ResourceLedger from becoming a second copy
    of target prompt material or target_server debug payloads.
    """

    runtime = target_data.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    target_info = target_data.get("target_info", {})
    target_info = target_info if isinstance(target_info, dict) else {}
    service_names = sorted(str(name) for name in target_info)
    public_service_names = sorted(
        str(name)
        for name, service in target_info.items()
        if isinstance(service, dict) and service.get("external_port") is not None
    )
    metadata: dict[str, object] = {
        "challenge_id": chal_id,
        "public_service_names": public_service_names,
        "service_names": service_names,
    }
    for output_key, runtime_key in (
        ("target_run_id", "run_id"),
        ("project_name", "project_name"),
        ("network_name", "network_name"),
        ("network_subnet", "network_subnet"),
        ("network_gateway", "network_gateway"),
    ):
        value = runtime.get(runtime_key)
        if value not in (None, ""):
            metadata[output_key] = str(value)
    target_status = target_data.get("target_status")
    if target_status not in (None, ""):
        metadata["target_status"] = str(target_status)
    return dict(sorted(metadata.items()))

def _record_target_runtime_resource(
    *,
    storage: RunStorage,
    run_id: str,
    agent: AgentInstance,
    trial: Trial,
    chal_id: str,
    target_data: dict[str, Any],
    status: str,
    cleanup_error: str | None = None,
) -> None:
    """Best-effort ledger record for one target_server runtime/project.

    ``created`` records come from target launch metadata. Cleanup records should
    use ``_target_teardown_resource_status`` on the ``ChallengeClient`` teardown
    outcome so the ledger only says ``released`` when the target backend proved
    deletion, and otherwise keeps ``cleanup_requested`` or ``cleanup_failed``.
    """

    canonical_trial_id = plan_trial_id(agent, trial)
    metadata = _target_runtime_resource_metadata(chal_id=chal_id, target_data=target_data)
    target_run_id = str(metadata.get("target_run_id") or "")
    project_name = str(metadata.get("project_name") or "")
    external_id = project_name or target_run_id or chal_id
    cleanup_action = f"target_server DELETE /launch/{chal_id}"
    if target_run_id:
        cleanup_action = f"{cleanup_action}?run_id={target_run_id}"

    try:
        TrialRuntimeSession(
            run_dir=storage.run_dir,
            trial_id=canonical_trial_id,
        ).record_resource(
            run_id=run_id,
            resource_id=f"target_runtime:{canonical_trial_id}:{chal_id}",
            kind="target_runtime",
            provider="target_server",
            external_id=external_id,
            status=status,
            cleanup_action=cleanup_action,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata=metadata,
            cleanup_error=cleanup_error,
        )
    except Exception as exc:
        logger.warning(
            "resource ledger update failed for target runtime %s: %s",
            chal_id,
            exc,
        )

def _target_teardown_resource_status(teardown_result: object) -> tuple[str, str | None]:
    """Translate a target teardown outcome into ResourceLedger fields.

    ``ChallengeClient.finish_challenge`` now returns a small outcome object,
    but this helper is deliberately permissive because out-of-tree target
    clients may still return ``None`` or a simple bool during the migration.
    Unknown results stay as ``cleanup_requested`` rather than being promoted to
    ``released`` without proof.
    """
    if teardown_result is None:
        return "cleanup_requested", None
    if isinstance(teardown_result, bool):
        return ("released", None) if teardown_result else (
            "cleanup_failed",
            "target teardown returned False",
        )

    status = str(getattr(teardown_result, "status", "") or "").strip()
    cleanup_error = getattr(teardown_result, "error", None)
    cleanup_error = str(cleanup_error) if cleanup_error not in (None, "") else None
    if status in {"released", "cleanup_requested", "cleanup_failed"}:
        return status, cleanup_error

    succeeded = getattr(teardown_result, "succeeded", None)
    if succeeded is True:
        return "released", cleanup_error
    if succeeded is False:
        return "cleanup_failed", cleanup_error or "target teardown failed"
    return "cleanup_requested", cleanup_error
