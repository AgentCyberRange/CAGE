"""Tests for ResourceLedger cleanup planning."""
from __future__ import annotations

from cage.experiment.model import ResourceRecord
from cage.gc.plan import build_resource_cleanup_plan


def _resource(
    *,
    resource_id: str,
    kind: str,
    provider: str = "docker",
    external_id: str = "",
    status: str = "started",
    namespace: str = "",
) -> ResourceRecord:
    """Create a ResourceRecord for cleanup-plan tests."""

    labels = {}
    if namespace:
        labels["cage.target.namespace"] = namespace
    return ResourceRecord(
        schema_version="resource_record.v1",
        record_id=f"res-{resource_id}",
        run_id="run-cleanup",
        resource_id=resource_id,
        kind=kind,
        provider=provider,
        external_id=external_id,
        status=status,
        cleanup_action="",
        timestamp="2026-06-05T00:00:00Z",
        metadata={"labels": labels} if labels else {},
    )


def test_build_resource_cleanup_plan_filters_and_groups_docker_resources() -> None:
    records = (
        _resource(
            resource_id="docker_container:agent-a",
            kind="docker_container",
            external_id="agent-a",
            namespace="ns-a",
        ),
        _resource(
            resource_id="docker_container:agent-a-alias",
            kind="docker_container",
            external_id="agent-a",
            namespace="ns-a",
        ),
        _resource(
            resource_id="docker_network:trial-net",
            kind="docker_network",
            external_id="trial-net",
            namespace="ns-b",
        ),
        _resource(
            resource_id="docker_volume:released-vol",
            kind="docker_volume",
            external_id="released-vol",
            status="released",
            namespace="ns-a",
        ),
        _resource(
            resource_id="target_runtime:sample/pass_1:pb-siyucms",
            kind="target_runtime",
            provider="target_server",
            external_id="cage_pb_siyucms",
            status="cleanup_failed",
            namespace="ns-a",
        ),
        _resource(
            resource_id="docker_volume:missing-external-id",
            kind="docker_volume",
            external_id="",
            namespace="ns-a",
        ),
    )

    plan = build_resource_cleanup_plan(records, namespace="ns-a")

    assert plan.counts() == {"containers": 2, "networks": 0, "volumes": 1}
    assert plan.docker_ids() == {
        "containers": ("agent-a",),
        "networks": (),
        "volumes": (),
    }
    assert [
        record.resource_id
        for record in plan.cleanup_records_by_key()["containers"]
    ] == ["docker_container:agent-a"]
