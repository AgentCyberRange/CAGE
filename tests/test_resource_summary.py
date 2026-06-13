"""Tests for canonical ResourceLedger summary projections."""
from __future__ import annotations

from pathlib import Path

from cage.artifacts.resources import ResourceLedgerWriter
from cage.gc.summary import summarize_resource_ledger


def test_summarize_resource_ledger_uses_latest_record_per_resource(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = ResourceLedgerWriter(run_dir)
    writer.append_resource(
        run_id="run",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:00Z",
        trial_id="sample/pass_1",
    )
    writer.append_resource(
        run_id="run",
        resource_id="docker_network:trial-net",
        kind="docker_network",
        provider="docker",
        external_id="trial-net",
        status="released",
        cleanup_action="docker network rm trial-net",
        timestamp="2026-06-05T00:00:01Z",
        trial_id="sample/pass_1",
    )
    writer.append_resource(
        run_id="run",
        resource_id="target_runtime:sample/pass_1:pb-siyucms",
        kind="target_runtime",
        provider="target_server",
        external_id="cage_pb_siyucms",
        status="cleanup_failed",
        cleanup_action="target_server DELETE /launch/pb-siyucms",
        timestamp="2026-06-05T00:00:02Z",
        trial_id="sample/pass_1",
        cleanup_error="target_server refused teardown",
    )
    writer.append_resource(
        run_id="run",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="released",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:03Z",
        trial_id="sample/pass_1",
    )

    summary = summarize_resource_ledger(run_dir)

    assert summary.to_mapping() == {
        "total": 3,
        "active": 0,
        "released": 2,
        "cleanup_failed": 1,
        "by_kind": {
            "docker_container": 1,
            "docker_network": 1,
            "target_runtime": 1,
        },
        "by_status": {
            "cleanup_failed": 1,
            "released": 2,
        },
        "items": [
            {
                "resource_id": "docker_container:agent-a",
                "kind": "docker_container",
                "provider": "docker",
                "external_id": "agent-a",
                "status": "released",
                "trial_id": "sample/pass_1",
                "cleanup_error": "",
            },
            {
                "resource_id": "docker_network:trial-net",
                "kind": "docker_network",
                "provider": "docker",
                "external_id": "trial-net",
                "status": "released",
                "trial_id": "sample/pass_1",
                "cleanup_error": "",
            },
            {
                "resource_id": "target_runtime:sample/pass_1:pb-siyucms",
                "kind": "target_runtime",
                "provider": "target_server",
                "external_id": "cage_pb_siyucms",
                "status": "cleanup_failed",
                "trial_id": "sample/pass_1",
                "cleanup_error": "target_server refused teardown",
            },
        ],
    }
