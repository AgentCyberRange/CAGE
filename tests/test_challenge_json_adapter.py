from __future__ import annotations

import json

from cage.target.adapters.base import BenchmarkSource
from cage.target.adapters.challenge_json import ChallengeJsonAdapter
from cage.target.scope import resolve_target_scope


_COMPOSE_WITH_PRIVATE_NETWORK = """
services:
  target:
    image: example/target:latest
    ports:
      - "9090"
    networks:
      - target_network
networks:
  target_network: {}
"""


def _write_challenge(tmp_path, challenge_json: dict, *, compose: str | None = None):
    challenge_root = tmp_path / "chal-1"
    challenge_root.mkdir()
    (tmp_path / "bench.json").write_text(
        json.dumps({"chal-1": {"path": "chal-1"}}), encoding="utf-8"
    )
    challenge_root.joinpath("challenge.json").write_text(
        json.dumps(challenge_json), encoding="utf-8"
    )
    if compose is not None:
        challenge_root.joinpath("docker-compose.yml").write_text(compose, encoding="utf-8")
    adapter = ChallengeJsonAdapter()
    challenges = adapter.discover(
        BenchmarkSource(adapter_kind="challenge_json", root=tmp_path)
    )
    return adapter, challenges["chal-1"]


def test_nested_source_fields_override_flat_challenge_fields(tmp_path):
    """Nested source_fields are the canonical internal fields when present."""
    challenge_root = tmp_path / "range-1"
    challenge_root.mkdir()
    index_path = tmp_path / "postexp.json"
    index_path.write_text(
        json.dumps({"pb-postexp-range-1": {"path": "range-1"}}),
        encoding="utf-8",
    )
    challenge_root.joinpath("challenge.json").write_text(
        json.dumps(
            {
                "adapter_kind": "challenge_json",
                "benchmark_family": "agent_pentest_bench",
                "task_profile": "post_exploitation",
                "marker_stages": [
                    {
                        "id": "report_user_shell",
                        "service": "1_report",
                        "marker_path": "${VERIFY_USER_MARKER_PATH:-/tmp/range1_user_shell_marker",
                    }
                ],
                "source_fields": {
                    "marker_stages": [
                        {
                            "id": "report_user_shell",
                            "service": "1_report",
                            "marker_path": "/tmp/range1_user_shell_marker",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    challenges = ChallengeJsonAdapter().discover(
        BenchmarkSource(adapter_kind="challenge_json", root=tmp_path)
    )

    source_fields = challenges["pb-postexp-range-1"]["source_fields"]
    assert source_fields["marker_stages"][0]["marker_path"] == "/tmp/range1_user_shell_marker"


def test_declared_target_fields_flow_without_benchmark_name_branch(tmp_path):
    """A challenge that declares its target runtime fields drives launch wiring
    entirely from data — the framework never keys off the benchmark name."""
    adapter, challenge = _write_challenge(
        tmp_path,
        {
            "adapter_kind": "challenge_json",
            "benchmark_family": "anything",
            "target_scope": "per_agent",
            "network_mode": "compose_project_local",
            "agent_network": "target_network",
            "project_local_subnet_pool": "172.31.0.0/16",
            "project_local_subnet_prefix": 24,
            "runtime_scoring": {"kind": "http_poll", "service": "target", "port": 9091, "path": "/done"},
        },
        compose=_COMPOSE_WITH_PRIVATE_NETWORK,
    )

    assert resolve_target_scope(chal_data=challenge, runtime_args={}) == "per_agent"
    patches = adapter.build_launch_spec(challenge).runtime_patches
    assert patches["network_mode"] == "compose_project_local"
    assert patches["agent_network"] == "target_network"
    assert patches["project_local_subnet_pool"] == "172.31.0.0/16"
    assert patches["project_local_subnet_prefix"] == 24
    assert challenge["source_fields"]["runtime_scoring"]["kind"] == "http_poll"


def test_benchmark_family_alone_injects_nothing(tmp_path):
    """Regression guard for the deleted Layer-1 name branches: a challenge whose
    only benchmark signal is ``benchmark_family`` (no declared target fields)
    must get ZERO network/scope/scoring injection."""
    adapter, challenge = _write_challenge(
        tmp_path,
        {
            "adapter_kind": "challenge_json",
            "benchmark_family": "cvebench",
        },
        compose=_COMPOSE_WITH_PRIVATE_NETWORK,
    )

    # No deprecation shim: family alone no longer forces per_agent.
    assert resolve_target_scope(chal_data=challenge, runtime_args={}) == "per_challenge"
    patches = adapter.build_launch_spec(challenge).runtime_patches
    assert "network_mode" not in patches
    assert "agent_network" not in patches
    assert "project_local_subnet_pool" not in patches
    # No derived runtime_scoring either.
    assert "runtime_scoring" not in challenge["source_fields"]
