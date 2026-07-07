"""Live integration tests for per-agent target allocation.

These tests talk to a real target_server server and tear targets up/down
through the HTTP API. They are skipped unless the configured server is
reachable.

Run with::

    pytest tests/test_target_allocation_live.py -v -s
"""

from __future__ import annotations

import ipaddress
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest

# Skip the whole module when ``requests`` is unavailable — same fallback the
# RemoteBackend uses.
requests = pytest.importorskip("requests")

from cage.target.client import ChallengeClient, ChallengeClientConfig  # noqa: E402

SERVER_URL = os.getenv("CAGE_TEST_TARGET_SERVER_URL", "http://192.0.2.10:8000")
CHAL_ID = os.getenv("CAGE_TEST_CHAL_ID", "cvb-CVE-2023-37999")
BENCHMARK_FAMILY = os.getenv("CAGE_TEST_CHAL_FAMILY", "cvebench")


def _server_reachable() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/openapi.json", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=f"target_server server at {SERVER_URL} not reachable",
)


def _make_client(label: str) -> ChallengeClient:
    return ChallengeClient(
        ChallengeClientConfig(
            challenges={CHAL_ID: {"benchmark_family": BENCHMARK_FAMILY}},
            run_mode="remote",
            server_url=SERVER_URL,
        ),
        logger=logging.getLogger(f"target-test.{label}"),
    )


def _runtime(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("runtime", {})


def test_two_clients_get_distinct_runtimes():
    """Two independent clients launching the same challenge must each get
    their own run_id / network_name / network_subnet under per_agent scope.

    This is the cornerstone of multi-agent isolation: if it fails, two
    agents evaluating the same task can see each other's side effects.
    """
    a = _make_client("A")
    b = _make_client("B")
    try:
        rec_a = _runtime(a.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"}))
        rec_b = _runtime(b.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"}))

        assert rec_a["run_id"] != rec_b["run_id"], rec_a["run_id"]
        assert rec_a["network_name"] != rec_b["network_name"]
        assert rec_a["network_subnet"] != rec_b["network_subnet"]

        net_a = ipaddress.ip_network(rec_a["network_subnet"], strict=False)
        net_b = ipaddress.ip_network(rec_b["network_subnet"], strict=False)
        assert not net_a.overlaps(net_b), f"subnets overlap: {net_a} & {net_b}"
    finally:
        for client in (a, b):
            try:
                client.finish_challenge(CHAL_ID)
            except Exception:
                pass


def test_teardown_a_does_not_affect_b():
    """Tearing down A's per_agent instance must leave B's intact.

    We verify by tearing down A, then launching a probe C: if C inherits A's
    run_id the server short-circuited (didn't actually teardown), which
    would break per-agent isolation.
    """
    a = _make_client("A")
    b = _make_client("B")
    probe = _make_client("probe")
    try:
        rec_a = _runtime(a.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"}))
        rec_b = _runtime(b.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"}))
        original_a_run_id = rec_a["run_id"]
        original_b_run_id = rec_b["run_id"]

        a.finish_challenge(CHAL_ID)

        rec_probe = _runtime(
            probe.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"}),
        )
        assert rec_probe["run_id"] != original_a_run_id, (
            "Probe inherited A's run_id — server didn't actually tear A down."
        )
        assert rec_probe["run_id"] != original_b_run_id, (
            "Probe inherited B's run_id — should be a fresh instance."
        )
        probe.finish_challenge(CHAL_ID)

        # B should still be runnable (cache still maps to its run_id).
        cached = _runtime(b.get_challenge_data(CHAL_ID))
        assert cached["run_id"] == original_b_run_id, (
            "B's cached run_id was clobbered."
        )
    finally:
        for client in (a, b, probe):
            try:
                client.finish_challenge(CHAL_ID)
            except Exception:
                pass


def test_three_concurrent_launches_get_non_overlapping_subnets():
    """IPAM stress: 3 parallel per_agent launches → 3 disjoint /N subnets."""
    n = 3
    clients: list[ChallengeClient] = []

    def _one(i: int) -> tuple[ChallengeClient, dict[str, Any]]:
        c = _make_client(f"stress.{i}")
        rec = _runtime(c.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"}))
        return c, rec

    try:
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(_one, i) for i in range(n)]
            results = [fut.result() for fut in as_completed(futs)]

        clients = [c for c, _ in results]
        runtimes = [r for _, r in results]

        run_ids = {r["run_id"] for r in runtimes}
        assert len(run_ids) == n, f"run_id collision: {run_ids}"

        nets = [ipaddress.ip_network(r["network_subnet"], strict=False) for r in runtimes]
        for i in range(n):
            for j in range(i + 1, n):
                assert not nets[i].overlaps(nets[j]), (
                    f"subnet overlap between concurrent launches: {nets[i]} & {nets[j]}"
                )
    finally:
        for c in clients:
            try:
                c.finish_challenge(CHAL_ID)
            except Exception:
                pass


def _docker_available() -> bool:
    import subprocess

    try:
        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _docker_available(), reason="docker daemon not reachable")
def test_cage_isolation_bridge_blocks_db_reachable_target():
    """End-to-end isolation contract:

    1. Launch a real cvebench target.
    2. Create a cage-private bridge wired only to the ``target`` container.
    3. From a fresh probe attached to that bridge, ``target`` must be
       reachable AND every internal service (``db``, ``secrets_init``)
       must be unresolvable.
    """
    import subprocess

    from cage.target.provisioning import build_agent_isolation_network

    client = _make_client("isolation-e2e")
    rec = client.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"})
    target_data = rec  # rec is the challenge dict with runtime/target_info populated

    isolation = None
    try:
        isolation = build_agent_isolation_network("isolation-e2e-test", target_data)
        debug_network = (
            target_data.get("runtime", {}).get("debug", {}).get("network", {})
        )
        target_info_keys = list(target_data.get("target_info", {}).keys())
        assert isolation is not None, (
            "Expected to build a cage-private bridge; server response was: "
            f"runtime.debug.network={debug_network}, "
            f"target_info_keys={target_info_keys}"
        )

        probe = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", isolation.name,
                "alpine:3.19",
                "sh", "-c",
                # We resolve hostnames and run TCP probes from inside the
                # bridge. `target` must resolve + accept TCP; `db` must NOT
                # resolve. The `||` cascade keeps a non-zero exit visible.
                "getent hosts target > /tmp/t.txt 2>&1 || echo TARGET_NXDOMAIN > /tmp/t.txt;"
                "getent hosts db > /tmp/d.txt 2>&1 || echo DB_NXDOMAIN > /tmp/d.txt;"
                "(nc -w 2 -zv target 9090 2>&1 || echo TARGET_TCP_FAIL) > /tmp/tt.txt;"
                "(nc -w 2 -zv db 5432 2>&1 || echo DB_TCP_FAIL) > /tmp/dd.txt;"
                "echo ===; cat /tmp/t.txt; echo ===; cat /tmp/d.txt;"
                "echo ===; cat /tmp/tt.txt; echo ===; cat /tmp/dd.txt",
            ],
            capture_output=True, text=True, timeout=60,
        )
        out = probe.stdout

        assert "TARGET_NXDOMAIN" not in out, f"target should resolve, got:\n{out}"
        assert "DB_NXDOMAIN" in out or "bad address 'db'" in out, (
            f"db must NOT resolve from inside the cage bridge:\n{out}"
        )
        assert "open" in out, f"target:9090 must accept TCP:\n{out}"
        # `nc -zv db 5432` on Alpine prints either "bad address 'db'" (no
        # DNS) or fails the connect; either signals db is unreachable.
        assert "DB_TCP_FAIL" in out or "bad address 'db'" in out, (
            f"db:5432 must NOT be reachable from the cage bridge:\n{out}"
        )
    finally:
        if isolation is not None:
            try:
                isolation.teardown()
            except Exception:
                pass
        try:
            client.finish_challenge(CHAL_ID)
        except Exception:
            pass


def test_per_agent_scope_propagates_as_query_param():
    """The remote backend must serialize target_scope into the launch URL."""
    captured: dict[str, Any] = {}
    original_get = requests.get

    def _spy(url, params=None, timeout=None, **kw):
        captured["url"] = url
        captured["params"] = dict(params or {})
        return original_get(url, params=params, timeout=timeout, **kw)

    requests.get = _spy
    client = _make_client("scope-spy")
    try:
        client.get_challenge_data(CHAL_ID, runtime_args={"target_scope": "per_agent"})
    finally:
        requests.get = original_get
        try:
            client.finish_challenge(CHAL_ID)
        except Exception:
            pass

    assert captured["params"].get("target_scope") == "per_agent"
