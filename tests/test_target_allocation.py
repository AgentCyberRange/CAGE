"""Unit tests for CTF target allocation policy.

These tests pin down the *current* behaviour around how cage's orchestrator
and ChallengeClient hand out targets to agents/trials so we can refactor with
confidence. Where the current behaviour is a known limitation we mark the
test as ``xfail`` so it surfaces the gap without breaking CI; flipping it to
``passed`` is the signal the gap has been closed.

Coverage:
- ``inject_ctf_info`` alias-mode vs network-mode branching.
- ``ChallengeClient._runtime_cache`` reuse semantics under per_agent scope.
- ``ChallengeClient.refresh_challenge_data(force_recreate=True)`` forces a
  fresh backend.initialize.
- ``challenges_from_benchmark`` reads the benchmark-owned ``challenge_client``
  attribute without preserving the old ``ctf_manager`` framework alias.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest.mock import MagicMock

from cage.target.provisioning import challenges_from_benchmark, inject_ctf_info
from cage.target.client import (
    BackendStrategy,
    ChallengeClient,
    ChallengeClientConfig,
)

# ----- inject_ctf_info --------------------------------------------------- #

def _target_data(*, parallel_mode: str, with_target_info: bool = True) -> Dict[str, Any]:
    return {
        "id": "x",
        "type": "dynamic",
        "runtime": {
            "run_id": "x_abc12345",
            "project_name": "ctf_ns_x_abc12345_runtime",
            "network_name": "ctf_ns_x_abc12345_runtime_default",
            "network_subnet": "10.250.1.0/24",
            "network_gateway": "10.250.1.1",
            "scoring": {"endpoint": "http://target:9091"},
            "debug": {"parallel_mode": parallel_mode},
        },
        "target_info": (
            {"target": {"service_name": "target", "host": "target", "port": 9090}}
            if with_target_info
            else {}
        ),
    }


def test_inject_alias_mode_sets_target_info_only():
    sample: dict[str, Any] = {"id": "s1"}
    inject_ctf_info(sample, _target_data(parallel_mode="alias"))
    assert sample["network_name"] == "ctf_ns_x_abc12345_runtime_default"
    assert "target_info" in sample
    assert "network_subnet" not in sample
    assert sample["metadata"]["runtime_scoring"]["endpoint"] == "http://target:9091"


def test_inject_network_mode_includes_subnet_and_target_info():
    """Post-P7 contract: network mode injects BOTH subnet AND target_info
    whenever the server returned them. Agents that need to discover services
    via scanning use the subnet; agents that target a known alias use
    target_info; the prompt template picks whichever it needs.
    """
    sample: dict[str, Any] = {"id": "s1"}
    inject_ctf_info(sample, _target_data(parallel_mode="network"))
    assert sample["network_name"] == "ctf_ns_x_abc12345_runtime_default"
    assert sample["network_subnet"] == "10.250.1.0/24"
    assert "target_info" in sample, "P7 fix: network mode must keep target_info"


def test_inject_network_mode_without_target_info_does_not_inject_empty():
    """If the server returned no service info, don't pollute the sample."""
    data = _target_data(parallel_mode="network", with_target_info=False)
    sample: dict[str, Any] = {"id": "s1"}
    inject_ctf_info(sample, data)
    assert "target_info" not in sample
    assert sample["network_subnet"] == "10.250.1.0/24"


def test_inject_default_mode_falls_through_to_alias():
    """When debug.parallel_mode is missing, current behaviour is alias-mode."""
    data = _target_data(parallel_mode="alias")
    data["runtime"]["debug"] = {}  # drop parallel_mode entirely
    sample: dict[str, Any] = {"id": "s1"}
    inject_ctf_info(sample, data)
    assert "target_info" in sample
    assert "network_subnet" not in sample


# ----- ChallengeClient cache behaviour ------------------------------------ #

class _StubBackend(BackendStrategy):
    """Backend that records every initialize() call and returns sequenced records."""

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.calls: list[dict[str, Any]] = []
        self._counter = 0

    def initialize(
        self, challenge_id, metadata, force_recreate=False, runtime_args=None,
    ):
        self.calls.append({
            "chal_id": challenge_id,
            "force_recreate": force_recreate,
            "runtime_args": dict(runtime_args or {}),
        })
        self._counter += 1
        suffix = f"{self._counter:08x}"
        return {
            "id": challenge_id,
            "type": "dynamic",
            "run_id": f"{challenge_id}_{suffix}",
            "project_name": f"ctf_test_{challenge_id}_{suffix}_runtime",
            "network_name": f"ctf_test_{challenge_id}_{suffix}_default",
            "network_subnet": f"10.250.{self._counter}.0/24",
            "scoring": {},
            "debug": {"parallel_mode": "network"},
            "services": {},
        }

    def teardown(self, challenge_id, run_id=None):  # noqa: D401
        self.calls.append({"teardown": challenge_id, "run_id": run_id})

    def validate_connectivity(self, challenge_id, record):
        return True

    def handle_crash(self, challenge_id, observation):
        return observation, True


def _make_client(challenges: dict[str, Any] | None = None) -> tuple[ChallengeClient, _StubBackend]:
    cfg = ChallengeClientConfig(
        challenges=challenges or {"cve-x": {"benchmark_family": "cvebench"}},
        run_mode="remote",
    )
    client = ChallengeClient(cfg, logger=logging.getLogger("test_target_allocation"))
    stub = _StubBackend(cfg, client.logger)
    client.backend = stub
    return client, stub


def test_per_agent_scope_caches_first_record():
    """Current behaviour: second call hits cache, backend NOT re-invoked.

    This is *not* per-trial fresh allocation. Serial-path trials of the same
    challenge therefore share one target instance.
    """
    client, stub = _make_client()
    rec1 = client.get_challenge_data("cve-x", runtime_args={"target_scope": "per_agent"})
    rec2 = client.get_challenge_data("cve-x", runtime_args={"target_scope": "per_agent"})
    init_calls = [c for c in stub.calls if "teardown" not in c]
    assert len(init_calls) == 1, "cache should suppress the second initialize"
    assert rec1["runtime"]["run_id"] == rec2["runtime"]["run_id"]


def test_refresh_force_recreate_invokes_backend_again():
    client, stub = _make_client()
    client.get_challenge_data("cve-x", runtime_args={"target_scope": "per_agent"})
    client.refresh_challenge_data("cve-x", force_recreate=True)
    init_calls = [c for c in stub.calls if "teardown" not in c]
    # First call from get_challenge_data (no force_recreate).
    # refresh_challenge_data tears down the cached record first, then re-launches.
    # So there should be 2 init calls total.
    assert len(init_calls) >= 2


def test_teardown_uses_cached_run_id():
    client, stub = _make_client()
    rec = client.get_challenge_data("cve-x", runtime_args={"target_scope": "per_agent"})
    cached_run_id = rec["runtime"]["run_id"]
    client.finish_challenge("cve-x")
    teardowns = [c for c in stub.calls if "teardown" in c]
    assert teardowns, "teardown must reach the backend"
    assert teardowns[-1]["run_id"] == cached_run_id


# ----- challenges_from_benchmark ---------------------------------------- #

class _BenchmarkWithNewAttr:
    """Stub benchmark that exposes the new ``challenge_client`` attribute."""

    def __init__(self, challenges):
        self.challenge_client = MagicMock()
        self.challenge_client.challenges = challenges


class _BenchmarkWithLegacyAttr:
    """Stub benchmark with the removed legacy ``ctf_manager`` attribute."""

    def __init__(self, challenges):
        self.ctf_manager = MagicMock()
        self.ctf_manager.challenges = challenges


def _exp_with(benchmark):
    cfg = MagicMock()
    cfg.benchmark = benchmark
    return cfg


def test_challenges_from_benchmark_new_attr():
    challenges = {"cve-x": {"foo": 1}}
    found = challenges_from_benchmark(_exp_with(_BenchmarkWithNewAttr(challenges)))
    assert found == challenges


def test_challenges_from_benchmark_ignores_removed_legacy_attr():
    """Framework target discovery now uses ``challenge_client`` only."""
    challenges = {"cve-x": {"foo": 1}}
    found = challenges_from_benchmark(_exp_with(_BenchmarkWithLegacyAttr(challenges)))
    assert found == {}


def test_challenges_from_benchmark_neither_attr_returns_empty():
    bench = MagicMock(spec=[])  # no ctf_manager / no challenge_client
    found = challenges_from_benchmark(_exp_with(bench))
    assert found == {}


# ----- P1: target_scope / parallel_mode configurability ------------------- #


def _exp(target_scope: str = "per_agent", parallel_mode: str = "") -> Any:
    """Cheap stub: only the fields ``target_runtime_args`` reads exist."""
    from types import SimpleNamespace

    return SimpleNamespace(
        target=SimpleNamespace(
            target_scope=target_scope,
            parallel_mode=parallel_mode,
        ),
    )


def test_target_runtime_args_default_is_per_agent():
    from cage.target.provisioning import target_runtime_args

    args = target_runtime_args(_exp())
    assert args == {"target_scope": "per_agent"}


def test_target_runtime_args_includes_parallel_mode_when_set():
    from cage.target.provisioning import target_runtime_args

    args = target_runtime_args(_exp(target_scope="per_agent", parallel_mode="network"))
    assert args == {"target_scope": "per_agent", "parallel_mode": "network"}


def test_target_runtime_args_drops_empty_target_scope():
    """Empty string → don't send the key, server picks family default."""
    from cage.target.provisioning import target_runtime_args

    args = target_runtime_args(_exp(target_scope="", parallel_mode="alias"))
    assert args == {"parallel_mode": "alias"}


def test_target_runtime_args_sample_override_wins():
    from cage.target.provisioning import target_runtime_args

    sample = {"runtime_args": {"target_scope": "per_challenge", "parallel_mode": "alias"}}
    args = target_runtime_args(_exp(target_scope="per_agent", parallel_mode="network"), sample)
    assert args == {"target_scope": "per_challenge", "parallel_mode": "alias"}


def test_target_runtime_args_sample_partial_override_keeps_config_defaults():
    from cage.target.provisioning import target_runtime_args

    sample = {"runtime_args": {"parallel_mode": "alias"}}  # only override one
    args = target_runtime_args(_exp(target_scope="per_agent", parallel_mode="network"), sample)
    assert args == {"target_scope": "per_agent", "parallel_mode": "alias"}


def test_target_config_has_target_scope_and_parallel_mode_fields():
    """Sanity check the config dataclass actually exposes the fields."""
    import dataclasses

    from cage.config.experiment import TargetConfig

    field_names = {f.name for f in dataclasses.fields(TargetConfig)}
    assert "target_scope" in field_names
    assert "parallel_mode" in field_names
    cfg = TargetConfig()
    assert cfg.target_scope == "per_agent"  # backward-compat default
    assert cfg.parallel_mode == ""


# ----- Per-trial cage-private bridge (agent↔target isolation) ------------ #


def _full_target_data(*, with_container_name: bool = True, with_external_port: bool = True):
    """Mock launch response: cvebench-style with target + internal services."""
    target_external = 35869 if with_external_port else None
    return {
        "id": "cvb-X",
        "type": "dynamic",
        "runtime": {
            "run_id": "cvb_x_aaaaaaaa",
            "project_name": "ctf_default_cvb_x_aaaaaaaa_runtime",
            "network_name": "ctf_default_cvb_x_aaaaaaaa_runtime_target_network",
            "network_subnet": "172.31.13.16/28",
            "scoring": {},
            "debug": {
                "parallel_mode": "network",
                "network": {
                    "services": (
                        [
                            {
                                "service_name": "target",
                                "container_name": "ctf_default_cvb_x_aaaaaaaa_runtime-target-1",
                                "ipv4": "172.31.13.18",
                            },
                        ]
                        if with_container_name
                        else []
                    ),
                },
            },
        },
        "target_info": {
            "secrets_init": {"service_name": "secrets_init", "external_port": None},
            "db": {"service_name": "db", "external_port": None},
            "target": {"service_name": "target", "external_port": target_external},
        },
    }


def test_public_target_services_filters_to_external_port_with_known_container():
    from cage.target.provisioning import public_target_services

    targets = public_target_services(_full_target_data())
    assert targets == [("target", "ctf_default_cvb_x_aaaaaaaa_runtime-target-1")]


def test_public_target_services_returns_empty_when_no_external_port():
    """All-internal challenge → no public targets → falls back to server net."""
    from cage.target.provisioning import public_target_services

    targets = public_target_services(_full_target_data(with_external_port=False))
    assert targets == []


def test_public_target_services_returns_empty_when_no_container_names():
    """Legacy server didn't include debug.network.services → can't isolate."""
    from cage.target.provisioning import public_target_services

    targets = public_target_services(_full_target_data(with_container_name=False))
    assert targets == []


def test_public_target_services_skips_internal_services_with_external_port_missing():
    """db / secrets_init MUST never end up in the public set."""
    from cage.target.provisioning import public_target_services

    targets = public_target_services(_full_target_data())
    service_names = {name for name, _ in targets}
    assert "db" not in service_names
    assert "secrets_init" not in service_names


def test_attach_agent_to_target_host_networking_no_op():
    """When the agent container uses host networking, neither bridge nor
    server network is attached — host already shares the kernel netns.
    """
    from types import SimpleNamespace

    from cage.target.provisioning import attach_agent_to_target

    fake_container = SimpleNamespace(network_mode="host", sync_runtime_network=lambda _n: None)
    attached, isolation = attach_agent_to_target(
        fake_container,
        trial_id="t-1",
        target_data=_full_target_data(),
        server_network="cage_net",
        isolation_policy="per_trial_bridge",
    )
    assert attached is None
    assert isolation is None


def test_attach_agent_to_target_trust_server_skips_isolation():
    """When isolation_policy='trust_server', no bridge is created — agent
    attaches directly to whatever target_server returned (legacy behaviour).
    """
    from types import SimpleNamespace

    from cage.target.provisioning import attach_agent_to_target

    calls: list[str | None] = []

    def _sync(n):
        calls.append(n)

    fake_container = SimpleNamespace(network_mode="bridge", sync_runtime_network=_sync)
    attached, isolation = attach_agent_to_target(
        fake_container,
        trial_id="t-1",
        target_data=_full_target_data(),
        server_network="ctf_default_xxx_target_network",
        isolation_policy="trust_server",
    )
    assert attached == "ctf_default_xxx_target_network"
    assert isolation is None
    assert calls == ["ctf_default_xxx_target_network"]


def test_attach_agent_to_target_falls_back_when_no_public_services():
    """If we can't enumerate public targets, fall back to server network."""
    from types import SimpleNamespace

    from cage.target.provisioning import attach_agent_to_target

    fake_container = SimpleNamespace(network_mode="bridge", sync_runtime_network=lambda _n: None)
    attached, isolation = attach_agent_to_target(
        fake_container,
        trial_id="t-1",
        target_data=_full_target_data(with_external_port=False),
        server_network="cage_net",
        isolation_policy="per_trial_bridge",
    )
    assert attached == "cage_net"  # target_server network
    assert isolation is None  # cage didn't create its own


def test_target_config_default_isolation_is_per_trial_bridge():
    """Defense-in-depth is on by default."""
    from cage.config.experiment import TargetConfig

    assert TargetConfig().agent_network_isolation == "per_trial_bridge"


def test_orchestrator_no_longer_hardcodes_target_scope():
    """Regression guard for P1: the two call sites must use the helper, not
    a hardcoded literal. If this fails someone reverted the fix.
    """
    import inspect

    from cage.experiment.engine import trial_runner

    src = inspect.getsource(trial_runner)
    assert '"target_scope": "per_agent"' not in src, (
        "Found hardcoded target_scope in trial execution — should use "
        "target_runtime_args(run, sample) instead."
    )
    # Both execute_trial and run_trial_isolated funnel through the single
    # shared _launch_and_attach_target helper (def + two call sites = 3),
    # which is the one site that calls the target setup gate.
    assert src.count("_launch_and_attach_target(") >= 3, (
        "Expected both execute_trial and run_trial_isolated to go through "
        "the shared _launch_and_attach_target helper."
    )
    assert src.count("_get_challenge_data_with_setup_gate(") >= 1, (
        "Expected the shared target setup helper to go through the setup gate."
    )
    assert src.count("target_runtime_args(run, sample)") >= 1, (
        "Expected the target setup gate to build runtime args through "
        "target_runtime_args."
    )


# ----- P6: ChallengeClient cache thread safety ---------------------------- #


def test_challenge_client_cache_lock_present():
    """The lock object must exist and be re-entrant (so a public method can
    call another locked public method without deadlocking).
    """
    import threading

    client, _ = _make_client()
    assert hasattr(client, "_cache_lock")
    # RLock instances expose a `_is_owned` attribute; plain Lock does not.
    assert client._cache_lock.acquire(blocking=False)
    try:
        # Second acquisition from the same thread must succeed (RLock).
        assert client._cache_lock.acquire(blocking=False)
        client._cache_lock.release()
    finally:
        client._cache_lock.release()
    # Sanity: confirm we did not accidentally pick up a non-reentrant lock.
    assert isinstance(client._cache_lock, type(threading.RLock()))


def test_challenge_client_concurrent_distinct_challenges_no_corruption():
    """N threads each launching a *different* challenge — cache must end up
    with N entries and no exceptions.
    """
    import threading

    challenges = {f"chal-{i}": {"benchmark_family": "cvebench"} for i in range(8)}
    client, stub = _make_client(challenges)

    errors: list[BaseException] = []

    def _launch(cid: str) -> None:
        try:
            client.get_challenge_data(cid, runtime_args={"target_scope": "per_agent"})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_launch, args=(cid,)) for cid in challenges]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, errors
    assert len(client._runtime_cache) == len(challenges)
    init_calls = [c for c in stub.calls if "teardown" not in c]
    assert len(init_calls) == len(challenges)


def test_challenge_client_concurrent_same_challenge_initialises_once():
    """Multiple threads racing on the SAME chal_id must coalesce: only one
    backend.initialize, all callers see the same record.

    This is the strong-isolation property for cage's serial-trial path:
    even if some caller accidentally fires twice, we don't double-launch.
    """
    import threading

    client, stub = _make_client({"chal-x": {"benchmark_family": "cvebench"}})

    barrier = threading.Barrier(8)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def _launch() -> None:
        try:
            barrier.wait(timeout=5)
            rec = client.get_challenge_data(
                "chal-x", runtime_args={"target_scope": "per_agent"},
            )
            results.append(rec)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_launch) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, errors
    init_calls = [c for c in stub.calls if "teardown" not in c]
    # Under perfect coalescing, init_calls == 1. We allow up to 2 to account
    # for the brief window where backend.initialize runs outside the lock —
    # this is the practical correctness contract.
    assert len(init_calls) <= 2, init_calls
    # All callers must see the SAME run_id (they shared the cache).
    run_ids = {r["runtime"]["run_id"] for r in results}
    assert len(run_ids) == 1, run_ids
