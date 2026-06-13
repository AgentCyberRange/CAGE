import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cage.target.server.launch import (
    _launch_challenge_impl,
    _compose_build_locked,
    _compose_up_cmd,
    _missing_build_images_detail,
    _missing_build_images,
)
from cage.target.server.launch_runtime import ComposeRuntimePlan


def test_compose_up_cmd_always_passes_no_build(tmp_path):
    # ``up`` is always ``--no-build``: the build step runs separately under
    # ``challenge_build_locks`` via ``_compose_build_locked``. Passing
    # ``--build`` here would re-trigger the classic-builder ``AlreadyExists``
    # race the lock prevents.
    cmd = _compose_up_cmd(
        chal_path=tmp_path,
        project_name="cage_test_project",
        runtime_compose_rel=".cage_runtime/docker-compose.runtime.test.yml",
    )

    assert "--no-build" in cmd
    assert "--build" not in cmd
    assert cmd[-2:] == ["--no-build", "--force-recreate"]


def test_compose_build_locked_serialises_same_chal(tmp_path):
    # Concurrent builds for the SAME chal_id must run one at a time:
    # otherwise N parallel passk trials would all hit classic builder's
    # non-atomic tag-create step and one or more would fail with
    # ``AlreadyExists: image <tag> already exists``.
    inflight = 0
    max_inflight = 0
    lock = threading.Lock()

    def fake_run(*args, **kwargs):
        nonlocal inflight, max_inflight
        with lock:
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        # Hold the "build" long enough that a non-serialised version would
        # observably overlap.
        time.sleep(0.05)
        with lock:
            inflight -= 1

        class _Res:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Res()

    def _call(chal_id):
        with patch("cage.target.server.launch.subprocess.run", side_effect=fake_run):
            _compose_build_locked(
                chal_id=chal_id,
                chal_path=tmp_path,
                project_name=f"cage_test_{chal_id}_{threading.get_ident()}",
                runtime_compose_rel=".cage_runtime/test.yml",
                env={},
            )

    threads = [threading.Thread(target=_call, args=("pb-postexp-range-1",)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_inflight == 1, f"expected serialisation, saw {max_inflight} concurrent builds"


def _make_plan(services: dict) -> ComposeRuntimePlan:
    # Helper: minimal ComposeRuntimePlan stub for the missing-build-images
    # check. Only the ``config.services`` field is read by
    # ``_missing_build_images``; everything else can stay empty.
    from pathlib import Path
    return ComposeRuntimePlan(
        compose_path=Path("/tmp/unused.yml"),
        config={"services": services},
        services=[],
        public_service_names=[],
        external_ports={},
    )


def test_missing_build_images_returns_empty_when_no_build_services():
    # If no service has ``build:``, nothing needs to be built regardless of
    # what's on disk — we should NOT inspect any image (returns []).
    plan = _make_plan({"db": {"image": "mysql:5.7"}})
    with patch("cage.target.server.launch._image_exists", return_value=False) as mock_exists:
        assert _missing_build_images(plan, env={}) == []
    mock_exists.assert_not_called()


def test_missing_build_images_returns_missing_image_for_each_build_service():
    # Two services both build; one image exists, one doesn't → only the
    # missing one is returned.
    plan = _make_plan({
        "web":       {"build": {"context": "."}, "image": "cage-pb-target-foo-web:latest"},
        "evaluator": {"build": {"context": "."}, "image": "cage-pb-target-foo-eval:latest"},
    })
    exists_map = {
        "cage-pb-target-foo-web:latest": True,
        "cage-pb-target-foo-eval:latest": False,
    }
    with patch(
        "cage.target.server.launch._image_exists",
        side_effect=lambda img, env: exists_map[img],
    ):
        assert _missing_build_images(plan, env={}) == ["cage-pb-target-foo-eval:latest"]


def test_missing_build_images_returns_empty_when_all_present():
    # Hot path: every build-required image is already in the local image
    # store, so the build phase can be skipped entirely.
    plan = _make_plan({
        "web":       {"build": {"context": "."}, "image": "cage-pb-target-foo-web:latest"},
        "evaluator": {"build": {"context": "."}, "image": "cage-pb-target-foo-eval:latest"},
        "db":        {"image": "mysql:5.7"},  # no build, ignored
    })
    with patch("cage.target.server.launch._image_exists", return_value=True):
        assert _missing_build_images(plan, env={}) == []


def test_missing_build_images_detail_points_to_benchmark_build():
    detail = _missing_build_images_detail(
        chal_id="pb-demo",
        meta={"benchmark": "web_exploit_bench"},
        missing_images=["cage-pb-demo-web:latest"],
    )

    assert "target launch does not build images" in detail
    assert "cage-pb-demo-web:latest" in detail
    assert "Run: cage benchmark build web_exploit_bench --sample pb-demo" in detail
    assert "docker compose build" not in detail


def test_launch_challenge_rejects_missing_build_images_without_compose_build(
    tmp_path,
    monkeypatch,
):
    import cage.target.server.launch as launch_workflow

    meta = {
        "id": "pb-demo",
        "adapter_kind": "fake",
        "benchmark": "web_exploit_bench",
        "source_fields": {"challenge_id": "pb-demo"},
    }
    runtime_plan = _make_plan(
        {
            "web": {
                "build": {"context": "."},
                "image": "cage-pb-demo-web:latest",
            }
        }
    )

    class FakeAdapter:
        def build_launch_spec(self, _meta):
            return SimpleNamespace(
                mode="compose",
                working_directory=str(tmp_path),
                runtime_patches={},
            )

    monkeypatch.setenv("TARGET_SERVER_BUILD_IF_MISSING", "1")
    monkeypatch.setattr(launch_workflow, "load_all_challenges", lambda: {"pb-demo": meta})
    monkeypatch.setattr(launch_workflow, "get_running_instance", lambda _chal_id: None)
    monkeypatch.setattr(
        launch_workflow,
        "build_default_registry",
        lambda: SimpleNamespace(get=lambda _kind: FakeAdapter()),
    )
    monkeypatch.setattr(
        launch_workflow,
        "materialize_compose_runtime",
        lambda **_kwargs: runtime_plan,
    )
    monkeypatch.setattr(launch_workflow, "self_heal_docker_network", lambda: None)
    monkeypatch.setattr(launch_workflow, "_image_exists", lambda _image, env: False)
    monkeypatch.setattr(
        launch_workflow,
        "_compose_build_locked",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("target launch must not run docker compose build")
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        _launch_challenge_impl("pb-demo", force_recreate=False)

    assert exc_info.value.status_code == 500
    assert "target launch does not build images" in str(exc_info.value.detail)
    assert "Run: cage benchmark build web_exploit_bench --sample pb-demo" in str(
        exc_info.value.detail
    )


def test_compose_build_locked_does_not_serialise_across_chals(tmp_path):
    # Builds for DIFFERENT challenges must NOT serialise — that would tank
    # Internal target readiness checks with every sample queueing
    # behind every other sample's cold build). The lock is per-chal_id by
    # design.
    inflight = 0
    max_inflight = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3, timeout=2.0)

    def fake_run(*args, **kwargs):
        nonlocal inflight, max_inflight
        with lock:
            inflight += 1
            max_inflight = max(max_inflight, inflight)
        # Wait for all three threads to be inside the build at the same
        # time. If the lock is incorrectly shared the barrier times out.
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        with lock:
            inflight -= 1

        class _Res:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Res()

    def _call(chal_id):
        with patch("cage.target.server.launch.subprocess.run", side_effect=fake_run):
            _compose_build_locked(
                chal_id=chal_id,
                chal_path=tmp_path,
                project_name=f"cage_test_{chal_id}",
                runtime_compose_rel=".cage_runtime/test.yml",
                env={},
            )

    threads = [
        threading.Thread(target=_call, args=(f"chal-{i}",)) for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_inflight == 3, f"expected 3 concurrent builds across distinct chal_ids, saw {max_inflight}"


# ---------------------------------------------------------------------------
# Default network policy: compose_project_local + agent_network auto-pick
# ---------------------------------------------------------------------------
#
# These tests drive ``materialize_compose_runtime`` against synthetic compose
# files to verify the default-network-mode flip:
#   * empty / unset ``network_mode`` → ``compose_project_local``.
#   * 0 declared networks → inject a ``default`` entry, agent_network=default.
#   * 1 declared network → auto-pick it as agent_network.
#   * >1 declared networks → raise ValueError (caller must disambiguate).
#   * ``shared_external`` / ``alias`` opt-in → legacy alias-on-shared-bridge.
#   * Explicit ``agent_network`` always wins, and is materialised into
#     ``config["networks"]`` even if compose didn't declare it (so IPAM
#     allocation + label-stamp run on it).


def _materialize(tmp_path, compose_yaml: str, runtime_patches: dict):
    """Helper: write ``compose_yaml`` to tmp + call materialize_compose_runtime.

    Returns the resulting ``ComposeRuntimePlan``. Uses a counter-based fake
    ``find_free_port_fn`` so tests don't actually bind sockets, and mocks
    ``_collect_used_docker_subnets`` so subnet allocation doesn't depend on
    whatever leftover networks the host has from real cage runs.
    """
    from pathlib import Path
    from cage.target.adapters.base import LaunchSpec
    from cage.target.server.launch_runtime import materialize_compose_runtime
    from cage.target.server.network_alloc import _release_reserved_project_local_subnets

    compose_path = Path(tmp_path) / "docker-compose.yml"
    compose_path.write_text(compose_yaml)
    runtime_compose = Path(tmp_path) / ".cage_runtime" / "runtime.yml"
    runtime_compose.parent.mkdir(parents=True, exist_ok=True)
    port = [40000]

    def find_port() -> int:
        port[0] += 1
        return port[0]

    spec = LaunchSpec(
        mode="compose",
        working_directory=str(tmp_path),
        compose_files=[str(compose_path)],
        target_services=["web"],
        dependency_services=[],
        runtime_patches=dict(runtime_patches),
        exposure_mode="host_ports",
    )
    # Reset reservation set so consecutive tests get deterministic subnets,
    # and skip the live docker query so the host's existing networks don't
    # collide with the synthetic pool.
    _release_reserved_project_local_subnets()
    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        return_value=[],
    ):
        return materialize_compose_runtime(
            spec=spec,
            project_name="cage_bench_test_pb_x_aaaa1111_runtime",
            docker_network="cage_bench_test",
            host_ip="127.0.0.1",
            runtime_compose_path=runtime_compose,
            find_free_port_fn=find_port,
            existing_external_ports=None,
            challenge_id="pb-x",
        )


def test_default_network_mode_is_compose_project_local_with_injected_default(tmp_path):
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    ports: ["8080:80"]
""",
        runtime_patches={},
    )

    nets = plan.config.get("networks") or {}
    assert list(nets.keys()) == ["default"]
    # IPAM allocated from the project-local pool, labelled for cleanup.
    ipam = (nets["default"].get("ipam") or {}).get("config") or []
    assert ipam and ipam[0]["subnet"].startswith("172.31.")
    labels = nets["default"].get("labels") or {}
    assert labels.get("cage.target.network") == "cage_bench_test"
    assert labels.get("cage.target.role") == "runtime"
    # External bridge MUST NOT appear under the new default.
    assert "cage_bench_test" not in nets
    assert plan.agent_network_name == "cage_bench_test_pb_x_aaaa1111_runtime_default"


def test_default_picks_single_declared_network_when_agent_network_unset(tmp_path):
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    networks: [internal]
    ports: ["8080:80"]
networks:
  internal: {}
""",
        runtime_patches={},
    )

    assert plan.agent_network_name == "cage_bench_test_pb_x_aaaa1111_runtime_internal"
    assert "internal" in (plan.config.get("networks") or {})


def test_default_rejects_multiple_networks_without_explicit_agent_network(tmp_path):
    import pytest

    with pytest.raises(ValueError, match=r"Multiple compose networks declared"):
        _materialize(
            tmp_path,
            """
services:
  web:
    image: nginx:alpine
    networks: [internal, external_seg]
    ports: ["8080:80"]
networks:
  internal: {}
  external_seg: {}
""",
            runtime_patches={},
        )


def test_explicit_agent_network_lands_in_networks_dict_even_when_not_declared(tmp_path):
    # PrestaShop's regression: challenge.json declared
    # ``agent_network: default`` but its compose has no ``networks:`` block.
    # Without the ensure-exists pass, IPAM + labels would be skipped and
    # docker would allocate from its global default pool (172.17/16,
    # 172.18/16, …) — fine in isolation but prone to VPN clashes and
    # unlabelled for namespace-scoped orphan cleanup.
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    ports: ["8080:80"]
""",
        runtime_patches={"agent_network": "default"},
    )

    nets = plan.config.get("networks") or {}
    assert "default" in nets
    ipam = (nets["default"].get("ipam") or {}).get("config") or []
    assert ipam and ipam[0]["subnet"].startswith("172.31.")
    labels = nets["default"].get("labels") or {}
    assert labels.get("cage.target.network") == "cage_bench_test"


def test_shared_external_opt_in_attaches_services_to_docker_network(tmp_path):
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    ports: ["8080:80"]
""",
        runtime_patches={"network_mode": "shared_external"},
    )

    nets = plan.config.get("networks") or {}
    assert "cage_bench_test" in nets
    assert nets["cage_bench_test"].get("external") is True
    # Service got attached to docker_network with project-scoped aliases.
    svc = plan.config["services"]["web"]
    assert "networks" in svc
    assert "cage_bench_test" in svc["networks"]
    aliases = svc["networks"]["cage_bench_test"].get("aliases") or []
    assert any("web" in a for a in aliases)
    # No project-local ``default`` entry was injected in legacy mode.
    assert "default" not in nets
    # agent_network_name falls back to docker_network in legacy mode.
    assert plan.agent_network_name == "cage_bench_test"


def test_legacy_alias_keyword_maps_to_shared_external(tmp_path):
    # Older challenge.json files may have used ``network_mode: alias`` — it
    # must still trigger the shared-external path so the flip doesn't
    # silently change semantics for those.
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    ports: ["8080:80"]
""",
        runtime_patches={"network_mode": "alias"},
    )

    nets = plan.config.get("networks") or {}
    assert "cage_bench_test" in nets
    assert nets["cage_bench_test"].get("external") is True


def test_unknown_network_mode_falls_back_to_compose_project_local(tmp_path):
    # A bad value in challenge.json must not kill the whole batch — we
    # log a warning and fall back to the new safe default.
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    ports: ["8080:80"]
""",
        runtime_patches={"network_mode": "something_typoed"},
    )

    assert plan.agent_network_name.endswith("_default")
    nets = plan.config.get("networks") or {}
    assert "default" in nets
    assert "cage_bench_test" not in nets


def test_build_without_image_is_rejected(tmp_path):
    # Cage requires explicit image tags on every build: service. Without
    # this lint, docker compose would tag the build output with its
    # per-trial project-default name (``<project>-<service>:latest``),
    # producing one new image tag per trial — the tag-pollution we just
    # spent a commit cleaning up.
    import pytest

    with pytest.raises(ValueError, match=r"declares 'build:' but no 'image:'"):
        _materialize(
            tmp_path,
            """
services:
  web:
    build: ./web
    ports: ["8080:80"]
""",
            runtime_patches={},
        )


def test_build_with_empty_image_is_rejected(tmp_path):
    # ``image: ""`` (or whitespace) must trip the same lint — otherwise
    # compose silently falls back to its default-tagging.
    import pytest

    with pytest.raises(ValueError, match=r"declares 'build:' but no 'image:'"):
        _materialize(
            tmp_path,
            """
services:
  web:
    build: ./web
    image: "   "
    ports: ["8080:80"]
""",
            runtime_patches={},
        )


def test_build_with_explicit_image_is_accepted(tmp_path):
    # Sanity: the explicit-image path must not trip the lint.
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    build: ./web
    image: pentestbench-x-web:latest
    ports: ["8080:80"]
""",
        runtime_patches={},
    )

    assert plan.config["services"]["web"]["image"] == "pentestbench-x-web:latest"


def test_default_subnet_prefix_is_24(tmp_path):
    # /24 default gives 254 usable host IPs — enough for Dify-class targets
    # (14 services). Previous /28 cap was too tight. Test pins the value so
    # a future tweak triggers a deliberate review.
    plan = _materialize(
        tmp_path,
        """
services:
  web:
    image: nginx:alpine
    ports: ["8080:80"]
""",
        runtime_patches={},
    )

    nets = plan.config.get("networks") or {}
    subnet = (nets["default"].get("ipam") or {}).get("config")[0]["subnet"]
    assert subnet.endswith("/24"), f"expected /24, got {subnet}"


# ---------------------------------------------------------------------------
# Cage-managed subnet pools (agent_home + cage-trial) — P0a / P0c
# ---------------------------------------------------------------------------
#
# Verify the two new allocators carve from their pools, avoid overlap with
# existing docker networks, return distinct cidrs on repeat calls, and
# release them back to the pool.


def _release_all_cage_pools():
    """Helper: reset both new reservation sets between tests."""
    from cage.target.server.network_alloc import (
        _AGENT_HOME_RESERVED_SUBNETS,
        _CAGE_TRIAL_RESERVED_SUBNETS,
    )
    _AGENT_HOME_RESERVED_SUBNETS.clear()
    _CAGE_TRIAL_RESERVED_SUBNETS.clear()


def test_allocate_agent_home_subnet_returns_pool_24(monkeypatch):
    from cage.target.server.network_alloc import allocate_agent_home_subnet
    _release_all_cage_pools()
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_POOL", "10.200.0.0/16")
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_PREFIX", "24")
    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        return_value=[],
    ):
        cidr = allocate_agent_home_subnet("cage_bench_namespace_x")
    assert cidr.startswith("10.200.")
    assert cidr.endswith("/24")


def test_allocate_agent_home_subnet_avoids_repeat(monkeypatch):
    from cage.target.server.network_alloc import (
        allocate_agent_home_subnet,
        release_agent_home_subnet,
    )
    _release_all_cage_pools()
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_POOL", "10.200.0.0/16")
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_PREFIX", "24")
    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        return_value=[],
    ):
        a = allocate_agent_home_subnet("ns_a")
        b = allocate_agent_home_subnet("ns_b")
    assert a != b
    release_agent_home_subnet(a)
    release_agent_home_subnet(b)


def test_allocate_agent_home_subnet_raises_when_pool_exhausted(monkeypatch):
    # /24 pool with prefix /24 → exactly one slot. Pre-occupy it via a
    # mocked ``_collect_used_docker_subnets`` and confirm the allocator
    # raises rather than overlapping.
    from cage.target.server.network_alloc import allocate_agent_home_subnet
    import ipaddress
    _release_all_cage_pools()
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_POOL", "10.200.0.0/24")
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_PREFIX", "24")
    import pytest
    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        return_value=[ipaddress.ip_network("10.200.0.0/24")],
    ):
        with pytest.raises(RuntimeError, match="fully occupied"):
            allocate_agent_home_subnet("ns_overflow")


def test_allocate_cage_trial_subnet_returns_pool_26(monkeypatch):
    from cage.target.server.network_alloc import allocate_cage_trial_subnet
    _release_all_cage_pools()
    monkeypatch.setenv("CAGE_TRIAL_SUBNET_POOL", "10.201.0.0/16")
    monkeypatch.setenv("CAGE_TRIAL_SUBNET_PREFIX", "26")
    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        return_value=[],
    ):
        cidr = allocate_cage_trial_subnet("cage-trial-pb-X-pass_1-abcd1234")
    assert cidr.startswith("10.201.")
    assert cidr.endswith("/26")


def test_release_agent_home_subnet_handles_garbage():
    from cage.target.server.network_alloc import release_agent_home_subnet
    # Garbage / empty → no-op, no exception.
    release_agent_home_subnet("")
    release_agent_home_subnet(None)
    release_agent_home_subnet("not-a-cidr")


def test_release_cage_trial_subnet_handles_garbage():
    from cage.target.server.network_alloc import release_cage_trial_subnet
    release_cage_trial_subnet("")
    release_cage_trial_subnet(None)
    release_cage_trial_subnet("not-a-cidr")


def test_agent_home_and_cage_trial_pools_dont_share_subnets(monkeypatch):
    """The two pools must allocate from disjoint ranges by default.

    Sharing would defeat the point of separating them — agent_home and
    cage-trial fight over slots and one exhausting the other's pool.
    """
    from cage.target.server.network_alloc import (
        _agent_home_pool_settings,
        _cage_trial_pool_settings,
    )
    import ipaddress
    home_pool, _ = _agent_home_pool_settings()
    trial_pool, _ = _cage_trial_pool_settings()
    home_net = ipaddress.ip_network(home_pool, strict=False)
    trial_net = ipaddress.ip_network(trial_pool, strict=False)
    assert not home_net.overlaps(trial_net), (
        f"agent_home pool {home_pool} overlaps cage-trial pool {trial_pool}; "
        "they must be disjoint by default"
    )


def test_resolve_parallel_mode_defaults_to_network():
    """The bug we just fixed: agent_pentest_bench (and every non-cvebench family)
    used to get ``parallel_mode = 'alias'`` by default. That gated IPAM
    injection off, causing compose to fall back to docker's /16 default
    pool. Lock the new universal-default behavior in."""
    from cage.target.server.launch import resolve_parallel_mode

    # agent_pentest_bench meta — used to return "alias", must return "network" now
    assert resolve_parallel_mode({"benchmark_family": "agent_pentest_bench"}, None) == "network"
    # cvebench / autopenbench / etc. all use the same default — no family check
    assert resolve_parallel_mode({"benchmark_family": "cvebench"}, None) == "network"
    assert resolve_parallel_mode({"benchmark_family": "autopenbench"}, None) == "network"
    # No family info at all → still network
    assert resolve_parallel_mode({}, None) == "network"
    # Explicit override wins
    assert resolve_parallel_mode({"benchmark_family": "agent_pentest_bench"}, "alias") == "alias"
    assert resolve_parallel_mode({"benchmark_family": "agent_pentest_bench"}, "network") == "network"


def test_resolve_target_scope_reads_from_chal_data():
    """target_scope reads chal_data first; runtime_args overrides; default else.

    The framework keys off the declared ``target_scope`` field only — never the
    benchmark family. Challenges that need per-agent instances declare it in
    challenge.json (see the cvebench/autopenbench datasets).
    """
    from cage.target.scope import resolve_target_scope

    # Challenge-level declaration honored
    assert resolve_target_scope({"target_scope": "per_agent"}, {}) == "per_agent"
    # Runtime args always win
    assert resolve_target_scope(
        {"target_scope": "per_agent"}, {"target_scope": "per_challenge"}
    ) == "per_challenge"
    # Benchmark family alone never forces a scope — no name branch survives.
    assert resolve_target_scope({"benchmark_family": "agent_pentest_bench"}, {}) == "per_challenge"
    assert resolve_target_scope({"benchmark_family": "cvebench"}, {}) == "per_challenge"


def test_allocator_does_not_hold_lock_during_docker_rpc(monkeypatch):
    """Reviewer H2 regression guard.

    The docker daemon list call (``_collect_used_docker_subnets``) must
    NOT run inside the per-pool lock — otherwise concurrent passk
    allocations serialise behind a single RPC. We assert this by
    inspecting the wall-clock latency of 16 concurrent allocations:
    if the slow RPC were serialised, total wall time ≥ N * rpc_latency;
    parallelised, total wall time ≤ ~1 * rpc_latency + small overhead.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from cage.target.server.network_alloc import (
        _AGENT_HOME_RESERVED_SUBNETS,
        allocate_agent_home_subnet,
        release_agent_home_subnet,
    )

    _AGENT_HOME_RESERVED_SUBNETS.clear()
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_POOL", "10.200.0.0/16")
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_PREFIX", "24")

    RPC_DELAY = 0.05  # 50ms simulated slow daemon
    N = 16

    def slow_collect(_pool):
        time.sleep(RPC_DELAY)
        return []

    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        side_effect=slow_collect,
    ):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=N) as ex:
            results = list(ex.map(lambda i: allocate_agent_home_subnet(f"ns_{i}"), range(N)))
        wall = time.perf_counter() - start

    # If lock held during RPC: wall ≈ N * RPC_DELAY = 0.8s
    # If lock NOT held: wall ≈ 1 * RPC_DELAY + scheduling = ~0.1s
    # Give 4x RPC_DELAY headroom for slow CI; if we're past ~5x someone
    # serialised again.
    serialised_threshold = N * RPC_DELAY * 0.5  # 0.4s — well above parallel, well below serial
    assert wall < serialised_threshold, (
        f"allocations took {wall:.3f}s for {N} concurrent calls with "
        f"{RPC_DELAY}s simulated RPC; expected parallel (~{RPC_DELAY:.3f}s), "
        f"got serialised (~{N*RPC_DELAY:.3f}s) — the lock is being held "
        f"during the docker daemon RPC again."
    )
    assert len(set(results)) == N, "concurrent allocations produced duplicates"

    for cidr in results:
        release_agent_home_subnet(cidr)


def test_allocator_internal_helpers_still_signed_correctly():
    """Sanity-check the H2 split: _pool_candidates_and_existing must run
    without a lock; _pick_subnet_with_lock takes already-computed input."""
    from cage.target.server.network_alloc import (
        _pick_subnet_with_lock,
        _pool_candidates_and_existing,
    )
    candidates, existing, pool = _pool_candidates_and_existing("10.42.0.0/16", 24)
    assert len(candidates) == 256
    assert isinstance(existing, list)
    assert pool.prefixlen == 16

    chosen = _pick_subnet_with_lock(
        candidates=candidates,
        existing_used=[],
        seed="hello",
        extra_reserved=set(),
        pool_cidr="10.42.0.0/16",
        prefix=24,
    )
    assert chosen.startswith("10.42.")
    assert chosen.endswith("/24")


def test_isolation_teardown_releases_subnet_only_on_rm_success(monkeypatch):
    """Reviewer H1 regression guard.

    ``AgentIsolationNetwork.teardown`` previously released the subnet
    reservation unconditionally — even when ``docker network rm`` failed,
    in which case the network is still alive in the daemon but the slot
    is freed in our process-local pool. The next allocation could then
    pick a candidate that overlaps the still-alive network. (The daemon's
    own conflict check saves us in practice, but the pool reports
    spurious exhaustion as it fills with reserved-but-not-removed slots.)
    """
    import subprocess as sp
    from cage.target.provisioning import AgentIsolationNetwork
    from cage.target.server.network_alloc import _CAGE_TRIAL_RESERVED_SUBNETS

    _CAGE_TRIAL_RESERVED_SUBNETS.clear()
    fake_subnet = "10.201.7.64/26"
    _CAGE_TRIAL_RESERVED_SUBNETS.add(fake_subnet)
    iso = AgentIsolationNetwork(
        name="cage-trial-zz-test",
        connected_targets=[],
        subnet=fake_subnet,
    )

    def fake_run_fail(*args, **kwargs):
        return sp.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="busy")

    with patch("subprocess.run", side_effect=fake_run_fail):
        removed = iso.teardown()

    # rm FAILED → subnet must still be reserved
    assert removed is False
    assert fake_subnet in _CAGE_TRIAL_RESERVED_SUBNETS, (
        "teardown released the subnet despite docker network rm failing"
    )

    # Now simulate successful rm.
    _CAGE_TRIAL_RESERVED_SUBNETS.clear()
    _CAGE_TRIAL_RESERVED_SUBNETS.add(fake_subnet)
    iso2 = AgentIsolationNetwork(
        name="cage-trial-zz-test-2",
        connected_targets=[],
        subnet=fake_subnet,
    )

    def fake_run_success(*args, **kwargs):
        return sp.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

    with patch("subprocess.run", side_effect=fake_run_success):
        removed = iso2.teardown()

    assert removed is True
    assert fake_subnet not in _CAGE_TRIAL_RESERVED_SUBNETS, (
        "teardown failed to release the subnet after successful docker network rm"
    )


def test_isolation_teardown_treats_not_found_as_success(monkeypatch):
    """``docker network rm`` returning ``not found`` means the network is
    already gone — equivalent to success for reservation-release purposes.
    """
    import subprocess as sp
    from cage.target.provisioning import AgentIsolationNetwork
    from cage.target.server.network_alloc import _CAGE_TRIAL_RESERVED_SUBNETS

    _CAGE_TRIAL_RESERVED_SUBNETS.clear()
    fake_subnet = "10.201.9.0/26"
    _CAGE_TRIAL_RESERVED_SUBNETS.add(fake_subnet)
    iso = AgentIsolationNetwork(
        name="cage-trial-already-gone",
        connected_targets=[],
        subnet=fake_subnet,
    )

    def fake_run_notfound(*args, **kwargs):
        return sp.CompletedProcess(
            args=args[0], returncode=1, stdout="",
            stderr='Error: No such network: cage-trial-already-gone',
        )

    with patch("subprocess.run", side_effect=fake_run_notfound):
        iso.teardown()

    assert fake_subnet not in _CAGE_TRIAL_RESERVED_SUBNETS, (
        "teardown failed to release subnet when network was already gone"
    )


def test_isolation_network_retries_on_subnet_overlap(monkeypatch):
    """Reviewer M1 regression guard.

    Between ``_collect_used_docker_subnets`` and ``docker network create``,
    a peer process can claim our chosen /26. Without retry, we'd
    fall back to docker's default pool (eating a /16). With retry, we
    allocate a different /26 and try again.
    """
    import subprocess as sp
    from cage.target.provisioning import build_agent_isolation_network
    from cage.target.server.network_alloc import _CAGE_TRIAL_RESERVED_SUBNETS

    _CAGE_TRIAL_RESERVED_SUBNETS.clear()
    monkeypatch.setenv("CAGE_TRIAL_SUBNET_POOL", "10.201.0.0/16")
    monkeypatch.setenv("CAGE_TRIAL_SUBNET_PREFIX", "26")

    # First call to ``docker network create`` returns "overlaps", second succeeds.
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return sp.CompletedProcess(
                args=argv, returncode=1, stdout="",
                stderr="Error response from daemon: Pool overlaps with other one on this address space",
            )
        return sp.CompletedProcess(args=argv, returncode=0, stdout="id\n", stderr="")

    target_data = {
        "target_info": {"web": {"external_port": 8080}},
        "runtime": {"debug": {"network": {"services": [
            {"service_name": "web", "container_name": "test-web-1"},
        ]}}},
    }
    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        return_value=[],
    ), patch("subprocess.run", side_effect=fake_run):
        result = build_agent_isolation_network(
            trial_id="pb-X-pass_1", target_data=target_data,
            agent_container_name="agent-fake",
        )

    # We must have made multiple ``docker network create`` calls (retry).
    create_calls = [c for c in calls if len(c) > 2 and c[1] == "network" and c[2] == "create"]
    assert len(create_calls) >= 2, (
        f"expected retry after overlap conflict, got {len(create_calls)} create calls"
    )
    assert result is not None, "isolation network creation should succeed on retry"
    assert result.subnet is not None
    # The two attempts must have used DIFFERENT /26s.
    subnets = [arg for c in create_calls for i, arg in enumerate(c)
               if i > 0 and c[i-1] == "--subnet"]
    assert len(set(subnets)) == len(subnets), f"retry reused same subnet: {subnets}"
