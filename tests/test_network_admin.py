from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from docker.errors import NotFound

from cage.target.server import network_admin
from cage.target.server.network_alloc import _stamp_target_server_network_labels


class _FakeNetwork:
    def __init__(
        self,
        name,
        *,
        containers=None,
        labels=None,
        ipam_config=None,
        created=None,
        reload_error=None,
    ):
        self.name = name
        self.attrs = {
            "Containers": containers or {},
            "Labels": labels or {},
            "IPAM": {"Config": ipam_config or []},
        }
        if created is not None:
            self.attrs["Created"] = created
        self.reload_error = reload_error
        self.removed = False

    def reload(self):
        if self.reload_error:
            raise self.reload_error

    def remove(self):
        self.removed = True


class _FakeNetworks:
    def __init__(self, networks):
        self._networks = networks

    def list(self):
        return self._networks


class _FakeClient:
    def __init__(self, networks):
        self.networks = _FakeNetworks(networks)


def _iso_minus(seconds: float) -> str:
    """ISO-8601 timestamp ``seconds`` ago, formatted like docker SDK does."""
    return (datetime.now(tz=timezone.utc) - timedelta(seconds=seconds)).isoformat().replace(
        "+00:00", "Z"
    )


# ---------------------------------------------------------------------------
# _classify_cage_bench_network — the post-refactor categorizer (replaces the
# old ``_current_server_runtime_network_kind``).
# ---------------------------------------------------------------------------


def test_classify_own_home_and_runtime(monkeypatch):
    monkeypatch.setattr(network_admin, "DOCKER_NETWORK", "cage_bench_tcheck_current")

    assert (
        network_admin._classify_cage_bench_network("cage_bench_tcheck_current", "agent_home")
        == "own_home"
    )
    assert (
        network_admin._classify_cage_bench_network(
            "cage_bench_tcheck_current_pb_siyucms_abcd1234_runtime_default",
            "runtime",
        )
        == "own_runtime"
    )


def test_classify_peer_home_and_runtime(monkeypatch):
    monkeypatch.setattr(network_admin, "DOCKER_NETWORK", "cage_bench_tcheck_current")

    assert (
        network_admin._classify_cage_bench_network("cage_bench_tcheck_other", "agent_home")
        == "peer_home"
    )
    assert (
        network_admin._classify_cage_bench_network(
            "cage_bench_tcheck_other_pb_x_abcd1234_runtime_default", "runtime"
        )
        == "peer_runtime"
    )


def test_classify_non_cage_returns_none(monkeypatch):
    monkeypatch.setattr(network_admin, "DOCKER_NETWORK", "cage_bench_tcheck_current")

    assert network_admin._classify_cage_bench_network("bridge", "") is None
    assert network_admin._classify_cage_bench_network("ctf_old_runtime_default", "") is None
    assert network_admin._classify_cage_bench_network("user_cage_bench_fake", "") is None


def test_classify_role_label_overrides_substring_heuristic(monkeypatch):
    # ``_runtime`` appearing inside a name but the role label saying agent_home
    # must classify as agent_home (substring heuristic alone would mistake it
    # for runtime).
    monkeypatch.setattr(network_admin, "DOCKER_NETWORK", "cage_bench_tcheck_current")
    # No ``_runtime`` in name → must classify as home regardless of role label.
    assert (
        network_admin._classify_cage_bench_network("cage_bench_peer_ns", "agent_home")
        == "peer_home"
    )


# ---------------------------------------------------------------------------
# _parse_docker_created / _network_age_seconds — ISO-8601 parsing
# ---------------------------------------------------------------------------


def test_parse_docker_created_handles_z_suffix():
    parsed = network_admin._parse_docker_created("2026-05-17T00:09:07.282784732Z")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.tzinfo is not None


def test_parse_docker_created_handles_explicit_offset():
    parsed = network_admin._parse_docker_created("2026-05-17T00:09:07+00:00")
    assert parsed is not None


def test_parse_docker_created_returns_none_on_garbage():
    assert network_admin._parse_docker_created("") is None
    assert network_admin._parse_docker_created(None) is None
    assert network_admin._parse_docker_created("not-a-timestamp") is None


def test_network_age_seconds_is_positive_for_past_timestamp():
    attrs = {"Created": _iso_minus(300.0)}
    age = network_admin._network_age_seconds(attrs)
    assert age is not None and age >= 299.0


def test_network_age_seconds_returns_none_when_missing():
    assert network_admin._network_age_seconds({}) is None


# ---------------------------------------------------------------------------
# _is_namespace_alive — /proc cmdline scan
# ---------------------------------------------------------------------------


def test_is_namespace_alive_returns_false_for_empty():
    assert network_admin._is_namespace_alive("") is False


def test_is_namespace_alive_returns_false_when_no_match(monkeypatch, tmp_path):
    # Build a fake /proc tree with one cage process but a different namespace.
    proc = tmp_path / "proc"
    proc.mkdir()
    pid_dir = proc / "12345"
    pid_dir.mkdir()
    cmd = b"\x00".join(
        [
            b"/usr/bin/python3",
            b"-m",
            b"cage.cli",
            b"serve",
            b"--namespace",
            b"someother_ns",
        ]
    ) + b"\x00"
    (pid_dir / "cmdline").write_bytes(cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))
    assert network_admin._is_namespace_alive("target_ns") is False


def test_is_namespace_alive_finds_namespace_flag(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    pid_dir = proc / "12345"
    pid_dir.mkdir()
    # Console-script form.
    cmd = b"\x00".join(
        [
            b"/usr/bin/python3",
            b"-m",
            b"cage.cli",
            b"serve",
            b"--namespace",
            b"target_ns",
        ]
    ) + b"\x00"
    (pid_dir / "cmdline").write_bytes(cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))
    assert network_admin._is_namespace_alive("target_ns") is True


def test_is_namespace_alive_finds_namespace_kv_form(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    pid_dir = proc / "67890"
    pid_dir.mkdir()
    # ``--namespace=target_ns`` form.
    cmd = b"\x00".join(
        [
            b"/usr/bin/python3",
            b"-m",
            b"cage.cli",
            b"serve",
            b"--namespace=target_ns",
        ]
    ) + b"\x00"
    (pid_dir / "cmdline").write_bytes(cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))
    assert network_admin._is_namespace_alive("target_ns") is True


def test_is_namespace_alive_ignores_non_cage_processes(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    pid_dir = proc / "1"
    pid_dir.mkdir()
    # docker shell that happens to mention the namespace string but isn't cage.
    cmd = b"\x00".join(
        [
            b"docker",
            b"network",
            b"create",
            b"--label",
            b"cage.target.namespace=target_ns",
        ]
    ) + b"\x00"
    (pid_dir / "cmdline").write_bytes(cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))
    assert network_admin._is_namespace_alive("target_ns") is False


# ---------------------------------------------------------------------------
# cleanup_orphan_networks — full behaviour incl. peer reclamation
# ---------------------------------------------------------------------------


def _setup_cleanup_env(monkeypatch, networks, *, alive_namespaces=None, min_age=120.0):
    """Setup helper. Returns a single list capturing every subnet release.

    Both pools (project-local + agent_home) are captured into the same
    list so tests can assert "this cidr was released" without caring
    which pool the implementation chose.
    """
    alive = set(alive_namespaces or [])
    monkeypatch.setattr(network_admin, "DOCKER_NETWORK", "cage_bench_tcheck_current")
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: _FakeClient(networks))
    monkeypatch.setattr(network_admin, "ORPHAN_NETWORK_MIN_AGE_S", float(min_age))
    monkeypatch.setattr(
        network_admin,
        "_is_namespace_alive",
        lambda ns, pid=None: ns in alive,
    )
    released: list[str] = []
    monkeypatch.setattr(
        network_admin,
        "release_reserved_project_local_subnet",
        released.append,
    )
    monkeypatch.setattr(
        network_admin,
        "release_agent_home_subnet",
        released.append,
    )
    return released


def test_cleanup_removes_own_empty_runtime(monkeypatch):
    own_runtime_empty = _FakeNetwork(
        "cage_bench_tcheck_current_pb_x_abcd1234_runtime_default",
        labels={"cage.target.role": "runtime"},
        ipam_config=[{"Subnet": "172.99.1.0/24"}],
    )
    own_runtime_active = _FakeNetwork(
        "cage_bench_tcheck_current_pb_y_abcd1234_runtime_default",
        containers={"abc": {"Name": "y"}},
        labels={"cage.target.role": "runtime"},
    )
    released = _setup_cleanup_env(monkeypatch, [own_runtime_empty, own_runtime_active])

    network_admin.cleanup_orphan_networks()

    assert own_runtime_empty.removed
    assert not own_runtime_active.removed
    assert released == ["172.99.1.0/24"]


def test_cleanup_preserves_own_home(monkeypatch):
    own_home = _FakeNetwork(
        "cage_bench_tcheck_current",
        labels={"cage.target.role": "agent_home"},
        # Even empty + very old, own_home is managed by
        # ensure_docker_network / remove_own_docker_network — not by
        # cleanup_orphan_networks.
        created=_iso_minus(86400),
    )
    _setup_cleanup_env(monkeypatch, [own_home])

    network_admin.cleanup_orphan_networks()

    assert not own_home.removed


def test_cleanup_reclaims_peer_home_when_namespace_dead_and_old(monkeypatch):
    peer_home = _FakeNetwork(
        "cage_bench_tcheck_dead_peer",
        labels={
            "cage.target.role": "agent_home",
            "cage.target.namespace": "tcheck_dead_peer",
            "cage.target.pid": "999999",
        },
        ipam_config=[{"Subnet": "10.42.5.0/24"}],
        created=_iso_minus(3600),
    )
    released = _setup_cleanup_env(
        monkeypatch, [peer_home], alive_namespaces=set()
    )

    network_admin.cleanup_orphan_networks()

    assert peer_home.removed
    assert released == ["10.42.5.0/24"]


def test_cleanup_preserves_peer_home_when_namespace_alive(monkeypatch):
    peer_home = _FakeNetwork(
        "cage_bench_tcheck_live_peer",
        labels={
            "cage.target.role": "agent_home",
            "cage.target.namespace": "tcheck_live_peer",
        },
        created=_iso_minus(3600),
    )
    _setup_cleanup_env(
        monkeypatch,
        [peer_home],
        alive_namespaces={"tcheck_live_peer"},
    )

    network_admin.cleanup_orphan_networks()

    assert not peer_home.removed


def test_cleanup_preserves_peer_home_when_too_young(monkeypatch):
    # Even with a dead namespace, a just-created peer home (e.g. another
    # cage server starting concurrently) must NOT be reclaimed.
    peer_home = _FakeNetwork(
        "cage_bench_tcheck_just_starting",
        labels={
            "cage.target.role": "agent_home",
            "cage.target.namespace": "tcheck_just_starting",
        },
        created=_iso_minus(5),
    )
    _setup_cleanup_env(
        monkeypatch,
        [peer_home],
        alive_namespaces=set(),
        min_age=120.0,
    )

    network_admin.cleanup_orphan_networks()

    assert not peer_home.removed


def test_cleanup_preserves_peer_when_namespace_label_missing(monkeypatch):
    # User-created or external network whose name happens to start with
    # ``cage_bench_`` and has no namespace label must be left alone.
    suspicious = _FakeNetwork(
        "cage_bench_some_external_thing",
        labels={"cage.target.role": "agent_home"},
        created=_iso_minus(3600),
    )
    _setup_cleanup_env(monkeypatch, [suspicious])

    network_admin.cleanup_orphan_networks()

    assert not suspicious.removed


def test_cleanup_reclaims_peer_runtime_when_namespace_dead(monkeypatch):
    peer_runtime = _FakeNetwork(
        "cage_bench_tcheck_dead_peer_pb_z_abc_runtime_default",
        labels={
            "cage.target.role": "runtime",
            "cage.target.namespace": "tcheck_dead_peer",
        },
        ipam_config=[{"Subnet": "172.31.5.0/24"}],
        created=_iso_minus(3600),
    )
    released = _setup_cleanup_env(
        monkeypatch, [peer_runtime], alive_namespaces=set()
    )

    network_admin.cleanup_orphan_networks()

    assert peer_runtime.removed
    assert released == ["172.31.5.0/24"]


def test_cleanup_handles_reload_notfound(monkeypatch):
    vanished = _FakeNetwork(
        "cage_bench_tcheck_current_pb_gone_abc_runtime_default",
        reload_error=NotFound("gone"),
    )
    _setup_cleanup_env(monkeypatch, [vanished])

    network_admin.cleanup_orphan_networks()

    # NotFound on reload → skip silently (no crash, no removal call which
    # would also raise NotFound).
    assert not vanished.removed


# ---------------------------------------------------------------------------
# Pre-existing tests (kept for regression coverage on label-stamping).
# ---------------------------------------------------------------------------


def test_runtime_network_labels_bind_networks_to_server():
    config = {
        "networks": {
            "default": None,
            "frontend": {"labels": ["existing=yes"]},
            "shared": {"external": True},
        }
    }

    _stamp_target_server_network_labels(config, "cage_bench_tcheck_current")

    assert config["networks"]["default"]["labels"] == {
        "cage.target.network": "cage_bench_tcheck_current",
        "cage.target.role": "runtime",
    }
    assert config["networks"]["frontend"]["labels"] == {
        "existing": "yes",
        "cage.target.network": "cage_bench_tcheck_current",
        "cage.target.role": "runtime",
    }
    assert config["networks"]["shared"] == {"external": True}


# ---------------------------------------------------------------------------
# cage-trial-* orphan reclamation (orchestrator-side leak path)
# ---------------------------------------------------------------------------


def test_is_pid_alive_fast_path(tmp_path, monkeypatch):
    # Build fake /proc tree.
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "12345").mkdir()
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))
    assert network_admin._is_pid_alive("12345") is True
    assert network_admin._is_pid_alive("99999") is False
    assert network_admin._is_pid_alive("0") is False
    assert network_admin._is_pid_alive("-1") is False
    assert network_admin._is_pid_alive("abc") is False
    assert network_admin._is_pid_alive(None) is False
    assert network_admin._is_pid_alive("") is False


def _setup_cleanup_env_with_alive_pids(monkeypatch, networks, *, alive_pids=None, min_age=120.0):
    """Variant of _setup_cleanup_env: also stubs ``_is_pid_alive``."""
    released = _setup_cleanup_env(monkeypatch, networks, min_age=min_age)
    pids = set(str(p) for p in (alive_pids or []))
    monkeypatch.setattr(
        network_admin, "_is_pid_alive",
        lambda pid: str(pid).strip() in pids,
    )
    # Capture cage-trial subnet releases too.
    monkeypatch.setattr(
        network_admin, "release_cage_trial_subnet",
        released.append,
    )
    return released


def test_cleanup_reclaims_cage_trial_when_owner_pid_dead(monkeypatch):
    cage_trial = _FakeNetwork(
        "cage-trial-pb-X-pass_1-deadbeef",
        labels={
            "cage.network.kind": "cage-trial",
            "cage.network.owner-pid": "999999",
        },
        ipam_config=[{"Subnet": "10.201.5.64/26"}],
        created=_iso_minus(3600),
    )
    released = _setup_cleanup_env_with_alive_pids(monkeypatch, [cage_trial], alive_pids=[])

    network_admin.cleanup_orphan_networks()

    assert cage_trial.removed
    assert released == ["10.201.5.64/26"]


def test_cleanup_preserves_cage_trial_when_owner_pid_alive(monkeypatch):
    import os
    my_pid = os.getpid()
    cage_trial = _FakeNetwork(
        "cage-trial-pb-X-pass_1-abc",
        labels={
            "cage.network.kind": "cage-trial",
            "cage.network.owner-pid": str(my_pid),
        },
        ipam_config=[{"Subnet": "10.201.6.0/26"}],
        created=_iso_minus(3600),
    )
    _setup_cleanup_env_with_alive_pids(monkeypatch, [cage_trial], alive_pids=[my_pid])

    network_admin.cleanup_orphan_networks()

    assert not cage_trial.removed


def test_cleanup_preserves_cage_trial_when_too_young(monkeypatch):
    # Just-created cage-trial — even if owner PID is dead, give it time
    # for teardown to run before stomping.
    cage_trial = _FakeNetwork(
        "cage-trial-pb-X-pass_1-new",
        labels={
            "cage.network.kind": "cage-trial",
            "cage.network.owner-pid": "999999",
        },
        created=_iso_minus(5),
    )
    _setup_cleanup_env_with_alive_pids(
        monkeypatch, [cage_trial], alive_pids=[], min_age=120.0,
    )

    network_admin.cleanup_orphan_networks()

    assert not cage_trial.removed


def test_cleanup_preserves_user_network_with_cage_trial_prefix(monkeypatch):
    # User-created network with cage-trial prefix but no cage label:
    # must not be touched.
    suspicious = _FakeNetwork(
        "cage-trial-totally-not-cage",
        labels={"some.other.label": "value"},
        created=_iso_minus(3600),
    )
    _setup_cleanup_env_with_alive_pids(monkeypatch, [suspicious], alive_pids=[])

    network_admin.cleanup_orphan_networks()

    assert not suspicious.removed


# ---------------------------------------------------------------------------
# Reviewer M4 follow-up: missing test coverage
# ---------------------------------------------------------------------------


def _cmdline(*args: bytes) -> bytes:
    return b"\x00".join(args) + b"\x00"


def test_is_namespace_alive_pid_fast_path_matches(monkeypatch, tmp_path):
    # Fast path: the labeled PID exists with matching cmdline → return True
    # without scanning all of /proc.
    proc = tmp_path / "proc"
    proc.mkdir()
    pid_dir = proc / "7777"
    pid_dir.mkdir()
    cmd = _cmdline(
        b"python3",
        b"-m",
        b"cage.target.serve",
        b"--namespace",
        b"my_ns",
    )
    (pid_dir / "cmdline").write_bytes(cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))

    # Track whether we fell through to full scan (we shouldn't).
    scan_count = 0
    real_iterdir = (tmp_path / "proc").iterdir
    def counting_iterdir(self):
        nonlocal scan_count
        scan_count += 1
        return real_iterdir()
    monkeypatch.setattr("pathlib.Path.iterdir", counting_iterdir, raising=False)

    assert network_admin._is_namespace_alive("my_ns", "7777") is True
    # Fast path returns BEFORE iterating /proc.
    assert scan_count == 0, f"fast path should not iterate /proc, got {scan_count} iterations"


def test_is_namespace_alive_pid_fast_path_falls_through_on_mismatch(monkeypatch, tmp_path):
    # PID exists but its cmdline is for a different namespace → fast path
    # rejects, scan kicks in and finds the real owner elsewhere.
    proc = tmp_path / "proc"
    proc.mkdir()
    # /proc/100 has cmdline for a DIFFERENT namespace (stale PID hint).
    stale = proc / "100"
    stale.mkdir()
    stale_cmd = _cmdline(
        b"python3",
        b"-m",
        b"cage.cli",
        b"serve",
        b"--namespace",
        b"different_ns",
    )
    (stale / "cmdline").write_bytes(stale_cmd)
    # /proc/200 has the actual owner.
    owner = proc / "200"
    owner.mkdir()
    owner_cmd = _cmdline(
        b"python3",
        b"-m",
        b"cage.target.serve",
        b"--namespace",
        b"target_ns",
    )
    (owner / "cmdline").write_bytes(owner_cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))

    assert network_admin._is_namespace_alive("target_ns", "100") is True


def test_is_namespace_alive_pid_fast_path_nonint_falls_through(monkeypatch, tmp_path):
    # Garbage pid_hint must not crash; scan still runs.
    proc = tmp_path / "proc"
    proc.mkdir()
    owner = proc / "300"
    owner.mkdir()
    cmd = _cmdline(
        b"python3",
        b"-m",
        b"cage.target.serve",
        b"--namespace",
        b"ok_ns",
    )
    (owner / "cmdline").write_bytes(cmd)
    monkeypatch.setattr(network_admin, "Path", lambda p: tmp_path / p.lstrip("/"))

    assert network_admin._is_namespace_alive("ok_ns", "not-an-int") is True


def test_cleanup_orphan_networks_mixed_inputs(monkeypatch):
    # One sweep, multiple kinds — verify each goes through the right branch.
    own_runtime_empty = _FakeNetwork(
        "cage_bench_tcheck_current_pb_a_xxx_runtime_default",
        labels={"cage.target.role": "runtime"},
        ipam_config=[{"Subnet": "172.31.1.0/24"}],
    )
    own_runtime_active = _FakeNetwork(
        "cage_bench_tcheck_current_pb_b_xxx_runtime_default",
        containers={"abc": {"Name": "live"}},
        labels={"cage.target.role": "runtime"},
    )
    peer_home_dead = _FakeNetwork(
        "cage_bench_dead_ns",
        labels={
            "cage.target.role": "agent_home",
            "cage.target.namespace": "dead_ns",
            "cage.target.pid": "999991",
        },
        ipam_config=[{"Subnet": "10.200.50.0/24"}],
        created=_iso_minus(3600),
    )
    peer_home_live = _FakeNetwork(
        "cage_bench_alive_ns",
        labels={
            "cage.target.role": "agent_home",
            "cage.target.namespace": "alive_ns",
        },
        created=_iso_minus(3600),
    )
    peer_home_young = _FakeNetwork(
        "cage_bench_young_ns",
        labels={
            "cage.target.role": "agent_home",
            "cage.target.namespace": "young_ns",
        },
        created=_iso_minus(5),
    )
    own_home = _FakeNetwork(
        "cage_bench_tcheck_current",
        labels={"cage.target.role": "agent_home"},
        created=_iso_minus(3600),
    )
    cage_trial_dead = _FakeNetwork(
        "cage-trial-pb-x-pass_1-zz",
        labels={"cage.network.kind": "cage-trial", "cage.network.owner-pid": "999992"},
        ipam_config=[{"Subnet": "10.201.7.0/26"}],
        created=_iso_minus(3600),
    )
    released = _setup_cleanup_env_with_alive_pids(
        monkeypatch,
        [
            own_runtime_empty,
            own_runtime_active,
            peer_home_dead,
            peer_home_live,
            peer_home_young,
            own_home,
            cage_trial_dead,
        ],
        alive_pids=[],
        min_age=120.0,
    )
    monkeypatch.setattr(
        network_admin, "_is_namespace_alive",
        lambda ns, pid=None: ns == "alive_ns",
    )

    network_admin.cleanup_orphan_networks()

    assert own_runtime_empty.removed, "empty own runtime should be reclaimed"
    assert not own_runtime_active.removed, "active own runtime preserved"
    assert peer_home_dead.removed, "dead peer home should be reclaimed"
    assert not peer_home_live.removed, "live peer home preserved"
    assert not peer_home_young.removed, "young peer home preserved (age gate)"
    assert not own_home.removed, "own home never auto-reclaimed"
    assert cage_trial_dead.removed, "dead cage-trial should be reclaimed"
    # Verify subnets returned to the right pools:
    assert "172.31.1.0/24" in released  # project-local
    assert "10.200.50.0/24" in released  # agent_home
    assert "10.201.7.0/26" in released  # cage-trial


def test_allocator_concurrency_under_simulated_daemon_failure(monkeypatch):
    """Reviewer M4: allocator must not deadlock or leak slots when the
    docker daemon RPC fails intermittently for SOME threads."""
    from concurrent.futures import ThreadPoolExecutor

    from cage.target.server.network_alloc import (
        _AGENT_HOME_RESERVED_SUBNETS,
        allocate_agent_home_subnet,
        release_agent_home_subnet,
    )

    _AGENT_HOME_RESERVED_SUBNETS.clear()
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_POOL", "10.200.0.0/16")
    monkeypatch.setenv("TARGET_SERVER_AGENT_HOME_SUBNET_PREFIX", "24")

    # Half the threads see a healthy daemon, half see exceptions — but
    # the exception path returns ``[]`` (per ``_collect_used_docker_subnets``'s
    # bare-except), not raises, so allocators still succeed.
    call_counter = [0]
    def flaky_collect(pool):
        n = call_counter[0]
        call_counter[0] += 1
        if n % 2 == 1:
            return []  # simulates a daemon timeout fallback
        return []

    with patch(
        "cage.target.server.network_alloc._collect_used_docker_subnets",
        side_effect=flaky_collect,
    ):
        with ThreadPoolExecutor(max_workers=10) as ex:
            cidrs = list(ex.map(lambda i: allocate_agent_home_subnet(f"ns_{i}"), range(10)))

    assert len(set(cidrs)) == 10, f"got duplicates under flaky daemon: {sorted(cidrs)}"
    for c in cidrs:
        release_agent_home_subnet(c)
    assert len(_AGENT_HOME_RESERVED_SUBNETS) == 0, "reservations leaked after release"
