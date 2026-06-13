"""Tests for P0-1 volume cleanup.

Covers three call paths:

1. ``_stamp_cage_run_id_labels`` stamps the ``cage.run_id`` /
   ``cage.component=target`` labels onto every non-external named
   volume in a compose config.
2. ``sweep_run`` removes labelled volumes alongside containers /
   networks (and only when the ``"target"`` component is requested).
3. ``cleanup_orphan_volumes`` removes namespace-scoped volumes that are
   not currently attached (best-effort; ``in use`` errors are skipped).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from docker.errors import APIError, NotFound

from cage.target import local_cleanup
from cage.target.local_cleanup import SweepResult, sweep_run
from cage.target.server import network_admin
from cage.target.server.network_alloc import _stamp_cage_run_id_labels

# ---------------------------------------------------------------------------
# _stamp_cage_run_id_labels
# ---------------------------------------------------------------------------


def test_stamp_labels_named_volume_gets_cage_labels():
    config = {
        "services": {"db": {"image": "mysql"}},
        "networks": {"default": {}},
        "volumes": {
            "db_data": {},
            "redis_cache": {"driver": "local"},
        },
    }

    _stamp_cage_run_id_labels(config, "run-abc", namespace="tcheck-a")

    for vol_name in ("db_data", "redis_cache"):
        labels = config["volumes"][vol_name]["labels"]
        assert labels["cage.run_id"] == "run-abc"
        assert labels["cage.component"] == "target"
        # P0-1 fix: namespace must be on volumes so cleanup_orphan_volumes
        # can safely filter cross-namespace.
        assert labels["cage.target.namespace"] == "tcheck-a"


def test_stamp_labels_omits_namespace_when_not_provided():
    """Backwards-compat: namespace arg is optional; absent → no label."""
    config = {"volumes": {"db_data": {}}}
    _stamp_cage_run_id_labels(config, "run-x")
    labels = config["volumes"]["db_data"]["labels"]
    assert labels["cage.run_id"] == "run-x"
    assert "cage.target.namespace" not in labels


def test_stamp_labels_namespace_propagates_to_networks_and_services():
    config = {
        "services": {"db": {"image": "mysql"}},
        "networks": {"default": {}},
        "volumes": {"db_data": {}},
    }
    _stamp_cage_run_id_labels(config, "run-abc", namespace="tcheck-a")

    assert config["services"]["db"]["labels"]["cage.target.namespace"] == "tcheck-a"
    assert config["networks"]["default"]["labels"]["cage.target.namespace"] == "tcheck-a"
    assert config["volumes"]["db_data"]["labels"]["cage.target.namespace"] == "tcheck-a"


def test_stamp_labels_external_volume_left_alone():
    config = {
        "volumes": {
            "user_shared": {"external": True, "name": "user_volume"},
        },
    }

    _stamp_cage_run_id_labels(config, "run-abc")

    # External volumes must remain unchanged — we don't own their labels.
    assert "labels" not in config["volumes"]["user_shared"]


def test_stamp_labels_handles_null_volume_config():
    config = {"volumes": {"anon_named": None}}

    _stamp_cage_run_id_labels(config, "run-xyz")

    assert config["volumes"]["anon_named"]["labels"]["cage.run_id"] == "run-xyz"


def test_stamp_labels_preserves_existing_volume_labels():
    config = {
        "volumes": {
            "db_data": {"labels": {"app": "wordpress"}},
        },
    }

    _stamp_cage_run_id_labels(config, "run-1")

    labels = config["volumes"]["db_data"]["labels"]
    assert labels["app"] == "wordpress"
    assert labels["cage.run_id"] == "run-1"
    assert labels["cage.component"] == "target"


def test_stamp_labels_volume_list_form_normalized_to_dict():
    """Compose accepts labels as either dict or ``key=value`` list."""
    config = {
        "volumes": {
            "db_data": {"labels": ["app=wordpress", "team=core"]},
        },
    }

    _stamp_cage_run_id_labels(config, "run-2")

    labels = config["volumes"]["db_data"]["labels"]
    assert isinstance(labels, dict)
    assert labels["app"] == "wordpress"
    assert labels["team"] == "core"
    assert labels["cage.run_id"] == "run-2"


# ---------------------------------------------------------------------------
# sweep_run volume branch
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _make_subprocess_recorder():
    """Return (recorder_list, callable) that records all subprocess.run calls.

    Each entry in the recorder list is the ``cmd`` arg. The callable
    routes ``docker ps``, ``docker network ls``, ``docker volume ls``
    to canned output, and treats every ``rm`` as success.
    """
    recorded: list[list[str]] = []
    canned: dict[tuple[str, ...], _FakeProc] = {}

    def runner(cmd, capture_output, text, timeout, check):
        recorded.append(list(cmd))
        key = tuple(cmd[:4])  # e.g. ('docker','ps','-aq','--filter')
        if key in canned:
            return canned[key]
        # ``ps -aq`` / ``network ls -q`` / ``volume ls -q`` → empty by default
        if cmd[:2] == ["docker", "ps"] and "-aq" in cmd:
            return _FakeProc(stdout="")
        if cmd[:3] == ["docker", "network", "ls"]:
            return _FakeProc(stdout="")
        if cmd[:3] == ["docker", "volume", "ls"]:
            return _FakeProc(stdout="")
        # rm always succeeds
        return _FakeProc(stdout="", returncode=0)

    return recorded, canned, runner


def test_sweep_run_removes_labelled_volumes(monkeypatch):
    recorded, canned, runner = _make_subprocess_recorder()
    # Canned listings: 1 container, 1 network, 2 volumes returned by ls.
    canned[("docker", "ps", "-aq", "--filter")] = _FakeProc(stdout="cid1\n")
    canned[("docker", "network", "ls", "-q")] = _FakeProc(stdout="nid1\n")
    canned[("docker", "volume", "ls", "-q")] = _FakeProc(stdout="vol-mysql\nvol-redis\n")

    monkeypatch.setattr(local_cleanup.subprocess, "run", runner)

    result = sweep_run("run-42", components=("target",))

    assert result.containers_removed == 1
    assert result.networks_removed == 1
    assert result.volumes_removed == 2
    assert result.errors == []
    # Volume rm commands were issued.
    rm_cmds = [c for c in recorded if c[:3] == ["docker", "volume", "rm"]]
    assert len(rm_cmds) == 2
    assert {c[-1] for c in rm_cmds} == {"vol-mysql", "vol-redis"}


def test_sweep_run_skips_volumes_for_agent_component(monkeypatch):
    recorded, canned, runner = _make_subprocess_recorder()
    canned[("docker", "volume", "ls", "-q")] = _FakeProc(stdout="vol-x\n")
    monkeypatch.setattr(local_cleanup.subprocess, "run", runner)

    result = sweep_run("run-42", components=("agent",))

    assert result.volumes_removed == 0
    # No volume ls / volume rm commands should have been issued.
    assert not any(c[:2] == ["docker", "volume"] for c in recorded)


def test_sweep_run_records_volume_rm_failures(monkeypatch):
    recorded, canned, runner = _make_subprocess_recorder()
    canned[("docker", "volume", "ls", "-q")] = _FakeProc(stdout="bad-vol\n")

    def runner_with_rm_failure(cmd, capture_output, text, timeout, check):
        if cmd[:3] == ["docker", "volume", "rm"]:
            return _FakeProc(stderr="Error: volume is in use", returncode=1)
        return runner(cmd, capture_output, text, timeout, check)

    monkeypatch.setattr(local_cleanup.subprocess, "run", runner_with_rm_failure)

    result = sweep_run("run-42", components=("target",))

    assert result.volumes_removed == 0
    assert any("volume bad-vol" in err for err in (result.errors or []))


def test_sweep_run_no_components_returns_empty():
    result = sweep_run("run-42", components=())
    assert result == SweepResult(run_id="run-42")


def test_sweep_result_removed_anything_includes_volumes():
    r = SweepResult(run_id="x", volumes_removed=1)
    assert r.removed_anything is True


def test_sweep_docker_resources_removes_explicit_resource_ids(monkeypatch):
    """Canonical ledgers can name resources even when Docker labels are gone."""

    recorded, _canned, runner = _make_subprocess_recorder()
    monkeypatch.setattr(local_cleanup.subprocess, "run", runner)

    result = local_cleanup.sweep_docker_resources(
        "run-ledger",
        containers=("agent-a",),
        networks=("trial-net",),
        volumes=("target-vol",),
    )

    assert result.containers_removed == 1
    assert result.networks_removed == 1
    assert result.volumes_removed == 1
    assert result.errors == []
    assert ["docker", "rm", "-f", "-v", "agent-a"] in recorded
    assert ["docker", "network", "rm", "trial-net"] in recorded
    assert ["docker", "volume", "rm", "-f", "target-vol"] in recorded


def test_sweep_run_namespace_propagates_to_target_filters(monkeypatch):
    """When ``namespace=`` is set, target-side ls calls include the
    namespace filter; agent-side calls don't (agents don't carry it)."""
    recorded, canned, runner = _make_subprocess_recorder()
    canned[("docker", "ps", "-aq", "--filter")] = _FakeProc(stdout="cid1\n")
    canned[("docker", "network", "ls", "-q")] = _FakeProc(stdout="")
    canned[("docker", "volume", "ls", "-q")] = _FakeProc(stdout="")
    monkeypatch.setattr(local_cleanup.subprocess, "run", runner)

    sweep_run("run-42", components=("agent", "target"), namespace="tcheck_a")

    target_ps = [c for c in recorded
                 if c[:2] == ["docker", "ps"] and "cage.component=target" in " ".join(c)]
    agent_ps = [c for c in recorded
                if c[:2] == ["docker", "ps"] and "cage.component=agent" in " ".join(c)]
    target_ns_filter = "label=cage.target.namespace=tcheck_a"

    assert target_ps and any(target_ns_filter in " ".join(c) for c in target_ps), \
        "namespace filter must propagate to target ps"
    assert agent_ps and not any(target_ns_filter in " ".join(c) for c in agent_ps), \
        "agent ps must NOT carry namespace filter"


def test_sweep_run_namespace_none_keeps_legacy_behavior(monkeypatch):
    """``namespace=None`` (default) → no namespace filter, matches old API."""
    recorded, canned, runner = _make_subprocess_recorder()
    monkeypatch.setattr(local_cleanup.subprocess, "run", runner)

    sweep_run("run-42", components=("target",))  # namespace omitted

    target_calls = [c for c in recorded if "cage.run_id=run-42" in " ".join(c)]
    assert target_calls, "sweep must have made some target calls"
    assert not any("cage.target.namespace" in " ".join(c) for c in target_calls), \
        "no namespace filter expected when namespace=None"


# ---------------------------------------------------------------------------
# cleanup_orphan_volumes
# ---------------------------------------------------------------------------


class _FakeVolume:
    def __init__(self, name, *, labels=None, remove_error=None):
        self.name = name
        self.attrs = {"Labels": labels or {}}
        self.removed = False
        self._remove_error = remove_error

    def remove(self, force=False):
        if self._remove_error is not None:
            raise self._remove_error
        self.removed = True


class _FakeVolumes:
    def __init__(self, volumes):
        self._volumes = volumes
        self.list_calls: list[dict] = []

    def list(self, filters=None):
        self.list_calls.append(filters or {})
        return list(self._volumes)


class _FakeDockerClient:
    def __init__(self, volumes):
        self.volumes = _FakeVolumes(volumes)


def test_cleanup_orphan_volumes_removes_matching_ns(monkeypatch):
    monkeypatch.setattr(network_admin, "TARGET_SERVER_NAMESPACE", "tcheck_a")
    v1 = _FakeVolume(
        "cage_bench_tcheck_a_db_data",
        labels={"cage.target.namespace": "tcheck_a"},
    )
    v2 = _FakeVolume(
        "cage_bench_tcheck_a_redis",
        labels={"cage.target.namespace": "tcheck_a"},
    )
    fake = _FakeDockerClient([v1, v2])
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: fake)

    network_admin.cleanup_orphan_volumes()

    assert v1.removed
    assert v2.removed


def test_cleanup_orphan_volumes_skips_other_namespace(monkeypatch):
    monkeypatch.setattr(network_admin, "TARGET_SERVER_NAMESPACE", "tcheck_a")
    mine = _FakeVolume(
        "vol_a", labels={"cage.target.namespace": "tcheck_a"}
    )
    theirs = _FakeVolume(
        "vol_b", labels={"cage.target.namespace": "tcheck_b"}
    )
    fake = _FakeDockerClient([mine, theirs])
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: fake)

    network_admin.cleanup_orphan_volumes()

    assert mine.removed
    assert not theirs.removed


def test_cleanup_orphan_volumes_skips_in_use(monkeypatch):
    monkeypatch.setattr(network_admin, "TARGET_SERVER_NAMESPACE", "tcheck_a")
    in_use = _FakeVolume(
        "vol_busy",
        labels={"cage.target.namespace": "tcheck_a"},
        remove_error=APIError("volume is in use"),
    )
    free = _FakeVolume(
        "vol_free",
        labels={"cage.target.namespace": "tcheck_a"},
    )
    fake = _FakeDockerClient([in_use, free])
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: fake)

    # Should not raise.
    network_admin.cleanup_orphan_volumes()
    assert not in_use.removed
    assert free.removed


def test_cleanup_orphan_volumes_swallows_notfound(monkeypatch):
    monkeypatch.setattr(network_admin, "TARGET_SERVER_NAMESPACE", "tcheck_a")
    gone = _FakeVolume(
        "vol_gone",
        labels={"cage.target.namespace": "tcheck_a"},
        remove_error=NotFound("vol_gone"),
    )
    fake = _FakeDockerClient([gone])
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: fake)

    network_admin.cleanup_orphan_volumes()
    # No assertion needed — must not raise.


def test_cleanup_orphan_volumes_bails_when_list_fails(monkeypatch):
    """When docker list blows up we bail safely — no fallback that could
    cross-namespace delete (the dangerous fallback was removed in P0-1 fix)."""
    monkeypatch.setattr(network_admin, "TARGET_SERVER_NAMESPACE", "tcheck_a")

    class _BrokenVolumes:
        def __init__(self):
            self.calls = 0

        def list(self, filters=None):
            self.calls += 1
            raise RuntimeError("docker socket down")

    client = SimpleNamespace(volumes=_BrokenVolumes())
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: client)

    network_admin.cleanup_orphan_volumes()
    # Exactly one list call — no fallback.
    assert client.volumes.calls == 1


def test_cleanup_orphan_volumes_handles_apierror_409_as_in_use(monkeypatch):
    """``APIError.status_code == 409`` is the docker daemon's "volume in use"
    response — handled via the structured status code, not string match
    on the error message (so daemon localization can't break it)."""
    from docker.errors import APIError as _APIError

    monkeypatch.setattr(network_admin, "TARGET_SERVER_NAMESPACE", "tcheck_a")

    class _Resp:
        status_code = 409
    err = _APIError("some translated message", response=_Resp())
    busy = _FakeVolume(
        "vol_busy",
        labels={"cage.target.namespace": "tcheck_a"},
        remove_error=err,
    )
    fake = _FakeDockerClient([busy])
    monkeypatch.setattr(network_admin, "get_docker_client", lambda: fake)

    network_admin.cleanup_orphan_volumes()
    assert not busy.removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
