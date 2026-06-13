"""Tests for the shared liveness module and ``cage gc``.

Coverage:

  * Five branches of ``is_run_running`` against synthetic ``.cage_runs/``
    layouts on tmp_path.
  * ``locate_run_dir`` / ``iter_known_run_ids`` over a multi-agent root.
  * ``gc`` decision tree + ``--apply`` propagation via subprocess.run
    mocking (no real docker daemon).
  * ``default_cage_runs_roots`` precedence (env var > cwd > examples/).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.artifacts.resources import ResourceLedgerReader, ResourceLedgerWriter
from cage.gc import runner as cage_gc
from cage.experiment.model import (
    ResourceRecord,
    build_experiment_plan,
    load_experiment_spec,
)
from cage.experiment.engine.live.liveness import (
    is_run_running,
    iter_known_run_ids,
    locate_run_dir,
)

# ---------------------------------------------------------------------------
# is_run_running — five branches
# ---------------------------------------------------------------------------


def _write_dashboard(run_dir: Path, **fields) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_dir.name, **fields}
    (run_dir / "dashboard.json").write_text(json.dumps(payload))


def _write_planned(run_dir: Path, *, fresh: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "planned_trials.json"
    path.write_text("[]")
    if not fresh:
        # Older than the 5-minute live window.
        old = time.time() - 3600
        os.utime(path, (old, old))


def _write_experiment_record(run_dir: Path, **fields) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_dir.name, "status": "planned", **fields}
    (run_dir / "experiment_record.json").write_text(json.dumps(payload))


def _write_contract_project(project_dir: Path) -> Path:
    """Write a tiny project file that can produce canonical trial records."""

    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: liveness-demo
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBenchmark
runtime:
  max_rounds: 2
agents:
  - id: agent
    kind: codex
    model: model
""".lstrip(),
        encoding="utf-8",
    )
    return project_file


def _write_active_trial(run_dir: Path, trial_id: str, *, recent: bool = True) -> Path:
    """Create a trial dir with a progress.json (active) but no task_output."""
    trial = run_dir / "trials" / trial_id
    proxy = trial / "proxy"
    proxy.mkdir(parents=True, exist_ok=True)
    p = proxy / "progress.json"
    p.write_text(json.dumps({"successful_requests": 0}))
    if not recent:
        old = time.time() - 3600
        os.utime(p, (old, old))
    return trial


def _write_completed_trial(run_dir: Path, trial_id: str) -> Path:
    """Create a trial dir with task_output.json (terminal)."""
    trial = run_dir / "trials" / trial_id
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "task_output.json").write_text("{}")
    return trial


def test_liveness_dead_when_completed_at_set(tmp_path):
    run = tmp_path / "run-a"
    _write_dashboard(run, completed_at="2026-05-17T00:00:00")
    _write_active_trial(run, "t1", recent=True)  # should not matter

    result = is_run_running(run)
    assert result.running is False
    assert "completed_at" in result.reason


def test_liveness_alive_when_progress_recent(tmp_path):
    run = tmp_path / "run-b"
    _write_dashboard(run)  # no completed_at
    _write_active_trial(run, "t1", recent=True)

    result = is_run_running(run)
    assert result.running is True
    assert "active trial" in result.reason


def test_liveness_dead_when_progress_stale(tmp_path):
    run = tmp_path / "run-c"
    _write_dashboard(run)
    _write_active_trial(run, "t1", recent=False)

    result = is_run_running(run)
    assert result.running is False
    assert "stalled" in result.reason


def test_liveness_alive_when_dashboard_pending_no_progress(tmp_path):
    """Real dashboard schema: agents[label] = {total, completed, failed}.

    Pending = max(0, total - completed - failed). This is the shared
    formula in ``cage.experiment.engine.live.fs_signals.dashboard_pending_count`` used
    by both the web inspector and ``cage gc``. If they diverge, the GC
    can kill a live run.
    """
    run = tmp_path / "run-d"
    _write_dashboard(
        run,
        agents={"codex": {"total": 5, "completed": 1, "failed": 0}},
    )
    # No trial dirs at all yet — pending=4 (5-1-0).

    result = is_run_running(run)
    assert result.running is True
    assert "pending" in result.reason


def test_liveness_dead_when_dashboard_all_done(tmp_path):
    """All completed/failed but no completed_at — still classified dead."""
    run = tmp_path / "run-allcomplete"
    _write_dashboard(
        run,
        agents={"codex": {"total": 5, "completed": 3, "failed": 2}},
    )
    # No active trials, dashboard shows 0 pending.
    result = is_run_running(run)
    assert result.running is False


def test_liveness_pending_formula_matches_web_inspector():
    """Contract: the GC's pending-counter MUST be the same function as
    the web inspector's. If they diverge, GC can kill live runs."""
    from cage.experiment.engine.live.fs_signals import dashboard_pending_count
    from cage.web.data import _dashboard_pending_count as web_pending

    assert dashboard_pending_count is web_pending, (
        "liveness/gc and web/inspector diverged on dashboard_pending_count — "
        "fix cage/experiment/engine/live/fs_signals.py or cage/web/data/__init__.py"
    )
    sample = {"agents": {
        "codex": {"total": 5, "completed": 2, "failed": 1},
        "claude": {"total": 3, "completed": 3, "failed": 0},
    }}
    assert dashboard_pending_count(sample) == 2  # (5-2-1) + (3-3-0)
    assert dashboard_pending_count({}) == 0
    assert dashboard_pending_count({"agents": None}) == 0
    assert dashboard_pending_count({"agents": {"codex": "bad"}}) == 0


def test_liveness_alive_when_planned_fresh_no_dashboard(tmp_path):
    run = tmp_path / "run-e"
    run.mkdir(parents=True)
    _write_planned(run, fresh=True)
    # No dashboard.json yet.

    result = is_run_running(run)
    assert result.running is True
    assert "planned_trials" in result.reason


def test_liveness_alive_when_canonical_record_fresh_no_dashboard(tmp_path):
    run = tmp_path / "run-record"
    _write_experiment_record(run, status="planned")
    # No dashboard.json or planned_trials.json yet.

    result = is_run_running(run)
    assert result.running is True
    assert "experiment_record" in result.reason


def test_liveness_alive_when_fresh_canonical_record_overrides_stale_dashboard(tmp_path):
    run = tmp_path / "run-record-active"
    _write_dashboard(run, completed_at="2026-05-17T00:00:00")
    old = time.time() - 3600
    os.utime(run / "dashboard.json", (old, old))
    _write_experiment_record(run, status="running")

    result = is_run_running(run)

    assert result.running is True
    assert "experiment_record" in result.reason


def test_liveness_dead_when_canonical_record_is_terminal(tmp_path):
    run = tmp_path / "run-record-done"
    _write_experiment_record(run, status="completed")
    # No dashboard.json yet; the canonical record is still terminal truth.

    result = is_run_running(run)
    assert result.running is False
    assert "experiment_record status=completed" in result.reason


def test_liveness_alive_when_fresh_canonical_trial_record_is_running(tmp_path):
    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    run = tmp_path / ".cage_runs" / "agent:model" / "run-record-trial-active"
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id=run.name,
        created_at="2026-06-05T00:00:00Z",
    )
    old = time.time() - 3600
    os.utime(run / "experiment_record.json", (old, old))
    writer.mark_trial_started(
        plan.trials[0].trial_id,
        started_at="2026-06-05T00:00:10Z",
    )

    result = is_run_running(run)

    assert result.running is True
    assert "trial_record" in result.reason


def test_liveness_dead_when_terminal_trial_record_overrides_legacy_progress(tmp_path):
    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    run = tmp_path / ".cage_runs" / "agent:model" / "run-record-trial-done"
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id=run.name,
        created_at="2026-06-05T00:00:00Z",
    )
    writer.mark_trial_finished(
        plan.trials[0].trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    old = time.time() - 3600
    os.utime(run / "experiment_record.json", (old, old))
    _write_active_trial(run, "sample-a/pass_1", recent=True)

    result = is_run_running(run)

    assert result.running is False
    assert "trial_record" in result.reason


def test_liveness_dead_when_terminal_trial_records_override_fresh_running_record(tmp_path):
    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    run = tmp_path / ".cage_runs" / "agent:model" / "run-record-fresh-trial-done"
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id=run.name,
        created_at="2026-06-05T00:00:00Z",
    )
    writer.mark_run_started(started_at="2026-06-05T00:00:05Z")
    writer.mark_trial_finished(
        plan.trials[0].trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    _write_active_trial(run, "sample-a/pass_1", recent=True)

    result = is_run_running(run)

    assert result.running is False
    assert "trial_record" in result.reason


def test_liveness_dead_when_nothing(tmp_path):
    run = tmp_path / "run-f"
    run.mkdir(parents=True)
    # No dashboard, no planned, no trials.

    result = is_run_running(run)
    assert result.running is False


def test_liveness_dead_when_dir_missing(tmp_path):
    result = is_run_running(tmp_path / "nope")
    assert result.running is False
    assert "missing" in result.reason


# ---------------------------------------------------------------------------
# locate_run_dir / iter_known_run_ids
# ---------------------------------------------------------------------------


def _scaffold(root: Path, agent: str, rid: str) -> Path:
    p = root / agent / rid
    p.mkdir(parents=True, exist_ok=True)
    (p / "dashboard.json").write_text("{}")
    return p


def test_locate_run_dir_walks_agent_layout(tmp_path):
    root = tmp_path / ".cage_runs"
    _scaffold(root, "codex:gpt-5.5:stateless", "run-2026-aaa")
    _scaffold(root, "claude:c4:stateful", "run-2026-bbb")

    found = locate_run_dir("run-2026-bbb", search_roots=[root])
    assert found is not None
    assert found.name == "run-2026-bbb"
    assert found.parent.name == "claude:c4:stateful"


def test_locate_run_dir_returns_none_when_unknown(tmp_path):
    root = tmp_path / ".cage_runs"
    _scaffold(root, "codex", "run-x")
    assert locate_run_dir("run-not-here", search_roots=[root]) is None


def test_iter_known_run_ids_yields_each_once(tmp_path):
    root1 = tmp_path / ".cage_runs"
    root2 = tmp_path / "examples" / "pb" / ".cage_runs"
    _scaffold(root1, "codex", "run-1")
    _scaffold(root1, "codex", "run-2")
    _scaffold(root2, "claude", "run-3")
    # Same run-id under both roots — should only yield once.
    _scaffold(root2, "claude", "run-1")

    seen = list(iter_known_run_ids([root1, root2]))
    ids = sorted(rid for rid, _ in seen)
    assert ids == ["run-1", "run-2", "run-3"]


# ---------------------------------------------------------------------------
# gc decision tree
# ---------------------------------------------------------------------------


def test_gc_run_classifies_orphan_when_no_artifact(tmp_path):
    decision = cage_gc.gc_run(
        "ghost-rid",
        {"containers": 1, "networks": 1, "volumes": 0},
        search_roots=[tmp_path],
        apply=False,
    )
    assert decision.decision == cage_gc.DECISION_ORPHAN
    assert decision.run_dir is None
    assert decision.swept is None  # dry-run


def test_gc_run_classifies_alive_when_actively_ticking(tmp_path):
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "codex", "run-alive")
    _write_active_trial(run_dir, "t1", recent=True)

    decision = cage_gc.gc_run(
        "run-alive",
        {"containers": 1, "networks": 0, "volumes": 0},
        search_roots=[root],
        apply=True,  # even with apply, alive runs must NOT be swept
    )
    assert decision.decision == cage_gc.DECISION_ALIVE
    assert decision.swept is None


def test_gc_run_classifies_dead_when_completed(tmp_path):
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "codex", "run-dead")
    _write_dashboard(run_dir, completed_at="2026-05-17")
    decision = cage_gc.gc_run(
        "run-dead",
        {"containers": 2, "networks": 1, "volumes": 1},
        search_roots=[root],
        apply=False,
    )
    assert decision.decision == cage_gc.DECISION_DEAD
    assert decision.swept is None  # still dry-run


def test_gc_run_no_resources_skips_sweep(tmp_path):
    """Dead run with 0 resources: nothing to sweep, but recorded as dead."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "codex", "run-dead-empty")
    _write_dashboard(run_dir, completed_at="2026-05-17")

    decision = cage_gc.gc_run(
        "run-dead-empty",
        {"containers": 0, "networks": 0, "volumes": 0},
        search_roots=[root],
        apply=True,
    )
    assert decision.decision == cage_gc.DECISION_DEAD
    assert decision.swept is None  # nothing to sweep


def test_gc_run_apply_invokes_sweep(monkeypatch, tmp_path):
    """When ``apply=True`` and decision is dead, sweep_run is called."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "codex", "run-dead-with-stuff")
    _write_dashboard(run_dir, completed_at="2026-05-17")

    swept_args = {}

    def fake_sweep(run_id, *, components, namespace=None, docker_timeout=60.0):
        from cage.target.local_cleanup import SweepResult
        swept_args["run_id"] = run_id
        swept_args["components"] = components
        swept_args["namespace"] = namespace
        return SweepResult(
            run_id=run_id, containers_removed=2,
            networks_removed=1, volumes_removed=1,
        )

    monkeypatch.setattr(cage_gc, "sweep_run", fake_sweep)

    decision = cage_gc.gc_run(
        "run-dead-with-stuff",
        {"containers": 2, "networks": 1, "volumes": 1},
        search_roots=[root],
        apply=True,
    )
    assert decision.decision == cage_gc.DECISION_DEAD
    assert decision.swept is not None
    assert swept_args["run_id"] == "run-dead-with-stuff"
    assert "agent" in swept_args["components"] and "target" in swept_args["components"]
    assert swept_args["namespace"] is None  # no namespace passed
    assert decision.swept.containers_removed == 2
    assert decision.swept.volumes_removed == 1


def test_gc_run_namespace_propagates_to_sweep(monkeypatch, tmp_path):
    """H2 fix: namespace param must reach sweep_run."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "codex", "run-dead-ns")
    _write_dashboard(run_dir, completed_at="2026-05-17")

    captured = {}

    def fake_sweep(run_id, *, components, namespace=None, docker_timeout=60.0):
        from cage.target.local_cleanup import SweepResult
        captured["namespace"] = namespace
        return SweepResult(run_id=run_id)

    monkeypatch.setattr(cage_gc, "sweep_run", fake_sweep)

    cage_gc.gc_run(
        "run-dead-ns",
        {"containers": 1, "networks": 0, "volumes": 0},
        search_roots=[root],
        apply=True,
        namespace="tcheck_a",
    )
    assert captured["namespace"] == "tcheck_a"


def test_gc_all_propagates_namespace_to_each_gc_run(monkeypatch, tmp_path):
    """End-to-end namespace propagation from ``gc_all`` to ``sweep_run``."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "codex", "run-dead")
    _write_dashboard(run_dir, completed_at="2026-05-17")

    monkeypatch.setattr(
        cage_gc, "collect_docker_run_ids",
        lambda namespace=None: {
            "run-dead": {"containers": 1, "networks": 0, "volumes": 0},
        },
    )

    sweeps: list[dict] = []

    def fake_sweep(run_id, *, components, namespace=None, docker_timeout=60.0):
        from cage.target.local_cleanup import SweepResult
        sweeps.append({"run_id": run_id, "namespace": namespace})
        return SweepResult(run_id=run_id)

    monkeypatch.setattr(cage_gc, "sweep_run", fake_sweep)

    cage_gc.gc_all(apply=True, namespace="tcheck_a", search_roots=[root])
    assert sweeps == [{"run_id": "run-dead", "namespace": "tcheck_a"}]


# ---------------------------------------------------------------------------
# default_cage_runs_roots
# ---------------------------------------------------------------------------


def test_default_roots_env_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("CAGE_RUNS_ROOT", str(tmp_path / "custom"))
    # cwd has its own .cage_runs — should be IGNORED in favor of env.
    (tmp_path / ".cage_runs").mkdir()
    roots = cage_gc.default_cage_runs_roots(cwd=tmp_path)
    assert roots == [tmp_path / "custom"]


def test_default_roots_picks_up_cwd_cage_runs(monkeypatch, tmp_path):
    monkeypatch.delenv("CAGE_RUNS_ROOT", raising=False)
    (tmp_path / ".cage_runs").mkdir()
    (tmp_path / "examples" / "pb" / ".cage_runs").mkdir(parents=True)

    roots = cage_gc.default_cage_runs_roots(cwd=tmp_path)
    assert tmp_path / ".cage_runs" in roots
    assert tmp_path / "examples" / "pb" / ".cage_runs" in roots


def test_default_roots_anchors_on_cli_package_marker(monkeypatch, tmp_path):
    """Auto-discovery should use the package CLI marker."""

    monkeypatch.delenv("CAGE_RUNS_ROOT", raising=False)
    repo = tmp_path / "repo"
    (repo / "cage" / "cli").mkdir(parents=True)
    (repo / "cage" / "cli" / "main.py").write_text("", encoding="utf-8")
    expected = repo / "examples" / "pb" / ".cage_runs"
    expected.mkdir(parents=True)

    roots = cage_gc.default_cage_runs_roots(cwd=repo / "examples" / "pb")
    assert expected in roots


def test_default_roots_empty_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("CAGE_RUNS_ROOT", raising=False)
    roots = cage_gc.default_cage_runs_roots(cwd=tmp_path)
    assert roots == []


# ---------------------------------------------------------------------------
# collect_docker_run_ids — subprocess mocked
# ---------------------------------------------------------------------------


def test_collect_docker_run_ids_groups_by_rid(monkeypatch):
    """Each resource emits a single id|run_id line via ``docker inspect``."""

    def fake_docker_ls(cmd, timeout):
        # Dispatch by detecting kind + ls/inspect.
        head2 = cmd[:2]
        head3 = cmd[:3]
        # ``docker ps -aq ...``
        if head2 == ["docker", "ps"]:
            return ["c1", "c2", "c3"]
        # ``docker network ls -q ...`` vs ``docker network inspect ...``
        if head3 == ["docker", "network", "ls"]:
            return ["n1"]
        if head3 == ["docker", "network", "inspect"]:
            return ["n1|run-A"]
        # ``docker volume ls -q ...`` vs ``docker volume inspect ...``
        if head3 == ["docker", "volume", "ls"]:
            return ["v1", "v2"]
        if head3 == ["docker", "volume", "inspect"]:
            return ["v1|run-A", "v2|run-B"]
        # ``docker inspect ...`` (containers)
        if head2 == ["docker", "inspect"]:
            return ["c1|run-A", "c2|run-A", "c3|run-B"]
        return []

    monkeypatch.setattr(cage_gc, "_docker_ls", fake_docker_ls)

    grouped = cage_gc.collect_docker_run_ids()
    assert grouped["run-A"]["containers"] == 2
    assert grouped["run-A"]["networks"] == 1
    assert grouped["run-A"]["volumes"] == 1
    assert grouped["run-B"]["containers"] == 1
    assert grouped["run-B"]["networks"] == 0
    assert grouped["run-B"]["volumes"] == 1


def test_collect_docker_run_ids_handles_empty(monkeypatch):
    monkeypatch.setattr(cage_gc, "_docker_ls", lambda cmd, timeout: [])
    assert cage_gc.collect_docker_run_ids() == {}


# ---------------------------------------------------------------------------
# canonical resource ledger collection
# ---------------------------------------------------------------------------


def test_collect_ledger_resource_counts_uses_latest_resource_status(tmp_path):
    """GC should count unreleased ledger resources, not every ledger line."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "claude", "run-ledger")
    ledger = ResourceLedgerWriter(run_dir)

    ledger.append_resource(
        run_id="run-ledger",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:01Z",
    )
    ledger.append_resource(
        run_id="run-ledger",
        resource_id="docker_container:agent-b",
        kind="docker_container",
        provider="docker",
        external_id="agent-b",
        status="started",
        cleanup_action="docker rm -f agent-b",
        timestamp="2026-06-05T00:00:02Z",
    )
    ledger.append_resource(
        run_id="run-ledger",
        resource_id="docker_container:agent-b",
        kind="docker_container",
        provider="docker",
        external_id="agent-b",
        status="released",
        cleanup_action="docker rm -f agent-b",
        timestamp="2026-06-05T00:00:03Z",
    )
    ledger.append_resource(
        run_id="run-ledger",
        resource_id="docker_network:trial-net",
        kind="docker_network",
        provider="docker",
        external_id="trial-net",
        status="cleanup_failed",
        cleanup_action="docker network rm trial-net",
        timestamp="2026-06-05T00:00:04Z",
    )
    ledger.append_resource(
        run_id="run-ledger",
        resource_id="docker_volume:target-vol",
        kind="docker_volume",
        provider="docker",
        external_id="target-vol",
        status="created",
        cleanup_action="docker volume rm target-vol",
        timestamp="2026-06-05T00:00:05Z",
    )

    assert cage_gc.collect_ledger_resource_counts([root]) == {
        "run-ledger": {"containers": 1, "networks": 1, "volumes": 1},
    }


def test_collect_ledger_resource_counts_prefers_canonical_snapshot(
    monkeypatch,
    tmp_path,
):
    """GC should consume the shared canonical run snapshot when available."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "claude", "run-snapshot-ledger")
    load_snapshot_calls: list[Path] = []

    def fake_load_snapshot(self: ExperimentArtifactReader):
        load_snapshot_calls.append(self.run_dir)
        return SimpleNamespace(
            resources=(
                ResourceRecord(
                    schema_version="resource_record.v1",
                    record_id="res_000001",
                    run_id="run-snapshot-ledger",
                    resource_id="docker_container:agent-a",
                    kind="docker_container",
                    provider="docker",
                    external_id="agent-a",
                    status="started",
                    cleanup_action="docker rm -f agent-a",
                    timestamp="2026-06-05T00:00:01Z",
                ),
            )
        )

    monkeypatch.setattr(ExperimentArtifactReader, "load_snapshot", fake_load_snapshot)

    assert cage_gc.collect_ledger_resource_counts([root]) == {
        "run-snapshot-ledger": {"containers": 1, "networks": 0, "volumes": 0},
    }
    assert load_snapshot_calls == [run_dir.resolve()]


def test_gc_all_reports_ledger_only_resources(monkeypatch, tmp_path):
    """Dry-run visibility should not depend on Docker labels alone."""
    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "claude", "run-ledger-dead")
    _write_dashboard(run_dir, completed_at="2026-06-05T00:00:00Z")
    ledger = ResourceLedgerWriter(run_dir)
    ledger.append_resource(
        run_id="run-ledger-dead",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:01Z",
    )
    ledger.append_resource(
        run_id="run-ledger-dead",
        resource_id="docker_network:trial-net",
        kind="docker_network",
        provider="docker",
        external_id="trial-net",
        status="cleanup_failed",
        cleanup_action="docker network rm trial-net",
        timestamp="2026-06-05T00:00:02Z",
    )

    monkeypatch.setattr(cage_gc, "collect_docker_run_ids", lambda namespace=None: {})
    monkeypatch.setattr(cage_gc, "sweep_run", lambda *a, **k: pytest.fail("dry-run should not sweep"))

    report = cage_gc.gc_all(apply=False, search_roots=[root])

    assert len(report.decisions) == 1
    decision = report.decisions[0]
    assert decision.run_id == "run-ledger-dead"
    assert decision.decision == cage_gc.DECISION_DEAD
    assert decision.container_count == 1
    assert decision.network_count == 1
    assert decision.volume_count == 0
    assert decision.swept is None


def test_gc_all_apply_sweeps_ledger_only_docker_resources(monkeypatch, tmp_path):
    """Apply mode should use canonical Docker resource ids, not labels only."""

    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "claude", "run-ledger-apply")
    _write_dashboard(run_dir, completed_at="2026-06-05T00:00:00Z")
    ledger = ResourceLedgerWriter(run_dir)
    ledger.append_resource(
        run_id="run-ledger-apply",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:01Z",
    )
    ledger.append_resource(
        run_id="run-ledger-apply",
        resource_id="docker_network:trial-net",
        kind="docker_network",
        provider="docker",
        external_id="trial-net",
        status="cleanup_failed",
        cleanup_action="docker network rm trial-net",
        timestamp="2026-06-05T00:00:02Z",
    )
    ledger.append_resource(
        run_id="run-ledger-apply",
        resource_id="docker_volume:target-vol",
        kind="docker_volume",
        provider="docker",
        external_id="target-vol",
        status="created",
        cleanup_action="docker volume rm target-vol",
        timestamp="2026-06-05T00:00:03Z",
    )

    monkeypatch.setattr(cage_gc, "collect_docker_run_ids", lambda namespace=None: {})
    label_sweeps: list[str] = []
    ledger_sweeps: list[dict[str, object]] = []

    def fake_sweep_run(run_id, *, components, namespace=None, docker_timeout=60.0):
        from cage.target.local_cleanup import SweepResult

        label_sweeps.append(run_id)
        return SweepResult(run_id=run_id)

    def fake_sweep_docker_resources(
        run_id,
        *,
        containers=(),
        networks=(),
        volumes=(),
        docker_timeout=60.0,
    ):
        from cage.target.local_cleanup import SweepResult

        ledger_sweeps.append({
            "run_id": run_id,
            "containers": tuple(containers),
            "networks": tuple(networks),
            "volumes": tuple(volumes),
        })
        return SweepResult(
            run_id=run_id,
            containers_removed=len(containers),
            networks_removed=len(networks),
            volumes_removed=len(volumes),
        )

    monkeypatch.setattr(cage_gc, "sweep_run", fake_sweep_run)
    monkeypatch.setattr(cage_gc, "sweep_docker_resources", fake_sweep_docker_resources, raising=False)

    report = cage_gc.gc_all(apply=True, search_roots=[root])

    assert label_sweeps == ["run-ledger-apply"]
    assert ledger_sweeps == [{
        "run_id": "run-ledger-apply",
        "containers": ("agent-a",),
        "networks": ("trial-net",),
        "volumes": ("target-vol",),
    }]
    assert report.summary()["removed"] == {
        "containers": 1,
        "networks": 1,
        "volumes": 1,
    }


def test_gc_all_apply_records_cleanup_status_for_ledger_docker_resources(
    monkeypatch,
    tmp_path,
):
    """Apply mode should append ResourceLedger outcome records after cleanup."""

    root = tmp_path / ".cage_runs"
    run_dir = _scaffold(root, "claude", "run-ledger-record-cleanup")
    _write_dashboard(run_dir, completed_at="2026-06-05T00:00:00Z")
    ledger = ResourceLedgerWriter(run_dir)
    ledger.append_resource(
        run_id="run-ledger-record-cleanup",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:01Z",
    )
    ledger.append_resource(
        run_id="run-ledger-record-cleanup",
        resource_id="docker_network:trial-net",
        kind="docker_network",
        provider="docker",
        external_id="trial-net",
        status="started",
        cleanup_action="docker network rm trial-net",
        timestamp="2026-06-05T00:00:02Z",
    )

    monkeypatch.setattr(cage_gc, "collect_docker_run_ids", lambda namespace=None: {})
    monkeypatch.setattr(
        cage_gc,
        "sweep_run",
        lambda run_id, **kwargs: SimpleNamespace(
            run_id=run_id,
            containers_removed=0,
            networks_removed=0,
            volumes_removed=0,
            errors=[],
        ),
    )
    monkeypatch.setattr(
        cage_gc,
        "sweep_docker_resources",
        lambda run_id, **kwargs: SimpleNamespace(
            run_id=run_id,
            containers_removed=1,
            networks_removed=1,
            volumes_removed=0,
            errors=[],
        ),
    )

    cage_gc.gc_all(apply=True, search_roots=[root])

    latest = ResourceLedgerReader(run_dir).latest_by_resource_id()
    assert latest["docker_container:agent-a"].status == "released"
    assert latest["docker_network:trial-net"].status == "released"


# ---------------------------------------------------------------------------
# gc_all end-to-end with mocked docker + tmp .cage_runs/
# ---------------------------------------------------------------------------


def test_gc_all_dry_run_reports_without_sweeping(monkeypatch, tmp_path):
    root = tmp_path / ".cage_runs"
    alive_run = _scaffold(root, "codex", "run-alive")
    _write_active_trial(alive_run, "t1", recent=True)
    dead_run = _scaffold(root, "codex", "run-dead")
    _write_dashboard(dead_run, completed_at="2026-05-17")

    # Docker reports both run-ids plus an orphan with no .cage_runs/ trace.
    monkeypatch.setattr(
        cage_gc, "collect_docker_run_ids",
        lambda namespace=None: {
            "run-alive": {"containers": 1, "networks": 0, "volumes": 0},
            "run-dead": {"containers": 2, "networks": 1, "volumes": 1},
            "run-ghost": {"containers": 1, "networks": 0, "volumes": 0},
        },
    )
    monkeypatch.setattr(cage_gc, "sweep_run", lambda *a, **k: pytest.fail("dry-run should not sweep"))

    report = cage_gc.gc_all(apply=False, search_roots=[root])

    by_rid = {d.run_id: d for d in report.decisions}
    assert by_rid["run-alive"].decision == cage_gc.DECISION_ALIVE
    assert by_rid["run-dead"].decision == cage_gc.DECISION_DEAD
    assert by_rid["run-ghost"].decision == cage_gc.DECISION_ORPHAN
    assert all(d.swept is None for d in report.decisions)
    summary = report.summary()
    assert summary["alive"] == 1 and summary["dead"] == 1 and summary["orphan"] == 1
    assert summary["removed"] == {"containers": 0, "networks": 0, "volumes": 0}


def test_gc_all_apply_sweeps_dead_and_orphan(monkeypatch, tmp_path):
    root = tmp_path / ".cage_runs"
    alive_run = _scaffold(root, "codex", "run-alive")
    _write_active_trial(alive_run, "t1", recent=True)
    dead_run = _scaffold(root, "codex", "run-dead")
    _write_dashboard(dead_run, completed_at="2026-05-17")

    monkeypatch.setattr(
        cage_gc, "collect_docker_run_ids",
        lambda namespace=None: {
            "run-alive": {"containers": 1, "networks": 0, "volumes": 0},
            "run-dead": {"containers": 2, "networks": 1, "volumes": 1},
            "run-ghost": {"containers": 1, "networks": 0, "volumes": 0},
        },
    )

    swept_ids: list[str] = []

    def fake_sweep(run_id, *, components, namespace=None, docker_timeout=60.0):
        from cage.target.local_cleanup import SweepResult
        swept_ids.append(run_id)
        return SweepResult(
            run_id=run_id, containers_removed=1, networks_removed=0,
            volumes_removed=0,
        )

    monkeypatch.setattr(cage_gc, "sweep_run", fake_sweep)

    report = cage_gc.gc_all(apply=True, search_roots=[root])

    # run-alive must NOT have been swept.
    assert "run-alive" not in swept_ids
    assert "run-dead" in swept_ids
    assert "run-ghost" in swept_ids
    summary = report.summary()
    assert summary["alive"] == 1 and summary["dead"] == 1 and summary["orphan"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
