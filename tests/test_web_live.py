"""Tests for the live-polling layer: caches, signatures, delta API."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import pytest

from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.experiment.model import (
    TrialTermination,
    build_experiment_plan,
    load_experiment_spec,
)
from cage.web.app import create_app
from cage.web.cache import (
    SignatureCache,
    _signals_cache,
    discovery_cache,
    is_recently_active,
    run_summary_cache,
    safe_mtime_ns,
    scan_run_signals,
    trial_summary_cache,
)
from cage.web.data import (
    _read_progress_cached,
    load_dashboard,
    load_live_check_evidence,
    load_trial,
    load_trial_summary_cached,
    scan_runs,
    trial_summary_signature,
)
from cage.web.delta import build_run_trials_delta, build_runs_delta

# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Each test starts with empty caches so order doesn't matter."""
    run_summary_cache.clear()
    trial_summary_cache.clear()
    discovery_cache.clear()
    _signals_cache.clear()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write newline-delimited JSON test artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def _write_contract_project(project_dir: Path) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: record-only-demo
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


def _write_canonical_run_snapshot(run_dir: Path, *, run_id: str) -> None:
    project_file = _write_contract_project(run_dir.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id=run_id,
        created_at="2026-06-05T00:00:00Z",
    )


def _write_canonical_run_with_indexed_proxy_log(tmp_path: Path) -> tuple[Path, str]:
    """Create a canonical run whose raw proxy log lives outside legacy trials."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    project_file = _write_contract_project(run.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-record",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    proxy_ref = "canonical_proxy/proxy.jsonl"
    _write_jsonl(
        run / proxy_ref,
        [
            {
                "request_id": "req-0001",
                "trial_id": trial_id,
                "status": "success",
                "ts_ms": 1000,
                "openai_request": {
                    "model": "gpt-test",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "upstream_response": {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            }
        ],
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.proxy_log",
        path=proxy_ref,
        kind="proxy_log",
        schema_version="proxy_log.jsonl.v1",
        producer="proxy session",
        replayability="audit",
        content_type="application/x-ndjson",
    )
    return run, trial_id


def _bump_mtime(path: Path, delta_sec: float = 0.5) -> None:
    """Touch *path* into the future so signatures change reliably."""
    st = path.stat()
    new = st.st_atime + delta_sec, st.st_mtime + delta_sec
    Path(path).touch()  # ensure file exists
    import os
    os.utime(path, new)


# --------------------------------------------------------------------
# SignatureCache primitives
# --------------------------------------------------------------------

def test_signature_cache_hit_when_signature_matches(tmp_path: Path) -> None:
    cache = SignatureCache(max_size=4)
    key = tmp_path / "a"
    cache.put(key, "sig-1", {"v": 1})
    assert cache.get(key, "sig-1") == {"v": 1}


def test_signature_cache_miss_when_signature_changes(tmp_path: Path) -> None:
    cache = SignatureCache(max_size=4)
    key = tmp_path / "a"
    cache.put(key, "sig-1", {"v": 1})
    assert cache.get(key, "sig-2") is None


def test_signature_cache_evicts_lru(tmp_path: Path) -> None:
    cache = SignatureCache(max_size=2)
    cache.put(tmp_path / "a", "s", 1)
    cache.put(tmp_path / "b", "s", 2)
    cache.put(tmp_path / "c", "s", 3)
    assert cache.get(tmp_path / "a", "s") is None  # evicted
    assert cache.get(tmp_path / "b", "s") == 2
    assert cache.get(tmp_path / "c", "s") == 3


# --------------------------------------------------------------------
# scan_run_signals
# --------------------------------------------------------------------

def test_scan_run_signals_classifies_trial_dirs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    # Completed trial (has task_output.json)
    done = run / "trials" / "t1"
    done.mkdir(parents=True)
    _write_json(done / "task_output.json", {"output": "x"})
    # Running trial (has only progress.json)
    live = run / "trials" / "t2"
    live.mkdir(parents=True)
    _write_json(live / "proxy" / "progress.json", {"total_requests": 3})

    signals = scan_run_signals(run)

    assert signals.completed_count == 1
    assert signals.active_count == 1
    assert signals.newest_progress_mtime_ns > 0
    assert len(signals.progress_files) == 1
    assert signals.progress_files[0].name == "progress.json"


def test_scan_run_signals_supports_nested_layout(tmp_path: Path) -> None:
    run = tmp_path / "run"
    # nested challenge/variant/ layout
    nested = run / "trials" / "CVE-1" / "zero_day"
    nested.mkdir(parents=True)
    _write_json(nested / "proxy" / "progress.json", {"total_requests": 1})

    signals = scan_run_signals(run)
    assert signals.active_count == 1


def test_scan_run_signals_ttl_dedupes_repeat_calls(tmp_path: Path) -> None:
    run = tmp_path / "run"
    live = run / "trials" / "t1"
    live.mkdir(parents=True)
    _write_json(live / "proxy" / "progress.json", {"total_requests": 1})

    s1 = scan_run_signals(run)
    s2 = scan_run_signals(run)
    # Same identity object (memoised within the 1s window)
    assert s1 is s2


def test_is_recently_active_window() -> None:
    assert is_recently_active(time.time_ns())
    assert not is_recently_active(time.time_ns() - 600 * 1_000_000_000)
    assert not is_recently_active(0)
    assert not is_recently_active(-1)


# --------------------------------------------------------------------
# scan_runs caching behaviour
# --------------------------------------------------------------------

def test_scan_runs_returns_running_run_without_dashboard(tmp_path: Path) -> None:
    """A run with planned_trials.json but no dashboard.json must surface."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-live"
    )
    _write_json(
        run / "planned_trials.json",
        [{"trial_id": "t1", "trial_index": 0, "sample_id": "s1"}],
    )
    _write_json(
        run / "trials" / "t1" / "proxy" / "progress.json",
        {
            "total_requests": 2,
            "errors": 0,
            "last_ts_ms": int(time.time() * 1000),
        },
    )

    runs = scan_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].run_id == "run-live"
    assert runs[0].running is True
    assert runs[0].running_trials == 1
    assert runs[0].live_total_requests == 2


def test_scan_runs_discovers_run_from_experiment_record_without_legacy_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run with only canonical ExperimentRecord artifacts must surface."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    _write_canonical_run_snapshot(run, run_id="run-record")
    original_load_snapshot = ExperimentArtifactReader.load_snapshot
    load_snapshot_calls: list[Path] = []

    def load_snapshot_spy(self: ExperimentArtifactReader):
        load_snapshot_calls.append(self.run_dir)
        return original_load_snapshot(self)

    monkeypatch.setattr(ExperimentArtifactReader, "load_snapshot", load_snapshot_spy)

    runs = scan_runs(tmp_path)

    assert load_snapshot_calls == [run.resolve()]
    assert len(runs) == 1
    assert runs[0].run_id == "run-record"
    assert runs[0].experiment == "record-only-demo"
    assert runs[0].status == "running"
    assert runs[0].running is True
    assert runs[0].running_trials == 1
    assert runs[0].agents["agent:model:stateless"]["total"] == 1


def test_scan_runs_falls_back_to_legacy_record_json_when_snapshot_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical partial records still render when typed snapshot loading fails."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    _write_canonical_run_snapshot(run, run_id="run-record")
    load_snapshot_calls: list[Path] = []

    def load_snapshot_fails(self: ExperimentArtifactReader):
        load_snapshot_calls.append(self.run_dir)
        raise RuntimeError("simulated incomplete canonical snapshot")

    monkeypatch.setattr(ExperimentArtifactReader, "load_snapshot", load_snapshot_fails)

    runs = scan_runs(tmp_path)

    assert load_snapshot_calls == [run.resolve()]
    assert len(runs) == 1
    assert runs[0].run_id == "run-record"
    assert runs[0].experiment == "record-only-demo"
    assert runs[0].status == "running"
    assert runs[0].running_trials == 1
    assert runs[0].agents["agent:model:stateless"]["total"] == 1


def test_scan_runs_projects_canonical_score_summary_into_agent_payload(
    tmp_path: Path,
) -> None:
    """Offline canonical score summaries should appear on the index cards."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-scored"
    )
    _write_canonical_run_snapshot(run, run_id="run-scored")
    summary_path = run / "scores" / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": "score_summary.v1",
            "scores": {"canonical": {"count": 1, "mean": 1.0}},
        },
    )
    writer = ExperimentArtifactWriter(run)
    writer.mark_run_artifact(
        artifact_id="run.score_summary",
        path=summary_path,
        kind="score_summary",
        schema_version="score_summary.v1",
        producer="cage score",
        replayability="replayable",
    )
    writer.mark_run_scored(summary_ref="scores/summary.json")

    [info] = scan_runs(tmp_path)

    assert info.agents["agent:model:stateless"]["mean_scores"] == {"canonical": 1.0}


def test_load_dashboard_overlays_canonical_run_record_on_stale_dashboard(
    tmp_path: Path,
) -> None:
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    _write_canonical_run_snapshot(run, run_id="run-record")
    writer = ExperimentArtifactWriter(run)
    trial_id = ExperimentArtifactReader(run).load_snapshot().trial_records[0].trial_id
    writer.mark_run_started(started_at="2026-06-05T00:00:03Z")
    writer.mark_trial_finished(
        trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    writer.mark_run_finished(
        status="completed",
        completed_at="2026-06-05T00:01:05Z",
    )
    _write_json(
        run / "dashboard.json",
        {
            "run_id": "stale-run",
            "experiment": "stale-experiment",
            "started_at": "2026-06-04T00:00:00Z",
            "completed_at": "",
            "status": "running",
            "agents": {
                "agent:model:stateless": {
                    "total": 99,
                    "completed": 0,
                    "failed": 0,
                    "running": 99,
                    "trials": [{"trial_id": trial_id}],
                }
            },
        },
    )

    dashboard = load_dashboard(run)

    assert dashboard["run_id"] == "run-record"
    assert dashboard["experiment"] == "record-only-demo"
    assert dashboard["started_at"] == "2026-06-05T00:00:03Z"
    assert dashboard["completed_at"] == "2026-06-05T00:01:05Z"
    assert dashboard["status"] == "completed"
    assert dashboard["agents"]["agent:model:stateless"]["total"] == 1
    assert dashboard["agents"]["agent:model:stateless"]["completed"] == 1
    assert dashboard["agents"]["agent:model:stateless"]["running"] == 0
    assert dashboard["agents"]["agent:model:stateless"]["trials"] == [
        {"trial_id": trial_id}
    ]


def test_scan_runs_prefers_canonical_trial_counts_over_stale_legacy_trial_dirs(
    tmp_path: Path,
) -> None:
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    _write_canonical_run_snapshot(run, run_id="run-record")
    trial_id = ExperimentArtifactReader(run).load_snapshot().trial_records[0].trial_id
    ExperimentArtifactWriter(run).mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="agent_exit_nonzero",
        termination=TrialTermination(reason="agent_exit_nonzero", exit_code=2),
    )
    legacy_trial = run / "trials" / Path(*trial_id.split("/"))
    _write_json(
        legacy_trial / "meta.json",
        {
            "trial_id": trial_id,
            "termination_reason": "completed",
            "exit_code": 0,
        },
    )
    _write_json(legacy_trial / "task_output.json", {"output": "stale"})

    [info] = scan_runs(tmp_path)

    assert info.agents["agent:model:stateless"]["total"] == 1
    assert info.agents["agent:model:stateless"]["completed"] == 0
    assert info.agents["agent:model:stateless"]["failed"] == 1


def test_scan_runs_does_not_guess_unindexed_canonical_score_summary(
    tmp_path: Path,
) -> None:
    """Record score refs must be backed by the canonical artifact index."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-unindexed-score"
    )
    _write_canonical_run_snapshot(run, run_id="run-unindexed-score")
    _write_json(
        run / "scores" / "summary.json",
        {
            "schema_version": "score_summary.v1",
            "scores": {"canonical": {"count": 1, "mean": 1.0}},
        },
    )
    ExperimentArtifactWriter(run).mark_run_scored(summary_ref="scores/summary.json")

    [info] = scan_runs(tmp_path)

    assert "mean_scores" not in info.agents["agent:model:stateless"]


def test_scan_runs_recursive_prunes_git_worktrees(tmp_path: Path, monkeypatch) -> None:
    """Inspecting a repo root should not absorb sibling feature worktrees."""
    monkeypatch.setenv("CAGE_INSPECT_RECURSIVE", "1")
    visible = tmp_path / "project" / ".cage_runs" / "agent:model:stateless" / "run-visible"
    hidden = (
        tmp_path
        / ".worktrees"
        / "feature-copy"
        / "project"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-hidden"
    )
    for run_dir, run_id in ((visible, "run-visible"), (hidden, "run-hidden")):
        _write_json(
            run_dir / "dashboard.json",
            {
                "run_id": run_id,
                "started_at": "2026-04-28T10:00:00",
                "completed_at": "2026-04-28T11:00:00",
                "status": "completed",
                "agents": {"agent:model:stateless": {"trials": []}},
            },
        )

    runs = scan_runs(tmp_path)

    assert [run.run_id for run in runs] == ["run-visible"]


def test_scan_runs_ignores_top_level_logs_only_cage_runs(tmp_path: Path) -> None:
    """Repo-level target-server logs must not hide real project runs.

    Canonical layout: the real run lives one level below the inspect root, in
    ``<root>/<project>/.cage_runs/`` — the logs-only top-level ``.cage_runs/``
    must not shadow it.
    """
    logs_only = tmp_path / ".cage_runs"
    logs_only.mkdir()
    (logs_only / "target_server-tcheck-demo.log").write_text("ok\n", encoding="utf-8")
    run_dir = tmp_path / "demo" / ".cage_runs" / "agent:model:stateless" / "run-visible"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-visible",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )

    runs = scan_runs(tmp_path)

    assert [run.run_id for run in runs] == ["run-visible"]


def test_scan_runs_caches_completed_run_to_cheap_signature(tmp_path: Path) -> None:
    """Completed runs hit the cheap-signature cache path on every call."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-done"
    )
    _write_json(
        run / "dashboard.json",
        {
            "run_id": "run-done",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {
                "agent:model:stateless": {
                    "completed": 1, "failed": 0, "total": 1, "trials": [],
                },
            },
        },
    )

    # Warm caches.
    scan_runs(tmp_path)
    assert len(run_summary_cache) >= 1

    # Touch a per-trial mtime to simulate filesystem churn that should
    # NOT invalidate the completed-run cache (dashboard didn't move).
    trial = run / "trials" / "t1"
    trial.mkdir(parents=True, exist_ok=True)
    _write_json(trial / "task_output.json", {"output": "x"})

    runs = scan_runs(tmp_path)
    # The run is still classified as not running, with its cached info.
    assert any(r.run_id == "run-done" and not r.running for r in runs)


def test_scan_runs_treats_stale_unfinished_run_as_not_running(tmp_path: Path) -> None:
    """If progress.json hasn't moved in 5+ min, the run is not running."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-stale"
    )
    _write_json(
        run / "planned_trials.json",
        [{"trial_id": "t1", "trial_index": 0, "sample_id": "s1"}],
    )
    progress = run / "trials" / "t1" / "proxy" / "progress.json"
    _write_json(progress, {"total_requests": 5, "last_ts_ms": 0})
    # Backdate the file to ~1 hour ago.
    old = time.time() - 3600
    import os
    os.utime(progress, (old, old))

    runs = scan_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].running is False


# --------------------------------------------------------------------
# load_trial_summary_cached
# --------------------------------------------------------------------

def test_load_trial_summary_cached_hits_on_repeat(tmp_path: Path) -> None:
    trial = tmp_path / "run-1" / "trials" / "t1"
    _write_json(
        trial / "proxy" / "progress.json",
        {"total_requests": 4, "errors": 0, "last_ts_ms": 100, "tokens_in": 1, "tokens_out": 2},
    )

    a = load_trial_summary_cached(trial, {})
    b = load_trial_summary_cached(trial, {})
    assert a is b  # identical cached object


def test_load_trial_summary_cached_invalidates_on_mtime_change(tmp_path: Path) -> None:
    trial = tmp_path / "run-1" / "trials" / "t1"
    progress = trial / "proxy" / "progress.json"
    _write_json(progress, {"total_requests": 1, "last_ts_ms": 100})
    first = load_trial_summary_cached(trial, {})

    # Rewrite with new content + future mtime.
    _write_json(progress, {"total_requests": 7, "last_ts_ms": 200})
    import os
    future = time.time() + 1.0
    os.utime(progress, (future, future))
    second = load_trial_summary_cached(trial, {})

    assert second is not first
    assert second["progress"]["total_requests"] == 7


def test_trial_summary_signature_changes_with_progress_writes(tmp_path: Path) -> None:
    trial = tmp_path / "run" / "trials" / "t1"
    _write_json(trial / "proxy" / "progress.json", {"total_requests": 1})
    sig1 = trial_summary_signature(trial)

    import os
    future = time.time() + 5.0
    os.utime(trial / "proxy" / "progress.json", (future, future))
    sig2 = trial_summary_signature(trial)

    assert sig1 != sig2


def test_trial_summary_signature_tracks_indexed_artifacts_for_real_trial_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    trial_dir = run / "trials" / Path(*trial_id.split("/"))
    trial_dir.mkdir(parents=True, exist_ok=True)
    task_output_ref = "canonical_outputs/task_output.json"
    task_output_path = run / task_output_ref
    _write_json(
        task_output_path,
        {"output": "first", "sample": {"id": "sample-a"}},
    )
    ExperimentArtifactWriter(run).mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    sig1 = trial_summary_signature(trial_dir)

    _write_json(
        task_output_path,
        {"output": "second", "sample": {"id": "sample-a"}},
    )
    future = time.time() + 5.0
    os.utime(task_output_path, (future, future))
    sig2 = trial_summary_signature(trial_dir)

    assert sig1 != sig2


# --------------------------------------------------------------------
# Delta API
# --------------------------------------------------------------------

def _setup_one_running_run(root: Path) -> Path:
    run = root / "proj" / ".cage_runs" / "agent:model:stateless" / "run-A"
    _write_json(
        run / "planned_trials.json",
        [{"trial_id": "t1", "trial_index": 0, "sample_id": "s1"}],
    )
    _write_json(
        run / "trials" / "t1" / "proxy" / "progress.json",
        {
            "total_requests": 3,
            "errors": 0,
            "last_ts_ms": int(time.time() * 1000),
            "tokens_in": 10,
            "tokens_out": 5,
            "cost_usd": 0.0579,
        },
    )
    return run


def test_runs_delta_returns_all_when_since_is_zero(tmp_path: Path) -> None:
    _setup_one_running_run(tmp_path)
    runs = scan_runs(tmp_path)
    payload = build_runs_delta(runs, root=tmp_path, since_ms=0)
    assert payload["since_ms"] == 0
    assert len(payload["runs"]) == 1
    assert payload["summary"]["running_runs"] == 1
    assert payload["max_signature_ms"] > 0


def test_runs_delta_filters_settled_runs_above_cursor(tmp_path: Path) -> None:
    """A completed run with sig <= since must be omitted from the payload."""
    done = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-done"
    )
    _write_json(
        done / "dashboard.json",
        {
            "run_id": "run-done",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {
                "agent:model:stateless": {
                    "completed": 1, "failed": 0, "total": 1, "trials": [],
                },
            },
        },
    )
    _setup_one_running_run(tmp_path)

    runs = scan_runs(tmp_path)
    first = build_runs_delta(runs, root=tmp_path, since_ms=0)
    assert {r["run_id"] for r in first["runs"]} == {"run-A", "run-done"}

    later = build_runs_delta(runs, root=tmp_path, since_ms=first["max_signature_ms"])
    # The running run still ticks (its signature == max), but the completed
    # run is excluded because it's settled and below/at cursor.
    ids = {r["run_id"] for r in later["runs"]}
    assert "run-done" not in ids


def test_runs_delta_includes_settled_run_when_canonical_record_changes(
    tmp_path: Path,
) -> None:
    """Record-only updates such as offline scoring must refresh the run index."""
    done = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-done"
    )
    dashboard_path = done / "dashboard.json"
    _write_json(
        dashboard_path,
        {
            "run_id": "run-done",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {
                "agent:model:stateless": {
                    "completed": 1, "failed": 0, "total": 1, "trials": [],
                },
            },
        },
    )
    _write_canonical_run_snapshot(done, run_id="run-done")
    record_mtime_s = (done / "experiment_record.json").stat().st_mtime
    os.utime(dashboard_path, (record_mtime_s - 2, record_mtime_s - 2))

    [run] = scan_runs(tmp_path)
    since_ms = safe_mtime_ns(dashboard_path) // 1_000_000

    payload = build_runs_delta([run], root=tmp_path, since_ms=since_ms)

    assert [item["run_id"] for item in payload["runs"]] == ["run-done"]


def test_runs_delta_emits_completed_transition_after_start_cursor(tmp_path: Path) -> None:
    """Terminal dashboard writes must reach already-open index pages."""
    done = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-done"
    )
    _write_json(
        done / "dashboard.json",
        {
            "run_id": "run-done",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {
                "agent:model:stateless": {
                    "completed": 1, "failed": 0, "total": 1, "trials": [],
                },
            },
        },
    )

    [run] = scan_runs(tmp_path)
    # Simulate a browser cursor taken while the run was already started
    # but before dashboard.json recorded the terminal state.
    since_ms = int(time.mktime(time.strptime("2026-04-28T10:30:00", "%Y-%m-%dT%H:%M:%S"))) * 1000

    payload = build_runs_delta([run], root=tmp_path, since_ms=since_ms)

    assert [r["run_id"] for r in payload["runs"]] == ["run-done"]
    assert payload["runs"][0]["status"] == "completed"


def test_run_trials_delta_returns_compact_payload(tmp_path: Path) -> None:
    run = _setup_one_running_run(tmp_path)
    payload = build_run_trials_delta(run, root=tmp_path, since_ms=0)
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["running"] == 1
    [trial] = payload["trials"]
    assert trial["id"] == "t1"
    assert trial["status_kind"] == "running"
    assert trial["progress"]["total"] == 3
    assert trial["usage"]["input_tokens"] == 10
    assert trial["usage"]["cost_usd"] == 0.0579
    assert trial["signature_ms"] > 0


def test_run_trials_delta_keeps_running_trial_live_for_duration_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _setup_one_running_run(tmp_path)
    progress_path = run / "trials" / "t1" / "proxy" / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["started_at_ms"] = 1_000_000
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    now = {"value": 1_600.0}
    monkeypatch.setattr("cage.web.data.time.time", lambda: now["value"])
    first = build_run_trials_delta(run, root=tmp_path, since_ms=0)
    assert first["trials"][0]["duration_ms"] == 600_000

    now["value"] = 1_605.0
    second = build_run_trials_delta(run, root=tmp_path, since_ms=first["max_signature_ms"])

    assert second["trials"][0]["id"] == "t1"
    assert second["trials"][0]["duration_ms"] == 605_000


def test_run_trials_delta_includes_pending_planned_trials(tmp_path: Path) -> None:
    run = tmp_path / "proj" / ".cage_runs" / "agent:model:stateless" / "run-live"
    _write_json(
        run / "planned_trials.json",
        [
            {"trial_id": "running", "trial_index": 0, "sample_id": "s-running"},
            {"trial_id": "pending-a", "trial_index": 1, "sample_id": "s-pending-a"},
            {"trial_id": "group/pending-b", "trial_index": 2, "sample_id": "s-pending-b"},
        ],
    )
    _write_json(
        run / "trials" / "running" / "proxy" / "progress.json",
        {"total_requests": 3, "last_status": "tool", "last_ts_ms": int(time.time() * 1000)},
    )

    payload = build_run_trials_delta(run, root=tmp_path, since_ms=0)

    assert payload["summary"]["total"] == 3
    assert payload["summary"]["running"] == 1
    assert payload["summary"]["other"] == 2
    trials = {trial["id"]: trial for trial in payload["trials"]}
    assert trials["running"]["status_kind"] == "running"
    assert trials["pending-a"]["status_kind"] == "pending"
    assert trials["pending-a"]["status_label"] == "Pending"
    assert trials["pending-a"]["status_detail"] == "Not started yet"
    assert trials["pending-a"]["trial_index"] == 1
    assert trials["group/pending-b"]["status_kind"] == "pending"


def test_run_trials_delta_includes_canonical_trial_records_without_legacy_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    project_file = _write_contract_project(run.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-record",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    writer.mark_trial_started(trial_id, started_at="2026-06-05T00:00:05Z")
    writer.mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="agent_exit_nonzero",
        termination=TrialTermination(reason="agent_exit_nonzero", exit_code=2),
    )
    score_ref = f"trials/{trial_id}/scores/demo.json"
    _write_json(run / score_ref, {"demo": {"value": 1.0, "answer": "ok"}})
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.score.demo",
        path=score_ref,
        kind="trial_score",
        schema_version="trial_score.v1",
        producer="demo scorer",
        replayability="replayable",
    )
    writer.mark_trial_scored(trial_id, score_ref=score_ref, scoring_id="demo")
    _write_json(
        run / "planned_trials.json",
        [
            {
                "trial_id": trial_id,
                "trial_index": 0,
                "sample_id": "stale-sample",
                "status": "planned",
            }
        ],
    )
    original_load_snapshot = ExperimentArtifactReader.load_snapshot
    load_snapshot_calls: list[Path] = []

    def load_snapshot_spy(self: ExperimentArtifactReader):
        load_snapshot_calls.append(self.run_dir)
        return original_load_snapshot(self)

    monkeypatch.setattr(ExperimentArtifactReader, "load_snapshot", load_snapshot_spy)

    payload = build_run_trials_delta(run, root=tmp_path, since_ms=0)

    assert load_snapshot_calls == [run.resolve()]
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["failed"] == 1
    [trial] = payload["trials"]
    assert trial["id"] == trial_id
    assert trial["status_kind"] == "error"
    assert trial["status_label"] == "Agent failed"
    assert trial["status_detail"] == "Exited with code 2"
    assert trial["scores"] == {"demo": 1.0}


def test_run_trials_delta_prefers_canonical_record_over_stale_legacy_trial_dir(
    tmp_path: Path,
) -> None:
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    project_file = _write_contract_project(run.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-record",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    writer.mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="agent_exit_nonzero",
        termination=TrialTermination(reason="agent_exit_nonzero", exit_code=2),
    )
    legacy_trial = run / "trials" / Path(*trial_id.split("/"))
    _write_json(
        legacy_trial / "meta.json",
        {
            "trial_id": trial_id,
            "termination_reason": "completed",
            "exit_code": 0,
        },
    )
    _write_json(legacy_trial / "task_output.json", {"output": "stale"})

    payload = build_run_trials_delta(run, root=tmp_path, since_ms=0)

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["failed"] == 1
    [trial] = payload["trials"]
    assert trial["id"] == trial_id
    assert trial["status_kind"] == "error"
    assert trial["status_label"] == "Agent failed"


def test_run_trials_delta_does_not_guess_unindexed_canonical_trial_score(
    tmp_path: Path,
) -> None:
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    project_file = _write_contract_project(run.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-record",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    writer.mark_trial_finished(
        trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    score_ref = f"trials/{trial_id}/scores/demo.json"
    _write_json(run / score_ref, {"demo": {"value": 1.0, "answer": "ok"}})
    writer.mark_trial_scored(trial_id, score_ref=score_ref, scoring_id="demo")

    payload = build_run_trials_delta(run, root=tmp_path, since_ms=0)

    [trial] = payload["trials"]
    assert trial["scores"] == {}


def test_run_trials_delta_refreshes_when_canonical_score_artifact_changes(
    tmp_path: Path,
) -> None:
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    project_file = _write_contract_project(run.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    writer = ExperimentArtifactWriter(run)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-record",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    writer.mark_trial_finished(
        trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    score_ref = f"trials/{trial_id}/scores/demo.json"
    score_path = run / score_ref
    _write_json(score_path, {"demo": {"value": 1.0}})
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.score.demo",
        path=score_ref,
        kind="trial_score",
        schema_version="trial_score.v1",
        producer="demo scorer",
        replayability="replayable",
    )
    writer.mark_trial_scored(trial_id, score_ref=score_ref, scoring_id="demo")
    first = build_run_trials_delta(run, root=tmp_path, since_ms=0)
    _write_json(score_path, {"demo": {"value": 2.0}})
    _bump_mtime(score_path)

    later = build_run_trials_delta(run, root=tmp_path, since_ms=first["max_signature_ms"])

    [trial] = later["trials"]
    assert trial["id"] == trial_id
    assert trial["scores"] == {"demo": 2.0}


def test_run_trials_delta_skips_unchanged_settled_trials_above_cursor(tmp_path: Path) -> None:
    run = _setup_one_running_run(tmp_path)
    _write_json(run / "trials" / "t1" / "task_output.json", {"output": "done"})
    first = build_run_trials_delta(run, root=tmp_path, since_ms=0)
    # Poll again with the cursor set to last seen signature → no deltas.
    later = build_run_trials_delta(run, root=tmp_path, since_ms=first["max_signature_ms"])
    assert later["trials"] == []
    assert later["summary"]["total"] == 1  # counts still accurate


# --------------------------------------------------------------------
# Flask routes — integration
# --------------------------------------------------------------------

def test_api_runs_route_returns_json(tmp_path: Path) -> None:
    _setup_one_running_run(tmp_path)
    app = create_app(tmp_path)
    client = app.test_client()
    r = client.get("/api/runs")
    assert r.status_code == 200
    data = r.get_json()
    assert "now_ms" in data
    assert "max_signature_ms" in data
    assert len(data["runs"]) == 1


def test_api_trajectory_resolves_indexed_proxy_log_without_legacy_trial_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))
    encoded = base64.urlsafe_b64encode(
        str(virtual_trial_dir.resolve().relative_to(tmp_path.resolve())).encode()
    ).decode()
    client = create_app(tmp_path).test_client()

    trajectory = client.get(f"/api/trajectory/{encoded}?offset=0&limit=10")
    context = client.get(f"/api/trajectory/{encoded}/step/0")

    assert trajectory.status_code == 200
    payload = trajectory.get_json()
    assert payload["total_steps"] == 1
    assert payload["summary"]["total_tokens"]["in"] == 2
    assert payload["summary"]["total_tokens"]["out"] == 1
    assert context.status_code == 200
    assert context.get_json()["response"]["content"] == [
        {"type": "text", "text": "ok"}
    ]


def test_load_trial_marks_indexed_proxy_log_as_trajectory_available(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))

    trial = load_trial(virtual_trial_dir)

    assert trial.has_trajectory is True


def test_load_trial_hydrates_indexed_task_output_and_scores_without_legacy_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    writer = ExperimentArtifactWriter(run)
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {
            "output": "finished exploit",
            "sample": {"id": "sample-a", "difficulty": "smoke"},
        },
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    score_ref = "canonical_scores/demo.json"
    _write_json(run / score_ref, {"demo": {"value": 0.75, "answer": "ok"}})
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.score.demo",
        path=score_ref,
        kind="trial_score",
        schema_version="trial_score.v1",
        producer="test scorer",
        replayability="replayable",
    )
    writer.mark_trial_scored(trial_id, score_ref=score_ref, scoring_id="demo")
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))

    trial = load_trial(virtual_trial_dir)

    assert trial.output == "finished exploit"
    assert trial.sample == {"id": "sample-a", "difficulty": "smoke"}
    assert trial.scores == {"demo": 0.75}


def test_project_trial_route_accepts_canonical_trial_without_legacy_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {"output": "canonical detail output", "sample": {"id": "sample-a"}},
    )
    ExperimentArtifactWriter(run).mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    client = create_app(tmp_path).test_client()

    response = client.get(
        f"/projects/proj/runs/{run.name}/trials/{trial_id}",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"canonical detail output" in response.data


def test_agent_trial_route_accepts_canonical_trial_without_legacy_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {"output": "canonical agent route output", "sample": {"id": "sample-a"}},
    )
    ExperimentArtifactWriter(run).mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    client = create_app(tmp_path).test_client()

    response = client.get(
        f"/trial/{run.parent.name}/{run.name}/{trial_id}",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"canonical agent route output" in response.data


def test_load_trial_hydrates_indexed_prompt_without_legacy_file(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    prompt_ref = "canonical_prompts/prompt.txt"
    prompt_path = run / prompt_ref
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("canonical prompt text", encoding="utf-8")
    ExperimentArtifactWriter(run).mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.prompt",
        path=prompt_ref,
        kind="prompt",
        schema_version="prompt.txt.v1",
        producer="test",
        replayability="replayable",
        content_type="text/plain",
    )
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))

    trial = load_trial(virtual_trial_dir)

    assert trial.prompt == "canonical prompt text"


def test_load_trial_summary_hydrates_indexed_task_output_and_scores_without_legacy_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    writer = ExperimentArtifactWriter(run)
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {
            "output": "finished exploit",
            "sample": {
                "id": "sample-a",
                "category": "web",
                "tags": ["smoke"],
            },
        },
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    score_ref = "canonical_scores/demo.json"
    _write_json(run / score_ref, {"demo": {"value": 0.75, "answer": "ok"}})
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.score.demo",
        path=score_ref,
        kind="trial_score",
        schema_version="trial_score.v1",
        producer="test scorer",
        replayability="replayable",
    )
    writer.mark_trial_scored(trial_id, score_ref=score_ref, scoring_id="demo")
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))

    summary = load_trial_summary_cached(virtual_trial_dir, {})

    assert summary["output"] == "finished exploit"
    assert summary["scores"] == {"demo": 0.75}
    assert "category:web" in summary["tags"]
    assert "smoke" in summary["tags"]


def test_api_trial_summary_accepts_canonical_trial_record_without_legacy_dir(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    writer = ExperimentArtifactWriter(run)
    writer.mark_trial_started(trial_id, started_at="2026-06-05T00:00:05Z")
    writer.mark_trial_finished(
        trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {"output": "finished exploit", "sample": {"id": "sample-a"}},
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    score_ref = "canonical_scores/demo.json"
    _write_json(run / score_ref, {"demo": {"value": 0.75, "answer": "ok"}})
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.score.demo",
        path=score_ref,
        kind="trial_score",
        schema_version="trial_score.v1",
        producer="test scorer",
        replayability="replayable",
    )
    writer.mark_trial_scored(trial_id, score_ref=score_ref, scoring_id="demo")
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))
    encoded = base64.urlsafe_b64encode(
        str(virtual_trial_dir.resolve().relative_to(tmp_path.resolve())).encode()
    ).decode()
    client = create_app(tmp_path).test_client()

    response = client.get(f"/api/trial/{encoded}/summary")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["present"] is True
    assert payload["summary"]["trial_id"] == trial_id
    assert payload["summary"]["benchmark_outcome"]["kind"] == "passed"


def test_load_trial_summary_uses_canonical_trial_record_status_without_meta_json(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    writer = ExperimentArtifactWriter(run)
    writer.mark_trial_started(trial_id, started_at="2026-06-05T00:00:05Z")
    writer.mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="agent_exit_nonzero",
        termination=TrialTermination(reason="agent_exit_nonzero", exit_code=2),
    )
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {"output": "", "sample": {"id": "sample-a"}},
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))

    summary = load_trial_summary_cached(virtual_trial_dir, {})

    assert summary["trial_id"] == trial_id
    assert summary["exit_code"] == 2
    assert summary["status_kind"] == "error"
    assert summary["status_label"] == "Agent failed"
    assert summary["status_detail"] == "Exited with code 2"


def test_load_trial_summary_does_not_mark_indexed_output_trial_running(
    tmp_path: Path,
) -> None:
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(tmp_path)
    writer = ExperimentArtifactWriter(run)
    writer.mark_trial_started(trial_id, started_at="2026-06-05T00:00:05Z")
    writer.mark_trial_finished(
        trial_id,
        status="completed",
        completed_at="2026-06-05T00:01:00Z",
    )
    task_output_ref = "canonical_outputs/task_output.json"
    _write_json(
        run / task_output_ref,
        {"output": "finished exploit", "sample": {"id": "sample-a"}},
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))
    _write_json(
        virtual_trial_dir / "proxy" / "progress.json",
        {
            "total_requests": 1,
            "successful_requests": 1,
            "tokens_in": 10,
            "tokens_out": 5,
        },
    )

    summary = load_trial_summary_cached(virtual_trial_dir, {})

    assert summary["running"] is False
    assert summary["status_kind"] == "success"
    assert summary["status_label"] == "Completed"


def test_api_run_trials_route_uses_encoded_path(tmp_path: Path) -> None:
    run = _setup_one_running_run(tmp_path)
    app = create_app(tmp_path)
    client = app.test_client()
    rel = str(run.resolve().relative_to(tmp_path.resolve()))
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()
    r = client.get(f"/api/run/{encoded}/trials")
    assert r.status_code == 200
    data = r.get_json()
    assert data["summary"]["total"] == 1
    assert data["trials"][0]["id"] == "t1"


def test_run_detail_page_renders_for_live_run_without_dashboard(tmp_path: Path) -> None:
    """The dashboard-gate must not block running runs from /run/."""
    run = _setup_one_running_run(tmp_path)
    app = create_app(tmp_path)
    client = app.test_client()
    rel = str(run.resolve().relative_to(tmp_path.resolve()))
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()
    r = client.get(f"/run/{encoded}", follow_redirects=True)
    assert r.status_code == 200
    body = r.data.decode()
    assert "run-live-banner" in body
    assert b"data-trial-id" in r.data


def test_run_detail_page_renders_for_canonical_run_without_legacy_files(
    tmp_path: Path,
) -> None:
    """A canonical ExperimentRecord is enough to open the run detail page."""
    run = (
        tmp_path
        / "proj"
        / ".cage_runs"
        / "agent:model:stateless"
        / "run-record"
    )
    _write_canonical_run_snapshot(run, run_id="run-record")
    app = create_app(tmp_path)
    client = app.test_client()
    rel = str(run.resolve().relative_to(tmp_path.resolve()))
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()

    r = client.get(f"/run/{encoded}", follow_redirects=True)

    assert r.status_code == 200
    body = r.data.decode()
    assert "run-record" in body
    assert "sample-a" in body


def test_dashboard_page_renders_live_snapshot_without_dashboard_view(tmp_path: Path) -> None:
    """The dashboard route should show current state before final artifacts exist."""
    run = _setup_one_running_run(tmp_path)
    app = create_app(tmp_path)
    client = app.test_client()
    rel = str(run.resolve().relative_to(tmp_path.resolve()))
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()

    r = client.get(f"/run/{encoded}/dashboard")

    assert r.status_code == 200
    body = r.data.decode()
    assert "Live snapshot" in body
    assert "No dashboard view yet" not in body
    assert "t1" in body
    assert "Running" in body
    assert "3" in body


def test_dashboard_api_returns_live_snapshot_without_dashboard_view(tmp_path: Path) -> None:
    """The dashboard poller must not 404 while the page renders a live snapshot."""
    run = _setup_one_running_run(tmp_path)
    app = create_app(tmp_path)
    client = app.test_client()
    rel = str(run.resolve().relative_to(tmp_path.resolve()))
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()

    r = client.get(f"/api/run/{encoded}/dashboard_view")

    assert r.status_code == 200
    payload = r.get_json()
    assert payload["present"] is True
    assert payload["changed"] is True
    assert payload["mode"] == "live"
    assert payload["view"]["title"] == "Current run dashboard"
    assert payload["freshness"]["source"] == "live artifacts"
    assert payload["max_signature_ms"] > 0


def test_dashboard_page_prefers_benchmark_snapshot_before_final_view(tmp_path: Path) -> None:
    """Dashboard should use benchmark semantics, including score columns, when available."""
    run = tmp_path / "proj" / ".cage_runs" / "agent:model:stateless" / "run-score"
    run.mkdir(parents=True)
    (run / "models.yml").write_text(
        """
models:
  dummy:
    provider: openai
    model: gpt-test
""".strip(),
        encoding="utf-8",
    )
    (run / "project.yml").write_text(
        """
project:
  name: demo
models_file: models.yml
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBench
agents: []
""".strip(),
        encoding="utf-8",
    )
    (run / "benchmark.py").write_text(
        """
from cage.benchmarks import Benchmark
from cage.artifacts.dashboard import Column, Dashboard, Section, Stat


class DemoBench(Benchmark):
    name = "demo"

    def iter_samples(self):
        return iter(())

    def prepare_trial(self, container, sample, workspace_dir):
        return None

    def build_prompt(self, sample):
        return ""

    def scorer(self):
        return None

    def build_dashboard(self, run_dir):
        return Dashboard(
            title="Benchmark score dashboard",
            sections=(
                Section(
                    kind="summary",
                    title="Score summary",
                    stats=(Stat(label="Mean score", value="0.875"),),
                ),
                Section(
                    kind="table",
                    title="Per target",
                    columns=(Column(key="target", label="Target"), Column(key="score", label="Score", align="right")),
                    rows=({"target": "range-1", "score": "0.875"},),
                ),
            ),
        )
""".strip(),
        encoding="utf-8",
    )

    app = create_app(tmp_path)
    client = app.test_client()
    rel = str(run.resolve().relative_to(tmp_path.resolve()))
    encoded = base64.urlsafe_b64encode(rel.encode()).decode()

    r = client.get(f"/run/{encoded}/dashboard")

    assert r.status_code == 200
    body = r.data.decode()
    assert "Benchmark score dashboard" in body
    assert "Mean score" in body
    assert "0.875" in body
    assert "Current run dashboard" not in body
    assert "No dashboard view yet" not in body


def test_dashboard_page_loads_benchmark_from_project_dir_when_run_snapshot_is_provenance_only(
    tmp_path: Path,
) -> None:
    """Run project.yml snapshots may not include benchmark.py; inspect root should fill that in."""
    project_dir = tmp_path / "proj"
    run = project_dir / ".cage_runs" / "agent:model:stateless" / "run-score"
    run.mkdir(parents=True)
    project_yaml = """
project:
  name: score-demo
models_file: models.yml
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBench
agents: []
""".strip()
    (run / "project.yml").write_text(project_yaml, encoding="utf-8")
    (project_dir / "project_score.yml").write_text(project_yaml, encoding="utf-8")
    (project_dir / "models.yml").write_text(
        """
models:
  dummy:
    provider: openai
    model: gpt-test
""".strip(),
        encoding="utf-8",
    )
    (project_dir / "benchmark.py").write_text(
        """
from cage.benchmarks import Benchmark
from cage.artifacts.dashboard import Dashboard, Section, Stat


class DemoBench(Benchmark):
    name = "demo"

    def iter_samples(self):
        return iter(())

    def prepare_trial(self, container, sample, workspace_dir):
        return None

    def build_prompt(self, sample):
        return ""

    def scorer(self):
        return None

    def build_dashboard(self, run_dir):
        return Dashboard(
            title="Project directory dashboard",
            sections=(Section(kind="summary", title="Score summary", stats=(Stat(label="Mean score", value="0.625"),)),),
        )
""".strip(),
        encoding="utf-8",
    )

    app = create_app(project_dir)
    client = app.test_client()
    r = client.get("/run/agent:model:stateless/run-score/dashboard", follow_redirects=True)

    assert r.status_code == 200
    body = r.data.decode()
    assert "Project directory dashboard" in body
    assert "Mean score" in body
    assert "0.625" in body
    assert "No dashboard view yet" not in body


def test_progress_cache_hits_until_mtime_moves(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    _write_json(progress, {"total_requests": 1})
    a = _read_progress_cached(progress)
    b = _read_progress_cached(progress)
    assert a == b
    assert a is b  # cache hit returns the same parsed dict


# --------------------------------------------------------------------
# Live-check audit semantics
# --------------------------------------------------------------------

def _setup_completed_no_live_success(root: Path) -> Path:
    """Trial that exited cleanly but never satisfied the live check —
    the "Max rounds" case we want to surface differently from a real win."""
    run = root / "proj" / ".cage_runs" / "agent:model:stateless" / "run-exhausted"
    trial = run / "trials" / "challenge-x" / "pass_1"
    _write_json(
        trial / "meta.json",
        {
            "trial_id": "challenge-x/pass_1",
            "trial_index": 0,
            "exit_code": 0,
            "status": "completed",
            "termination_reason": "completed",
            "termination_detail": "Task finished",
            "termination_source": "agent",
            "live_success": False,
            "live_success_verdict": {},
            "timing": {"duration_ms": 1000},
        },
    )
    _write_json(trial / "task_output.json", {"output": "ran out of rounds"})
    # Polls all returned false → live-check was enabled but never passed.
    polls_path = trial / "runtime" / "check_done_polls.jsonl"
    polls_path.parent.mkdir(parents=True, exist_ok=True)
    polls_path.write_text(
        "\n".join(
            json.dumps({
                "mode": "polling", "source": "check_done", "poll_index": i,
                "output": json.dumps({"message": "Attack unsuccessful", "status": False}),
            }) for i in range(1, 6)
        ),
        encoding="utf-8",
    )
    return trial


def test_classify_relabels_completed_without_live_success_as_exhausted(tmp_path: Path) -> None:
    trial_dir = _setup_completed_no_live_success(tmp_path)
    summary = load_trial_summary_cached(trial_dir, {})
    # The old behaviour would mark this 'success' (green); we now want 'warning'.
    assert summary["status_kind"] == "warning"
    assert summary["status_label"] == "Max rounds"
    assert "never satisfied" in summary["status_detail"].lower()


def test_classify_keeps_live_success_kind_distinct_from_warning(tmp_path: Path) -> None:
    """Trials terminated by live_success get their own status_kind so
    the UI banner can count them separately and the row badge can be amber."""
    trial = tmp_path / "run" / "trials" / "challenge-y" / "pass_1"
    _write_json(
        trial / "meta.json",
        {
            "trial_id": "challenge-y/pass_1",
            "trial_index": 0,
            "exit_code": 0,
            "termination_reason": "live_success",
            "termination_detail": "Stopped after a successful live-check verdict.",
            "live_success": True,
            "live_success_verdict": {
                "success": True,
                "source": "check_done",
                "mode": "polling",
                "evidence": {
                    "check_done": {"message": "Privilege escalation successful", "status": True},
                },
            },
        },
    )
    _write_json(trial / "task_output.json", {"output": "exit"})
    summary = load_trial_summary_cached(trial, {})
    assert summary["status_kind"] == "live_success"
    assert summary["status_label"] == "Target passed"


def test_classify_does_not_relabel_completed_when_live_check_was_not_configured(tmp_path: Path) -> None:
    """Regression: the orchestrator writes ``live_success: False`` to every
    trial's meta.json regardless of whether live-check was configured.
    The old heuristic in ``_live_check_was_enabled`` treated the presence
    of that key as a signal — turning every cleanly-completed trial in a
    agent-pentest-bench / nyu run (where ``live_check.enabled: false``) into a
    yellow "Max rounds" badge. ``runtime/check_done_polls.jsonl`` is the
    only reliable signal.
    """
    trial = tmp_path / "run" / "trials" / "challenge-x" / "pass_1"
    _write_json(
        trial / "meta.json",
        {
            "trial_id": "challenge-x/pass_1",
            "trial_index": 0,
            "exit_code": 0,
            "status": "completed",
            "termination_reason": "completed",
            "termination_detail": "Task finished",
            "live_success": False,
            "live_success_verdict": {},
            "timing": {"duration_ms": 1000},
        },
    )
    _write_json(trial / "task_output.json", {"output": "done"})
    # NOTE: no runtime/check_done_polls.jsonl — live-check was never wired up.
    summary = load_trial_summary_cached(trial, {})
    assert summary["status_kind"] == "success"
    assert summary["status_label"] == "Completed"


def test_target_unavailable_trial_reports_error_kind_with_target_label(tmp_path: Path) -> None:
    """Trials that fail-fast because their target stack never launched must
    surface as a red, distinctly-labelled row — not the generic
    'Target Unavailable' from the catch-all title-case branch."""
    trial = tmp_path / "run" / "trials" / "pb-siyucms" / "pass_2"
    _write_json(
        trial / "meta.json",
        {
            "trial_id": "pb-siyucms/pass_2",
            "trial_index": 0,
            "exit_code": -1,
            "status": "failed",
            "termination_reason": "target_unavailable",
            "termination_detail": "target_status=stopped — server-side launch failed",
            "termination_source": "orchestrator",
            "live_success": False,
            "live_success_verdict": {},
        },
    )
    _write_json(trial / "task_output.json", {"output": ""})
    summary = load_trial_summary_cached(trial, {})
    assert summary["status_label"] == "Target unavailable"
    assert summary["status_kind"] == "error"
    assert "stopped" in summary["status_detail"].lower()


def test_target_unavailable_inline_chip_uses_first_line_when_detail_is_multiline(
    tmp_path: Path,
) -> None:
    """When the orchestrator now embeds the full upstream response body
    (multi-line docker compose log) in ``termination_detail``, the inline
    summary chip on the trial header and trial-list row must not balloon
    into a wall of text — it should show the first non-empty line and
    leave the full body to the Termination block."""
    trial = tmp_path / "run" / "trials" / "pb-siyucms" / "pass_2"
    multiline_detail = (
        "target_status=stopped — server-side launch failed:\n"
        "\n"
        "500 Internal Server Error for url: http://127.0.0.1:42743/launch/pb-siyucms\n"
        "body: {\"detail\":\"Docker up failed: ...\n"
        "Container runtime-mysql-1  Error\n"
        "dependency failed to start: container runtime-mysql-1 is unhealthy\"}"
    )
    _write_json(
        trial / "meta.json",
        {
            "trial_id": "pb-siyucms/pass_2",
            "trial_index": 0,
            "exit_code": -1,
            "status": "failed",
            "termination_reason": "target_unavailable",
            "termination_detail": multiline_detail,
            "termination_source": "orchestrator",
        },
    )
    _write_json(trial / "task_output.json", {"output": ""})
    summary = load_trial_summary_cached(trial, {})
    # The chip detail is the first non-empty line, no newlines.
    assert summary["status_label"] == "Target unavailable"
    assert summary["status_kind"] == "error"
    assert "\n" not in summary["status_detail"]
    assert summary["status_detail"].startswith("target_status=stopped")
    # And termination.detail (rendered in the trial.html Termination
    # block with whitespace-pre-wrap) still has the full body so the
    # operator sees the real cause without opening a log file.
    from cage.web.data import build_trial_termination
    term = build_trial_termination(summary)
    assert "mysql-1 is unhealthy" in term["detail"]
    assert "Docker up failed" in term["detail"]


def test_completed_trial_with_live_success_true_still_counts_as_success(tmp_path: Path) -> None:
    """A trial whose termination_reason='completed' AND live_success=True
    is a real win — must NOT be relabelled to Max rounds."""
    trial = tmp_path / "run" / "trials" / "challenge-z" / "pass_1"
    _write_json(
        trial / "meta.json",
        {
            "trial_id": "challenge-z/pass_1",
            "trial_index": 0,
            "exit_code": 0,
            "termination_reason": "completed",
            "termination_detail": "Task finished",
            "live_success": True,
            "live_success_verdict": {"success": True, "source": "check_done"},
        },
    )
    _write_json(trial / "task_output.json", {"output": "done"})
    summary = load_trial_summary_cached(trial, {})
    assert summary["status_kind"] == "success"
    assert summary["status_label"] == "Completed"


def test_load_live_check_evidence_summarises_polls(tmp_path: Path) -> None:
    trial = tmp_path / "run" / "trials" / "challenge-z" / "pass_1"
    trial.mkdir(parents=True)
    (trial / "runtime").mkdir()
    polls = "\n".join([
        json.dumps({"poll_index": i, "mode": "polling", "source": "check_done",
                    "output": json.dumps({"message": "Attack unsuccessful", "status": False})})
        for i in range(1, 7)
    ] + [
        json.dumps({"poll_index": 7, "mode": "polling", "source": "check_done",
                    "output": json.dumps({"message": "RCE successful", "status": True})}),
    ])
    (trial / "runtime/check_done_polls.jsonl").write_text(polls, encoding="utf-8")
    _write_json(
        trial / "runtime/live_success.json",
        {
            "success": True, "source": "check_done", "mode": "polling",
            "evidence": {
                "check_done": {"message": "RCE successful", "status": True},
                "poll_index": 7,
            },
        },
    )

    evidence = load_live_check_evidence(trial)
    assert evidence is not None
    assert evidence["total_polls"] == 7
    assert evidence["first_true_index"] == 7
    assert evidence["had_passes"] is True
    assert evidence["verdict"]["evidence"]["check_done"]["message"] == "RCE successful"
    assert evidence["polls"][0]["status"] is False
    assert evidence["polls"][-1]["status"] is True


def test_load_live_check_evidence_returns_none_when_no_artifacts(tmp_path: Path) -> None:
    trial = tmp_path / "run" / "trials" / "no-live-check"
    trial.mkdir(parents=True)
    assert load_live_check_evidence(trial) is None


def test_delta_summary_buckets_sum_to_total(tmp_path: Path) -> None:
    """The five visible counters (running/completed/live_success/warnings/failed)
    plus 'other' must always sum to total."""
    base = tmp_path / "proj" / ".cage_runs" / "agent:model:stateless" / "run-mix"
    # one live_success
    _write_json(
        base / "trials" / "a" / "meta.json",
        {"trial_id": "a", "termination_reason": "live_success",
         "live_success": True, "live_success_verdict": {"success": True}},
    )
    _write_json(base / "trials" / "a" / "task_output.json", {"output": "x"})
    # one exhausted
    _setup_completed_no_live_success(tmp_path)  # writes to its own run
    # one failed
    _write_json(
        base / "trials" / "b" / "meta.json",
        {"trial_id": "b", "termination_reason": "agent_exit_nonzero", "exit_code": 1},
    )
    _write_json(base / "trials" / "b" / "task_output.json", {"output": ""})

    payload = build_run_trials_delta(base, root=tmp_path, since_ms=0)
    s = payload["summary"]
    assert s["live_success"] == 1
    assert s["failed"] == 1
    # Sum the visible parts
    parts = sum(
        s[k] for k in ("running", "completed", "live_success", "warnings", "failed", "other")
    )
    assert parts == s["total"]
