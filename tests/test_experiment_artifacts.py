import json
from pathlib import Path

import pytest

from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.resources import ResourceLedgerReader, ResourceLedgerWriter
from cage.artifacts.trial_session import TrialRuntimeSession
from cage.artifacts.writer import ExperimentArtifactWriter, ExperimentEventWriter
from cage.experiment.model import (
    ExperimentPlan,
    ExperimentSpec,
    TrialTermination,
    build_experiment_plan,
    load_experiment_spec,
)
from cage.scoring import ScoringContext


def _write_contract_project(project_dir: Path) -> Path:
    """Create a tiny project file for artifact-writer tests.

    The benchmark module intentionally raises if imported. The writer receives
    already-built contract objects, so writing initial run artifacts must not
    import or set up benchmark code.
    """

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "benchmark.py").write_text(
        'raise RuntimeError("artifact writer imported benchmark code")\n',
        encoding="utf-8",
    )
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: artifact-smoke
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBenchmark
    prompt_levels: [l2]
runtime:
  timeout: 600
  max_trials_global: 1
  passk: 1
  max_rounds: 3
agents:
  - id: claude_code
    kind: claude_code
    model: deepseek-v4-pro
""".lstrip(),
        encoding="utf-8",
    )
    return project_file


def _build_plan(project_dir: Path) -> tuple[ExperimentSpec, ExperimentPlan]:
    """Build a side-effect-free spec and plan for one sample."""

    project_file = _write_contract_project(project_dir)
    spec = load_experiment_spec(project_file, sample_ids=("pb-siyucms",))
    return spec, build_experiment_plan(spec)


def test_experiment_artifact_writer_writes_initial_canonical_snapshot(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "claude_code:deepseek-v4-pro:stateless" / "run-1"

    snapshot = ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    assert snapshot.spec_path == run_dir / "experiment_spec.json"
    assert snapshot.plan_path == run_dir / "experiment_plan.json"
    assert snapshot.record_path == run_dir / "experiment_record.json"
    assert snapshot.artifact_index_path == run_dir / "artifact_index.json"
    assert all(path.exists() for path in snapshot.trial_record_paths.values())
    assert all(path.exists() for path in snapshot.trial_event_log_paths.values())
    assert all(
        (path.parent / "resources.json").exists()
        for path in snapshot.trial_record_paths.values()
    )

    saved_plan = json.loads(snapshot.plan_path.read_text(encoding="utf-8"))
    saved_record = json.loads(snapshot.record_path.read_text(encoding="utf-8"))
    saved_index = json.loads(snapshot.artifact_index_path.read_text(encoding="utf-8"))

    assert saved_plan["plan_id"] == plan.plan_id
    assert saved_record["run_id"] == "run-1"
    assert saved_record["status"] == "planned"
    assert saved_record["spec_ref"] == "experiment_spec.json"
    assert saved_record["plan_ref"] == "experiment_plan.json"
    assert saved_record["event_log_ref"] == "events.jsonl"
    assert saved_record["resource_ledger_ref"] == "resources.jsonl"
    assert saved_record["trials"]["total"] == len(plan.trials)
    assert saved_record["trials"]["records"][0]["trial_id"] == plan.trials[0].trial_id
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "resources.jsonl").is_file()

    artifact_paths = {artifact["path"] for artifact in saved_index["artifacts"]}
    assert {
        "experiment_spec.json",
        "experiment_plan.json",
        "experiment_record.json",
        "artifact_index.json",
        "events.jsonl",
        "resources.jsonl",
    }.issubset(artifact_paths)
    first_trial_ref = saved_record["trials"]["records"][0]["record_ref"]
    first_trial_events_ref = str(Path(first_trial_ref).parent / "events.jsonl")
    first_trial_resources_ref = str(Path(first_trial_ref).parent / "resources.json")
    assert first_trial_ref in artifact_paths
    assert first_trial_events_ref in artifact_paths
    assert first_trial_resources_ref in artifact_paths
    first_trial_path = run_dir / first_trial_ref
    first_trial = json.loads(first_trial_path.read_text(encoding="utf-8"))
    first_trial_artifact_paths = {artifact["path"] for artifact in first_trial["artifacts"]}
    assert (first_trial_path.parent / "events.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads(
        (first_trial_path.parent / "resources.json").read_text(encoding="utf-8")
    ) == {
        "resources": [],
        "run_id": "run-1",
        "schema_version": "trial_resources.v1",
        "trial_id": plan.trials[0].trial_id,
    }
    assert (first_trial_path.parent / first_trial["plan_ref"]).resolve() == snapshot.plan_path
    assert "events.jsonl" in first_trial_artifact_paths
    assert "resources.json" in first_trial_artifact_paths


def test_experiment_artifact_writer_overwrites_json_atomically(tmp_path: Path) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment_plan.json").write_text("{not-json", encoding="utf-8")

    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    assert json.loads((run_dir / "experiment_plan.json").read_text(encoding="utf-8"))[
        "plan_id"
    ] == plan.plan_id
    assert not list(run_dir.rglob("*.tmp"))


def test_experiment_event_writer_appends_trial_events(tmp_path: Path) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    event = ExperimentEventWriter(run_dir).append_trial_event(
        run_id="run-1",
        trial_id=plan.trials[0].trial_id,
        phase="running",
        event_type="trial_started",
        timestamp="2026-06-05T00:00:01Z",
        payload={"target_id": "target-1"},
        artifact_refs=("trials/demo/record.json",),
    )

    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    saved = json.loads(lines[0])
    assert event.event_id == "evt_000001"
    assert saved["schema_version"] == "trial_event.v1"
    assert saved["event_id"] == "evt_000001"
    assert saved["run_id"] == "run-1"
    assert saved["trial_id"] == plan.trials[0].trial_id
    assert saved["phase"] == "running"
    assert saved["type"] == "trial_started"
    assert saved["payload"] == {"target_id": "target-1"}
    assert saved["artifact_refs"] == ["trials/demo/record.json"]


def test_resolve_trial_artifacts_returns_indexed_files_and_excludes_missing(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    reader = ExperimentArtifactReader(run_dir)

    # Attach a durable artifact (record + index, run-relative path) and resolve.
    proxy_file = run_dir / "proxy-log.jsonl"
    proxy_file.write_text("{}\n", encoding="utf-8")
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="proxy-1",
        path=proxy_file.name,
        kind="proxy_jsonl",
    )
    record_with_proxy = reader.load_trial_records()[0]
    resolved = reader.resolve_trial_artifacts(record_with_proxy)
    proxy = [a for a in resolved if a.kind == "proxy_jsonl"]
    assert proxy, "indexed trial artifact should resolve"
    assert proxy[0].path == proxy_file
    assert proxy[0].path.is_absolute() and proxy[0].path.exists()
    assert proxy[0].ref_path == "proxy-log.jsonl"

    # An artifact on the record + index but whose file no longer exists is
    # excluded — the reader never blesses a path that is not actually present.
    proxy_file.unlink()
    assert not any(
        a.kind == "proxy_jsonl"
        for a in reader.resolve_trial_artifacts(record_with_proxy)
    )


def test_experiment_artifact_reader_loads_run_and_trial_events(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id).append_event(
        phase="proxy_ready",
        event_type="proxy_ready",
        timestamp="2026-06-05T00:00:03Z",
        payload={"port": 41234},
        resource_refs=("res_000001",),
    )

    reader = ExperimentArtifactReader(run_dir)
    run_events = reader.load_events()
    trial_events = reader.load_trial_events(trial_id)

    assert [event.type for event in run_events] == ["proxy_ready"]
    assert [event.type for event in trial_events] == ["proxy_ready"]
    assert trial_events[0].trial_id == trial_id
    assert trial_events[0].phase == "proxy_ready"
    assert trial_events[0].payload == {"port": 41234}
    assert trial_events[0].resource_refs == ("res_000001",)


def test_resource_ledger_writer_appends_and_reads_latest_resource_state(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    writer = ResourceLedgerWriter(run_dir)

    created = writer.append_resource(
        run_id="run-1",
        resource_id="container:agent-1",
        kind="docker_container",
        provider="docker",
        external_id="agent-1",
        status="created",
        cleanup_action="docker rm -f agent-1",
        timestamp="2026-06-05T00:00:01Z",
        trial_id=plan.trials[0].trial_id,
        metadata={"image": "cage/claude-code:pentestenv"},
    )
    released = writer.append_resource(
        run_id="run-1",
        resource_id="container:agent-1",
        kind="docker_container",
        provider="docker",
        external_id="agent-1",
        status="released",
        cleanup_action="docker rm -f agent-1",
        timestamp="2026-06-05T00:00:05Z",
        trial_id=plan.trials[0].trial_id,
    )

    lines = (run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    saved = [json.loads(line) for line in lines]
    latest = ResourceLedgerReader(run_dir).latest_by_resource_id()

    assert created.record_id == "res_000001"
    assert released.record_id == "res_000002"
    assert [item["record_id"] for item in saved] == ["res_000001", "res_000002"]
    assert saved[0]["schema_version"] == "resource_record.v1"
    assert saved[0]["metadata"] == {"image": "cage/claude-code:pentestenv"}
    assert latest["container:agent-1"].status == "released"
    assert latest["container:agent-1"].cleanup_action == "docker rm -f agent-1"


def test_experiment_artifact_reader_loads_canonical_snapshot(tmp_path: Path) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    snapshot = ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    reader = ExperimentArtifactReader(run_dir)

    assert reader.load_plan() == plan
    assert reader.load_record() == snapshot.record
    assert reader.load_trial_records() == snapshot.trial_records


def test_trial_record_colocates_with_runtime_dir_not_subject_tree(
    tmp_path: Path,
) -> None:
    """P3: the durable record lives in ``trials/<runtime_id>/`` — one tree.

    The record ref must equal the runtime trial directory (no ``<subject>/``
    prefix, no ``replace(':','_')`` parallel tree), so ``record.json``
    co-locates with the runtime ``meta.json``/``proxy`` the trial runner writes.
    """

    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    snapshot = ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    for trial_ref in snapshot.record.trials.records:
        plan_trial = next(
            t for t in plan.trials if t.trial_id == trial_ref.trial_id
        )
        assert trial_ref.record_ref == f"trials/{plan_trial.runtime_id}/record.json"
        # The subject leg (agent:model:mode) must not leak into the disk path —
        # that prefix is what used to spawn the parallel subject-keyed tree.
        assert plan_trial.subject_id not in trial_ref.record_ref
        assert (run_dir / trial_ref.record_ref).is_file()


def test_reset_trial_planned_record_restores_archived_record(tmp_path: Path) -> None:
    """Resume re-run recreates the live record after the trial dir is archived.

    Co-location means archiving the prior attempt's trial directory carries its
    ``record.json`` away. ``reset_trial_planned_record`` must restore a live
    planned record at the run record's ref so ``experiment_record.json`` stays
    resolvable and the re-run's lifecycle marks have a record to update.
    """

    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec, plan=plan, run_id="run-1", created_at="2026-06-05T00:00:00Z"
    )
    trial_ref = snapshot.record.trials.records[0]
    live = run_dir / trial_ref.record_ref

    # Simulate the resume archive moving the whole trial dir aside.
    archived = live.parent.with_name(live.parent.name + ".before_resume_20260605")
    live.parent.rename(archived)
    assert not live.is_file()
    assert (archived / "record.json").is_file()  # prior attempt preserved

    restored = writer.reset_trial_planned_record(trial_ref.trial_id)
    assert restored is not None
    assert live.is_file()  # live ref resolves again
    assert restored.status == "planned"
    # The re-run's marks can now load and update the restored record.
    ExperimentArtifactReader(run_dir).load_trial_record(trial_ref.record_ref)


def test_experiment_artifact_reader_loads_full_run_snapshot(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id).append_event(
        phase="proxy_ready",
        event_type="proxy_ready",
        timestamp="2026-06-05T00:00:03Z",
        payload={"port": 41234},
    )
    ResourceLedgerWriter(run_dir).append_resource(
        run_id="run-1",
        resource_id="container:agent-1",
        kind="docker_container",
        provider="docker",
        external_id="agent-1",
        status="started",
        cleanup_action="docker rm -f agent-1",
        timestamp="2026-06-05T00:00:04Z",
        trial_id=trial_id,
    )

    loaded = ExperimentArtifactReader(run_dir).load_snapshot()

    assert loaded.run_dir == run_dir.resolve()
    assert loaded.spec.identity.experiment_id == spec.identity.experiment_id
    assert loaded.plan.plan_id == plan.plan_id
    assert loaded.record.run_id == "run-1"
    assert [trial.trial_id for trial in loaded.trial_records] == [trial_id]
    assert [event.type for event in loaded.events] == ["proxy_ready"]
    assert [event.type for event in loaded.trial_events[trial_id]] == ["proxy_ready"]
    assert [resource.resource_id for resource in loaded.resources] == ["container:agent-1"]


def test_experiment_artifact_reader_try_load_snapshot_returns_none_on_bad_runs(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    assert ExperimentArtifactReader(run_dir).try_load_snapshot() is not None

    (run_dir / "experiment_plan.json").write_text("{not-json", encoding="utf-8")

    assert ExperimentArtifactReader(run_dir).try_load_snapshot() is None
    assert ExperimentArtifactReader(tmp_path / "missing-run").try_load_snapshot() is None


def test_experiment_artifact_writer_updates_run_lifecycle_record(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    started = writer.mark_run_started(started_at="2026-06-05T00:00:03Z")
    finished = writer.mark_run_finished(
        status="interrupted",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="ctrl_c",
    )

    record = ExperimentArtifactReader(run_dir).load_record()
    assert started.status == "running"
    assert started.started_at == "2026-06-05T00:00:03Z"
    assert finished.status == "interrupted"
    assert record.status == "interrupted"
    assert record.status_reason == "ctrl_c"
    assert record.started_at == "2026-06-05T00:00:03Z"
    assert record.completed_at == "2026-06-05T00:01:00Z"
    assert record.interrupted_at == "2026-06-05T00:01:00Z"
    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    record_artifact = next(
        item
        for item in saved_index["artifacts"]
        if item["path"] == "experiment_record.json"
    )
    assert record_artifact["sha256"] == _sha256(run_dir / "experiment_record.json")


def test_experiment_artifact_writer_updates_trial_record_and_run_counts(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id

    started = writer.mark_trial_started(
        trial_id,
        started_at="2026-06-05T00:00:05Z",
        target_id="target-1",
    )
    finished = writer.mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:02:00Z",
        status_reason="container_error",
        termination=TrialTermination(reason="container_error", exit_code=137),
    )

    reader = ExperimentArtifactReader(run_dir)
    record = reader.load_record()
    trial = reader.load_trial_record(snapshot.record.trials.records[0].record_ref)
    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    trial_artifact = next(
        item
        for item in saved_index["artifacts"]
        if item["path"] == snapshot.record.trials.records[0].record_ref
    )

    assert started.status == "running"
    assert started.target_id == "target-1"
    assert finished.status == "failed"
    assert record.trials.total == len(plan.trials)
    assert record.trials.failed == 1
    assert record.trials.completed == 0
    assert record.trials.interrupted == 0
    assert trial.status == "failed"
    assert trial.status_reason == "container_error"
    assert trial.started_at == "2026-06-05T00:00:05Z"
    assert trial.completed_at == "2026-06-05T00:02:00Z"
    assert trial.termination.reason == "container_error"
    assert trial.termination.exit_code == 137
    assert trial_artifact["sha256"] == _sha256(run_dir / trial_artifact["path"])


def test_experiment_artifact_writer_updates_trial_scoring_record(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    score_ref = f"trials/{trial_id}/scores/demo.json"

    scored = writer.mark_trial_scored(
        trial_id,
        score_ref=score_ref,
        scoring_id="demo",
    )

    trial = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )
    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    trial_artifact = next(
        item
        for item in saved_index["artifacts"]
        if item["path"] == snapshot.record.trials.records[0].record_ref
    )

    assert scored.scoring.status == "scored"
    assert scored.scoring.score_ref == score_ref
    assert scored.scoring_id == "demo"
    assert trial.scoring.status == "scored"
    assert trial.scoring.score_ref == score_ref
    assert trial.scoring_id == "demo"
    assert trial_artifact["sha256"] == _sha256(run_dir / trial_artifact["path"])


def test_experiment_artifact_writer_updates_run_score_summary_record(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )

    scored = writer.mark_run_scored(summary_ref="summary.json")

    record = ExperimentArtifactReader(run_dir).load_record()
    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    record_artifact = next(
        item for item in saved_index["artifacts"]
        if item["path"] == "experiment_record.json"
    )

    assert scored.score_summary.status == "scored"
    assert scored.score_summary.summary_ref == "summary.json"
    assert record.score_summary.status == "scored"
    assert record.score_summary.summary_ref == "summary.json"
    assert record_artifact["sha256"] == _sha256(run_dir / "experiment_record.json")


def test_experiment_artifact_writer_indexes_run_level_artifacts(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    summary_path = run_dir / "scores" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text('{"scores": {}}\n', encoding="utf-8")

    artifact = writer.mark_run_artifact(
        artifact_id="run.score_summary",
        path=summary_path,
        kind="score_summary",
        schema_version="score_summary.v1",
        producer="cage score",
        replayability="replayable",
    )

    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    indexed = next(
        item for item in saved_index["artifacts"]
        if item["path"] == "scores/summary.json"
    )

    assert artifact.path == "scores/summary.json"
    assert indexed["artifact_id"] == "run.score_summary"
    assert indexed["kind"] == "score_summary"
    assert indexed["schema_version"] == "score_summary.v1"
    assert indexed["producer"] == "cage score"
    assert indexed["sha256"] == _sha256(summary_path)
    with pytest.raises(FileNotFoundError):
        writer.mark_run_artifact(
            artifact_id="missing",
            path="missing.json",
            kind="missing",
        )


def test_experiment_artifact_reader_loads_and_resolves_indexed_artifacts(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    summary_path = run_dir / "scores" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text('{"scores": {}}\n', encoding="utf-8")
    writer.mark_run_artifact(
        artifact_id="run.score_summary",
        path=summary_path,
        kind="score_summary",
        schema_version="score_summary.v1",
        producer="cage score",
        replayability="replayable",
    )

    reader = ExperimentArtifactReader(run_dir)
    index = reader.load_artifact_index()
    artifact = reader.find_artifact(path="scores/summary.json")

    assert {item.path for item in index.artifacts}.issuperset(
        {"experiment_record.json", "scores/summary.json"}
    )
    assert artifact is not None
    assert artifact.artifact_id == "run.score_summary"
    assert reader.resolve_artifact_path(artifact) == summary_path.resolve()
    assert reader.resolve_artifact_path("scores/summary.json") == summary_path.resolve()
    with pytest.raises(KeyError):
        reader.resolve_artifact_path("missing.json")
    with pytest.raises(ValueError):
        reader.resolve_artifact_path("../outside.json")


def test_experiment_artifact_reader_rejects_indexed_symlink_escape(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    outside = tmp_path / "outside-summary.json"
    outside.write_text('{"scores": {}}\n', encoding="utf-8")
    link = run_dir / "scores" / "summary.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    writer.mark_run_artifact(
        artifact_id="run.score_summary",
        path=link,
        kind="score_summary",
        schema_version="score_summary.v1",
        producer="cage score",
        replayability="replayable",
    )

    with pytest.raises(ValueError):
        ExperimentArtifactReader(run_dir).resolve_artifact_path("scores/summary.json")


def test_experiment_artifact_writer_records_trial_task_output_ref(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "trials" / "sample-a" / "pass_1" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "trial_index": 7,
                "sample": {"id": "sample-a"},
                "output": "finished",
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )

    updated = writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        producer="RunStorage",
        replayability="replayable",
    )

    trial = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )
    assert updated.artifacts == trial.artifacts
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in trial.artifacts}
    assert {
        f"trial.{trial_id}.events",
        f"trial.{trial_id}.resources",
        "trial.output",
    } == set(artifacts_by_id)
    assert artifacts_by_id["trial.output"].path == (
        "trials/sample-a/pass_1/task_output.json"
    )
    assert artifacts_by_id["trial.output"].kind == "task_output"
    assert artifacts_by_id["trial.output"].sha256 == _sha256(output_path)

    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    output_artifact = next(
        item
        for item in saved_index["artifacts"]
        if item["path"] == "trials/sample-a/pass_1/task_output.json"
    )
    assert output_artifact["kind"] == "task_output"
    assert output_artifact["sha256"] == _sha256(output_path)


def test_experiment_artifact_writer_records_trial_directory_artifact(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    state_dir = run_dir / "trials" / "sample-a" / "pass_1" / "state_pre"
    state_dir.mkdir(parents=True)
    (state_dir / "home" / "agent").mkdir(parents=True)
    (state_dir / "home" / "agent" / ".bashrc").write_text(
        "export CAGE=1\n",
        encoding="utf-8",
    )

    updated = writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.state_pre",
        path=state_dir,
        kind="state_snapshot_pre",
        schema_version="state_snapshot.directory.v1",
        producer="snapshot_state",
        replayability="audit",
        content_type="inode/directory",
    )

    trial = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )
    assert updated.artifacts == trial.artifacts
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in trial.artifacts}
    assert {
        f"trial.{trial_id}.events",
        f"trial.{trial_id}.resources",
        "trial.state_pre",
    } == set(artifacts_by_id)
    assert artifacts_by_id["trial.state_pre"].path == (
        "trials/sample-a/pass_1/state_pre"
    )
    assert artifacts_by_id["trial.state_pre"].kind == "state_snapshot_pre"
    assert artifacts_by_id["trial.state_pre"].content_type == "inode/directory"
    assert artifacts_by_id["trial.state_pre"].sha256

    saved_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    state_artifact = next(
        item
        for item in saved_index["artifacts"]
        if item["path"] == "trials/sample-a/pass_1/state_pre"
    )
    assert state_artifact["kind"] == "state_snapshot_pre"
    assert state_artifact["sha256"] == artifacts_by_id["trial.state_pre"].sha256
    assert ExperimentArtifactReader(run_dir).resolve_artifact_path(
        "trials/sample-a/pass_1/state_pre"
    ) == state_dir.resolve()


def test_scoring_context_loads_from_canonical_trial_record_ref(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    trial_dir = run_dir / "trials" / "sample-a" / "pass_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "task_output.json").write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "trial_index": 2,
                "sample": {"id": "sample-a"},
                "output": "final answer",
                "exit_code": 3,
            }
        ),
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=trial_dir / "task_output.json",
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    ctx = ScoringContext.from_trial_record(run_dir, trial_record)

    assert ctx is not None
    assert ctx.trial_id == trial_id
    assert ctx.trial_index == 2
    assert ctx.sample == {"id": "sample-a"}
    assert ctx.output == "final answer"
    assert ctx.exit_code == 3
    assert ctx.trial_dir == trial_dir


def test_scoring_context_loads_prompt_and_proxy_from_canonical_artifact_refs(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "canonical" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": trial_id, "output": "final answer"}),
        encoding="utf-8",
    )
    prompt_path = run_dir / "canonical_prompt" / "prompt.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("rendered prompt", encoding="utf-8")
    proxy_path = run_dir / "canonical_proxy" / "proxy.jsonl"
    proxy_path.parent.mkdir(parents=True)
    proxy_path.write_text(
        json.dumps({"request_id": "req-1", "status": "success"}) + "\n",
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.prompt",
        path=prompt_path,
        kind="prompt",
        schema_version="prompt.txt.v1",
        replayability="replayable",
        content_type="text/plain",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.proxy",
        path=proxy_path,
        kind="proxy_log",
        schema_version="proxy_log.jsonl.v1",
        replayability="audit",
        content_type="application/x-ndjson",
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    ctx = ScoringContext.from_trial_record(run_dir, trial_record)

    assert ctx is not None
    assert ctx.prompt == "rendered prompt"
    assert ctx.proxy_log == [{"request_id": "req-1", "status": "success"}]


def test_scoring_context_from_trial_record_uses_record_trial_id_over_adjacent_meta(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "canonical" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": "stale-output-id", "output": "final answer"}),
        encoding="utf-8",
    )
    (output_path.parent / "meta.json").write_text(
        json.dumps({"trial_id": "stale-meta-id"}),
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    ctx = ScoringContext.from_trial_record(run_dir, trial_record)

    assert ctx is not None
    assert ctx.trial_id == trial_id
    assert ctx.canonical_trial_id == trial_id
    assert ctx.metadata["trial_id"] == trial_id


def test_scoring_context_metadata_projects_canonical_trial_record(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "canonical" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": trial_id, "output": "final answer"}),
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    writer.mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="model_timeout",
        termination=TrialTermination(reason="model_timeout", exit_code=124),
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    ctx = ScoringContext.from_trial_record(run_dir, trial_record)

    assert ctx is not None
    assert ctx.metadata["trial_id"] == trial_id
    assert ctx.metadata["status"] == "failed"
    assert ctx.metadata["termination_reason"] == "model_timeout"
    assert ctx.metadata["exit_code"] == 124


def test_scoring_context_reads_check_done_output_from_final_evidence_ref(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "canonical" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": trial_id, "output": "final answer"}),
        encoding="utf-8",
    )
    evidence_path = run_dir / "canonical_evidence" / "check_done_output.txt"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("target reported solved", encoding="utf-8")
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.final_evidence",
        path=evidence_path,
        kind="final_evidence",
        schema_version="check_done_output.txt.v1",
        replayability="audit",
        content_type="text/plain",
    )
    writer.mark_trial_scored(
        trial_id,
        final_evidence_ref="canonical_evidence/check_done_output.txt",
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    ctx = ScoringContext.from_trial_record(run_dir, trial_record)

    assert ctx is not None
    assert ctx.check_done_output == "target reported solved"


def test_scoring_context_reads_live_success_from_live_evidence_ref(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "canonical" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": trial_id, "output": "final answer"}),
        encoding="utf-8",
    )
    verdict_path = run_dir / "canonical_evidence" / "live_success.json"
    verdict_path.parent.mkdir(parents=True)
    verdict = {
        "success": True,
        "trial_id": trial_id,
        "source": "check_done",
    }
    verdict_path.write_text(json.dumps(verdict), encoding="utf-8")
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.live_evidence",
        path=verdict_path,
        kind="live_evidence",
        schema_version="live_success.json.v1",
        replayability="audit",
        content_type="application/json",
    )
    writer.mark_trial_scored(
        trial_id,
        live_evidence_ref="canonical_evidence/live_success.json",
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    ctx = ScoringContext.from_trial_record(run_dir, trial_record)

    assert ctx is not None
    assert ctx.live_success == verdict


def test_scoring_context_requires_indexed_task_output_artifact(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "trials" / "sample-a" / "pass_1" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "sample": {"id": "sample-a"},
                "output": "final answer",
                "exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    index_path = run_dir / "artifact_index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    index_data["artifacts"] = [
        artifact
        for artifact in index_data["artifacts"]
        if artifact.get("path") != "trials/sample-a/pass_1/task_output.json"
    ]
    index_path.write_text(json.dumps(index_data), encoding="utf-8")
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    assert ScoringContext.from_trial_record(run_dir, trial_record) is None


def test_scoring_context_requires_matching_indexed_artifact_id(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "canonical" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": trial_id, "output": "final answer"}),
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    index_path = run_dir / "artifact_index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    for artifact in index_data["artifacts"]:
        if artifact.get("path") == "canonical/task_output.json":
            artifact["artifact_id"] = "trial.other-output"
    index_path.write_text(json.dumps(index_data), encoding="utf-8")
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )

    assert ScoringContext.from_trial_record(run_dir, trial_record) is None


def test_scoring_context_returns_none_when_artifact_index_is_missing(
    tmp_path: Path,
) -> None:
    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    output_path = run_dir / "trials" / "sample-a" / "pass_1" / "task_output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"trial_id": trial_id, "output": "final answer"}),
        encoding="utf-8",
    )
    writer.mark_trial_artifact(
        trial_id,
        artifact_id="trial.output",
        path=output_path,
        kind="task_output",
        schema_version="task_output.v1",
        replayability="replayable",
    )
    trial_record = ExperimentArtifactReader(run_dir).load_trial_record(
        snapshot.record.trials.records[0].record_ref
    )
    (run_dir / "artifact_index.json").unlink()

    assert ScoringContext.from_trial_record(run_dir, trial_record) is None


def test_finalize_running_trials_marks_killed_trials_interrupted(tmp_path: Path) -> None:
    import json

    spec, plan = _build_plan(tmp_path / "project")
    run_dir = tmp_path / ".cage_runs" / "claude_code:deepseek-v4-pro:stateless" / "run-1"
    writer = ExperimentArtifactWriter(run_dir)
    snapshot = writer.write_initial_snapshot(
        spec=spec, plan=plan, run_id="run-1", created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    record_path = snapshot.trial_record_paths[trial_id]

    # A trial killed mid-flight is left at "running" — mark_trial_finished never ran.
    writer.mark_trial_started(trial_id, started_at="2026-06-05T00:00:10Z")
    assert json.loads(record_path.read_text())["status"] == "running"

    finalized = writer.finalize_running_trials_as_interrupted(
        completed_at="2026-06-05T00:05:00Z",
    )

    assert finalized == [trial_id]
    record = json.loads(record_path.read_text())
    assert record["status"] == "interrupted"
    assert record["status_reason"] == "user_interrupted"
    assert record["completed_at"] == "2026-06-05T00:05:00Z"
    # Idempotent: a second sweep finds nothing still running.
    assert writer.finalize_running_trials_as_interrupted(
        completed_at="2026-06-05T00:06:00Z",
    ) == []


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
