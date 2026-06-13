import json
from pathlib import Path

from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.experiment.model import (
    TrialTermination,
    build_experiment_plan,
    load_experiment_spec,
)
from cage.artifacts.trial_session import TrialRuntimeSession


def _write_project(project_dir: Path) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "benchmark.py").write_text(
        'raise RuntimeError("trial runtime test must not import benchmark")\n',
        encoding="utf-8",
    )
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: runtime-session-demo
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBenchmark
runtime:
  timeout: 60
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


def _write_snapshot(tmp_path: Path):
    project_file = _write_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    run_dir = tmp_path / ".cage_runs" / "agent" / "run-1"
    ExperimentArtifactWriter(run_dir).write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    return run_dir, plan.trials[0].trial_id


def _trial_event_log_path(run_dir: Path, trial_id: str) -> Path:
    record = ExperimentArtifactReader(run_dir).load_record()
    for trial_ref in record.trials.records:
        if trial_ref.trial_id == trial_id:
            return run_dir / Path(trial_ref.record_ref).parent / "events.jsonl"
    raise AssertionError(f"unknown trial_id in test snapshot: {trial_id}")


def _trial_resource_path(run_dir: Path, trial_id: str) -> Path:
    record = ExperimentArtifactReader(run_dir).load_record()
    for trial_ref in record.trials.records:
        if trial_ref.trial_id == trial_id:
            return run_dir / Path(trial_ref.record_ref).parent / "resources.json"
    raise AssertionError(f"unknown trial_id in test snapshot: {trial_id}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_trial_runtime_session_marks_started_and_finished(tmp_path: Path) -> None:
    run_dir, trial_id = _write_snapshot(tmp_path)
    session = TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id)

    started = session.mark_started(started_at="2026-06-05T00:00:01Z")
    finished = session.mark_finished(
        status="completed",
        completed_at="2026-06-05T00:00:05Z",
        status_reason="completed",
        termination=TrialTermination(reason="completed", exit_code=0),
        payload={"termination_reason": "completed", "exit_code": 0},
    )

    reader = ExperimentArtifactReader(run_dir)
    trial_record = reader.load_trial_records()[0]
    events = _read_jsonl(run_dir / "events.jsonl")
    trial_events = _read_jsonl(_trial_event_log_path(run_dir, trial_id))

    assert started.status == "running"
    assert finished.status == "completed"
    assert trial_record.status == "completed"
    assert trial_record.termination.reason == "completed"
    assert [event["type"] for event in events] == ["trial_started", "trial_finished"]
    assert [event["type"] for event in trial_events] == ["trial_started", "trial_finished"]
    assert events[0]["trial_id"] == trial_id
    assert events[0]["phase"] == "running"
    assert events[1]["phase"] == "completed"
    assert events[1]["payload"] == {"exit_code": 0, "termination_reason": "completed"}
    assert trial_events[1]["payload"] == {
        "exit_code": 0,
        "termination_reason": "completed",
    }


def test_trial_runtime_session_appends_non_terminal_lifecycle_event(
    tmp_path: Path,
) -> None:
    run_dir, trial_id = _write_snapshot(tmp_path)
    session = TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id)

    event = session.append_event(
        phase="proxy_ready",
        event_type="proxy_ready",
        timestamp="2026-06-05T00:00:03Z",
        payload={"port": 41234},
        resource_refs=("res_000001",),
    )

    run_events = _read_jsonl(run_dir / "events.jsonl")
    trial_events = _read_jsonl(_trial_event_log_path(run_dir, trial_id))
    trial_record = ExperimentArtifactReader(run_dir).load_trial_records()[0]

    assert event.trial_id == trial_id
    assert event.subject_id == trial_record.subject_id
    assert event.task_id == trial_record.task_id
    assert run_events[0]["type"] == "proxy_ready"
    assert trial_events[0]["type"] == "proxy_ready"
    assert trial_events[0]["payload"] == {"port": 41234}
    assert trial_events[0]["resource_refs"] == ["res_000001"]


def test_trial_runtime_session_records_resource_lifecycle(tmp_path: Path) -> None:
    run_dir, trial_id = _write_snapshot(tmp_path)
    session = TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id)

    record = session.record_resource(
        run_id="run-1",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:02Z",
        metadata={"image": "cage/claude-code:pentestenv"},
    )

    saved = _read_jsonl(run_dir / "resources.jsonl")

    assert record.trial_id == trial_id
    assert saved == [
        {
            "cleanup_action": "docker rm -f agent-a",
            "cleanup_error": None,
            "external_id": "agent-a",
            "kind": "docker_container",
            "metadata": {"image": "cage/claude-code:pentestenv"},
            "provider": "docker",
            "record_id": "res_000001",
            "resource_id": "docker_container:agent-a",
            "run_id": "run-1",
            "schema_version": "resource_record.v1",
            "status": "started",
            "timestamp": "2026-06-05T00:00:02Z",
            "trial_id": trial_id,
        }
    ]


def test_trial_runtime_session_records_resource_lifecycle_events(
    tmp_path: Path,
) -> None:
    run_dir, trial_id = _write_snapshot(tmp_path)
    session = TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id)

    record = session.record_resource(
        run_id="run-1",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:02Z",
    )

    run_events = _read_jsonl(run_dir / "events.jsonl")
    trial_events = _read_jsonl(_trial_event_log_path(run_dir, trial_id))

    assert record.record_id == "res_000001"
    assert [event["type"] for event in run_events] == ["resource_started"]
    assert [event["type"] for event in trial_events] == ["resource_started"]
    assert run_events[0]["phase"] == "resource:started"
    assert run_events[0]["resource_refs"] == ["res_000001"]
    assert run_events[0]["payload"] == {
        "cleanup_action": "docker rm -f agent-a",
        "external_id": "agent-a",
        "kind": "docker_container",
        "provider": "docker",
        "resource_id": "docker_container:agent-a",
        "status": "started",
    }


def test_trial_runtime_session_updates_trial_local_resource_projection(
    tmp_path: Path,
) -> None:
    run_dir, trial_id = _write_snapshot(tmp_path)
    session = TrialRuntimeSession(run_dir=run_dir, trial_id=trial_id)

    session.record_resource(
        run_id="run-1",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:02Z",
        metadata={"image": "cage/claude-code:pentestenv"},
    )
    session.record_resource(
        run_id="run-1",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="released",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-06-05T00:00:06Z",
    )

    saved = json.loads(_trial_resource_path(run_dir, trial_id).read_text(encoding="utf-8"))

    assert saved["schema_version"] == "trial_resources.v1"
    assert saved["run_id"] == "run-1"
    assert saved["trial_id"] == trial_id
    assert [record["record_id"] for record in saved["resources"]] == [
        "res_000001",
        "res_000002",
    ]
    assert [record["status"] for record in saved["resources"]] == [
        "started",
        "released",
    ]
    assert saved["resources"][0]["metadata"] == {
        "image": "cage/claude-code:pentestenv"
    }
