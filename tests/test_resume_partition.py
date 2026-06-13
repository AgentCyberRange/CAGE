"""Tests for resume policy in _partition_resumed_trials.

Resume's policy:
  * Default re-runs ONLY trials whose previous attempt couldn't yield a
    valid result (target_unavailable, trial_error, cancelled_before_start,
    missing/incomplete meta).
  * Completed trials always replay from disk.
  * Other failure modes (agent_exit_nonzero, tool_limit, execution_timeout,
    model_timeout, user_interrupted) replay from disk by default and only
    re-run when explicitly opted in via project.yml::resume.retry_reasons.
"""

import json
from pathlib import Path

from cage.artifacts.writer import ExperimentArtifactWriter
from cage.artifacts.run_storage import RunStorage
from cage.experiment.model import (
    TrialTermination,
    build_experiment_plan,
    load_experiment_spec,
)
from cage.experiment.model import Trial, TrialType
from cage.experiment.engine.resume import _archive_trial_dir_before_resume, _partition_resumed_trials


def _make_trial(idx: int, trial_id: str) -> Trial:
    return Trial(
        id=trial_id,
        index=idx,
        type=TrialType.TASK,
        sample={"id": f"sample-{idx}"},
    )


def _write_meta(storage: RunStorage, trial_id: str, payload: dict) -> None:
    storage.trial_dir(trial_id)  # ensures directory exists
    (storage.run_dir / "trials" / trial_id / "meta.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_contract_project(project_dir: Path) -> Path:
    """Create a minimal project file for canonical resume-record tests."""

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "benchmark.py").write_text(
        'raise RuntimeError("resume test imported benchmark code")\n',
        encoding="utf-8",
    )
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: resume-contract-smoke
eval:
  benchmark:
    module: ./benchmark.py
    class: DemoBenchmark
runtime:
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


def _write_canonical_resume_snapshot(
    tmp_path: Path,
    *,
    status: str = "failed",
    reason: str = "model_error",
    exit_code: int = 1,
) -> tuple[RunStorage, ExperimentArtifactWriter, str, Trial]:
    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    storage = RunStorage(run_dir=tmp_path / ".cage_runs" / "agent" / "run-1")
    writer = ExperimentArtifactWriter(storage.run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    writer.mark_trial_finished(
        trial_id,
        status=status,
        completed_at="2026-06-05T00:01:00Z",
        status_reason=reason,
        termination=TrialTermination(reason=reason, exit_code=exit_code),
    )
    trial = Trial(
        id=trial_id,
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-a", "content": "task"},
    )
    return storage, writer, trial_id, trial


def test_default_reruns_target_unavailable(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "target_unavailable",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_default_reruns_trial_error(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "trial_error",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_default_reruns_cancelled_before_start(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "cancelled",
        "termination_reason": "cancelled_before_start",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_default_reruns_coarse_interrupted_reason(tmp_path: Path) -> None:
    # Some records store an interrupted trial as status=failed,
    # reason="interrupted" (no meta) rather than the canonical
    # "user_interrupted". It still means "didn't finish" ⇒ re-run.
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "interrupted",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_default_reruns_stale_running_without_reason(tmp_path: Path) -> None:
    # A "running" record during resume is stale (the run is not active). With no
    # recorded reason, the trial was interrupted mid-flight ⇒ re-run.
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {"status": "running"})

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_missing_meta_reruns(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    storage.trial_dir("t0")  # dir exists but no meta.json

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_blank_status_reruns(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {"termination_reason": "agent_exit_nonzero"})

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_default_replays_agent_exit_nonzero(tmp_path: Path) -> None:
    """agent_exit_nonzero is a valid result by default — DO NOT re-run."""
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "agent_exit_nonzero",
        "exit_code": 2,
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_default_replays_tool_limit(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "interrupted",
        "termination_reason": "tool_limit",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_default_replays_execution_timeout(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "execution_timeout",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_default_replays_completed(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "completed",
        "termination_reason": "completed",
    })

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_extra_retry_reasons_opt_in_tool_limit(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "interrupted",
        "termination_reason": "tool_limit",
    })

    replayed, pending = _partition_resumed_trials(
        storage, [trial], extra_retry_reasons=["tool_limit"]
    )

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_extra_retry_reasons_normalized(tmp_path: Path) -> None:
    """Reason strings are normalized to lower-case + stripped."""
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "model_timeout",
    })

    replayed, pending = _partition_resumed_trials(
        storage, [trial], extra_retry_reasons=["  Model_Timeout  "]
    )

    assert [t.id for t in pending] == ["t0"]
    assert replayed == []


def test_replay_preserves_failed_status(tmp_path: Path) -> None:
    """Previously _replay_trial_result_from_disk hardcoded trial.status=COMPLETED.
    Now it must reflect the on-disk status so failed trials don't masquerade
    as successes in the merged result list.
    """
    from cage.experiment.model import TrialStatus

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "agent_exit_nonzero",
        "exit_code": 1,
    })
    # output file is optional but the replay helper reads it
    (tmp_path / "trials" / "t0" / "task_output.json").write_text(
        json.dumps({"output": "boom"}), encoding="utf-8"
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert len(replayed) == 1
    assert trial.status is TrialStatus.FAILED
    assert replayed[0].metadata["status"] == "failed"
    assert replayed[0].metadata["termination_reason"] == "agent_exit_nonzero"


def test_replay_preserves_completed_status(tmp_path: Path) -> None:
    from cage.experiment.model import TrialStatus

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "completed",
        "termination_reason": "completed",
        "exit_code": 0,
    })
    (tmp_path / "trials" / "t0" / "task_output.json").write_text(
        json.dumps({"output": "done"}), encoding="utf-8"
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert trial.status is TrialStatus.COMPLETED


def test_resume_replays_canonical_trial_record_when_meta_json_missing(
    tmp_path: Path,
) -> None:
    """Canonical TrialRecord should drive resume when legacy meta is absent."""

    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    storage = RunStorage(run_dir=tmp_path / ".cage_runs" / "agent" / "run-1")
    writer = ExperimentArtifactWriter(storage.run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
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
    trial = Trial(
        id=trial_id,
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-a", "content": "task"},
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert [result.trial_id for result in replayed] == [trial_id]
    assert replayed[0].metadata["termination_reason"] == "agent_exit_nonzero"


def test_resume_prefers_canonical_trial_record_over_stale_meta_json(
    tmp_path: Path,
) -> None:
    storage, _writer, trial_id, trial = _write_canonical_resume_snapshot(
        tmp_path,
        reason="model_error",
        exit_code=1,
    )
    _write_meta(
        storage,
        trial_id,
        {
            "status": "completed",
            "termination_reason": "completed",
            "exit_code": 0,
        },
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert replayed == []
    assert [item.id for item in pending] == [trial_id]


def test_resume_reruns_never_started_planned_trial(tmp_path: Path) -> None:
    """A trial left ``planned`` (never executed) must be (re)run, not replayed.

    Regression: the initial snapshot writes a canonical TrialRecord with
    ``status="planned"`` for every trial. A run that is interrupted (or capped
    by ``max_trial``) before reaching a trial leaves that record at "planned"
    with no meta.json and no result on disk. Resume previously classified it as
    "(no reason) (not a retry reason)" → replay, silently dropping every
    still-pending trial instead of continuing them.
    """
    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    storage = RunStorage(run_dir=tmp_path / ".cage_runs" / "agent" / "run-1")
    writer = ExperimentArtifactWriter(storage.run_dir)
    # Snapshot only — never mark the trial finished, so its record stays
    # "planned" exactly as it would for a never-reached pending trial.
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    trial = Trial(
        id=trial_id,
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-a", "content": "task"},
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert replayed == []
    assert [item.id for item in pending] == [trial_id]


def test_resume_replay_reads_canonical_task_output_when_legacy_file_missing(
    tmp_path: Path,
) -> None:
    """Replay should reconstruct TrialResult output through ArtifactIndex."""

    project_file = _write_contract_project(tmp_path / "project")
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    storage = RunStorage(run_dir=tmp_path / ".cage_runs" / "agent" / "run-1")
    writer = ExperimentArtifactWriter(storage.run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id="run-1",
        created_at="2026-06-05T00:00:00Z",
    )
    trial_id = plan.trials[0].trial_id
    task_output_ref = "canonical_outputs/task_output.json"
    task_output_path = storage.run_dir / task_output_ref
    task_output_path.parent.mkdir(parents=True)
    task_output_path.write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "trial_index": 0,
                "output": "canonical replay output",
                "exit_code": 2,
                "sample": {"id": "sample-a"},
            }
        ),
        encoding="utf-8",
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
    writer.mark_trial_finished(
        trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="agent_exit_nonzero",
        termination=TrialTermination(reason="agent_exit_nonzero", exit_code=2),
    )
    trial = Trial(
        id=trial_id,
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-a", "content": "task"},
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert len(replayed) == 1
    assert replayed[0].trial_id == trial_id
    assert replayed[0].output == "canonical replay output"
    assert replayed[0].exit_code == 2
    assert replayed[0].metadata["termination_reason"] == "agent_exit_nonzero"


def test_resume_replay_prefers_canonical_task_output_over_stale_legacy_file(
    tmp_path: Path,
) -> None:
    storage, writer, trial_id, trial = _write_canonical_resume_snapshot(
        tmp_path,
        reason="agent_exit_nonzero",
        exit_code=2,
    )
    task_output_ref = "canonical_outputs/task_output.json"
    task_output_path = storage.run_dir / task_output_ref
    task_output_path.parent.mkdir(parents=True)
    task_output_path.write_text(
        json.dumps(
            {
                "trial_id": trial_id,
                "trial_index": 0,
                "output": "canonical replay output",
                "exit_code": 2,
                "sample": {"id": "sample-a"},
            }
        ),
        encoding="utf-8",
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
    legacy_output = storage.run_dir / "trials" / trial_id / "task_output.json"
    legacy_output.parent.mkdir(parents=True, exist_ok=True)
    legacy_output.write_text(
        json.dumps({"output": "stale legacy output"}),
        encoding="utf-8",
    )

    replayed, pending = _partition_resumed_trials(storage, [trial])

    assert pending == []
    assert len(replayed) == 1
    assert replayed[0].output == "canonical replay output"


def test_load_experiment_parses_resume_retry_reasons(tmp_path: Path) -> None:
    """resume.retry_reasons in project.yml must land on ExperimentConfig."""
    # Minimal benchmark module
    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n"
        "  retry_reasons:\n"
        "    - Model_Timeout\n"
        "    - tool_limit\n"
        "    - '  '\n",
        encoding="utf-8",
    )

    from cage.config.experiment import resolve

    cfg = resolve(project_file)

    # normalized + blank stripped
    assert cfg.resume_retry_reasons == ["model_timeout", "tool_limit"]


def test_load_experiment_rejects_non_list_retry_reasons(tmp_path: Path) -> None:
    import pytest

    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n  retry_reasons: not-a-list\n",
        encoding="utf-8",
    )

    from cage.config.experiment import resolve

    with pytest.raises(ValueError, match="resume.retry_reasons must be a list"):
        resolve(project_file)


def test_archive_renames_existing_trial_dir(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "target_unavailable",
    })
    # Simulate the append-mode artifacts that motivate archiving
    (tmp_path / "trials" / "t0" / "proxy").mkdir(parents=True, exist_ok=True)
    (tmp_path / "trials" / "t0" / "proxy" / "proxy.jsonl").write_text(
        '{"old": "request"}\n', encoding="utf-8"
    )

    archived = _archive_trial_dir_before_resume(storage, trial)

    assert archived is not None
    assert archived.exists()
    assert archived.name.startswith("t0.before_resume_")
    # Original dir is gone (will be lazily recreated when re-run writes)
    assert not (tmp_path / "trials" / "t0").exists()
    # Old artifact preserved inside the archive
    assert (archived / "meta.json").exists()
    assert (archived / "proxy" / "proxy.jsonl").read_text() == '{"old": "request"}\n'


def test_archive_skips_missing_or_empty_dir(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")

    # Missing dir
    assert _archive_trial_dir_before_resume(storage, trial) is None

    # Empty dir
    (tmp_path / "trials" / "t0").mkdir(parents=True)
    assert _archive_trial_dir_before_resume(storage, trial) is None
    assert (tmp_path / "trials" / "t0").exists()


def test_archive_disambiguates_same_second(tmp_path: Path) -> None:
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {"status": "failed", "termination_reason": "trial_error"})
    first = _archive_trial_dir_before_resume(storage, trial)

    # Second resume cycle on the same trial (same wall-clock second possible)
    _write_meta(storage, "t0", {"status": "failed", "termination_reason": "trial_error"})
    second = _archive_trial_dir_before_resume(storage, trial)

    assert first is not None and second is not None
    assert first != second
    assert first.exists() and second.exists()


def test_archive_handles_nested_passk_id(tmp_path: Path) -> None:
    """Trial id like 'sample/pass_2' should archive only that pass dir."""
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "sample/pass_2")
    # Sibling pass dirs must survive archiving
    _write_meta(storage, "sample/pass_1", {"status": "completed", "termination_reason": "completed"})
    _write_meta(storage, "sample/pass_2", {"status": "failed", "termination_reason": "target_unavailable"})
    _write_meta(storage, "sample/pass_3", {"status": "completed", "termination_reason": "completed"})

    archive = _archive_trial_dir_before_resume(storage, trial)

    assert archive is not None
    assert archive.parent == tmp_path / "trials" / "sample"
    assert archive.name.startswith("pass_2.before_resume_")
    assert (tmp_path / "trials" / "sample" / "pass_1" / "meta.json").exists()
    assert (tmp_path / "trials" / "sample" / "pass_3" / "meta.json").exists()
    assert not (tmp_path / "trials" / "sample" / "pass_2").exists()


def test_find_trial_dirs_skips_archives(tmp_path: Path) -> None:
    """Web index must not list ``<id>.before_resume_<ts>`` as a separate trial."""
    from cage.web.data import find_trial_dirs

    run_dir = tmp_path
    trials = run_dir / "trials"
    # One live trial dir
    (trials / "t0").mkdir(parents=True)
    (trials / "t0" / "meta.json").write_text("{}", encoding="utf-8")
    # One archive sibling carrying the same artifact shape
    archive = trials / "t0.before_resume_20260521T143022"
    archive.mkdir(parents=True)
    (archive / "meta.json").write_text("{}", encoding="utf-8")
    (archive / "task_output.json").write_text("{}", encoding="utf-8")

    found = find_trial_dirs(run_dir)
    names = sorted(p.name for p in found)
    assert names == ["t0"]


def test_load_resume_attempts_lists_live_and_archives(tmp_path: Path) -> None:
    from cage.web.data import load_resume_attempts

    run_dir = tmp_path
    trials = run_dir / "trials"
    live = trials / "t0"
    live.mkdir(parents=True)
    (live / "meta.json").write_text(
        json.dumps({"status": "completed", "termination_reason": "completed"}),
        encoding="utf-8",
    )
    older = trials / "t0.before_resume_20260520T120000"
    older.mkdir()
    (older / "meta.json").write_text(
        json.dumps({"status": "failed", "termination_reason": "target_unavailable"}),
        encoding="utf-8",
    )
    newer = trials / "t0.before_resume_20260521T143022"
    newer.mkdir()
    (newer / "meta.json").write_text(
        json.dumps({"status": "failed", "termination_reason": "trial_error"}),
        encoding="utf-8",
    )

    # Viewing live trial → live is current, archives listed newest-first.
    attempts = load_resume_attempts(live)
    assert len(attempts) == 3
    assert attempts[0].is_live is True
    assert attempts[0].is_current is True
    assert attempts[0].label == "Current"
    # Newest archive first
    assert attempts[1].trial_dir == newer
    assert attempts[1].is_current is False
    assert attempts[1].label == "2026-05-21 14:30:22"
    assert attempts[1].termination_reason == "trial_error"
    assert attempts[2].trial_dir == older
    assert attempts[2].label == "2026-05-20 12:00:00"

    # Viewing an archive → that archive is marked current; live is still listed.
    attempts_from_archive = load_resume_attempts(newer)
    assert attempts_from_archive[0].is_live is True
    assert attempts_from_archive[0].is_current is False
    archive_entries = [a for a in attempts_from_archive if not a.is_live]
    current_archive = [a for a in archive_entries if a.is_current]
    assert len(current_archive) == 1
    assert current_archive[0].trial_dir == newer


def test_load_resume_attempts_empty_when_no_archives(tmp_path: Path) -> None:
    from cage.web.data import load_resume_attempts

    live = tmp_path / "trials" / "t0"
    live.mkdir(parents=True)
    (live / "meta.json").write_text("{}", encoding="utf-8")

    assert load_resume_attempts(live) == []


def test_find_trial_dirs_surfaces_archive_only_orphans(tmp_path: Path) -> None:
    """An interrupted resume can leave a parent with only archive
    children (live dir was renamed but the new one was never created).
    The dashboard must still show that trial as a row — use the newest
    archive as the row representative."""
    from cage.web.data import find_trial_dirs

    run_dir = tmp_path
    trials = run_dir / "trials"
    # Normal trial with live + archive — must still surface live only.
    (trials / "t_live").mkdir(parents=True)
    (trials / "t_live" / "meta.json").write_text("{}", encoding="utf-8")
    (trials / "t_live.before_resume_20260520T120000").mkdir(parents=True)
    (trials / "t_live.before_resume_20260520T120000" / "meta.json").write_text(
        "{}", encoding="utf-8"
    )
    # Orphan trial: only archives, no live.
    older = trials / "t_orphan.before_resume_20260520T120000"
    older.mkdir(parents=True)
    (older / "meta.json").write_text("{}", encoding="utf-8")
    newer = trials / "t_orphan.before_resume_20260521T143022"
    newer.mkdir(parents=True)
    (newer / "meta.json").write_text("{}", encoding="utf-8")

    found = find_trial_dirs(run_dir)
    names = sorted(p.name for p in found)
    # t_live appears once (live), t_orphan appears once (newest archive).
    assert names == ["t_live", "t_orphan.before_resume_20260521T143022"]


def test_find_trial_dirs_orphan_in_nested_layout(tmp_path: Path) -> None:
    """Orphan promotion must work in the nested ``<challenge>/<pass>``
    layout used by agent_pentest_bench."""
    from cage.web.data import find_trial_dirs

    run_dir = tmp_path
    trials = run_dir / "trials"
    challenge = trials / "chal-1"
    challenge.mkdir(parents=True)
    # pass_1: live exists.
    (challenge / "pass_1").mkdir()
    (challenge / "pass_1" / "meta.json").write_text("{}", encoding="utf-8")
    # pass_2: only archives — orphan.
    arc = challenge / "pass_2.before_resume_20260521T155122"
    arc.mkdir()
    (arc / "meta.json").write_text("{}", encoding="utf-8")

    found = find_trial_dirs(run_dir)
    names = sorted(str(p.relative_to(trials)) for p in found)
    assert names == [
        "chal-1/pass_1",
        "chal-1/pass_2.before_resume_20260521T155122",
    ]


def test_load_resume_attempts_skips_current_for_orphan(tmp_path: Path) -> None:
    """When a trial has only archives (no live dir), the attempts bar
    must not render a fake ``Current`` entry — just the archives."""
    from cage.web.data import load_resume_attempts

    run_dir = tmp_path
    trials = run_dir / "trials"
    archive = trials / "t0.before_resume_20260521T143022"
    archive.mkdir(parents=True)
    (archive / "meta.json").write_text(
        json.dumps({"status": "failed", "termination_reason": "model_quota_exhausted"}),
        encoding="utf-8",
    )

    attempts = load_resume_attempts(archive)
    assert len(attempts) == 1
    assert attempts[0].is_live is False
    assert attempts[0].is_current is True
    assert attempts[0].label == "2026-05-21 14:30:22"


def test_max_attempts_caps_retry(tmp_path: Path) -> None:
    """Once a trial has been re-run max_attempts times, resume stops
    re-running it even if the reason is in the retry set. The most recent
    failed meta replays from disk so the trial still shows in summary."""
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "model_quota_exhausted",
    })
    # Simulate 2 prior archives (= 2 attempts already exhausted)
    for i, ts in enumerate(["20260520T120000", "20260521T080000"]):
        archive = tmp_path / "trials" / f"t0.before_resume_{ts}"
        archive.mkdir(parents=True)
        (archive / "meta.json").write_text(
            json.dumps({"status": "failed", "termination_reason": "model_quota_exhausted"}),
            encoding="utf-8",
        )
    # 2 archives + 1 live = 3 attempts so far. cap=3 → STOP.
    replayed, pending = _partition_resumed_trials(storage, [trial], max_attempts=3)
    assert pending == []
    assert len(replayed) == 1
    # cap=4 still allows one more retry
    replayed2, pending2 = _partition_resumed_trials(storage, [trial], max_attempts=4)
    assert [t.id for t in pending2] == ["t0"]


def test_max_attempts_zero_disables_cap(tmp_path: Path) -> None:
    """max_attempts=0 → no cap (legacy behavior)."""
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "target_unavailable",
    })
    # Even with 100 prior archives, cap=0 disables the bound.
    for i in range(5):
        archive = tmp_path / "trials" / f"t0.before_resume_20260521T00000{i}"
        archive.mkdir(parents=True)
        (archive / "meta.json").write_text("{}", encoding="utf-8")
    _, pending = _partition_resumed_trials(storage, [trial], max_attempts=0)
    assert [t.id for t in pending] == ["t0"]


def test_max_attempts_does_not_affect_completed_trials(tmp_path: Path) -> None:
    """A trial that already completed successfully should never be re-run,
    even if max_attempts has not been reached."""
    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "completed",
        "termination_reason": "completed",
    })
    replayed, pending = _partition_resumed_trials(storage, [trial], max_attempts=10)
    assert pending == []
    assert len(replayed) == 1


def test_load_experiment_parses_resume_max_attempts(tmp_path: Path) -> None:
    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n"
        "  max_attempts: 5\n",
        encoding="utf-8",
    )
    from cage.config.experiment import resolve

    cfg = resolve(project_file)
    assert cfg.resume_max_attempts == 5


def test_load_experiment_defaults_resume_max_attempts_to_unlimited(tmp_path: Path) -> None:
    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n",
        encoding="utf-8",
    )
    from cage.config.experiment import resolve

    cfg = resolve(project_file)
    assert cfg.resume_max_attempts == 0


def test_load_experiment_rejects_invalid_max_attempts(tmp_path: Path) -> None:
    import pytest

    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n"
        "  max_attempts: -1\n",
        encoding="utf-8",
    )
    from cage.config.experiment import resolve

    with pytest.raises(ValueError, match="resume.max_attempts must be >= 0"):
        resolve(project_file)


def _write_progress(storage: RunStorage, trial_id: str, total_requests: int) -> None:
    proxy_dir = storage.run_dir / "trials" / trial_id / "proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    (proxy_dir / "progress.json").write_text(
        json.dumps({"total_requests": total_requests}), encoding="utf-8"
    )


def test_keep_if_min_rounds_salvages_long_trial(tmp_path: Path) -> None:
    """A model_error trial that ran >= min_rounds is KEPT, not re-run."""
    from cage.config.experiment import ResumeKeepIf

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "model_error",
    })
    _write_progress(storage, "t0", 157)

    # Without keep_if: model_error re-runs.
    _, pending = _partition_resumed_trials(storage, [trial])
    assert [t.id for t in pending] == ["t0"]

    # With keep_if min_rounds=100: 157 >= 100 → salvaged (replayed).
    replayed, pending = _partition_resumed_trials(
        storage, [trial], keep_if=ResumeKeepIf(min_rounds=100)
    )
    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_keep_if_min_rounds_below_threshold_reruns(tmp_path: Path) -> None:
    """A model_error trial that ran < min_rounds is still re-run."""
    from cage.config.experiment import ResumeKeepIf

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "model_error",
    })
    _write_progress(storage, "t0", 1)  # rejected on the very first request

    _, pending = _partition_resumed_trials(
        storage, [trial], keep_if=ResumeKeepIf(min_rounds=100)
    )
    assert [t.id for t in pending] == ["t0"]


def test_keep_if_missing_progress_reruns(tmp_path: Path) -> None:
    """No progress.json → cannot prove enough rounds → re-run (no veto)."""
    from cage.config.experiment import ResumeKeepIf

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "target_unavailable",
    })
    _, pending = _partition_resumed_trials(
        storage, [trial], keep_if=ResumeKeepIf(min_rounds=100)
    )
    assert [t.id for t in pending] == ["t0"]


def test_keep_if_min_duration_salvages(tmp_path: Path) -> None:
    from cage.config.experiment import ResumeKeepIf

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "model_error",
        "timing": {"duration_ms": 2_552_606},  # ~42 min
    })
    replayed, pending = _partition_resumed_trials(
        storage, [trial], keep_if=ResumeKeepIf(min_duration_s=1800)
    )
    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_keep_if_id_matches_salvages(tmp_path: Path) -> None:
    from cage.config.experiment import ResumeKeepIf

    storage = RunStorage(run_dir=tmp_path)
    keep = _make_trial(0, "range-8/pass_3")
    drop = _make_trial(1, "range-2/pass_3")
    for t in (keep, drop):
        _write_meta(storage, t.id, {
            "status": "failed", "termination_reason": "model_error",
        })
    replayed, pending = _partition_resumed_trials(
        storage, [keep, drop], keep_if=ResumeKeepIf(id_matches=r"range-8")
    )
    assert [t.id for t in pending] == ["range-2/pass_3"]
    assert [r.trial_id for r in replayed] == ["range-8/pass_3"]


def test_select_id_pattern_gates_rerun(tmp_path: Path) -> None:
    """Only trials whose id matches select are eligible to re-run; the rest
    replay regardless of (retryable) termination reason."""
    storage = RunStorage(run_dir=tmp_path)
    t_match = _make_trial(0, "range-8/pass_3")
    t_other = _make_trial(1, "range-2/pass_3")
    for t in (t_match, t_other):
        _write_meta(storage, t.id, {
            "status": "failed", "termination_reason": "target_unavailable",
        })
    replayed, pending = _partition_resumed_trials(
        storage, [t_match, t_other], select_id_pattern=r"range-8"
    )
    assert [t.id for t in pending] == ["range-8/pass_3"]
    assert [r.trial_id for r in replayed] == ["range-2/pass_3"]


def test_completed_trial_ignores_keep_if_and_select(tmp_path: Path) -> None:
    """keep_if/select only ever affect retry-eligible trials — a completed
    trial always replays."""
    from cage.config.experiment import ResumeKeepIf

    storage = RunStorage(run_dir=tmp_path)
    trial = _make_trial(0, "t0")
    _write_meta(storage, "t0", {
        "status": "completed", "termination_reason": "completed",
    })
    replayed, pending = _partition_resumed_trials(
        storage, [trial],
        keep_if=ResumeKeepIf(min_rounds=100),
        select_id_pattern=r"nomatch",
    )
    assert pending == []
    assert [r.trial_id for r in replayed] == ["t0"]


def test_decision_labels_and_categories(tmp_path: Path) -> None:
    """The shared decision helper tags each trial with a stable category and a
    detailed label (consumed by the dry-run preview / resume log)."""
    from cage.config.experiment import ResumeKeepIf
    from cage.experiment.engine.resume import _resolve_retry_reasons, _resume_decisions

    storage = RunStorage(run_dir=tmp_path)
    salvage = _make_trial(0, "t-keep")
    rerun = _make_trial(1, "t-rerun")
    _write_meta(storage, "t-keep", {
        "status": "failed", "termination_reason": "model_error",
    })
    _write_progress(storage, "t-keep", 200)
    _write_meta(storage, "t-rerun", {
        "status": "failed", "termination_reason": "model_error",
    })
    _write_progress(storage, "t-rerun", 5)

    decisions = dict(
        (t.id, d) for t, d in _resume_decisions(
            storage, [salvage, rerun], _resolve_retry_reasons([]),
            keep_if=ResumeKeepIf(min_rounds=100),
        )
    )
    assert decisions["t-keep"].action == "replay"
    assert decisions["t-keep"].category == "keep_if:min_rounds"
    assert "ran 200 ≥ 100" in decisions["t-keep"].label
    assert decisions["t-rerun"].action == "rerun"
    assert decisions["t-rerun"].category == "model_error"


def test_load_experiment_parses_keep_if_and_select(tmp_path: Path) -> None:
    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n"
        "  retry_reasons: [model_error]\n"
        "  select:\n"
        "    id_matches: 'range-8'\n"
        "  keep_if:\n"
        "    min_rounds: 100\n"
        "    min_duration_s: 1800\n"
        "    id_matches: 'range-3'\n",
        encoding="utf-8",
    )
    from cage.config.experiment import resolve

    cfg = resolve(project_file)
    assert cfg.resume_select_id_pattern == "range-8"
    assert cfg.resume_keep_if.min_rounds == 100
    assert cfg.resume_keep_if.min_duration_s == 1800.0
    assert cfg.resume_keep_if.id_matches == "range-3"


def test_load_experiment_rejects_bad_keep_if_regex(tmp_path: Path) -> None:
    import pytest

    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n"
        "  keep_if:\n"
        "    id_matches: '(unclosed'\n",
        encoding="utf-8",
    )
    from cage.config.experiment import resolve

    with pytest.raises(ValueError, match="resume.keep_if.id_matches is not a valid regex"):
        resolve(project_file)


def test_load_experiment_rejects_negative_min_rounds(tmp_path: Path) -> None:
    import pytest

    bench_module = tmp_path / "benchmark.py"
    bench_module.write_text(
        "from cage.benchmarks import Benchmark\n"
        "from cage.scoring import Scorer\n"
        "class FakeScorer(Scorer):\n"
        "    def score(self, ctx):\n"
        "        return {}\n"
        "class FakeBench(Benchmark):\n"
        "    name = 'fake'\n"
        "    def iter_samples(self):\n"
        "        return iter([])\n"
        "    def build_prompt(self, sample):\n"
        "        return ''\n"
        "    def prepare_trial(self, container, sample, workspace_dir):\n"
        "        return None\n"
        "    def scorer(self):\n"
        "        return FakeScorer()\n",
        encoding="utf-8",
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        "models:\n"
        "  fake-model:\n"
        "    provider: openai\n"
        "    model_id: gpt-x\n"
        "    api_base: http://localhost\n"
        "    api_key_env: NONEXISTENT_KEY\n",
        encoding="utf-8",
    )
    project_file = tmp_path / "project.yml"
    project_file.write_text(
        "project:\n  name: t\n"
        f"models_file: {models_file.name}\n"
        "eval:\n  benchmark:\n    module: benchmark.py\n    class: FakeBench\n"
        "agents: []\n"
        "resume:\n"
        "  keep_if:\n"
        "    min_rounds: -5\n",
        encoding="utf-8",
    )
    from cage.config.experiment import resolve

    with pytest.raises(ValueError, match="resume.keep_if.min_rounds must be >= 0"):
        resolve(project_file)


def test_mixed_batch(tmp_path: Path) -> None:
    """One target_unavailable re-runs, one agent_exit_nonzero replays, one completed replays."""
    storage = RunStorage(run_dir=tmp_path)
    trials = [_make_trial(i, f"t{i}") for i in range(3)]
    _write_meta(storage, "t0", {
        "status": "failed",
        "termination_reason": "target_unavailable",
    })
    _write_meta(storage, "t1", {
        "status": "failed",
        "termination_reason": "agent_exit_nonzero",
    })
    _write_meta(storage, "t2", {
        "status": "completed",
        "termination_reason": "completed",
    })

    replayed, pending = _partition_resumed_trials(storage, trials)

    assert [t.id for t in pending] == ["t0"]
    assert sorted(r.trial_id for r in replayed) == ["t1", "t2"]
