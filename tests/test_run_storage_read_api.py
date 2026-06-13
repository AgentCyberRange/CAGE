"""Tests for ``RunStorage``'s read half — the artifact access authority.

Covers the wrapped run-level / trial-level loaders, the dashboard-projection
overview read (``iter_trial_summaries``), the ``discover_runs`` navigation
helper, and the canonical-first → legacy-fallback policy.
"""

from __future__ import annotations

import json
from pathlib import Path

from cage.artifacts.run_storage import (
    RUN_ARTIFACTS,
    TRIAL_ARTIFACTS,
    RunRef,
    RunStorage,
    discover_runs,
)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_run(root: Path, *, benchmark="cybergym",
              agent_label="claude_code:nex-n2:stateless", run_id="nex-n2-100") -> Path:
    """Build a minimal legacy-layout run under root/<benchmark>/.cage_runs/..."""
    run_dir = root / benchmark / ".cage_runs" / agent_label / run_id
    trial_dir = run_dir / "trials" / "arvo_1"
    trial_dir.mkdir(parents=True)

    _write_json(run_dir / "dashboard.json", {
        "run_id": run_id,
        "status": "completed",
        "completed_at": "2026-06-10T20:56:00",
        "agents": {
            agent_label: {
                "total": 1, "completed": 1, "failed": 0,
                "trials": [{
                    "trial_id": "arvo_1", "trial_index": 0, "status": "completed",
                    "exit_code": 0, "duration_ms": 1234, "sample_id": "arvo_1",
                    "scores": {"cybergym": 1.0}, "usage": {"input_tokens": 10},
                }],
            },
        },
    })
    _write_json(run_dir / "experiment_record.json", {"run_id": run_id, "status": "completed"})
    _write_json(run_dir / "planned_trials.json", [{"trial_id": "arvo_1"}])
    _write_json(run_dir / "run_history.json", [{"invocation": 1}])
    (run_dir / "config.yaml").write_text("models: [m1]\n", encoding="utf-8")

    _write_json(trial_dir / "meta.json", {"status": "completed", "exit_code": 0,
                                          "timing": {"duration_ms": 1234}})
    _write_json(trial_dir / "task_output.json", {"output": "hello", "sample": {"id": "arvo_1"}})
    _write_json(trial_dir / "scores" / "cybergym.json", {"cybergym": {"value": 1.0}})
    _write_json(trial_dir / "proxy" / "progress.json", {"total_requests": 7})
    (trial_dir / "prompt.txt").write_text("do the task", encoding="utf-8")
    return run_dir


def test_manifest_declares_every_loader() -> None:
    # Each artifact spec names a non-empty purpose + accessor (the semantic map).
    for spec in RUN_ARTIFACTS + TRIAL_ARTIFACTS:
        assert spec.name and spec.purpose and spec.accessor
        assert spec.level in {"run", "trial"}


def test_discover_runs_parses_coordinates(tmp_path: Path) -> None:
    _make_run(tmp_path)
    refs = discover_runs(tmp_path)
    assert len(refs) == 1
    ref = refs[0]
    assert isinstance(ref, RunRef)
    assert (ref.benchmark, ref.agent, ref.model, ref.lifecycle, ref.run_id) == (
        "cybergym", "claude_code", "nex-n2", "stateless", "nex-n2-100"
    )
    assert ref.run_dir.is_dir()


def test_iter_trial_summaries_reads_projection(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    rows = RunStorage(run_dir).iter_trial_summaries()
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["scores"] == {"cybergym": 1.0}


def test_iter_trial_summaries_empty_without_dashboard(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    (run_dir / "trials").mkdir(parents=True)
    assert RunStorage(run_dir).iter_trial_summaries() == []


def test_run_level_loaders(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    st = RunStorage(run_dir)
    assert st.load_experiment_record()["status"] == "completed"
    assert st.load_planned_trials() == [{"trial_id": "arvo_1"}]
    assert st.load_run_history() == [{"invocation": 1}]
    assert st.load_config() == {"models": ["m1"]}
    # absent files -> empty containers, never raise
    assert st.load_metrics() == {}
    assert st.load_preflight() == {}
    assert st.has_artifact_index() is False


def test_trial_level_loaders_legacy(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    st = RunStorage(run_dir)
    trial_dir = st.iter_trial_dirs()[0]
    assert st.load_trial_meta(trial_dir)["status"] == "completed"
    assert st.load_trial_output(trial_dir)["output"] == "hello"
    assert st.load_trial_scores(trial_dir) == {"cybergym": {"value": 1.0}}
    assert st.load_trial_progress(trial_dir)["total_requests"] == 7
    assert st.load_trial_prompt(trial_dir) == "do the task"


def test_trial_loaders_missing_files_are_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "r" / ".cage_runs" / "a:b:stateless" / "run-1"
    trial_dir = run_dir / "trials" / "t1"
    trial_dir.mkdir(parents=True)
    st = RunStorage(run_dir)
    assert st.load_trial_meta(trial_dir) == {}
    assert st.load_trial_output(trial_dir) == {}
    assert st.load_trial_scores(trial_dir) == {}


def test_canonical_artifact_wins_over_legacy(tmp_path: Path, monkeypatch) -> None:
    """When a canonical task_output artifact resolves, it is preferred."""
    run_dir = _make_run(tmp_path)
    st = RunStorage(run_dir)
    trial_dir = st.iter_trial_dirs()[0]

    canonical_path = tmp_path / "canonical_output.json"
    _write_json(canonical_path, {"output": "canonical", "sample": {}})

    class _Art:
        kind = "task_output"
        path = canonical_path

    monkeypatch.setattr(RunStorage, "_resolved_trial_artifacts",
                        lambda self, td: [_Art()])
    assert st.load_trial_output(trial_dir)["output"] == "canonical"


def test_split_trial_dir_flat_and_nested() -> None:
    flat = Path("/x/run-1/trials/arvo_1")
    run_dir, tid = RunStorage._split_trial_dir(flat)
    assert run_dir == Path("/x/run-1") and tid == "arvo_1"

    nested = Path("/x/run-1/trials/chal/variant")
    run_dir, tid = RunStorage._split_trial_dir(nested)
    assert run_dir == Path("/x/run-1") and tid == "chal/variant"
