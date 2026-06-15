"""Tests for live trial progress in the web inspector."""

from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path

from cage.config import WebInspectorUIConfig
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.artifacts.resources import ResourceLedgerWriter
from cage.experiment.model import (
    TrialTermination,
    build_experiment_plan,
    load_experiment_spec,
)
from cage.web.app import create_app
from cage.web.data import (
    find_trial_dirs,
    group_runs,
    load_run_history,
    load_trial_summary,
    scan_runs,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


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


def _dt_ms(text: str) -> int:
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def _codex_stream(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def _extract_json_script(html: str, element_id: str) -> dict:
    match = re.search(
        rf'<script id="{re.escape(element_id)}" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match, f"missing JSON script {element_id}"
    return json.loads(match.group(1))


def test_load_run_history_reconstructs_resume_windows_from_trial_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-fixed"

    _write_json(
        run_dir / "trials" / "trial-a.before_resume_20260428T110000" / "meta.json",
        {
            "status": "failed",
            "timing": {
                "started_at_ms": _dt_ms("2026-04-28T10:00:00"),
                "ended_at_ms": _dt_ms("2026-04-28T10:20:00"),
                "duration_ms": 20 * 60 * 1000,
            },
        },
    )
    _write_json(
        run_dir / "trials" / "trial-a.before_resume_20260428T130000" / "meta.json",
        {
            "status": "failed",
            "timing": {
                "started_at_ms": _dt_ms("2026-04-28T11:05:00"),
                "ended_at_ms": _dt_ms("2026-04-28T11:25:00"),
                "duration_ms": 20 * 60 * 1000,
            },
        },
    )
    _write_json(
        run_dir / "trials" / "trial-a" / "meta.json",
        {
            "status": "completed",
            "timing": {
                "started_at_ms": _dt_ms("2026-04-28T13:10:00"),
                "ended_at_ms": _dt_ms("2026-04-28T13:18:00"),
                "duration_ms": 8 * 60 * 1000,
            },
        },
    )
    # Completed on the original run and never retried; this should widen
    # only the first run window, not the latest resume window.
    _write_json(
        run_dir / "trials" / "trial-b" / "meta.json",
        {
            "status": "completed",
            "timing": {
                "started_at_ms": _dt_ms("2026-04-28T10:02:00"),
                "ended_at_ms": _dt_ms("2026-04-28T10:08:00"),
                "duration_ms": 6 * 60 * 1000,
            },
        },
    )

    history = load_run_history(run_dir, dashboard={})

    assert [entry["label"] for entry in history] == [
        "Initial run",
        "Resume #1",
        "Resume #2",
    ]
    assert [entry["trial_attempts"] for entry in history] == [2, 1, 1]
    assert history[0]["started_at"] == "2026-04-28T10:00:00"
    assert history[0]["completed_at"] == "2026-04-28T10:20:00"
    assert history[1]["started_at"] == "2026-04-28T11:05:00"
    assert history[1]["completed_at"] == "2026-04-28T11:25:00"
    assert history[2]["started_at"] == "2026-04-28T13:10:00"
    assert history[2]["completed_at"] == "2026-04-28T13:18:00"
    assert history[2]["is_latest"] is True
    assert history[2]["source"] == "reconstructed"


def test_run_detail_renders_recorded_run_history(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "cvebench" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "cvebench",
            "started_at": "2026-04-28T11:00:00",
            "completed_at": "2026-04-28T11:05:00",
            "run_history": {
                "version": 1,
                "attempts": [
                    {
                        "label": "Initial run",
                        "started_at": "2026-04-28T10:00:00",
                        "completed_at": "2026-04-28T10:20:00",
                        "duration_ms": 20 * 60 * 1000,
                        "status": "interrupted",
                    },
                    {
                        "label": "Resume #1",
                        "started_at": "2026-04-28T11:00:00",
                        "completed_at": "2026-04-28T11:05:00",
                        "duration_ms": 5 * 60 * 1000,
                        "status": "completed",
                    },
                ],
            },
            "agents": {
                "agent:model:stateless": {
                    "completed": 0,
                    "failed": 0,
                    "total": 0,
                    "trials": [],
                }
            },
        },
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Run history" in html
    assert 'data-run-history-summary' not in html
    assert "Latest invocation" not in html
    assert "Initial run" in html
    assert "Resume #1" in html
    assert "Last run" in html
    assert "2026-04-28T10:00:00" in html
    assert "2026-04-28T11:00:00" in html
    assert '<details class="mt-5 border-t border-slate-700/50 pt-4">' in html
    assert '<details class="mt-5 border-t border-slate-700/50 pt-4" open>' not in html


def test_find_trial_dirs_includes_progress_only_running_trial(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 3,
            "errors": 1,
            "last_status": "success",
            "last_ts_ms": 1777322280300,
            "tokens_in": 1234,
            "tokens_out": 567,
            "tools_used": {"Bash": 2},
        },
    )

    assert find_trial_dirs(run_dir) == [trial_dir]


def test_load_trial_summary_marks_progress_only_trial_as_running(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 3,
            "errors": 1,
            "last_status": "success",
            "last_ts_ms": 1777322280300,
            "tokens_in": 1234,
            "tokens_out": 567,
            "tools_used": {"Bash": 2},
        },
    )

    summary = load_trial_summary(trial_dir, {})

    assert summary["running"] is True
    assert summary["progress"]["total_requests"] == 4
    assert summary["usage"] == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "reasoning_tokens": 0,
        "num_requests": 3,
    }
    assert summary["status_label"] == "Running"


def test_load_trial_summary_marks_in_flight_trial_interrupted_when_run_interrupted(
    tmp_path: Path,
) -> None:
    """A trial in-flight at Ctrl+C must classify as Interrupted, not Running.

    Regression: the partial-results table (and a settled interrupted run) showed
    ``Running`` for trials whose meta.json never finalized past "running",
    because the ``running`` branch short-circuited before the interrupted-run
    override. The run loop has stopped, so nothing is actually running.
    """
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 3,
            "last_status": "success",
        },
    )

    summary = load_trial_summary(trial_dir, {}, run_status="interrupted")

    assert summary["status_label"] == "Interrupted"
    assert summary["status_kind"] == "warning"


def test_load_trial_summary_carries_nonzero_progress_cost(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 3,
            "errors": 1,
            "tokens_in": 1234,
            "tokens_out": 567,
            "cost_usd": 0.0579,
        },
    )

    summary = load_trial_summary(trial_dir, {})

    assert summary["usage"]["cost_usd"] == 0.0579


def test_load_trial_summary_refreshes_cached_running_duration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 3,
            "errors": 1,
            "started_at_ms": 1_000_000,
        },
    )
    now = {"value": 1_600.0}
    monkeypatch.setattr("cage.web.data.time.time", lambda: now["value"])

    first = load_trial_summary(trial_dir, {})
    now["value"] = 1_605.0
    second = load_trial_summary(trial_dir, {})

    assert first["duration_ms"] == 600_000
    assert second["duration_ms"] == 605_000


def test_load_trial_summary_extracts_filter_tags_from_sample(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range2-L1"
    _write_json(
        trial_dir / "meta.json",
        {"trial_id": "range2-L1", "trial_index": 1, "sample_id": "range2"},
    )
    _write_json(
        trial_dir / "task_output.json",
        {
            "output": "",
            "sample": {
                "id": "range2",
                "benchmark": "agent_pentest_bench",
                "category": "web",
                "name": "Range 2 (L1)",
                "hint_level": 1,
                "tags": ["wordpress", "pivot"],
            },
        },
    )

    summary = load_trial_summary(trial_dir, {})

    assert summary["tags"] == [
        "L1",
        "benchmark:agent_pentest_bench",
        "category:web",
        "pivot",
        "range2",
        "wordpress",
    ]


def test_load_trial_summary_explains_user_interrupted_trial(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
        },
    )

    summary = load_trial_summary(trial_dir, {}, run_status="interrupted")

    assert summary["status_label"] == "Interrupted"
    assert summary["status_detail"] == "Stopped by user (Ctrl+C)"
    assert summary["status_kind"] == "warning"


def test_load_trial_summary_explains_model_timeout_from_metadata(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
            "exit_code": 1,
            "termination_reason": "model_timeout",
            "termination_detail": "timed out after 300 seconds",
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "", "sample": {}})

    summary = load_trial_summary(trial_dir, {})

    assert summary["status_label"] == "Model timeout"
    assert summary["status_detail"] == "timed out after 300 seconds"
    assert summary["status_kind"] == "error"


def test_load_trial_summary_labels_runtime_budget_reasons(tmp_path: Path) -> None:
    cases = {
        "max_input_tokens_reached": "Input tokens",
        "max_output_tokens_reached": "Output tokens",
        "max_cost_reached": "Max cost",
    }
    for reason, label in cases.items():
        trial_dir = tmp_path / reason
        _write_json(
            trial_dir / "meta.json",
            {
                "trial_id": reason,
                "trial_index": 0,
                "sample_id": "range1",
                "exit_code": 1,
                "termination_reason": reason,
                "termination_detail": "budget reached",
            },
        )
        _write_json(trial_dir / "task_output.json", {"output": "", "sample": {}})

        summary = load_trial_summary(trial_dir, {})

        assert summary["status_label"] == label
        assert summary["status_detail"] == "budget reached"
        assert summary["status_kind"] == "warning"


def test_load_trial_summary_hides_resume_policy_repair_detail(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    internal_detail = (
        "Restored after overly broad resume selection; excluded from final L2-only "
        "rerun policy. Previous resume-policy marker was status=failed, "
        "reason=model_error."
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
            "exit_code": 0,
            "termination_reason": "completed",
            "termination_detail": internal_detail,
            "termination_source": "resume_policy_repair",
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "done", "sample": {}})

    summary = load_trial_summary(trial_dir, {})

    assert summary["status_label"] == "Completed"
    assert summary["status_detail"] == "Task finished"
    assert summary["status_kind"] == "success"
    assert "Restored after overly broad resume selection" not in summary["status_detail"]
    assert "Previous resume-policy marker" not in summary["status_detail"]
    assert "model_error" not in summary["status_detail"]


def test_load_trial_summary_parses_codex_output_and_hides_raw_model_timeout_stream(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    stream = _codex_stream(
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": "The first foothold is established.",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "agent_message",
                "text": "ActiveMQ RCE is confirmed as root, and I retrieved flag3.",
            },
        },
        {
            "type": "turn.failed",
            "error": {
                "message": "unexpected status 503 Service Unavailable",
            },
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
            "exit_code": 1,
            "termination_reason": "model_timeout",
            "termination_detail": stream,
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": stream, "sample": {}})

    summary = load_trial_summary(trial_dir, {})

    assert summary["status_label"] == "Model timeout"
    assert summary["status_detail"] == "Upstream request timed out"
    assert "503 Service Unavailable" not in summary["status_detail"]
    assert summary["output"] == "ActiveMQ RCE is confirmed as root, and I retrieved flag3."


def test_load_trial_summary_does_not_read_proxy_to_infer_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L0"
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
            "exit_code": 1,
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "", "sample": {}})
    (trial_dir / "proxy").mkdir()
    (trial_dir / "proxy" / "proxy.jsonl").write_text(
        json.dumps({"status": "error", "error": "timed out after 300 seconds"}) + "\n",
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def fail_if_proxy_is_read(path: Path, *args, **kwargs) -> str:
        if path.name == "proxy.jsonl":
            raise AssertionError("proxy.jsonl should not be read for status inference")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_if_proxy_is_read)

    summary = load_trial_summary(trial_dir, {})

    assert summary["status_label"] == "Agent failed"
    assert summary["status_detail"] == "Exited with code 1"
    assert summary["status_kind"] == "error"


def test_load_trial_summary_marks_exit_zero_trial_interrupted_when_run_interrupted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "range1-L3"
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L3",
            "trial_index": 2,
            "sample_id": "range1-L3",
            "exit_code": 0,
        },
    )
    _write_json(
        trial_dir / "task_output.json",
        {
            "output": "There's a metasploit module for CVE-2023-46604. Let me use it.",
            "sample": {},
        },
    )
    (trial_dir / "proxy").mkdir()
    (trial_dir / "proxy" / "proxy.jsonl").write_text(
        json.dumps({"status": "error", "error": "The read operation timed out"}) + "\n",
        encoding="utf-8",
    )

    original_read_text = Path.read_text

    def fail_if_proxy_is_read(path: Path, *args, **kwargs) -> str:
        if path.name == "proxy.jsonl":
            raise AssertionError("proxy.jsonl should not be read for run-level interrupted status")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_if_proxy_is_read)

    summary = load_trial_summary(trial_dir, {}, run_status="interrupted")

    assert summary["status_label"] == "Interrupted"
    assert summary["status_detail"] == "Stopped by user (Ctrl+C)"
    assert summary["status_kind"] == "warning"


def test_run_page_renders_running_trial_progress(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "agents": {"agent:model": {"completed": 0, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(
        run_dir / "trials" / "range1-L0" / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 3,
            "errors": 1,
            "last_status": "success",
            "last_ts_ms": 1777322280300,
            "tokens_in": 1234,
            "tokens_out": 567,
            "tools_used": {"Bash": 2},
        },
    )

    app = create_app(root)
    encoded = base64.urlsafe_b64encode(
        str(run_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()

    response = app.test_client().get(f"/run/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "range1-L0" in html
    assert "Status" in html
    assert "Running" in html
    assert "4 steps" in html
    assert "25.0% errors" in html
    assert "active" in html


def test_run_page_renders_trial_filters_and_tag_data(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:20:00",
            "agents": {"agent:model": {"completed": 2, "failed": 0, "total": 2, "trials": []}},
        },
    )
    _write_json(
        run_dir / "trials" / "range1-L0" / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "timing": {"duration_ms": 600000},
            "exit_code": 0,
        },
    )
    _write_json(
        run_dir / "trials" / "range1-L0" / "task_output.json",
        {
            "output": "long",
            "sample": {
                "id": "range1",
                "benchmark": "agent_pentest_bench",
                "category": "web",
                "hint_level": 0,
            },
        },
    )
    _write_json(
        run_dir / "trials" / "range2-L1" / "meta.json",
        {
            "trial_id": "range2-L1",
            "trial_index": 1,
            "timing": {"duration_ms": 120000},
            "exit_code": 0,
        },
    )
    _write_json(
        run_dir / "trials" / "range2-L1" / "task_output.json",
        {
            "output": "short",
            "sample": {
                "id": "range2",
                "benchmark": "agent_pentest_bench",
                "category": "cms",
                "hint_level": 1,
            },
        },
    )

    app = create_app(root)
    encoded = base64.urlsafe_b64encode(
        str(run_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()

    response = app.test_client().get(f"/run/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="trial-filters"' in html
    assert 'data-default-min-duration-ms="0"' in html
    assert 'data-duration-ms="600000"' in html
    assert 'data-duration-ms="120000"' in html
    assert 'data-tags="L0||benchmark:agent_pentest_bench||category:web||range1"' in html
    assert 'data-tags="L1||benchmark:agent_pentest_bench||category:cms||range2"' in html
    assert "category:web" in html
    assert "category:cms" in html


def test_run_page_uses_configured_trial_duration_filter_default(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:20:00",
            "agents": {"agent:model": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(
        run_dir / "trials" / "range1-L0" / "meta.json",
        {
            "trial_id": "range1-L0",
            "timing": {"duration_ms": 600000},
            "exit_code": 0,
        },
    )
    _write_json(run_dir / "trials" / "range1-L0" / "task_output.json", {"output": "done"})
    encoded = base64.urlsafe_b64encode(
        str(run_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()

    response = create_app(
        root,
        ui=WebInspectorUIConfig(default_min_trial_duration_ms=480000),
    ).test_client().get(f"/run/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-default-min-duration-ms="480000"' in html
    assert '<option value="480000" selected>8m+</option>' in html
    assert '<option value="0">Any duration</option>' in html


def test_run_page_normalizes_json_output_preview(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    trial_dir = run_dir / "trials" / "range-json"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:20:00",
            "agents": {"agent:model": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range-json",
            "trial_index": 0,
            "sample_id": "range-json",
            "exit_code": 0,
        },
    )
    raw_stream = json.dumps([
        {"type": "system", "subtype": "init", "message": "bootstrap"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "I verified the target and captured the flag.",
                    }
                ]
            },
        },
    ])
    _write_json(trial_dir / "task_output.json", {"output": raw_stream, "sample": {}})

    summary = load_trial_summary(trial_dir, {})

    assert summary["output"] == "I verified the target and captured the flag."

    app = create_app(root)
    encoded = base64.urlsafe_b64encode(
        str(run_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()
    response = app.test_client().get(f"/run/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "I verified the target and captured the flag." in html
    assert "system&quot;" not in html
    assert "[{&quot;type&quot;" not in html


def test_run_page_renders_specific_trial_termination_reason(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:05:00",
            "agents": {
                "agent:model": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [
                        {
                            "trial_id": "range1-L0",
                            "exit_code": -1,
                            "termination_reason": "execution_timeout",
                            "termination_detail": "Agent execution exceeded 300s",
                        }
                    ],
                }
            },
        },
    )
    _write_json(
        run_dir / "trials" / "range1-L0" / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
            "exit_code": -1,
        },
    )
    _write_json(run_dir / "trials" / "range1-L0" / "task_output.json", {"output": "", "sample": {}})

    app = create_app(root)
    encoded = base64.urlsafe_b64encode(
        str(run_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()

    response = app.test_client().get(f"/run/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Timed out" in html
    assert "Agent execution exceeded 300s" in html


def test_trial_page_renders_specific_termination_reason(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "status": "completed",
            "agents": {
                "agent:model": {
                    "trials": [
                        {
                            "trial_id": "range1-L0",
                            "exit_code": -1,
                            "termination_reason": "execution_timeout",
                            "termination_detail": "Agent execution exceeded 300s",
                        }
                    ]
                }
            },
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "trial_index": 0,
            "sample_id": "range1",
            "exit_code": -1,
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "", "sample": {}})

    app = create_app(root)
    encoded = base64.urlsafe_b64encode(
        str(trial_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()

    response = app.test_client().get(f"/trial/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Execution:" in html
    assert "Benchmark outcome:" in html
    assert "Timed out" in html
    assert "Agent execution exceeded 300s" in html
    assert ">More termination evidence<" in html
    evidence_index = html.index(">More termination evidence<")
    evidence_prefix = html[evidence_index - 140:evidence_index]
    assert "open>\n  <summary" not in evidence_prefix
    assert "execution_timeout" in html


def test_scan_runs_marks_run_running_from_trial_progress(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = (
        root
        / "cvebench"
        / ".cage_runs"
        / "claude_code_baseline:minimax-2.5-sii:stateless"
        / "run-live"
    )
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-live",
            "experiment": "cvebench",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "agents": {
                "claude_code_baseline:minimax-2.5-sii:stateless": {
                    "completed": 0,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )
    _write_json(
        run_dir / "trials" / "cvb-CVE-2023-37999" / "proxy" / "progress.json",
        {
            "trial_id": "cvb-CVE-2023-37999",
            "total_requests": 9,
            "success": 7,
            "errors": 2,
            "last_status": "success",
            "last_ts_ms": 1777322280300,
            "tokens_in": 1234,
            "tokens_out": 567,
        },
    )

    runs = scan_runs(root)

    assert len(runs) == 1
    run = runs[0]
    assert run.running is True
    assert run.running_trials == 1
    assert run.live_total_requests == 7
    assert run.live_errors == 2
    assert run.last_active_ts_ms == 1777322280300
    assert run.agent_label == "claude_code_baseline:minimax-2.5-sii:stateless"
    assert run.agent_name == "claude_code_baseline"
    assert run.model_name == "minimax-2.5-sii"
    assert run.mode == "stateless"


def test_scan_runs_does_not_descend_into_existing_cage_runs(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "cvebench" / ".cage_runs" / "agent:model:stateless" / "run-good"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-good",
            "experiment": "cvebench",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "agents": {
                "agent:model:stateless": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )
    nested_run = (
        run_dir
        / "trials"
        / "range1"
        / "proxy"
        / ".cage_runs"
        / "nested:model:stateless"
        / "run-bad"
    )
    _write_json(
        nested_run / "dashboard.json",
        {
            "run_id": "run-bad",
            "experiment": "nested",
            "started_at": "2026-04-28T12:00:00",
            "completed_at": "2026-04-28T13:00:00",
            "agents": {
                "nested:model:stateless": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )

    runs = scan_runs(root)

    assert [run.run_id for run in runs] == ["run-good"]


def test_scan_runs_marks_incomplete_dashboard_as_running_before_progress(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "cvebench" / ".cage_runs" / "agent:model:stateless" / "run-starting"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-starting",
            "experiment": "cvebench",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "agents": {
                "agent:model:stateless": {
                    "completed": 0,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )

    run = scan_runs(root)[0]

    assert run.running is True
    assert run.running_trials == 1
    assert run.live_total_requests == 0


def test_index_groups_runs_by_benchmark_agent_model_with_running_first(tmp_path: Path) -> None:
    root = tmp_path
    live_run = (
        root
        / "cvebench"
        / ".cage_runs"
        / "claude_code_baseline:minimax-2.5-sii:stateless"
        / "run-live"
    )
    done_run = (
        root
        / "cvebench"
        / ".cage_runs"
        / "claude_code_baseline:minimax-2.5-sii:stateless"
        / "run-done"
    )
    _write_json(
        live_run / "dashboard.json",
        {
            "run_id": "run-live",
            "experiment": "cvebench",
            "started_at": "2026-04-27T10:00:00",
            "completed_at": "",
            "agents": {
                "claude_code_baseline:minimax-2.5-sii:stateless": {
                    "completed": 0,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )
    _write_json(
        live_run / "trials" / "range1-L0" / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 4,
            "success": 4,
            "errors": 0,
            "last_status": "success",
            "last_ts_ms": 1777322280300,
            "tokens_in": 100,
            "tokens_out": 50,
        },
    )
    _write_json(
        done_run / "dashboard.json",
        {
            "run_id": "run-done",
            "experiment": "cvebench",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "agents": {
                "claude_code_baseline:minimax-2.5-sii:stateless": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )

    runs = scan_runs(root)
    grouped = group_runs(runs)

    assert grouped[0]["project"] == "cvebench"
    model_group = grouped[0]["models"][0]
    assert model_group["model_name"] == "minimax-2.5-sii"
    assert model_group["running_count"] == 1
    assert [run.run_id for run in model_group["runs"]] == ["run-live", "run-done"]

    response = create_app(root).test_client().get("/benchmark/cvebench")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "cvebench" in html
    assert "claude_code_baseline" in html
    assert "minimax-2.5-sii" in html
    assert "Running" in html
    assert "4 steps" in html
    assert 'href="/cvebench/minimax-2.5-sii/run-live"' in html
    assert html.index("run-live") < html.index("run-done")


def test_root_page_lists_benchmarks_only_and_links_to_benchmark_pages(tmp_path: Path) -> None:
    root = tmp_path
    for project, run_id in (("cvebench", "run-a"), ("nyuctfbench", "run-b")):
        _write_json(
            root / project / ".cage_runs" / "codex:gpt-5.5:stateless" / run_id / "dashboard.json",
            {
                "run_id": run_id,
                "experiment": project,
                "started_at": "2026-04-27T10:00:00",
                "completed_at": "2026-04-27T10:10:00",
                "agents": {},
            },
        )

    response = create_app(root).test_client().get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    # Root lists benchmarks and links to their pages — not individual runs.
    assert "cvebench" in html
    assert "nyuctfbench" in html
    assert 'href="/benchmark/cvebench"' in html
    assert 'href="/benchmark/nyuctfbench"' in html
    # Root must NOT eagerly render per-run rows or the run-filter machinery.
    assert "run-a" not in html
    assert 'id="run-filters"' not in html


def test_benchmark_page_shows_runs_and_unknown_project_404s(tmp_path: Path) -> None:
    root = tmp_path
    _write_json(
        root / "cvebench" / ".cage_runs" / "codex:gpt-5.5:stateless" / "run-a" / "dashboard.json",
        {
            "run_id": "run-a",
            "experiment": "cvebench",
            "started_at": "2026-04-27T10:00:00",
            "completed_at": "2026-04-27T10:10:00",
            "agents": {},
        },
    )
    client = create_app(root).test_client()

    ok = client.get("/benchmark/cvebench")
    assert ok.status_code == 200
    assert "run-a" in ok.get_data(as_text=True)

    missing = client.get("/benchmark/does-not-exist")
    assert missing.status_code == 404


def test_group_runs_merges_stateful_and_stateless_under_same_model(tmp_path: Path) -> None:
    root = tmp_path
    stateless_run = root / "cvebench" / ".cage_runs" / "codex:gpt-5.5:stateless" / "run-a"
    stateful_run = root / "cvebench" / ".cage_runs" / "codex:gpt-5.5:stateful" / "run-b"
    _write_json(
        stateless_run / "dashboard.json",
        {
            "run_id": "run-a",
            "experiment": "cvebench",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:10:00",
            "agents": {
                "codex:gpt-5.5:stateless": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )
    _write_json(
        stateful_run / "dashboard.json",
        {
            "run_id": "run-b",
            "experiment": "cvebench",
            "started_at": "2026-04-28T11:00:00",
            "completed_at": "2026-04-28T11:10:00",
            "agents": {
                "codex:gpt-5.5:stateful": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )

    grouped = group_runs(scan_runs(root))
    model_group = grouped[0]["models"][0]

    assert model_group["model_name"] == "gpt-5.5"
    assert len(model_group["runs"]) == 2
    assert sorted(run.mode for run in model_group["runs"]) == ["stateful", "stateless"]


def test_index_page_renders_open_run_filters_with_configured_duration_default(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = (
        root
        / "agent_pentest_bench"
        / ".cage_runs"
        / "codex_ctfenv:gpt-5.5:stateless"
        / "run-fixed"
    )
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "agent-pentest-bench",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:12:00",
            "agents": {
                "codex_ctfenv:gpt-5.5:stateless": {
                    "completed": 1,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )

    response = create_app(
        root,
        ui=WebInspectorUIConfig(default_min_run_duration_ms=0),
    ).test_client().get("/benchmark/agent_pentest_bench")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="run-filters"' in html
    assert 'data-preserve-model-panels="true"' in html
    assert 'data-default-min-duration-ms="0"' in html
    assert "const shouldAutoOpenGroups = Boolean(agent || model || status);" in html
    assert '<option value="0" selected>Any duration</option>' in html
    assert '<option value="480000">8m+</option>' in html
    assert 'data-run-agent="codex_ctfenv"' in html
    assert 'data-run-model="gpt-5.5"' in html
    assert 'data-run-duration-ms="720000"' in html
    assert '<details class="run-project-group ' in html
    assert '<details class="run-filter-panel ' in html
    assert "Live operations" not in html
    assert "Archive analysis" not in html
    assert 'data-mode-jump="live"' not in html
    assert 'data-mode-jump="archive"' not in html
    assert 'run-filter-panel bg-surface border border-slate-700/50 rounded-lg mb-6" open' in html
    assert 'run-project-group rounded-lg border border-slate-800/70 bg-slate-950/20" open' in html
    assert 'data-model-panel="gpt-5.5"' in html
    assert 'data-model-chip="gpt-5.5"' in html
    assert (
        "run-model-panel rounded-xl border border-slate-700/60 bg-slate-900/35 p-4 "
        'shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]" open'
        not in html
    )


def test_index_page_uses_configured_run_duration_filter_default(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:12:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )

    response = create_app(
        root,
        ui=WebInspectorUIConfig(default_min_run_duration_ms=1800000),
    ).test_client().get("/benchmark/project")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-default-min-duration-ms="1800000"' in html
    assert '<option value="1800000" selected>30m+</option>' in html
    assert '<option value="0">Any duration</option>' in html


def test_index_page_keeps_entry_dashboard_plain(tmp_path: Path) -> None:
    root = tmp_path
    recent_ms = int(time.time() * 1000)
    healthy_run = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-healthy"
    failing_run = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-failing"
    _write_json(
        healthy_run / "dashboard.json",
        {
            "run_id": "run-healthy",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:20:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 3, "failed": 0, "total": 3, "trials": []}},
        },
    )
    _write_json(
        failing_run / "dashboard.json",
        {
            "run_id": "run-failing",
            "experiment": "demo",
            "started_at": "2026-04-28T11:00:00",
            "completed_at": "",
            "status": "running",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 3, "total": 4, "trials": []}},
        },
    )
    _write_json(
        failing_run / "trials" / "active" / "proxy" / "progress.json",
        {
            "total_requests": 9,
            "success": 8,
            "errors": 1,
            "last_ts_ms": recent_ms,
        },
    )
    for trial_id in ("good-1", "good-2", "good-3"):
        _write_json(
            healthy_run / "trials" / trial_id / "meta.json",
            {"trial_id": trial_id, "exit_code": 0, "termination_reason": "completed"},
        )
        _write_json(healthy_run / "trials" / trial_id / "task_output.json", {"output": "ok", "sample": {}})
    _write_json(
        failing_run / "trials" / "good" / "meta.json",
        {"trial_id": "good", "exit_code": 0, "termination_reason": "completed"},
    )
    _write_json(failing_run / "trials" / "good" / "task_output.json", {"output": "ok", "sample": {}})
    for trial_id in ("bad-1", "bad-2", "bad-3"):
        _write_json(
            failing_run / "trials" / trial_id / "meta.json",
            {"trial_id": trial_id, "exit_code": 2, "termination_reason": "agent_exit_nonzero"},
        )
        _write_json(failing_run / "trials" / trial_id / "task_output.json", {"output": "", "sample": {}})

    response = create_app(root).test_client().get("/benchmark/project")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Active now" in html
    assert "Scan overview" not in html
    assert "What this inspector found under the selected root." not in html
    assert "Live operations" not in html
    assert "Archive analysis" not in html
    assert "Failure-heavy" not in html
    assert "Needs audit" not in html
    assert "Need Audit" not in html
    assert 'data-index-failure-heavy-runs' not in html
    assert "Last activity" not in html
    assert f'data-index-last-active="{recent_ms}"' not in html


def test_index_page_polls_even_when_no_runs_are_active(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-done"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-done",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:12:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )

    response = create_app(root).test_client().get("/benchmark/project")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Active now" in html
    assert "No active runs" in html
    assert "endpoint: '/api/runs'" in html
    assert "if (anyStatusChanged) needsReload = true;" in html


def test_index_uses_encoded_run_urls_when_readable_identity_collides(tmp_path: Path) -> None:
    root = tmp_path
    # Two agents under the same benchmark share model + run_id, so the canonical
    # (benchmark, model, run_id) address collides and must fall back to encoded.
    run_dirs = [
        root / "project" / ".cage_runs" / "agent-a:model:stateless" / "run-fixed",
        root / "project" / ".cage_runs" / "agent-b:model:stateless" / "run-fixed",
    ]
    for index, run_dir in enumerate(run_dirs):
        _write_json(
            run_dir / "dashboard.json",
            {
                "run_id": "run-fixed",
                "experiment": f"demo-{index}",
                "started_at": "2026-04-28T10:00:00",
                "completed_at": "2026-04-28T10:12:00",
                "status": "completed",
                "agents": {f"agent-{'ab'[index]}:model:stateless": {"trials": []}},
            },
        )

    response = create_app(root).test_client().get("/benchmark/project")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'href="/project/model/run-fixed"' not in html
    for run_dir in run_dirs:
        rel = str(run_dir.resolve().relative_to(root.resolve()))
        encoded = base64.urlsafe_b64encode(rel.encode()).decode()
        assert f'href="/run/{encoded}"' in html


def test_run_detail_uses_encoded_trial_urls_when_run_identity_collides(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dirs = [
        root / "left" / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed",
        root / "right" / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed",
    ]
    for index, run_dir in enumerate(run_dirs):
        _write_json(
            run_dir / "dashboard.json",
            {
                "run_id": "run-fixed",
                "experiment": f"demo-{index}",
                "started_at": "2026-04-28T10:00:00",
                "completed_at": "2026-04-28T10:12:00",
                "status": "completed",
                "agents": {"agent:model:stateless": {"trials": []}},
            },
        )
        _write_json(
            run_dir / "trials" / "t1" / "meta.json",
            {"trial_id": "t1", "exit_code": 0},
        )
        _write_json(
            run_dir / "trials" / "t1" / "task_output.json",
            {"output": "ok", "sample": {}},
        )

    rel = str(run_dirs[0].resolve().relative_to(root.resolve()))
    encoded_run = base64.urlsafe_b64encode(rel.encode()).decode()
    response = create_app(root).test_client().get(f"/run/{encoded_run}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    trial_rel = str((run_dirs[0] / "trials" / "t1").resolve().relative_to(root.resolve()))
    encoded_trial = base64.urlsafe_b64encode(trial_rel.encode()).decode()
    assert f'href="/trial/{encoded_trial}"' in html
    assert 'href="/trial/agent%3Amodel%3Astateless/run-fixed/t1"' not in html


def test_index_page_shows_live_elapsed_and_labeled_trial_counts(tmp_path: Path) -> None:
    root = tmp_path
    now_ms = int(time.time() * 1000)
    started_at = datetime.fromtimestamp((now_ms - 10 * 60 * 1000) / 1000).isoformat(
        timespec="seconds"
    )
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-live"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-live",
            "experiment": "demo",
            "started_at": started_at,
            "completed_at": "",
            "status": "running",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        run_dir / "planned_trials.json",
        [{"trial_id": "running"}, {"trial_id": "done"}, {"trial_id": "timeout"}, {"trial_id": "failed"}],
    )
    _write_json(
        run_dir / "trials" / "running" / "proxy" / "progress.json",
        {
            "total_requests": 6,
            "success": 5,
            "errors": 1,
            "last_status": "tool",
            "last_ts_ms": now_ms,
        },
    )
    _write_json(
        run_dir / "trials" / "done" / "meta.json",
        {"trial_id": "done", "exit_code": 0, "timing": {"duration_ms": 120000}},
    )
    _write_json(run_dir / "trials" / "done" / "task_output.json", {"output": "done", "sample": {}})
    _write_json(
        run_dir / "trials" / "timeout" / "meta.json",
        {
            "trial_id": "timeout",
            "exit_code": -1,
            "termination_reason": "execution_timeout",
            "timing": {"duration_ms": 300000},
        },
    )
    _write_json(run_dir / "trials" / "timeout" / "task_output.json", {"output": "", "sample": {}})
    _write_json(
        run_dir / "trials" / "failed" / "meta.json",
        {"trial_id": "failed", "exit_code": 2, "timing": {"duration_ms": 240000}},
    )
    _write_json(run_dir / "trials" / "failed" / "task_output.json", {"output": "", "sample": {}})

    done_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-done"
    _write_json(
        done_dir / "dashboard.json",
        {
            "run_id": "run-done",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        done_dir / "trials" / "done" / "meta.json",
        {"trial_id": "done", "exit_code": 0, "timing": {"duration_ms": 120000}},
    )
    _write_json(done_dir / "trials" / "done" / "task_output.json", {"output": "done", "sample": {}})
    _write_json(
        done_dir / "trials" / "timeout" / "meta.json",
        {
            "trial_id": "timeout",
            "exit_code": -1,
            "termination_reason": "execution_timeout",
            "timing": {"duration_ms": 300000},
        },
    )
    _write_json(done_dir / "trials" / "timeout" / "task_output.json", {"output": "", "sample": {}})
    _write_json(
        done_dir / "trials" / "failed" / "meta.json",
        {"trial_id": "failed", "exit_code": 2, "timing": {"duration_ms": 240000}},
    )
    _write_json(done_dir / "trials" / "failed" / "task_output.json", {"output": "", "sample": {}})

    response = create_app(root).test_client().get("/benchmark/project")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "elapsed" in html
    assert 'data-run-live="elapsed"' in html
    assert 'data-active-run-live="elapsed"' in html
    assert 'data-run-agent-count="completed"' in html
    assert 'data-run-agent-count="stopped"' in html
    assert 'data-run-agent-count="failed"' in html
    assert "Completed" in html
    assert "Stopped" in html
    assert "Verified" not in html
    assert "Needs audit" not in html
    assert "completed + warnings + failed" not in html


def test_index_archive_filter_uses_lifecycle_language(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-done"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-done",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T10:12:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(run_dir / "trials" / "done" / "meta.json", {"trial_id": "done", "exit_code": 0})
    _write_json(run_dir / "trials" / "done" / "task_output.json", {"output": "ok", "sample": {}})

    response = create_app(root).test_client().get("/benchmark/project")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Lifecycle" in html
    assert '<option value="completed">Finished</option>' in html
    assert '<option value="interrupted">Stopped</option>' in html
    assert ">Status<" not in html
    assert '<option value="completed">Completed</option>' not in html
    assert '<option value="interrupted">Interrupted</option>' not in html


def test_run_detail_sorts_trials_in_natural_trial_order(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "status": "running",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 1, "total": 4, "trials": []}},
        },
    )
    _write_json(run_dir / "trials" / "done" / "meta.json", {"trial_id": "done", "exit_code": 0, "timing": {"duration_ms": 120000}})
    _write_json(run_dir / "trials" / "done" / "task_output.json", {"output": "done", "sample": {}})
    _write_json(
        run_dir / "trials" / "timeout" / "meta.json",
        {"trial_id": "timeout", "exit_code": -1, "termination_reason": "execution_timeout", "timing": {"duration_ms": 300000}},
    )
    _write_json(run_dir / "trials" / "timeout" / "task_output.json", {"output": "", "sample": {}})
    _write_json(
        run_dir / "trials" / "failed" / "meta.json",
        {"trial_id": "failed", "exit_code": 2, "timing": {"duration_ms": 240000}},
    )
    _write_json(run_dir / "trials" / "failed" / "task_output.json", {"output": "", "sample": {}})
    _write_json(
        run_dir / "trials" / "running" / "proxy" / "progress.json",
        {
            "total_requests": 6,
            "success": 5,
            "errors": 1,
            "last_status": "tool",
            "last_ts_ms": 1777322280300,
            "tokens_in": 1234,
            "tokens_out": 567,
            "cost_usd": 0.0579,
        },
    )
    _write_json(
        run_dir / "trials" / "running" / "meta.json",
        {"trial_id": "running", "timing": {"duration_ms": 600000}},
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "project.yml summary" in html
    assert "Run selection" in html
    assert "Stop conditions" in html
    assert "Agent / model" in html
    assert "Lifecycle" not in html
    assert "Action needed" not in html
    assert "Trial scope" not in html
    assert "<th" in html
    assert ">Trial ID<" in html
    assert ">Status<" in html
    assert ">Score<" in html
    assert ">Duration<" in html
    assert ">Exit<" in html
    assert ">Progress<" in html
    assert ">Tokens<" in html
    assert ">Output<" in html
    assert ">Why<" not in html
    assert ">Last output<" not in html
    assert "Benchmark result" not in html
    assert 'data-cell-target="detail"' in html
    assert 'data-cell-target="usage-cost"' in html
    assert "$0.0579" in html
    assert "function reorderTrialRows()" in html
    assert "reorderTrialRows();" in html
    # Rows render in natural trial (plan) order regardless of live status — the
    # "running" trial sits in its trial-order position, NOT floated to the top.
    pos_done = html.index('data-trial-id="done"')
    pos_failed = html.index('data-trial-id="failed"')
    pos_running = html.index('data-trial-id="running"')
    pos_timeout = html.index('data-trial-id="timeout"')
    assert pos_done < pos_failed < pos_running < pos_timeout


def test_run_detail_prefers_canonical_record_over_stale_legacy_trial_dir(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-record"
    project_file = _write_contract_project(run_dir.parents[2])
    spec = load_experiment_spec(project_file, sample_ids=("sample-a",))
    plan = build_experiment_plan(spec)
    writer = ExperimentArtifactWriter(run_dir)
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
    legacy_trial = run_dir / "trials" / Path(*trial_id.split("/"))
    _write_json(
        legacy_trial / "meta.json",
        {
            "trial_id": trial_id,
            "termination_reason": "completed",
            "exit_code": 0,
        },
    )
    _write_json(legacy_trial / "task_output.json", {"output": "stale"})

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-record", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert f'data-trial-id="{trial_id}"' in html
    assert 'data-status-kind="error"' in html
    assert 'data-status-label="Agent failed"' in html


def test_run_detail_lists_pending_planned_trials_with_reason_status(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-live"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-live",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "status": "running",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        run_dir / "planned_trials.json",
        [
            {"trial_id": "running", "trial_index": 0, "sample_id": "s-running"},
            {"trial_id": "pending-a", "trial_index": 1, "sample_id": "s-pending-a"},
            {"trial_id": "pending-b", "trial_index": 2, "sample_id": "s-pending-b"},
        ],
    )
    _write_json(
        run_dir / "trials" / "running" / "proxy" / "progress.json",
        {"total_requests": 6, "success": 5, "errors": 1, "last_status": "tool"},
    )
    _write_json(
        run_dir / "trials" / "done" / "meta.json",
        {"trial_id": "done", "exit_code": 0, "termination_reason": "completed"},
    )
    _write_json(run_dir / "trials" / "done" / "task_output.json", {"output": "done"})
    _write_json(
        run_dir / "trials" / "timeout" / "meta.json",
        {"trial_id": "timeout", "exit_code": -1, "termination_reason": "execution_timeout"},
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-live", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-trial-id="pending-a"' in html
    assert 'data-status-kind="pending"' in html
    assert "Not started yet" in html
    # A pending trial with no artifacts still renders a dimmed but clickable link
    # to its detail page (never an unclickable span) — see run.html trial-link.
    assert 'data-cell-target="trial-link"' in html
    assert 'class="font-mono text-xs text-slate-400 hover:text-slate-200"' in html
    assert 'title="No agent artifacts yet — opens status, meta, and events">pending-a</a>' in html
    assert "Completed normally" not in html
    assert "Stopped by limit or timeout" not in html
    assert "Benchmark result" not in html
    assert "Task finished" in html
    assert "Agent execution exceeded the configured trial timeout" in html
    assert "warning: 'bg-yellow-500/10 text-yellow-300 border-yellow-500/20'" in html
    assert "success: 'bg-green-500/10 text-green-300 border-green-500/20'" in html
    assert "running: 'bg-sky-500/15 text-sky-300 border-sky-500/30'" in html


def test_run_overview_summarizes_recorded_run_configuration(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(run_dir / "trials" / "sample-a" / "meta.json", {"trial_id": "sample-a", "exit_code": 0})
    _write_json(run_dir / "trials" / "sample-a" / "task_output.json", {"output": "ok", "sample": {"id": "sample-a"}})
    (run_dir / "project.yml").write_text(
        """
project:
  name: demo
proxy:
  request_timeout: 1800
eval:
  benchmark:
    module: ./benchmark.py
    class: PentestBench
    benchmark_root: ./datasets/postexp/
    hint_levels: [0, 1, 2]
  limit: 5
runtime:
  passk: 2
  max_trials_global: 3
  max_target_setups: 1
  timeout: 3600
  max_rounds: 100
  max_trial: 7
  live_check:
    enabled: true
    max_calls: 4
target:
  enabled: true
  run_mode: remote
  target_scope: per_agent
  startup_timeout: 1800
  compose_up_timeout: 3600
agents:
  - id: agent
    kind: qwen_code
    model: qwen3.7-max
    image: cage/qwen-code:pentestenv
    home: /home/agent/workspace
    max_concurrent: 2
    max_rounds: 80
    session_args: ["--fast", "--json"]
    shared_paths: ["/mnt/data"]
    extra_env:
      OPENAI_BASE_URL: http://proxy.local/v1
      MODEL_REASONING_EFFORT: high
""".lstrip(),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "dashboard_view.json",
        {
            "schema_version": 1,
            "title": "Saved benchmark dashboard",
            "subtitle": "run-fixed",
            "sections": [
                {
                    "kind": "summary",
                    "title": "Run summary",
                    "stats": [{"label": "Trials", "value": "1"}],
                }
            ],
        },
    )

    client = create_app(root).test_client()
    run_response = client.get("/run/agent:model:stateless/run-fixed", follow_redirects=True)
    dashboard_response = client.get("/run/agent:model:stateless/run-fixed/dashboard", follow_redirects=True)

    assert run_response.status_code == 200
    assert dashboard_response.status_code == 200
    for html in (
        run_response.get_data(as_text=True),
        dashboard_response.get_data(as_text=True),
    ):
        assert "Run selection" in html
        assert "Stop conditions" in html
        assert "Agent / model" in html
        assert "Overall result" not in html
        assert "Run setup" not in html
        assert "Agent configuration" not in html
        assert "Benchmark:" in html
        assert "PentestBench" in html
        assert "Benchmark root:" in html
        assert "./datasets/postexp/" in html
        assert "Prompt/hint levels:" in html
        assert "l0, l1, l2" in html
        assert "Max rounds:" in html
        assert "100" in html
        assert "Trial timeout:" in html
        assert "3600s (1h)" in html
        assert "Model request timeout:" in html
        assert "1800s (30m)" in html
        assert "Request timeout:" not in html
        assert "Agent kind:" in html
        assert "qwen_code" in html
        assert "Model:" in html
        assert "qwen3.7-max" in html
        assert "Image:" in html
        assert "cage/qwen-code:pentestenv" in html
        assert "Home:" not in html
        assert "Environment:" in html
        assert "OPENAI_BASE_URL=http://proxy.local/v1, MODEL_REASONING_EFFORT=high" in html
        assert "Session args:" in html
        assert "--fast, --json" in html
        assert "Agent summaries" not in html
        assert "Run limits" not in html
        assert "Pass@k attempts:" in html
        assert "2 per sample" in html
        assert "Concurrency:" in html
        assert "max_trials_global=3, max_target_setups=1, max_concurrent=2" in html
        assert "Trials to run:" in html
        assert "Target timeout:" in html
        assert "startup=1800s, compose=3600s" in html
        assert "Stop run:" in html
        assert "Ctrl-C" in html
        assert "limit 5" in html

    run_payload = _extract_json_script(run_response.get_data(as_text=True), "run-debug-bundle")
    dashboard_payload = _extract_json_script(dashboard_response.get_data(as_text=True), "dashboard-debug-bundle")
    assert run_payload["overview"]["project_summary"][0]["title"] == "Run selection"
    assert dashboard_payload["overview"]["project_summary"] == run_payload["overview"]["project_summary"]


def test_run_detail_warns_when_run_claims_active_without_active_trials(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "status": "running",
            "agents": {
                "agent:model:stateless": {
                    "completed": 0,
                    "failed": 0,
                    "total": 1,
                    "trials": [],
                }
            },
        },
    )
    _write_json(
        run_dir / "trials" / "done" / "meta.json",
        {
            "trial_id": "done",
            "exit_code": 0,
            "timing": {"duration_ms": 120000},
        },
    )
    _write_json(
        run_dir / "trials" / "done" / "task_output.json",
        {"output": "done", "sample": {}},
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Run metadata still says active" in html
    assert "No active trial process found" in html
    payload = _extract_json_script(html, "run-debug-bundle")
    assert payload["overview"]["lifecycle"] == "Recently active"
    assert payload["overview"]["activity_warnings"] == [
        "Run metadata still says active. No active trial process found."
    ]


def test_run_detail_new_trial_notice_auto_refreshes_with_pause(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "",
            "status": "running",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        run_dir / "trials" / "running" / "proxy" / "progress.json",
        {"total_requests": 6, "success": 5, "errors": 1},
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="run-new-trials-hint"' in html
    assert 'data-new-trials-auto-refresh-ms="8000"' in html
    assert 'data-new-trials-countdown' in html
    assert 'id="run-new-trials-refresh"' in html
    assert 'id="run-new-trials-pause"' in html
    assert "refreshing in" in html
    assert "Pause auto-refresh" in html
    assert "function scheduleNewTrialsReload()" in html
    assert "newTrialsReloadPaused" in html


def test_run_detail_exposes_specific_audit_filter_presets(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        run_dir / "trials" / "timeout" / "meta.json",
        {
            "trial_id": "timeout",
            "exit_code": -1,
            "termination_reason": "execution_timeout",
        },
    )
    _write_json(run_dir / "trials" / "timeout" / "task_output.json", {"output": "", "sample": {}})
    _write_json(
        run_dir / "trials" / "failed" / "meta.json",
        {"trial_id": "failed", "exit_code": 2, "termination_reason": "model_error"},
    )
    _write_json(run_dir / "trials" / "failed" / "task_output.json", {"output": "", "sample": {}})

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-audit-preset="running"' in html
    assert 'data-audit-preset="max-rounds"' in html
    assert 'data-audit-preset="timed-out"' in html
    assert 'data-summary-filter="failed"' in html
    assert 'data-audit-preset="failed"' in html
    # The "Target passed" filter chip + legend were removed as confusing.
    assert "Target passed" not in html
    assert 'data-audit-preset="target-passed"' not in html
    assert "Max rounds" in html
    assert "Timed out" in html
    assert "Failed" in html
    assert "Verified" not in html
    assert "Needs audit" not in html
    assert "Exhausted" not in html
    assert "Infra/model failures" not in html
    assert "All warnings" not in html
    assert "Live-check" not in html
    # The filter-help legend was removed as confusing clutter.
    assert 'data-audit-filter-help' not in html
    assert "Target passed means the target validator reported success." not in html
    assert "Max rounds means the agent stopped at the configured round budget." not in html
    assert "Failed includes model, target, tool, process, and nonzero-exit errors." not in html
    assert "Review audit/warning trials" not in html
    assert "running: (kind) => kind === 'running'" in html
    # The target-passed filter preset was removed with its chip.
    assert "'target-passed':" not in html
    assert "'timed-out': (kind, label)" in html
    assert "'max-rounds': (kind, label)" in html


def test_run_detail_exposes_copyable_debug_bundle(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(
        run_dir / "trials" / "done" / "meta.json",
        {"trial_id": "done", "exit_code": 0, "timing": {"duration_ms": 120000}},
    )
    _write_json(run_dir / "trials" / "done" / "task_output.json", {"output": "done", "sample": {}})

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-copy-debug-bundle="run"' in html
    assert "Copy debug bundle" in html
    payload = _extract_json_script(html, "run-debug-bundle")
    assert payload["kind"] == "run"
    assert payload["url"] == "/project/model/run-fixed"
    assert payload["run_id"] == "run-fixed"
    assert payload["experiment"] == "demo"
    assert payload["run_dir"].endswith("/project/.cage_runs/agent:model:stateless/run-fixed")
    assert payload["overview"]["lifecycle"] == "Finished"
    assert payload["overview"]["action"] == "No action needed"
    assert payload["counts"] == {
        "running": 0,
        "completed": 1,
        "live_success": 0,
        "warnings": 0,
        "failed": 0,
        "other": 0,
        "total": 1,
    }


def test_run_debug_bundle_summarizes_resource_ledger(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-resources"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-resources",
            "experiment": "demo",
            "status": "completed",
            "completed_at": "2026-04-28T11:00:00",
            "agents": {"agent:model:stateless": {"completed": 0, "failed": 0, "total": 0, "trials": []}},
        },
    )
    ledger = ResourceLedgerWriter(run_dir)
    ledger.append_resource(
        run_id="run-resources",
        resource_id="docker_container:agent-a",
        kind="docker_container",
        provider="docker",
        external_id="agent-a",
        status="started",
        cleanup_action="docker rm -f agent-a",
        timestamp="2026-04-28T10:00:00",
        trial_id="sample/pass_1",
    )
    ledger.append_resource(
        run_id="run-resources",
        resource_id="docker_network:trial-net",
        kind="docker_network",
        provider="docker",
        external_id="trial-net",
        status="released",
        cleanup_action="docker network rm trial-net",
        timestamp="2026-04-28T10:05:00",
        trial_id="sample/pass_1",
    )
    ledger.append_resource(
        run_id="run-resources",
        resource_id="target_runtime:sample/pass_1:pb-siyucms",
        kind="target_runtime",
        provider="target_server",
        external_id="cage_pb_siyucms",
        status="cleanup_failed",
        cleanup_action="target_server DELETE /launch/pb-siyucms",
        timestamp="2026-04-28T10:06:00",
        trial_id="sample/pass_1",
        cleanup_error="target_server refused teardown",
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-resources", follow_redirects=True)

    assert response.status_code == 200
    payload = _extract_json_script(response.get_data(as_text=True), "run-debug-bundle")
    assert payload["resources"]["total"] == 3
    assert payload["resources"]["active"] == 1
    assert payload["resources"]["released"] == 1
    assert payload["resources"]["cleanup_failed"] == 1
    assert payload["resources"]["by_kind"] == {
        "docker_container": 1,
        "docker_network": 1,
        "target_runtime": 1,
    }
    assert payload["resources"]["items"] == [
        {
            "resource_id": "docker_container:agent-a",
            "kind": "docker_container",
            "provider": "docker",
            "external_id": "agent-a",
            "status": "started",
            "trial_id": "sample/pass_1",
            "cleanup_error": "",
        },
        {
            "resource_id": "docker_network:trial-net",
            "kind": "docker_network",
            "provider": "docker",
            "external_id": "trial-net",
            "status": "released",
            "trial_id": "sample/pass_1",
            "cleanup_error": "",
        },
        {
            "resource_id": "target_runtime:sample/pass_1:pb-siyucms",
            "kind": "target_runtime",
            "provider": "target_server",
            "external_id": "cage_pb_siyucms",
            "status": "cleanup_failed",
            "trial_id": "sample/pass_1",
            "cleanup_error": "target_server refused teardown",
        },
    ]


def test_trial_page_is_diagnosis_first_and_uses_current_attempt_label(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "exit_code": 0,
            "termination_reason": "completed",
            "timing": {"duration_ms": 120000},
        },
    )
    _write_json(trial_dir / "prompt.txt", {"prompt": "ignored"})
    (trial_dir / "prompt.txt").write_text("Long prompt text", encoding="utf-8")
    _write_json(trial_dir / "task_output.json", {"output": "final answer", "sample": {}})

    encoded = base64.urlsafe_b64encode(
        str(trial_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()
    response = create_app(root).test_client().get(f"/trial/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Trial diagnosis" in html
    assert "Verdict" in html
    assert "Current attempt" in html
    assert "● LIVE" not in html
    assert "Readable evidence" in html
    assert "Tool timeline" in html
    assert "Raw JSON" in html
    assert 'data-context-mode="readable"' in html
    assert 'data-context-mode="raw"' in html
    assert 'data-trial-summary-endpoint="/api/trial/' in html
    assert 'data-trial-summary-field="status"' in html
    assert 'data-trial-summary-field="duration"' in html
    assert 'data-trial-summary-field="tokens"' in html
    # Header is slimmed: the redundant context-nav rail and verbose
    # "Token context"/"Trial updated" lines were removed (see Task 3). The
    # diagnosis section still comes before the (now default-expanded) Prompt.
    assert 'data-trial-context-rail' not in html
    assert "Token context:" not in html
    assert html.index("Trial diagnosis") < html.index("Prompt")


def test_trial_overview_explains_why_trial_ended_before_raw_sections(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "exit_code": -1,
            "termination_reason": "execution_timeout",
            "termination_detail": "Agent execution exceeded 300s",
            "termination_source": "orchestrator",
            "timing": {"duration_ms": 300000},
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "partial", "sample": {}})
    _write_json(
        trial_dir / "scores" / "score.json",
        {"score": {"value": 0.25, "explanation": "one host compromised"}},
    )

    encoded = base64.urlsafe_b64encode(
        str(trial_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()
    response = create_app(root).test_client().get(f"/trial/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    why_index = html.index("Why this trial ended")
    evidence_index = html.index(">More termination evidence<")
    assert "Trial diagnosis" in html[why_index:]
    assert "Timed out" in html[why_index:evidence_index]
    assert "Agent execution exceeded 300s" in html[why_index:evidence_index]
    assert "orchestrator" in html[why_index:evidence_index]
    assert "score 0.25" in html[why_index:evidence_index]
    assert why_index < evidence_index < html.index("Prompt")


def test_trial_page_splits_completed_execution_from_partial_benchmark_outcome(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "exit_code": 0,
            "termination_reason": "completed",
            "timing": {"duration_ms": 120000},
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "partial", "sample": {}})
    _write_json(
        trial_dir / "scores" / "score.json",
        {"score": {"value": 0.25, "explanation": "one host compromised"}},
    )

    encoded = base64.urlsafe_b64encode(
        str(trial_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()
    response = create_app(root).test_client().get(f"/trial/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Execution:" in html
    assert "Benchmark outcome:" in html
    assert "Partial result" in html
    assert "score 0.25" in html
    assert 'data-trial-outcome-kind="partial"' in html
    status_start = html.index('data-trial-summary-field="status"')
    status_end = html.index("Completed", status_start)
    assert "text-green-300" not in html[status_start:status_end]


def test_trial_page_exposes_copyable_debug_bundle(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "exit_code": -1,
            "termination_reason": "execution_timeout",
            "termination_detail": "Agent execution exceeded 300s",
            "timing": {"duration_ms": 300000},
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "final answer", "sample": {}})
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "total_requests": 7,
            "success": 6,
            "errors": 1,
            "tokens_in": 1234,
            "tokens_out": 567,
            "tokens_reasoning": 89,
        },
    )

    encoded = base64.urlsafe_b64encode(
        str(trial_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()
    response = create_app(root).test_client().get(f"/trial/{encoded}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-copy-debug-bundle="trial"' in html
    assert "Copy debug bundle" in html
    payload = _extract_json_script(html, "trial-debug-bundle")
    assert payload["kind"] == "trial"
    # The debug bundle now carries the canonical, shareable trial URL.
    assert payload["url"] == "/project/model/run-fixed/range1-L0"
    assert payload["run_id"] == "run-fixed"
    assert payload["trial_id"] == "range1-L0"
    assert payload["run_dir"].endswith("/project/.cage_runs/agent:model:stateless/run-fixed")
    assert payload["trial_dir"].endswith("/project/.cage_runs/agent:model:stateless/run-fixed/trials/range1-L0")
    assert payload["status"]["label"] == "Timed out"
    assert payload["status"]["kind"] == "warning"
    assert payload["termination"]["reason"] == "execution_timeout"
    assert payload["termination"]["detail"] == "Agent execution exceeded 300s"
    assert payload["usage"] == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "reasoning_tokens": 89,
        "num_requests": 6,
    }


def test_trial_summary_api_returns_live_status_usage_and_termination(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1-L0"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "range1-L0",
            "exit_code": -1,
            "termination_reason": "execution_timeout",
            "termination_detail": "Agent execution exceeded 300s",
            "timing": {"duration_ms": 300000},
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "", "sample": {}})
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {
            "trial_id": "range1-L0",
            "total_requests": 7,
            "success": 6,
            "errors": 1,
            "tokens_in": 1234,
            "tokens_out": 567,
            "tokens_reasoning": 89,
        },
    )

    encoded = base64.urlsafe_b64encode(
        str(trial_dir.resolve().relative_to(root.resolve())).encode()
    ).decode()
    response = create_app(root).test_client().get(f"/api/trial/{encoded}/summary")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["present"] is True
    assert payload["changed"] is True
    assert payload["max_signature_ms"] > 0
    summary = payload["summary"]
    assert summary["status_label"] == "Timed out"
    assert summary["status_kind"] == "warning"
    assert summary["duration_ms"] == 300000
    assert summary["exit_code"] == -1
    assert summary["usage"] == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "reasoning_tokens": 89,
        "num_requests": 6,
    }
    assert summary["termination"]["reason"] == "execution_timeout"
    assert summary["termination"]["detail"] == "Agent execution exceeded 300s"

    unchanged = create_app(root).test_client().get(
        f"/api/trial/{encoded}/summary?since={payload['max_signature_ms']}"
    )
    assert unchanged.status_code == 200
    unchanged_payload = unchanged.get_json()
    assert unchanged_payload["changed"] is False
    assert unchanged_payload["summary"] is None


def test_dashboard_warns_when_saved_view_is_stale(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 2, "failed": 0, "total": 2, "trials": []}},
        },
    )
    _write_json(
        run_dir / "dashboard_view.json",
        {
            "schema_version": 1,
            "title": "Saved benchmark dashboard",
            "subtitle": "run-fixed",
            "sections": [
                {
                    "kind": "summary",
                    "title": "Run summary",
                    "stats": [
                        {
                            "label": "Trials",
                            "value": "1",
                            "hint": "Saved dashboard row count.",
                        }
                    ],
                },
                {
                    "kind": "table",
                    "title": "Per-trial",
                    "columns": [
                        {"key": "trial", "label": "Trial"},
                        {"key": "status", "label": "Status"},
                    ],
                    "rows": [
                        {"trial": "t1", "status": "completed"},
                        {"trial": "t2", "status": "timed out"},
                        {"trial": "t3", "status": "failed"},
                    ],
                }
            ],
        },
    )
    for trial_id in ("t1", "t2"):
        _write_json(run_dir / "trials" / trial_id / "meta.json", {"trial_id": trial_id, "exit_code": 0})
        _write_json(run_dir / "trials" / trial_id / "task_output.json", {"output": "ok", "sample": {}})

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed/dashboard", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Benchmark metrics view" in html
    assert "Outcome: 1/3 completed, 1 stopped, 1 failed" in html
    assert "Diagnosis shortcuts" in html
    assert 'data-dashboard-stat-hint="Trials"' in html
    assert re.search(r'data-dashboard-stat-hint="Trials"[^>]*>\s*Saved dashboard row count\.\s*<', html)
    assert "Stopped rows 1" in html
    assert "Failed rows 1" in html
    assert "Completed rows 1" in html
    assert "need review" not in html
    assert "Review stops" not in html
    assert "Audit stops" not in html
    assert "Verified rows" not in html
    assert "Needs audit" not in html
    assert "Verified" not in html
    assert "project.yml summary" in html
    assert "Run selection" in html
    assert "Stop conditions" in html
    assert "Agent / model" in html
    assert "Trial scope" not in html
    assert "Config:" not in html
    assert "Termination conditions:" not in html
    assert "Output:" not in html
    assert "Dashboard data is stale" in html
    assert "Saved dashboard: 1 trial" in html
    assert "Current artifacts: 2 trials" in html
    assert "Dashboard updated" in html
    assert 'data-copy-debug-bundle="dashboard"' in html
    assert "Copy debug bundle" in html
    assert 'data-dashboard-live-endpoint="/api/run/' in html
    assert "data-dashboard-last-updated" in html
    assert "data-dashboard-status-filter" in html
    assert '<option value="stopped">Stopped</option>' in html
    assert '<option value="completed">Completed</option>' in html
    assert '<option value="review">Needs audit</option>' not in html
    assert '<option value="completed">Verified</option>' not in html
    assert '<option value="review">Needs review</option>' not in html
    assert "data-dashboard-failures-first" in html
    assert 'data-dashboard-sort="status"' in html
    assert 'data-dashboard-sort="trial"' in html
    assert 'data-dashboard-row-status="completed"' in html
    assert 'data-dashboard-row-status="stopped"' in html
    assert 'data-dashboard-row-status="failed"' in html
    assert 'data-dashboard-row-tone="completed"' in html
    assert 'data-dashboard-row-tone="stopped"' in html
    assert 'data-dashboard-row-tone="failed"' in html
    assert "border-l-red-500/60" in html
    assert "border-l-amber-400/60" in html
    assert 'data-dashboard-status-label="Completed"' in html
    assert 'data-dashboard-status-label="Stopped"' in html
    assert 'data-dashboard-status-label="Failed"' in html
    assert 'data-dashboard-status-raw="timed out"' in html
    assert 'data-dashboard-status-kind="stopped"' in html
    payload = _extract_json_script(html, "dashboard-debug-bundle")
    assert payload["kind"] == "dashboard"
    assert payload["url"] == "/project/model/run-fixed/dashboard"
    assert payload["run_id"] == "run-fixed"
    assert payload["run_dir"].endswith("/project/.cage_runs/agent:model:stateless/run-fixed")
    assert payload["dashboard_mode"] == "final"
    assert payload["freshness"]["stale"] is True
    assert payload["freshness"]["saved_count"] == 1
    assert payload["freshness"]["current_count"] == 2


def test_dashboard_renders_typed_cells_for_sorting_and_severity(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"completed": 1, "failed": 0, "total": 1, "trials": []}},
        },
    )
    _write_json(
        run_dir / "dashboard_view.json",
        {
            "schema_version": 1,
            "title": "Typed dashboard",
            "sections": [
                {
                    "kind": "table",
                    "title": "Per-trial",
                    "columns": [
                        {"key": "trial", "label": "Trial"},
                        {"key": "score", "label": "Score", "align": "right", "unit": "ratio", "hint": "Higher is better"},
                        {"key": "status", "label": "Status"},
                    ],
                    "rows": [
                        {
                            "trial": "t1",
                            "score": {
                                "display_value": "0.25",
                                "raw_value": 0.25,
                                "sort_key": 0.25,
                                "unit": "ratio",
                                "severity": "warning",
                                "hint": "Partial benchmark result",
                            },
                            "status": {
                                "display_value": "timed out",
                                "sort_key": "timeout",
                                "severity": "warning",
                            },
                        }
                    ],
                }
            ],
        },
    )
    _write_json(run_dir / "trials" / "t1" / "meta.json", {"trial_id": "t1", "exit_code": 0})
    _write_json(run_dir / "trials" / "t1" / "task_output.json", {"output": "ok", "sample": {}})

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed/dashboard", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'data-dashboard-sort-value="0.25"' in html
    assert 'data-dashboard-cell-severity="warning"' in html
    assert 'data-dashboard-cell-unit="ratio"' in html
    assert re.search(r">\s*Higher is better\s*<", html)
    assert 'data-dashboard-cell-hint="score"' in html
    assert re.search(r'data-dashboard-cell-hint="score"[^>]*>\s*Partial benchmark result\s*<', html)
    assert 'title="Partial benchmark result"' in html
    assert "0.25" in html
    assert 'data-dashboard-status-kind="stopped"' in html
    assert 'href="/project/model/run-fixed/t1"' in html


def test_run_detail_labels_live_success_as_target_check_audit(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    trial_dir = run_dir / "trials" / "audit-me"
    _write_json(
        trial_dir / "meta.json",
        {
            "trial_id": "audit-me",
            "exit_code": 0,
            "termination_reason": "live_success",
            "live_success": True,
            "timing": {"duration_ms": 120000},
        },
    )
    _write_json(trial_dir / "task_output.json", {"output": "candidate answer", "sample": {}})

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Target passed" in html
    assert "Target validator reported success." in html


def test_dashboard_api_returns_freshness_on_semantic_project_route(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    _write_json(
        run_dir / "dashboard_view.json",
        {
            "schema_version": 1,
            "title": "Saved benchmark dashboard",
            "sections": [
                {
                    "kind": "summary",
                    "title": "Run summary",
                    "stats": [{"label": "Trials", "value": "1"}],
                }
            ],
        },
    )
    for trial_id in ("t1", "t2"):
        _write_json(run_dir / "trials" / trial_id / "meta.json", {"trial_id": trial_id, "exit_code": 0})
        _write_json(run_dir / "trials" / trial_id / "task_output.json", {"output": "ok", "sample": {}})

    response = create_app(root).test_client().get(
        "/api/projects/project/runs/run-fixed/dashboard_view"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["present"] is True
    assert payload["changed"] is True
    assert payload["max_signature_ms"] > 0
    assert payload["freshness"]["signature_ms"] == payload["max_signature_ms"]
    assert payload["freshness"]["stale"] is True
    assert payload["freshness"]["saved_count"] == 1
    assert payload["freshness"]["current_count"] == 2

    unchanged = create_app(root).test_client().get(
        f"/api/projects/project/runs/run-fixed/dashboard_view?since={payload['max_signature_ms']}"
    )
    assert unchanged.status_code == 200
    unchanged_payload = unchanged.get_json()
    assert unchanged_payload["present"] is True
    assert unchanged_payload["changed"] is False
    assert unchanged_payload["view"] is None
    assert unchanged_payload["max_signature_ms"] == payload["max_signature_ms"]


def test_dashboard_no_view_offers_next_actions_and_artifact_path(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )

    response = create_app(root).test_client().get("/run/agent:model:stateless/run-fixed/dashboard", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "No dashboard view yet" in html
    assert "Open generic run overview" in html
    assert 'href="/project/model/run-fixed"' in html
    assert "Expected artifact path" in html
    assert str(run_dir / "dashboard_view.json") in html
    assert "Why this benchmark may not provide a dashboard" in html
    assert "benchmarks can opt out" in html


def test_dashboard_page_reports_malformed_dashboard_view(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    _write_json(
        run_dir / "dashboard.json",
        {
            "run_id": "run-fixed",
            "experiment": "demo",
            "started_at": "2026-04-28T10:00:00",
            "completed_at": "2026-04-28T11:00:00",
            "status": "completed",
            "agents": {"agent:model:stateless": {"trials": []}},
        },
    )
    (run_dir / "dashboard_view.json").write_text("{not json", encoding="utf-8")

    client = create_app(root).test_client()
    response = client.get(
        "/run/agent:model:stateless/run-fixed/dashboard"
    , follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Dashboard view file is malformed" in html
    assert "dashboard_view.json" in html
    assert "not valid JSON" in html
    assert "No dashboard view yet" not in html

    api_response = client.get("/api/projects/project/runs/run-fixed/dashboard_view")
    assert api_response.status_code == 200
    payload = api_response.get_json()
    assert payload["present"] is False
    assert payload["error"]["title"] == "Dashboard view file is malformed"
