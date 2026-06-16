"""Tests for trial artifact file browsing in the web inspector."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

from cage.artifacts.writer import ExperimentArtifactWriter
from cage.experiment.model import build_experiment_plan, load_experiment_spec
from cage.web.app import create_app
from cage.web.data import build_trial_file_tree


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_contract_project(project_dir: Path) -> Path:
    """Write the smallest project file needed for contract artifact tests."""
    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: trial-files-demo
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


def _write_canonical_run_with_indexed_proxy_log(
    root: Path,
    *,
    proxy_ref: str = "canonical_proxy/proxy.jsonl",
) -> tuple[Path, str]:
    """Create a canonical run whose raw proxy log is only artifact-indexed."""
    run = root / "project" / ".cage_runs" / "agent:model" / "run-record"
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
    proxy_path = run / proxy_ref
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_path.write_text('{"status": "success"}\n', encoding="utf-8")
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


def _encode(root: Path, path: Path) -> str:
    rel = str(path.resolve().relative_to(root.resolve()))
    return base64.urlsafe_b64encode(rel.encode()).decode()


def _codex_stream(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def test_build_trial_file_tree_includes_nested_files_and_sizes(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run-fixed" / "trials" / "trial-1"
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1"})
    (trial_dir / "prompt.txt").write_text("hello", encoding="utf-8")
    (trial_dir / "proxy").mkdir()
    (trial_dir / "proxy" / "proxy.jsonl").write_bytes(b"x" * 1536)

    entries = build_trial_file_tree(trial_dir)

    by_path = {entry.relative_path: entry for entry in entries}
    assert by_path["meta.json"].is_dir is False
    assert by_path["meta.json"].size_bytes > 0
    assert by_path["prompt.txt"].size_label == "5 B"
    assert by_path["proxy"].is_dir is True
    assert by_path["proxy"].depth == 0
    assert by_path["proxy/proxy.jsonl"].is_dir is False
    assert by_path["proxy/proxy.jsonl"].depth == 1
    assert by_path["proxy/proxy.jsonl"].size_label == "1.5 KB"


def test_trial_file_download_serves_file_as_attachment(tmp_path: Path) -> None:
    root = tmp_path
    trial_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed" / "trials" / "trial-1"
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1"})
    artifact = trial_dir / "proxy" / "proxy.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ok": true}\n', encoding="utf-8")

    app = create_app(root)
    response = app.test_client().get(
        f"/trial/{_encode(root, trial_dir)}/download/{_encode(root, artifact)}"
    )

    assert response.status_code == 200
    assert response.get_data(as_text=True) == '{"ok": true}\n'
    assert "attachment" in response.headers["Content-Disposition"]
    assert "proxy.jsonl" in response.headers["Content-Disposition"]


def test_trial_file_download_rejects_file_outside_trial_dir(tmp_path: Path) -> None:
    root = tmp_path
    trials_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed" / "trials"
    trial_dir = trials_dir / "trial-1"
    other_trial_dir = trials_dir / "trial-2"
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1"})
    other_file = other_trial_dir / "secret.txt"
    other_file.parent.mkdir(parents=True)
    other_file.write_text("nope", encoding="utf-8")

    app = create_app(root)
    response = app.test_client().get(
        f"/trial/{_encode(root, trial_dir)}/download/{_encode(root, other_file)}"
    )

    assert response.status_code == 403


def test_trial_page_renders_file_tree_panel(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    trial_dir = run_dir / "trials" / "trial-1"
    _write_json(run_dir / "dashboard.json", {"agents": {}})
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1", "exit_code": 0})
    _write_json(trial_dir / "task_output.json", {"output": "done", "sample": {}})
    (trial_dir / "proxy").mkdir()
    (trial_dir / "proxy" / "proxy.jsonl").write_bytes(b"x" * 1536)

    app = create_app(root)
    response = app.test_client().get(f"/trial/{_encode(root, trial_dir)}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Files" in html
    assert '<details\n  id="trial-files-panel"' in html
    assert 'class="bg-surface rounded-xl border border-slate-700/50 mb-4"' in html
    assert "bg-white rounded-lg border border-slate-200 mb-4 text-slate-900" not in html
    encoded_trial = _encode(root, trial_dir)
    assert f'data-files-endpoint="/api/trial/{encoded_trial}/files"' in html
    assert "Load file tree on demand." in html
    assert "renderTrialDiagnostics" in html
    assert "data-file-diagnostic" in html
    assert "workspace" in html
    assert "score files" in html
    assert "screenshots" in html
    assert "downloads" in html
    assert "proxy.jsonl" not in html
    assert '<summary class="px-6 py-4 text-sm font-semibold text-slate-300 hover:text-white">Prompt</summary>' in html
    # Prompt + Final Output are expanded by default (key info should not be hidden).
    assert '<details id="trial-raw" class="bg-surface rounded-xl border border-slate-700/50 mb-4" open>' in html
    assert '<summary class="px-6 py-4 text-sm font-semibold text-slate-300 hover:text-white">Final Output</summary>' in html


def test_trial_files_api_returns_tree_on_demand(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    trial_dir = run_dir / "trials" / "trial-1"
    _write_json(run_dir / "dashboard.json", {"agents": {}})
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1", "exit_code": 0})
    _write_json(trial_dir / "task_output.json", {"output": "done", "sample": {}})
    (trial_dir / "proxy").mkdir()
    _write_json(
        trial_dir / "proxy" / "progress.json",
        {"total_requests": 4, "errors": 1},
    )
    (trial_dir / "proxy" / "proxy.jsonl").write_bytes(b"x" * 1536)
    _write_json(trial_dir / "target_inspect.json", {"status": "ok"})
    (trial_dir / "target_server.log").write_text("target booted\n", encoding="utf-8")

    app = create_app(root)
    response = app.test_client().get(f"/api/trial/{_encode(root, trial_dir)}/files")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] >= 2
    names = {entry["relative_path"] for entry in payload["entries"]}
    assert "proxy" in names
    assert "proxy/proxy.jsonl" in names
    expected_total = sum(
        entry["size_bytes"]
        for entry in payload["entries"]
        if not entry["is_dir"]
    )
    assert payload["total_size_bytes"] == expected_total
    assert payload["total_size_label"]
    assert payload["large_file_count"] == 0
    proxy_entry = next(entry for entry in payload["entries"] if entry["relative_path"] == "proxy/proxy.jsonl")
    assert proxy_entry["size_label"] == "1.5 KB"
    encoded_file = _encode(root, trial_dir / "proxy" / "proxy.jsonl")
    assert proxy_entry["download_url"] == f"/trial/{_encode(root, trial_dir)}/download/{encoded_file}"
    diagnostics = {item["key"]: item for item in payload["diagnostics"]}
    assert diagnostics["progress"]["relative_path"] == "proxy/progress.json"
    assert diagnostics["metadata"]["relative_path"] == "meta.json"
    assert diagnostics["final_output"]["relative_path"] == "task_output.json"
    assert diagnostics["target_inspect"]["relative_path"] == "target_inspect.json"
    assert diagnostics["target_logs"]["relative_path"] == "target_server.log"
    assert diagnostics["proxy_trace"]["relative_path"] == "proxy/proxy.jsonl"
    assert diagnostics["proxy_trace"]["size_label"] == "1.5 KB"
    assert diagnostics["proxy_trace"]["is_large"] is False
    assert diagnostics["proxy_trace"]["download_url"] == proxy_entry["download_url"]


def test_trial_files_api_lists_and_downloads_indexed_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(root)
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))
    encoded_trial = _encode(root, virtual_trial_dir)
    client = create_app(root).test_client()

    response = client.get(f"/api/trial/{encoded_trial}/files")

    assert response.status_code == 200
    payload = response.get_json()
    by_path = {entry["relative_path"]: entry for entry in payload["entries"]}
    proxy_entry = by_path["canonical_proxy/proxy.jsonl"]
    assert proxy_entry["is_dir"] is False
    assert proxy_entry["size_label"] == "22 B"
    diagnostics = {item["key"]: item for item in payload["diagnostics"]}
    assert diagnostics["proxy_trace"]["relative_path"] == "canonical_proxy/proxy.jsonl"

    download = client.get(proxy_entry["download_url"])

    assert download.status_code == 200
    assert download.get_data(as_text=True) == '{"status": "success"}\n'


def test_trial_files_api_diagnostics_use_indexed_artifact_kind(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(
        root,
        proxy_ref="model_events/raw.ndjson",
    )
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))

    response = create_app(root).test_client().get(
        f"/api/trial/{_encode(root, virtual_trial_dir)}/files"
    )

    assert response.status_code == 200
    payload = response.get_json()
    by_path = {entry["relative_path"]: entry for entry in payload["entries"]}
    assert by_path["model_events/raw.ndjson"]["artifact_kind"] == "proxy_log"
    diagnostics = {item["key"]: item for item in payload["diagnostics"]}
    assert diagnostics["proxy_trace"]["relative_path"] == "model_events/raw.ndjson"


def test_trial_files_api_lists_and_zips_indexed_directory_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run, trial_id = _write_canonical_run_with_indexed_proxy_log(root)
    snapshot_dir = run / "canonical_state" / "state_pre"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "home_agent.txt").write_text("captured state\n", encoding="utf-8")
    ExperimentArtifactWriter(run).mark_trial_artifact(
        trial_id,
        artifact_id=f"trial.{trial_id}.state_snapshot_pre",
        path="canonical_state/state_pre",
        kind="state_snapshot_pre",
        schema_version="state_snapshot.directory.v1",
        producer="runtime state",
        replayability="audit",
        content_type="inode/directory",
    )
    virtual_trial_dir = run / "trials" / Path(*trial_id.split("/"))
    encoded_trial = _encode(root, virtual_trial_dir)
    client = create_app(root).test_client()

    response = client.get(f"/api/trial/{encoded_trial}/files")

    assert response.status_code == 200
    payload = response.get_json()
    by_path = {entry["relative_path"]: entry for entry in payload["entries"]}
    snapshot_entry = by_path["canonical_state/state_pre"]
    assert snapshot_entry["is_dir"] is True
    assert snapshot_entry["size_bytes"] == 0
    assert snapshot_entry["size_label"] == ""
    assert "download_url" not in snapshot_entry
    assert snapshot_entry["download_zip_url"]

    download = client.get(snapshot_entry["download_zip_url"])

    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.data)) as archive:
        assert archive.read("state_pre/home_agent.txt").decode() == "captured state\n"


def test_trial_files_api_flags_large_proxy_trace(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    trial_dir = run_dir / "trials" / "trial-1"
    _write_json(run_dir / "dashboard.json", {"agents": {}})
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1", "exit_code": 0})
    trace = trial_dir / "proxy" / "proxy.jsonl"
    trace.parent.mkdir(parents=True)
    with trace.open("wb") as handle:
        handle.truncate(11 * 1024 * 1024)

    response = create_app(root).test_client().get(f"/api/trial/{_encode(root, trial_dir)}/files")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["large_file_count"] == 1
    diagnostics = {item["key"]: item for item in payload["diagnostics"]}
    assert diagnostics["proxy_trace"]["relative_path"] == "proxy/proxy.jsonl"
    assert diagnostics["proxy_trace"]["is_large"] is True
    assert "Large file" in diagnostics["proxy_trace"]["note"]


def test_trial_files_panel_summarizes_total_size_and_large_files(tmp_path: Path) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "agent:model" / "run-fixed"
    trial_dir = run_dir / "trials" / "trial-1"
    _write_json(run_dir / "dashboard.json", {"agents": {}})
    _write_json(trial_dir / "meta.json", {"trial_id": "trial-1", "exit_code": 0})
    _write_json(trial_dir / "task_output.json", {"output": "done", "sample": {}})

    response = create_app(root).test_client().get(f"/trial/{_encode(root, trial_dir)}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "total_size_label" in html
    assert "large_file_count" in html
    assert "large file" in html


def test_trial_page_renders_parsed_codex_final_output_instead_of_raw_stream(
    tmp_path: Path,
) -> None:
    root = tmp_path
    run_dir = root / "project" / ".cage_runs" / "codex:gpt-5.5:stateless" / "run-fixed"
    trial_dir = run_dir / "trials" / "range1"
    stream = _codex_stream(
        {"type": "thread.started", "thread_id": "thread-1"},
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "Initial foothold."},
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
            "error": {"message": "unexpected status 503 Service Unavailable"},
        },
    )
    _write_json(
        run_dir / "dashboard.json",
        {
            "agents": {
                "codex:gpt-5.5:stateless": {
                    "trials": [
                        {
                            "trial_id": "range1",
                            "exit_code": 1,
                            "termination_reason": "model_timeout",
                            "termination_detail": stream,
                        }
                    ]
                }
            }
        },
    )
    _write_json(trial_dir / "meta.json", {"trial_id": "range1", "exit_code": 1})
    _write_json(trial_dir / "task_output.json", {"output": stream, "sample": {}})

    app = create_app(root)
    response = app.test_client().get(f"/trial/{_encode(root, trial_dir)}", follow_redirects=True)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "ActiveMQ RCE is confirmed as root, and I retrieved flag3." in html
    assert ">Final Output<" in html
