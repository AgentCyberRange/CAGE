"""Cairn graph gallery: Cairn's own frontend, multi-project, in the inspector.

When a Cairn agent trial ends, its container (and the live graph it served) is
gone, but the entrypoint leaves a ``workspace/cairn_graph.yaml`` snapshot. The
inspector serves Cairn's *own* frontend and backs it with a synthesized
multi-project API: every trial of a run is an isolated project (id = trial
slug), its graph synthesized from that snapshot (or proxied live). These tests
lock the synthesis + the read-only gallery contract, with no Docker / no server.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cage.web import app as webapp
from cage.web.app import (
    _CAIRN_STATIC_DIR,
    _cairn_gallery_projects,
    _cairn_load_snapshot,
    _cairn_slug,
    _cairn_trial_label,
    _cairn_unslug,
    create_app,
)

# Export format the entrypoint writes: project + facts (with ids) + intents
# (WITHOUT ids -- the snapshot drops them, the inspector re-mints them).
_SNAPSHOT_YAML = """
project:
  title: cage-trial
  origin: pwn the range and drop the markers
  goal: compromise the in-scope hosts
  bootstrap_enabled: true
facts:
  - id: origin
    description: target brief
  - id: goal
    description: win condition
  - id: f001
    description: recon only, nothing rooted
intents:
  - from: [origin]
    to: f001
    description: bootstrap
    creator: dispatcher.bootstrap
    worker: claudecode_worker
    created_at: '2026-06-29 10:33:32'
    concluded_at: '2026-06-29 10:43:57'
  - from: [f001]
    to: null
    description: exploit AJ-Report on the entry host
    creator: claudecode_worker
    worker: claudecode_worker
    created_at: '2026-06-29 10:44:00'
    concluded_at: null
  - from: [f001]
    to: null
    description: pivot inward once a host is owned
    creator: claudecode_worker
    worker: null
    created_at: '2026-06-29 10:44:05'
    concluded_at: null
""".lstrip()


def _make_trial(run: Path, trial_rel: str, *, snapshot: bool) -> None:
    trial = run / "trials" / trial_rel
    trial.mkdir(parents=True)
    (trial / "meta.json").write_text("{}", encoding="utf-8")  # trial-dir marker
    if snapshot:
        (trial / "workspace").mkdir()
        (trial / "workspace" / "cairn_graph.yaml").write_text(_SNAPSHOT_YAML, encoding="utf-8")


def _run_with_trials(root: Path) -> Path:
    """A cairn run with one settled trial (snapshot) + one pending (no snapshot)."""
    run = root / "bench" / ".cage_runs" / "cairn:model:stateless" / "run"
    _make_trial(run, "pb-range-1-l0/pass_1", snapshot=True)
    _make_trial(run, "pb-range-1-l1/pass_1", snapshot=False)
    return run


def _gallery_client(root: Path, run: Path):
    """A client whose run-resolver returns ``run`` (skips project.yml resolution)."""
    app = create_app(root)
    app.view_functions  # ensure routes registered
    import cage.web.app as mod
    mod._find_benchmark_model_run = lambda b, m, r: run  # type: ignore[assignment]
    return app.test_client()


# --- pure synthesis ---------------------------------------------------------

def test_slug_roundtrips_trial_path() -> None:
    assert _cairn_slug("pb-range-1-l0/pass_1") == "pb-range-1-l0~pass_1"
    assert _cairn_unslug("pb-range-1-l0~pass_1") == "pb-range-1-l0/pass_1"


def test_trial_label_is_path_under_trials(tmp_path: Path) -> None:
    trial = tmp_path / "run" / "trials" / "pb-postexp-range-1-l0" / "pass_1"
    assert _cairn_trial_label(trial) == "pb-postexp-range-1-l0/pass_1"
    assert _cairn_trial_label(tmp_path / "flat" / "trial-x") == "trial-x"


def test_load_snapshot_synthesizes_frontend_shape(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run" / "trials" / "sample" / "pass_1"
    (trial_dir / "workspace").mkdir(parents=True)
    (trial_dir / "workspace" / "cairn_graph.yaml").write_text(_SNAPSHOT_YAML, encoding="utf-8")

    snap = _cairn_load_snapshot(trial_dir, "sample~pass_1", title="sample/pass_1")

    assert snap is not None
    # Settled snapshot -> read-only: status 'completed' disables mutation controls.
    assert snap["project"]["id"] == "sample~pass_1"
    assert snap["project"]["status"] == "completed"
    assert snap["project"]["title"] == "sample/pass_1"
    assert snap["project"]["created_at"] == "2026-06-29 10:33:32"
    assert [f["id"] for f in snap["facts"]] == ["origin", "goal", "f001"]
    # Intents get deterministic ids minted in file order (export omits them).
    assert [i["id"] for i in snap["intents"]] == ["i001", "i002", "i003"]
    assert snap["intents"][0]["from"] == ["origin"] and snap["intents"][0]["to"] == "f001"
    assert sum(1 for i in snap["intents"] if i["to"]) == 1      # 1 concluded edge
    assert sum(1 for i in snap["intents"] if i["to"] is None) == 2  # 2 open intents


def test_load_snapshot_default_id_and_absent(tmp_path: Path) -> None:
    trial_dir = tmp_path / "run" / "trials" / "s" / "pass_1"
    trial_dir.mkdir(parents=True)
    assert _cairn_load_snapshot(trial_dir) is None  # no snapshot
    (trial_dir / "workspace").mkdir()
    (trial_dir / "workspace" / "cairn_graph.yaml").write_text(_SNAPSHOT_YAML, encoding="utf-8")
    snap = _cairn_load_snapshot(trial_dir)
    assert snap["project"]["id"] == "proj_001"  # container-side default
    assert snap["project"]["title"] == "s/pass_1"  # falls back to trial label


def test_gallery_projects_lists_every_trial(tmp_path: Path) -> None:
    run = _run_with_trials(tmp_path)
    items = _cairn_gallery_projects(run)
    by_id = {it["id"]: it for it in items}
    # The settled trial is a completed project with real counts...
    settled = by_id["pb-range-1-l0~pass_1"]
    assert settled["status"] == "completed"
    assert settled["fact_count"] == 3 and settled["intent_count"] == 3
    assert settled["title"] == "pb-range-1-l0/pass_1"
    # ...the snapshot-less trial shows as a pending (active) project, 0 counts.
    pending = by_id["pb-range-1-l1~pass_1"]
    assert pending["status"] == "active" and pending["fact_count"] == 0


# --- gallery route (Cairn's own frontend + synthesized data) ----------------

def test_gallery_serves_cairns_own_frontend(tmp_path: Path) -> None:
    run = _run_with_trials(tmp_path)
    client = _gallery_client(tmp_path, run)
    resp = client.get("/cairn-gallery/bench/model/run/")
    assert resp.status_code == 200
    html = resp.data.decode()
    # It is Cairn's OWN rich frontend (not a custom viewer): its real assets +
    # absolute paths rewritten under the per-run prefix + a fetch shim.
    assert "/cairn-gallery/bench/model/run/static/vendor/cytoscape.min.js" in html
    assert "window.fetch=function" in html


def test_gallery_project_list_and_detail_and_export(tmp_path: Path) -> None:
    run = _run_with_trials(tmp_path)
    client = _gallery_client(tmp_path, run)
    base = "/cairn-gallery/bench/model/run"

    listing = json.loads(client.get(f"{base}/projects").data)
    ids = {p["id"] for p in listing}
    assert "pb-range-1-l0~pass_1" in ids and "pb-range-1-l1~pass_1" in ids

    detail = json.loads(client.get(f"{base}/projects/pb-range-1-l0~pass_1").data)
    assert detail["project"]["id"] == "pb-range-1-l0~pass_1"  # relabeled to slug
    assert len(detail["facts"]) == 3
    assert [i["id"] for i in detail["intents"]] == ["i001", "i002", "i003"]

    export = client.get(f"{base}/projects/pb-range-1-l0~pass_1/export?format=yaml")
    assert export.status_code == 200 and b"title: cage-trial" in export.data

    assert json.loads(client.get(f"{base}/settings").data)["intent_timeout"] == 5


def test_live_proxy_snapshot_reads_as_active(tmp_path: Path) -> None:
    # A trial that wrote its graph to the bind-mounted proxy dir (live, survives
    # interruption) but has no settled workspace snapshot -> served as "active"
    # so the frontend keeps polling and the graph grows mid-run.
    run = tmp_path / "bench" / ".cage_runs" / "cairn:model:stateless" / "run"
    trial = run / "trials" / "pb-range-1-l2" / "pass_1"
    (trial / "proxy").mkdir(parents=True)
    (trial / "meta.json").write_text("{}", encoding="utf-8")
    (trial / "proxy" / "cairn_graph.yaml").write_text(_SNAPSHOT_YAML, encoding="utf-8")

    snap = _cairn_load_snapshot(trial, "pb-range-1-l2~pass_1", title="x", status="active")
    assert snap["project"]["status"] == "active" and len(snap["facts"]) == 3

    client = _gallery_client(tmp_path, run)
    detail = json.loads(
        client.get("/cairn-gallery/bench/model/run/projects/pb-range-1-l2~pass_1").data)
    assert detail["project"]["status"] == "active"  # live, not "completed"
    assert len(detail["facts"]) == 3
    # And it lists as active too.
    listing = json.loads(client.get("/cairn-gallery/bench/model/run/projects").data)
    assert any(p["id"] == "pb-range-1-l2~pass_1" and p["status"] == "active" for p in listing)


def test_gallery_graphless_trial_returns_empty_json_not_html(tmp_path: Path) -> None:
    # The smoke_test_cairn_2 bug: a trial dir exists (preflight failed -> no
    # snapshot, no container). The data API must return an EMPTY graph as JSON,
    # never an HTML 404 the frontend chokes on ("Unexpected token '<'").
    run = _run_with_trials(tmp_path)
    client = _gallery_client(tmp_path, run)
    resp = client.get("/cairn-gallery/bench/model/run/projects/pb-range-1-l1~pass_1")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("application/json")
    body = json.loads(resp.data)
    assert body["facts"] == [] and body["intents"] == []
    assert body["project"]["id"] == "pb-range-1-l1~pass_1"


def test_gallery_unknown_trial_and_writes_return_json_errors(tmp_path: Path) -> None:
    run = _run_with_trials(tmp_path)
    client = _gallery_client(tmp_path, run)
    # Unknown slug -> JSON 404 (not HTML), so r.json() in the frontend works.
    missing = client.get("/cairn-gallery/bench/model/run/projects/nope~pass_9")
    assert missing.status_code == 404
    assert missing.headers["Content-Type"].startswith("application/json")
    assert "detail" in json.loads(missing.data)
    # Writes -> JSON 405.
    wr = client.post("/cairn-gallery/bench/model/run/projects")
    assert wr.status_code == 405
    assert wr.headers["Content-Type"].startswith("application/json")


def test_gallery_static_served_from_disk(tmp_path: Path) -> None:
    if not (_CAIRN_STATIC_DIR / "vendor" / "cytoscape.min.js").is_file():
        import pytest
        pytest.skip("vendored Cairn frontend not present")
    run = _run_with_trials(tmp_path)
    client = _gallery_client(tmp_path, run)
    resp = client.get("/cairn-gallery/bench/model/run/static/vendor/cytoscape.min.js")
    assert resp.status_code == 200 and int(resp.headers["Content-Length"]) > 100_000


def test_gallery_static_rejects_traversal(tmp_path: Path) -> None:
    run = _run_with_trials(tmp_path)
    client = _gallery_client(tmp_path, run)
    resp = client.get("/cairn-gallery/bench/model/run/static/../../../etc/passwd")
    assert resp.status_code in (400, 404)
