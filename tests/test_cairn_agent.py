"""Cairn custom-agent orchestrator: pure-helper unit tests.

The Cairn agent's launch entrypoint (``cage/agents/custom/cairn/
cairn_cage_entry.py``) keeps all Cage<->Cairn glue in pure functions —
dispatch.yaml rendering, project body, status/answer extraction — so they can be
tested with no Docker and no network. Loaded by path (it ships in the agent
source dir, not on sys.path).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ENTRY = (
    Path(__file__).resolve().parents[1]
    / "cage" / "agents" / "custom" / "cairn" / "cairn_cage_entry.py"
)


@pytest.fixture(scope="module")
def entry():
    if not _ENTRY.is_file():
        pytest.skip(f"cairn agent entrypoint not present: {_ENTRY}")
    spec = importlib.util.spec_from_file_location("cairn_cage_entry", _ENTRY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_build_project_body_uses_prompt_as_origin(entry):
    body = entry.build_project_body(
        title="cage-trial", origin="pwn http://10.0.0.5 and drop /tmp/u.txt", goal="g",
    )
    assert body["origin"] == "pwn http://10.0.0.5 and drop /tmp/u.txt"
    assert body["goal"] == "g"
    assert body["bootstrap_enabled"] is True
    assert body["hints"] == []


def test_anthropic_base_url_strips_trailing_v1(entry):
    # openai-protocol model => CustomAgent appended /v1 => strip it so the
    # claude worker (which appends /v1/messages itself) doesn't hit /v1/v1/messages.
    assert entry.anthropic_base_url("http://localhost:33145/v1") == "http://localhost:33145"
    assert entry.anthropic_base_url("http://localhost:33145/v1/") == "http://localhost:33145"
    # anthropic-protocol model => already bare => unchanged.
    assert entry.anthropic_base_url("http://localhost:33145") == "http://localhost:33145"
    assert entry.anthropic_base_url("http://localhost:33145/") == "http://localhost:33145"


def test_render_dispatch_config_single_worker_to_proxy(entry):
    cfg = entry.render_dispatch_config(
        server_url="http://127.0.0.1:8000",
        worker_image="cage/cairn-worker:latest",
        base_url="http://127.0.0.1:9000",
        api_key="sk-proxy",
        model="claude-opus",
        workers=3,
    )
    # one project, N workers, all concurrency knobs agree
    assert cfg["runtime"]["max_running_projects"] == 1
    assert cfg["runtime"]["max_workers"] == 3
    assert cfg["runtime"]["max_project_workers"] == 3
    assert cfg["runtime"]["worker_healthcheck"] == "disabled"
    # worker container shares the trial netns under the inner daemon
    assert cfg["container"]["network_mode"] == "host"
    assert cfg["container"]["image"] == "cage/cairn-worker:latest"
    # the single claudecode worker points every call at the Cage proxy
    assert len(cfg["workers"]) == 1
    w = cfg["workers"][0]
    assert w["type"] == "claudecode"
    assert w["max_running"] == 3
    assert w["task_types"] == ["bootstrap", "reason", "explore"]
    assert w["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert w["env"]["ANTHROPIC_MODEL"] == "claude-opus"
    assert w["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"


def test_render_dispatch_config_clamps_workers(entry):
    cfg = entry.render_dispatch_config(
        server_url="s", worker_image="i", base_url="b", api_key="k",
        model="m", workers=0,
    )
    assert cfg["runtime"]["max_workers"] == 1
    assert cfg["workers"][0]["max_running"] == 1


def test_project_status_reads_nested_field(entry):
    assert entry.project_status({"project": {"status": "completed"}}) == "completed"
    assert entry.project_status({"project": {}}) == ""
    assert entry.project_status({}) == ""


def test_goal_facts_extracts_answer_set(entry):
    detail = {
        "facts": [
            {"id": "origin", "description": "target"},
            {"id": "goal", "description": "win"},
            {"id": "f001", "description": "rooted host-A, dropped /root/r.txt"},
            {"id": "f002", "description": "noise"},
        ],
        "intents": [
            {"from": ["f001"], "to": "goal", "description": "done"},
            {"from": ["origin"], "to": "f002", "description": "scan"},
        ],
    }
    assert entry.goal_facts(detail) == ["rooted host-A, dropped /root/r.txt"]


def test_summarize_includes_status_and_answer(entry):
    detail = {
        "project": {"status": "completed"},
        "facts": [
            {"id": "origin", "description": "t"},
            {"id": "goal", "description": "w"},
            {"id": "f001", "description": "rooted host-A"},
        ],
        "intents": [{"from": ["f001"], "to": "goal", "description": "d"}],
    }
    out = entry.summarize(detail)
    assert "status: completed" in out
    assert "rooted host-A" in out
