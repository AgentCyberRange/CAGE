"""Tests for compact trajectory event export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "export_trajectory_events.py"


def load_export_module():
    spec = importlib.util.spec_from_file_location("export_trajectory_events", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def sample_proxy_entries() -> list[dict]:
    return [
        {
            "request_id": "req-0001",
            "status": "success",
            "ts_ms": 1700000000000,
            "openai_request": {
                "model": "glm-5.1",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "start"}]}],
            },
            "upstream_response": {
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "plan"},
                    {"type": "text", "text": "I will inspect it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "id"},
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 3},
            },
        },
        {
            "request_id": "req-0002",
            "status": "success",
            "ts_ms": 1700000001000,
            "openai_request": {
                "model": "glm-5.1",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "start"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "plan"},
                            {"type": "text", "text": "I will inspect it."},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Bash",
                                "input": {"command": "id"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "uid=1000(agent)",
                            }
                        ],
                    },
                ],
            },
            "upstream_response": {
                "type": "message",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 11, "output_tokens": 1},
            },
        },
    ]


def test_build_step_records_preserves_thinking_tool_and_observation(tmp_path: Path) -> None:
    module = load_export_module()
    proxy_jsonl = tmp_path / "proxy.jsonl"
    write_jsonl(proxy_jsonl, sample_proxy_entries())

    records = module.build_step_records(proxy_jsonl)

    assert records[0]["blocks"] == [
        {"type": "thinking", "content": "plan"},
        {"type": "response", "content": "I will inspect it."},
        {
            "type": "tool_call",
            "name": "Bash",
            "input": {"command": "id"},
            "observation": "uid=1000(agent)",
        },
    ]
    assert records[1]["blocks"] == [{"type": "response", "content": "done"}]


def test_build_claude_event_records_splits_tool_observation(tmp_path: Path) -> None:
    module = load_export_module()
    trial_dir = tmp_path / "trial"
    proxy_jsonl = trial_dir / "proxy" / "proxy.jsonl"
    write_jsonl(proxy_jsonl, sample_proxy_entries())
    (trial_dir / "meta.json").write_text(
        json.dumps({"status": "completed", "termination_reason": "completed"}),
        encoding="utf-8",
    )
    (trial_dir / "task_output.json").write_text(
        json.dumps({"output": "final answer"}),
        encoding="utf-8",
    )

    records = module.build_claude_event_records(
        proxy_jsonl,
        session_id="s1",
        model="glm-5.1",
    )

    assert [record["type"] for record in records] == [
        "system",
        "assistant",
        "user",
        "assistant",
        "result",
    ]
    assert records[1]["message"]["content"][0] == {"type": "thinking", "thinking": "plan"}
    assert records[1]["message"]["content"][2]["type"] == "tool_use"
    tool_id = records[1]["message"]["content"][2]["id"]
    assert records[2]["message"]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": "uid=1000(agent)",
        }
    ]
    assert records[-1]["result"] == "final answer"
