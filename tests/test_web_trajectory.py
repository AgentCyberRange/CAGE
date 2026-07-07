"""Tests for web trajectory parsing."""

from __future__ import annotations

import json
from pathlib import Path

from cage.web.data import load_step_context, parse_trajectory


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def test_parse_trajectory_caches_full_read_across_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Harness-structure reconstruction needs the whole stream (a subagent's
    # parent edge can point anywhere), so the first parse reads the full log;
    # subsequent page fetches of the same (unchanged) log are served from the
    # (path, mtime, size)-keyed cache without re-opening the file.
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": f"req-{idx:04d}",
                "status": "success",
                "ts_ms": idx,
                "openai_request": {
                    "model": "gpt-test",
                    "messages": [{"role": "user", "content": f"step {idx}"}],
                },
                "upstream_response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": f"ok {idx}"}}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            }
            for idx in range(120)
        ],
    )
    from cage.web.data import _TRAJECTORY_SOURCE_CACHE

    _TRAJECTORY_SOURCE_CACHE.clear()

    # First page: cold cache, full read is allowed.
    first = parse_trajectory(proxy_jsonl, offset=0, limit=2)
    assert [step["index"] for step in first["steps"]] == [0, 1]
    assert first["total_steps"] == 120
    assert first["has_more"] is True

    # Second page of the unchanged log must not re-open the file.
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path == proxy_jsonl:
            raise AssertionError("paginated trajectory re-read a cached log")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    second = parse_trajectory(proxy_jsonl, offset=2, limit=2)
    assert [step["index"] for step in second["steps"]] == [2, 3]
    assert second["total_steps"] == 120


def test_parse_trajectory_includes_operator_summary(tmp_path: Path) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-1",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {"messages": [{"role": "user", "content": "start"}]},
                "upstream_response": {
                    "choices": [{"message": {"content": "checking files"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
            },
            {
                "request_id": "req-2",
                "status": "success",
                "ts_ms": 2,
                "openai_request": {"messages": [{"role": "user", "content": "run shell"}]},
                "upstream_response": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {"name": "bash", "arguments": "{}"},
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                },
            },
            {
                "request_id": "req-3",
                "status": "success",
                "ts_ms": 3,
                "error": "tool stderr was non-empty",
                "openai_request": {"messages": [{"role": "user", "content": "run tools"}]},
                "upstream_response": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call-2",
                                        "function": {"name": "python", "arguments": "{}"},
                                    },
                                    {
                                        "id": "call-3",
                                        "function": {"name": "bash", "arguments": "{}"},
                                    },
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 5},
                },
            },
        ],
    )

    data = parse_trajectory(proxy_jsonl, offset=0, limit=3)

    assert data["summary"] == {
        "requests": 3,
        "loaded_steps": 3,
        "total_steps": 3,
        "total_steps_known": True,
        "total_tokens": {"in": 60, "out": 12, "reasoning": 0},
        "top_tools": [{"name": "bash", "count": 2}, {"name": "python", "count": 1}],
        "tools": [{"name": "bash", "count": 2}, {"name": "python", "count": 1}],
        "tool_calls": 3,
        "last_action": "tool: bash",
        "error_count": 1,
    }


def test_parse_trajectory_infers_anthropic_stream_blocks_from_next_request(
    tmp_path: Path,
) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-0001",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {
                    "model": "glm-5.1",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "start"}]}],
                },
                "upstream_response": {
                    "type": "message",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
            {
                "request_id": "req-0002",
                "status": "success",
                "ts_ms": 2,
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
                    "content": [],
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ],
    )

    data = parse_trajectory(proxy_jsonl, offset=0, limit=1)

    assert data["total_steps"] == 2
    assert data["steps"][0]["blocks"] == [
        {"type": "thinking", "content": "plan"},
        {"type": "text", "content": "I will inspect it."},
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "id"},
            "result": "uid=1000(agent)",
        },
    ]


def test_parse_trajectory_extracts_openai_responses_output_blocks(
    tmp_path: Path,
) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-0001",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "start"}],
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "uid=1000(agent)",
                        },
                    ],
                },
                "upstream_response": {
                    "object": "response",
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "plan"}],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "I will inspect it."}
                            ],
                        },
                        {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": "{\"cmd\":\"id\"}",
                            "call_id": "call_1",
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            }
        ],
    )

    data = parse_trajectory(proxy_jsonl, offset=0, limit=1)

    assert data["steps"][0]["blocks"] == [
        {"type": "thinking", "content": "plan"},
        {"type": "text", "content": "I will inspect it."},
        {
            "type": "tool_use",
            "name": "exec_command",
            "input": {"cmd": "id"},
            "result": "uid=1000(agent)",
        },
    ]


def test_parse_trajectory_infers_openai_responses_blocks_from_next_request(
    tmp_path: Path,
) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-0001",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "start"}],
                        }
                    ],
                },
                "upstream_response": {
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
            {
                "request_id": "req-0002",
                "status": "success",
                "ts_ms": 2,
                "openai_request": {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "start"}],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "I will inspect it."}
                            ],
                            "phase": "commentary",
                        },
                        {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": "{\"cmd\":\"id\"}",
                            "call_id": "call_1",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "uid=1000(agent)",
                        },
                    ],
                },
                "upstream_response": {
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ],
    )

    data = parse_trajectory(proxy_jsonl, offset=0, limit=1)

    assert data["steps"][0]["blocks"] == [
        {"type": "text", "content": "I will inspect it."},
        {
            "type": "tool_use",
            "name": "exec_command",
            "input": {"cmd": "id"},
            "result": "uid=1000(agent)",
        },
    ]


def test_load_step_context_infers_openai_responses_response_content(
    tmp_path: Path,
) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-0001",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "start"}],
                        }
                    ],
                },
                "upstream_response": {
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
            {
                "request_id": "req-0002",
                "status": "success",
                "ts_ms": 2,
                "openai_request": {
                    "model": "gpt-5.5",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "start"}],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        },
                        {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": "{\"cmd\":\"id\"}",
                            "call_id": "call_1",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "uid=1000(agent)",
                        },
                    ],
                },
                "upstream_response": {
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ],
    )

    context = load_step_context(proxy_jsonl, 0)

    assert context["response"]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "name": "exec_command", "input": {"cmd": "id"}},
    ]
    assert context["summary"]["response_tool_count"] == 1


def test_parse_trajectory_does_not_attach_unrelated_future_tool_result(
    tmp_path: Path,
) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-0001",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {
                    "model": "qwen",
                    "messages": [
                        {"role": "user", "content": "start"},
                    ],
                },
                "upstream_response": {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Checking size",
                                "tool_calls": [
                                    {
                                        "id": "call_bash",
                                        "type": "function",
                                        "function": {
                                            "name": "Bash",
                                            "arguments": json.dumps(
                                                {
                                                    "command": "wc -c files/stage-1.asm",
                                                    "description": "Count bytes",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
            },
            {
                "request_id": "req-0002",
                "status": "success",
                "ts_ms": 2,
                "openai_request": {
                    "model": "qwen",
                    "messages": [
                        {"role": "user", "content": "start"},
                        {
                            "role": "assistant",
                            "content": "Reading makefile",
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": json.dumps(
                                            {"file_path": "/home/agent/workspace/files/Makefile"}
                                        ),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_read",
                            "content": "tacOS.bin: stage-1.asm stage-2.bin",
                        },
                    ],
                },
                "upstream_response": {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Done",
                                "tool_calls": [],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 1},
                },
            },
        ],
    )

    data = parse_trajectory(proxy_jsonl, offset=0, limit=1)

    assert data["steps"][0]["blocks"] == [
        {"type": "text", "content": "Checking size"},
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "wc -c files/stage-1.asm", "description": "Count bytes"},
            "result": "",
        },
    ]


def test_parse_trajectory_extracts_zai_upstream_content_blocks(
    tmp_path: Path,
) -> None:
    proxy_jsonl = tmp_path / "proxy.jsonl"
    _write_jsonl(
        proxy_jsonl,
        [
            {
                "request_id": "req-0001",
                "status": "success",
                "ts_ms": 1,
                "openai_request": {
                    "model": "glm-5.1",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "start"}]}],
                },
                "upstream_response": {
                    "type": "message",
                    "content": [
                        {"type": "thinking", "thinking": "plan"},
                        {"type": "text", "text": "hello"},
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
                "ts_ms": 2,
                "openai_request": {
                    "model": "glm-5.1",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "start"}]},
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "plan"},
                                {"type": "text", "text": "hello"},
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
                    "content": [],
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ],
    )

    data = parse_trajectory(proxy_jsonl, offset=0, limit=1)

    assert data["steps"][0]["blocks"] == [
        {"type": "thinking", "content": "plan"},
        {"type": "text", "content": "hello"},
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "id"},
            "result": "uid=1000(agent)",
        },
    ]
