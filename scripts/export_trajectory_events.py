#!/usr/bin/env python3
"""Export compact trajectory events from a Cage ``proxy.jsonl``.

The web inspector renders trajectories from ``proxy/proxy.jsonl`` via
``cage.web.data.parse_trajectory``. This script exposes that same parser as a
file-oriented exporter so review bundles can carry a compact event stream
instead of the very repetitive raw proxy capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cage.web.data import parse_trajectory  # noqa: E402


def resolve_proxy_jsonl(path: Path) -> Path:
    """Resolve *path* as either a proxy.jsonl file or a Cage trial directory."""
    if path.is_file():
        return path
    candidate = path / "proxy" / "proxy.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no proxy.jsonl found at {path} or {candidate}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def trial_dir_for_proxy(proxy_jsonl: Path) -> Path | None:
    if proxy_jsonl.name == "proxy.jsonl" and proxy_jsonl.parent.name == "proxy":
        return proxy_jsonl.parent.parent
    return None


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def iso_timestamp(ts_ms: Any) -> str:
    try:
        ts = int(ts_ms or 0) / 1000.0
    except (TypeError, ValueError):
        ts = 0.0
    if ts <= 0:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def infer_model(steps: list[dict[str, Any]], fallback: str = "") -> str:
    for step in steps:
        blocks = step.get("blocks")
        if isinstance(blocks, list):
            # ``parse_trajectory`` keeps model only in the raw step context API,
            # not the paginated step payload. Keep this hook for future schema
            # expansion and fall back to CLI/meta data for now.
            break
    return fallback


def normalize_step_block(block: dict[str, Any]) -> dict[str, Any]:
    """Map web-inspector block shape to the compact review-export shape."""
    btype = block.get("type")
    if btype == "thinking":
        return {"type": "thinking", "content": str(block.get("content") or "")}
    if btype == "text":
        return {"type": "response", "content": str(block.get("content") or "")}
    if btype == "tool_use":
        return {
            "type": "tool_call",
            "name": block.get("name", "unknown"),
            "input": block.get("input", {}),
            "observation": str(block.get("result") or ""),
        }
    return {"type": str(btype or "unknown"), "value": block}


def build_step_records(proxy_jsonl: Path) -> list[dict[str, Any]]:
    """Return one compact JSON record per model step."""
    parsed = parse_trajectory(proxy_jsonl, offset=0, limit=999999)
    records: list[dict[str, Any]] = []
    for step in parsed.get("steps", []):
        records.append(
            {
                "type": "step",
                "step": step.get("index"),
                "ts_ms": step.get("ts_ms", 0),
                "timestamp": iso_timestamp(step.get("ts_ms", 0)),
                "tokens": step.get("tokens", {}),
                "cumulative": step.get("cumulative", {}),
                "context": {
                    "message_count": step.get("context_msg_count", 0),
                    "roles": step.get("context_summary", {}),
                },
                "blocks": [
                    normalize_step_block(block)
                    for block in step.get("blocks", [])
                    if isinstance(block, dict)
                ],
            }
        )
    return records


def content_block_to_claude(
    block: dict[str, Any],
    *,
    step_index: int,
    block_index: int,
    tool_ids: dict[int, str],
) -> dict[str, Any] | None:
    """Convert a compact block to a Claude Code stream-json content block."""
    btype = block.get("type")
    if btype == "thinking":
        text = str(block.get("content") or "")
        return {"type": "thinking", "thinking": text} if text else None
    if btype == "response":
        text = str(block.get("content") or "")
        return {"type": "text", "text": text} if text else None
    if btype == "tool_call":
        block_fingerprint = json.dumps(block, sort_keys=True)
        tool_id = stable_id("toolu", f"{step_index}:{block_index}:{block_fingerprint}")
        tool_ids[block_index] = tool_id
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": block.get("name", "unknown"),
            "input": block.get("input", {}),
        }
    return None


def build_claude_event_records(
    proxy_jsonl: Path,
    *,
    session_id: str = "",
    model: str = "",
) -> list[dict[str, Any]]:
    """Build a synthetic Claude Code stream-json-like event list.

    The shape intentionally mirrors Claude Code ``--output-format stream-json``
    enough for downstream review tooling: system, assistant, user tool_result,
    and result records. It is synthetic because Cage records proxy traffic, not
    the original Claude CLI stdout stream.
    """
    step_records = build_step_records(proxy_jsonl)
    trial_dir = trial_dir_for_proxy(proxy_jsonl)
    meta = load_json(trial_dir / "meta.json") if trial_dir else {}
    task_output = load_json(trial_dir / "task_output.json") if trial_dir else {}
    progress = load_json(proxy_jsonl.parent / "progress.json")
    if not session_id:
        session_id = stable_id("cage_session", str(proxy_jsonl.resolve()))

    tool_names: list[str] = []
    for name in (progress.get("tools_used") or {}).keys() if isinstance(progress, dict) else []:
        tool_names.append(str(name))
    if not tool_names:
        seen: set[str] = set()
        for record in step_records:
            for block in record.get("blocks", []):
                if isinstance(block, dict) and block.get("type") == "tool_call":
                    name = str(block.get("name") or "")
                    if name and name not in seen:
                        seen.add(name)
                        tool_names.append(name)

    events: list[dict[str, Any]] = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": session_id,
            "uuid": stable_id("evt", f"{session_id}:system"),
            "model": model,
            "tools": tool_names,
            "permissionMode": "",
            "apiKeySource": "",
            "claude_code_version": "",
            "synthetic": True,
            "source": "cage.proxy.parse_trajectory",
        }
    ]

    total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    for record in step_records:
        step_index = int(record.get("step") or 0)
        timestamp = record.get("timestamp") or ""
        tokens = record.get("tokens") if isinstance(record.get("tokens"), dict) else {}
        total_usage["input_tokens"] += int(tokens.get("in", 0) or 0)
        total_usage["output_tokens"] += int(tokens.get("out", 0) or 0)
        total_usage["reasoning_tokens"] += int(tokens.get("reasoning", 0) or 0)

        assistant_content: list[dict[str, Any]] = []
        tool_result_content: list[dict[str, Any]] = []
        tool_ids: dict[int, str] = {}
        blocks = [b for b in record.get("blocks", []) if isinstance(b, dict)]
        for block_index, block in enumerate(blocks):
            content_block = content_block_to_claude(
                block,
                step_index=step_index,
                block_index=block_index,
                tool_ids=tool_ids,
            )
            if content_block:
                assistant_content.append(content_block)

        for block_index, block in enumerate(blocks):
            if block.get("type") != "tool_call":
                continue
            observation = str(block.get("observation") or "")
            if not observation:
                continue
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_ids.get(
                        block_index,
                        stable_id("toolu", f"{step_index}:{block_index}"),
                    ),
                    "content": observation,
                }
            )

        if assistant_content:
            event: dict[str, Any] = {
                "type": "assistant",
                "message": {
                    "id": stable_id("msg", f"{session_id}:assistant:{step_index}"),
                    "type": "message",
                    "role": "assistant",
                    "content": assistant_content,
                    "model": model,
                    "usage": {
                        "input_tokens": int(tokens.get("in", 0) or 0),
                        "output_tokens": int(tokens.get("out", 0) or 0),
                        "reasoning_tokens": int(tokens.get("reasoning", 0) or 0),
                    },
                },
                "parent_tool_use_id": None,
                "session_id": session_id,
                "uuid": stable_id("evt", f"{session_id}:assistant:{step_index}"),
                "synthetic": True,
            }
            if timestamp:
                event["timestamp"] = timestamp
            events.append(event)

        if tool_result_content:
            event = {
                "type": "user",
                "message": {"role": "user", "content": tool_result_content},
                "parent_tool_use_id": None,
                "session_id": session_id,
                "uuid": stable_id("evt", f"{session_id}:user:{step_index}"),
                "synthetic": True,
            }
            if timestamp:
                event["timestamp"] = timestamp
            events.append(event)

    output = str(task_output.get("output") or "")
    terminal_reason = str(meta.get("termination_reason") or "")
    status = str(meta.get("status") or "")
    timing = meta.get("timing") if isinstance(meta.get("timing"), dict) else {}
    events.append(
        {
            "type": "result",
            "subtype": terminal_reason or status or "unknown",
            "is_error": bool(status and status not in {"completed"}),
            "duration_ms": int(timing.get("duration_ms") or 0),
            "num_turns": len(step_records),
            "result": output,
            "stop_reason": terminal_reason,
            "session_id": session_id,
            "total_cost_usd": None,
            "usage": total_usage,
            "terminal_reason": terminal_reason,
            "uuid": stable_id("evt", f"{session_id}:result"),
            "synthetic": True,
        }
    )
    return events


def write_jsonl(records: Iterable[dict[str, Any]], output: Path | None) -> None:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    text = "\n".join(lines) + ("\n" if lines else "")
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export compact review trajectory events from Cage proxy.jsonl.",
    )
    parser.add_argument("path", type=Path, help="Trial directory or proxy/proxy.jsonl file")
    parser.add_argument(
        "--format",
        choices=("steps-jsonl", "claude-events-jsonl"),
        default="steps-jsonl",
        help="Output event schema",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output path; defaults to stdout")
    parser.add_argument("--session-id", default="", help="Session id for claude-events-jsonl")
    parser.add_argument("--model", default="", help="Model label for claude-events-jsonl")
    args = parser.parse_args(argv)

    proxy_jsonl = resolve_proxy_jsonl(args.path)
    if args.format == "steps-jsonl":
        records = build_step_records(proxy_jsonl)
    else:
        records = build_claude_event_records(
            proxy_jsonl,
            session_id=args.session_id,
            model=args.model,
        )
    write_jsonl(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
