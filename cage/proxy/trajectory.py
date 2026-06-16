"""Canonical agent-trajectory parsing from ``proxy.jsonl``.

``proxy.jsonl`` is the single source of truth for an agent trial's trajectory.
This module owns the one parser that turns a raw proxy record into structured
content blocks (thinking / text / tool_use, with tool results paired in), across
every wire format the in-container proxy records (Anthropic messages, OpenAI
chat completions, and the OpenAI Responses API).

Consumers share this parser instead of each re-deriving blocks:
  - the web inspector renders these blocks into the trajectory view;
  - :func:`generate_traj` serializes them into a human-readable ``.traj`` file.

Block shape (presentation-free):
  - ``{"type": "thinking", "content": str}``
  - ``{"type": "text", "content": str}``
  - ``{"type": "tool_use", "name": str, "input": Any, "result": str}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cage.proxy.conversations import reconstruct_forest
from cage.proxy.usage import extract_entry_usage


def _truncate_tool_result(result: str) -> str:
    if len(result) > 5000:
        return result[:5000] + f"\n... ({len(result)} chars)"
    return result


def _parse_tool_arguments(value: Any) -> Any:
    if isinstance(value, str):
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    if value is None:
        return {}
    return value


def _responses_reasoning_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    summary = item.get("summary")
    if isinstance(summary, list):
        for block in summary:
            if isinstance(block, dict):
                text = block.get("text") or block.get("summary_text")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))

    content = item.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("summary_text")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
    elif isinstance(content, str) and content:
        parts.append(content)

    text = item.get("text")
    if isinstance(text, str) and text:
        parts.append(text)
    return "\n".join(parts)


def _blocks_from_anthropic_content(
    content: list[Any],
    tool_results: dict[str, str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            text = str(block.get("thinking") or block.get("text") or "")
            if text:
                blocks.append({"type": "thinking", "content": text})
        elif btype in {"text", "input_text", "output_text"}:
            text = str(block.get("text") or "")
            if text:
                blocks.append({"type": "text", "content": text})
        elif btype == "tool_use":
            tool_id = str(block.get("id") or "")
            blocks.append({
                "type": "tool_use",
                "name": block.get("name", "unknown"),
                "input": block.get("input", {}),
                "result": _truncate_tool_result(tool_results.get(tool_id, "")),
            })
    return blocks


def _blocks_from_openai_chat_message(
    message: dict[str, Any],
    tool_results: dict[str, str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        blocks.append({"type": "thinking", "content": str(reasoning)})

    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "content": content})
    elif isinstance(content, list):
        blocks.extend(_blocks_from_anthropic_content(content, tool_results))

    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
        name = func.get("name") or tc.get("name") or "unknown"
        args = _parse_tool_arguments(func.get("arguments", tc.get("arguments", {})))
        tool_id = str(tc.get("id") or tc.get("call_id") or "")
        blocks.append({
            "type": "tool_use",
            "name": name,
            "input": args,
            "result": _truncate_tool_result(tool_results.get(tool_id, "")),
        })
    return blocks


def _blocks_from_responses_items(
    items: list[Any],
    tool_results: dict[str, str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            text = _responses_reasoning_text(item)
            if text:
                blocks.append({"type": "thinking", "content": text})
        elif item_type in (None, "message"):
            if item.get("role") not in (None, "assistant"):
                continue
            content = item.get("content")
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "content": content})
            elif isinstance(content, list):
                blocks.extend(_blocks_from_anthropic_content(content, tool_results))
        elif item_type in {"function_call", "tool_call", "custom_tool_call"}:
            tool_id = str(item.get("call_id") or item.get("id") or "")
            blocks.append({
                "type": "tool_use",
                "name": item.get("name", item_type),
                "input": _parse_tool_arguments(item.get("arguments", item.get("input", {}))),
                "result": _truncate_tool_result(tool_results.get(tool_id, "")),
            })
        elif item_type == "computer_call":
            tool_id = str(item.get("call_id") or item.get("id") or "")
            blocks.append({
                "type": "tool_use",
                "name": "computer",
                "input": item.get("action", {}),
                "result": _truncate_tool_result(tool_results.get(tool_id, "")),
            })
    return blocks


def _extract_response_blocks_from_body(
    response: dict[str, Any],
    tool_results: dict[str, str],
) -> list[dict[str, Any]]:
    content = response.get("content")
    if isinstance(content, list) and content:
        return _blocks_from_anthropic_content(content, tool_results)

    output = response.get("output")
    if isinstance(output, list) and output:
        return _blocks_from_responses_items(output, tool_results)

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message") if isinstance(first.get("message"), dict) else {}
        return _blocks_from_openai_chat_message(msg, tool_results)

    return []


def _extract_blocks(
    entry: dict[str, Any],
    tool_results: dict[str, str],
) -> list[dict[str, Any]]:
    """Extract structured content blocks for one ``proxy.jsonl`` record.

    Prefers the Anthropic-shaped response (what the agent actually received) and
    grafts on any ``thinking`` block the upstream response carries that the
    Anthropic view dropped.
    """

    anthropic_resp = entry.get("anthropic_response") or {}
    upstream = entry.get("upstream_response") or {}

    blocks: list[dict[str, Any]] = []
    if isinstance(anthropic_resp, dict):
        blocks = _extract_response_blocks_from_body(anthropic_resp, tool_results)
    upstream_blocks = (
        _extract_response_blocks_from_body(upstream, tool_results)
        if isinstance(upstream, dict)
        else []
    )
    if blocks:
        if upstream_blocks and not any(block.get("type") == "thinking" for block in blocks):
            thinking = [block for block in upstream_blocks if block.get("type") == "thinking"]
            return thinking + blocks
        return blocks
    return upstream_blocks


def generate_traj(proxy_jsonl_path: Path, output_path: Path) -> None:
    """Parse a ``proxy.jsonl`` file and write a human-readable ``.traj`` file.

    A derived, browsable text projection of the same blocks the web inspector
    renders. ``proxy.jsonl`` stays the source of truth.
    """

    if not proxy_jsonl_path or not proxy_jsonl_path.exists():
        return

    entries: list[dict[str, Any]] = []
    for line in proxy_jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("status") != "success":
            continue
        entries.append(entry)

    if not entries:
        return

    # Pair tool results with the tool_use that triggered them, collected from
    # the tool messages carried in subsequent requests.
    tool_results_by_id: dict[str, str] = {}
    for entry in entries:
        req = entry.get("openai_request") or entry.get("anthropic_request") or {}
        for msg in req.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                if tid:
                    tool_results_by_id[tid] = str(msg.get("content", ""))

    # Reconstruct the harness conversation structure so the projection shows the
    # delegation tree (root vs subagents), compaction boundaries, and demotes the
    # harness's own background calls instead of flattening everything into one
    # misleading linear stream.
    forest = reconstruct_forest(entries)
    conv_by_id = {c.id: c for c in forest.conversations}
    compaction_call_indices = {i for c in forest.conversations for i in c.compaction_calls}
    compaction_resume_indices = {i for c in forest.conversations for i in c.compaction_at}
    spawns_by_index: dict[int, list[str]] = {}
    returns_by_index: dict[int, list[str]] = {}
    for conv in forest.conversations:
        if conv.spawned_by and conv.spawned_by.get("parent_index") is not None:
            spawns_by_index.setdefault(int(conv.spawned_by["parent_index"]), []).append(conv.id)
        if conv.returns_at is not None:
            returns_by_index.setdefault(int(conv.returns_at), []).append(conv.id)
    cumulative_by_conv: dict[str, dict[str, int]] = {}

    parts: list[str] = []
    s = forest.structure
    parts.append("#" * 72)
    parts.append(
        f"# Harness structure: root_rounds={forest.root_rounds}  "
        f"subagents={s['num_subagents']}  background_calls={s['num_background_calls']}  "
        f"compactions={s['num_compactions']}  max_depth={s['max_depth']}  "
        f"(raw success calls={len(entries)})"
    )
    parts.append("#" * 72)
    parts.append("")

    for step_idx, entry in enumerate(entries):
        usage = extract_entry_usage(entry)
        in_tok = usage["input_tokens"]
        out_tok = usage["output_tokens"]
        reason_tok = usage["reasoning_tokens"]

        conv = forest.conversation_of(step_idx)
        conv_id = conv.id if conv else "?"
        kind = conv.kind if conv else "root"
        depth = conv.depth if conv else 0
        indent = "    " * depth

        # Background calls are the harness's own auxiliary blips — collapse them.
        if kind == "background":
            text = ""
            for block in _extract_blocks(entry, tool_results_by_id):
                if block.get("type") == "text":
                    text = str(block.get("content") or "").strip().replace("\n", " ")[:60]
                    break
            parts.append(f"{indent}· [background call] in={in_tok} out={out_tok}  {text}")
            continue

        cum = cumulative_by_conv.setdefault(conv_id, {"in": 0, "out": 0, "reasoning": 0})
        cum["in"] += in_tok
        cum["out"] += out_tok
        cum["reasoning"] += reason_tok

        if step_idx in compaction_call_indices:
            parts.append("")
            parts.append(f"{indent}{'✂' * 36}")
            parts.append(f"{indent}✂ COMPACTION CALL — model summarizes the conversation "
                         f"(input={in_tok}: full history + summarize prompt; output below = the summary)")
            parts.append(f"{indent}{'✂' * 36}")
        if step_idx in compaction_resume_indices:
            parts.append(f"{indent}↩ resumes from the compaction summary (context_in now {in_tok})")

        # A subagent's result folds back into the parent's context at this step.
        for child_id in returns_by_index.get(step_idx, []):
            child = conv_by_id.get(child_id)
            if child:
                parts.append(
                    f"{indent}  ⤶ subagent:{child.subagent_type or '?'} ({child_id}) "
                    f"result merged into context here (ran at step "
                    f"{child.call_indices[0] if child.call_indices else '?'})"
                )

        tag = kind if kind != "subagent" else f"subagent:{conv.subagent_type or '?'}"
        parts.append(f"{indent}{'=' * (72 - len(indent))}")
        parts.append(
            f"{indent}  Step {step_idx}  [{conv_id} {tag}]  |  "
            f"context_in={in_tok}  out={out_tok}  reasoning={reason_tok}  |  "
            f"conv cumulative: in={cum['in']} out={cum['out']}"
        )
        parts.append(f"{indent}{'=' * (72 - len(indent))}")

        for block in _extract_blocks(entry, tool_results_by_id):
            btype = block.get("type")
            if btype == "thinking":
                text = str(block.get("content") or "").strip()
                if text:
                    parts.append("")
                    parts.append(f"{indent}--- thinking ---")
                    parts.append(_indent_text(text, indent))
            elif btype == "text":
                text = str(block.get("content") or "").strip()
                if text:
                    parts.append("")
                    parts.append(f"{indent}--- text ---")
                    parts.append(_indent_text(text, indent))
            elif btype == "tool_use":
                name = block.get("name", "unknown")
                parts.append("")
                parts.append(f"{indent}>>> tool: {name}")
                tool_input = block.get("input") or {}
                if isinstance(tool_input, dict):
                    for k, v in tool_input.items():
                        val_str = str(v)
                        if len(val_str) > 200:
                            val_str = val_str[:200] + "..."
                        parts.append(f"{indent}    {k}: {val_str}")
                else:
                    parts.append(f"{indent}    {tool_input}")
                # Subagent spawn point: announce the child conversation(s).
                if str(name).lower() in {"task", "agent"}:
                    for child_id in spawns_by_index.get(step_idx, []):
                        child = conv_by_id.get(child_id)
                        if child:
                            first = child.call_indices[0] if child.call_indices else "?"
                            ret = (
                                f", result returns at step {child.returns_at}"
                                if child.returns_at is not None
                                else ""
                            )
                            parts.append(
                                f"{indent}    ↳ spawns {child_id} "
                                f"(subagent:{child.subagent_type or '?'}, "
                                f"{len(child.call_indices)} calls, runs at step {first}{ret})"
                            )
                result_text = str(block.get("result") or "")
                if result_text:
                    parts.append(f"{indent}<<< result:")
                    parts.append(_indent_text(result_text, indent))

        parts.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _indent_text(text: str, indent: str) -> str:
    if not indent:
        return text
    return "\n".join(indent + line for line in text.splitlines())
