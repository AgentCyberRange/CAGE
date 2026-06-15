"""Anthropic<->OpenAI message/tool/response translation for the host proxy.

Pure request/response shape conversion + text tool-call parsing used by the
host proxy. NOTE: deliberately duplicated (not shared) with the in-container
sidecar, which must stay httpx-only with zero cage imports.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProxyModifyRule:
    """A rule for modifying proxy requests."""

    target: str  # "system_prompt"
    rule: str  # "append" | "prepend" | "replace"
    content: str


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _translate_tools_anthropic_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def _translate_messages_anthropic_to_openai(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    openai_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": str(content or "")})
            continue

        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "tool_use":
                tool_uses.append(block)
            elif block_type == "tool_result":
                tool_results.append(block)

        if role == "assistant" and tool_uses:
            text = "\n".join(text_parts) if text_parts else None
            tool_calls = []
            for tu in tool_uses:
                inp = tu.get("input", {})
                tool_calls.append({
                    "id": tu.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(inp) if not isinstance(inp, str) else inp,
                    },
                })
            openai_messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": tool_calls,
            })
            continue

        if tool_results:
            if text_parts:
                openai_messages.append({"role": "user", "content": "\n".join(text_parts)})
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = "\n".join(
                        str(b.get("text", "")) for b in tr_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": str(tr_content or ""),
                })
            continue

        openai_messages.append({"role": role, "content": "\n".join(text_parts)})

    return openai_messages


_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


_XML_FUNCTION_RE = re.compile(
    r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE,
)


_XML_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL | re.IGNORECASE,
)


def _coerce_xml_param_value(value: str) -> Any:
    stripped = value.strip()
    if stripped in ("true", "True"):
        return True
    if stripped in ("false", "False"):
        return False
    if stripped in ("null", "None"):
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
    except ValueError:
        pass
    return value


def _parse_xml_tool_call(raw: str) -> dict[str, Any] | None:
    fn_match = _XML_FUNCTION_RE.search(raw)
    if not fn_match:
        return None
    name = fn_match.group(1).strip()
    if not name:
        return None
    args: dict[str, Any] = {}
    for param in _XML_PARAMETER_RE.finditer(fn_match.group(2)):
        key = param.group(1).strip()
        if not key:
            continue
        args[key] = _coerce_xml_param_value(param.group(2))
    return {"name": name, "arguments": args}


def _loads_text_tool_call(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    candidates = [text]
    trimmed = text
    for _ in range(3):
        if not trimmed.endswith("}"):
            break
        trimmed = trimmed[:-1].rstrip()
        candidates.append(trimmed)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return _parse_xml_tool_call(text)


def _normalize_text_tool_input(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if args is None:
        args = call.get("input", {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
        return parsed if isinstance(parsed, dict) else {"raw": args}
    return {}


def _extract_text_tool_calls(raw_text: str, request_id: str) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []
    cursor = 0

    for match in _TEXT_TOOL_CALL_RE.finditer(raw_text):
        text_parts.append(raw_text[cursor:match.start()])
        call = _loads_text_tool_call(match.group(1))
        if call is None or not call.get("name"):
            text_parts.append(match.group(0))
        else:
            tool_blocks.append({
                "type": "tool_use",
                "id": f"{request_id}-text-tool-{len(tool_blocks)}",
                "name": str(call.get("name") or ""),
                "input": _normalize_text_tool_input(call),
            })
        cursor = match.end()

    text_parts.append(raw_text[cursor:])
    return "".join(text_parts).strip(), tool_blocks


def _translate_response_openai_to_anthropic(
    *,
    request_id: str,
    model: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    choices = response.get("choices", []) or []
    first = (choices[0] if choices else {}) or {}
    message = first.get("message", {}) or {}
    finish_reason = first.get("finish_reason", "stop")
    usage = response.get("usage", {}) or {}

    content_blocks: list[dict[str, Any]] = []

    raw_text = message.get("content") or ""
    native_tool_calls = message.get("tool_calls") or []
    text_tool_blocks: list[dict[str, Any]] = []
    if raw_text and not native_tool_calls:
        raw_text, text_tool_blocks = _extract_text_tool_calls(raw_text, request_id)

    if raw_text:
        content_blocks.append({"type": "text", "text": raw_text})

    for tc in native_tool_calls:
        func = tc.get("function", {}) or {}
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {"raw": args_str}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": args,
        })

    content_blocks.extend(text_tool_blocks)

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason_map = {"tool_calls": "tool_use", "length": "max_tokens"}
    stop_reason = "tool_use" if text_tool_blocks else stop_reason_map.get(finish_reason, "end_turn")

    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def _apply_modify_rules(
    system_content: str,
    rules: list[ProxyModifyRule],
) -> tuple[str, str]:
    """Apply modification rules to system prompt.

    Returns (original, modified).
    """
    original = system_content
    modified = system_content
    for rule in rules:
        if rule.target != "system_prompt":
            continue
        if rule.rule == "append":
            modified = f"{modified}\n\n{rule.content}" if modified else rule.content
        elif rule.rule == "prepend":
            modified = f"{rule.content}\n\n{modified}" if modified else rule.content
        elif rule.rule == "replace":
            modified = rule.content
    return original, modified


def _apply_system_template(original: str, template: str) -> str:
    """Render system template. {{ system_raw }} is replaced with original."""
    return template.replace("{{ system_raw }}", original).replace("{{system_raw}}", original)


def _build_openai_request(
    anthropic_request: dict[str, Any],
    *,
    modify_rules: list[ProxyModifyRule],
    system_template: str = "",
    max_output_tokens_cap: int | None = None,
    upstream_extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Build OpenAI chat request from Anthropic request.

    Returns (openai_payload, original_system, modified_system).
    """
    messages: list[dict[str, Any]] = []
    system_content = _normalize_content(anthropic_request.get("system", ""))

    if system_template:
        original_system = system_content
        modified_system = _apply_system_template(system_content, system_template)
    else:
        original_system, modified_system = _apply_modify_rules(system_content, modify_rules)

    if modified_system:
        messages.append({"role": "system", "content": modified_system})

    messages.extend(
        _translate_messages_anthropic_to_openai(
            anthropic_request.get("messages", []) or []
        )
    )

    payload: dict[str, Any] = {
        "model": anthropic_request.get("model", ""),
        "messages": messages,
        # Explicit non-streaming so resp.json() doesn't have to handle SSE
        # bytes and proxy.jsonl gets one record per agent call.
        "stream": False,
    }

    anthropic_tools = anthropic_request.get("tools")
    if anthropic_tools:
        payload["tools"] = _translate_tools_anthropic_to_openai(anthropic_tools)

    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if key in anthropic_request and anthropic_request[key] is not None:
            payload[key] = anthropic_request[key]

    if upstream_extra_body:
        # Registry-pinned inference config (e.g. Qwen ``chat_template_kwargs``
        # / sampling) overrides the same-named params the agent CLI sent. Runs
        # before the cap clamp so ``max_output_tokens_cap`` stays a hard ceiling.
        for key, value in upstream_extra_body.items():
            payload[key] = value

    if max_output_tokens_cap is not None:
        requested = payload.get("max_tokens")
        if isinstance(requested, int):
            payload["max_tokens"] = min(requested, max_output_tokens_cap)

    return payload, original_system, modified_system

