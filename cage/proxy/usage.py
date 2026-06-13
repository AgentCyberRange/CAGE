"""Token usage extraction helpers for proxy responses."""

from __future__ import annotations

from cage.contracts.coerce import int_or_zero

from typing import Any


def extract_response_usage(response: dict[str, Any] | None) -> dict[str, int]:
    """Return normalized token usage from OpenAI or Anthropic response bodies."""
    if not isinstance(response, dict):
        response = {}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}
    output_details = usage.get("output_tokens_details") or {}
    if not isinstance(output_details, dict):
        output_details = {}

    input_tokens = int_or_zero(usage.get("prompt_tokens")) or int_or_zero(usage.get("input_tokens"))
    input_tokens += int_or_zero(usage.get("cache_read_input_tokens"))
    input_tokens += int_or_zero(usage.get("cache_creation_input_tokens"))

    return {
        "input_tokens": input_tokens,
        "output_tokens": int_or_zero(usage.get("completion_tokens")) or int_or_zero(usage.get("output_tokens")),
        "reasoning_tokens": (
            int_or_zero(usage.get("reasoning_tokens"))
            or int_or_zero(completion_details.get("reasoning_tokens"))
            or int_or_zero(output_details.get("reasoning_tokens"))
        ),
    }


def extract_entry_usage(entry: dict[str, Any]) -> dict[str, int]:
    """Return normalized token usage from a proxy.jsonl entry."""
    for key in ("upstream_response", "anthropic_response"):
        usage = extract_response_usage(entry.get(key) or {})
        if usage["input_tokens"] or usage["output_tokens"] or usage["reasoning_tokens"]:
            return usage
    return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
