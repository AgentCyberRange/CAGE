"""LLM-judge helper for the StrongReject benchmark.

Generic enough to be a framework utility, but currently used by exactly one
benchmark (StrongReject) — kept local until a second caller appears.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from cage.models import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Raw result from an LLM judge call."""

    parsed: dict[str, Any]
    raw_text: str
    success: bool = True
    error: str = ""


def call_judge(
    *,
    model: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> JudgeResult:
    """POST to an OpenAI-compatible ``/chat/completions`` and parse JSON content."""
    url = f"{model.base_url.rstrip('/')}/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if model.api_key:
        headers["Authorization"] = f"Bearer {model.api_key}"

    payload = {
        "model": model.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        with httpx.Client(timeout=model.timeout, trust_env=False) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        raw_text = data["choices"][0]["message"]["content"] or ""
        parsed = _extract_json(raw_text)
        return JudgeResult(parsed=parsed, raw_text=raw_text)
    except Exception as exc:
        logger.warning("Judge call failed: %s", exc)
        return JudgeResult(parsed={}, raw_text="", success=False, error=str(exc))


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of text — handles ```` ```json ``` ```` fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}
