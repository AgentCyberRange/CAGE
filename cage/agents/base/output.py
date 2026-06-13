"""Shared output-parsing primitives for Claude-Code-shaped CLI streams.

``claude`` and ``hermes`` emit the same NDJSON ``stream-json`` event shapes
(``type == "result"`` carrying the final answer, ``type == "assistant"``
messages carrying text blocks); their ``parse_output`` extraction loops were
line-identical twins. This module owns that skeleton once.

Deliberately NOT used by the other agents: kimi's CLI schemas churned across
versions so its loop tolerates extra shapes (``role`` keys, string content,
bare ``message`` events), qwen speaks single-object ``--output-format json``
first, and codex has a dedicated event-stream parser. Forcing those onto one
extractor would change what each agent accepts — they share only the
:func:`failure_banner` guard.
"""

from __future__ import annotations

import json

from cage.sandbox.exec import ExecResult


def failure_banner(result: ExecResult) -> str | None:
    """The shared "agent died with no output" banner, or None to keep parsing."""

    if result.exit_code != 0 and not result.stdout.strip():
        return f"[Agent exited with code {result.exit_code}]\n{result.stderr[:1000]}"
    return None


def normalize_recorded_output(output: str) -> str:
    """Repair a recorded ``task_output.json`` output field on read.

    Runtime always runs ``AgentType.parse_output`` before persisting, but two
    sources still leave raw CLI streams in recorded outputs: run dirs written
    by older cage versions (re-scored offline via ``cage score``), and
    codex's parse-time fallback that keeps raw stdout when the event stream
    carried no final message. The knowledge of *which* recorded formats need
    repair belongs to the agents package — scorers call this one primitive
    and never learn an agent name.

    Today the only repair is the codex ``--json`` event stream.
    """

    try:
        from cage.agents.codex.output import parse_codex_event_stream

        summary = parse_codex_event_stream(output)
        if summary.is_event_stream:
            return summary.final_output()
    except Exception:
        pass
    return output


def extract_stream_json_text(stdout: str) -> tuple[str, bool]:
    """Extract the final result/assistant text from an NDJSON event stream.

    Returns ``(last_text, saw_json)`` — callers choose their own raw-stdout
    fallback when the stream carried no parseable answer.
    """

    last_text = ""
    saw_json = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        saw_json = True

        # Final result event wins when present.
        if obj.get("type") == "result":
            result_text = obj.get("result", "")
            if result_text:
                last_text = result_text
            continue

        # Otherwise keep the latest assistant text block.
        if obj.get("type") == "assistant" and obj.get("message"):
            msg = obj["message"]
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_text = block.get("text", "")
    return last_text, saw_json
