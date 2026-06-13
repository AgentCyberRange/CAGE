"""Utilities for parsing Codex CLI ``--json`` event streams."""

from __future__ import annotations

import json
from dataclasses import dataclass


_KNOWN_CODEX_EVENT_TYPES = {
    "thread.started",
    "turn.started",
    "turn.completed",
    "turn.failed",
    "item.started",
    "item.updated",
    "item.completed",
    "error",
}


@dataclass(frozen=True)
class CodexEventStreamSummary:
    """Parsed view of a Codex CLI event stream."""

    is_event_stream: bool
    last_agent_message: str = ""
    terminal_error: str = ""
    last_error: str = ""
    last_command_output: str = ""
    error_messages: tuple[str, ...] = ()

    def final_output(self) -> str:
        if self.last_agent_message:
            return self.last_agent_message
        if self.terminal_error:
            return self.terminal_error
        if self.last_error:
            return self.last_error
        return self.last_command_output


def parse_codex_event_stream(text: str) -> CodexEventStreamSummary:
    """Parse a Codex CLI ``--json`` output stream.

    Returns a ``CodexEventStreamSummary`` with ``is_event_stream=False`` when
    the input does not look like Codex NDJSON.
    """

    source = str(text or "")
    first_type = ""
    last_agent_message = ""
    terminal_error = ""
    last_error = ""
    last_command_output = ""
    error_messages: list[str] = []

    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "").strip()
        if not event_type:
            continue
        if not first_type:
            first_type = event_type
        if event_type == "error":
            message = _clean_text(event.get("message"))
            if message:
                last_error = message
                error_messages.append(message)
            continue
        if event_type == "turn.failed":
            message = _extract_error_message(event.get("error"))
            if message:
                terminal_error = message
                error_messages.append(message)
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type == "agent_message":
            message = _clean_text(item.get("text"))
            if message:
                last_agent_message = message
            continue
        if event_type == "item.completed" and item_type == "command_execution":
            aggregated = _clean_text(item.get("aggregated_output"))
            if aggregated:
                last_command_output = aggregated[:1200]

    if first_type not in _KNOWN_CODEX_EVENT_TYPES:
        return CodexEventStreamSummary(is_event_stream=False)
    return CodexEventStreamSummary(
        is_event_stream=True,
        last_agent_message=last_agent_message,
        terminal_error=terminal_error,
        last_error=last_error,
        last_command_output=last_command_output,
        error_messages=tuple(error_messages),
    )


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_error_message(value: object) -> str:
    if isinstance(value, dict):
        return _clean_text(value.get("message") or value.get("error") or value)
    return _clean_text(value)
