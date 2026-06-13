"""Host-side proxy monitoring for trials.

This module owns everything related to *observing* a trial's in-container HTTP
proxy from the host: polling the per-trial ``progress.json`` the sidecar writes,
emitting reporter progress / model-request events, enforcing the ``max_rounds``
round budget by terminating the agent process, and parsing the raw
``proxy.jsonl`` log into token/cost/reasoning statistics after a trial ends.

It is a pure collaborator of the trial runtime: it does not own the trial
lifecycle, only the proxy-observation slice of it.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from cage.contracts.telemetry import ModelRequestEvent
from cage.proxy.usage import extract_entry_usage
from cage.sandbox.containers import Container

logger = logging.getLogger("cage.proxy.monitor")


def _progress_int(progress: dict[str, Any], key: str, *, fallback: int = 0) -> int:
    try:
        return int(progress.get(key, fallback) or fallback)
    except (TypeError, ValueError):
        return fallback

def _progress_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

class _ProxyMonitor:
    """Periodically read host-side proxy progress and log stats."""

    def __init__(
        self,
        container: Container,
        log_dir: str,
        trial_id: str,
        artifact_dir: Path | None = None,
        reporter: Any | None = None,
        agent_label: str = "",
        poll_interval: float = 10.0,
        process: Any | None = None,
        max_rounds: int = -1,
    ) -> None:
        del container, log_dir
        self.trial_id = trial_id
        self.artifact_dir = artifact_dir
        self.reporter = reporter
        self.agent_label = agent_label
        self.poll_interval = poll_interval
        self.process = process
        self.max_rounds = max_rounds
        self.terminated_by_max_rounds = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_count = 0
        self._last_tokens_in = 0
        self._last_tokens_out = 0
        self._last_tokens_reasoning = 0
        self._last_cost_usd: float | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        try:
            self._report()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.poll_interval)
            if self._stop.is_set():
                break
            try:
                self._report()
            except Exception:
                pass

    def _report(self) -> None:
        progress = self._read_progress()
        if progress is None:
            return
        try:
            count = int(progress.get("total_requests", 0) or 0)
        except (TypeError, ValueError):
            return
        self._emit_reporter_progress(progress)
        self._maybe_stop_on_max_rounds(progress)
        if count == self._last_count:
            return
        new_requests = count - self._last_count
        self._emit_reporter_request(progress)
        self._last_count = count
        logger.info(
            "  [proxy] trial=%s requests=%d (+%d) last_status=%s "
            "context_in=%d out=%d errors=%d",
            self.trial_id,
            count,
            new_requests,
            progress.get("last_status", "?"),
            int(progress.get("tokens_in", 0) or 0),
            int(progress.get("tokens_out", 0) or 0),
            int(progress.get("errors", 0) or 0),
        )

    def _maybe_stop_on_max_rounds(self, progress: dict[str, Any]) -> None:
        if self.terminated_by_max_rounds:
            return
        if self.process is None or self.max_rounds <= 0:
            return
        rounds = _progress_int(
            progress,
            "successful_requests",
            fallback=_progress_int(
                progress,
                "success",
                fallback=_progress_int(progress, "total_requests"),
            ),
        )
        if rounds < self.max_rounds:
            return
        if not _process_is_running(self.process):
            return
        self.terminated_by_max_rounds = True
        logger.info(
            "  [proxy] trial=%s max_rounds reached (%d/%d), terminating agent",
            self.trial_id,
            rounds,
            self.max_rounds,
        )
        _terminate_process(self.process)

    def _emit_reporter_progress(self, progress: dict[str, Any]) -> None:
        if self.reporter is None:
            return
        update = getattr(self.reporter, "update_trial_progress", None)
        if callable(update):
            update(
                agent_label=self.agent_label,
                trial_id=self.trial_id,
                progress=progress,
            )

    def _emit_reporter_request(self, progress: dict[str, Any]) -> None:
        if self.reporter is None:
            return
        record = getattr(self.reporter, "record_model_request", None)
        if not callable(record):
            return
        tokens_in = _progress_int(progress, "tokens_in")
        tokens_out = _progress_int(progress, "tokens_out")
        tokens_reasoning = _progress_int(progress, "tokens_reasoning")
        cost_usd = _progress_float(progress.get("cost_usd"))
        step = _progress_int(
            progress,
            "successful_requests",
            fallback=_progress_int(
                progress,
                "success",
                fallback=_progress_int(progress, "total_requests"),
            ),
        )
        last_status = str(progress.get("last_status") or "")
        event = ModelRequestEvent(
            trial_id=self.trial_id,
            step=step,
            status=last_status or "?",
            input_tokens=max(0, tokens_in - self._last_tokens_in),
            output_tokens=max(0, tokens_out - self._last_tokens_out),
            reasoning_tokens=max(0, tokens_reasoning - self._last_tokens_reasoning),
            cost_usd=(
                None if cost_usd is None else max(0.0, cost_usd - (self._last_cost_usd or 0.0))
            ),
            error="" if last_status == "success" else last_status,
        )
        self._last_tokens_in = tokens_in
        self._last_tokens_out = tokens_out
        self._last_tokens_reasoning = tokens_reasoning
        self._last_cost_usd = cost_usd
        record(event)

    def _read_progress(self) -> dict[str, Any] | None:
        if self.artifact_dir is None:
            return None
        progress_path = self.artifact_dir / "progress.json"
        try:
            raw = progress_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

def _process_is_running(process: Any) -> bool:
    try:
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
    except Exception:
        return True
    return True

def _terminate_process(process: Any) -> None:
    """Best-effort termination for an async agent process."""
    if not _process_is_running(process):
        return
    try:
        process.terminate()
    except Exception:
        pass

def _start_live_success_stop_thread(
    process: Any,
    monitors: list[tuple[Any, bool]],
) -> tuple[threading.Event, threading.Thread | None]:
    """Terminate the agent process when a live-success monitor asks us to stop."""
    stop_event = threading.Event()
    if not monitors:
        return stop_event, None

    def _watch() -> None:
        while not stop_event.is_set():
            for event, should_stop in monitors:
                if event.is_set():
                    if should_stop:
                        _terminate_process(process)
                    return
            stop_event.wait(0.2)

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    return stop_event, thread

def _parse_proxy_stats(proxy_log_path: Path | None) -> dict[str, Any]:
    """Extract token usage and reasoning content from a trial's proxy.jsonl.

    Returns a dict with:
      - input_tokens, output_tokens, reasoning_tokens (totals)
      - num_requests (successful upstream calls)
      - reasoning_content (concatenated thinking from all requests)
    """
    stats: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "num_requests": 0,
        "reasoning_content": "",
    }
    if not proxy_log_path or not proxy_log_path.exists():
        return stats

    import json as json_mod

    reasoning_parts: list[str] = []
    for line in proxy_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json_mod.loads(line)
        except json_mod.JSONDecodeError:
            continue
        if entry.get("status") != "success":
            continue

        usage = extract_entry_usage(entry)
        stats["input_tokens"] += usage["input_tokens"]
        stats["output_tokens"] += usage["output_tokens"]
        stats["reasoning_tokens"] += usage["reasoning_tokens"]
        stats["num_requests"] += 1

        # Extract reasoning_content from message
        resp = entry.get("upstream_response") or entry.get("anthropic_response") or {}
        choices = resp.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            rc = msg.get("reasoning_content", "")
            if rc:
                reasoning_parts.append(rc)

    stats["reasoning_content"] = "\n---\n".join(reasoning_parts)
    return stats
