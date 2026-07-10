"""Runtime monitors for reactive and polling live-success checks.

Live monitors call ``Benchmark.check_done`` to fetch the raw check-endpoint
response, then feed it into ``Benchmark.scorer().score(ctx)`` via
``ScoringContext.live_payload``. A trial is considered successful when any
returned ``Score.value >= 1.0``. This keeps mid-trial and end-of-trial
scoring on the same code path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from cage.artifacts.live_success import (
    load_live_success,
    parse_live_checks_success,
    record_live_success,
)
from cage.scoring import GatherRuntime, ScoringContext

logger = logging.getLogger(__name__)

# Canonical file: every check_done invocation (polling AND reactive) is
# appended here so a single timeline can be reconstructed for audit.
CHECK_DONE_LOG_NAME = "check_done_polls.jsonl"


def _json_lines(path: Path, start: int) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], start
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    new_lines = lines[start:]
    entries: list[dict[str, Any]] = []
    for line in new_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries, len(lines)


def _extract_tool_commands(entry: dict[str, Any]) -> list[str]:
    """Extract newly requested tool command text from a proxy entry response.

    The container proxy normalises *all* upstream shapes (Anthropic native,
    OpenAI tool_calls, qwen/Hermes-style inline ``<tool_call>...`` XML in
    ``message.content``) into structured Anthropic ``tool_use`` blocks
    before writing ``anthropic_response`` to ``proxy.jsonl`` (see
    ``cage/proxy/host.py::_translate_response_openai_to_anthropic``).
    So this reader only needs to handle the canonical translated shape —
    no inline parsing fallback required.
    """
    commands: list[str] = []
    for block in _response_tool_blocks(entry):
        commands.extend(_command_values(block.get("input", {})))
    return commands


def _response_tool_blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    # Prefer the proxy's translated Anthropic response — it carries
    # canonical tool_use blocks regardless of upstream model dialect.
    # Only fall back to ``upstream_response`` if the translation field
    # is absent (older trial logs, or non-translating proxy mode).
    response = entry.get("anthropic_response") or entry.get("upstream_response") or {}
    if isinstance(response.get("content"), list):
        blocks = []
        for block in response["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                blocks.append(
                    {
                        "name": block.get("name", ""),
                        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                    }
                )
        return blocks

    choices = response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message", {})
    blocks = []
    for tool_call in message.get("tool_calls") or []:
        func = tool_call.get("function", {})
        raw_args = func.get("arguments", "{}")
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"raw": raw_args}
        if isinstance(args, dict):
            blocks.append({"name": func.get("name", ""), "input": args})
    return blocks


def _command_values(value: Any) -> list[str]:
    commands: list[str] = []
    if isinstance(value, str):
        commands.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"command", "cmd", "script", "code", "raw"}:
                commands.extend(_command_values(item))
            elif isinstance(item, (dict, list)):
                commands.extend(_command_values(item))
    elif isinstance(value, list):
        for item in value:
            commands.extend(_command_values(item))
    return commands


_SCHEME_HINTS = ("http://", "https://")


def _command_matches_trigger(command: str, triggers: list[str]) -> str | None:
    """Return the matching trigger substring iff the command looks like a
    real network call to one of the benchmark's live-check endpoints.

    Match rule: trigger substring is in the command AND the command either
    carries an explicit URL scheme (``http://``/``https://``) or the trigger
    itself is host-qualified (contains a non-leading colon, e.g.
    ``target:9091``). The URL-shape guard stops false matches on prompt
    text or filenames that happen to mention a port number.

    What constitutes a trigger comes from the benchmark via
    :meth:`Benchmark.live_check_triggers`; cage's runtime stays
    benchmark-agnostic.
    """
    if not triggers:
        return None
    text = command.lower()
    has_scheme = any(hint in text for hint in _SCHEME_HINTS)
    for trigger in triggers:
        if not trigger:
            continue
        needle = trigger.lower()
        if needle not in text:
            continue
        host_qualified = ":" in trigger.lstrip(":")
        if has_scheme or host_qualified:
            return trigger
    return None


def _command_touches_9091(command: str) -> bool:
    """Backwards-compat shim — pre-refactor entry point still used by tests
    and external callers. Equivalent to ``_command_matches_trigger`` with
    the default proof-upload trigger list."""
    return _command_matches_trigger(command, [":9091", "target:9091"]) is not None


command_touches_9091 = _command_touches_9091


def _score_live_payload(
    benchmark: Any,
    sample: dict[str, Any],
    trial_dir: Path,
    trial_id: str,
    raw_output: str,
) -> tuple[bool, dict[str, Any]]:
    """Run the benchmark's scorer with the raw check-done payload.

    Returns ``(success, scores_dict)`` where ``success`` is True iff any
    returned score value is >= 1.0. ``scores_dict`` is the serialised score
    map for inclusion in the verdict evidence.
    """
    scorer = benchmark.scorer()
    ctx = ScoringContext(
        trial_id=trial_id,
        sample=sample,
        trial_dir=trial_dir,
        live_payload=raw_output,
    )
    try:
        scores = scorer.score(ctx)
    except Exception:
        logger.debug("Live scorer call failed", exc_info=True)
        return False, {}
    success = any(s.value >= 1.0 for s in scores.values())
    serialised = {
        name: {"value": s.value, "answer": s.answer, "explanation": s.explanation}
        for name, s in scores.items()
    }
    return success, serialised


class ReactiveLiveCheckMonitor:
    """Watch agent-triggered live checks and record success verdicts."""

    def __init__(
        self,
        *,
        benchmark: Any,
        container: Any,
        sample: dict[str, Any],
        trial_dir: Path,
        trial_id: str,
        proxy_jsonl: Path,
        live_checks_jsonl: Path,
        poll_interval: float = 1.0,
        check_on_submit: bool = True,
        check_on_9091_call: bool = True,
        call_counter: "_CheckDoneCounter | None" = None,
    ) -> None:
        self.benchmark = benchmark
        self.container = container
        self.sample = sample
        self.trial_dir = trial_dir
        self.trial_id = trial_id
        self.proxy_jsonl = proxy_jsonl
        self.live_checks_jsonl = live_checks_jsonl
        self.poll_interval = poll_interval
        self.check_on_submit = check_on_submit
        self.check_on_9091_call = check_on_9091_call
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proxy_line_count = 0
        self._live_checks_line_count = 0
        self.success_event = threading.Event()
        self.verdict: dict[str, Any] | None = None
        # Shared counter so polling + reactive emit a monotonic poll_index
        # across the merged ``check_done_polls.jsonl`` audit log.
        self._counter = call_counter or _CheckDoneCounter()
        # The benchmark decides which URL fragments mean "the agent just
        # hit our live-check endpoint" — cage stays agnostic. Empty list
        # => reactive checking is effectively disabled.
        try:
            triggers = list(benchmark.live_check_triggers(sample) or [])
        except Exception:
            logger.debug("live_check_triggers lookup failed", exc_info=True)
            triggers = []
        self._triggers = [t for t in triggers if isinstance(t, str) and t]

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def poll_once(self) -> dict[str, Any] | None:
        """Scan new artifacts once and return a success verdict if found."""
        existing = load_live_success(self.trial_dir)
        if existing:
            self._set_success(existing)
            return existing
        if self.check_on_submit:
            verdict = self._scan_live_checks()
            if verdict:
                return verdict
        if self.check_on_9091_call:
            return self._scan_proxy_for_9091_triggers()
        return None

    def _run(self) -> None:
        while not self._stop.is_set() and not self.success_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.debug("Reactive live-check monitor poll failed", exc_info=True)
            if self.success_event.is_set():
                return
            self._stop.wait(self.poll_interval)

    def _scan_live_checks(self) -> dict[str, Any] | None:
        if not self.live_checks_jsonl.is_file():
            return None
        lines = self.live_checks_jsonl.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[self._live_checks_line_count:]:
            entry = parse_live_checks_success(line)
            if entry is None:
                continue
            verdict = record_live_success(
                trial_dir=self.trial_dir,
                trial_id=self.trial_id,
                benchmark=str(getattr(self.benchmark, "name", "")),
                mode="reactive",
                source="submit",
                evidence={
                    "path": str(self.live_checks_jsonl),
                    "answer_hash": entry.get("answer_hash", ""),
                    "source": entry.get("source", ""),
                },
            )
            self._set_success(verdict)
            return verdict
        self._live_checks_line_count = len(lines)
        return None

    def _scan_proxy_for_9091_triggers(self) -> dict[str, Any] | None:
        if not self._triggers:
            return None
        entries, new_count = _json_lines(self.proxy_jsonl, self._proxy_line_count)
        self._proxy_line_count = new_count

        for entry in entries:
            if entry.get("status") != "success":
                continue
            for command in _extract_tool_commands(entry):
                matched = _command_matches_trigger(command, self._triggers)
                if matched is None:
                    continue
                try:
                    raw = str(self.benchmark.scorer().gather(
                        GatherRuntime(sample=self.sample, container=self.container)
                    ))
                except Exception:
                    logger.debug("scorer.gather failed during reactive scan", exc_info=True)
                    continue
                if not raw:
                    continue
                poll_index = self._counter.next()
                success, scores = _score_live_payload(
                    self.benchmark, self.sample, self.trial_dir, self.trial_id, raw,
                )
                self._append_reactive_check_done(
                    command=command,
                    output=raw,
                    poll_index=poll_index,
                    success=success,
                    proxy_ts_ms=int(entry.get("ts_ms") or 0),
                    matched=matched,
                )
                if not success:
                    continue
                verdict = record_live_success(
                    trial_dir=self.trial_dir,
                    trial_id=self.trial_id,
                    benchmark=str(getattr(self.benchmark, "name", "")),
                    mode="reactive",
                    source="agent_live_check_trigger",
                    evidence={
                        "poll_index": poll_index,
                        "trigger": {"matched": matched, "command": command[:500]},
                        "check_done": raw,
                        "scores": scores,
                    },
                )
                self._set_success(verdict)
                return verdict
        return None

    def _append_reactive_check_done(
        self,
        *,
        command: str,
        output: str,
        poll_index: int,
        success: bool,
        matched: str,
        proxy_ts_ms: int = 0,
    ) -> None:
        runtime_dir = self.trial_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        path = runtime_dir / CHECK_DONE_LOG_NAME
        entry = {
            "mode": "reactive",
            "source": "agent_live_check_trigger",
            "poll_index": poll_index,
            "ts_ms": int(time.time() * 1000),
            "proxy_ts_ms": proxy_ts_ms,
            "trigger": {"matched": matched, "command": command[:500]},
            "success": success,
            "output": output,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _set_success(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict
        self.success_event.set()


class CheckDonePoller:
    """Actively poll the benchmark's check-done endpoint."""

    def __init__(
        self,
        *,
        benchmark: Any,
        container: Any,
        sample: dict[str, Any],
        trial_dir: Path,
        trial_id: str,
        poll_interval: float = 5.0,
        call_counter: "_CheckDoneCounter | None" = None,
        confirm_polls: int = 1,
    ) -> None:
        self.benchmark = benchmark
        self.container = container
        self.sample = sample
        self.trial_dir = trial_dir
        self.trial_id = trial_id
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.success_event = threading.Event()
        self.verdict: dict[str, Any] | None = None
        self._counter = call_counter or _CheckDoneCounter()
        # Kept as a derived property for backwards-compatibility with tests
        # that inspect ``poller.poll_count``; the canonical counter is
        # ``self._counter``.
        self.poll_count = 0
        # Debounce state: lock-in only after N consecutive positive polls
        # for the same verdict class. Defends against transient validator
        # flips (target slowdowns, restart jitter) that briefly mark
        # success while the agent has done nothing.
        self._confirm_polls = max(1, int(confirm_polls))
        self._success_streak = 0
        # Track the verdict class so a NEW message resets the streak.
        # Mixing "Outbound service successful" and "DoS successful" should
        # NOT count as 2 confirmations.
        self._last_success_class = ""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def poll_once(self) -> dict[str, Any] | None:
        existing = load_live_success(self.trial_dir)
        if existing:
            self._set_success(existing)
            return existing

        try:
            raw = str(self.benchmark.scorer().gather(
                GatherRuntime(sample=self.sample, container=self.container)
            ))
        except Exception:
            logger.debug("scorer.gather failed during poll", exc_info=True)
            return None
        if not raw:
            return None

        poll_index = self._counter.next()
        self.poll_count = poll_index
        success, scores = _score_live_payload(
            self.benchmark, self.sample, self.trial_dir, self.trial_id, raw,
        )

        verdict_class = self._extract_verdict_class(raw)
        threshold = self._resolved_confirm_polls(raw)

        # Benchmark-side second-opinion: reject verdicts where the
        # underlying validator is known to fire spuriously (e.g. a
        # DoS check tripping on its own /done reload).
        # Treat a rejected positive as if scored false — keeps the
        # debounce streak coherent.
        rejected = False
        if success:
            try:
                accepted = bool(self.benchmark.validate_live_verdict(
                    self.container, self.sample, verdict_class, raw,
                ))
            except Exception:
                logger.debug("validate_live_verdict failed", exc_info=True)
                accepted = True
            if not accepted:
                success = False
                rejected = True

        # Debounce: a success must repeat (same verdict class) ``threshold``
        # times in a row before we commit it. Any not-success or
        # different-class success resets the streak so a single transient
        # validator flip can't lock in a verdict.
        if success and verdict_class == self._last_success_class:
            self._success_streak += 1
        elif success:
            self._success_streak = 1
            self._last_success_class = verdict_class
        else:
            self._success_streak = 0
            self._last_success_class = ""
        confirmed = success and self._success_streak >= threshold

        self._append_poll(
            output=raw,
            poll_index=poll_index,
            success=success,
            confirmed=confirmed,
            streak=self._success_streak,
            required=threshold,
            verdict_class=verdict_class,
            rejected_by_benchmark=rejected,
        )

        if not confirmed:
            return None
        verdict = record_live_success(
            trial_dir=self.trial_dir,
            trial_id=self.trial_id,
            benchmark=str(getattr(self.benchmark, "name", "")),
            mode="polling",
            source="check_done",
            evidence={
                "poll_index": poll_index,
                "check_done": raw,
                "scores": scores,
                "consecutive_confirmations": self._success_streak,
                "required_confirmations": threshold,
                "verdict_class": verdict_class,
            },
        )
        self._set_success(verdict)
        return verdict

    def _resolved_confirm_polls(self, raw: str) -> int:
        """Benchmark override → global config → 1."""
        try:
            override = self.benchmark.live_check_confirm_polls(self.sample, raw)
        except Exception:
            logger.debug("live_check_confirm_polls failed", exc_info=True)
            override = None
        if isinstance(override, int) and override >= 1:
            return override
        return self._confirm_polls

    def _extract_verdict_class(self, raw: str) -> str:
        """Stable class key derived from the check_done output, used to
        bucket the success streak. Falls back to the raw text so even
        benchmarks that don't follow the ``{"message": ...}``
        shape get coherent debouncing."""
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw.strip()[:120]
        if isinstance(payload, dict):
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        return raw.strip()[:120]

    def _run(self) -> None:
        while not self._stop.is_set() and not self.success_event.is_set():
            interval = self._resolved_interval()
            if self._stop.wait(interval):
                return
            try:
                self.poll_once()
            except Exception:
                logger.debug("Check_done poller failed", exc_info=True)

    def _resolved_interval(self) -> float:
        """Per-sample interval override → global ``poll_interval``.

        Re-resolved on every iteration so a benchmark could in principle
        adapt the cadence based on trial state. In practice the override
        is static per sample (e.g. lollms always wants a sparse poll).
        """
        try:
            override = self.benchmark.live_check_polling_interval(self.sample)
        except Exception:
            logger.debug("live_check_polling_interval failed", exc_info=True)
            override = None
        if isinstance(override, (int, float)) and override > 0:
            return float(override)
        return self.poll_interval

    def _append_poll(
        self,
        *,
        output: str,
        poll_index: int,
        success: bool,
        confirmed: bool = False,
        streak: int = 0,
        required: int = 1,
        verdict_class: str = "",
        rejected_by_benchmark: bool = False,
    ) -> None:
        runtime_dir = self.trial_dir / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        path = runtime_dir / CHECK_DONE_LOG_NAME
        entry = {
            "mode": "polling",
            "source": "check_done",
            "poll_index": poll_index,
            "ts_ms": int(time.time() * 1000),
            "success": success,
            # Auditable debounce telemetry — needed to explain WHY a
            # transient validator flip didn't lock in success.
            "confirmed": confirmed,
            "streak": streak,
            "required": required,
            "verdict_class": verdict_class,
            # True when the underlying scorer said success but the
            # benchmark's validate_live_verdict rejected (e.g. a
            # DoS verdict overruled by a direct target-health probe).
            "rejected_by_benchmark": rejected_by_benchmark,
            "output": output,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _set_success(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict
        self.success_event.set()


class _CheckDoneCounter:
    """Thread-safe monotonic counter shared by poller + reactive monitor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
