"""Plain terminal progress output for ``cage run``.

The :class:`RunProgressReporter` and its factories live here, split out of
:mod:`cage.cli.ui.run` which keeps the run banner, contract model and the
parameter/selection table rendering.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from cage.cli.ui.run import RunContract, _inspector_browser_urls, build_run_contract
from cage.contracts.coerce import int_or_zero
from cage.contracts.duration import split_duration_hms
from cage.contracts.logging import (
    register_active_progress_line,
    unregister_active_progress_line,
)
from cage.contracts.style import should_color, style
from cage.contracts.telemetry import ModelRequestEvent
from cage.contracts.trial_status import (
    FAILED_TRIAL_STATUSES,
    INTERRUPTED_TRIAL_STATUSES,
)
from cage.experiment.model.trial_id import parse_trial_id


@dataclass
class _TrialProgress:
    step: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    errors: int = 0
    cost_usd: float | None = None


@dataclass
class _ActiveTrial:
    agent_label: str
    trial_id: str
    sample_id: str
    trial_index: int
    started_at: float
    status: str = "running"


class RunProgressReporter:
    """Track run progress and print a small plain-text progress line."""

    def __init__(
        self,
        total_trials: int,
        *,
        enabled: bool = True,
        contract: RunContract | None = None,
        stream: Any | None = None,
        max_request_events: int = 12,
        refresh_per_second: int = 4,
        runnable_trials: int | None = None,
        resume_replayed_trials: int = 0,
    ) -> None:
        del refresh_per_second
        self.total_trials = max(0, int(total_trials or 0))
        # The LIVE progress bar denominator is the plan-to-run count, not the
        # grand total. On ``--resume`` the grand total folds in the trials kept
        # from disk (e.g. 60 = 33 replayed + 27 to run); counting those toward a
        # live "X/60" makes the bar jump to "33/60" before a single trial runs
        # this session, which reads as a bug. We track only what this invocation
        # actually executes — ``runnable_trials`` — and drop replayed trials from
        # the live "done" count (the banner already reports "N kept from disk").
        self.resume_replayed_trials = max(0, int(resume_replayed_trials or 0))
        if runnable_trials is None:
            self.live_total = self.total_trials
        else:
            self.live_total = max(0, int(runnable_trials or 0))
        self.enabled = bool(enabled)
        self.contract = contract
        self.stream = stream or sys.stderr
        self._color_enabled = should_color(self.stream)
        self.max_request_events = max(1, int(max_request_events or 1))

        self._lock = Lock()
        self._start_time = time.time()
        self._active: dict[tuple[str, str], _ActiveTrial] = {}
        self._trial_progress: dict[tuple[str, str], _TrialProgress] = {}
        self._request_events: list[ModelRequestEvent] = []
        self._trials_by_status: defaultdict[str, list[str]] = defaultdict(list)
        self._recent_trials: list[tuple[str, str]] = []
        self._agents: dict[str, int] = {}
        self._model_calls = 0
        self._progress_model_calls_seen = False
        self._completed_progress_model_calls = 0
        self._tokens_in = 0
        self._tokens_out = 0
        self._cost_usd = 0.0
        self._started = False
        self._uses_carriage_return = True
        self._last_line_len = 0
        self._last_progress_line = ""
        self._line_visible = False
        self._stopping = False
        # Set once the forced-exit banner is printed: worker threads still alive
        # during teardown must not redraw a progress line *after* the final
        # banner (it raced one in below the URLs before ``os._exit``).
        self._final_banner_printed = False

    @property
    def n_completed(self) -> int:
        return sum(len(trials) for trials in self._trials_by_status.values())

    @contextmanager
    def live(self) -> Iterator["RunProgressReporter"]:
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        register_active_progress_line(self)
        if self.contract is not None:
            print(
                self.contract.to_plain_text(color=self._color_enabled),
                file=self.stream,
                flush=True,
            )
            print(file=self.stream, flush=True)
        self._print_progress()

    def stop(self) -> None:
        if not self.enabled or not self._started:
            return
        self._print_progress()
        if self._uses_carriage_return:
            print(file=self.stream, flush=True)
        unregister_active_progress_line(self)
        self._started = False

    def agent_started(self, agent_label: str, total_trials: int) -> None:
        with self._lock:
            self._agents[agent_label] = max(0, int(total_trials or 0))
        self._print_progress()

    def trial_started(
        self,
        *,
        agent_label: str,
        trial_id: str,
        sample_id: str,
        trial_index: int,
    ) -> None:
        key = (agent_label, trial_id)
        with self._lock:
            self._active[key] = _ActiveTrial(
                agent_label=agent_label,
                trial_id=trial_id,
                sample_id=sample_id,
                trial_index=trial_index,
                started_at=time.time(),
            )
        self._print_progress()

    def update_trial_status(
        self,
        *,
        agent_label: str,
        trial_id: str,
        message: str,
    ) -> None:
        key = (agent_label, trial_id)
        with self._lock:
            if active := self._active.get(key):
                active.status = message or "running"
        self._print_progress()

    def update_trial_progress(
        self,
        *,
        agent_label: str,
        trial_id: str,
        progress: dict[str, Any],
    ) -> None:
        key = (agent_label, trial_id)
        with self._lock:
            self._trial_progress[key] = _progress_from_mapping(progress)
            self._progress_model_calls_seen = True
        self._print_progress()

    def record_model_request(self, event: ModelRequestEvent) -> None:
        with self._lock:
            self._request_events.append(event)
            if len(self._request_events) > self.max_request_events:
                self._request_events = self._request_events[-self.max_request_events :]
            self._model_calls += 1
            self._tokens_in += int_or_zero(event.input_tokens)
            self._tokens_out += int_or_zero(event.output_tokens)
            self._cost_usd += float(event.cost_usd or 0.0)
        self._print_progress()

    def trial_finished(
        self,
        *,
        agent_label: str,
        trial_id: str,
        status: str,
        duration_ms: int | float | None = None,
        exit_code: int | None = None,
    ) -> None:
        del duration_ms, exit_code
        status_key = _normalize_status(status)
        key = (agent_label, trial_id)
        with self._lock:
            self._active.pop(key, None)
            progress = self._trial_progress.pop(key, None)
            if progress is not None:
                self._completed_progress_model_calls += max(0, int_or_zero(progress.step))
            self._trials_by_status[status_key].append(trial_id)
            self._recent_trials.append((status_key, trial_id))
        self._print_progress()

    def trial_replayed(
        self,
        *,
        agent_label: str,
        trial_id: str,
        sample_id: str,
        trial_index: int,
        status: str = "completed",
    ) -> None:
        del sample_id, trial_index, status
        key = (agent_label, trial_id)
        with self._lock:
            self._active.pop(key, None)
            self._trial_progress.pop(key, None)
            self._trials_by_status["replayed"].append(trial_id)
            self._recent_trials.append(("replayed", trial_id))
        self._print_progress()

    def on_uncaught_exception(
        self,
        *,
        agent_label: str,
        trial_id: str,
        exception: Exception,
    ) -> None:
        self.trial_finished(
            agent_label=agent_label,
            trial_id=trial_id,
            status=f"uncaught {type(exception).__name__}",
        )

    def print_report(self) -> None:
        for status, trials in self._trials_by_status.items():
            print(f"{status}: {len(trials)}")
            for trial_id in trials:
                print(f"  {trial_id}")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = [
                {
                    "agent_label": item.agent_label,
                    "trial_id": item.trial_id,
                    "sample_id": item.sample_id,
                    "trial_index": item.trial_index,
                    "status": item.status,
                    "elapsed_s": max(0.0, time.time() - item.started_at),
                }
                for item in sorted(
                    self._active.values(),
                    key=lambda value: (value.agent_label, value.trial_index, value.trial_id),
                )
            ]
            by_status = {
                status: len(trials)
                for status, trials in self._trials_by_status.items()
            }
            progress_model_calls = self._completed_progress_model_calls + sum(
                max(0, int_or_zero(progress.step))
                for progress in self._trial_progress.values()
            )
            model_calls = (
                progress_model_calls
                if self._progress_model_calls_seen
                else self._model_calls
            )
            tokens_in = self._tokens_in
            tokens_out = self._tokens_out
            cost_usd = self._cost_usd
        # The live banner has no separate interrupted slot, so its "failed"
        # number is an at-a-glance "did not succeed" count: every non-success
        # terminal status (canonical failed + interrupted/cancelled) plus any
        # uncaught-exception status the reporter synthesizes. The canonical
        # record keeps the precise failed-vs-interrupted split.
        _not_ok_statuses = FAILED_TRIAL_STATUSES | INTERRUPTED_TRIAL_STATUSES
        failed = sum(
            count
            for status, count in by_status.items()
            if status in _not_ok_statuses or status.startswith("uncaught ")
        )
        # "done" tracks progress against this invocation's plan-to-run, so it
        # excludes trials replayed from disk (those were resolved in a prior
        # session and are reported separately as "replayed"). Without this, a
        # resumed run shows the bar pre-filled to the replayed count.
        replayed = by_status.get("replayed", 0)
        done = sum(by_status.values()) - replayed
        return {
            "total": self.live_total,
            "done": done,
            "completed": by_status.get("completed", 0),
            "failed": failed,
            "replayed": replayed,
            "active": active,
            "by_status": by_status,
            "model_calls": model_calls,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "elapsed_s": max(0.0, time.time() - self._start_time),
        }

    def recent_trials(self, limit: int = 5) -> list[str]:
        return [
            f"{status}: {trial_id}"
            for status, trial_id in reversed(self._recent_trials[-limit:])
        ]

    def _print_progress(self) -> None:
        if not self.enabled or not self._started or self._final_banner_printed:
            return
        line = self._progress_line()
        if self._line_visible and line == self._last_progress_line:
            return
        if self._uses_carriage_return:
            padding = " " * max(0, self._last_line_len - len(line))
            print("\r" + line + padding, end="", file=self.stream, flush=True)
            self._last_line_len = len(line)
            self._line_visible = True
        else:
            print(line, file=self.stream, flush=True)
            self._line_visible = True
        self._last_progress_line = line

    def print_graceful_stop_notice(self) -> None:
        """Explain the graceful-stop semantics on the FIRST Ctrl+C.

        The first SIGINT requests a graceful drain — in-flight trials run to
        completion, queued trials are cancelled — but the only visible effect was
        the progress bar flipping to "⏹ stopping". The handler's ``logger``
        message is swallowed by ``quiet_console_logging`` during a live run, so
        the user saw no explanation of what just happened or how to force-quit.
        Print it straight to the stream (signal-safe: no ``self._lock``, no disk
        I/O) so the contract is unambiguous.
        """
        if not self.enabled:
            return
        try:
            lead = "\n" if (self._uses_carriage_return and self._line_visible) else ""
            self.stream.write(
                f"{lead}"
                "⏸  Ctrl+C — stopping gracefully:\n"
                "   • in-flight trials keep running until they finish;\n"
                "   • queued trials that haven't started are cancelled;\n"
                "   • press Ctrl+C again to force-quit now (kills running trials).\n"
            )
            self.stream.flush()
            # Force the next progress redraw onto a fresh line below the notice.
            self._line_visible = False
        except Exception:  # noqa: BLE001
            pass

    def print_interrupt_banner(self) -> None:
        """Print a final summary on a forced (Ctrl+C×2 / SIGTERM) exit.

        The forced-exit path ``os._exit``s straight from the signal handler, so
        the normal end-of-run results table never gets a chance to render. This
        guarantees the user always sees the final situation — the last resolved/
        failed/token counts — plus the *live* inspector URL (the managed board is
        a detached process that outlives ``cage run``, so the link still works)
        and the durable run dir for ``cage inspect`` later. It runs inside a
        signal handler, so it must be allocation-light and lock-free: we write
        the already-rendered ``_last_progress_line`` and pre-built contract URLs
        directly rather than re-taking ``self._lock`` (which the interrupted main
        thread may be holding) or reading per-trial records off disk.
        """
        if not self.enabled:
            return
        # Stop any worker thread still alive during teardown from redrawing a
        # progress line below this final banner. Set before writing so a redraw
        # racing between here and the write is already gated out.
        self._final_banner_printed = True
        # Erase the live progress line so the results table starts on a clean row.
        try:
            if self._uses_carriage_return and self._line_visible:
                self.stream.write("\r" + (" " * self._last_line_len) + "\r")
                self.stream.flush()
                self._line_visible = False
        except Exception:  # noqa: BLE001
            pass
        # Immediate acknowledgment that the SECOND Ctrl+C registered, flushed
        # *before* the slower work below (reading every trial summary off disk
        # to render the table). The container/network teardown itself no longer
        # blocks here — it runs in a detached background sweep — so the banner
        # spells out that cleanup outlives this exit and that mashing Ctrl+C
        # again changes nothing.
        try:
            self.stream.write(
                "\n⏹  Ctrl+C×2 — force-quitting now:\n"
                "   • in-flight trials killed; their containers & networks are\n"
                "     removed by a background sweep that keeps running after exit.\n"
                "   • pressing Ctrl+C again does nothing — cleanup finishes on its\n"
                "     own (verify later with `docker ps` / `cage gc`).\n"
            )
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass
        run_dir = self._interrupt_run_dir()
        # The full per-trial results table — the SAME renderer the graceful exit
        # path uses — reading each trial's incrementally-written summary off disk
        # so a forced Ctrl+C×2 still shows what every trial reached, not just a
        # one-line tally. Best-effort: a signal-handler failure (disk error,
        # reentrant print) falls back to the last rendered progress line.
        rendered_table = False
        try:
            if run_dir:
                print_run_results([run_dir], interrupted=True, stream=self.stream)
                rendered_table = True
        except Exception:  # noqa: BLE001
            rendered_table = False
        if not rendered_table:
            try:
                line = self._last_progress_line or "progress [⏹ stopping] interrupted"
                self.stream.write(
                    f"\n{line}\nrun force-interrupted (Ctrl+C×2) — partial results saved.\n"
                )
                self.stream.flush()
            except Exception:  # noqa: BLE001
                pass
        # The live inspector URL(s) (the managed board outlives the run) plus the
        # durable ``cage inspect`` re-open command — same hint as the normal exit.
        try:
            urls = self._interrupt_inspector_urls()
            inspect_command = f"cage inspect {run_dir}" if run_dir else ""
            print_inspector_hint(urls, inspect_command, stream=self.stream)
        except Exception:  # noqa: BLE001
            pass

    def _interrupt_inspector_urls(self) -> list[str]:
        """Live inspector browser URL(s) for the forced-exit banner (no I/O)."""
        contract = self.contract
        if contract is None:
            return []
        try:
            return _inspector_browser_urls(contract)
        except Exception:  # noqa: BLE001
            return []

    def _interrupt_run_dir(self) -> str:
        """Run dir for the durable ``cage inspect`` hint in the forced banner."""
        contract = self.contract
        if contract is None:
            return ""
        return str(getattr(contract, "run_dir", "") or "")

    def clear_for_external_write(self) -> None:
        """Clear the live progress line before another console writer logs."""
        if (
            not self.enabled
            or not self._started
            or not self._uses_carriage_return
            or not self._line_visible
        ):
            return
        print("\r" + (" " * self._last_line_len) + "\r", end="", file=self.stream, flush=True)
        self._line_visible = False

    def redraw_after_external_write(self) -> None:
        """Restore the live progress line after another console writer logs."""
        if not self.enabled or not self._started or not self._uses_carriage_return:
            return
        self._print_progress()

    def _progress_line(self) -> str:
        snapshot = self.snapshot()
        raw_total = int(snapshot["total"] or 0)
        if raw_total <= 0:
            tokens_text = (
                f"{_format_count(snapshot['tokens_in'])}/"
                f"{_format_count(snapshot['tokens_out'])}"
            )
            parts = [
                f"{style('progress', 'cyan', 'bold', enabled=self._color_enabled)} "
                f"[{style('------------------------', 'yellow', enabled=self._color_enabled)}] "
                f"{style('no trials to run', 'yellow', 'bold', enabled=self._color_enabled)}",
                _progress_metric(
                    "llm_calls",
                    snapshot["model_calls"],
                    color="cyan",
                    enabled=self._color_enabled,
                ),
                _progress_metric(
                    "tokens",
                    tokens_text,
                    color="yellow",
                    enabled=self._color_enabled,
                ),
                *_progress_cost_parts(snapshot["cost_usd"], enabled=self._color_enabled),
                _progress_metric(
                    "elapsed",
                    _format_elapsed(snapshot["elapsed_s"]),
                    color="dim",
                    enabled=self._color_enabled,
                ),
            ]
            return ", ".join(parts)
        total = max(1, raw_total)
        done = int(snapshot["done"])
        filled = int(24 * min(done, total) / total)
        percent: object = int(100 * min(done, total) / total)
        # Interrupted: don't render a filling/100% bar (it reads as a clean
        # finish). Show a clear "stopping" indicator with the resolved count.
        if self._stopping:
            colored_bar = style("⏹ stopping", "yellow", "bold", enabled=self._color_enabled)
            done_text = style(
                f"{done}/{snapshot['total']} resolved (interrupted)",
                "yellow", "bold", enabled=self._color_enabled,
            )
            percent = ""
        else:
            colored_bar = (
                style("#" * filled, "green", enabled=self._color_enabled)
                + style("-" * (24 - filled), "dim", enabled=self._color_enabled)
            )
            done_text = style(
                f"{done}/{snapshot['total']} done",
                "green",
                "bold",
                enabled=self._color_enabled,
            )
        failed_color = "red" if int(snapshot["failed"]) else "green"
        tokens_text = (
            f"{_format_count(snapshot['tokens_in'])}/"
            f"{_format_count(snapshot['tokens_out'])}"
        )
        percent_text = (
            ""
            if self._stopping
            else style(str(percent) + "%", "green", "bold", enabled=self._color_enabled) + " "
        )
        parts = [
            f"{style('progress', 'cyan', 'bold', enabled=self._color_enabled)} "
            f"[{colored_bar}] "
            f"{percent_text}"
            f"{done_text}",
            _progress_metric(
                "running",
                len(snapshot["active"]),
                color="blue",
                enabled=self._color_enabled,
            ),
            _progress_metric(
                "failed",
                snapshot["failed"],
                color=failed_color,
                enabled=self._color_enabled,
            ),
            _progress_metric(
                "llm_calls",
                snapshot["model_calls"],
                color="cyan",
                enabled=self._color_enabled,
            ),
            _progress_metric(
                "tokens",
                tokens_text,
                color="yellow",
                enabled=self._color_enabled,
            ),
            *_progress_cost_parts(snapshot["cost_usd"], enabled=self._color_enabled),
            _progress_metric(
                "elapsed",
                _format_elapsed(snapshot["elapsed_s"]),
                color="dim",
                enabled=self._color_enabled,
            ),
        ]
        return ", ".join(parts)


def _progress_metric(label: str, value: object, *, color: str, enabled: bool) -> str:
    text_value = str(value)
    if not enabled:
        return f"{label}={text_value}"
    return (
        f"{style(label, 'dim', enabled=True)}="
        f"{style(text_value, color, 'bold', enabled=True)}"
    )


def _progress_cost_parts(cost_usd: object, *, enabled: bool) -> list[str]:
    try:
        cost = float(cost_usd or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    if cost <= 0.0:
        return []
    return [
        _progress_metric(
            "cost",
            _format_cost(cost),
            color="yellow",
            enabled=enabled,
        )
    ]


def create_run_reporter(
    *,
    enabled: bool,
    total_trials: int,
    contract: RunContract | None = None,
    stream: Any | None = None,
    mode: str | None = None,
    runnable_trials: int | None = None,
    resume_replayed_trials: int = 0,
) -> RunProgressReporter:
    del mode
    return RunProgressReporter(
        total_trials=total_trials,
        enabled=bool(enabled),
        contract=contract,
        stream=stream,
        runnable_trials=runnable_trials,
        resume_replayed_trials=resume_replayed_trials,
    )


def create_run_reporter_with_contract(
    *,
    config: Any,
    run_id: str,
    run_dir: Any,
    enabled: bool,
    total_trials: int,
    planned_trials: int,
    runnable_trials: int,
    resume_replayed_trials: int = 0,
    samples: list[dict[str, Any]],
    board_url: str = "",
    run_url: str = "",
    dashboard_url: str = "",
    view_links: list[Any] | None = None,
) -> RunProgressReporter:
    """Build the run-parameter contract and the terminal reporter together.

    This is the factory the conductor receives by injection: it owns the CLI's
    display knowledge (the run contract + the progress reporter) so the conductor
    never imports either. The conductor passes raw run parameters; the reporter
    carries the resulting ``RunContract`` for the disabled-output banner.
    """
    contract = build_run_contract(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        board_url=board_url,
        run_url=run_url,
        dashboard_url=dashboard_url,
        view_links=view_links,
        planned_trials=planned_trials,
        runnable_trials=runnable_trials,
        resume_replayed_trials=resume_replayed_trials,
        samples=samples,
    )
    return create_run_reporter(
        enabled=enabled,
        total_trials=total_trials,
        contract=contract,
        runnable_trials=runnable_trials,
        resume_replayed_trials=resume_replayed_trials,
    )


def _progress_from_mapping(progress: dict[str, Any]) -> _TrialProgress:
    step = int_or_zero(
        progress.get("successful_requests")
        if progress.get("successful_requests") is not None
        else progress.get("success", progress.get("total_requests"))
    )
    cost = progress.get("cost_usd")
    try:
        parsed_cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        parsed_cost = None
    return _TrialProgress(
        step=step,
        tokens_in=int_or_zero(progress.get("tokens_in")),
        tokens_out=int_or_zero(progress.get("tokens_out")),
        tokens_reasoning=int_or_zero(progress.get("tokens_reasoning")),
        errors=int_or_zero(progress.get("errors")),
        cost_usd=parsed_cost,
    )


def _normalize_status(status: str) -> str:
    return (status or "unknown").strip().lower() or "unknown"


def _format_count(value: int | float) -> str:
    try:
        return f"{int(value or 0):,}"
    except (TypeError, ValueError):
        return "0"


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):.2f}"


def _format_elapsed(value: float | int) -> str:
    hours, minutes, secs = split_duration_hms(int(value or 0))
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def print_run_results(
    run_dirs: list[Path] | list[str],
    *,
    interrupted: bool = False,
    stream: Any = None,
) -> None:
    """Print a per-trial results table mirroring the web run page.

    Reads each trial's incrementally-written summary (present even after an
    interrupt) through the same data layer the web inspector uses, so the
    terminal and the browser agree. Shows one row per trial — status, score,
    duration, model-call steps and token usage — plus a one-line tally matching
    the run page's status banner (target-passed / stopped / failed). Best-effort:
    never raises.
    """
    out = stream or sys.stdout
    header = "Run interrupted — partial results" if interrupted else "Run results"

    # Reuse the inspector's shared data layer (same source the web run page is
    # built on) so terminal and web agree: meta.json is the source of truth,
    # scores overlaid. Lazy import to avoid a static cli/ui → web dependency.
    try:
        from cage.web.data import find_trial_dirs, load_trial_summary
    except Exception:  # noqa: BLE001
        find_trial_dirs = None  # type: ignore[assignment]

    rows: list[dict[str, Any]] = []
    if find_trial_dirs is not None:
        run_status = "interrupted" if interrupted else "completed"
        for rd in run_dirs:
            rd = Path(rd)
            try:
                trial_dirs = find_trial_dirs(rd)
            except Exception:  # noqa: BLE001
                trial_dirs = []
            for trial_dir in trial_dirs:
                try:
                    info = load_trial_summary(trial_dir, run_status=run_status)
                except Exception:  # noqa: BLE001
                    info = {}
                rows.append(_result_row(info, trial_dir))

    if not rows:
        print("\n" + "=" * 60, file=out)
        print(header, file=out)
        print("=" * 60, file=out)
        print("  (no trial results found)", file=out, flush=True)
        return

    sample_w = min(34, max(8, max(len(r["sample"]) for r in rows)))
    status_w = min(14, max(6, max(len(r["status"]) for r in rows)))

    def _fmt(index: str, sample: str, status: str, score: str, dur: str, steps: str, tokens: str) -> str:
        return (
            f"  {index:>3}  {_truncate(sample, sample_w):<{sample_w}}  "
            f"{_truncate(status, status_w):<{status_w}}  {score:>6}  "
            f"{dur:>9}  {steps:>6}  {tokens:>15}"
        )

    table: list[str] = [_fmt("#", "sample", "status", "score", "duration", "steps", "tokens")]
    tok_in_total = tok_out_total = 0
    cost_total = 0.0
    for i, r in enumerate(rows, 1):
        tok_in_total += r["tok_in"]
        tok_out_total += r["tok_out"]
        cost_total += r["cost"]
        tokens = f"{_fmt_tokens_compact(r['tok_in'])}/{_fmt_tokens_compact(r['tok_out'])}"
        table.append(_fmt(str(i), r["sample"], r["status"], r["score"], r["duration"], r["steps"], tokens))

    title = f"{header} — {_results_tally(rows)}"
    width = min(120, max(len(title), max(len(line) for line in table)))
    print("\n" + "=" * width, file=out)
    print(title, file=out)
    print("=" * width, file=out)
    for line in table:
        print(line, file=out)
    footer = (
        f"  totals: {len(rows)} trials · "
        f"tokens {_fmt_tokens_compact(tok_in_total)} in / {_fmt_tokens_compact(tok_out_total)} out"
    )
    if cost_total > 0:
        footer += f" · ${cost_total:.2f}"
    print("\n" + footer, file=out, flush=True)


def _sample_from_trial_dir(trial_dir: Path) -> str:
    """Sample (task) id for a trial dir when ``sample_id`` is not yet recorded.

    A trial dir is ``.../trials/<task>[/pass_<n>]``; the sample is the task part
    of that runtime subpath — the same derivation the web inspector uses. Using
    it instead of the bare ``trial_dir.name`` stops in-flight pass@k trials
    (whose ``meta.json`` hasn't recorded ``sample_id`` yet) from printing the
    pass directory name (``pass_1``) as the sample.
    """
    parts = trial_dir.parts
    if "trials" in parts:
        idx = len(parts) - 1 - parts[::-1].index("trials")
        rel = "/".join(parts[idx + 1:])
        if rel:
            return parse_trial_id(rel)[0]
    return trial_dir.name


def _result_row(info: dict[str, Any], trial_dir: Path) -> dict[str, Any]:
    """Flatten one ``load_trial_summary`` result into a printable result row."""
    scores = info.get("scores") or {}
    score = "-"
    if isinstance(scores, dict) and scores:
        try:
            score = f"{float(next(iter(scores.values()))):.2f}"
        except (TypeError, ValueError):
            score = "-"
    duration_ms = int_or_zero(info.get("duration_ms"))
    progress = info.get("progress") or {}
    steps = int_or_zero(progress.get("total_requests") or progress.get("success"))
    usage = info.get("usage") or {}
    try:
        cost = float(usage.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return {
        "sample": str(info.get("sample_id") or _sample_from_trial_dir(trial_dir)),
        "status": str(info.get("status_label") or info.get("status") or "—"),
        "status_kind": str(info.get("status_kind") or ""),
        # Drive "running" off the *classified* kind, not the raw meta flag: a
        # trial in-flight at an interrupt keeps ``running=True`` in its summary
        # but classifies as Interrupted, and the tally must agree with the row.
        "running": str(info.get("status_kind") or "") == "running",
        "score": score,
        "duration": _format_elapsed(duration_ms / 1000.0) if duration_ms else "-",
        "steps": str(steps) if steps else "-",
        "tok_in": int_or_zero(usage.get("input_tokens")),
        "tok_out": int_or_zero(usage.get("output_tokens")),
        "cost": cost,
    }


def _results_tally(rows: list[dict[str, Any]]) -> str:
    """One-line status tally matching the web run page's banner buckets."""
    buckets: dict[str, int] = {
        "completed": 0,
        "target-passed": 0,
        "stopped": 0,
        "failed": 0,
        "running": 0,
        "pending": 0,
    }
    kind_to_bucket = {
        "success": "completed",
        "live_success": "target-passed",
        "warning": "stopped",
        "error": "failed",
        "pending": "pending",
    }
    for r in rows:
        if r["running"]:
            buckets["running"] += 1
            continue
        bucket = kind_to_bucket.get(r["status_kind"])
        if bucket:
            buckets[bucket] += 1
    parts = [f"{len(rows)} trials"]
    parts.extend(f"{name}={count}" for name, count in buckets.items() if count)
    return " · ".join(parts)


def _truncate(text: str, width: int) -> str:
    if len(text) <= width or width <= 1:
        return text[:width] if len(text) > width else text
    return text[: width - 1] + "…"


def _fmt_tokens_compact(value: int | float | None) -> str:
    n = int_or_zero(value)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_inspector_hint(
    browser_urls: list[str],
    inspect_command: str = "",
    *,
    stream: Any = None,
) -> None:
    """Remind the operator how to open the full run view once a run ends.

    The managed inspector board is a detached subprocess that outlives ``cage
    run``, so its browser URL keeps working after the command returns. Print it
    (plus the durable ``cage inspect`` re-open command) so users don't have to
    scroll back to the launch banner to find the link. Best-effort: never raises.
    """
    out = stream or sys.stdout
    if not browser_urls and not inspect_command:
        return
    print("", file=out)
    if browser_urls:
        print("Open the full run view in your browser (inspector still running):", file=out)
        for url in browser_urls:
            print(f"  {url}", file=out)
        if inspect_command:
            print(f"  Re-open later:  {inspect_command}", file=out)
    elif inspect_command:
        print(f"Open the run view:  {inspect_command}", file=out)
    print("", end="", file=out, flush=True)
