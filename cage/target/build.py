"""Dispatch benchmark-owned target image builds."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from threading import Lock

from cage.benchmarks import BenchmarkBuildSummary
from cage.benchmarks.registry import load_project_yaml
from cage.benchmarks.loader import load_benchmark_from_module

_PRINT_LOCK = Lock()

_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m",
}


def _color() -> bool:
    """True when stdout is an interactive terminal that should get ANSI styling.

    Honours ``NO_COLOR`` and degrades to plain text whenever output is piped or
    captured (logs, ``capsys`` in tests) so the wire text stays stable.
    """
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:  # noqa: BLE001 - a missing/exotic stream just means "no color"
        return False


def _style(text: str, *, color: str | None = None, bold: bool = False, dim: bool = False) -> str:
    if not _color():
        return text
    codes = ""
    if bold:
        codes += _ANSI["bold"]
    if dim:
        codes += _ANSI["dim"]
    if color:
        codes += _ANSI[color]
    return f"{codes}{text}{_ANSI['reset']}" if codes else text


def _sym(symbol: str) -> str:
    """A leading status glyph — only on color terminals, so piped logs stay plain."""
    return f"{symbol} " if _color() else ""


def build_benchmark_targets(
    project_path: Path,
    *,
    limit: int | None = None,
    only: list[str] | None = None,
    max_workers: int = 1,
    dry_run: bool = False,
    rebuild: bool = False,
) -> BenchmarkBuildSummary:
    """Call the benchmark's build hook for selected samples."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    benchmark = load_benchmark_from_project(project_path)
    try:
        samples = list(benchmark.iter_samples_limited(limit=limit, sample_ids=only))
        _print_build_header(
            project_path,
            samples=samples,
            only=only,
            limit=limit,
            max_workers=max_workers,
            dry_run=dry_run,
        )
        try:
            return benchmark.build_targets(
                samples,
                reporter=_print_build_event,
                max_workers=max_workers,
                dry_run=dry_run,
                rebuild=rebuild,
            )
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
    finally:
        try:
            benchmark.teardown()
        except Exception:
            pass


def load_benchmark_from_project(project_path: Path):
    """Load only the benchmark object from a project file.

    Target-only commands use this instead of ``resolve`` so unrelated
    agent/model auth does not block build or target lifecycle checks.
    """
    project_path = Path(project_path).expanduser().resolve()
    raw = load_project_yaml(project_path)
    base_dir = project_path.parent
    eval_raw = raw.get("eval", {})
    bench_cfg = eval_raw.get("benchmark", eval_raw)

    if isinstance(bench_cfg, dict) and "module" in bench_cfg:
        module_path = base_dir / str(bench_cfg["module"])
        bench_kwargs = {
            k: v
            for k, v in bench_cfg.items()
            if k not in ("module", "class")
        }
        benchmark = load_benchmark_from_module(
            module_path,
            class_name=bench_cfg.get("class"),
            kwargs=bench_kwargs,
        )
    elif isinstance(bench_cfg, str):
        module_path = base_dir / "benchmark.py"
        benchmark = load_benchmark_from_module(module_path)
    else:
        raise ValueError("eval.benchmark must specify a module path or benchmark name")

    benchmark.setup()
    return benchmark


def _load_benchmark_for_build(project_path: Path):
    """Backward-compatible alias for older tests/imports."""
    return load_benchmark_from_project(project_path)


def _print_build_header(
    project_path: Path,
    *,
    samples: list[dict],
    only: list[str] | None,
    limit: int | None,
    max_workers: int,
    dry_run: bool,
) -> None:
    title = "Benchmark build dry-run" if dry_run else "Benchmark build"
    rule = "─" * 60
    print(_style(rule, color="cyan", dim=True), flush=True)
    print(_style(f" {title}", color="cyan", bold=True), flush=True)
    print(_style(rule, color="cyan", dim=True), flush=True)
    if not dry_run:
        # The very first thing on screen: this step can run for a VERY long time.
        # Surface it loudly and up front so nobody assumes the CLI has hung and
        # kills it mid-build.
        print(
            _style(f"{_sym('⏳')}This build can take a VERY long time.", color="yellow", bold=True),
            flush=True,
        )
        for line in (
            "Often tens of minutes — and possibly an hour or more on the first",
            "run, when Docker pulls large base images before building each",
            "target. This is expected; the CLI has NOT hung. A status line",
            "prints every 30s so you can see a long build is still alive.",
            "Press Ctrl+C to abort.",
        ):
            print(_style(f"   {line}", color="yellow"), flush=True)
        print("", flush=True)
    print(f" Project: {project_path}", flush=True)
    print(f" Selected samples: {len(samples)}", flush=True)
    print(f" Build workers: {max_workers}", flush=True)
    if dry_run:
        print(_style(" Mode: dry-run (no build commands executed)", dim=True), flush=True)
    filters: list[str] = []
    if only:
        filters.append("only=" + ",".join(only))
    if limit is not None:
        filters.append(f"limit={limit}")
    if filters:
        print(f" Filters: {' '.join(filters)}", flush=True)
    print("", flush=True)


def _print_build_event(event: str, payload: dict) -> None:
    target_id = str(payload.get("target_id") or "?")
    index = payload.get("index")
    total = payload.get("total")
    prefix = f"[{index}/{total}] " if index and total else ""
    lines: list[str] = []

    if event == "plan":
        total_targets = payload.get("total_targets")
        if total_targets is not None:
            selected_samples = payload.get("selected_samples")
            if selected_samples is not None:
                lines.append(_style(
                    f"Build targets: {total_targets} "
                    f"(from {selected_samples} selected samples)",
                    color="cyan", bold=True,
                ))
            else:
                lines.append(_style(f"Build targets: {total_targets}", color="cyan", bold=True))
            _print_lines(lines)
        return

    if event == "start":
        kind = str(payload.get("kind") or "build")
        lines.append(
            _sym("▸") + _style(f"{prefix}{target_id}", bold=True)
            + _style(f": building {kind}…", dim=True)
        )
        command = payload.get("command")
        if command:
            lines.append(_style(f"  command: {_format_command(command)}", dim=True))
        lines.extend(_payload_detail_lines(payload))
        _print_lines(lines)
        return

    if event == "dry-run":
        kind = str(payload.get("kind") or "build")
        status = str(payload.get("status") or "planned")
        action = str(payload.get("action") or "build")
        verb = "would skip" if status == "skipped" else f"would {action}"
        lines.append(_sym("▸") + _style(f"{prefix}{target_id}: {verb} {kind}", color="cyan"))
        command = payload.get("command")
        if command and status != "skipped":
            lines.append(_style(f"  command: {_format_command(command)}", dim=True))
        lines.extend(_payload_detail_lines(payload))
        error = str(payload.get("error") or "").strip()
        if error:
            lines.append(f"  note: {error}")
        _print_lines(lines)
        return

    if event == "heartbeat":
        duration = _format_duration(float(payload.get("duration_s") or 0.0))
        _print_lines([
            _sym("·")
            + _style(f"{prefix}{target_id}: still building — {duration} elapsed", color="yellow")
        ])
        return

    if event == "finish":
        status = str(payload.get("status") or "done")
        duration = _format_duration(float(payload.get("duration_s") or 0.0))
        exit_text = ""
        if status == "failed" and payload.get("returncode") is not None:
            exit_text = f" (exit {payload.get('returncode')})"
        head = f"{prefix}{target_id}: {status}{exit_text} in {duration}"
        if status == "failed":
            lines.append(_sym("✗") + _style(head, color="red", bold=True))
        else:
            lines.append(_sym("✓") + _style(head, color="green"))
        error = str(payload.get("error") or "").strip()
        if error:
            lines.append(_style(f"  error: {error}", color="red"))
        _print_lines(lines)
        return


def _print_lines(lines: list[str]) -> None:
    with _PRINT_LOCK:
        for line in lines:
            print(line, flush=True)


def _payload_detail_lines(payload: dict) -> list[str]:
    raw_details = payload.get("details")
    details: list[str] = []
    if isinstance(raw_details, (list, tuple)):
        for item in raw_details:
            details.extend(_split_detail(str(item)))
    detail = str(payload.get("detail") or "").strip()
    if detail:
        details.extend(_split_detail(detail))
    return [f"  detail: {item}" for item in details]


def _split_detail(detail: str) -> list[str]:
    return [line.strip() for line in str(detail or "").splitlines() if line.strip()]


def _format_command(command: object) -> str:
    if isinstance(command, (list, tuple)):
        return " ".join(shlex.quote(str(part)) for part in command)
    return str(command)


def _format_duration(duration_s: float) -> str:
    if duration_s < 60:
        return f"{duration_s:.1f}s"
    minutes, seconds = divmod(int(duration_s), 60)
    return f"{minutes}m{seconds:02d}s"


def print_build_summary(
    summary: BenchmarkBuildSummary,
    *,
    compact: bool = False,
) -> None:
    if not summary.results:
        print("No targets selected.", flush=True)
        return

    if not compact:
        for result in summary.results:
            duration = (
                f" ({_format_duration(result.duration_s)})"
                if result.duration_s > 0
                else ""
            )
            if result.status == "built":
                print(_style(f"[built] {result.target_id}{duration}", color="green"), flush=True)
            elif result.status == "planned":
                detail_lines = _split_detail(result.detail) or ["would run build hook"]
                print(_style(f"[plan]  {result.target_id}", color="cyan"), flush=True)
                for line in detail_lines:
                    print(_style(f"        {line}", dim=True), flush=True)
            elif result.status == "skipped":
                detail = result.error or "no build step"
                print(
                    _style(f"[skip]  {result.target_id}{duration}: {detail}", color="yellow"),
                    flush=True,
                )
                if result.detail:
                    for line in _split_detail(result.detail):
                        print(_style(f"        {line}", dim=True), flush=True)
            else:
                detail = _failure_summary_line(result.error)
                duration = (
                    f" ({_format_duration(result.duration_s)})"
                    if result.duration_s > 0
                    else ""
                )
                if not detail and result.command:
                    detail = " ".join(result.command)
                print(
                    _style(f"[fail]  {result.target_id}{duration}: {detail}".rstrip(),
                           color="red", bold=True),
                    flush=True,
                )
    planned = getattr(summary, "planned", 0)
    planned_text = f" planned={planned}" if planned else ""
    summary_line = (
        f"Summary: total={summary.total}{planned_text} built={summary.built} "
        f"skipped={summary.skipped} failed={summary.failed}"
    )
    print(_style("─" * 60, color="cyan", dim=True), flush=True)
    print(_style(summary_line, color=("red" if summary.failed else "green"), bold=True), flush=True)


def _failure_summary_line(error: str) -> str:
    lines = [line.strip() for line in str(error or "").splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        if "ERROR:" in line or "error:" in line.lower():
            return _truncate_line(line)
    return _truncate_line(lines[-1])


def _truncate_line(line: str, max_chars: int = 240) -> str:
    return line if len(line) <= max_chars else line[: max_chars - 3] + "..."
