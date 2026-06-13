"""Dashboard and results.csv writing for completed/interrupted runs.

Owns the human-facing run output: per-agent ``dashboard.json`` (latest snapshot),
the append-only ``run_history.json`` sidecar, ``results.csv``, and the optional
benchmark-provided ``dashboard_view.json``. Reads scores/usage/summary produced by
the scoring and proxy collaborators and the planned-trial reconstruction from the
resume module; registers its outputs as canonical run artifacts.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from cage.config.sections import OutputConfig
from cage.experiment.engine.run_context import ExperimentRun
from cage.experiment.model import TrialResult
from cage.experiment.engine.resume import (
    _merge_interrupted_planned_trials,
    _summary_from_trial_infos,
)
from cage.artifacts.canonical_marks import _mark_canonical_run_output_artifacts
from cage.artifacts.jsonio import (
    _load_json_file,
)
from cage.sandbox.naming import (
    _parse_agent_label,
)
from cage.scoring.lifecycle import (
    _build_summary,
)
from cage.proxy.monitor import (
    _parse_proxy_stats,
)


logger = logging.getLogger("cage.experiment.engine.reporting")

RUN_HISTORY_FILE = "run_history.json"


def _run_history_label(index: int) -> str:
    return "Initial run" if index == 0 else f"Resume #{index}"

def _timestamp_ms_from_iso(text: str) -> int:
    text = str(text or "").strip()
    if not text:
        return 0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return 0

def _duration_ms_between(started_at: str, completed_at: str) -> int:
    start = _timestamp_ms_from_iso(started_at)
    end = _timestamp_ms_from_iso(completed_at)
    if start and end and end >= start:
        return end - start
    return 0

def _append_run_history(
    run_dir: Path,
    *,
    started_at: str,
    completed_at: str,
    status: str,
) -> dict[str, Any]:
    """Append one run invocation to ``run_history.json``.

    ``dashboard.json`` is intentionally still a latest-snapshot file for
    compatibility. This append-only sidecar preserves the run-id's full
    lifecycle across ``cage run --resume`` invocations.
    """
    history_path = run_dir / RUN_HISTORY_FILE
    raw = _load_json_file(history_path)
    attempts = raw.get("attempts", []) if isinstance(raw, dict) else []
    if not isinstance(attempts, list):
        attempts = []
    attempts = [dict(item) for item in attempts if isinstance(item, dict)]

    for index, attempt in enumerate(attempts):
        attempt["sequence"] = index + 1
        attempt["label"] = _run_history_label(index)
        attempt["is_latest"] = False

    entry = {
        "sequence": len(attempts) + 1,
        "label": _run_history_label(len(attempts)),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": _duration_ms_between(started_at, completed_at),
        "status": status,
        "source": "recorded",
        "is_latest": True,
    }
    attempts.append(entry)

    history = {
        "version": 1,
        "attempts": attempts,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return history

def _write_dashboards(
    run: ExperimentRun,
    all_results: dict[str, list[TrialResult]],
    cage_runs: Path,
    run_id: str,
    started_at: str,
    completed_at: str,
    *,
    interrupted: bool = False,
) -> dict[str, dict[str, str]]:
    """Write dashboard.json + results.csv for each agent. Returns artifacts dict.

    Called from both normal completion and interrupt handler, so partial
    results are always persisted.
    """
    import json as json_mod

    ordered_labels = list(all_results.keys())
    agent_artifacts: dict[str, dict[str, str]] = {}

    for label in ordered_labels:
        results = all_results[label]
        agent_dir_name, _mode = _parse_agent_label(label)
        run_dir = cage_runs / agent_dir_name / run_id
        status = "interrupted" if interrupted else "completed"
        run_history = _append_run_history(
            run_dir,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
        )
        dashboard: dict[str, Any] = {
            "run_id": run_id,
            "experiment": run.name,
            "started_at": started_at,
            "completed_at": completed_at,
            "status": status,
            "run_history": run_history,
            "agents": {},
        }
        summary = _build_summary(results)
        trials = []
        for r in results:
            trial_info: dict[str, Any] = {
                "trial_id": r.trial_id,
                "trial_index": r.trial_index,
                "sample_id": r.sample_id,
                "exit_code": r.exit_code,
            }
            if r.timing:
                trial_info["duration_ms"] = r.timing.duration_ms
            if r.error:
                trial_info["error"] = r.error
            if r.metadata.get("status"):
                trial_info["status"] = r.metadata["status"]
            if r.metadata.get("termination_reason"):
                trial_info["termination_reason"] = r.metadata["termination_reason"]
            if r.metadata.get("termination_detail"):
                trial_info["termination_detail"] = r.metadata["termination_detail"]
            if r.metadata.get("termination_source"):
                trial_info["termination_source"] = r.metadata["termination_source"]
            if r.scores:
                trial_info["scores"] = {n: s.value for n, s in r.scores.items()}
            sample = r.metadata.get("sample", {})
            output_cfg = run.output
            if output_cfg.dashboard_prompt:
                trial_info["prompt"] = sample.get("forbidden_prompt") or sample.get("content", "")
            if output_cfg.dashboard_output:
                trial_info["output"] = r.output
            proxy_stats = _parse_proxy_stats(r.proxy_log)
            trial_info["usage"] = {
                "input_tokens": proxy_stats["input_tokens"],
                "output_tokens": proxy_stats["output_tokens"],
                "reasoning_tokens": proxy_stats["reasoning_tokens"],
                "num_requests": proxy_stats["num_requests"],
            }
            if output_cfg.dashboard_reasoning and proxy_stats["reasoning_content"]:
                trial_info["reasoning_content"] = proxy_stats["reasoning_content"]
            if r.scores:
                for n, s in r.scores.items():
                    if s.answer:
                        trial_info.setdefault("score_details", {})[n] = {
                            "value": s.value,
                            "answer": s.answer,
                            "explanation": s.explanation,
                        }
            trials.append(trial_info)
        if interrupted:
            trials = _merge_interrupted_planned_trials(run_dir, trials)
            summary = _summary_from_trial_infos(trials, fallback=summary)
        dashboard["agents"][label] = {**summary, "trials": trials}

        dashboard_path = run_dir / "dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(
            json_mod.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.info("Dashboard written (%s): %s", status, dashboard_path)

        results_path = run_dir / "results.csv"
        _write_results_csv(results_path, {label: results}, output_cfg=run.output)

        view_path = _maybe_write_dashboard_view(run, run_dir)
        _mark_canonical_run_output_artifacts(
            run_dir=run_dir,
            dashboard_path=dashboard_path,
            results_path=results_path,
            dashboard_view_path=view_path,
        )

        artifact: dict[str, str] = {
            "run_dir": str(run_dir),
            "dashboard_path": str(dashboard_path),
            "results_path": str(results_path),
        }
        if view_path is not None:
            artifact["dashboard_view_path"] = str(view_path)
        agent_artifacts[label] = artifact

    return agent_artifacts

def _maybe_write_dashboard_view(
    run: ExperimentRun,
    run_dir: Path,
) -> Path | None:
    """Ask the benchmark for a visualization spec and persist it.

    Returns the file path on success, ``None`` if the benchmark opted out or
    raised. Benchmark errors are logged but never block run completion —
    a missing view just makes the inspector fall back to the generic table.
    """
    try:
        spec = run.benchmark.build_dashboard(run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_dashboard failed for %s: %s", run_dir, exc)
        return None
    if spec is None:
        return None
    try:
        path = run_dir / "dashboard_view.json"
        spec.write(path)
        logger.info("Dashboard view written: %s", path)
        return path
    except OSError as exc:
        logger.warning("Failed to write dashboard_view.json under %s: %s", run_dir, exc)
        return None

def _write_results_csv(
    path: Path,
    all_results: dict[str, list[TrialResult]],
    output_cfg: "OutputConfig | None" = None,
) -> None:
    """Write a flat CSV with one row per trial across all agents.

    Which columns are included is controlled by OutputConfig.
    By default all columns are written; set csv_* flags to False to exclude
    heavy text fields (prompt, output, reasoning_content).
    """
    import csv

    if output_cfg is None:
        from cage.config.sections import OutputConfig
        output_cfg = OutputConfig()

    metric_names: list[str] = []
    seen: set[str] = set()
    for results in all_results.values():
        for r in results:
            for name in r.scores:
                if name not in seen:
                    metric_names.append(name)
                    seen.add(name)

    fieldnames = ["agent", "trial_id", "sample_id"]
    if output_cfg.csv_prompt:
        fieldnames.append("prompt")
    if output_cfg.csv_output:
        fieldnames.append("output")
    fieldnames.extend([
        "exit_code", "duration_s",
        "input_tokens", "output_tokens", "reasoning_tokens", "num_requests",
    ])
    if output_cfg.csv_reasoning:
        fieldnames.append("reasoning_content")
    fieldnames.extend(["tool_call_count", "terminated_by_limit", "error"])
    for m in metric_names:
        fieldnames.extend([f"score_{m}", f"answer_{m}"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for label, results in all_results.items():
            for r in results:
                sample = r.metadata.get("sample", {})
                proxy_stats = _parse_proxy_stats(r.proxy_log)
                row: dict[str, Any] = {
                    "agent": label,
                    "trial_id": r.trial_id,
                    "sample_id": r.sample_id,
                    "exit_code": r.exit_code,
                    "duration_s": round(r.timing.duration_ms / 1000, 1) if r.timing else "",
                    "input_tokens": proxy_stats["input_tokens"],
                    "output_tokens": proxy_stats["output_tokens"],
                    "reasoning_tokens": proxy_stats["reasoning_tokens"],
                    "num_requests": proxy_stats["num_requests"],
                    "tool_call_count": r.tool_call_count,
                    "terminated_by_limit": r.terminated_by_limit,
                    "error": r.error or "",
                }
                if output_cfg.csv_prompt:
                    row["prompt"] = sample.get("forbidden_prompt") or sample.get("content", "")
                if output_cfg.csv_output:
                    row["output"] = r.output
                if output_cfg.csv_reasoning:
                    row["reasoning_content"] = proxy_stats["reasoning_content"]
                for m in metric_names:
                    s = r.scores.get(m)
                    row[f"score_{m}"] = s.value if s else ""
                    row[f"answer_{m}"] = s.answer if s else ""
                writer.writerow(row)
