"""Delta-builders for the live polling API.

Each function takes a fully-resolved run/trial summary and a ``since_ms``
cursor; it returns a compact JSON-serialisable payload that contains
only entries whose underlying state has changed since the cursor.

The wire format is intentionally minimal so a 320-trial poll with
nothing changed is well under 1 KB.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cage.contracts.coerce import int_or_zero
from cage.web.cache import (
    dashboard_projection_is_stale,
    run_has_live_activity,
    safe_mtime_ns,
)
from cage.web.data import (
    RunInfo,
    find_trial_dirs,
    is_trial_dir,
    load_dashboard,
    load_planned_trial_records,
    load_trial_activity,
    load_trial_summary_cached,
    pending_trial_summary,
    planned_trial_dir,
    trial_summary_from_dashboard_entry,
    trial_summary_signature,
)


def relative_to_root(path: Path, root: Path) -> str:
    """Path of ``path`` relative to ``root``, lexically (no symlink follow).

    Run dirs may be symlinks under ``<root>/.cage_runs`` that resolve outside
    ``root`` (e.g. linked in from another worktree). Resolving them would break
    ``relative_to(root)``, so normalize ``..``/abs without following symlinks
    and keep the symlink as a path component. Falls back to a fully-resolved
    relative path for the normal (non-symlinked) layout.
    """
    rp = os.path.normpath(os.path.abspath(path))
    rr = os.path.normpath(os.path.abspath(root))
    if rp == rr:
        return ""
    prefix = rr + os.sep
    if rp.startswith(prefix):
        return rp[len(prefix):]
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _encode_path(path: Path, root: Path) -> str:
    """Mirror of ``app.encode_path``. Kept here so tests don't need Flask."""
    rel = relative_to_root(path, root)
    return base64.urlsafe_b64encode(rel.encode()).decode()


def _ms(value_ns: int) -> int:
    if value_ns <= 0:
        return 0
    return value_ns // 1_000_000


def _demote_info_if_stale(info: dict[str, Any], run_live: bool) -> dict[str, Any]:
    """Present a phantom "running" trial as Interrupted when nothing is live.

    Mirrors ``app._demote_stale_running_rows`` for the polling payload so the
    page doesn't flip a row back to Running on the next delta.
    """
    if not run_live and (info.get("running") or info.get("status_kind") == "running"):
        info["running"] = False
        info["status_kind"] = "error"
        info["status_label"] = "Interrupted"
        info["status_detail"] = "Run stopped while this trial was still running"
    return info


def build_runs_delta(
    runs: list[RunInfo],
    *,
    root: Path,
    since_ms: int = 0,
) -> dict[str, Any]:
    """Build the ``/api/runs`` payload.

    Includes a run iff its summary signature ms > ``since_ms``. Always
    includes the count totals so the index header can refresh without
    needing per-run detail.
    """
    now_ms = int(time.time() * 1000)
    payload_runs: list[dict[str, Any]] = []
    max_sig_ms = since_ms

    for run in runs:
        sig_ms = _run_signature_ms(run)
        max_sig_ms = max(max_sig_ms, sig_ms)
        if since_ms and sig_ms <= since_ms and not run.running:
            # Settled run that hasn't moved — skip.
            continue
        payload_runs.append(_run_to_json(run, root=root, signature_ms=sig_ms))

    summary = {
        "total_runs": len(runs),
        "running_runs": sum(1 for r in runs if r.running),
    }

    return {
        "now_ms": now_ms,
        "since_ms": since_ms,
        "max_signature_ms": max_sig_ms,
        "runs": payload_runs,
        "summary": summary,
    }


def _run_to_json(run: RunInfo, *, root: Path, signature_ms: int) -> dict[str, Any]:
    return {
        "encoded_path": _encode_path(run.path, root),
        "run_id": run.run_id,
        "project": run.project,
        "agent_label": run.agent_label,
        "agent_name": run.agent_name,
        "model_name": run.model_name,
        "experiment": run.experiment,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "status": run.status,
        "running": run.running,
        "running_trials": run.running_trials,
        "live_total_requests": run.live_total_requests,
        "live_errors": run.live_errors,
        "last_active_ts_ms": run.last_active_ts_ms,
        "duration_ms": run.duration_ms,
        "run_history": run.run_history,
        "signature_ms": signature_ms,
    }


def _run_signature_ms(run: RunInfo) -> int:
    """Signature used by the index poller for lifecycle-visible changes."""
    file_sig_ms = max(
        _ms(safe_mtime_ns(run.path / "dashboard.json")),
        _ms(safe_mtime_ns(run.path / "dashboard_view.json")),
        _ms(safe_mtime_ns(run.path / "run_history.json")),
        _ms(safe_mtime_ns(run.path / "planned_trials.json")),
        _ms(safe_mtime_ns(run.path / "experiment_record.json")),
        _ms(safe_mtime_ns(run.path / "artifact_index.json")),
    )
    return max(
        _started_at_ms(run.started_at),
        _started_at_ms(run.completed_at),
        int(run.last_active_ts_ms or 0),
        int(run.duration_ms or 0),
        file_sig_ms,
    )


def build_run_trials_delta(
    run_dir: Path,
    *,
    root: Path,
    since_ms: int = 0,
    trial_url: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Build the ``/api/run/<id>/trials`` payload.

    A trial is included iff its mtime signature exceeds ``since_ms``,
    or if ``since_ms == 0`` (full refresh / first load).
    """
    now_ms = int(time.time() * 1000)
    dashboard = load_dashboard(run_dir)
    run_status = str(dashboard.get("status") or "")
    # Stash the dashboard's per-trial sub-records for combine.
    trial_map: dict[str, dict[str, Any]] = {}
    for agent_data in dashboard.get("agents", {}).values():
        if not isinstance(agent_data, dict):
            continue
        for t in agent_data.get("trials", []) or []:
            if isinstance(t, dict) and t.get("trial_id"):
                trial_map[t["trial_id"]] = t
    # A settled run's dashboard.json is the final projection of every trial, so
    # the delta is built from it without walking/stat-ing trial dirs each poll
    # (the per-trial 5-stat signature × N-trial NAS cost) and WITHOUT loading the
    # planned/canonical records (``load_planned_trial_records`` parses the whole
    # ``artifact_index.json`` — hundreds of KB — for canonical runs). Live/
    # in-progress runs use ``find_trial_dirs`` (same source as the page-render
    # banner) so totals match first load and live trials keep updating;
    # ``scan_run_signals`` would miss trials that failed before the agent ran
    # (target_unavailable: meta.json terminal, no task_output).
    # A resumed run's dashboard.json is frozen at its previous completion while
    # the canonical store + per-trial records move on, so its rows go stale.
    # Treat such a run as live (authoritative per-trial scan), matching the
    # page-render fast-path guard in app._trial_rows_for_run.
    settled = (
        bool(dashboard.get("completed_at"))
        and bool(trial_map)
        and not dashboard_projection_is_stale(run_dir)
    )
    planned = [] if settled else load_planned_trial_records(run_dir)
    planned_map = {str(item.get("trial_id")): item for item in planned if item.get("trial_id")}
    planned_sig_ns = safe_mtime_ns(run_dir / "planned_trials.json")
    trial_dirs = [] if settled else find_trial_dirs(run_dir)
    dashboard_sig_ms = _ms(safe_mtime_ns(run_dir / "dashboard.json")) if settled else 0
    # When the run has no live activity, a trial record stuck at "running" (the
    # loop was Ctrl+C-killed before finalizing it) must not surface as Running.
    # Skip the scan on the settled fast path, where nothing is live anyway.
    run_live = True if settled else run_has_live_activity(run_dir)

    out_trials: list[dict[str, Any]] = []
    max_sig_ms = since_ms
    # All status_kind buckets are listed here so the banner can show
    # every trial in the run and the parts sum to ``total``. The five
    # countable kinds (running + completed + live_success + warnings +
    # failed) MUST partition the total — the run.html banner relies on
    # this for its arithmetic display.
    counts = {
        "running": 0,
        "completed": 0,
        "live_success": 0,
        "warnings": 0,
        "failed": 0,
        "other": 0,
        "total": 0,
    }

    def add_counts(info: dict[str, Any]) -> None:
        counts["total"] += 1
        kind = info.get("status_kind")
        if info.get("running"):
            counts["running"] += 1
        elif kind == "success":
            counts["completed"] += 1
        elif kind == "live_success":
            counts["live_success"] += 1
        elif kind == "warning":
            # Budget-bound stops: max-rounds, token/cost budgets,
            # execution_timeout, tool_limit, user_interrupted, and Exhausted
            # (clean exit without live-check). All folded into one warnings
            # bucket so the banner's arithmetic stays simple.
            counts["warnings"] += 1
        elif kind == "error":
            counts["failed"] += 1
        else:
            counts["other"] += 1

    seen: set[str] = set()
    canonical_planned = any(_is_canonical_trial_record(item) for item in planned)
    if canonical_planned:
        for item in planned:
            tid = str(item.get("trial_id") or "")
            if not tid:
                continue
            trial_dir = planned_trial_dir(run_dir, tid)
            info = pending_trial_summary(item)
            # Canonical status stays authoritative, but a trial that
            # materialized artifacts on disk is still a browsable blue link.
            # Mirrors the page-render path (_trial_rows_for_run).
            info["has_artifacts"] = is_trial_dir(trial_dir)
            if _canonical_row_has_activity_artifacts(item, trial_dir):
                activity = load_trial_activity(trial_dir)
                if activity:
                    info.update(activity)
            _demote_info_if_stale(info, run_live)
            sig_ms = _ms(_planned_record_signature_ns(item, planned_sig_ns))
            max_sig_ms = max(max_sig_ms, sig_ms)
            add_counts(info)
            seen.add(tid)
            if since_ms and sig_ms <= since_ms and not info.get("running"):
                continue
            out_trials.append(_trial_to_json(
                trial_dir,
                info,
                signature_ms=sig_ms,
                root=root,
                display_id=tid,
                trial_url=trial_url,
            ))

    if settled:
        # Settled fast path: one signature (dashboard mtime) for every row, rows
        # straight from the projection — no per-trial filesystem access.
        for tid, entry in trial_map.items():
            if tid in seen:
                continue
            seed = {
                **planned_map.get(tid, {}),
                **entry,
            }
            info = trial_summary_from_dashboard_entry(seed, run_status=run_status)
            _demote_info_if_stale(info, run_live)
            max_sig_ms = max(max_sig_ms, dashboard_sig_ms)
            add_counts(info)
            seen.add(tid)
            if since_ms and dashboard_sig_ms <= since_ms and not info.get("running"):
                continue
            out_trials.append(_trial_to_json(
                planned_trial_dir(run_dir, tid), info,
                signature_ms=dashboard_sig_ms, root=root, display_id=tid,
                trial_url=trial_url,
            ))

    # Live/in-progress runs only — ``trial_dirs`` is empty when settled.
    for trial_dir in trial_dirs:
        sig_ns = max(trial_summary_signature(trial_dir))
        sig_ms = _ms(sig_ns)
        max_sig_ms = max(max_sig_ms, sig_ms)
        tid = _trial_display_id(trial_dir, run_dir)
        if tid in seen:
            continue
        seed = {
            **planned_map.get(tid, {}),
            **trial_map.get(tid, {}),
        }
        info = load_trial_summary_cached(
            trial_dir,
            seed,
            run_status=run_status,
        )
        _demote_info_if_stale(info, run_live)
        add_counts(info)
        seen.add(tid)

        if since_ms and sig_ms <= since_ms and not info.get("running"):
            # Settled trial that hasn't moved — skip the payload.
            continue
        out_trials.append(_trial_to_json(
            trial_dir, info, signature_ms=sig_ms, root=root, display_id=tid,
            trial_url=trial_url,
        ))
    for item in planned:
        tid = str(item.get("trial_id") or "")
        if not tid or tid in seen:
            continue
        info = pending_trial_summary(item)
        _demote_info_if_stale(info, run_live)
        sig_ms = _ms(_planned_record_signature_ns(item, planned_sig_ns))
        max_sig_ms = max(max_sig_ms, sig_ms)
        add_counts(info)
        if since_ms and sig_ms <= since_ms:
            continue
        out_trials.append(_trial_to_json(
            planned_trial_dir(run_dir, tid),
            info,
            signature_ms=sig_ms,
            root=root,
            display_id=tid,
            trial_url=trial_url,
        ))

    return {
        "now_ms": now_ms,
        "since_ms": since_ms,
        "max_signature_ms": max_sig_ms,
        "run_status": run_status or ("running" if counts["running"] > 0 else ""),
        "summary": counts,
        "trials": out_trials,
    }


def _is_canonical_trial_record(item: dict[str, Any]) -> bool:
    """Return whether a row came from a canonical TrialRecord projection."""

    return (
        str(item.get("schema_version") or "") == "trial_record.v1"
        or bool(item.get("record_ref"))
    )


def _canonical_row_has_activity_artifacts(item: dict[str, Any], trial_dir: Path) -> bool:
    """Return whether a canonical row needs activity hydration.

    Most canonical table rows already carry enough record information, and
    reading ``ArtifactIndex`` again would waste work. Activity hydration is
    needed only when live ``proxy/progress.json`` exists.
    """

    del item
    return (trial_dir / "proxy" / "progress.json").is_file()


def _planned_record_signature_ns(item: dict[str, Any], default_ns: int) -> int:
    """Return the mtime signature for a planned/canonical trial row."""

    try:
        return int(item.get("_signature_ns") or default_ns or 0)
    except (TypeError, ValueError):
        return default_ns


def _trial_display_id(trial_dir: Path, run_dir: Path) -> str:
    """Stable display id: relative path under ``<run>/trials/``,
    with consecutive duplicate path components collapsed."""
    trials_root = run_dir / "trials"
    try:
        rel = trial_dir.relative_to(trials_root)
    except ValueError:
        return trial_dir.name
    parts = list(rel.parts)
    deduped: list[str] = []
    for part in parts:
        if deduped and part == deduped[-1]:
            continue
        if deduped and (deduped[-1].endswith("-" + part) or deduped[-1].endswith("_" + part)):
            continue
        deduped.append(part)
    return "/".join(deduped) if deduped else trial_dir.name


def _trial_to_json(
    trial_dir: Path,
    info: dict[str, Any],
    *,
    signature_ms: int,
    root: Path,
    display_id: str,
    trial_url: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    progress = info.get("progress") if isinstance(info.get("progress"), dict) else {}
    usage = info.get("usage") if isinstance(info.get("usage"), dict) else {}
    scores = info.get("scores") if isinstance(info.get("scores"), dict) else {}
    output = str(info.get("output") or "")
    if len(output) > 200:
        output = output[:200]

    # Hint: a brief result-relevant line shown as the badge tooltip.
    audit_hint = ""
    if info.get("status_kind") == "live_success":
        verdict = info.get("live_success_verdict") or {}
        evidence = verdict.get("evidence") if isinstance(verdict, dict) else None
        check_done = evidence.get("check_done") if isinstance(evidence, dict) else None
        verdict_msg = check_done.get("message") if isinstance(check_done, dict) else None
        audit_hint = (
            f"Target validator: {verdict_msg}. Verify the agent's commands."
            if verdict_msg else
            "Stopped on a positive target-check verdict. Open the trial for details."
        )
    elif (info.get("status_label") or "") == "Max rounds":
        audit_hint = "Agent exited cleanly but never satisfied the live check."

    usage_payload = {
        "input_tokens": int_or_zero(usage.get("input_tokens")),
        "output_tokens": int_or_zero(usage.get("output_tokens")),
        "reasoning_tokens": int_or_zero(usage.get("reasoning_tokens")),
        "num_requests": int_or_zero(usage.get("num_requests")),
    }
    cost_usd = _as_float(usage.get("cost_usd"))
    if cost_usd > 0:
        usage_payload["cost_usd"] = cost_usd

    return {
        "id": display_id,
        "trial_id_full": info.get("trial_id") or display_id,
        "encoded_path": _encode_path(trial_dir, root),
        # Canonical /<benchmark>/<model>/<run_id>/<trial_rel> link so the
        # client-rendered rows match the server-rendered ones; the run.html
        # JS uses ``url`` when present and falls back to /trial/<encoded_path>.
        "url": trial_url(trial_dir) if trial_url else None,
        "has_artifacts": bool(info.get("has_artifacts", True)),
        "audit_hint": audit_hint,
        "trial_index": info.get("trial_index"),
        "status_kind": info.get("status_kind", "pending"),
        "status_label": info.get("status_label", "Pending"),
        "status_detail": info.get("status_detail", ""),
        "running": bool(info.get("running")),
        "duration_ms": int_or_zero(info.get("duration_ms")),
        "exit_code": info.get("exit_code"),
        "scores": {k: _as_number(v) for k, v in scores.items()},
        "tags": list(info.get("tags") or []),
        "progress": {
            "total": int_or_zero(
                progress.get("total_requests")
                if progress.get("total_requests") is not None
                else progress.get("successful_requests", progress.get("success"))
            ),
            "successful": int_or_zero(
                progress.get("successful_requests")
                if progress.get("successful_requests") is not None
                else progress.get("success")
            ),
            "errors": int_or_zero(progress.get("errors")),
            "last_ts_ms": int_or_zero(progress.get("last_ts_ms")),
            "last_status": str(progress.get("last_status") or ""),
        } if progress else None,
        "usage": usage_payload,
        "output_preview": output,
        "signature_ms": signature_ms,
    }


def _as_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _started_at_ms(text: str) -> int:
    if not text:
        return 0
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0
