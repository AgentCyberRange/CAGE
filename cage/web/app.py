"""Flask application for the Cage web inspector.

Usage (via CLI)::

    cage inspect [path]          # scan path for .cage_runs
    cage inspect --port 8080     # custom port
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from flask import (
    Blueprint,
    Flask,
    abort,
    current_app,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
)

from cage.artifacts.dashboard import load_dashboard_view
from cage.artifacts.reader import ExperimentArtifactReader
from cage.config import WebInspectorAuthConfig, WebInspectorUIConfig
from cage.contracts.coerce import int_or_zero
from cage.contracts.duration import split_duration_hms
from cage.contracts.scoring import extract_numeric_score_value
from cage.gc.summary import summarize_resource_ledger

from .cache import (
    dashboard_projection_is_stale,
    run_has_live_activity,
    safe_mtime_ns,
)
from .data import (
    _format_file_size,
    _iter_cage_runs_dirs,
    build_trial_file_tree,
    build_trial_termination,
    find_trial_dirs,
    group_runs,
    is_indexed_trial_artifact_path,
    is_known_trial_path,
    is_trial_dir,
    list_benchmarks,
    load_dashboard,
    load_live_check_evidence,
    load_planned_trial_records,
    load_resume_attempts,
    load_run_history,
    load_trial,
    load_trial_activity,
    load_trial_step_context,
    load_trial_summary,
    load_trial_summary_cached,
    parse_trial_trajectory,
    pending_trial_summary,
    planned_trial_dir,
    scan_runs,
    scan_runs_for_project,
    trial_summary_from_dashboard_entry,
    trial_summary_signature,
)
from .data import (
    load_preflight_summary as _load_preflight_summary,
)
from .delta import build_run_trials_delta, build_runs_delta, relative_to_root

inspector_bp = Blueprint("inspector", __name__)


@inspector_bp.before_app_request
def _require_inspector_auth() -> tuple[str, int] | None:
    cfg: WebInspectorAuthConfig = current_app.config["CAGE_INSPECTOR_AUTH"]
    if not cfg.enabled:
        return None
    if _inspector_request_authorized(cfg):
        if request.args.get("token"):
            g.cage_set_inspector_cookie = True
        return None
    return ("Cage Inspector authentication required\n", 401)


@inspector_bp.after_app_request
def _persist_query_token(response):
    cfg: WebInspectorAuthConfig = current_app.config["CAGE_INSPECTOR_AUTH"]
    if cfg.enabled and getattr(g, "cage_set_inspector_cookie", False):
        response.set_cookie(
            "cage_inspector_token",
            cfg.token,
            httponly=True,
            samesite="Lax",
        )
    return response


@inspector_bp.app_template_filter("fmt_tokens")
def fmt_tokens(n: int | float) -> str:
    if not n:
        return "0"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@inspector_bp.app_template_filter("fmt_duration")
def fmt_duration(ms: int | float) -> str:
    if not ms:
        return "-"
    hours, minutes, seconds = split_duration_hms(int(ms) // 1000)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


@inspector_bp.app_template_filter("fmt_age")
def fmt_age(ts_ms: int | float) -> str:
    if not ts_ms:
        return "-"
    delta_ms = max(0, int(time.time() * 1000) - int(ts_ms))
    s = delta_ms // 1000
    if s < 5:
        return "just now"
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m ago"
    return f"{s // 86400}d ago"


@inspector_bp.app_template_filter("encode_path")
def encode_path(p: Path | str) -> str:
    rel = relative_to_root(Path(p), current_app.config["ROOT_DIR"])
    return base64.urlsafe_b64encode(rel.encode()).decode()


def _decode_path(encoded: str) -> Path:
    rel = base64.urlsafe_b64decode(encoded.encode()).decode()
    rel_path = Path(rel)
    # Sandbox lexically (no ``..``/absolute) instead of resolving, so symlinked
    # run dirs under ``<root>/.cage_runs`` (which resolve outside root) still
    # load. The symlink is followed at file-open time.
    if rel_path.is_absolute() or ".." in rel_path.parts:
        abort(403)
    return current_app.config["ROOT_DIR"] / rel_path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _inspector_request_authorized(cfg: WebInspectorAuthConfig) -> bool:
    candidates: list[str] = []
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        candidates.append(auth_header[7:].strip())
    query_token = request.args.get("token")
    if query_token:
        candidates.append(query_token)
    cookie_token = request.cookies.get("cage_inspector_token")
    if cookie_token:
        candidates.append(cookie_token)
    return any(secrets.compare_digest(candidate, cfg.token) for candidate in candidates)


def _run_lookup_indexes() -> tuple[
    dict[tuple[str, str], Path],
    dict[tuple[str, str], Path],
    dict[tuple[str, str, str], Path],
]:
    now = time.monotonic()
    cache = current_app.config["CAGE_RUN_LOOKUP_CACHE"]
    if cache["expires_at"] > now:
        return cache["by_agent"], cache["by_project"], cache["by_bmr"]

    agent_candidates: dict[tuple[str, str], list[Path]] = {}
    project_candidates: dict[tuple[str, str], list[Path]] = {}
    bmr_candidates: dict[tuple[str, str, str], list[Path]] = {}
    for cage_runs_dir in _iter_cage_runs_dirs(current_app.config["ROOT_DIR"]):
        project = cage_runs_dir.parent.name
        try:
            agent_dirs = [p for p in cage_runs_dir.iterdir() if p.is_dir()]
        except OSError:
            continue
        for agent_dir in agent_dirs:
            _, model, _ = _parse_agent_label_for_view(agent_dir.name)
            try:
                run_dirs = [p for p in agent_dir.iterdir() if p.is_dir()]
            except OSError:
                continue
            for run_dir in run_dirs:
                agent_candidates.setdefault((agent_dir.name, run_dir.name), []).append(run_dir)
                project_candidates.setdefault((project, run_dir.name), []).append(run_dir)
                if model:
                    bmr_candidates.setdefault((project, model, run_dir.name), []).append(run_dir)

    by_agent = {
        key: matches[0]
        for key, matches in agent_candidates.items()
        if len(matches) == 1
    }
    by_project = {
        key: matches[0]
        for key, matches in project_candidates.items()
        if len(matches) == 1
    }
    by_bmr = {
        key: matches[0]
        for key, matches in bmr_candidates.items()
        if len(matches) == 1
    }
    cache.update({
        "expires_at": now + 2.0,
        "by_agent": by_agent,
        "by_project": by_project,
        "by_bmr": by_bmr,
    })
    return by_agent, by_project, by_bmr


def _find_run_dir(agent_label: str, run_id: str) -> Path | None:
    """Resolve ``/run/<agent_label>/<run_id>`` to a real run directory.

    Searches every ``.cage_runs/`` under root (the same set
    ``scan_runs`` walks; the result is cached briefly). Returns a
    directory only when the agent/run pair is unique, so readable URLs
    never silently pick the first match in multi-root scans. Path
    traversal is blocked up front by rejecting any
    segment containing ``/`` or equal to ``.``/``..``, which means
    the candidate is necessarily a 2-deep child of a ``.cage_runs/``
    dir already known to be under root. ``is_dir()`` alone is the
    only syscall in the success path — keep it that way; this is
    called once per trial-detail / run-detail URL hit.
    """
    if "/" in agent_label or "/" in run_id:
        return None
    if agent_label in {"", ".", ".."} or run_id in {"", ".", ".."}:
        return None
    by_agent, _, _ = _run_lookup_indexes()
    return by_agent.get((agent_label, run_id))


def _project_slug_for_run_dir(run_dir: Path) -> str:
    for parent in run_dir.parents:
        if parent.name == ".cage_runs":
            return parent.parent.name
    return ""


def _find_project_run_dir(project: str, run_id: str) -> Path | None:
    if "/" in project or "/" in run_id:
        return None
    _, by_project, _ = _run_lookup_indexes()
    return by_project.get((project, run_id))


def _find_benchmark_model_run(benchmark: str, model: str, run_id: str) -> Path | None:
    """Resolve the canonical ``/<benchmark>/<model>/<run_id>`` URL to a run dir.

    ``benchmark`` is the project (``examples/<benchmark>/``), ``model`` is the
    model segment of the agent label. Returns a dir only when the triple is
    unique, so a model shared by two agents under one benchmark never silently
    resolves to the wrong run.
    """
    if "/" in benchmark or "/" in model or "/" in run_id:
        return None
    if benchmark in {"", ".", ".."} or run_id in {"", ".", ".."}:
        return None
    _, _, by_bmr = _run_lookup_indexes()
    return by_bmr.get((benchmark, model, run_id))


def _build_run_lookup_from_runs(
    runs: Iterable[Any],
) -> tuple[
    dict[tuple[str, str], Path],
    dict[tuple[str, str], Path],
    dict[tuple[str, str, str], Path],
    set[str],
]:
    agent_candidates: dict[tuple[str, str], list[str]] = {}
    candidates: dict[tuple[str, str], list[str]] = {}
    bmr_candidates: dict[tuple[str, str, str], list[str]] = {}
    path_by_text: dict[str, Path] = {}
    for run in runs:
        project = str(getattr(run, "project", "") or "")
        run_id = str(getattr(run, "run_id", "") or "")
        agent_label = str(getattr(run, "agent_label", "") or "")
        path = getattr(run, "path", None)
        if not run_id or path is None:
            continue
        run_path = Path(path)
        path_text = str(run_path)
        path_by_text[path_text] = run_path
        if agent_label:
            agent_candidates.setdefault((agent_label, run_id), []).append(path_text)
        if project:
            candidates.setdefault((project, run_id), []).append(path_text)
        _, model, _ = _parse_agent_label_for_view(agent_label)
        if project and model:
            bmr_candidates.setdefault((project, model, run_id), []).append(path_text)
    def _pick(paths: list[str], run_id: str) -> str | None:
        """Resolve a lookup key to a single run path.

        A run and its ``.previous_<ts>`` / ``.before_resume_<ts>`` archive both
        report the same dashboard ``run_id`` (the archive is a copy), so they
        collide on every (…, run_id) key. The *live* run is the one whose
        directory name IS the run_id; archives carry a suffix. Prefer it so the
        canonical ``/<benchmark>/<model>/<run_id>`` URL — which ``cage run``
        builds from the live dir name — still resolves instead of 404-ing
        whenever a page render warms this index.
        """
        if len(paths) == 1:
            return paths[0]
        live = [p for p in paths if Path(p).name == run_id]
        return live[0] if len(live) == 1 else None

    by_agent = {
        key: path_by_text[picked]
        for key, paths in agent_candidates.items()
        if (picked := _pick(paths, key[1])) is not None
    }
    semantic_paths = {
        picked
        for key, paths in candidates.items()
        if (picked := _pick(paths, key[1])) is not None
    }
    by_project = {
        key: path_by_text[picked]
        for key, paths in candidates.items()
        if (picked := _pick(paths, key[1])) is not None
    }
    by_bmr = {
        key: path_by_text[picked]
        for key, paths in bmr_candidates.items()
        if (picked := _pick(paths, key[2])) is not None
    }
    return by_agent, by_project, by_bmr, semantic_paths


def _warm_run_lookup_indexes_from_runs(runs: Iterable[Any]) -> set[str]:
    by_agent, by_project, by_bmr, semantic_paths = _build_run_lookup_from_runs(runs)
    current_app.config["CAGE_RUN_LOOKUP_CACHE"].update({
        "expires_at": time.monotonic() + 60.0,
        "by_agent": by_agent,
        "by_project": by_project,
        "by_bmr": by_bmr,
    })
    return semantic_paths


def _install_request_run_url_index(runs: Iterable[Any]) -> None:
    g._cage_semantic_run_paths = _warm_run_lookup_indexes_from_runs(runs)


def _request_project_run_url_allowed(path: Path) -> bool | None:
    try:
        semantic_paths = getattr(g, "_cage_semantic_run_paths", None)
    except RuntimeError:
        return None
    if semantic_paths is None:
        return None
    return str(path) in semantic_paths


def _run_url_for(run_or_path) -> str:
    """Build the canonical ``/<benchmark>/<model>/<run_id>`` URL.

    ``benchmark`` is the project dir (``examples/<benchmark>/``), ``model`` is
    the model segment of the agent label. Accepts a ``RunInfo`` (with ``.path``)
    or a run-directory ``Path``. Falls back to base64 ``/run/<encoded>`` when the
    path is not a canonical ``<benchmark>/.cage_runs/<agent_label>/<run_id>`` run
    (e.g. out-of-tree, or an agent label with no model segment), so such runs
    still render a working link.
    """
    path = Path(getattr(run_or_path, "path", run_or_path))
    agent_label = path.parent.name
    run_id = path.name
    project = _project_slug_for_run_dir(path)
    if project and path.parent.parent.name == ".cage_runs" and "/" not in agent_label:
        _, model, _ = _parse_agent_label_for_view(agent_label)
        if model:
            # A unique (project, run_id) implies a unique (project, model,
            # run_id), so the request-warmed project-uniqueness set is a valid
            # fast path that avoids a per-row index lookup on the index page.
            request_allowed = _request_project_run_url_allowed(path)
            if request_allowed is True or (
                request_allowed is None
                and _find_benchmark_model_run(project, model, run_id) == path
            ):
                return f"/{quote(project)}/{quote(model)}/{quote(run_id)}"
    return f"/run/{encode_path(path)}"


def _trial_url_for(trial_dir, run_dir) -> str:
    """Build a readable ``/trial/<agent_label>/<run_id>/<trial_rel>`` URL.

    Falls back to ``/trial/<encoded>`` when either dir doesn't fit the
    canonical ``<.cage_runs>/<agent_label>/<run_id>/trials/...`` layout
    (e.g. legacy flat runs without a ``trials/`` subdir).

    Hot path: called once per trial row on the run detail page; for a
    100-trial run that is 100 calls per render. The implementation is
    intentionally I/O-free — paths from ``find_trial_dirs`` and
    ``_resolve_run_dir`` are already canonical so the string-only
    ``relative_to`` is correct and cheap.
    """
    trial_dir = Path(trial_dir)
    run_dir = Path(run_dir)
    agent_label = run_dir.parent.name
    run_id = run_dir.name
    if run_dir.parent.parent.name != ".cage_runs" or "/" in agent_label:
        return f"/trial/{encode_path(trial_dir)}"
    trials_root = run_dir / "trials"
    try:
        trial_rel = trial_dir.relative_to(trials_root).as_posix()
    except ValueError:
        return f"/trial/{encode_path(trial_dir)}"
    project = _project_slug_for_run_dir(run_dir)
    _, model, _ = _parse_agent_label_for_view(agent_label)
    if project and model and _find_benchmark_model_run(project, model, run_id) == run_dir:
        trial_part = quote(trial_rel, safe="/")
        return f"/{quote(project)}/{quote(model)}/{quote(run_id)}/{trial_part}"
    return f"/trial/{encode_path(trial_dir)}"


def _dashboard_url_for(run_or_path) -> str:
    """Canonical ``/<benchmark>/<model>/<run_id>/dashboard`` URL."""
    base = _run_url_for(run_or_path)
    return f"{base}/dashboard"


def _canonical_run_url_or_none(run_dir: Path) -> str | None:
    """Canonical ``/<benchmark>/<model>/<run_id>`` URL, or ``None`` out-of-tree.

    ``_run_url_for`` returns the encoded ``/run/<encoded>`` form only when a run
    has no canonical benchmark/model/run address; that is the signal that there
    is nothing to redirect a legacy URL to.
    """
    url = _run_url_for(run_dir)
    return None if url.startswith("/run/") else url


def _redirect_or_render_run(run_dir: Path):
    canonical = _canonical_run_url_or_none(run_dir)
    if canonical is not None:
        return redirect(canonical, code=301)
    return _render_run_detail(run_dir)


def _redirect_or_render_dashboard(run_dir: Path):
    canonical = _canonical_run_url_or_none(run_dir)
    if canonical is not None:
        return redirect(f"{canonical}/dashboard", code=301)
    return _render_dashboard(run_dir)


def _redirect_or_render_trial(trial_dir: Path, run_dir: Path):
    canonical = _trial_url_for(trial_dir, run_dir)
    if not canonical.startswith("/trial/"):
        return redirect(canonical, code=301)
    return _render_trial_detail(trial_dir)


def _trial_display_id(trial_dir: Path, run_dir: Path) -> str:
    """Stable display id: relative path under ``<run>/trials/``.

    Falls back to ``trial_dir.name`` for legacy layouts where the
    trial dir doesn't live directly under ``trials/``. Strips
    consecutive duplicate components (a benchmark may nest the
    challenge dir + a same-named variant dir, producing
    ``<chal>-one_day/one_day/pass_1`` — we collapse the duplicate).
    """
    trials_root = run_dir / "trials"
    try:
        rel = trial_dir.relative_to(trials_root)
    except ValueError:
        return trial_dir.name
    parts = list(rel.parts)
    # Collapse consecutive duplicates like a
    # ``challenge/variant/pass_k`` layout where challenge encodes variant.
    deduped: list[str] = []
    for part in parts:
        if deduped and part == deduped[-1]:
            continue
        if deduped and (deduped[-1].endswith("-" + part) or deduped[-1].endswith("_" + part)):
            # parent already encodes this part (e.g. "cvb-X-one_day" + "one_day")
            continue
        deduped.append(part)
    return "/".join(deduped) if deduped else trial_dir.name


def _resolve_run_dir(trial_dir: Path) -> Path:
    """Find the run dir for a trial.

    Trial dirs always live under ``<run>/trials/[<sample>/...]<id>``,
    regardless of flat / nested / passk layout. Walk up from
    ``trial_dir`` and return the parent of the first ancestor named
    ``trials``. This is invariant to dashboard.json existence (which
    the orchestrator only writes at run end), unlike the previous
    heuristic that overshot to ``<agent_label>`` for in-progress runs.
    """
    p = trial_dir
    while p.parent != p:
        if p.name == "trials":
            return p.parent
        p = p.parent
    # Fallback for malformed paths: keep the old two-up behaviour.
    return trial_dir.parent.parent


def _aggregate_agent_counts(trial_infos: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Per-agent header counts derived from on-disk trial classification.

    ``status_kind`` is the canonical web classifier (set by
    ``_classify_trial_status``); it's also what the live banner uses,
    so the top header and the banner now agree. Definitions:

    * ``completed`` — success or live-check-success outcomes.
    * ``failed`` — genuine errors (model/infra/target/agent crash).
    * ``total`` — every trial dir we can see on disk for this run.

    Trials that are still running or in a warning state (timed out,
    exhausted, max-rounds) intentionally fall outside both buckets;
    the live banner exposes them separately so the user can drill in.
    """
    total = running = completed = live_success = warnings = failed = other = 0
    for info in trial_infos:
        total += 1
        kind = str(info.get("status_kind") or "").lower()
        if info.get("running"):
            running += 1
        elif kind == "success":
            completed += 1
        elif kind == "live_success":
            live_success += 1
        elif kind == "warning":
            warnings += 1
        elif kind == "error":
            failed += 1
        else:
            other += 1
    return {
        "total": total,
        "running": running,
        "completed": completed,
        "live_success": live_success,
        "warnings": warnings,
        "failed": failed,
        "other": other,
    }


def _sort_trials_for_display(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Natural trial order — the order trials were planned/submitted, captured in
    # ``sort_index`` (canonical record-ref order for current runs, dashboard order
    # for settled ones). Rows read top-to-bottom in that one stable order
    # regardless of live status; running trials are NOT floated to the top, which
    # previously reshuffled the table as trials started and finished.
    for index, trial in enumerate(trials):
        trial.setdefault("sort_index", index)
    return sorted(trials, key=lambda t: int(t.get("sort_index") or 0))


def _trial_map_from_dashboard(dashboard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trial_map: dict[str, dict[str, Any]] = {}
    for agent_data in dashboard.get("agents", {}).values():
        if not isinstance(agent_data, dict):
            continue
        for trial in agent_data.get("trials", []) or []:
            if isinstance(trial, dict) and trial.get("trial_id"):
                trial_map[str(trial["trial_id"])] = trial
    return trial_map


def _is_canonical_trial_record(item: dict[str, Any]) -> bool:
    return (
        str(item.get("schema_version") or "") == "trial_record.v1"
        or bool(item.get("record_ref"))
    )


def _settled_trial_rows(
    run_dir: Path,
    trial_map: dict[str, dict[str, Any]],
    run_status: str,
) -> list[dict[str, Any]]:
    """Overview rows for a settled run, straight from the dashboard projection.

    No per-trial filesystem walk and no planned/canonical-record load: a settled
    run's dashboard.json already lists every trial with full status/scores, so
    the projection is the complete, authoritative source for the overview. (For
    canonical runs ``load_planned_trial_records`` parses the run's whole
    ``artifact_index.json`` — hundreds of KB — which is exactly the per-render
    cost this fast path exists to avoid.)
    """
    rows: list[dict[str, Any]] = []
    for trial_id, entry in trial_map.items():
        rows.append({
            "dir": planned_trial_dir(run_dir, trial_id),
            "info": trial_summary_from_dashboard_entry(entry, run_status=run_status),
            "id": trial_id,
            "sort_index": len(rows),
        })
    return _sort_trials_for_display(rows)


def _trial_rows_for_run(
    run_dir: Path,
    dashboard: dict[str, Any],
) -> list[dict[str, Any]]:
    trial_map = _trial_map_from_dashboard(dashboard)
    run_status = str(dashboard.get("status", "") or "")
    # Fast path: a settled run's dashboard.json is the final projection of every
    # trial (status/scores/duration/termination/usage), so render rows straight
    # from it — no per-trial filesystem walk (the 100×-trial NAS cost). Live or
    # in-progress runs (no completed_at) fall through to the authoritative
    # per-trial scan below, which keeps live updates accurate.
    #
    # But a RESUMED run keeps a dashboard.json frozen at its previous completion
    # (completed_at still set) while the canonical store and per-trial records
    # move on — so the projection would show stale "cancelled/pending" rows (or
    # omit re-run trials entirely). Skip the fast path when the projection is
    # stale and let the authoritative scan below report the real outcomes.
    if (
        dashboard.get("completed_at")
        and trial_map
        and not dashboard_projection_is_stale(run_dir)
    ):
        return _settled_trial_rows(run_dir, trial_map, run_status)
    planned = load_planned_trial_records(run_dir)
    planned_map = {str(item.get("trial_id")): item for item in planned if item.get("trial_id")}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    canonical_planned = any(_is_canonical_trial_record(item) for item in planned)
    if canonical_planned:
        for item in planned:
            trial_id = str(item.get("trial_id") or "")
            if not trial_id:
                continue
            trial_dir = planned_trial_dir(run_dir, trial_id)
            info = pending_trial_summary(item)
            # Canonical status stays authoritative (the record wins over a stale
            # legacy meta.json), but the row is still a browsable blue link once
            # the trial has materialized artifacts on disk. Without this, every
            # trial in a canonical run renders gray even after producing a full
            # trial dir — the planned stub hardcodes has_artifacts=False.
            info["has_artifacts"] = is_trial_dir(trial_dir)
            rows.append({
                "dir": trial_dir,
                "info": info,
                "id": trial_id,
                "sort_index": len(rows),
            })
            seen.add(trial_id)
    for trial_dir in find_trial_dirs(run_dir):
        trial_id = _trial_display_id(trial_dir, run_dir)
        if trial_id in seen:
            continue
        seed = {
            **planned_map.get(trial_id, {}),
            **trial_map.get(trial_id, {}),
        }
        info = load_trial_summary_cached(trial_dir, seed, run_status=run_status)
        rows.append({
            "dir": trial_dir,
            "info": info,
            "id": trial_id,
            "sort_index": len(rows),
        })
        seen.add(trial_id)
    for item in planned:
        trial_id = str(item.get("trial_id") or "")
        if not trial_id or trial_id in seen:
            continue
        rows.append({
            "dir": planned_trial_dir(run_dir, trial_id),
            "info": pending_trial_summary(item),
            "id": trial_id,
            "sort_index": len(rows),
        })
    rows = _sort_trials_for_display(rows)
    _demote_stale_running_rows(rows, run_dir)
    return rows


def _demote_stale_running_rows(rows: list[dict[str, Any]], run_dir: Path) -> None:
    """Rewrite phantom "running" rows when the run has no live activity.

    A trial Ctrl+C-killed mid-flight keeps a record stuck at ``status="running"``
    (the loop died before finalizing it). Once nothing is actually live, a
    "Running" chip is plain wrong — present those trials as Interrupted instead.
    Genuinely live runs keep their running rows.
    """
    if run_has_live_activity(run_dir):
        return
    for row in rows:
        info = row.get("info") or {}
        if info.get("running") or info.get("status_kind") == "running":
            info["running"] = False
            info["status_kind"] = "error"
            info["status_label"] = "Interrupted"
            info["status_detail"] = "Run stopped while this trial was still running"


def _run_lifecycle_label(dashboard: dict[str, Any], has_running: bool) -> str:
    if has_running:
        return "Running"
    raw = str(dashboard.get("status") or "").strip().lower()
    if raw in {"running", "active", "in_progress"}:
        return "Recently active"
    if raw == "interrupted":
        return "Stopped"
    if raw == "failed":
        return "Failed"
    if raw == "cancelled":
        return "Cancelled"
    if dashboard.get("completed_at") or raw == "completed":
        return "Finished"
    return "Pending"


def _run_action_label(counts: dict[str, int]) -> str:
    if counts.get("running"):
        return "Monitor active trials"
    if counts.get("live_success") or counts.get("warnings"):
        return "Check target-passed or stopped trials"
    if counts.get("failed"):
        return "Investigate failures"
    if counts.get("completed") and counts.get("completed") == counts.get("total"):
        return "No action needed"
    return "Check incomplete trials"


def _run_health_sentence(counts: dict[str, int]) -> str:
    total = counts.get("total", 0)
    if not total:
        return "No trial artifacts found yet."
    parts = [
        f"{counts.get('running', 0)} running",
        f"{counts.get('completed', 0)} completed",
        f"{counts.get('live_success', 0)} target passed",
        f"{counts.get('warnings', 0)} stopped",
        f"{counts.get('failed', 0)} failed",
    ]
    if counts.get("other"):
        parts.append(f"{counts.get('other', 0)} pending")
    return f"{' / '.join(parts)} out of {total} trials."


def _run_activity_warnings(dashboard: dict[str, Any], counts: dict[str, int]) -> list[str]:
    raw = str(dashboard.get("status") or "").strip().lower()
    if raw in {"running", "active", "in_progress"} and not counts.get("running"):
        return ["Run metadata still says active. No active trial process found."]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _present_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _load_run_config_yaml(run_dir: Path) -> dict[str, Any]:
    for name in ("project.yml", "config.yaml", "config.yml"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}
    return {}


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _format_duration(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds <= 0:
        return "unlimited"
    if seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    if seconds % 60 == 0:
        return f"{int(seconds // 60)}m"
    return f"{seconds:g}s"


def _format_rounds(value: Any) -> str:
    try:
        rounds = int(value)
    except (TypeError, ValueError):
        return f"{value} rounds"
    if rounds <= 0:
        return "unlimited"
    return f"{rounds} rounds"


def _plural_count(value: Any, singular: str, plural: str) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return f"{value} {plural}"
    noun = singular if count == 1 else plural
    return f"{count} {noun}"


def _append_config_fact(summary: list[dict[str, str]], label: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        summary.append({"label": label, "value": text})


def _matching_agent_config(raw_agents: Any, agent_name: str) -> dict[str, Any]:
    agents = []
    if isinstance(raw_agents, list):
        agents = [item for item in raw_agents if isinstance(item, dict)]
    if not agents:
        return {}
    wanted = str(agent_name or "").strip()
    for agent in agents:
        for key in ("id", "kind", "agent_type"):
            if wanted and str(agent.get(key) or "").strip() == wanted:
                return agent
    return agents[0] if len(agents) == 1 else {}


def _run_config_summary(run_dir: Path, agent_label: str) -> list[dict[str, str]]:
    raw = _load_run_config_yaml(run_dir)
    if not raw:
        return []

    runtime = _mapping(raw.get("runtime") or raw.get("execution"))
    proxy = _mapping(raw.get("proxy"))
    target = _mapping(raw.get("target"))
    eval_config = _mapping(raw.get("eval"))
    agent_name, _, _ = _parse_agent_label_for_view(agent_label)
    agent_config = _matching_agent_config(raw.get("agents"), agent_name)
    summary: list[dict[str, str]] = []

    passk = _present_value(runtime, "passk")
    if passk is not None:
        _append_config_fact(summary, "Passes", f"{passk} per sample")

    workers = _present_value(runtime, "max_trials_global", "n_concurrent", "max_running_trials", "max_workers")
    if workers is not None:
        _append_config_fact(summary, "Trial concurrency", f"{workers} at once")

    target_setups = _present_value(runtime, "max_target_setups", "max_sample_target_setups")
    if target_setups is not None:
        _append_config_fact(summary, "Target setup concurrency", f"{target_setups} at once")

    timeout = _present_value(runtime, "timeout", "timeout_seconds")
    if timeout is not None:
        _append_config_fact(summary, "Trial timeout", _format_duration(timeout))

    max_rounds = _present_value(runtime, "max_rounds")
    if max_rounds is not None:
        _append_config_fact(summary, "Round budget", _format_rounds(max_rounds))

    max_trial = _present_value(runtime, "max_trial")
    if max_trial is not None:
        _append_config_fact(summary, "Invocation cap", f"first {max_trial} planned trials")

    live_check = _mapping(runtime.get("live_check"))
    if "enabled" in live_check:
        live_bits = ["on" if live_check.get("enabled") else "off"]
        max_calls = _present_value(live_check, "max_calls")
        if max_calls is not None:
            live_bits.append(_plural_count(max_calls, "check", "checks"))
        _append_config_fact(summary, "Live check", ", ".join(live_bits))

    proxy_timeout = _present_value(proxy, "request_timeout")
    if proxy_timeout is not None:
        _append_config_fact(summary, "Model request timeout", _format_duration(proxy_timeout))

    target_bits: list[str] = []
    if "enabled" in target:
        target_bits.append("enabled" if target.get("enabled") else "disabled")
    for key in ("run_mode", "target_scope", "parallel_mode"):
        value = _present_value(target, key)
        if value is not None:
            target_bits.append(str(value))
    _append_config_fact(summary, "Target runtime", ", ".join(target_bits))

    sample_limit = _present_value(eval_config, "limit", "sample_limit")
    if sample_limit is not None:
        _append_config_fact(summary, "Sample selection", f"limit {sample_limit}")

    agent_bits: list[str] = []
    max_concurrent = _present_value(agent_config, "max_concurrent")
    if max_concurrent is not None:
        agent_bits.append(f"{max_concurrent} at once")
    agent_rounds = _present_value(agent_config, "max_rounds")
    if agent_rounds is not None:
        agent_bits.append(_format_rounds(agent_rounds))
    session_args = agent_config.get("session_args") if isinstance(agent_config, dict) else []
    if isinstance(session_args, list) and session_args:
        agent_bits.append(_plural_count(len(session_args), "launch arg", "launch args"))
    shared_paths = agent_config.get("shared_paths") if isinstance(agent_config, dict) else []
    if isinstance(shared_paths, list) and shared_paths:
        agent_bits.append(_plural_count(len(shared_paths), "shared path", "shared paths"))
    _append_config_fact(summary, "Agent limits", ", ".join(agent_bits))

    return summary


def _run_project_root(run_dir: Path) -> Path:
    current = run_dir.resolve()
    for parent in current.parents:
        if parent.name == ".cage_runs":
            return parent.parent
    return current_app.config["ROOT_DIR"]


def _load_model_registry_entry(
    run_dir: Path,
    raw: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    models_file = str(raw.get("models_file") or "").strip()
    if not models_file or not model_id:
        return {}
    candidate = Path(models_file)
    if not candidate.is_absolute():
        candidate = _run_project_root(run_dir) / candidate
    if not candidate.exists():
        return {}
    try:
        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    models = loaded.get("models") if isinstance(loaded, dict) else {}
    if not isinstance(models, dict):
        return {}
    entry = models.get(model_id)
    return entry if isinstance(entry, dict) else {}


def _format_seconds_detail(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        text = str(value or "").strip()
        return text or "unlimited"
    if seconds <= 0:
        return "unlimited"
    raw = f"{int(seconds)}s" if seconds.is_integer() else f"{seconds:g}s"
    human = _format_duration(seconds)
    return raw if human == raw else f"{raw} ({human})"


def _format_prompt_levels(value: Any) -> str:
    if isinstance(value, list | tuple):
        out = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            lowered = text.lower()
            out.append(lowered if lowered.startswith("l") else f"l{text}")
        return ", ".join(out)
    text = str(value or "").strip()
    if not text:
        return "n/a"
    return text if text.lower().startswith("l") else f"l{text}"


def _format_max_rounds(value: Any) -> str:
    if value is None:
        return "benchmark default"
    try:
        rounds = int(value)
    except (TypeError, ValueError):
        return str(value)
    if rounds <= 0:
        return f"benchmark default (project.yml max_rounds={rounds})"
    return str(rounds)


def _append_summary_row(rows: list[dict[str, str]], label: str, value: Any) -> None:
    text = str(value or "").strip()
    if text:
        rows.append({"label": label, "value": text})


def _run_project_summary(
    run_dir: Path,
    dashboard: dict[str, Any],
    agent_label: str,
    sample_ids: list[str],
    trial_ids: list[str],
    trials_to_run: int,
) -> list[dict[str, Any]]:
    raw = _load_run_config_yaml(run_dir)
    config_yaml = _load_yaml_file(run_dir / "config.yaml")
    agent_name, label_model, mode = _parse_agent_label_for_view(agent_label)
    agent_config = _matching_agent_config(raw.get("agents"), agent_name) if raw else {}
    eval_config = _mapping(raw.get("eval")) if raw else {}
    benchmark = _mapping(eval_config.get("benchmark"))
    runtime = _mapping(raw.get("runtime") or raw.get("execution")) if raw else {}
    target = _mapping(raw.get("target")) if raw else {}
    proxy = _mapping(raw.get("proxy")) if raw else {}
    project = _mapping(raw.get("project")) if raw else {}
    model_id = str(
        _present_value(agent_config, "model")
        or config_yaml.get("model")
        or label_model
        or ""
    )
    model_entry = _load_model_registry_entry(run_dir, raw, model_id)

    run_selection: list[dict[str, str]] = []
    _append_summary_row(run_selection, "Benchmark", _present_value(benchmark, "class", "name"))
    _append_summary_row(run_selection, "Suite", _run_project_root(run_dir).name)
    _append_summary_row(
        run_selection,
        "Project",
        project.get("name") or dashboard.get("experiment"),
    )
    _append_summary_row(run_selection, "Run id", dashboard.get("run_id") or run_dir.name)
    _append_summary_row(run_selection, "Samples", _sample_scope_summary(sample_ids, trial_ids))
    _append_summary_row(
        run_selection,
        "Planned trials",
        f"{len(trial_ids)} total before filters and caps",
    )
    _append_summary_row(
        run_selection,
        "Trials to run",
        f"{trials_to_run} runnable after filters and caps",
    )
    passk = _present_value(runtime, "passk")
    if passk is not None:
        _append_summary_row(run_selection, "Pass@k attempts", f"{passk} per sample")
    levels = _present_value(benchmark, "hint_levels", "prompt_levels", "levels")
    if levels is not None:
        _append_summary_row(run_selection, "Prompt/hint levels", _format_prompt_levels(levels))
    limit = _present_value(eval_config, "limit", "sample_limit")
    if limit is not None:
        _append_summary_row(run_selection, "Sample selection", f"limit {limit}")
    _append_summary_row(
        run_selection,
        "Benchmark root",
        _present_value(benchmark, "benchmark_root", "root", "dataset"),
    )

    stop_conditions: list[dict[str, str]] = []
    _append_summary_row(
        stop_conditions,
        "Max rounds",
        _format_max_rounds(_present_value(runtime, "max_rounds")),
    )
    timeout = _present_value(runtime, "timeout", "timeout_seconds")
    if timeout is not None:
        _append_summary_row(stop_conditions, "Trial timeout", _format_seconds_detail(timeout))
    request_timeout = _present_value(proxy, "request_timeout")
    if request_timeout is not None:
        _append_summary_row(
            stop_conditions,
            "Model request timeout",
            _format_seconds_detail(request_timeout),
        )
    target_timeouts = []
    startup_timeout = _present_value(target, "startup_timeout")
    compose_timeout = _present_value(target, "compose_up_timeout", "compose_timeout")
    if startup_timeout is not None:
        startup_text = _format_seconds_detail(startup_timeout).split(" ")[0]
        target_timeouts.append(f"startup={startup_text}")
    if compose_timeout is not None:
        compose_text = _format_seconds_detail(compose_timeout).split(" ")[0]
        target_timeouts.append(f"compose={compose_text}")
    if target_timeouts:
        _append_summary_row(stop_conditions, "Target timeout", ", ".join(target_timeouts))
    for label, keys in (
        ("Max input tokens", ("max_input_tokens", "input_token_limit")),
        (
            "Max output tokens",
            ("max_output_tokens", "max_output_tokens_cap", "output_token_limit"),
        ),
        ("Max cost", ("max_cost", "max_cost_usd", "cost_limit_usd")),
    ):
        value = _present_value(runtime, *keys)
        _append_summary_row(stop_conditions, label, value if value is not None else "unlimited")
    _append_summary_row(stop_conditions, "Stop run", "Ctrl-C")

    agent_rows: list[dict[str, str]] = []
    _append_summary_row(agent_rows, "Agent", agent_config.get("id") or agent_name)
    _append_summary_row(agent_rows, "Agent label", agent_label)
    _append_summary_row(
        agent_rows,
        "Agent kind",
        agent_config.get("kind") or agent_config.get("agent_type"),
    )
    _append_summary_row(agent_rows, "Model", model_id)
    _append_summary_row(agent_rows, "Provider", model_entry.get("provider"))
    _append_summary_row(agent_rows, "Endpoint model", model_id)
    _append_summary_row(agent_rows, "Agent model", model_entry.get("model") or model_id)
    concurrency = []
    workers = _present_value(runtime, "max_trials_global", "n_concurrent", "max_running_trials", "max_workers")
    target_setups = _present_value(runtime, "max_target_setups", "max_sample_target_setups")
    max_concurrent = _present_value(agent_config, "max_concurrent")
    if workers is not None:
        concurrency.append(f"max_trials_global={workers}")
    if target_setups is not None:
        concurrency.append(f"max_target_setups={target_setups}")
    if max_concurrent is not None:
        concurrency.append(f"max_concurrent={max_concurrent}")
    _append_summary_row(agent_rows, "Concurrency", ", ".join(concurrency))
    _append_summary_row(agent_rows, "Image", agent_config.get("image"))
    session_args = agent_config.get("session_args")
    _append_summary_row(
        agent_rows,
        "Session args",
        _join_values(session_args) if session_args else "none",
    )
    env_text = _agent_environment_summary(agent_config)
    if env_text != "framework-managed proxy/model env; no extra_env in project.yml":
        _append_summary_row(agent_rows, "Environment", env_text)

    return [
        {"title": "Run selection", "rows": run_selection},
        {"title": "Stop conditions", "rows": stop_conditions},
        {"title": "Agent / model", "rows": agent_rows},
    ]


def _join_values(values: Any) -> str:
    if isinstance(values, list):
        return ", ".join(str(item) for item in values if str(item).strip())
    if isinstance(values, tuple):
        return ", ".join(str(item) for item in values if str(item).strip())
    if isinstance(values, dict):
        return ", ".join(
            f"{key}={value}"
            for key, value in values.items()
            if str(key).strip() and str(value).strip()
        )
    return str(values or "").strip()


def _sample_scope_summary(sample_ids: list[str], trial_ids: list[str]) -> str:
    if sample_ids:
        if len(sample_ids) <= 8:
            return ", ".join(sample_ids)
        return f"{len(sample_ids)} samples ({sample_ids[0]} to {sample_ids[-1]})"
    if trial_ids:
        return f"{len(trial_ids)} planned trials"
    return "not recorded"


def _run_result_summary(counts: dict[str, int]) -> str:
    total = int(counts.get("total") or 0)
    noun = "trial" if total == 1 else "trials"
    parts = [
        f"{int(counts.get('running') or 0)} running",
        f"{int(counts.get('completed') or 0)} completed",
        f"{int(counts.get('live_success') or 0)} target passed",
        f"{int(counts.get('warnings') or 0)} stopped",
        f"{int(counts.get('failed') or 0)} failed",
    ]
    if counts.get("other"):
        parts.append(f"{int(counts.get('other') or 0)} pending")
    return f"{total} {noun}: {' / '.join(parts)}"


def _termination_summary(raw: dict[str, Any]) -> str:
    runtime = _mapping(raw.get("runtime") or raw.get("execution"))
    bits: list[str] = []
    live_check = _mapping(runtime.get("live_check"))
    if "enabled" in live_check:
        if live_check.get("enabled"):
            live_bits = ["live check on"]
            max_calls = _present_value(live_check, "max_calls")
            if max_calls is not None:
                live_bits.append(_plural_count(max_calls, "check", "checks"))
            bits.append(", ".join(live_bits))
        else:
            bits.append("live check off")
    max_rounds = _present_value(runtime, "max_rounds")
    if max_rounds is not None:
        try:
            rounds = int(max_rounds)
        except (TypeError, ValueError):
            bits.append(f"round budget {max_rounds}")
        else:
            if rounds <= 0:
                bits.append("benchmark default round budget")
            else:
                bits.append(f"round budget {_format_rounds(rounds)}")
    timeout = _present_value(runtime, "timeout", "timeout_seconds")
    if timeout is not None:
        bits.append(f"timeout {_format_duration(timeout)}")
    return "; ".join(bits) if bits else "not recorded"


def _agent_environment_summary(agent_config: dict[str, Any]) -> str:
    extra_env = agent_config.get("extra_env")
    if isinstance(extra_env, dict) and extra_env:
        return _join_values(extra_env)
    env = agent_config.get("env")
    if isinstance(env, dict) and env:
        return _join_values(env)
    return "framework-managed proxy/model env; no extra_env in project.yml"


def _run_setup_summary(
    run_dir: Path,
    agent_label: str,
    sample_ids: list[str],
    trial_ids: list[str],
    counts: dict[str, int],
) -> dict[str, Any]:
    raw = _load_run_config_yaml(run_dir)
    agent_name, _, _ = _parse_agent_label_for_view(agent_label)
    agent_config = _matching_agent_config(raw.get("agents"), agent_name) if raw else {}
    eval_config = _mapping(raw.get("eval")) if raw else {}
    benchmark = _mapping(eval_config.get("benchmark"))
    runtime = _mapping(raw.get("runtime") or raw.get("execution")) if raw else {}
    target = _mapping(raw.get("target")) if raw else {}
    proxy = _mapping(raw.get("proxy")) if raw else {}

    evaluation_facts: list[dict[str, str]] = [
        {"label": "Overall result", "value": _run_result_summary(counts)},
        {"label": "Samples", "value": _sample_scope_summary(sample_ids, trial_ids)},
    ]
    project_name = _mapping(raw.get("project")).get("name") if raw else ""
    if project_name:
        evaluation_facts.append({"label": "Project", "value": str(project_name)})
    benchmark_class = _present_value(benchmark, "class", "name")
    benchmark_module = _present_value(benchmark, "module")
    if benchmark_class:
        value = str(benchmark_class)
        if benchmark_module:
            value = f"{value} ({benchmark_module})"
        evaluation_facts.append({"label": "Benchmark", "value": value})
    benchmark_root = _present_value(benchmark, "benchmark_root", "root", "dataset")
    if benchmark_root:
        evaluation_facts.append({"label": "Benchmark root", "value": str(benchmark_root)})
    hint_levels = _present_value(benchmark, "hint_levels")
    if hint_levels is not None:
        evaluation_facts.append({"label": "Hint levels", "value": _join_values(hint_levels)})
    limit = _present_value(eval_config, "limit", "sample_limit")
    if limit is not None:
        evaluation_facts.append({"label": "Sample selection", "value": f"limit {limit}"})
    evaluation_facts.append({"label": "Termination", "value": _termination_summary(raw)})
    workers = _present_value(runtime, "max_trials_global", "n_concurrent", "max_running_trials", "max_workers")
    if workers is not None:
        evaluation_facts.append({"label": "Trial concurrency", "value": f"{workers} at once"})
    proxy_timeout = _present_value(proxy, "request_timeout")
    if proxy_timeout is not None:
        evaluation_facts.append({
            "label": "Model request timeout",
            "value": _format_duration(proxy_timeout),
        })
    target_bits = []
    if "enabled" in target:
        target_bits.append("enabled" if target.get("enabled") else "disabled")
    for key in ("run_mode", "target_scope", "parallel_mode"):
        value = _present_value(target, key)
        if value is not None:
            target_bits.append(str(value))
    if target_bits:
        evaluation_facts.append({"label": "Target runtime", "value": ", ".join(target_bits)})

    agent_facts: list[dict[str, str]] = []
    for label, value in (
        ("Kind", agent_config.get("kind") or agent_config.get("agent_type")),
        ("Model", agent_config.get("model")),
        ("Image", agent_config.get("image")),
        ("Home", agent_config.get("home")),
    ):
        if value:
            agent_facts.append({"label": label, "value": str(value)})
    max_concurrent = _present_value(agent_config, "max_concurrent")
    if max_concurrent is not None:
        agent_facts.append({"label": "Max concurrent", "value": f"{max_concurrent} at once"})
    agent_rounds = _present_value(agent_config, "max_rounds")
    if agent_rounds is not None:
        agent_facts.append({
            "label": "Agent round budget",
            "value": _format_rounds(agent_rounds),
        })
    session_args = agent_config.get("session_args")
    if session_args:
        agent_facts.append({"label": "Session args", "value": _join_values(session_args)})
    shared_paths = agent_config.get("shared_paths")
    if shared_paths:
        agent_facts.append({"label": "Shared paths", "value": _join_values(shared_paths)})
    agent_facts.append({
        "label": "Environment",
        "value": _agent_environment_summary(agent_config),
    })
    if raw.get("models_file"):
        agent_facts.append({"label": "Model registry", "value": str(raw.get("models_file"))})

    return {
        "evaluation": evaluation_facts,
        "agent": agent_facts,
    }


def _build_run_overview(
    run_dir: Path,
    dashboard: dict[str, Any],
    trials: list[dict[str, Any]],
    run_history: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = _aggregate_agent_counts(t["info"] for t in trials)
    has_running = counts.get("running", 0) > 0
    ids = [str(t.get("id") or "") for t in trials]
    sample_ids = sorted({
        str(t["info"].get("sample_id") or "")
        for t in trials
        if t["info"].get("sample_id")
    })
    status_labels = sorted({
        str(t["info"].get("status_label") or "")
        for t in trials
        if t["info"].get("status_label")
    })
    config_files = [
        name
        for name in (
            "project.yml",
            "config.yaml",
            "config.yml",
            "planned_trials.json",
            "dashboard.json",
            "dashboard_view.json",
        )
        if (run_dir / name).exists()
    ]
    agent_label = run_dir.parent.name
    agent_name, model_name, mode = _parse_agent_label_for_view(agent_label)
    latest = run_history[-1] if run_history else {}
    config_summary = _run_config_summary(run_dir, agent_label)
    attempts = len(run_history) or 1
    resume_count = max(0, attempts - 1)
    history_summary = {
        "latest_label": str(latest.get("label") or "Initial run"),
        "latest_status": str(latest.get("status") or dashboard.get("status") or ""),
        "latest_started_at": str(latest.get("started_at") or dashboard.get("started_at") or ""),
        "latest_completed_at": str(
            latest.get("completed_at") or dashboard.get("completed_at") or ""
        ),
        "latest_duration_ms": int(latest.get("duration_ms") or 0),
        "attempts": attempts,
        "resumes": resume_count,
        "resume_label": _plural_count(resume_count, "resume", "resumes"),
    }
    return {
        "lifecycle": _run_lifecycle_label(dashboard, has_running),
        "action": _run_action_label(counts),
        "health": _run_health_sentence(counts),
        "activity_warnings": _run_activity_warnings(dashboard, counts),
        "counts": counts,
        "agent_label": agent_label,
        "agent_name": agent_name,
        "model_name": model_name,
        "mode": mode,
        "config_files": config_files,
        "config_summary": config_summary,
        "project_summary": _run_project_summary(
            run_dir,
            dashboard,
            agent_label,
            sample_ids,
            ids,
            len(trials),
        ),
        "trial_scope": {
            "total": len(trials),
            "samples": len(sample_ids) or len(ids),
            "first": ids[0] if ids else "",
            "last": ids[-1] if ids else "",
        },
        "termination_labels": status_labels,
        "started_at": dashboard.get("started_at") or latest.get("started_at") or "",
        "completed_at": latest.get("completed_at") or dashboard.get("completed_at") or "",
        "attempts": attempts,
        "history_summary": history_summary,
        "output_dir": str(run_dir),
    }


def _counts_for_debug_bundle(counts: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(counts.get(key) or 0)
        for key in (
            "running",
            "completed",
            "live_success",
            "warnings",
            "failed",
            "other",
            "total",
        )
    }


def _run_resource_summary(run_dir: Path) -> dict[str, Any]:
    """Summarize latest ResourceLedger state for inspect/debug output.

    The canonical projection lives in ``cage.gc.summary`` so
    inspect, CLI, GC, and future dashboard surfaces do not each reinterpret
    the append-only ``resources.jsonl`` ledger differently.
    """

    return summarize_resource_ledger(run_dir).to_mapping()


def _run_debug_bundle(
    run_dir: Path,
    dashboard: dict[str, Any],
    run_overview: dict[str, Any],
    run_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "run",
        "url": request.path,
        "run_url": _run_url_for(run_dir),
        "dashboard_url": _dashboard_url_for(run_dir),
        "run_id": dashboard.get("run_id") or run_dir.name,
        "experiment": dashboard.get("experiment") or "",
        "status": dashboard.get("status") or "",
        "run_dir": str(run_dir),
        "overview": {
            "lifecycle": run_overview.get("lifecycle", ""),
            "action": run_overview.get("action", ""),
            "health": run_overview.get("health", ""),
            "activity_warnings": run_overview.get("activity_warnings", []),
            "agent_label": run_overview.get("agent_label", ""),
            "agent_name": run_overview.get("agent_name", ""),
            "model_name": run_overview.get("model_name", ""),
            "mode": run_overview.get("mode", ""),
            "started_at": run_overview.get("started_at", ""),
            "completed_at": run_overview.get("completed_at", ""),
            "trial_scope": run_overview.get("trial_scope", {}),
            "project_summary": run_overview.get("project_summary", []),
        },
        "counts": _counts_for_debug_bundle(run_overview.get("counts", {})),
        "run_history": [
            {
                "label": item.get("label", ""),
                "status": item.get("status", ""),
                "started_at": item.get("started_at", ""),
                "completed_at": item.get("completed_at", ""),
                "duration_ms": item.get("duration_ms", 0),
                "source": item.get("source", ""),
            }
            for item in run_history
        ],
        "resources": _run_resource_summary(run_dir),
    }


def _trial_debug_bundle(
    trial_dir: Path,
    run_dir: Path,
    trial: Any,
    trial_status: dict[str, Any],
    termination: dict[str, Any],
    usage: dict[str, Any],
    diagnosis: dict[str, Any],
    benchmark_outcome: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "trial",
        "url": request.path,
        "run_url": _run_url_for(run_dir),
        "trial_url": _trial_url_for(trial_dir, run_dir),
        "run_id": run_dir.name,
        "trial_id": trial.trial_id,
        "run_dir": str(run_dir),
        "trial_dir": str(trial_dir),
        "status": {
            "label": trial_status.get("status_label", "Pending"),
            "detail": trial_status.get("status_detail", ""),
            "kind": trial_status.get("status_kind", "pending"),
            "running": bool(trial_status.get("running")),
        },
        "duration_ms": (
            trial_status.get("duration_ms")
            or trial.meta.get("timing", {}).get("duration_ms", 0)
            or 0
        ),
        "exit_code": trial.meta.get("exit_code", "-"),
        "usage": dict(usage or {}),
        "termination": termination,
        "diagnosis": diagnosis,
        "benchmark_outcome": benchmark_outcome,
        "scores": trial.scores,
    }


def _dashboard_debug_bundle(
    run_dir: Path,
    dashboard: dict[str, Any],
    run_overview: dict[str, Any],
    dashboard_mode: str,
    dashboard_freshness: dict[str, Any],
    dashboard_insight: dict[str, Any],
    dashboard_error: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "kind": "dashboard",
        "url": request.path,
        "run_url": _run_url_for(run_dir),
        "dashboard_url": _dashboard_url_for(run_dir),
        "run_id": dashboard.get("run_id") or run_dir.name,
        "experiment": dashboard.get("experiment") or "",
        "run_dir": str(run_dir),
        "dashboard_mode": dashboard_mode,
        "freshness": dashboard_freshness,
        "insight": dashboard_insight,
        "error": dashboard_error,
        "overview": {
            "lifecycle": run_overview.get("lifecycle", ""),
            "action": run_overview.get("action", ""),
            "health": run_overview.get("health", ""),
            "trial_scope": run_overview.get("trial_scope", {}),
            "project_summary": run_overview.get("project_summary", []),
        },
        "counts": _counts_for_debug_bundle(run_overview.get("counts", {})),
    }


def _parse_agent_label_for_view(label: str) -> tuple[str, str, str]:
    parts = label.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return label, "", ""


def _build_trial_diagnosis(
    trial_status: dict[str, Any],
    termination: dict[str, Any],
    trial: Any,
) -> dict[str, str]:
    kind = str(trial_status.get("status_kind") or "pending")
    label = str(trial_status.get("status_label") or "Pending")
    scores = []
    for metric, data in getattr(trial, "scores", {}).items():
        value = extract_numeric_score_value(data)
        if value is not None:
            scores.append(f"{metric} {value:.2f}")
        else:
            raw = data.get("value") if isinstance(data, dict) else data
            scores.append(f"{metric} {raw}")
    if kind == "running":
        verdict = "Running"
        action = "Monitor progress"
    elif kind == "live_success":
        verdict = "Target passed"
        action = "Check evidence"
    elif kind == "warning":
        verdict = "Stopped"
        action = "Check stop reason"
    elif kind == "error":
        verdict = "Failed execution"
        action = "Investigate failure"
    elif scores:
        verdict = "Execution completed"
        action = "Check benchmark score"
    else:
        verdict = "No verdict yet"
        action = "Check artifacts"
    reason = termination.get("reason_label") or termination.get("reason") or label
    detail = termination.get("detail") or trial_status.get("status_detail") or ""
    return {
        "verdict": verdict,
        "action": action,
        "status": label,
        "reason": str(reason or "-"),
        "detail": str(detail or ""),
        "score": ", ".join(scores) if scores else "No score",
    }


def _trial_score_values(trial: Any) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for metric, data in getattr(trial, "scores", {}).items():
        value = extract_numeric_score_value(data)
        if value is not None:
            values.append((str(metric), value))
    return values


def _build_trial_benchmark_outcome(trial: Any) -> dict[str, str]:
    values = _trial_score_values(trial)
    if not values:
        return {
            "kind": "none",
            "label": "No benchmark score",
            "detail": "No score artifact was recorded for this trial.",
        }
    metric, best = max(values, key=lambda item: item[1])
    score_text = f"{metric} {best:.2f}"
    if best >= 0.7:
        return {
            "kind": "passed",
            "label": "Passed",
            "detail": score_text,
        }
    if best > 0:
        return {
            "kind": "partial",
            "label": "Partial result",
            "detail": f"{score_text}; review evidence before treating it as solved.",
        }
    return {
        "kind": "failed",
        "label": "Not solved",
        "detail": score_text,
    }


def _dashboard_trial_count(view: dict[str, Any] | None) -> int | None:
    if not isinstance(view, dict):
        return None
    for section in view.get("sections", []) or []:
        if not isinstance(section, dict):
            continue
        if section.get("kind") == "summary":
            for stat in section.get("stats", []) or []:
                if not isinstance(stat, dict):
                    continue
                if str(stat.get("label") or "").strip().lower() == "trials":
                    try:
                        return int(float(str(stat.get("value") or "0").replace(",", "")))
                    except ValueError:
                        pass
        title = str(section.get("title") or "").lower()
        if section.get("kind") == "table" and "per-trial" in title:
            rows = section.get("rows")
            if isinstance(rows, list):
                return len(rows)
    return None


def _dashboard_signature_ms(
    run_dir: Path,
    trial_dirs: Iterable[Path] | None = None,
) -> int:
    trial_dirs = trial_dirs if trial_dirs is not None else find_trial_dirs(run_dir)
    signature_ns = max(
        safe_mtime_ns(run_dir / "dashboard.json"),
        safe_mtime_ns(run_dir / "dashboard_view.json"),
        safe_mtime_ns(run_dir / "run_history.json"),
        safe_mtime_ns(run_dir / "planned_trials.json"),
    )
    for trial_dir in trial_dirs:
        signature_ns = max(
            signature_ns,
            safe_mtime_ns(trial_dir),
            safe_mtime_ns(trial_dir / "meta.json"),
            safe_mtime_ns(trial_dir / "task_output.json"),
            safe_mtime_ns(trial_dir / "proxy" / "progress.json"),
            safe_mtime_ns(trial_dir / "scores"),
        )
    if signature_ns <= 0:
        return 0
    return signature_ns // 1_000_000


def _dashboard_freshness(run_dir: Path, view: dict[str, Any] | None) -> dict[str, Any]:
    trial_dirs = find_trial_dirs(run_dir)
    current_count = len(trial_dirs)
    saved_count = _dashboard_trial_count(view)
    path = run_dir / "dashboard_view.json"
    mtime = path.stat().st_mtime if path.exists() else 0
    signature_ms = _dashboard_signature_ms(run_dir, trial_dirs)
    return {
        "saved_count": saved_count,
        "current_count": current_count,
        "stale": saved_count is not None and saved_count != current_count,
        "source": "dashboard_view.json" if path.exists() else "live artifacts",
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime)) if mtime else "",
        "signature_ms": signature_ms,
    }


def _dashboard_view_artifact_error(run_dir: Path) -> dict[str, str] | None:
    path = run_dir / "dashboard_view.json"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "title": "Dashboard view file is unreadable",
            "detail": str(exc),
            "path": str(path),
        }
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "title": "Dashboard view file is malformed",
            "detail": (
                "dashboard_view.json is not valid JSON at "
                f"line {exc.lineno}, column {exc.colno}."
            ),
            "path": str(path),
        }
    if not isinstance(data, dict):
        return {
            "title": "Dashboard view file is malformed",
            "detail": "dashboard_view.json must contain a JSON object at the top level.",
            "path": str(path),
        }
    return None


def _dashboard_status_kind(raw_status: Any) -> str:
    if isinstance(raw_status, dict):
        raw_status = (
            raw_status.get("status")
            or raw_status.get("display_value")
            or raw_status.get("value")
            or raw_status.get("raw_value")
            or raw_status.get("sort_key")
        )
    status = str(raw_status or "").strip().lower()
    if not status:
        return "other"
    if any(term in status for term in ("fail", "error", "unavailable", "crash")):
        return "failed"
    if any(
        term in status
        for term in ("timeout", "timed", "warning", "review", "audit", "interrupt")
    ):
        return "stopped"
    if "running" in status or "active" in status:
        return "running"
    if any(term in status for term in ("complete", "success", "passed", "verified")):
        return "completed"
    return "stopped"


def _dashboard_status_label(raw_status: Any) -> str:
    kind = _dashboard_status_kind(raw_status)
    if kind == "completed":
        return "Completed"
    if kind == "stopped":
        return "Stopped"
    if kind == "failed":
        return "Failed"
    if kind == "running":
        return "Running"
    return "Check"


def _dashboard_cell_display(cell: Any) -> str:
    if isinstance(cell, dict):
        for key in ("display_value", "value", "label", "raw_value"):
            if key in cell:
                return str(cell.get(key) or "")
        return ""
    return str(cell or "")


def _dashboard_cell_sort_value(cell: Any) -> str:
    if isinstance(cell, dict):
        for key in ("sort_key", "raw_value", "value", "display_value"):
            if key in cell:
                return str(cell.get(key) or "")
        return ""
    return str(cell or "")


def _dashboard_cell_attr(cell: Any, key: str) -> str:
    if isinstance(cell, dict):
        return str(cell.get(key) or "")
    return ""


def _dashboard_status_counts(view: dict[str, Any] | None) -> dict[str, int]:
    counts = {"total": 0, "completed": 0, "stopped": 0, "failed": 0}
    if not isinstance(view, dict):
        return counts
    for section in view.get("sections", []) or []:
        if not isinstance(section, dict) or section.get("kind") != "table":
            continue
        columns = section.get("columns") or []
        status_keys = {
            str(col.get("key"))
            for col in columns
            if isinstance(col, dict) and str(col.get("key") or "").lower() == "status"
        }
        if not status_keys:
            continue
        for row in section.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").strip().lower()
            if not status:
                continue
            counts["total"] += 1
            kind = _dashboard_status_kind(status)
            if kind == "failed":
                counts["failed"] += 1
            elif kind == "stopped":
                counts["stopped"] += 1
            elif kind == "completed":
                counts["completed"] += 1
            else:
                counts["stopped"] += 1
    return counts


def _dashboard_trial_identity(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("raw_value", "value", "display_value", "label", "sort_key"):
            if key in value:
                return str(value.get(key) or "").strip()
        return ""
    return str(value or "").strip()


def _attach_dashboard_trial_links(
    view: dict[str, Any] | None,
    run_dir: Path,
) -> dict[str, Any] | None:
    if not isinstance(view, dict):
        return view
    trial_dirs = find_trial_dirs(run_dir)
    trial_map = {
        _trial_display_id(trial_dir, run_dir): trial_dir
        for trial_dir in trial_dirs
    }
    for section in view.get("sections", []) or []:
        if not isinstance(section, dict) or section.get("kind") != "table":
            continue
        title = str(section.get("title") or "").lower()
        rows = section.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("_href"):
                continue
            trial_id = ""
            for key in ("_trial_id", "trial_id", "trial"):
                trial_id = _dashboard_trial_identity(row.get(key))
                if trial_id:
                    break
            if not trial_id and "per-trial" not in title:
                continue
            trial_dir = trial_map.get(trial_id)
            if trial_dir is not None:
                row["_href"] = _trial_url_for(trial_dir, run_dir)
    return view


def _dashboard_insight(
    view: dict[str, Any] | None,
    freshness: dict[str, Any],
) -> dict[str, Any]:
    counts = _dashboard_status_counts(view)
    if counts["total"]:
        outcome = (
            f"Outcome: {counts['completed']}/{counts['total']} completed, "
            f"{counts['stopped']} stopped, {counts['failed']} failed."
        )
    elif freshness.get("stale"):
        outcome = (
            f"Saved dashboard covers {freshness.get('saved_count')} trials; "
            f"current artifacts contain {freshness.get('current_count')}."
        )
    else:
        outcome = (
            "Use this page for benchmark-specific metrics; "
            "use the run overview for current state."
        )
    return {
        "role": "Benchmark metrics view",
        "outcome": outcome,
        "scope": (
            "Run overview remains the source of truth for current "
            "trial state and artifacts."
        ),
        "diagnostics": [
            {
                "label": "Stopped rows",
                "value": counts["stopped"],
                "tone": "stopped",
            },
            {
                "label": "Failed rows",
                "value": counts["failed"],
                "tone": "failed",
            },
            {
                "label": "Completed rows",
                "value": counts["completed"],
                "tone": "completed",
            },
        ] if counts["total"] else [],
    }


def _trial_usage_from_disk(
    trial_dir: Path, *, fallback: dict[str, Any],
) -> dict[str, Any]:
    """Build the trial-detail usage card from the shared activity reader.

    Trials write ``proxy/progress.json`` on each model request; that file is the
    usage source. ``load_trial_activity`` is the boundary that reads it, and this
    route-level helper only maps its progress-shaped payload to the template's
    usage fields.
    """
    activity = load_trial_activity(trial_dir)
    usage = activity.get("usage") if activity else {}
    if not isinstance(usage, dict) or not usage:
        return dict(fallback or {})
    return dict(usage)


def _usage_total_tokens(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    total = 0
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        try:
            total += int(usage.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _median_token_total(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _trial_token_context(
    run_dir: Path,
    trial_dir: Path,
    usage: dict[str, Any],
    dashboard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_total = _usage_total_tokens(usage)
    totals: list[int] = []
    if dashboard:
        # Per-trial usage is already aggregated in the dashboard projection, so
        # the run-wide median (for the "token outlier" badge) needs no per-trial
        # filesystem walk — rendering ONE trial must not stat/read all N trials.
        for agent_data in (dashboard.get("agents") or {}).values():
            if not isinstance(agent_data, dict):
                continue
            for entry in agent_data.get("trials") or []:
                if isinstance(entry, dict) and (
                    total := _usage_total_tokens(entry.get("usage") or {})
                ) > 0:
                    totals.append(total)
    if not totals:
        # Live / no-dashboard run: fall back to reading each trial's usage.
        totals = [
            total
            for candidate in find_trial_dirs(run_dir)
            if (
                total := _usage_total_tokens(
                    usage
                    if candidate == trial_dir
                    else _trial_usage_from_disk(candidate, fallback={})
                )
            ) > 0
        ]
    median_total = _median_token_total(totals)
    ratio = (current_total / median_total) if median_total else 0.0
    return {
        "total_tokens": current_total,
        "median_tokens": median_total,
        "trial_count": len(totals),
        "ratio": ratio,
        "is_high_outlier": len(totals) >= 3 and ratio >= 3.0,
        "available": current_total > 0 and median_total > 0 and len(totals) >= 2,
    }


def _stub_dashboard(run_dir: Path) -> dict[str, Any]:
    """Build a minimal dashboard-shaped dict for runs without one yet.

    The orchestrator only writes ``dashboard.json`` after the run
    finishes (or on Ctrl+C). While the run is in flight we still
    want the page to render. Canonical-only runs should also render from
    ``ExperimentRecord`` even when legacy ``planned_trials.json`` is gone.
    The trials list is left empty; the page reads per-trial state from the
    filesystem and canonical record refs.
    """

    config_path = run_dir / "config.yaml"
    planned_path = run_dir / "planned_trials.json"
    record_path = run_dir / "experiment_record.json"
    snapshot = (
        ExperimentArtifactReader(run_dir).try_load_snapshot()
        if record_path.exists()
        else None
    )
    if (
        not config_path.exists()
        and not planned_path.exists()
        and not record_path.exists()
    ):
        return {}
    agent_label = run_dir.parent.name
    agents: dict[str, Any] = {}
    if snapshot is not None:
        agents[agent_label] = {
            "total": snapshot.record.trials.total,
            "completed": snapshot.record.trials.completed,
            "failed": snapshot.record.trials.failed,
            "interrupted": snapshot.record.trials.interrupted,
            "running": max(
                0,
                snapshot.record.trials.total
                - snapshot.record.trials.completed
                - snapshot.record.trials.failed
                - snapshot.record.trials.interrupted,
            ),
        }
    return {
        "run_id": snapshot.record.run_id if snapshot is not None else run_dir.name,
        "experiment": (
            snapshot.spec.identity.experiment_id
            if snapshot is not None
            else ""
        ),
        "started_at": snapshot.record.started_at if snapshot is not None else "",
        "completed_at": snapshot.record.completed_at if snapshot is not None else "",
        "status": snapshot.record.status if snapshot is not None else "running",
        "agents": agents,
    }


def _generic_dashboard_snapshot(
    run_dir: Path,
    dashboard: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Build a transient dashboard from live trial artifacts.

    Benchmark-defined ``dashboard_view.json`` is usually written at the
    end of a run. This fallback keeps the dashboard route useful while a
    run is still in flight, using the same on-disk classifier as the run
    detail table.
    """
    trials_with_paths = _trial_rows_for_run(run_dir, dashboard)
    if not trials_with_paths:
        return None, "missing"

    counts = {
        "total": 0,
        "running": 0,
        "completed": 0,
        "warnings": 0,
        "failed": 0,
        "pending": 0,
    }
    total_requests = 0
    total_tokens = 0
    rows: list[dict[str, Any]] = []

    def _progress_requests(progress: dict[str, Any]) -> int:
        for key in ("successful_requests", "success", "total_requests"):
            try:
                return int(progress.get(key) or 0)
            except (TypeError, ValueError):
                return 0
        return 0

    for trial in trials_with_paths:
        trial_dir = trial["dir"]
        trial_id = trial["id"]
        info = trial["info"]
        kind = str(info.get("status_kind") or "pending").lower()
        label = str(info.get("status_label") or kind.title())
        progress = info.get("progress") if isinstance(info.get("progress"), dict) else {}
        usage = info.get("usage") if isinstance(info.get("usage"), dict) else {}
        requests = _progress_requests(progress)
        tokens = sum(
            int_or_zero(usage.get(key))
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")
        )
        duration_ms = int_or_zero(info.get("duration_ms"))

        counts["total"] += 1
        if kind == "running":
            counts["running"] += 1
        elif kind in {"success", "live_success"}:
            counts["completed"] += 1
        elif kind == "warning":
            counts["warnings"] += 1
        elif kind == "error":
            counts["failed"] += 1
        else:
            counts["pending"] += 1
        total_requests += requests
        total_tokens += tokens

        rows.append({
            "trial": trial_id,
            "sample": str(info.get("sample_id") or ""),
            "status": label,
            "requests": fmt_tokens(requests),
            "tokens": fmt_tokens(tokens),
            "duration": fmt_duration(duration_ms),
            "_href": _trial_url_for(trial_dir, run_dir) if info.get("has_artifacts") else "",
        })

    rows.sort(key=lambda row: str(row.get("trial") or ""))
    mode = "live" if counts["running"] else "partial"
    return (
        {
            "schema_version": 1,
            "title": "Current run dashboard",
            "subtitle": dashboard.get("run_id") or run_dir.name,
            "sections": [
                {
                    "kind": "summary",
                    "title": "Live snapshot" if mode == "live" else "Current snapshot",
                    "stats": [
                        {"label": "Trials", "value": fmt_tokens(counts["total"])},
                        {"label": "Running", "value": fmt_tokens(counts["running"])},
                        {"label": "Completed", "value": fmt_tokens(counts["completed"])},
                        {"label": "Warnings", "value": fmt_tokens(counts["warnings"])},
                        {"label": "Failed", "value": fmt_tokens(counts["failed"])},
                        {"label": "Requests", "value": fmt_tokens(total_requests)},
                        {"label": "Tokens", "value": fmt_tokens(total_tokens)},
                    ],
                },
                {
                    "kind": "table",
                    "title": "Current trials",
                    "columns": [
                        {"key": "trial", "label": "Trial"},
                        {"key": "sample", "label": "Sample"},
                        {"key": "status", "label": "Status"},
                        {"key": "requests", "label": "Requests", "align": "right"},
                        {"key": "tokens", "label": "Tokens", "align": "right"},
                        {"key": "duration", "label": "Duration", "align": "right"},
                    ],
                    "rows": rows,
                },
            ],
        },
        mode,
    )


def _benchmark_dashboard_snapshot(run_dir: Path) -> dict[str, Any] | None:
    """Ask the run's benchmark for its dashboard without writing it.

    The benchmark owns experiment semantics such as scores, pass@k, target
    grouping, and profile-specific columns. The web inspector should use
    that richer view whenever the run carries its project snapshot; the
    generic table below is only a last-resort fallback.
    """
    project_files = _benchmark_project_candidates(run_dir)
    if not project_files:
        return None

    from cage.config.experiment import resolve

    for project_file in project_files:
        try:
            config = resolve(project_file)
            spec = config.benchmark.build_dashboard(run_dir)
        except Exception as exc:  # noqa: BLE001
            current_app.logger.debug(
                "live dashboard build failed for %s via %s: %s",
                run_dir,
                project_file,
                exc,
            )
            continue
        if spec is None:
            continue
        if hasattr(spec, "to_dict"):
            return spec.to_dict()
        if isinstance(spec, dict):
            return spec
    current_app.logger.warning("live dashboard build unavailable for %s", run_dir)
    return None


def _best_dashboard_view(
    run_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    final_view = load_dashboard_view(run_dir)
    snapshot, snapshot_mode = _generic_dashboard_snapshot(run_dir, manifest)
    benchmark_view = None
    if snapshot_mode == "live" or final_view is None:
        benchmark_view = _benchmark_dashboard_snapshot(run_dir)
    if snapshot_mode == "live" or final_view is None:
        view = benchmark_view or snapshot
        return _attach_dashboard_trial_links(view, run_dir), snapshot_mode
    return _attach_dashboard_trial_links(final_view, run_dir), "final"


def _benchmark_project_candidates(run_dir: Path) -> list[Path]:
    """Return project files that can load the run's benchmark.

    ``cage run`` snapshots the original YAML to ``<run>/project.yml`` for
    provenance, but older/current snapshots do not include sibling files
    such as ``benchmark.py``, ``config/models.yml``, or datasets. For inspector
    usage, the live project directory next to ``.cage_runs`` is usually
    still available and can load the benchmark module correctly.
    """
    snapshot = run_dir / "project.yml"
    wanted_name = _project_name_from_yaml(snapshot)
    project_dir = _project_dir_for_run(run_dir)
    candidates: list[Path] = []

    def add(path: Path) -> None:
        if not path.is_file():
            return
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    add(snapshot)
    search_dirs = [p for p in (project_dir, current_app.config["ROOT_DIR"]) if p is not None]
    seen_dirs: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        add(directory / "project.yml")
        for pattern in ("*.yml", "*.yaml"):
            for candidate in sorted(directory.glob(pattern)):
                if wanted_name:
                    name = _project_name_from_yaml(candidate)
                    if name and name != wanted_name:
                        continue
                add(candidate)
    return candidates


def _project_dir_for_run(run_dir: Path) -> Path | None:
    """Return ``<project>`` for ``<project>/.cage_runs/<agent>/<run>``."""
    for parent in run_dir.parents:
        if parent.name == ".cage_runs":
            return parent.parent
    return None


def _project_name_from_yaml(path: Path) -> str:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ""
    project = raw.get("project") if isinstance(raw, dict) else {}
    if not isinstance(project, dict):
        return ""
    return str(project.get("name") or "").strip()


def _index_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _index_run_counts(run: Any) -> dict[str, int]:
    counts = {"total": 0, "completed": 0, "warnings": 0, "failed": 0, "running": 0}
    agents = getattr(run, "agents", {}) or {}
    values = agents.values() if isinstance(agents, dict) else []
    for agent in values:
        if not isinstance(agent, dict):
            continue
        for key in counts:
            counts[key] += _index_int(agent.get(key))
    return counts


def _build_index_summary(runs: list[Any], active_runs: list[Any]) -> dict[str, int]:
    run_counts = [_index_run_counts(run) for run in runs]
    return {
        "total": len(runs),
        "running": len(active_runs),
        "projects": len({run.project for run in runs}),
        "models": len({run.model_name for run in runs if run.model_name}),
        "trials": sum(counts["total"] for counts in run_counts),
        "review_runs": sum(
            1
            for counts in run_counts
            if counts["warnings"] > 0 or counts["failed"] > 0
        ),
        "failure_heavy_runs": sum(
            1
            for counts in run_counts
            if counts["total"] > 0 and counts["failed"] > counts["completed"]
        ),
        "last_active_ts_ms": max(
            (_index_int(getattr(run, "last_active_ts_ms", 0)) for run in runs),
            default=0,
        ),
    }


@inspector_bp.route("/")
def index():
    # Root page is benchmark-first and lazy: list benchmarks only (cheap dir
    # counts, zero run JSON read). Per-run detail loads on /benchmark/<project>.
    benchmarks = list_benchmarks(current_app.config["ROOT_DIR"])
    return render_template(
        "index.html",
        benchmarks=benchmarks,
        total_runs=sum(b.run_count for b in benchmarks),
        root=current_app.config["ROOT_DIR"],
    )


@inspector_bp.route("/benchmark/<path:project>")
def benchmark_detail(project: str):
    # Scope the scan to one benchmark so opening it never pays to summarize
    # every other benchmark's runs (the old whole-tree scan on the root page).
    runs = scan_runs_for_project(current_app.config["ROOT_DIR"], project)
    if not runs:
        abort(404)
    _install_request_run_url_index(runs)
    run_groups = group_runs(runs)
    active_runs = [run for run in runs if run.running]
    index_summary = _build_index_summary(runs, active_runs)
    run_filter_options = {
        "agents": sorted({run.agent_name for run in runs if run.agent_name}),
        "models": sorted({run.model_name for run in runs if run.model_name}),
    }
    return render_template(
        "benchmark.html",
        project=project,
        runs=runs,
        active_runs=active_runs,
        index_summary=index_summary,
        run_groups=run_groups,
        root=current_app.config["ROOT_DIR"],
        run_filter_options=run_filter_options,
    )


def _render_run_detail(run_dir: Path):
    # Live runs may not yet have dashboard.json — fall back to a
    # minimal stub built from planned_trials.json so the page can
    # still render while the agent is in flight.
    dashboard = load_dashboard(run_dir) or _stub_dashboard(run_dir)
    if not dashboard:
        abort(404)
    run_history = load_run_history(run_dir, dashboard=dashboard)
    trials_with_paths = _trial_rows_for_run(run_dir, dashboard)
    # Recompute the per-agent header counts from the actual on-disk
    # trial files we just classified, instead of trusting the
    # dashboard.json snapshot. The snapshot is a point-in-time write
    # by the orchestrator and can lag the filesystem after a resume
    # (or after an interrupt before the dashboard re-write ran). The
    # rule is: dashboard view always reflects the trials we can see
    # on disk right now.
    agent_counts = _aggregate_agent_counts(t["info"] for t in trials_with_paths)
    for agent_data in dashboard.get("agents", {}).values():
        agent_data.update(agent_counts)
    has_running = any(t["info"].get("running") for t in trials_with_paths)
    run_overview = _build_run_overview(run_dir, dashboard, trials_with_paths, run_history)
    has_review_work = any(
        t["info"].get("running")
        or str(t["info"].get("status_kind") or "") in {"warning", "live_success"}
        for t in trials_with_paths
    )
    trial_tag_options = sorted({
        tag
        for trial in trials_with_paths
        for tag in trial["info"].get("tags", [])
    }, key=lambda tag: (tag.lower(), tag))
    trial_status_options = sorted({
        str(trial["info"].get("status_kind") or "pending")
        for trial in trials_with_paths
    })
    preflight_summary = _load_preflight_summary(run_dir)
    # Breadcrumb coordinates: benchmark (project) / model / run_id. Empty for
    # out-of-tree runs, where the template falls back to a plain "Runs" crumb.
    project = _project_slug_for_run_dir(run_dir)
    _, model, _ = _parse_agent_label_for_view(run_dir.parent.name)
    return render_template(
        "run.html",
        dashboard=dashboard,
        trials=trials_with_paths,
        has_running=has_running,
        has_review_work=has_review_work,
        run_overview=run_overview,
        run_debug_bundle=_run_debug_bundle(run_dir, dashboard, run_overview, run_history),
        trial_tag_options=trial_tag_options,
        trial_status_options=trial_status_options,
        run_dir=run_dir,
        root=current_app.config["ROOT_DIR"],
        preflight=preflight_summary,
        run_history=run_history,
        project=project,
        model=model,
    )


@inspector_bp.route("/<benchmark>/<model>/<run_id>")
def run_detail_canonical(benchmark: str, model: str, run_id: str):
    run_dir = _find_benchmark_model_run(benchmark, model, run_id)
    if run_dir is None:
        abort(404)
    return _render_run_detail(run_dir)


@inspector_bp.route("/run/<encoded>")
def run_detail(encoded: str):
    # Encoded is the universal internal/fallback addressing. When the run has a
    # canonical /<benchmark>/<model>/<run_id> address, 301-redirect to it so the
    # address bar (and breadcrumb) show the readable form; out-of-tree runs with
    # no canonical address render in place.
    return _redirect_or_render_run(_decode_path(encoded))


@inspector_bp.route("/run/<agent_label>/<run_id>")
def run_detail_readable(agent_label: str, run_id: str):
    run_dir = _find_run_dir(agent_label, run_id)
    if run_dir is None:
        abort(404)
    return _redirect_or_render_run(run_dir)


@inspector_bp.route("/projects/<project>/runs/<run_id>")
def run_detail_project(project: str, run_id: str):
    run_dir = _find_project_run_dir(project, run_id)
    if run_dir is None:
        abort(404)
    return _redirect_or_render_run(run_dir)


def _render_dashboard(run_dir: Path):
    """Render the best available dashboard for a run."""
    manifest = load_dashboard(run_dir) or _stub_dashboard(run_dir)
    run_history = load_run_history(run_dir, dashboard=manifest)
    trials_with_paths = _trial_rows_for_run(run_dir, manifest)
    agent_counts = _aggregate_agent_counts(t["info"] for t in trials_with_paths)
    for agent_data in manifest.get("agents", {}).values():
        if isinstance(agent_data, dict):
            agent_data.update(agent_counts)
    run_overview = _build_run_overview(run_dir, manifest, trials_with_paths, run_history)
    view, dashboard_mode = _best_dashboard_view(run_dir, manifest)
    dashboard_freshness = _dashboard_freshness(run_dir, view)
    dashboard_insight = _dashboard_insight(view, dashboard_freshness)
    dashboard_error = _dashboard_view_artifact_error(run_dir)
    return render_template(
        "dashboard.html",
        run_dir=run_dir,
        root=current_app.config["ROOT_DIR"],
        dashboard=manifest,
        view=view,
        dashboard_mode=dashboard_mode,
        dashboard_freshness=dashboard_freshness,
        dashboard_insight=dashboard_insight,
        dashboard_error=dashboard_error,
        run_overview=run_overview,
        dashboard_debug_bundle=_dashboard_debug_bundle(
            run_dir,
            manifest,
            run_overview,
            dashboard_mode,
            dashboard_freshness,
            dashboard_insight,
            dashboard_error,
        ),
    )


@inspector_bp.route("/<benchmark>/<model>/<run_id>/dashboard")
def dashboard_view_canonical(benchmark: str, model: str, run_id: str):
    run_dir = _find_benchmark_model_run(benchmark, model, run_id)
    if run_dir is None:
        abort(404)
    return _render_dashboard(run_dir)


@inspector_bp.route("/run/<encoded>/dashboard")
def dashboard_view(encoded: str):
    return _render_dashboard(_decode_path(encoded))


@inspector_bp.route("/run/<agent_label>/<run_id>/dashboard")
def dashboard_view_readable(agent_label: str, run_id: str):
    run_dir = _find_run_dir(agent_label, run_id)
    if run_dir is None:
        abort(404)
    return _redirect_or_render_dashboard(run_dir)


@inspector_bp.route("/projects/<project>/runs/<run_id>/dashboard")
def dashboard_view_project(project: str, run_id: str):
    run_dir = _find_project_run_dir(project, run_id)
    if run_dir is None:
        abort(404)
    return _redirect_or_render_dashboard(run_dir)


@inspector_bp.route("/api/run/<encoded>/dashboard_view")
def api_dashboard_view(encoded: str):
    return _dashboard_view_json(_decode_path(encoded))


@inspector_bp.route("/api/projects/<project>/runs/<run_id>/dashboard_view")
def api_dashboard_view_project(project: str, run_id: str):
    run_dir = _find_project_run_dir(project, run_id)
    if run_dir is None:
        abort(404)
    return _dashboard_view_json(run_dir)


def _dashboard_view_json(run_dir: Path):
    manifest = load_dashboard(run_dir) or _stub_dashboard(run_dir)
    view, dashboard_mode = _best_dashboard_view(run_dir, manifest)
    dashboard_error = _dashboard_view_artifact_error(run_dir)
    if view is None:
        payload = {
            "present": False,
            "error": dashboard_error,
            "max_signature_ms": _dashboard_signature_ms(run_dir),
            "now_ms": int(time.time() * 1000),
        }
        return jsonify(payload), 200 if dashboard_error else 404
    freshness = _dashboard_freshness(run_dir, view)
    signature_ms = int(freshness.get("signature_ms") or 0)
    try:
        since_ms = int(request.args.get("since") or 0)
    except (TypeError, ValueError):
        since_ms = 0
    changed = not since_ms or signature_ms > since_ms
    return jsonify({
        "present": True,
        "changed": changed,
        "view": view if changed else None,
        "mode": dashboard_mode,
        "freshness": freshness,
        "error": dashboard_error,
        "max_signature_ms": signature_ms,
        "now_ms": int(time.time() * 1000),
    })


def _render_trial_detail(trial_dir: Path):
    trial = load_trial(trial_dir)
    # Find the run dir by anchoring on the ``trials/`` segment in the path.
    # Trial dirs always live under ``<run>/trials/...`` (flat / nested /
    # passk variants). The previous walk-up relied on ``dashboard.json``
    # but that file is written at run end, so in-progress runs would
    # overshoot one level to ``<agent_label>`` and produce broken
    # ``/run/<agent_label>`` breadcrumb links.
    run_dir = _resolve_run_dir(trial_dir)
    dashboard = load_dashboard(run_dir)
    # Find this trial's dashboard entry for non-usage fields (timing etc.).
    dashboard_trial: dict[str, Any] = {}
    for agent_data in dashboard.get("agents", {}).values():
        for t in agent_data.get("trials", []):
            if t.get("trial_id") == trial.trial_id:
                dashboard_trial = t
                break
    # Usage MUST come from on-disk progress.json — dashboard.json is a
    # snapshot the orchestrator writes at run-end / on Ctrl+C and goes
    # stale after archive↔live swaps or mid-run inspection. progress.json
    # is updated by the in-container proxy on every request, so it's
    # the authoritative live token/request counter.
    usage = _trial_usage_from_disk(trial_dir, fallback=dashboard_trial.get("usage", {}))
    trial_status = load_trial_summary(
        trial_dir,
        dashboard_trial,
        run_status=str(dashboard.get("status", "") or ""),
    )
    # meta.json on disk wins over dashboard.json cache — the latter is a
    # run-time snapshot that can lag behind out-of-band rewrites
    # (reclassify scripts, manual edits) without anyone refreshing it.
    termination = build_trial_termination({**dashboard_trial, **trial.meta})
    live_check = load_live_check_evidence(trial_dir)
    attempts = load_resume_attempts(trial_dir)
    attempt_links = [
        {
            "label": a.label,
            "is_current": a.is_current,
            "is_live": a.is_live,
            "display_label": "Current attempt" if a.is_live else f"Archived {a.label}",
            "status": a.status,
            "termination_reason": a.termination_reason,
            "reason_label": a.reason_label,
            "reason_kind": a.reason_kind,
            "url": _trial_url_for(a.trial_dir, run_dir),
        }
        for a in attempts
    ]
    if not attempt_links:
        attempt_links = [{
            "label": "current",
            "is_current": True,
            "is_live": bool(trial_status.get("running")),
            "display_label": "Current attempt",
            "status": trial_status.get("status_label", ""),
            "termination_reason": termination.get("reason", ""),
            "reason_label": trial_status.get("status_label", ""),
            "reason_kind": trial_status.get("status_kind", "pending"),
            "url": _trial_url_for(trial_dir, run_dir),
        }]
    trial_diagnosis = _build_trial_diagnosis(trial_status, termination, trial)
    trial_outcome = _build_trial_benchmark_outcome(trial)
    token_context = _trial_token_context(run_dir, trial_dir, usage, dashboard=dashboard)
    return render_template(
        "trial.html",
        trial=trial,
        trial_dir=trial_dir,
        trial_status=trial_status,
        termination=termination,
        trial_diagnosis=trial_diagnosis,
        trial_outcome=trial_outcome,
        token_context=token_context,
        usage=usage,
        live_check=live_check,
        root=current_app.config["ROOT_DIR"],
        run_dir=run_dir,
        attempts=attempt_links,
        trial_debug_bundle=_trial_debug_bundle(
            trial_dir,
            run_dir,
            trial,
            trial_status,
            termination,
            usage,
            trial_diagnosis,
            trial_outcome,
        ),
    )


def _trial_summary_signature_ms(trial_dir: Path) -> int:
    signature_ns = max(trial_summary_signature(trial_dir))
    if signature_ns <= 0:
        return 0
    return signature_ns // 1_000_000


def _dashboard_trial_for_trial(
    dashboard: dict[str, Any],
    trial_id: str,
) -> dict[str, Any]:
    for agent_data in dashboard.get("agents", {}).values():
        if not isinstance(agent_data, dict):
            continue
        for item in agent_data.get("trials", []) or []:
            if isinstance(item, dict) and item.get("trial_id") == trial_id:
                return item
    return {}


def _trial_summary_payload(trial_dir: Path) -> dict[str, Any]:
    trial = load_trial(trial_dir)
    run_dir = _resolve_run_dir(trial_dir)
    dashboard = load_dashboard(run_dir)
    dashboard_trial = _dashboard_trial_for_trial(dashboard, trial.trial_id)
    usage = _trial_usage_from_disk(
        trial_dir,
        fallback=dashboard_trial.get("usage", {}),
    )
    trial_status = load_trial_summary(
        trial_dir,
        dashboard_trial,
        run_status=str(dashboard.get("status", "") or ""),
    )
    termination = build_trial_termination({**dashboard_trial, **trial.meta})
    duration_ms = (
        trial_status.get("duration_ms")
        or trial.meta.get("timing", {}).get("duration_ms", 0)
        or 0
    )
    return {
        "trial_id": trial.trial_id,
        "running": bool(trial_status.get("running")),
        "status_label": trial_status.get("status_label", "Pending"),
        "status_detail": trial_status.get("status_detail", ""),
        "status_kind": trial_status.get("status_kind", "pending"),
        "duration_ms": duration_ms,
        "exit_code": trial.meta.get("exit_code", "-"),
        "usage": usage,
        "termination": termination,
        "diagnosis": _build_trial_diagnosis(trial_status, termination, trial),
        "benchmark_outcome": _build_trial_benchmark_outcome(trial),
    }


def _trial_file_diagnostics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    files = [
        entry
        for entry in entries
        if not entry.get("is_dir") and entry.get("relative_path")
    ]
    by_path = {str(entry["relative_path"]): entry for entry in files}

    def first_match(
        paths: tuple[str, ...] = (),
        *,
        kinds: tuple[str, ...] = (),
        contains: tuple[str, ...] = (),
        suffixes: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        if kinds:
            for entry in files:
                if str(entry.get("artifact_kind") or "") in kinds:
                    return entry
        for path in paths:
            if path in by_path:
                return by_path[path]
        if suffixes:
            for entry in files:
                rel = str(entry.get("relative_path") or "")
                name = str(entry.get("name") or "")
                if any(rel.endswith(suffix) or name.endswith(suffix) for suffix in suffixes):
                    return entry
        if contains:
            for entry in files:
                rel = str(entry.get("relative_path") or "")
                if all(part in rel for part in contains):
                    return entry
        return None

    specs = [
        (
            "progress",
            "Progress",
            "Live request, token, and error counters.",
            {"paths": ("proxy/progress.json", "progress.json")},
        ),
        (
            "metadata",
            "Metadata",
            "Trial id, exit code, timing, and termination fields.",
            {"paths": ("meta.json",)},
        ),
        (
            "final_output",
            "Final output",
            "Captured final answer or agent stream.",
            {"kinds": ("task_output",), "paths": ("task_output.json",)},
        ),
        (
            "score",
            "Score / eval",
            "Benchmark scoring artifact when present.",
            {
                "kinds": ("trial_score",),
                "suffixes": ("score.json", "scores.json", "eval.json", "evaluation.json"),
            },
        ),
        (
            "target_inspect",
            "Target inspect",
            "Target state snapshot from the run.",
            {"suffixes": ("target_inspect.json",)},
        ),
        (
            "target_logs",
            "Target logs",
            "Target server or service logs.",
            {"suffixes": ("target_server.log",), "contains": ("target",)},
        ),
        (
            "proxy_trace",
            "Proxy trace",
            "Full model/proxy event stream.",
            {
                "kinds": ("proxy_log", "proxy_jsonl"),
                "paths": ("proxy/proxy.jsonl",),
                "suffixes": ("proxy.jsonl",),
            },
        ),
    ]
    diagnostics = []
    large_threshold = 10 * 1024 * 1024
    for key, label, note, query in specs:
        entry = first_match(
            tuple(query.get("paths", ())),
            kinds=tuple(query.get("kinds", ())),
            contains=tuple(query.get("contains", ())),
            suffixes=tuple(query.get("suffixes", ())),
        )
        if not entry:
            continue
        size_bytes = int(entry.get("size_bytes") or 0)
        is_large = size_bytes >= large_threshold
        item_note = note
        if is_large:
            item_note = f"Large file; download instead of opening inline. {note}"
        diagnostics.append({
            "key": key,
            "label": label,
            "relative_path": entry.get("relative_path", ""),
            "size_bytes": size_bytes,
            "size_label": entry.get("size_label", ""),
            "download_url": entry.get("download_url", ""),
            "is_large": is_large,
            "note": item_note,
        })
    return diagnostics


def _resolve_trial_candidate(run_dir: Path, trial_rel: str) -> Path:
    """Resolve + sandbox a ``<run_dir>/trials/<trial_rel>`` request, or abort."""
    if ".." in Path(trial_rel).parts:
        abort(403)
    trials_root = (run_dir / "trials").resolve()
    candidate = (trials_root / trial_rel).resolve()
    if not _is_relative_to(candidate, trials_root):
        abort(403)
    if not is_known_trial_path(candidate):
        abort(404)
    return candidate


@inspector_bp.route("/<benchmark>/<model>/<run_id>/<path:trial_rel>")
def trial_detail_canonical(benchmark: str, model: str, run_id: str, trial_rel: str):
    run_dir = _find_benchmark_model_run(benchmark, model, run_id)
    if run_dir is None:
        abort(404)
    return _render_trial_detail(_resolve_trial_candidate(run_dir, trial_rel))


@inspector_bp.route("/trial/<encoded>")
def trial_detail(encoded: str):
    # Redirect to the canonical /<benchmark>/<model>/<run_id>/<trial_rel> form
    # when the trial has one, so the address bar shows the readable URL; encoded
    # stays the fallback for out-of-tree trials with no canonical address.
    trial_dir = _decode_path(encoded)
    return _redirect_or_render_trial(trial_dir, _resolve_run_dir(trial_dir))


@inspector_bp.route("/trial/<agent_label>/<run_id>/<path:trial_rel>")
def trial_detail_readable(agent_label: str, run_id: str, trial_rel: str):
    run_dir = _find_run_dir(agent_label, run_id)
    if run_dir is None:
        abort(404)
    return _redirect_or_render_trial(_resolve_trial_candidate(run_dir, trial_rel), run_dir)


@inspector_bp.route("/projects/<project>/runs/<run_id>/trials/<path:trial_rel>")
def trial_detail_project(project: str, run_id: str, trial_rel: str):
    run_dir = _find_project_run_dir(project, run_id)
    if run_dir is None:
        abort(404)
    return _redirect_or_render_trial(_resolve_trial_candidate(run_dir, trial_rel), run_dir)


@inspector_bp.route("/trial/<encoded>/download/<file_encoded>")
def download_trial_file(encoded: str, file_encoded: str):
    trial_dir = _decode_path(encoded)
    file_path = _decode_path(file_encoded)
    is_legacy_trial_file = _is_relative_to(file_path, trial_dir)
    is_canonical_trial_artifact = is_indexed_trial_artifact_path(
        trial_dir,
        file_path,
    )
    if not is_legacy_trial_file and not is_canonical_trial_artifact:
        abort(403)
    if not file_path.is_file():
        abort(404)
    return send_file(file_path, as_attachment=True, download_name=file_path.name)


@inspector_bp.route("/trial/<encoded>/zip/<dir_encoded>")
def download_trial_dir_zip(encoded: str, dir_encoded: str):
    """Stream a zip archive of a folder inside a trial.

    Sandbox: the folder must resolve under the trial dir. Names in the
    archive are rooted at the folder itself (so unzipping recreates
    ``<folder_name>/...``, not the full absolute path). Files that
    disappear mid-walk are skipped; symlinks are followed (the trial
    tree shouldn't contain user-controlled symlinks).
    """
    import io
    import zipfile

    from flask import Response

    trial_dir = _decode_path(encoded)
    folder = _decode_path(dir_encoded)
    is_legacy_trial_dir = _is_relative_to(folder, trial_dir)
    is_canonical_trial_artifact = is_indexed_trial_artifact_path(
        trial_dir,
        folder,
    )
    if not is_legacy_trial_dir and not is_canonical_trial_artifact:
        abort(403)
    if not folder.is_dir():
        abort(404)

    def _iter_files(base: Path):
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    yield p
            except OSError:
                continue

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in _iter_files(folder):
            try:
                arcname = Path(folder.name) / file_path.relative_to(folder)
                zf.write(file_path, arcname=str(arcname))
            except OSError:
                continue
    buf.seek(0)
    download_name = f"{folder.name}.zip"
    return Response(
        buf.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


@inspector_bp.route("/api/runs")
def api_runs():
    """Live-poll endpoint for the index page.

    Pass ``?since=<ms>`` to get only runs whose summary signature
    is newer than ``since``. The first call should omit the
    parameter (or send 0) to get the full set.
    """
    since_ms = request.args.get("since", 0, type=int)
    # ``?project=<slug>`` scopes the poll to one benchmark's runs. The per-benchmark
    # page only renders that project's run cards, so an unscoped (whole-tree) delta
    # would return runs with no card on the page and force a reload every tick.
    project = (request.args.get("project") or "").strip()
    root = current_app.config["ROOT_DIR"]
    runs = scan_runs_for_project(root, project) if project else scan_runs(root)
    _warm_run_lookup_indexes_from_runs(runs)
    return jsonify(build_runs_delta(runs, root=root, since_ms=since_ms))


@inspector_bp.route("/api/run/<encoded>/trials")
def api_run_trials(encoded: str):
    """Live-poll endpoint for the run detail page.

    Returns only trials whose mtime signature has advanced past
    ``since_ms``; cheap-poll friendly via the per-trial cache.
    """
    run_dir = _decode_path(encoded)
    since_ms = request.args.get("since", 0, type=int)
    return jsonify(build_run_trials_delta(
        run_dir,
        root=current_app.config["ROOT_DIR"],
        since_ms=since_ms,
        trial_url=lambda trial_dir: _trial_url_for(trial_dir, run_dir),
    ))


@inspector_bp.route("/api/trial/<encoded>/summary")
def api_trial_summary(encoded: str):
    trial_dir = _decode_path(encoded)
    if not is_known_trial_path(trial_dir):
        abort(404)
    signature_ms = _trial_summary_signature_ms(trial_dir)
    since_ms = request.args.get("since", 0, type=int)
    changed = not since_ms or signature_ms > since_ms
    return jsonify({
        "present": True,
        "changed": changed,
        "summary": _trial_summary_payload(trial_dir) if changed else None,
        "max_signature_ms": signature_ms,
        "now_ms": int(time.time() * 1000),
    })


@inspector_bp.route("/api/trial/<encoded>/files")
def api_trial_files(encoded: str):
    trial_dir = _decode_path(encoded)
    entries = build_trial_file_tree(trial_dir)
    payload = []
    for entry in entries:
        item = {
            "relative_path": entry.relative_path,
            "name": entry.name,
            "depth": entry.depth,
            "is_dir": entry.is_dir,
            "size_bytes": entry.size_bytes,
            "size_label": entry.size_label,
            "artifact_kind": entry.artifact_kind,
        }
        if entry.is_dir:
            item["download_zip_url"] = (
                f"/trial/{encoded}/zip/{encode_path(entry.path)}"
            )
        else:
            item["download_url"] = (
                f"/trial/{encoded}/download/{encode_path(entry.path)}"
            )
        payload.append(item)
    total_size_bytes = sum(
        int(item.get("size_bytes") or 0)
        for item in payload
        if not item.get("is_dir")
    )
    large_file_count = sum(
        1
        for item in payload
        if not item.get("is_dir") and int(item.get("size_bytes") or 0) >= 10 * 1024 * 1024
    )
    return jsonify({
        "count": len(payload),
        "diagnostics": _trial_file_diagnostics(payload),
        "entries": payload,
        "large_file_count": large_file_count,
        "total_size_bytes": total_size_bytes,
        "total_size_label": _format_file_size(total_size_bytes),
    })


@inspector_bp.route("/api/trajectory/<encoded>")
def api_trajectory(encoded: str):
    trial_dir = _decode_path(encoded)
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 50, type=int)
    data = parse_trial_trajectory(trial_dir, offset=offset, limit=limit)
    return jsonify(data)


@inspector_bp.route("/api/trajectory/<encoded>/all")
def api_trajectory_all(encoded: str):
    """Load entire trajectory at once (for client-side search)."""
    trial_dir = _decode_path(encoded)
    data = parse_trial_trajectory(trial_dir, offset=0, limit=999999)
    return jsonify(data)


@inspector_bp.route("/api/trajectory/<encoded>/step/<int:step_index>")
def api_step_context(encoded: str, step_index: int):
    """Load full request context for a single step (on-demand)."""
    trial_dir = _decode_path(encoded)
    data = load_trial_step_context(trial_dir, step_index)
    if not data:
        abort(404)
    return jsonify(data)


def create_app(
    root_dir: Path,
    *,
    auth: WebInspectorAuthConfig | None = None,
    ui: WebInspectorUIConfig | None = None,
) -> Flask:
    """Create and configure the Flask app."""
    root = root_dir.resolve()
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["ROOT_DIR"] = root
    auth = auth or WebInspectorAuthConfig()
    ui = ui or WebInspectorUIConfig()
    if auth.enabled and not auth.token:
        raise ValueError("web inspector auth token must be non-empty when auth is enabled")
    app.config["CAGE_INSPECTOR_AUTH"] = auth
    app.config["CAGE_INSPECTOR_UI"] = ui
    app.jinja_env.globals["web_ui"] = ui
    app.config["CAGE_RUN_LOOKUP_CACHE"] = {
        "expires_at": 0.0,
        "by_agent": {},
        "by_project": {},
        "by_bmr": {},
    }
    app.jinja_env.globals["run_url"] = _run_url_for
    app.jinja_env.globals["trial_url"] = _trial_url_for
    app.jinja_env.globals["dashboard_url"] = _dashboard_url_for
    app.jinja_env.globals["dashboard_status_kind"] = _dashboard_status_kind
    app.jinja_env.globals["dashboard_status_label"] = _dashboard_status_label
    app.jinja_env.globals["dashboard_cell_display"] = _dashboard_cell_display
    app.jinja_env.globals["dashboard_cell_sort_value"] = _dashboard_cell_sort_value
    app.jinja_env.globals["dashboard_cell_attr"] = _dashboard_cell_attr
    app.register_blueprint(inspector_bp)
    return app

