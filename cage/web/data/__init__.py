"""Data loading for the web inspector.

Reads run artifacts (.cage_runs/) and parses proxy.jsonl into structured
trajectory data for the frontend.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cage.agents.codex.output import parse_codex_event_stream
from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.records import ResolvedTrialArtifact
from cage.artifacts.run_storage import (
    DASHBOARD_FILENAME,
    EXPERIMENT_RECORD_FILENAME,
    META_FILENAME,
    PROGRESS_FILENAME,
    PROMPT_FILENAME,
    PROXY_DIRNAME,
    PROXY_LOG_FILENAME,
    RESUME_ARCHIVE_MARKER,
    TASK_OUTPUT_FILENAME,
    TRIALS_DIRNAME,
    discover_run_dirs,
    is_trial_dir,
    iter_live_trial_dirs,
)
from cage.contracts.coerce import int_or_zero
from cage.contracts.trial_status import (
    COMPLETED_TRIAL_STATUSES,
    FAILED_TRIAL_STATUSES,
    INTERRUPTED_TRIAL_STATUSES,
)
from cage.experiment.engine.live.fs_signals import (
    dashboard_pending_count as _dashboard_pending_count,
)
from cage.experiment.model import (
    TrialRecord,
    experiment_record_to_mapping,
    experiment_spec_to_mapping,
)
from cage.proxy.conversations import ConversationForest, reconstruct_forest
from cage.proxy.trajectory import (
    _blocks_from_responses_items,
    _extract_blocks,
    _extract_response_blocks_from_body,
    _parse_tool_arguments,
)
from cage.proxy.usage import extract_entry_usage
from cage.web.cache import (
    RunFsSignals,
    discovery_cache,
    discovery_signature,
    get_or_compute,
    is_recently_active,
    run_history_cache,
    run_summary_cache,
    run_tools_cache,
    safe_mtime_ns,
    scan_run_signals,
    trial_summary_cache,
)

_ENTRY_CACHE_MAX = 4
_FULL_TRAJECTORY_LIMIT = 100_000
_TRAJECTORY_LOOKAHEAD_STEPS = 6
_RAW_PROXY_ARTIFACT_KINDS = frozenset({"proxy_log", "proxy_jsonl"})
RUN_HISTORY_FILE = "run_history.json"
_entry_cache: dict[Path, tuple[tuple[int, int], list[dict[str, Any]]]] = {}
_entry_cache_lock = threading.RLock()
_SCAN_PRUNE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".worktrees",
    "__pycache__",
    "node_modules",
    "venv",
}

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class RunInfo:
    """Summary of a single experiment run."""
    path: Path
    run_id: str
    experiment: str
    started_at: str
    completed_at: str
    project: str
    agents: dict[str, Any] = field(default_factory=dict)
    agent_label: str = ""
    agent_name: str = ""
    model_name: str = ""
    mode: str = ""
    running: bool = False
    running_trials: int = 0
    live_total_requests: int = 0
    live_errors: int = 0
    last_active_ts_ms: int = 0
    duration_ms: int = 0
    run_history: list[dict[str, Any]] = field(default_factory=list)
    sort_ts: int = 0
    # Single string status used by the index page filter/badge. Values:
    # "running"     — actively ticking right now
    # "completed"   — dashboard wrote completed_at and reported "completed"
    # "interrupted" — orchestrator exited (Ctrl+C / crash) before finishing
    # "pending"     — fallback (planned but no progress, very early startup)
    # Computed once in ``_build_run_info`` so the template doesn't have
    # to re-derive it from a chain of conditionals.
    status: str = "pending"


@dataclass
class BenchmarkSummary:
    """Cheap root-page summary for one benchmark (project).

    Holds only what ``GET /`` needs to render a benchmark card — name and run
    count — plus the discovered ``.cage_runs/`` dirs so a follow-up
    per-benchmark scan does not have to re-discover them. No per-run data.
    """
    project: str
    run_count: int
    cage_runs_dirs: list[Path] = field(default_factory=list)


@dataclass
class TrialDetail:
    """Full detail of a single trial loaded from files."""
    trial_id: str
    meta: dict[str, Any]
    prompt: str
    output: str
    sample: dict[str, Any]
    scores: dict[str, Any]
    has_trajectory: bool = False


@dataclass
class TrialFileEntry:
    """A single row in the trial artifact file tree.

    ``artifact_kind`` is populated only when the row came from the canonical
    ``ArtifactIndex``. Legacy filesystem rows leave it empty, allowing web
    diagnostics to prefer contract metadata while preserving filename-based
    fallback for older runs.
    """
    path: Path
    relative_path: str
    name: str
    depth: int
    is_dir: bool
    size_bytes: int = 0
    size_label: str = ""
    artifact_kind: str = ""


# ------------------------------------------------------------------
# Scanning & loading
# ------------------------------------------------------------------

def _runs_under_cage_runs_dir(cage_runs_dir: Path) -> list[RunInfo]:
    """Build ``RunInfo`` for every run under one ``.cage_runs/`` directory."""
    runs: list[RunInfo] = []
    project = cage_runs_dir.parent.name
    try:
        agent_dirs = sorted(cage_runs_dir.iterdir())
    except OSError:
        return runs
    for agent_dir in agent_dirs:
        if not agent_dir.is_dir():
            continue
        try:
            run_dirs = sorted(agent_dir.iterdir(), reverse=True)
        except OSError:
            continue
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            info = _load_run_info(run_dir, project=project, agent_dir_name=agent_dir.name)
            if info is None:
                continue
            runs.append(info)
    return runs


def _sort_runs(runs: list[RunInfo]) -> list[RunInfo]:
    return sorted(
        runs,
        key=lambda run: (not run.running, -run.sort_ts, run.project, run.agent_label),
    )


def scan_runs(root: Path) -> list[RunInfo]:
    """Find all experiment runs under *root* (recursively finds .cage_runs/).

    Returns ``RunInfo`` objects sorted with running runs first, then by
    most-recently-active. Both discovery and per-run summary are cached:
    repeat calls during a polling tick are O(number-of-runs) stat calls,
    not O(number-of-trials).
    """
    runs: list[RunInfo] = []
    for cage_runs_dir in _iter_cage_runs_dirs(root):
        runs.extend(_runs_under_cage_runs_dir(cage_runs_dir))
    return _sort_runs(runs)


def warm_run_history_cache(run_dir: Path) -> None:
    """Prime the run-history reconstruction cache for one run, off the request path.

    ``load_run_history`` reconstructs from a full per-trial ``meta.json`` walk
    when a run never persisted ``run_history.json`` (live/crashed/abandoned) —
    multi-second on a large run over NAS. The result is cached on the cheap
    structural run signature, so paying it here (background warmer) means the
    first detail-page open is already warm. Best-effort: never raises.

    Deliberately does *not* warm per-trial summaries: that cache is a bounded
    4096-entry LRU sized for interactive browsing, and bulk-priming every run's
    trials would thrash it and slow the pages it is meant to speed up.
    """
    try:
        load_run_history(run_dir)
    except Exception:
        pass


def scan_runs_for_project(root: Path, project: str) -> list[RunInfo]:
    """Find runs for a single benchmark (project) only — the per-benchmark page.

    Avoids the whole-tree :func:`scan_runs` so opening one benchmark never pays
    to summarize every other benchmark's runs. Only the ``.cage_runs/`` dirs
    whose parent directory name equals ``project`` are scanned.
    """
    runs: list[RunInfo] = []
    for cage_runs_dir in _iter_cage_runs_dirs(root):
        if cage_runs_dir.parent.name != project:
            continue
        runs.extend(_runs_under_cage_runs_dir(cage_runs_dir))
    return _sort_runs(runs)


def list_benchmarks(root: Path) -> list["BenchmarkSummary"]:
    """Cheaply enumerate benchmarks (projects) for the root page.

    The root page only needs the benchmark list plus a run count — NOT a full
    per-run summary. This reads zero run JSON and never descends into a run's
    ``trials/`` tree: for each discovered ``.cage_runs/`` it counts run dirs with
    two levels of ``iterdir`` (agent dirs → run dirs). That keeps ``GET /``
    O(number-of-benchmarks) instead of O(number-of-runs).
    """
    by_project: dict[str, BenchmarkSummary] = {}
    for cage_runs_dir in _iter_cage_runs_dirs(root):
        project = cage_runs_dir.parent.name
        summary = by_project.setdefault(
            project,
            BenchmarkSummary(project=project, run_count=0, cage_runs_dirs=[]),
        )
        summary.cage_runs_dirs.append(cage_runs_dir)
        try:
            agent_dirs = list(cage_runs_dir.iterdir())
        except OSError:
            continue
        for agent_dir in agent_dirs:
            if not agent_dir.is_dir():
                continue
            try:
                summary.run_count += sum(1 for rd in agent_dir.iterdir() if rd.is_dir())
            except OSError:
                continue
    return sorted(
        by_project.values(),
        key=lambda b: (-b.run_count, b.project),
    )


def _iter_cage_runs_dirs(root: Path) -> list[Path]:
    """Find the ``.cage_runs/`` directories under *root*.

    Canonical layout: runs live in the inspect root itself
    (``<project>/.cage_runs/``) or exactly one level below it
    (``<root>/<project>/.cage_runs/`` — e.g. every ``examples/<benchmark>/``).
    Only these two depths are scanned, so the inspector never descends into a
    project's ``datasets/``, vendored submodules or image caches: gigabytes that
    never hold runs and used to make a cold index load take many seconds (a full
    ``os.walk`` of ``examples/`` did not even return within 10s here).

    A top-level ``.cage_runs/`` that holds only target-server logs contributes
    no runs and is therefore ignored downstream, so it never hides the real
    per-project runs one level below it.

    Deeper, non-conforming trees can opt back into the full recursive walk with
    ``CAGE_INSPECT_RECURSIVE=1``.
    """
    root = root.resolve()
    if root.name == ".cage_runs":
        return [root]

    if os.environ.get("CAGE_INSPECT_RECURSIVE"):
        return get_or_compute(
            discovery_cache,
            root,
            discovery_signature(),
            lambda: _recursive_cage_runs_dirs(root),
        )

    # Canonical two-level discovery is owned by the storage layout authority.
    return discover_run_dirs(root)


def _recursive_cage_runs_dirs(root: Path) -> list[Path]:
    """Full recursive ``.cage_runs/`` discovery (``CAGE_INSPECT_RECURSIVE=1``)."""
    found: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if ".cage_runs" in dirnames:
            found.append(Path(dirpath) / ".cage_runs")
            dirnames.remove(".cage_runs")
        dirnames[:] = sorted(d for d in dirnames if d not in _SCAN_PRUNE_DIRS)
    return found


# Signals for a settled run: no in-flight trials, nothing recently active. Used
# as a stand-in so completed runs skip the expensive ``scan_run_signals`` walk.
_EMPTY_RUN_SIGNALS = RunFsSignals(
    completed_count=0,
    active_count=0,
    newest_progress_mtime_ns=0,
    progress_files=(),
    completed_dirs=(),
    active_dirs=(),
)


def _run_recently_active(run_dir: Path, *artifact_mtimes_ns: int) -> bool:
    """Cheap 'is this run plausibly still live?' from run-level mtimes only.

    A live run keeps touching its run-level artifacts (the canonical record,
    dashboard, run-history, planned-trials) as trials complete, and adding a
    trial dir bumps ``run_dir``'s own mtime. If none of those moved within the
    live window the run is settled, and the expensive ``scan_run_signals`` walk
    over ``trials/`` — its only effect on a settled run being empty signals — can
    be skipped. Pure stat()s already taken by the caller, plus one for run_dir.
    """
    newest = safe_mtime_ns(run_dir)
    for mtime in artifact_mtimes_ns:
        if mtime > newest:
            newest = mtime
    return is_recently_active(newest)


def _shallow_run_signature(run_dir: Path) -> tuple[int, ...]:
    """Cheap structural fingerprint of a run's trial layout.

    Stats only ``run_dir`` and its ``trials/`` dir(s) — never descends into
    individual trial dirs. Adding a trial or resuming bumps one of these
    mtimes; an in-place ``progress.json`` tick inside an existing trial does
    not — but that only matters for *running* runs, which are never cached on
    this signature. Lets a settled run that never wrote ``dashboard.json``
    (crashed/abandoned) be served from cache without the hundreds-of-syscalls
    deep walk ``scan_run_signals`` performs — the walk that made a cold
    ``/api/runs`` over NAS-backed ``.cage_runs`` take minutes.
    """
    parts: list[int] = [safe_mtime_ns(run_dir)]
    direct = run_dir / "trials"
    if direct.is_dir():
        parts.append(safe_mtime_ns(direct))
    else:
        # Legacy nested layout: <session>/trials/, <mode>/trials/, ...
        try:
            children = sorted(run_dir.iterdir())
        except OSError:
            children = []
        for child in children:
            trials = child / "trials"
            if trials.is_dir():
                parts.append(safe_mtime_ns(child))
                parts.append(safe_mtime_ns(trials))
    return tuple(parts)


def _load_run_info(
    run_dir: Path,
    *,
    project: str,
    agent_dir_name: str,
) -> "RunInfo | None":
    """Build a ``RunInfo`` for one run dir, cached on fs signature.

    Two-level keying so completed runs don't pay for a deep trial-tree
    walk on every poll:

    * If ``dashboard.json`` reports ``status`` in {completed, interrupted}
      the cheap signature ``(dashboard_mtime, planned_mtime)`` is enough.
      That hit path is two ``stat`` calls per run — under 0.1 ms.
    * Otherwise (no dashboard yet, or a "running" status), we run the
      cheap-stat walk over ``trials/`` to detect activity. That's ~50 ms
      for a 320-trial active run on local disk.
    """
    dashboard_path = run_dir / DASHBOARD_FILENAME
    record_path = run_dir / EXPERIMENT_RECORD_FILENAME
    history_path = run_dir / RUN_HISTORY_FILE
    planned_path = run_dir / "planned_trials.json"
    dashboard_mtime = safe_mtime_ns(dashboard_path)
    record_mtime = safe_mtime_ns(record_path)
    history_mtime = safe_mtime_ns(history_path)
    planned_mtime = safe_mtime_ns(planned_path)
    if dashboard_mtime < 0 and planned_mtime < 0 and record_mtime < 0:
        return None

    # Fast path — completed runs have stable contents. Cheap signature
    # plus a lookup; on hit we skip the deep walk entirely.
    if dashboard_mtime >= 0:
        cheap_sig = ("done", dashboard_mtime, record_mtime, history_mtime, planned_mtime)
        cached = run_summary_cache.get(run_dir, cheap_sig)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    # Fast path for runs that never produced a dashboard.json (crashed or
    # abandoned mid-run). Once such a run has settled — last activity older than
    # the live window — its trials/ tree is frozen, so a shallow structural
    # fingerprint is enough to serve the previously-built RunInfo without the
    # expensive deep walk. Live/pre-settle runs miss here and fall through to a
    # real scan_run_signals so their "running" badge stays accurate.
    nodash_sig: tuple[Any, ...] | None = None
    if dashboard_mtime < 0:
        nodash_sig = (
            "settled-nodash",
            record_mtime,
            history_mtime,
            planned_mtime,
            _shallow_run_signature(run_dir),
        )
        cached = run_summary_cache.get(run_dir, nodash_sig)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

    dashboard_data = _load_json(dashboard_path) if dashboard_mtime >= 0 else {}
    if dashboard_mtime >= 0 and dashboard_data.get("completed_at"):
        # Settled run: the dashboard recorded completion, so the trials/ tree is
        # final and holds no in-flight work. Skip scan_run_signals — on
        # NAS-backed .cage_runs its hundreds of stat/scandir syscalls cost ~12s
        # per run, which made a cold ``examples/`` index (every benchmark's runs)
        # take minutes. Empty signals yield an identical RunInfo here because a
        # completed run has ``active_count == 0`` (see the ``completed_at`` branch
        # in _build_run_info's running logic).
        signals = _EMPTY_RUN_SIGNALS
    elif record_mtime >= 0:
        # Canonical record present: run-level trial counters come from it, and a
        # cheap record-mtime-based "running" signal is derived in _build_run_info
        # (the ``record_present and record_pending and is_recently_active``
        # branch). So the trials/ deep walk — thousands of scandir/stat syscalls,
        # ~9s on NAS for a 2600-trial run — is never needed on the list/index
        # path: a benchmark with dozens of such runs was re-walking every one on
        # every page load. The live *detail* page (polled) still shows precise
        # per-trial state; the list only needs counts + a running badge.
        signals = _EMPTY_RUN_SIGNALS
    elif not _run_recently_active(
        run_dir, dashboard_mtime, record_mtime, history_mtime, planned_mtime
    ):
        # No canonical record and no ``completed_at``, and nothing at the run
        # level moved within the live window — a crashed/abandoned legacy run.
        # Its trials/ tree is frozen, so the walk would only return empty
        # signals; skip it. A recently-active legacy run (no record to count
        # from) still falls through to the walk below.
        signals = _EMPTY_RUN_SIGNALS
    else:
        signals = scan_run_signals(run_dir)
    info = _build_run_info(
        run_dir,
        project=project,
        agent_dir_name=agent_dir_name,
        signals=signals,
        dashboard_present=dashboard_mtime >= 0,
        record_present=record_mtime >= 0,
        record_mtime_ns=record_mtime,
        planned_mtime_ns=planned_mtime,
        dashboard_data=dashboard_data,
    )

    if dashboard_mtime >= 0 and not info.running:
        # Settled run — stamp the cheap signature so we never deep-walk
        # this directory again until its dashboard mtime moves.
        run_summary_cache.put(
            run_dir,
            ("done", dashboard_mtime, record_mtime, history_mtime, planned_mtime),
            info,
        )
    elif nodash_sig is not None and not info.running:
        # Dashboard-less but settled (crashed/abandoned). Stamp the shallow
        # structural fingerprint so subsequent polls skip the deep trials/ walk
        # until a new trial or a resume bumps the fingerprint.
        run_summary_cache.put(run_dir, nodash_sig, info)
    else:
        # Live or pre-dashboard run — sign on progress + structure
        # mtimes so the entry naturally invalidates as activity ticks.
        # The 30s wall-clock bucket lets the "running" flag flip to
        # false when activity stops without us having to detect each
        # missing tick.
        signature = (
            "live",
            dashboard_mtime,
            record_mtime,
            history_mtime,
            planned_mtime,
            signals.completed_count,
            signals.active_count,
            signals.newest_progress_mtime_ns,
            time.time_ns() // 30_000_000_000,
        )
        run_summary_cache.put(run_dir, signature, info)
    return info


def _build_run_info(
    run_dir: Path,
    *,
    project: str,
    agent_dir_name: str,
    signals: RunFsSignals,
    dashboard_present: bool,
    record_present: bool,
    record_mtime_ns: int,
    planned_mtime_ns: int,
    dashboard_data: dict[str, Any] | None = None,
) -> "RunInfo":
    """Construct a RunInfo from cheap signals + (at most) one dashboard read.

    Callers that have already parsed ``dashboard.json`` (e.g. to detect a
    completed run) pass it via *dashboard_data* to avoid a second read.
    """
    agent_name, model_name, mode = _parse_agent_label(agent_dir_name)

    data: dict[str, Any] = {}
    if dashboard_present:
        if dashboard_data is not None:
            data = dashboard_data
        else:
            data = _load_json(run_dir / DASHBOARD_FILENAME)
    record_data, spec_data = _load_canonical_record_payloads(run_dir, record_present)

    run_id = str(data.get("run_id") or record_data.get("run_id") or run_dir.name)
    experiment = str(data.get("experiment") or _record_experiment_name(spec_data) or "")
    started_at = str(
        data.get("started_at")
        or record_data.get("started_at")
        or record_data.get("created_at")
        or ""
    )
    completed_at = str(data.get("completed_at") or record_data.get("completed_at") or "")
    agents_payload = data.get("agents") if isinstance(data.get("agents"), dict) else {}
    agents_payload = _merge_record_agent_payload(
        agents_payload,
        agent_label=agent_dir_name,
        record=record_data,
    )
    agents_payload = _merge_record_score_summary_payload(
        agents_payload,
        agent_label=agent_dir_name,
        run_dir=run_dir,
        record=record_data,
    )

    if not started_at and planned_mtime_ns > 0:
        started_at = datetime.fromtimestamp(planned_mtime_ns / 1e9).strftime(
            "%Y-%m-%dT%H:%M:%S",
        )
    run_history = load_run_history(run_dir, dashboard=data, reconstruct=False)
    if run_history:
        first_run = run_history[0]
        latest_run = run_history[-1]
        if first_run.get("started_at"):
            started_at = str(first_run["started_at"])
        if latest_run.get("completed_at"):
            completed_at = str(latest_run["completed_at"])

    # The agent-side ``last_ts_ms`` from progress.json is the canonical
    # "logical last activity" timestamp; the file mtime is a fallback
    # for cases where the JSON hasn't recorded a per-request stamp yet.
    last_active_ts_ms = 0
    fallback_mtime = signals.last_active_ms

    dashboard_pending = _dashboard_pending_count(data)
    record_pending = _record_pending_count(record_data)
    completed_ts_ns = _timestamp_ms(completed_at) * 1_000_000
    active_after_completed = (
        bool(completed_at)
        and signals.active_count > 0
        and signals.newest_progress_mtime_ns > completed_ts_ns
        and is_recently_active(signals.newest_progress_mtime_ns)
    )
    # "Running" iff there's pending work AND either the orchestrator
    # didn't write completed_at OR something has ticked recently. We
    # honour both signals (filesystem progress files + dashboard's
    # pending counter) so very-early startup (dashboard written but no
    # progress.json yet) still reports as running.
    if completed_at and not active_after_completed:
        running = False
    elif signals.active_count > 0 and is_recently_active(signals.newest_progress_mtime_ns):
        running = True
    elif signals.active_count > 0:
        # Has progress files but nothing has ticked recently — treat as
        # not running so we don't show a stale orchestrator as alive.
        running = False
    elif dashboard_present and dashboard_pending > 0:
        # Dashboard reports pending trials but no progress.json exists
        # yet — orchestrator is between trial start and first request.
        running = True
    elif record_present and record_pending > 0 and is_recently_active(record_mtime_ns):
        # Canonical record exists but legacy dashboard/planned artifacts may not
        # have been written yet. Treat a fresh planned record as startup/live
        # signal, mirroring the planned_trials.json fallback.
        running = True
    else:
        # No active progress files yet — might be very early in startup.
        # Treat as running if planned_trials.json is fresh and we have
        # no dashboard yet.
        running = (
            not dashboard_present
            and planned_mtime_ns > 0
            and is_recently_active(planned_mtime_ns)
        )

    # Aggregate live counts come from per-trial progress.json reads.
    # Each file is ~200 bytes and individually cached on its own mtime.
    # Run-level LLM calls count successful budgeted requests. ``total_requests``
    # includes failed upstream attempts and can drift from the orchestrator's
    # step accounting.
    total_requests = 0
    errors = 0
    if signals.active_count > 0:
        for progress_path in signals.progress_files:
            progress = _read_progress_cached(progress_path)
            if not progress:
                continue
            success = progress.get("successful_requests")
            if success is None:
                success = progress.get("success", progress.get("total_requests"))
            total_requests += int_or_zero(success)
            errors += int_or_zero(progress.get("errors"))
            last_active_ts_ms = max(
                last_active_ts_ms,
                int_or_zero(progress.get("last_ts_ms")),
            )

    if last_active_ts_ms == 0:
        last_active_ts_ms = fallback_mtime

    started_ts = _timestamp_sort_value(started_at, run_dir)
    latest_run_ts = 0
    duration_ms = _run_duration_ms(data, run_dir, last_active_ts_ms)
    if run_history:
        latest_run = run_history[-1]
        duration_ms = int_or_zero(latest_run.get("duration_ms")) or duration_ms
        latest_run_ts = (
            _timestamp_ms(latest_run.get("completed_at"))
            or _timestamp_ms(latest_run.get("started_at"))
        )

    # Single coarse-grained status used for index filtering + badge
    # rendering. The previous template-side chain
    # ``running ? running : completed_at ? completed : pending`` mis-classified
    # any run that died before writing dashboard.json (Ctrl+C at startup,
    # crashed preflight, etc.) as "pending" — surfacing them as if they
    # were still queued. Compute the truth here.
    raw_dashboard_status = str(data.get("status") or "").strip().lower()
    raw_record_status = str(record_data.get("status") or "").strip().lower()
    if running:
        status = "running"
    elif completed_at:
        status = raw_dashboard_status or "completed"
        if status not in {"completed", "interrupted", "failed", "cancelled"}:
            status = "completed"
    elif raw_dashboard_status in {"interrupted", "failed", "cancelled"}:
        # Dashboard written but no completed_at — preserve its self-reported
        # terminal state.
        status = raw_dashboard_status
    elif raw_record_status in {"completed", "interrupted", "failed", "cancelled"}:
        # Canonical ExperimentRecord may be the only durable run-level artifact.
        status = raw_record_status
    elif planned_mtime_ns > 0:
        # planned_trials.json exists (orchestrator started) but no
        # dashboard.json + no recent activity → orchestrator died before
        # finishing. Surface as interrupted, not pending.
        status = "interrupted"
    elif record_mtime_ns > 0:
        # Same policy as planned_trials.json: a stale planned ExperimentRecord
        # means the run began but never reached a terminal dashboard/record.
        status = "interrupted"
    else:
        status = "pending"

    if not record_data:
        # Legacy runs have no ExperimentRecord, so recompute from on-disk
        # meta.json instead of trusting dashboard.json. Canonical runs skip
        # this path because TrialRecord counts are the durable run truth and
        # stale legacy trial dirs must not overwrite them.
        agents_payload = _agents_payload_recount_from_disk(
            agents_payload,
            agent_label=agent_dir_name,
            run_dir=run_dir,
        )

    return RunInfo(
        path=run_dir,
        run_id=run_id,
        experiment=experiment,
        started_at=started_at,
        completed_at=completed_at,
        project=project,
        agents=agents_payload,
        agent_label=agent_dir_name,
        agent_name=agent_name,
        model_name=model_name,
        mode=mode,
        running=running,
        status=status,
        running_trials=(
            (signals.active_count or dashboard_pending or record_pending)
            if running else 0
        ),
        live_total_requests=total_requests,
        live_errors=errors,
        last_active_ts_ms=last_active_ts_ms,
        duration_ms=duration_ms,
        run_history=run_history,
        sort_ts=(
            last_active_ts_ms
            if running and last_active_ts_ms
            else latest_run_ts or started_ts
        ),
    )


def _load_canonical_record_payloads(
    run_dir: Path,
    record_present: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load canonical ExperimentRecord and referenced ExperimentSpec payloads.

    ``ExperimentArtifactReader.load_snapshot`` is the shared read boundary for
    canonical run truth. The web index still returns plain mappings internally
    because older dashboard merge helpers expect JSON-shaped payloads; converting
    from the typed contracts here keeps that compatibility boundary narrow.

    Bad, incomplete, or historical canonical artifacts fall back to the legacy
    direct JSON reads so old dashboard/planned paths render instead of crashing
    the index page.
    """

    if not record_present:
        return {}, {}
    # Read ONLY the run-level record + spec (one small file each), never a full
    # snapshot. ``load_snapshot`` pulls the artifact_index + EVERY trial record +
    # all events — thousands of files / tens of seconds on NAS for a large run
    # (a 2600-trial run measured at ~45s) — but the run-list summary needs only
    # the record's top-level status, trial counters, and score_summary, which
    # ``load_record`` returns directly. This is the same cheap read the
    # detail-page stub uses; the run-list path must not be heavier than it.
    try:
        reader = ExperimentArtifactReader(run_dir)
        record_obj = reader.load_record()
        spec_obj = reader.load_spec()
    except Exception:
        record_obj = None
        spec_obj = None
    if record_obj is not None:
        return (
            experiment_record_to_mapping(record_obj),
            experiment_spec_to_mapping(spec_obj) if spec_obj is not None else {},
        )
    # Fallback: raw JSON for bad/incomplete/historical canonical artifacts, so a
    # damaged record renders the legacy dashboard/planned view instead of 500ing.
    record = _load_json(run_dir / EXPERIMENT_RECORD_FILENAME)
    if not isinstance(record, dict):
        return {}, {}
    spec_ref = str(record.get("spec_ref") or "experiment_spec.json")
    spec = _load_json(run_dir / spec_ref)
    return record, spec if isinstance(spec, dict) else {}


def _record_experiment_name(spec: dict[str, Any]) -> str:
    """Return the experiment display name from a serialized ExperimentSpec."""

    identity = spec.get("identity") if isinstance(spec.get("identity"), dict) else {}
    return str(
        identity.get("experiment_id")
        or identity.get("display_name")
        or ""
    )


def _merge_record_agent_payload(
    agents_payload: dict[str, Any],
    *,
    agent_label: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Merge canonical record trial counters into the per-agent payload.

    Dashboard data still wins for rich fields, but a record-only run needs enough
    counters for the index page to show the run and its planned workload.
    """

    if not record:
        return agents_payload
    trials = record.get("trials") if isinstance(record.get("trials"), dict) else {}
    if not trials:
        return agents_payload
    payload = dict(agents_payload) if isinstance(agents_payload, dict) else {}
    base = payload.get(agent_label) if isinstance(payload.get(agent_label), dict) else {}
    base = dict(base)
    total = int_or_zero(trials.get("total"))
    completed = int_or_zero(trials.get("completed"))
    failed = int_or_zero(trials.get("failed"))
    interrupted = int_or_zero(trials.get("interrupted"))
    base["total"] = total
    base["completed"] = completed
    base["failed"] = failed
    base["interrupted"] = interrupted
    base["running"] = max(0, total - completed - failed - interrupted)
    payload[agent_label] = base
    return payload


def _merge_record_score_summary_payload(
    agents_payload: dict[str, Any],
    *,
    agent_label: str,
    run_dir: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Merge canonical run score summary means into the index agent payload.

    The index template already knows how to render ``agents[*].mean_scores``
    from legacy dashboards. Offline ``cage score`` now records equivalent
    aggregate data through ``ExperimentRecord.score_summary.summary_ref``. This
    helper keeps the schema translation at the web compatibility boundary:
    templates do not need to understand canonical record internals, and legacy
    dashboard-provided metrics still win when both sources provide a value.
    """

    means = _load_record_score_summary_means(run_dir, record)
    if not means:
        return agents_payload
    payload = dict(agents_payload) if isinstance(agents_payload, dict) else {}
    base = payload.get(agent_label) if isinstance(payload.get(agent_label), dict) else {}
    base = dict(base)
    existing = (
        base.get("mean_scores")
        if isinstance(base.get("mean_scores"), dict)
        else {}
    )
    merged = dict(existing)
    for metric, value in means.items():
        merged.setdefault(metric, value)
    base["mean_scores"] = merged
    payload[agent_label] = base
    return payload


def _load_record_score_summary_means(
    run_dir: Path,
    record: dict[str, Any],
) -> dict[str, float]:
    """Load numeric means from an indexed canonical score summary artifact."""

    summary = (
        record.get("score_summary")
        if isinstance(record.get("score_summary"), dict)
        else {}
    )
    ref = str(summary.get("summary_ref") or "").strip()
    if not ref:
        return {}
    reader = ExperimentArtifactReader(run_dir)
    try:
        artifact = reader.find_artifact(path=ref, kind="score_summary")
        if artifact is None:
            return {}
        ref_path = reader.resolve_artifact_path(artifact)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return {}
    data = _load_json(ref_path)
    scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    means: dict[str, float] = {}
    for metric, payload in scores.items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("mean")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        means[str(metric)] = float(value)
    return means


def _record_pending_count(record: dict[str, Any]) -> int:
    """Return pending trial count from a serialized ExperimentRecord."""

    trials = record.get("trials") if isinstance(record.get("trials"), dict) else {}
    if not trials:
        return 0
    total = int_or_zero(trials.get("total"))
    completed = int_or_zero(trials.get("completed"))
    failed = int_or_zero(trials.get("failed"))
    interrupted = int_or_zero(trials.get("interrupted"))
    return max(0, total - completed - failed - interrupted)


def _agents_payload_recount_from_disk(
    agents_payload: dict[str, Any],
    *,
    agent_label: str,
    run_dir: Path,
    preserve_when_no_trials: bool = False,
) -> dict[str, Any]:
    """Override total/completed/failed with on-disk truth.

    Walks every trial dir under the run (via ``find_trial_dirs`` — same
    source as the run-page banner), reads each meta.json, and buckets
    by status_kind (success/live_success/error) using the same
    classification table. Keeps other payload fields (mean_scores,
    pass_at_k, etc.) intact so the home page still shows them unchanged.

    A run dir always belongs to exactly one agent_label (it sits under
    ``.cage_runs/<agent_label>/<run_id>/``), so we recompute under that
    single key and leave any unexpected sibling keys alone.

    Crucially we count trial dirs that lack ``task_output.json`` (e.g.
    ``target_unavailable`` failures where the agent never ran): they
    still carry a meta.json with a terminal classification and the
    run-banner counts them, so the home card must too.
    """
    completed_count = 0
    failed_count = 0
    warning_count = 0
    running_count = 0
    total = 0
    for trial_dir in find_trial_dirs(run_dir):
        total += 1
        meta_path = trial_dir / META_FILENAME
        meta = _load_json(meta_path) if meta_path.is_file() else {}
        # No task_output.json AND no terminal meta → trial is still in flight.
        terminal = bool(meta.get("termination_reason")) or (
            trial_dir / TASK_OUTPUT_FILENAME
        ).is_file()
        if not terminal:
            running_count += 1
            continue
        reason = str(meta.get("termination_reason") or "").strip().lower()
        reason = _REASON_ALIASES.get(reason, reason)
        if bool(meta.get("live_success")):
            completed_count += 1
            continue
        config = _REASON_DISPLAY.get(reason)
        kind = (config or {}).get("kind", "")
        if kind in {"success", "live_success"}:
            completed_count += 1
        elif kind == "error":
            failed_count += 1
        elif kind == "warning":
            # Budget-bound stops (rounds, token/cost budgets, timeout,
            # tool_limit, user_interrupted): not a clean success but also
            # not an error.
            # Surfaced as its own bucket so completed+failed+warnings+running
            # adds up to total on the home card.
            warning_count += 1
        else:
            # Genuinely unclassified — count toward the failure bucket so
            # nothing silently drops out of the total.
            failed_count += 1

    payload = dict(agents_payload) if isinstance(agents_payload, dict) else {}
    if total == 0 and preserve_when_no_trials:
        return payload
    base = payload.get(agent_label) if isinstance(payload.get(agent_label), dict) else {}
    base = dict(base)
    base["total"] = total
    base["completed"] = completed_count
    base["failed"] = failed_count
    base["warnings"] = warning_count
    base["running"] = running_count
    payload[agent_label] = base
    return payload


def _read_progress_cached(progress_path: Path) -> dict[str, Any]:
    """Cached read of one ``progress.json``. Mtime-keyed; tiny payloads."""
    mtime = safe_mtime_ns(progress_path)
    if mtime < 0:
        return {}

    def _compute() -> dict[str, Any]:
        return _load_json(progress_path)

    return get_or_compute(trial_summary_cache, progress_path, ("progress", mtime), _compute)


def group_runs(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """Group runs for the index page by project and model."""
    project_map: dict[str, dict[str, Any]] = {}
    for run in runs:
        project_group = project_map.setdefault(
            run.project,
            {
                "project": run.project,
                "models": {},
                "running_count": 0,
                "total_count": 0,
            },
        )
        project_group["total_count"] += 1
        if run.running:
            project_group["running_count"] += 1

        model_key = run.model_name or "unknown"
        model_group = project_group["models"].setdefault(
            model_key,
            {
                "model_name": model_key,
                "agents": set(),
                "modes": set(),
                "runs": [],
                "running_count": 0,
                "total_count": 0,
            },
        )
        model_group["agents"].add(run.agent_name or run.agent_label or "unknown")
        if run.mode:
            model_group["modes"].add(run.mode)
        model_group["runs"].append(run)
        model_group["total_count"] += 1
        if run.running:
            model_group["running_count"] += 1

    grouped: list[dict[str, Any]] = []
    for project_group in project_map.values():
        models = []
        for model_group in project_group["models"].values():
            model_group["runs"].sort(
                key=lambda run: (not run.running, -run.sort_ts, run.run_id)
            )
            model_group["agents"] = sorted(model_group["agents"])
            model_group["modes"] = sorted(model_group["modes"])
            models.append(model_group)
        models.sort(
            key=lambda group: (
                -group["running_count"],
                str(group["model_name"]),
            )
        )
        project_group["models"] = models
        grouped.append(project_group)

    grouped.sort(key=lambda group: (-group["running_count"], str(group["project"])))
    return grouped


def _parse_agent_label(label: str) -> tuple[str, str, str]:
    parts = label.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], ":".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return label, "", ""


# ``_dashboard_pending_count`` is imported at the top alongside other
# cage imports — re-pointed here as a back-compat shim for old callers.


def _timestamp_sort_value(started_at: str, run_dir: Path) -> int:
    text = str(started_at or "").strip()
    if text:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return int(run_dir.stat().st_mtime * 1000)


def _timestamp_ms(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0


def _derive_duration_ms(started_at: Any, completed_at: Any) -> int:
    """Wall-clock trial duration in ms from its start/finish timestamps.

    The canonical ``TrialRecord`` stores only ``started_at``/``completed_at`` (no
    duration field), so the row template's ``duration_ms`` must be derived here —
    otherwise every finished trial renders ``-`` for duration. Returns 0 when
    either timestamp is missing/unparseable or the span is negative.
    """
    start = _timestamp_ms(started_at)
    end = _timestamp_ms(completed_at)
    return end - start if (start and end and end >= start) else 0


def _run_duration_ms(data: dict[str, Any], run_dir: Path, last_active_ts_ms: int = 0) -> int:
    explicit = int_or_zero(data.get("duration_ms"))
    if explicit:
        return explicit
    timing = data.get("timing")
    if isinstance(timing, dict):
        explicit = int_or_zero(timing.get("duration_ms"))
        if explicit:
            return explicit
    start = _timestamp_ms(data.get("started_at"))
    end = _timestamp_ms(data.get("completed_at")) or int_or_zero(last_active_ts_ms)
    if start and end and end >= start:
        return end - start
    return 0


def load_run_history(
    run_dir: Path,
    *,
    dashboard: dict[str, Any] | None = None,
    reconstruct: bool = True,
) -> list[dict[str, Any]]:
    """Return run-level invocation history for one ``run_id``.

    New runs persist ``run_history.json`` so the inspector can show every
    invocation even though ``dashboard.json`` remains a latest-snapshot file.
    Older runs do not have that file, so we reconstruct best-effort windows
    from live and ``.before_resume_*`` trial attempts.
    """
    dashboard = dashboard or {}
    history_path = run_dir / RUN_HISTORY_FILE
    raw: Any = None
    if history_path.is_file():
        raw = _load_json(history_path)
    if not raw and isinstance(dashboard.get("run_history"), dict):
        raw = dashboard.get("run_history")
    recorded = _normalize_recorded_run_history(raw)
    if recorded:
        return recorded

    if reconstruct:
        # Reconstruction reads every trial's (and every resume archive's)
        # meta.json off disk — thousands of NAS reads for a large run, repeated
        # on every page load. Cache on the cheap structural run signature: a
        # settled run hits cache forever; a live run only recomputes when a
        # trial dir is added or a resume happens (not on in-trial ticks, which
        # don't move history windows meaningfully).
        reconstructed = get_or_compute(
            run_history_cache,
            run_dir,
            _shallow_run_signature(run_dir),
            lambda: _reconstruct_run_history_from_trial_attempts(run_dir),
        )
        if reconstructed:
            return reconstructed

    fallback = _fallback_run_history_from_dashboard(dashboard)
    return fallback


def _normalize_recorded_run_history(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    attempts_raw = raw.get("attempts")
    if not isinstance(attempts_raw, list):
        return []

    attempts: list[dict[str, Any]] = []
    for index, item in enumerate(attempts_raw):
        if not isinstance(item, dict):
            continue
        started_at = str(item.get("started_at") or "")
        completed_at = str(item.get("completed_at") or "")
        if not started_at and not completed_at:
            continue
        start_ms = _timestamp_ms(started_at)
        end_ms = _timestamp_ms(completed_at)
        duration_ms = int_or_zero(item.get("duration_ms"))
        if not duration_ms and start_ms and end_ms and end_ms >= start_ms:
            duration_ms = end_ms - start_ms
        sequence = int_or_zero(item.get("sequence")) or len(attempts) + 1
        entry = {
            "sequence": sequence,
            "label": str(item.get("label") or _run_history_label(len(attempts))),
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "status": str(item.get("status") or ""),
            "trial_attempts": int_or_zero(item.get("trial_attempts")),
            "completed_trials": int_or_zero(item.get("completed_trials")),
            "failed_trials": int_or_zero(item.get("failed_trials")),
            "source": str(item.get("source") or "recorded"),
            "is_latest": False,
        }
        attempts.append(entry)

    attempts.sort(
        key=lambda e: (int_or_zero(e.get("sequence")), _timestamp_ms(e.get("started_at"))),
    )
    for index, entry in enumerate(attempts):
        entry["sequence"] = index + 1
        entry["label"] = _run_history_label(index)
        entry["is_latest"] = index == len(attempts) - 1
    return attempts


def _fallback_run_history_from_dashboard(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    started_at = str(dashboard.get("started_at") or "")
    completed_at = str(dashboard.get("completed_at") or "")
    if not started_at and not completed_at:
        return []
    start_ms = _timestamp_ms(started_at)
    end_ms = _timestamp_ms(completed_at)
    duration_ms = end_ms - start_ms if start_ms and end_ms and end_ms >= start_ms else 0
    return [
        {
            "sequence": 1,
            "label": "Initial run",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "status": str(dashboard.get("status") or ""),
            "trial_attempts": _dashboard_trial_count(dashboard),
            "completed_trials": 0,
            "failed_trials": 0,
            "source": "dashboard",
            "is_latest": True,
        }
    ]


def _dashboard_trial_count(dashboard: dict[str, Any]) -> int:
    total = 0
    agents = dashboard.get("agents")
    if not isinstance(agents, dict):
        return 0
    for agent in agents.values():
        if isinstance(agent, dict):
            total += int_or_zero(agent.get("total"))
    return total


def _iter_trial_meta_paths(trials_root: Path):
    """Yield trial ``meta.json`` paths without walking trial workspaces.

    ``meta.json`` lives at a fixed shallow depth — ``trials/<trial>/meta.json``
    or ``trials/<trial>/<attempt>/meta.json`` (pass@k). A naive
    ``rglob("meta.json")`` descends into every trial's ``workspace/`` tree,
    which can hold hundreds of thousands of files and multi-GB artifacts; that
    turns a per-render call into a 10s+ stall. Scan only the bounded depths and
    never recurse into ``workspace``.
    """
    try:
        trial_dirs = [p for p in trials_root.iterdir() if p.is_dir()]
    except OSError:
        return
    for trial_dir in trial_dirs:
        direct = trial_dir / META_FILENAME
        if direct.is_file():
            yield direct
        try:
            children = [p for p in trial_dir.iterdir() if p.is_dir()]
        except OSError:
            continue
        for attempt_dir in children:
            if attempt_dir.name == "workspace":
                continue
            nested = attempt_dir / META_FILENAME
            if nested.is_file():
                yield nested


# Bounded thread pool for the per-attempt meta.json walk during history
# reconstruction. Latency-bound NAS reads, so 16 is plenty; overridable.
_HISTORY_SCAN_WORKERS = max(
    4, int(os.environ.get("CAGE_INSPECT_TRIAL_WORKERS", "16") or "16")
)


def _reconstruct_run_history_from_trial_attempts(run_dir: Path) -> list[dict[str, Any]]:
    trials_root = run_dir / TRIALS_DIRNAME
    if not trials_root.is_dir():
        return []

    # Read every attempt's meta.json concurrently — the walk is dominated by
    # per-file NAS read latency, not parsing, so a small thread pool collapses a
    # serial scan of hundreds of trials to roughly its slowest single read. The
    # JSON parse and aggregation below stay sequential (cheap, order-independent).
    meta_paths = list(_iter_trial_meta_paths(trials_root))
    if not meta_paths:
        return []
    workers = min(_HISTORY_SCAN_WORKERS, len(meta_paths))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = pool.map(_load_json, meta_paths)
    else:
        loaded = map(_load_json, meta_paths)

    attempts_by_trial: dict[str, list[dict[str, Any]]] = {}
    for meta_path, meta in zip(meta_paths, loaded):
        attempt_dir = meta_path.parent
        if not isinstance(meta, dict):
            continue
        timing = meta.get("timing") if isinstance(meta.get("timing"), dict) else {}
        start_ms = int_or_zero(timing.get("started_at_ms"))
        end_ms = int_or_zero(timing.get("ended_at_ms"))
        duration_ms = int_or_zero(timing.get("duration_ms"))
        if not start_ms and not end_ms and not duration_ms:
            continue
        key = _logical_trial_attempt_key(trials_root, attempt_dir)
        attempts_by_trial.setdefault(key, []).append(
            {
                "dir": attempt_dir,
                "started_at_ms": start_ms,
                "ended_at_ms": end_ms,
                "duration_ms": duration_ms,
                "status": str(meta.get("status") or ""),
                "sort_key": _attempt_chronological_key(attempt_dir),
            }
        )

    buckets: dict[int, list[dict[str, Any]]] = {}
    for attempts in attempts_by_trial.values():
        attempts.sort(key=lambda attempt: attempt["sort_key"])
        for index, attempt in enumerate(attempts):
            buckets.setdefault(index, []).append(attempt)

    entries: list[dict[str, Any]] = []
    for index in sorted(buckets):
        bucket = buckets[index]
        starts = [
            int_or_zero(a.get("started_at_ms"))
            for a in bucket
            if int_or_zero(a.get("started_at_ms"))
        ]
        ends = [
            int_or_zero(a.get("ended_at_ms"))
            for a in bucket
            if int_or_zero(a.get("ended_at_ms"))
        ]
        start_ms = min(starts) if starts else 0
        end_ms = max(ends) if ends else 0
        duration_ms = (
            end_ms - start_ms
            if start_ms and end_ms and end_ms >= start_ms
            else 0
        )
        if not duration_ms:
            duration_ms = sum(int_or_zero(a.get("duration_ms")) for a in bucket)
        statuses = Counter(str(a.get("status") or "unknown") for a in bucket)
        completed_trials = (
            statuses.get("completed", 0)
            + statuses.get("success", 0)
        )
        failed_trials = sum(
            count
            for status, count in statuses.items()
            if status not in {"completed", "success", "unknown", ""}
        )
        entries.append(
            {
                "sequence": index + 1,
                "label": _run_history_label(index),
                "started_at": _format_ts_ms(start_ms),
                "completed_at": _format_ts_ms(end_ms),
                "duration_ms": duration_ms,
                "status": _history_status(statuses),
                "trial_attempts": len(bucket),
                "completed_trials": completed_trials,
                "failed_trials": failed_trials,
                "source": "reconstructed",
                "is_latest": False,
            }
        )

    for index, entry in enumerate(entries):
        entry["is_latest"] = index == len(entries) - 1
    return entries


def _logical_trial_attempt_key(trials_root: Path, attempt_dir: Path) -> str:
    rel = attempt_dir.relative_to(trials_root)
    parts = list(rel.parts)
    if not parts:
        return ""
    parts[-1] = parts[-1].split(RESUME_ARCHIVE_MARKER, 1)[0]
    return "/".join(parts)


def _attempt_chronological_key(attempt_dir: Path) -> tuple[int, str]:
    if RESUME_ARCHIVE_MARKER not in attempt_dir.name:
        return (1, "")
    tail = attempt_dir.name.split(RESUME_ARCHIVE_MARKER, 1)[1]
    return (0, tail)


def _history_status(statuses: Counter[str]) -> str:
    clean = Counter({k: v for k, v in statuses.items() if k and k != "unknown"})
    if not clean:
        return "unknown"
    if len(clean) == 1:
        return next(iter(clean))
    if any(
        status in clean
        for status in FAILED_TRIAL_STATUSES | INTERRUPTED_TRIAL_STATUSES
    ):
        return "mixed"
    return clean.most_common(1)[0][0]


def _run_history_label(index: int) -> str:
    return "Initial run" if index == 0 else f"Resume #{index}"


def _format_ts_ms(ts_ms: int) -> str:
    if not ts_ms:
        return ""
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%dT%H:%M:%S")


def load_dashboard(run_dir: Path) -> dict[str, Any]:
    """Load dashboard.json for a run."""
    f = run_dir / DASHBOARD_FILENAME
    if not f.exists():
        return {}
    dashboard = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(dashboard, dict):
        return {}
    return _overlay_record_dashboard_payload(run_dir, dashboard)


def _overlay_record_dashboard_payload(
    run_dir: Path,
    dashboard: dict[str, Any],
) -> dict[str, Any]:
    """Overlay canonical run facts onto a legacy dashboard payload.

    ``dashboard.json`` is now a compatibility projection: useful for older rich
    fields such as trial rows, but not the authority for run lifecycle facts.
    The canonical ``ExperimentRecord`` owns run identity, lifecycle timestamps,
    status, and trial counters. Keeping that translation at this boundary lets
    existing templates consume the familiar dashboard-shaped mapping while
    stale dashboards cannot override the durable record.
    """

    record_data, spec_data = _load_canonical_record_payloads(
        run_dir,
        (run_dir / EXPERIMENT_RECORD_FILENAME).is_file(),
    )
    if not record_data:
        return dashboard

    merged = dict(dashboard)
    run_id = str(record_data.get("run_id") or "").strip()
    if run_id:
        merged["run_id"] = run_id
    experiment = _record_experiment_name(spec_data)
    if experiment:
        merged["experiment"] = experiment
    started_at = str(record_data.get("started_at") or "").strip()
    if started_at:
        merged["started_at"] = started_at
    completed_at = str(record_data.get("completed_at") or "").strip()
    if completed_at:
        merged["completed_at"] = completed_at
    status = str(record_data.get("status") or "").strip().lower()
    if status and status != "planned":
        merged["status"] = status

    agent_label = run_dir.parent.name
    agents_payload = (
        merged.get("agents")
        if isinstance(merged.get("agents"), dict)
        else {}
    )
    agents_payload = _merge_record_agent_payload(
        agents_payload,
        agent_label=agent_label,
        record=record_data,
    )
    agents_payload = _merge_record_score_summary_payload(
        agents_payload,
        agent_label=agent_label,
        run_dir=run_dir,
        record=record_data,
    )
    merged["agents"] = agents_payload
    return merged


def load_preflight_summary(run_dir: Path) -> dict[str, Any] | None:
    """Load preflight.json and normalize for template rendering."""
    f = run_dir / "preflight.json"
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    passed = int(raw.get("passed", 0))
    failed = int(raw.get("failed", 0))
    warnings = int(raw.get("warnings", 0))
    if failed > 0:
        kind = "error"
    elif warnings > 0:
        kind = "warning"
    else:
        kind = "success"
    return {
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "kind": kind,
        "details": list(raw.get("details", [])),
    }


_NATURAL_KEY_SPLIT = re.compile(r"(\d+)")


def _natural_path_key(p: Path) -> list:
    """Sort key that treats embedded integers as numbers, not text.

    Without this, ``sorted(...)`` orders ``range-10`` between ``range-1`` and
    ``range-2`` because it's lexicographic. The inspector relies on this
    ordering to surface trials in the natural ``challenge → level → pass``
    order users expect (post-exp / main bench can both hit double-digit
    challenge indices).
    """
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in _NATURAL_KEY_SPLIT.split(p.name)
    ]


def find_trial_dirs(run_dir: Path) -> list[Path]:
    """Find all *live* trial directories under a run.

    Supports the flat layout (run-xxx/trials/<id>/), the nested
    layout with a per-challenge subdirectory (run-xxx/trials/<challenge>/<variant>/)
    used by benchmarks that emit a `variant` sample key, and the legacy
    layout with a mode subdirectory (run-xxx/stateless/trials/).

    Resume archives (``<id>.before_resume_<ts>/`` siblings) are normally
    *not* returned here — they're past attempts of the same logical
    trial, not independent trials. The trial-detail page surfaces them
    as an "Attempts" bar so users can navigate to a specific archive on
    demand.

    Exception: if a parent directory has only archive children and no
    live sibling (e.g. a resume was interrupted right after archiving
    the previous attempt, before the new live dir was created), the
    most recent archive is surfaced as the row so the trial does not
    silently disappear from the dashboard. The invariant we maintain is
    *every logical trial gets at least one row*.

    The scan itself is owned by the storage layout authority; this remains
    the web-facing name.
    """
    return iter_live_trial_dirs(run_dir)


def load_planned_trial_records(run_dir: Path) -> list[dict[str, Any]]:
    """Read planned trial rows recorded by the orchestrator.

    Canonical ``ExperimentRecord`` refs are the run truth for current runs.
    ``planned_trials.json`` is now a compatibility projection for older runs or
    incomplete migrations. The inspector uses these rows to render
    not-yet-started trials instead of making the list jump as directories
    appear, but stale projections must not mask canonical trial status/scores.
    """
    canonical = _load_canonical_trial_records(run_dir)
    if canonical:
        return canonical

    planned_path = run_dir / "planned_trials.json"
    raw = _load_json(planned_path)
    if isinstance(raw, list):
        planned: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                trial_id = str(item.get("trial_id") or item.get("id") or "").strip()
                if not trial_id:
                    continue
                record = dict(item)
            elif isinstance(item, str):
                trial_id = item.strip()
                if not trial_id:
                    continue
                record = {"trial_id": trial_id}
            else:
                continue
            record["trial_id"] = trial_id
            record.setdefault("trial_index", index)
            record.setdefault("_signature_ns", safe_mtime_ns(planned_path))
            planned.append(record)
        return planned
    return []


def _load_canonical_trial_records(run_dir: Path) -> list[dict[str, Any]]:
    """Read planned/current trial rows from canonical ExperimentRecord refs."""

    snapshot = _cached_run_snapshot(run_dir)
    if snapshot is not None:
        record_refs = {
            trial_ref.trial_id: trial_ref.record_ref
            for trial_ref in snapshot.record.trials.records
        }
        return [
            _canonical_trial_record_row_from_contract(
                run_dir=run_dir,
                trial_record=trial_record,
                record_ref=record_refs.get(trial_record.trial_id, ""),
                trial_index=index,
            )
            for index, trial_record in enumerate(snapshot.trial_records)
        ]

    record_path = run_dir / EXPERIMENT_RECORD_FILENAME
    run_record = _load_json(record_path)
    trials = run_record.get("trials") if isinstance(run_record.get("trials"), dict) else {}
    refs = trials.get("records") if isinstance(trials, dict) else []
    if not isinstance(refs, list):
        return []
    records: list[dict[str, Any]] = []
    record_mtime = safe_mtime_ns(record_path)
    for index, item in enumerate(refs):
        if not isinstance(item, dict):
            continue
        trial_id = str(item.get("trial_id") or "").strip()
        record_ref = str(item.get("record_ref") or "").strip()
        if not trial_id or not record_ref:
            continue
        trial_path = run_dir / record_ref
        trial_record = _load_json(trial_path)
        if not isinstance(trial_record, dict):
            trial_record = {}
        termination = (
            trial_record.get("termination")
            if isinstance(trial_record.get("termination"), dict)
            else {}
        )
        scoring = (
            trial_record.get("scoring")
            if isinstance(trial_record.get("scoring"), dict)
            else {}
        )
        score_ref = str(scoring.get("score_ref") or "").strip()
        row = {
            **trial_record,
            "trial_id": str(trial_record.get("trial_id") or trial_id),
            "trial_index": index,
            "sample_id": str(
                trial_record.get("sample_id")
                or trial_record.get("task_id")
                or trial_id
            ),
            "record_ref": record_ref,
            "termination_reason": termination.get("reason"),
            "exit_code": termination.get("exit_code"),
            "scores": _load_score_ref_scores(run_dir, score_ref),
            "_signature_ns": max(record_mtime, safe_mtime_ns(trial_path)),
        }
        records.append(row)
    return records


def ordered_canonical_trial_refs(run_dir: Path) -> list[dict[str, Any]] | None:
    """Cheap ordered trial list from the *run-level* record alone.

    Returns ``[{"id", "record_ref", "sort_index"}, ...]`` in planned order, or
    ``None`` if the run is not canonical (no ``experiment_record.json`` with
    trial refs). One ~100 KB JSON parse — it never reads per-trial record files
    or the full snapshot, so it is milliseconds even for a 1500-trial run. This
    is the index the detail page paginates over: order + identity for every
    trial, with per-trial status/scores deferred to :func:`enrich_canonical_trial_rows`
    for only the rows actually rendered.
    """
    record_path = run_dir / EXPERIMENT_RECORD_FILENAME
    if not record_path.is_file():
        return None
    try:
        record = ExperimentArtifactReader(run_dir).load_record()
    except Exception:
        return None
    refs: list[dict[str, Any]] = []
    for index, trial_ref in enumerate(record.trials.records):
        trial_id = str(trial_ref.trial_id or "").strip()
        if not trial_id:
            continue
        refs.append({
            "id": trial_id,
            "record_ref": str(trial_ref.record_ref or ""),
            "sort_index": index,
        })
    return refs


def enrich_canonical_trial_rows(
    run_dir: Path,
    refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build full display rows for a *slice* of :func:`ordered_canonical_trial_refs`.

    Reads only the given slice's per-trial record files (concurrently — the walk
    is NAS-latency-bound), producing rows byte-identical to the full
    ``_trial_rows_for_run`` canonical path. This is what makes a paginated detail
    page cheap: render 60 rows by reading 60 records, not 1500.
    """
    if not refs:
        return []
    reader = ExperimentArtifactReader(run_dir)

    def _row(ref: dict[str, Any]) -> dict[str, Any] | None:
        try:
            trial_record = reader.load_trial_record(ref["record_ref"])
        except Exception:
            return None
        record_row = _canonical_trial_record_row_from_contract(
            run_dir=run_dir,
            trial_record=trial_record,
            record_ref=ref["record_ref"],
            trial_index=ref["sort_index"],
        )
        info = pending_trial_summary(record_row)
        trial_dir = planned_trial_dir(run_dir, ref["id"])
        info["has_artifacts"] = is_trial_dir(trial_dir)
        # The canonical record carries no model-call counts, so overlay the
        # per-trial progress/usage projection. Without this the detail page's
        # first (lazy-rendered) slice shows "-" for steps until a background
        # scan warms it; reading the ~200B progress.json for the visible rows
        # makes steps appear the moment the page opens.
        if info.get("has_artifacts"):
            activity = load_trial_activity(trial_dir)
            if activity:
                info.update(activity)
        return {
            "dir": trial_dir,
            "info": info,
            "id": ref["id"],
            "sort_index": ref["sort_index"],
        }

    if len(refs) == 1:
        rows = [_row(refs[0])]
    else:
        workers = min(_HISTORY_SCAN_WORKERS, len(refs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_row, refs))
    return [row for row in rows if row is not None]


def _canonical_trial_record_row_from_contract(
    *,
    run_dir: Path,
    trial_record: TrialRecord,
    record_ref: str,
    trial_index: int,
) -> dict[str, Any]:
    """Build a web row from a canonical ``TrialRecord`` contract.

    The inspector still renders trial rows as dictionaries because a large
    amount of legacy UI code consumes that shape. This adapter is the narrow
    boundary between the typed canonical artifact reader and that existing web
    projection. Keeping it here lets future UI code move field-by-field without
    every consumer reparsing ``record.json``.
    """

    score_ref = trial_record.scoring.score_ref or ""
    trial_path = run_dir / record_ref if record_ref else run_dir / EXPERIMENT_RECORD_FILENAME
    artifact_signature_ns = _canonical_trial_artifact_signature_ns(
        run_dir,
        trial_record,
        score_ref,
    )
    return {
        "schema_version": trial_record.schema_version,
        "trial_id": trial_record.trial_id,
        "run_id": trial_record.run_id,
        "plan_ref": trial_record.plan_ref,
        "status": trial_record.status,
        "status_reason": trial_record.status_reason,
        "subject_id": trial_record.subject_id,
        "task_id": trial_record.task_id,
        "pass_index": trial_record.pass_index,
        "target_id": trial_record.target_id,
        "scoring_id": trial_record.scoring_id,
        "started_at": trial_record.started_at,
        "completed_at": trial_record.completed_at,
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "path": artifact.path,
                "kind": artifact.kind,
                "schema_version": artifact.schema_version,
                "producer": artifact.producer,
                "privacy": artifact.privacy,
                "replayability": artifact.replayability,
                "content_type": artifact.content_type,
                "created_at": artifact.created_at,
                "sha256": artifact.sha256,
            }
            for artifact in trial_record.artifacts
        ],
        "scoring": {
            "status": trial_record.scoring.status,
            "live_evidence_ref": trial_record.scoring.live_evidence_ref,
            "final_evidence_ref": trial_record.scoring.final_evidence_ref,
            "judgment_ref": trial_record.scoring.judgment_ref,
            "score_ref": score_ref,
        },
        "trial_index": trial_index,
        "sample_id": trial_record.task_id or trial_record.trial_id,
        "record_ref": record_ref,
        "termination_reason": trial_record.termination.reason,
        "exit_code": trial_record.termination.exit_code,
        "scores": _load_score_ref_scores(run_dir, score_ref, require_index=True),
        "_signature_ns": max(
            safe_mtime_ns(run_dir / EXPERIMENT_RECORD_FILENAME),
            safe_mtime_ns(run_dir / "artifact_index.json"),
            safe_mtime_ns(trial_path),
            artifact_signature_ns,
        ),
    }


def _canonical_trial_artifact_signature_ns(
    run_dir: Path,
    trial_record: TrialRecord,
    score_ref: str,
) -> int:
    """Return the newest mtime among canonical artifacts used by one row.

    Canonical trial rows can display values loaded from external artifact refs
    such as ``task_output`` or ``trial_score``. Their JSON content can change
    without mutating the trial record itself, so delta polling must include
    artifact mtimes in the row signature.
    """

    refs = {artifact.path for artifact in trial_record.artifacts}
    if score_ref:
        refs.add(score_ref)
    mtimes = []
    for ref in refs:
        path = _safe_run_relative_ref(run_dir, ref)
        if path is not None:
            mtimes.append(safe_mtime_ns(path))
    return max(mtimes, default=0)


def _load_score_ref_scores(
    run_dir: Path,
    score_ref: str,
    *,
    require_index: bool = False,
) -> dict[str, Any]:
    """Load a flat score mapping from a run-relative canonical score ref."""

    if not score_ref:
        return {}
    if require_index:
        reader = ExperimentArtifactReader(run_dir)
        try:
            raw = _load_json(reader.resolve_artifact_path(score_ref))
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return {}
    else:
        score_path = _safe_run_relative_ref(run_dir, score_ref)
        if score_path is None:
            return {}
        raw = _load_json(score_path)
    return _flatten_score_mapping(raw)


def _flatten_score_mapping(raw: Any) -> dict[str, Any]:
    """Return a template-friendly metric map from a score JSON object.

    Score artifacts often store rich metric payloads such as
    ``{"value": 0.75, "answer": "ok"}``, while web tables compare and render
    the scalar value directly. Keeping that projection in one helper prevents
    canonical and legacy score views from drifting.
    """

    if not isinstance(raw, dict):
        return {}
    scores: dict[str, Any] = {}
    for name, payload in raw.items():
        if isinstance(payload, dict) and "value" in payload:
            scores[str(name)] = payload.get("value")
        else:
            scores[str(name)] = payload
    return scores


def load_trial_score_details(trial_dir: Path) -> dict[str, Any]:
    """Full scorer payloads for the trial detail view, keyed by metric.

    ``_load_indexed_trial_scores`` flattens each metric to its scalar ``value``
    because tables and the diagnosis line only need the number. The detail page
    instead renders the *whole* scorer result — ``value``, ``answer``,
    ``explanation`` and any ``metadata`` the scorer attached (e.g. cybergym's
    ``vul_exit_code`` / ``fix_exit_code`` / ``poc_file``) — so a reader sees why
    the trial scored what it did without opening the raw score file under Files.
    """
    details: dict[str, Any] = {}
    for artifact in _indexed_trial_artifact_files(trial_dir):
        if artifact.kind != "trial_score":
            continue
        raw = _load_json(artifact.path)
        if not isinstance(raw, dict):
            continue
        for name, payload in raw.items():
            details[str(name)] = payload
    return details


def _safe_run_relative_ref(run_dir: Path, ref: str | Path) -> Path | None:
    """Return a run-local artifact path for legacy refs, or ``None`` if unsafe."""

    ref_path = Path(ref)
    if ref_path.is_absolute() or any(part == ".." for part in ref_path.parts):
        return None
    return run_dir / ref_path


def planned_trial_dir(run_dir: Path, trial_id: str) -> Path:
    """Best-effort future trial directory for a planned trial id."""
    parts = [
        part
        for part in str(trial_id or "").split("/")
        if part and part not in {".", ".."}
    ]
    if not parts:
        parts = ["pending"]
    return run_dir / TRIALS_DIRNAME / Path(*parts)


def pending_trial_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Build row info for a planned trial that has not started yet."""
    status = str(record.get("status") or "").strip().lower()
    if status and status not in {"planned", "pending"}:
        return _canonical_trial_record_summary(record)
    info = dict(record)
    info.update({
        "running": False,
        "status_kind": "pending",
        "status_label": "Pending",
        "status_detail": "Not started yet",
        "duration_ms": 0,
        "exit_code": None,
        "scores": {},
        "usage": {},
        "tags": [],
        "has_artifacts": False,
    })
    return info


def _canonical_trial_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Build row info from a canonical TrialRecord without legacy trial dirs."""

    info = dict(record)
    status = str(info.get("status") or "").strip().lower()
    reason = str(info.get("termination_reason") or info.get("status_reason") or "").strip()
    detail = str(info.get("termination_detail") or "").strip()
    exit_code = info.get("exit_code")
    info.update({
        "running": status == "running",
        # Canonical records store only start/finish timestamps (no duration
        # field), so derive the wall-clock duration the row template renders —
        # otherwise every finished trial shows "-".
        "duration_ms": _derive_duration_ms(
            info.get("started_at"), info.get("completed_at")
        ),
        "scores": info.get("scores") if isinstance(info.get("scores"), dict) else {},
        "usage": {},
        "tags": [],
        "has_artifacts": False,
        "termination_reason": reason,
        "termination_detail": detail,
        "exit_code": exit_code,
    })
    if status == "running":
        info.update({
            "status_kind": "running",
            "status_label": "Running",
            "status_detail": "Trial record is running",
        })
    elif reason:
        info.update(_status_from_reason(reason, detail=detail, exit_code=exit_code))
    elif status in COMPLETED_TRIAL_STATUSES:
        info.update({
            "status_kind": "success",
            "status_label": "Completed",
            "status_detail": "Task finished",
        })
    elif status in INTERRUPTED_TRIAL_STATUSES:
        info.update({
            "status_kind": "warning",
            "status_label": "Interrupted",
            "status_detail": "Stopped before completion",
        })
    else:
        info.update({
            "status_kind": "error",
            "status_label": "Trial failed",
            "status_detail": detail or "Trial did not finish successfully",
        })
    return info


def _live_trial_dir_for(trial_dir: Path) -> Path:
    """Return the current/live trial dir given any attempt (live or archive).

    Archives are named ``<live>.before_resume_<ts>``; stripping that suffix
    yields the live sibling.
    """
    name = trial_dir.name
    if RESUME_ARCHIVE_MARKER not in name:
        return trial_dir
    base = name.split(RESUME_ARCHIVE_MARKER, 1)[0]
    return trial_dir.parent / base


def _parse_archive_timestamp(name: str) -> tuple[str, str]:
    """Parse ``<live>.before_resume_<YYYYMMDDTHHMMSS>[_<n>]`` into (ts, label).

    Returns the raw ts portion and a human-readable label. On any parse
    failure returns ("", name) so the UI still has something to render.
    """
    if RESUME_ARCHIVE_MARKER not in name:
        return "", name
    tail = name.split(RESUME_ARCHIVE_MARKER, 1)[1]
    # Strip the same-second disambiguation suffix (e.g. "_2") if present
    ts = tail.split("_", 1)[0] if "_" in tail else tail
    try:
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%S")
        label = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        label = ts or name
    return ts, label


@dataclass
class ResumeAttempt:
    """One attempt of a logical trial — either the current live dir or
    an archived sibling from a previous --resume cycle. Carries the bits
    the template needs to render an entry in the attempts bar."""

    trial_dir: Path
    is_current: bool          # the attempt the user is currently viewing
    is_live: bool             # the live (non-archived) attempt
    label: str                # display string ("Current" or "2026-05-21 14:30:22")
    status: str               # trial status (completed/failed/...) or ""
    termination_reason: str   # canonical reason code (machine-readable) or ""
    reason_label: str = ""    # short human label for the chip ("Quota exhausted")
    reason_kind: str = ""     # chip-color hint (success/warning/error/pending)


def load_resume_attempts(trial_dir: Path) -> list[ResumeAttempt]:
    """All attempts of the same logical trial as ``trial_dir``.

    Returns the live attempt first, then archives in newest-first order.
    Returns an empty list when there are no archives — the template uses
    that to skip rendering the bar entirely.
    """
    live = _live_trial_dir_for(trial_dir)
    archives: list[Path] = []
    parent = live.parent
    if parent.is_dir():
        prefix = live.name + RESUME_ARCHIVE_MARKER
        for sibling in parent.iterdir():
            if sibling.is_dir() and sibling.name.startswith(prefix):
                archives.append(sibling)
    if not archives:
        return []

    def _meta(p: Path) -> dict[str, Any]:
        return _load_json(p / META_FILENAME) or {}

    def _build(p: Path, *, is_live: bool, label: str) -> ResumeAttempt:
        meta = _meta(p)
        reason = str(meta.get("termination_reason") or "")
        detail = str(meta.get("termination_detail") or "")
        exit_code = meta.get("exit_code")
        # Reuse the same chip mapping as the main termination card so the
        # attempts bar is consistent with whatever the user clicks into.
        chip = _status_from_reason(reason, detail=detail, exit_code=exit_code)
        return ResumeAttempt(
            trial_dir=p,
            is_current=(p.resolve() == current),
            is_live=is_live,
            label=label,
            status=str(meta.get("status") or ""),
            termination_reason=reason,
            reason_label=chip.get("status_label", ""),
            reason_kind=chip.get("status_kind", ""),
        )

    current = trial_dir.resolve()
    attempts: list[ResumeAttempt] = []
    # Skip a fake "Current" entry when the live dir was archived but no
    # new attempt has been created yet — the orphan case from an
    # interrupted resume. Archives below are still rendered, so the user
    # has a navigable history even without a live attempt.
    if live.exists() and _looks_like_trial_dir(live):
        attempts.append(_build(live, is_live=True, label="Current"))
    # Newest archive first (ts is sortable lex)
    for archive in sorted(archives, key=lambda p: p.name, reverse=True):
        _, ts_label = _parse_archive_timestamp(archive.name)
        attempts.append(_build(archive, is_live=False, label=ts_label))
    return attempts


def load_live_check_evidence(trial_dir: Path) -> dict[str, Any] | None:
    """Load the live-check audit trail for a trial, if it ran one.

    Returns ``None`` for trials that never had a live-check phase.
    Otherwise a dict with:

    * ``verdict`` — parsed ``runtime/live_success.json``, or ``{}``.
    * ``polls`` — list of ``{poll_index, ts_ms, status, message, raw}``.
    * ``first_true_index`` — 1-based index of the first poll that
      returned a positive verdict, or ``None``.
    * ``total_polls`` — count of poll lines.
    * ``had_passes`` — whether any poll ever reported success.
    """
    runtime = trial_dir / "runtime"
    polls_path = runtime / "check_done_polls.jsonl"
    verdict_path = runtime / "live_success.json"

    has_polls = polls_path.is_file()
    has_verdict = verdict_path.is_file()
    if not has_polls and not has_verdict:
        return None

    verdict: dict[str, Any] = {}
    if has_verdict:
        try:
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        except Exception:
            verdict = {}

    # ``evidence.check_done`` is written as a JSON-encoded string by the
    # poll runner ("{\"message\":...,\"status\":true}") but the trial
    # template accesses it as ``.get('message')``. Parse it here so
    # every consumer (template + API) sees a dict, matching the shape
    # the template was clearly written against.
    if isinstance(verdict, dict):
        evidence = verdict.get("evidence")
        if isinstance(evidence, dict):
            cd = evidence.get("check_done")
            if isinstance(cd, str):
                try:
                    evidence["check_done"] = json.loads(cd)
                except (json.JSONDecodeError, TypeError):
                    evidence["check_done"] = {"message": cd}

    polls: list[dict[str, Any]] = []
    first_true_index: int | None = None
    had_passes = False

    if has_polls:
        try:
            for line in polls_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # entry.output is itself JSON-encoded: {"message":..., "status":bool}
                raw_output = entry.get("output", "")
                parsed = {}
                if isinstance(raw_output, str):
                    try:
                        parsed = json.loads(raw_output)
                    except json.JSONDecodeError:
                        parsed = {"message": raw_output}
                elif isinstance(raw_output, dict):
                    parsed = raw_output
                status = bool(parsed.get("status"))
                message = str(parsed.get("message") or "").strip()
                idx = int_or_zero(entry.get("poll_index"))
                if status and first_true_index is None:
                    first_true_index = idx or (len(polls) + 1)
                if status:
                    had_passes = True
                polls.append({
                    "poll_index": idx or (len(polls) + 1),
                    "ts_ms": int_or_zero(entry.get("ts_ms")),
                    "mode": str(entry.get("mode") or ""),
                    "source": str(entry.get("source") or ""),
                    "status": status,
                    "message": message,
                    "raw": (
                        raw_output
                        if isinstance(raw_output, str)
                        else json.dumps(raw_output, ensure_ascii=False)
                    ),
                })
        except OSError:
            pass

    if not had_passes and isinstance(verdict, dict) and verdict.get("success"):
        had_passes = True

    return {
        "verdict": verdict if isinstance(verdict, dict) else {},
        "polls": polls,
        "first_true_index": first_true_index,
        "total_polls": len(polls),
        "had_passes": had_passes,
    }


def load_trial_progress(trial_dir: Path) -> dict[str, Any]:
    """Load the model-call progress snapshot for a trial.

    Live trials write ``proxy/progress.json`` on every model request; that file
    is the source consumed by summaries, delta polling, and trial-detail usage
    cards.
    """

    progress = _load_json(trial_dir / PROXY_DIRNAME / PROGRESS_FILENAME)
    if isinstance(progress, dict) and progress:
        return progress
    return {}


def load_trial_activity(trial_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the shared progress/usage projection for one trial.

    The web UI has several consumers for model-call activity: summary rows,
    delta polling, and the trial-detail usage card. They should not each map
    ``progress.json`` fields independently. This function
    exposes the two stable view objects those consumers need: ``progress`` for
    request status/counts and ``usage`` for token/cost totals.
    """

    progress = load_trial_progress(trial_dir)
    if not progress:
        return {}
    return {
        "progress": progress,
        "usage": _trial_usage_from_progress(progress),
    }


def _trial_usage_from_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Map a progress-shaped payload to the UI's token usage fields."""

    usage: dict[str, Any] = {
        "input_tokens": int(progress.get("tokens_in", 0) or 0),
        "output_tokens": int(progress.get("tokens_out", 0) or 0),
        "reasoning_tokens": int(progress.get("tokens_reasoning", 0) or 0),
        "num_requests": int(
            progress.get("success")
            or progress.get("successful_requests")
            or progress.get("total_requests")
            or 0
        ),
    }
    cost_usd = _progress_cost_usd(progress)
    if cost_usd > 0:
        usage["cost_usd"] = cost_usd
    return usage


def _normalize_trial_output_text(value: Any) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, (dict, list, tuple)):
        extracted = _extract_readable_output_from_json(value)
        if extracted:
            return extracted
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "")
    summary = parse_codex_event_stream(text)
    if summary.is_event_stream:
        return summary.final_output()
    structured = _load_json_text(text)
    if structured is not None:
        extracted = _extract_readable_output_from_json(structured)
        if extracted:
            return extracted
    return text


def _load_json_text(text: str) -> Any | None:
    stripped = str(text or "").strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _extract_readable_output_from_json(value: Any, *, depth: int = 0) -> str:
    if depth > 8 or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return ""
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            text = _extract_readable_output_from_json(item, depth=depth + 1)
            if text:
                return text
        return ""
    if not isinstance(value, dict):
        return ""

    kind = str(value.get("type") or value.get("role") or "").strip().lower()
    if kind in {"system", "developer", "thread.started", "turn.started", "item.started"}:
        return ""

    for key in (
        "text",
        "output_text",
        "answer",
        "output",
        "result",
        "aggregated_output",
    ):
        if key in value:
            text = _extract_readable_output_from_json(value.get(key), depth=depth + 1)
            if text:
                return text

    for key in ("message", "content", "item", "response", "error"):
        if key in value:
            text = _extract_readable_output_from_json(value.get(key), depth=depth + 1)
            if text:
                return text

    return ""


def _normalize_status_detail(reason: str, detail: Any) -> str:
    text = (
        detail.decode("utf-8", errors="replace")
        if isinstance(detail, bytes)
        else str(detail or "")
    )
    summary = parse_codex_event_stream(text)
    if not summary.is_event_stream:
        return text
    normalized_reason = str(reason or "").strip().lower()
    if normalized_reason in {"model_timeout", "proxy_timeout", "upstream_timeout"}:
        return ""
    return summary.terminal_error or summary.last_error or ""


def _is_resume_policy_repair_detail(source: Any, detail: str) -> bool:
    if str(source or "").strip() == "resume_policy_repair":
        return True
    lowered = str(detail or "").lower()
    return (
        "overly broad resume selection" in lowered
        or "previous resume-policy marker" in lowered
    )


def _progress_cost_usd(progress: dict[str, Any]) -> float:
    try:
        return float(progress.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _estimated_trial_start_ms(trial_dir: Path) -> int:
    candidates: list[int] = []
    for rel in ("prompt.txt", "state_pre", "proxy/stdout.log"):
        mtime_ns = safe_mtime_ns(trial_dir / rel)
        if mtime_ns > 0:
            candidates.append(mtime_ns // 1_000_000)
    return min(candidates) if candidates else 0


def _running_trial_duration_ms(trial_dir: Path, info: dict[str, Any]) -> int:
    if not info.get("running"):
        return int_or_zero(info.get("duration_ms"))
    progress = info.get("progress") if isinstance(info.get("progress"), dict) else {}
    timing = info.get("timing") if isinstance(info.get("timing"), dict) else {}
    start_ms = (
        int_or_zero(timing.get("started_at_ms"))
        or int_or_zero(info.get("started_at_ms"))
        or int_or_zero(progress.get("started_at_ms"))
        or int_or_zero(progress.get("first_ts_ms"))
        or _estimated_trial_start_ms(trial_dir)
    )
    if not start_ms:
        return int_or_zero(info.get("duration_ms"))
    elapsed = max(0, int(time.time() * 1000) - start_ms)
    return max(int_or_zero(info.get("duration_ms")), elapsed)


def refresh_running_trial_duration(trial_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    if not info.get("running"):
        return info
    duration_ms = _running_trial_duration_ms(trial_dir, info)
    if duration_ms == int_or_zero(info.get("duration_ms")):
        return info
    refreshed = dict(info)
    refreshed["duration_ms"] = duration_ms
    return refreshed


def build_trial_termination(info: dict[str, Any]) -> dict[str, Any]:
    """Build a display-friendly termination summary for the trial page."""
    reason = str(info.get("termination_reason") or "").strip()
    source = str(info.get("termination_source") or "").strip()
    raw_detail = info.get("termination_detail", "")
    text = (
        raw_detail.decode("utf-8", errors="replace")
        if isinstance(raw_detail, bytes)
        else str(raw_detail or "")
    )
    summary = parse_codex_event_stream(text)
    detail = _normalize_status_detail(reason, text).strip()
    if summary.is_event_stream and not detail:
        detail = summary.terminal_error or summary.last_error or ""
    if _is_resume_policy_repair_detail(source, detail):
        detail = ""

    errors: list[str] = []
    seen_errors: set[str] = set()
    for message in summary.error_messages:
        cleaned = str(message or "").strip()
        if not cleaned or cleaned in seen_errors:
            continue
        seen_errors.add(cleaned)
        errors.append(cleaned)

    reason_label = ""
    if reason:
        reason_status = _status_from_reason(
            reason,
            detail=detail,
            exit_code=info.get("exit_code"),
        )
        reason_label = reason_status.get("status_label", "")

    return {
        "reason": reason,
        "reason_label": reason_label,
        "source": source,
        "detail": detail,
        "raw_detail": text,
        "error_messages": errors,
        "is_event_stream": summary.is_event_stream,
        "last_agent_message": summary.last_agent_message,
        "has_content": bool(reason or source or detail or text),
    }


def trial_summary_signature(trial_dir: Path) -> tuple[int, ...]:
    """Cheap mtime tuple used to invalidate per-trial caches.

    Legacy trial directories are keyed by the files that drive the list row.
    Canonical trial rows may be virtual or may coexist with a legacy directory,
    so include the TrialRecord / ArtifactIndex signature whenever the path
    resolves to a canonical trial.
    """
    canonical_signature_ns = 0
    canonical_row = _indexed_trial_record_row(trial_dir)
    if canonical_row:
        canonical_signature_ns = int(canonical_row.get("_signature_ns") or 0)
    return (
        safe_mtime_ns(trial_dir / TASK_OUTPUT_FILENAME),
        safe_mtime_ns(trial_dir / META_FILENAME),
        safe_mtime_ns(trial_dir / PROXY_DIRNAME / PROGRESS_FILENAME),
        safe_mtime_ns(trial_dir / "scores"),
        canonical_signature_ns,
    )


def load_trial_summary_cached(
    trial_dir: Path,
    dashboard_info: dict[str, Any] | None = None,
    run_status: str = "",
) -> dict[str, Any]:
    """Cached front of :func:`load_trial_summary`.

    Cache key includes the trial's mtime signature, the run status (so
    "Running"→"Interrupted" relabelling propagates), and a digest of the
    dashboard-provided info (small dict; trials usually have a stable
    dashboard entry, so this rarely contributes to misses).
    """
    sig = (
        trial_summary_signature(trial_dir),
        run_status,
        _dashboard_info_signature(dashboard_info),
    )

    def _compute() -> dict[str, Any]:
        return load_trial_summary(trial_dir, dashboard_info, run_status=run_status)

    return refresh_running_trial_duration(
        trial_dir,
        get_or_compute(trial_summary_cache, trial_dir, sig, _compute),
    )


def _dashboard_info_signature(info: dict[str, Any] | None) -> tuple:
    if not info:
        return ()
    # We only care about fields load_trial_summary actually reads.
    keys = (
        "trial_id",
        "trial_index",
        "sample_id",
        "exit_code",
        "error",
        "status",
        "termination_reason",
        "termination_detail",
        "termination_source",
        "live_success",
        "live_success_verdict",
        "timing",
        "scores",
        "usage",
    )
    return tuple((k, _hashable(info.get(k))) for k in keys if k in info)


def _hashable(value: Any) -> Any:
    """Make nested dicts/lists hashable for cache key use."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def is_known_trial_path(trial_dir: Path) -> bool:
    """Return whether ``trial_dir`` names a legacy or canonical trial.

    Encoded web routes often receive paths shaped like ``<run>/trials/<id>``.
    For canonical runs that path may be a virtual projection rather than an
    actual directory; the TrialRecord is the durable existence check.
    """

    return trial_dir.is_dir() or bool(_indexed_trial_record_row(trial_dir))


def load_trial_summary(
    trial_dir: Path,
    dashboard_info: dict[str, Any] | None = None,
    run_status: str = "",
) -> dict[str, Any]:
    """Build list-page trial info from dashboard data plus live filesystem state.

    ``meta.json`` on disk is the source of truth — ``dashboard.json`` is just
    a snapshot the orchestrator writes during the run, and can lag behind
    out-of-band rewrites (``scripts/reclassify_meta.py``, manual edits).
    Overwrite termination + status fields from meta whenever it's present
    so the inspector reflects the current on-disk classification.

    Canonical runs may record summary artifacts in ``TrialRecord`` and
    ``ArtifactIndex`` without materializing legacy files below ``trial_dir``.
    Prefer those indexed artifacts for completed output and scores, then keep
    the historical filesystem layout as a compatibility fallback.
    """
    info: dict[str, Any] = dict(dashboard_info or {})
    record_info = _indexed_trial_record_row(trial_dir)
    if record_info:
        for key, value in record_info.items():
            if key not in info:
                info[key] = value
    info["has_artifacts"] = True
    meta = _load_json(trial_dir / META_FILENAME)
    if meta:
        for key in (
            "trial_id",
            "trial_index",
            "sample_id",
            "exit_code",
            "error",
            "status",
            "termination_reason",
            "termination_detail",
            "termination_source",
            "live_success",
            "live_success_verdict",
            "terminated_by_live_success",
        ):
            if key in meta:
                info[key] = meta[key]
        timing = meta.get("timing", {})
        if isinstance(timing, dict):
            info["timing"] = timing
            if "duration_ms" in timing:
                info["duration_ms"] = timing["duration_ms"]

    task_output = _load_indexed_trial_json_artifact(
        trial_dir,
        kind="task_output",
    ) or _load_json(trial_dir / TASK_OUTPUT_FILENAME)
    if task_output and "output" not in info:
        info["output"] = task_output.get("output", "")
    if "output" in info:
        info["output"] = _normalize_trial_output_text(info.get("output", ""))
    sample = task_output.get("sample") if isinstance(task_output, dict) else {}
    info["tags"] = _extract_trial_tags(info, sample if isinstance(sample, dict) else {})

    if "scores" not in info:
        scores = _load_indexed_trial_scores(trial_dir)
        if not scores:
            scores_dir = trial_dir / "scores"
            if scores_dir.is_dir():
                for sf in scores_dir.glob("*.json"):
                    try:
                        scores.update(json.loads(sf.read_text(encoding="utf-8")))
                    except Exception:
                        pass
        if scores:
            # ``Score`` artifacts on disk are dataclass dumps
            # ``{"value": float, "answer": str, "explanation": str}``;
            # the run.html template compares ``val >= 0.7`` directly, so
            # collapse to the scalar (matching dashboard.json's shape)
            # and keep the metadata reachable on a sibling key for the
            # trial-detail view.
            details: dict[str, Any] = {}
            for k, v in scores.items():
                if isinstance(v, dict) and "value" in v:
                    details[k] = v
            flat = _flatten_score_mapping(scores)
            info["scores"] = flat
            if details:
                info["score_details"] = details

    activity = load_trial_activity(trial_dir)
    progress = activity.get("progress") if activity else {}
    if progress:
        info["progress"] = progress
        has_task_output = bool(task_output) or (trial_dir / TASK_OUTPUT_FILENAME).exists()
        if not has_task_output and record_info:
            has_task_output = any(
                isinstance(artifact, dict) and artifact.get("kind") == "task_output"
                for artifact in record_info.get("artifacts", [])
            )
        terminal_status = str(info.get("status") or "").strip().lower() in {
            "completed",
            "failed",
            "interrupted",
            "cancelled",
            "not_scored",
            "scored",
        }
        info["running"] = not has_task_output and not terminal_status
        usage = info.get("usage")
        if not isinstance(usage, dict) or not any(usage.values()):
            usage = dict(activity.get("usage") or {})
            info["usage"] = usage
    else:
        info.setdefault("running", False)
    info.update(_classify_trial_status(trial_dir, info, run_status=run_status))
    info = refresh_running_trial_duration(trial_dir, info)
    return info


def trial_summary_from_dashboard_entry(
    entry: dict[str, Any],
    *,
    run_status: str = "",
) -> dict[str, Any]:
    """Build an overview trial row from a dashboard projection entry — no disk.

    The settled-run fast path. ``dashboard.json`` already aggregates every
    trial's status/scores/score_details/duration_ms/exit_code/termination_*/
    usage/sample_id, so the run overview can render rows without the per-trial
    filesystem walk :func:`load_trial_summary` performs. Presentation (status
    chip, tags) is computed here so it stays single-sourced with the on-disk
    path; the "Max rounds" live-check nuance is omitted (shows "Completed") and
    surfaces only on the trial-detail page, which still reads per-trial
    artifacts. Tags derive from the trial id alone (no ``task_output`` sample).
    """
    info: dict[str, Any] = dict(entry)
    info.setdefault("running", False)
    info["has_artifacts"] = True
    # Force-killed / resumed trials leave the projection's duration_ms at 0;
    # derive it from the entry's start/finish timestamps so the duration column
    # isn't blank for half the run.
    if not info.get("duration_ms"):
        info["duration_ms"] = _derive_duration_ms(
            info.get("started_at"), info.get("completed_at")
        )
    # Synthesize the model-call "progress" view the overview template renders
    # from the projection's usage counts, so a settled run shows steps with NO
    # per-trial filesystem read. num_requests is the canonical agent-round count
    # (= successful_requests, what max_rounds caps and we show as "steps").
    # total_requests/errors are present only on newer projections; when absent
    # the template hides the error rate rather than showing a bogus 0%.
    usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
    if "progress" not in info and usage:
        rounds = int_or_zero(usage.get("num_requests"))
        has_totals = usage.get("total_requests") is not None
        if rounds or has_totals:
            progress: dict[str, Any] = {"successful_requests": rounds}
            if has_totals:
                progress["total_requests"] = int_or_zero(usage.get("total_requests"))
                progress["errors"] = int_or_zero(usage.get("errors"))
            info["progress"] = progress
    info["tags"] = _extract_trial_tags(info, {})
    info.update(
        _classify_trial_status(
            Path(),  # trial_dir unused: live_check_enabled=False bypasses the disk probe
            info,
            run_status=run_status,
            live_check_enabled=False,
        )
    )
    return info


def _extract_trial_tags(info: dict[str, Any], sample: dict[str, Any]) -> list[str]:
    tags: set[str] = set()

    def add(value: Any, prefix: str = "") -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        if len(text) > 80 or "://" in text or "\n" in text:
            return
        tags.add(f"{prefix}{text}" if prefix else text)

    trial_id = str(info.get("trial_id") or sample.get("id") or "")
    for match in re.findall(r"(?i)\brange\d+\b", trial_id):
        add(match.lower())
    for match in re.findall(r"(?i)\bL\d+\b", trial_id):
        add(match.upper())

    for key in ("benchmark", "category", "variant", "test_type", "source", "type", "task_type"):
        if key in sample:
            add(sample.get(key), f"{key}:")

    has_explicit_level = any(key in sample for key in ("level", "hint_level"))
    for key in ("level", "hint_level"):
        if key in sample:
            level = sample.get(key)
            if isinstance(level, int) or str(level).isdigit():
                add(f"L{level}")
            else:
                add(level, "level:")

    name = str(sample.get("name") or "")
    for match in re.findall(r"(?i)\brange\s*\d+\b", name):
        add(match.lower().replace(" ", ""))
    if not has_explicit_level:
        for match in re.findall(r"(?i)\bL\d+\b", name):
            add(match.upper())

    sample_tags = sample.get("tags")
    if isinstance(sample_tags, list):
        for tag in sample_tags:
            add(tag)
    elif isinstance(sample_tags, str):
        add(sample_tags)

    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key in ("category", "variant", "test_type", "difficulty", "type", "event", "year"):
            if key in metadata:
                add(metadata.get(key), f"{key}:")

    def tag_sort_key(tag: str) -> tuple[int, str]:
        if re.fullmatch(r"L\d+", tag):
            return (0, tag)
        if tag.startswith("benchmark:"):
            return (1, tag)
        if tag.startswith("category:"):
            return (2, tag)
        return (3, tag.lower())

    return sorted(tags, key=tag_sort_key)


def load_trial(trial_dir: Path) -> TrialDetail:
    """Load full trial data for the trial detail page.

    Legacy runs store every displayed artifact directly under ``trial_dir``.
    Canonical runs may expose ``trial_dir`` as a virtual path while recording
    heavy artifacts, including raw proxy logs, through ``TrialRecord`` and
    ``ArtifactIndex``. Keep optional artifact discovery behind shared resolvers
    so the template reflects canonical runs without reopening path-guessing
    logic in the route layer.
    """
    meta = _load_json(trial_dir / META_FILENAME)
    if not meta:
        meta = _indexed_trial_record_row(trial_dir)
    prompt = _load_indexed_trial_text_artifact(
        trial_dir,
        kind="prompt",
    ) or _load_text(trial_dir / PROMPT_FILENAME)

    task_output = _load_indexed_trial_json_artifact(
        trial_dir,
        kind="task_output",
    ) or _load_json(trial_dir / TASK_OUTPUT_FILENAME)
    output = _normalize_trial_output_text(task_output.get("output", ""))
    sample = task_output.get("sample", {})

    scores = _load_indexed_trial_scores(trial_dir)
    if not scores:
        scores_dir = trial_dir / "scores"
        if scores_dir.is_dir():
            for sf in scores_dir.glob("*.json"):
                try:
                    scores.update(json.loads(sf.read_text(encoding="utf-8")))
                except Exception:
                    pass

    return TrialDetail(
        trial_id=meta.get("trial_id", trial_dir.name),
        meta=meta,
        prompt=prompt,
        output=output,
        sample=sample,
        scores=scores,
        has_trajectory=trial_has_trajectory(trial_dir),
    )


def build_trial_file_tree(trial_dir: Path) -> list[TrialFileEntry]:
    """Build a depth-aware file tree for a trial artifact directory.

    Historical runs expose a concrete directory tree under ``trial_dir``.
    Canonical runs can instead expose durable artifacts through the trial
    record and artifact index, sometimes outside the legacy trial directory.
    This function merges both views so the file browser remains an observation
    surface over recorded artifacts rather than a filesystem-only guess.
    """
    base = trial_dir.resolve()
    entries: list[TrialFileEntry] = []

    def add_children(parent: Path, depth: int) -> None:
        try:
            children = sorted(
                parent.iterdir(),
                key=lambda path: (not path.is_dir(), path.name.lower()),
            )
        except OSError:
            return

        for child in children:
            try:
                resolved = child.resolve()
                relative = resolved.relative_to(base)
            except (OSError, ValueError):
                continue

            is_dir = child.is_dir()
            size_bytes = 0
            size_label = ""
            if not is_dir:
                try:
                    size_bytes = child.stat().st_size
                except OSError:
                    continue
                size_label = _format_file_size(size_bytes)

            entries.append(
                TrialFileEntry(
                    path=child,
                    relative_path=relative.as_posix(),
                    name=child.name,
                    depth=depth,
                    is_dir=is_dir,
                    size_bytes=size_bytes,
                    size_label=size_label,
                )
            )
            if is_dir:
                add_children(child, depth + 1)

    add_children(base, 0)
    _add_indexed_trial_file_entries(trial_dir, entries)
    return entries


def is_indexed_trial_artifact_path(trial_dir: Path, artifact_path: Path) -> bool:
    """Return whether ``artifact_path`` is indexed for the given trial.

    Download routes allow legacy files under ``trial_dir`` by containment. For
    canonical runs, an artifact may live elsewhere under the run directory, so
    containment is not enough. This helper authorizes only artifacts explicitly
    attached to the matching ``TrialRecord`` and present in ``ArtifactIndex``.
    """

    target = artifact_path.expanduser().resolve()
    return any(
        artifact.path == target
        for artifact in _indexed_trial_artifact_files(trial_dir)
    )


def _load_indexed_trial_json_artifact(trial_dir: Path, *, kind: str) -> dict[str, Any]:
    """Load the first indexed JSON artifact of ``kind`` for a trial.

    Trial detail pages historically opened fixed filenames below
    ``trial_dir``. Canonical runs instead attach files to ``TrialRecord`` and
    ``ArtifactIndex``. This helper is the small bridge for single-artifact JSON
    payloads such as ``task_output`` while preserving legacy fallback behavior
    when no canonical artifact is recorded.
    """

    for artifact in _indexed_trial_artifact_files(trial_dir):
        if artifact.kind != kind:
            continue
        raw = _load_json(artifact.path)
        if isinstance(raw, dict):
            return raw
    return {}


def _load_indexed_trial_text_artifact(trial_dir: Path, *, kind: str) -> str:
    """Load the first indexed text artifact of ``kind`` for a trial.

    Rendered prompts are plain text rather than JSON. Keeping a separate text
    helper avoids overloading JSON artifact handling while still enforcing the
    same ``TrialRecord`` + ``ArtifactIndex`` membership rule.
    """

    for artifact in _indexed_trial_artifact_files(trial_dir):
        if artifact.kind != kind:
            continue
        return _load_text(artifact.path)
    return ""


def _load_indexed_trial_scores(trial_dir: Path) -> dict[str, Any]:
    """Load flat score values from indexed trial score artifacts.

    A trial may have multiple scoring artifacts over time. The web detail view
    keeps the legacy merged-dict behavior, but each candidate file must be
    explicitly attached to the trial and present in ``ArtifactIndex`` before it
    contributes values.
    """

    scores: dict[str, Any] = {}
    for artifact in _indexed_trial_artifact_files(trial_dir):
        if artifact.kind != "trial_score":
            continue
        raw = _load_json(artifact.path)
        scores.update(_flatten_score_mapping(raw))
    return scores


# ------------------------------------------------------------------
# Trajectory parsing (from proxy.jsonl)
# ------------------------------------------------------------------

def resolve_trial_proxy_jsonl(trial_dir: Path) -> Path:
    """Return the raw proxy JSONL path for a web trial view.

    Web routes receive a trial directory path. For historical runs that path is
    a real directory containing ``proxy/proxy.jsonl``. For canonical runs, the
    path may be only a virtual projection of ``trials/<trial_id>`` while the
    actual proxy log is recorded in ``TrialRecord.artifacts`` and
    ``ArtifactIndex``. This helper gives routes one read boundary: prefer the
    indexed canonical artifact, then fall back to the legacy file location.
    """

    legacy_path = trial_dir / PROXY_DIRNAME / PROXY_LOG_FILENAME
    canonical_path = _resolve_indexed_trial_proxy_jsonl(trial_dir)
    if canonical_path is not None:
        return canonical_path
    return legacy_path


def trial_has_trajectory(trial_dir: Path) -> bool:
    """Return whether a trial has a trajectory artifact (raw ``proxy.jsonl``)."""

    return resolve_trial_proxy_jsonl(trial_dir).exists()


def parse_trial_trajectory(
    trial_dir: Path,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Parse a trial's trajectory from its raw ``proxy.jsonl`` audit log.

    ``proxy.jsonl`` is the single source of truth for the agent trajectory
    (thinking + text + tool_use + tool_result per turn). Presentation shaping
    stays in the web layer.
    """

    return parse_trajectory(
        resolve_trial_proxy_jsonl(trial_dir),
        offset=offset,
        limit=limit,
    )


_TRAJECTORY_SOURCE_CACHE: "OrderedDict[tuple, tuple[list[dict[str, Any]], ConversationForest]]" = OrderedDict()
_TRAJECTORY_SOURCE_CACHE_MAX = 32
_TRAJECTORY_SOURCE_LOCK = threading.Lock()


def _trajectory_source(proxy_jsonl: Path) -> tuple[list[dict[str, Any]], ConversationForest]:
    """Return ``(success_entries, forest)`` for a proxy log, cached by file stat.

    Structure reconstruction needs the whole stream, so the file is read in full
    on a cache miss. The cache (keyed on path + mtime + size) means paging through
    a settled trajectory does not re-read the log; a live log's changing stat
    invalidates the entry automatically.
    """
    try:
        st = proxy_jsonl.stat()
        key = (str(proxy_jsonl), st.st_mtime_ns, st.st_size)
    except OSError:
        return [], reconstruct_forest([])
    with _TRAJECTORY_SOURCE_LOCK:
        cached = _TRAJECTORY_SOURCE_CACHE.get(key)
        if cached is not None:
            _TRAJECTORY_SOURCE_CACHE.move_to_end(key)
            return cached
    entries = _load_success_entries(proxy_jsonl)
    forest = reconstruct_forest(entries)
    with _TRAJECTORY_SOURCE_LOCK:
        _TRAJECTORY_SOURCE_CACHE[key] = (entries, forest)
        _TRAJECTORY_SOURCE_CACHE.move_to_end(key)
        while len(_TRAJECTORY_SOURCE_CACHE) > _TRAJECTORY_SOURCE_CACHE_MAX:
            _TRAJECTORY_SOURCE_CACHE.popitem(last=False)
    return entries, forest


def parse_trajectory(
    proxy_jsonl: Path,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Parse proxy.jsonl into structured trajectory steps.

    Returns ``{"total_steps": N, "steps": [...], "total_tokens": {...}}``.
    Supports pagination via *offset* and *limit*.
    """
    if not proxy_jsonl.exists():
        total_tokens: dict[str, Any] = {}
        return {
            "total_steps": 0,
            "steps": [],
            "total_tokens": total_tokens,
            "summary": _trajectory_operator_summary(
                [],
                total_tokens,
                total_steps=0,
                total_steps_known=True,
                tool_results_by_id={},
            ),
        }

    offset = max(0, int(offset or 0))
    limit = max(0, int(limit or 0))
    # Reconstructing harness structure (subagents/compaction/background) needs the
    # whole stream — a subagent's parent edge can point anywhere — so we load the
    # full success log and derive the conversation forest once, cache both keyed
    # on (path, mtime, size), and paginate the rendered steps over them. The cache
    # keeps repeated page fetches from re-reading the file (NAS-friendly); a live,
    # growing log changes size/mtime and is re-read automatically.
    entries, forest = _trajectory_source(proxy_jsonl)
    return _build_trajectory_payload(
        entries,
        offset=offset,
        limit=limit,
        total_steps_known=True,
        forest=forest,
    )


def _forest_web_view(forest: ConversationForest | None) -> dict[str, Any]:
    """Serialize a conversation forest for the trajectory payload."""
    if forest is None:
        return {"conversations": [], "spawns_by_index": {}, "returns_by_index": {}, "exits_by_index": {}, "structure": {}}
    conversations = []
    for conv in forest.conversations:
        conversations.append({
            "id": conv.id,
            "kind": conv.kind,
            "parent_id": conv.parent_id,
            "depth": conv.depth,
            "subagent_type": conv.subagent_type,
            "num_calls": len(conv.call_indices),
            "num_compactions": len(conv.compaction_calls),
            "spawned_by": conv.spawned_by,
            "spawn_prompt": (conv.spawned_by or {}).get("prompt", "") if conv.spawned_by else "",
            "returns_at": conv.returns_at,
            "usage": conv.usage,
            "first_index": conv.call_indices[0] if conv.call_indices else None,
            "last_index": conv.call_indices[-1] if conv.call_indices else None,
        })
    spawns_by_index: dict[int, list[str]] = {}
    returns_by_index: dict[int, list[str]] = {}
    exits_by_index: dict[int, list[str]] = {}
    for conv in forest.conversations:
        if conv.spawned_by and conv.spawned_by.get("parent_index") is not None:
            spawns_by_index.setdefault(int(conv.spawned_by["parent_index"]), []).append(conv.id)
        if conv.returns_at is not None:
            returns_by_index.setdefault(int(conv.returns_at), []).append(conv.id)
        # A subagent's last call is its end point — where its produced output is
        # shown inline so a reader sees what it concluded without chasing the
        # tool_result back into the parent.
        if conv.kind == "subagent" and conv.call_indices:
            exits_by_index.setdefault(int(conv.call_indices[-1]), []).append(conv.id)
    structure = dict(forest.structure)
    structure["root_rounds"] = forest.root_rounds
    structure["root_usage"] = forest.root_usage
    return {
        "conversations": conversations,
        "spawns_by_index": spawns_by_index,
        "returns_by_index": returns_by_index,
        "exits_by_index": exits_by_index,
        "structure": structure,
    }


def _build_trajectory_payload(
    entries: list[dict[str, Any]],
    *,
    offset: int,
    limit: int,
    total_steps_known: bool,
    forest: ConversationForest | None = None,
) -> dict[str, Any]:
    total = len(entries)
    fweb = _forest_web_view(forest)
    conv_by_id = {c["id"]: c for c in fweb["conversations"]}
    # A compaction is a real model call (the summary-generating call) followed by
    # a continuation that resumes from that summary.
    compaction_call_indices: set[int] = set()  # the call that PRODUCES the summary
    compaction_resume_indices: set[int] = set()  # the call that RESUMES from it
    if forest:
        for conv in forest.conversations:
            compaction_call_indices.update(conv.compaction_calls)
            compaction_resume_indices.update(conv.compaction_at)

    if not entries:
        total_tokens: dict[str, Any] = {}
        return {
            "total_steps": 0,
            "steps": [],
            "total_tokens": total_tokens,
            "summary": _trajectory_operator_summary(
                entries,
                total_tokens,
                total_steps=0,
                total_steps_known=True,
                tool_results_by_id={},
            ),
            "has_more": False,
            "total_steps_known": True,
            "total_tokens_known": True,
            "structure": fweb["structure"],
            "conversations": fweb["conversations"],
        }

    tool_results_by_id = _collect_tool_results_by_id(entries)

    # Build steps (paginated)
    cumulative_in = 0
    cumulative_out = 0
    cumulative_reasoning = 0
    total_tokens = {"in": 0, "out": 0, "reasoning": 0}
    inferred_blocks = _infer_response_blocks_from_next_requests(
        entries,
        tool_results_by_id,
        start=offset,
        end=offset + limit,
    )

    # Per-request observation (latest fed-back tool result). The result of step
    # idx's action shows up as the observation in request idx+1 — same as how a
    # native tool_use's result is matched from the following request.
    observations = [_latest_observation_text(e) for e in entries]

    # First request index per node (or the root agent when there is no node), so
    # each agent/node's first step can surface its system + user prompt.
    node_first_idx: dict[str, int] = {}
    for _i, _e in enumerate(entries):
        _span = _e.get("cage_span")
        _nd = _span.get("node") if isinstance(_span, dict) else None
        node_first_idx.setdefault(str(_nd) if _nd else "__root__", _i)

    steps: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        usage = _extract_usage(entry)
        in_tok = usage.get("in", 0)
        out_tok = usage.get("out", 0)
        reason_tok = usage.get("reasoning", 0)
        cumulative_in += in_tok
        cumulative_out += out_tok
        cumulative_reasoning += reason_tok
        total_tokens = {
            "in": cumulative_in,
            "out": cumulative_out,
            "reasoning": cumulative_reasoning,
        }

        if idx < offset:
            continue
        if idx >= offset + limit:
            continue

        blocks = _extract_blocks(entry, tool_results_by_id)
        if not blocks:
            blocks = inferred_blocks.get(idx, [])
        # Summarise request context (message count per role)
        req, _ = _extract_request_body(entry)
        req_msgs = _extract_request_messages(req)
        context_summary = {}
        for m in req_msgs:
            role = m.get("role", "?")
            context_summary[role] = context_summary.get(role, 0) + 1
        # Result of THIS step's action = the observation fed back in the next
        # request (None for the final step / native-tool agents whose results
        # already render inline under each tool_use block).
        result_text = observations[idx + 1] if idx + 1 < len(observations) else ""

        # First step of each node/agent surfaces its system + user prompt.
        _node_val = (entry.get("cage_span") or {}).get("node") if isinstance(entry.get("cage_span"), dict) else None
        node_first = node_first_idx.get(str(_node_val) if _node_val else "__root__") == idx
        system_prompt = user_prompt = ""
        if node_first:
            for m in req_msgs:
                role = str(m.get("role") or "")
                if not system_prompt and role == "system":
                    system_prompt = _message_text(m)
                elif not user_prompt and role in ("user", "human"):
                    user_prompt = _message_text(m)
                if system_prompt and user_prompt:
                    break

        # Harness-structure attribution for this call.
        conv_id = forest.call_to_conv.get(idx) if forest else None
        conv_meta = conv_by_id.get(conv_id or "", {})
        conversation = {
            "id": conv_id,
            "kind": conv_meta.get("kind", "root"),
            "depth": conv_meta.get("depth", 0),
            "subagent_type": conv_meta.get("subagent_type"),
            "parent_id": conv_meta.get("parent_id"),
        }
        is_compaction_call = idx in compaction_call_indices
        is_compaction_resume = idx in compaction_resume_indices
        # Subagent results that fold back into this step's context — the literal
        # point where a subagent's output enters the parent agent's prompt.
        returns_meta: list[dict[str, Any]] = []
        for cid in fweb["returns_by_index"].get(idx, []):
            cmeta = conv_by_id.get(cid, {})
            tuid = (cmeta.get("spawned_by") or {}).get("tool_use_id") or ""
            result_text = tool_results_by_id.get(tuid, "")
            returns_meta.append({
                "conversation_id": cid,
                "subagent_type": cmeta.get("subagent_type"),
                "tool_use_id": tuid,
                "first_index": cmeta.get("first_index"),
                "result_preview": result_text[:4000],
                "result_len": len(result_text),
                "result_truncated": len(result_text) > 4000,
            })
        # Subagents that end at this step — their produced output, shown inline at
        # the subagent's own last step (its natural end point).
        exits_meta: list[dict[str, Any]] = []
        for cid in fweb["exits_by_index"].get(idx, []):
            cmeta = conv_by_id.get(cid, {})
            tuid = (cmeta.get("spawned_by") or {}).get("tool_use_id") or ""
            result_text = tool_results_by_id.get(tuid, "")
            exits_meta.append({
                "conversation_id": cid,
                "subagent_type": cmeta.get("subagent_type"),
                "returns_at": cmeta.get("returns_at"),
                "result_preview": result_text[:4000],
                "result_len": len(result_text),
                "result_truncated": len(result_text) > 4000,
            })
        cage_span = entry.get("cage_span") if isinstance(entry.get("cage_span"), dict) else None
        steps.append({
            "index": idx,
            "ts_ms": entry.get("ts_ms", 0),
            # LangGraph node that issued this request (from the agent's X-Cage-Node
            # header, recorded by the proxy). None for agents without the hook.
            "node": (cage_span or {}).get("node"),
            "tokens": {"in": in_tok, "out": out_tok, "reasoning": reason_tok},
            "cumulative": {"in": cumulative_in, "out": cumulative_out},
            "blocks": blocks,
            "context_summary": context_summary,
            "context_msg_count": len(req_msgs),
            "result": result_text[:8000],
            "node_first": node_first,
            "system_prompt": system_prompt[:16000],
            "user_prompt": user_prompt[:16000],
            "conversation": conversation,
            "is_compaction_call": is_compaction_call,
            "is_compaction_resume": is_compaction_resume,
            "spawns": fweb["spawns_by_index"].get(idx, []),
            "returns": returns_meta,
            "exits": exits_meta,
        })

    has_more = (offset + limit) < total
    return {
        "total_steps": total,
        "steps": steps,
        "node_flow": _node_flow(entries),
        "total_tokens": total_tokens,
        "summary": _trajectory_operator_summary(
            entries,
            total_tokens,
            total_steps=total,
            total_steps_known=total_steps_known,
            tool_results_by_id=tool_results_by_id,
        ),
        "has_more": has_more,
        "total_steps_known": total_steps_known,
        "total_tokens_known": total_steps_known,
        "structure": fweb["structure"],
        "conversations": fweb["conversations"],
    }


def _latest_observation_text(entry: dict[str, Any]) -> str:
    """The latest non-system, non-assistant message in a request = a fed-back
    tool result. Agents that run tools in Python append the result as the next
    message, so request N+1's latest observation is the result of action N."""
    body, _ = _extract_request_body(entry)
    for msg in reversed(_extract_request_messages(body)):
        if str(msg.get("role") or "") in ("system", "assistant", "ai", "model"):
            continue
        return _message_text(msg)
    return ""


def _node_flow(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-request LangGraph nodes into the route the agent took.

    e.g. ``[{node: global_map, count: 12}, {node: candidate_dev, count: 23}]``.
    Empty for agents without the cage_trace hook (no ``cage_span`` on entries).
    Nodes that issue no model request (deterministic Python nodes) never appear —
    the trajectory is request-based by construction.
    """
    flow: list[dict[str, Any]] = []
    for entry in entries:
        span = entry.get("cage_span")
        node = span.get("node") if isinstance(span, dict) else None
        if not node:
            continue
        if flow and flow[-1]["node"] == node:
            flow[-1]["count"] += 1
        else:
            flow.append({"node": str(node), "count": 1})
    return flow


def _trajectory_operator_summary(
    entries: list[dict[str, Any]],
    total_tokens: dict[str, Any],
    *,
    total_steps: int,
    total_steps_known: bool,
    tool_results_by_id: dict[str, str],
) -> dict[str, Any]:
    tool_counts: dict[str, int] = {}
    last_action = ""
    error_count = 0
    for entry in entries:
        if _entry_has_error_signal(entry):
            error_count += 1
        blocks = _extract_blocks(entry, tool_results_by_id)
        if not blocks:
            continue
        last_action = _trajectory_last_action(blocks)
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1

    tools = [
        {"name": name, "count": count}
        for name, count in sorted(
            tool_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return {
        "requests": len(entries),
        "loaded_steps": len(entries),
        "total_steps": total_steps,
        "total_steps_known": total_steps_known,
        "total_tokens": dict(total_tokens or {}),
        "top_tools": tools[:3],
        "tools": tools,
        "tool_calls": sum(tool_counts.values()),
        "last_action": last_action,
        "error_count": error_count,
    }


def _progress_tools_used(trial_dir: Path) -> dict[str, int] | None:
    """Per-tool counts the proxy already persisted in ``proxy/progress.json``.

    The in-container proxy increments ``tools_used`` on every tool call and
    writes it to this tiny (~4 KB) file each request, so a run's tool
    distribution is *already on disk* — no need to re-parse the multi-MB
    ``proxy.jsonl``. Returns ``None`` only for older runs predating the field,
    which fall back to the proxy-log parse. An empty dict (trial made no tool
    calls) is authoritative and returned as-is.
    """
    progress = _read_progress_cached(trial_dir / PROXY_DIRNAME / PROGRESS_FILENAME)
    tools = progress.get("tools_used")
    if not isinstance(tools, dict):
        return None
    out: dict[str, int] = {}
    for name, count in tools.items():
        try:
            out[str(name)] = int(count)
        except (TypeError, ValueError):
            continue
    return out


def trial_tool_counts(trial_dir: Path) -> dict[str, int]:
    """Tool-use ``name`` → call count for one trial.

    Prefers the proxy's already-persisted ``tools_used`` (tiny file); falls back
    to parsing ``proxy.jsonl`` only for legacy runs that never wrote it.
    """
    from_progress = _progress_tools_used(trial_dir)
    if from_progress is not None:
        return from_progress
    proxy_jsonl = resolve_trial_proxy_jsonl(trial_dir)
    if not proxy_jsonl.exists():
        return {}
    return _count_proxy_tools(proxy_jsonl)


def _count_proxy_tools(proxy_jsonl: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in _load_success_entries(proxy_jsonl):
        for block in _extract_blocks(entry, {}):
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "unknown")
            counts[name] = counts.get(name, 0) + 1
    return counts


def aggregate_run_tool_counts(run_dir: Path) -> dict[str, Any]:
    """Aggregate tool-call distribution across every trial in a run.

    Each trial's per-tool counts are read from the tiny ``proxy/progress.json``
    the proxy already persisted during the run (legacy runs fall back to a
    ``proxy.jsonl`` parse), fanned out across trials. Cached in-process on the
    cheap structural run signature, so repeat loads within a session are
    instant; a live run recomputes only when a trial is added or resumed.
    Deliberately writes nothing back into the run tree — adding a file there
    would bump ``run_dir``'s mtime and invalidate the history/stub caches the
    detail page depends on.
    """
    return get_or_compute(
        run_tools_cache,
        run_dir,
        _shallow_run_signature(run_dir),
        lambda: _aggregate_run_tool_counts(run_dir),
    )


def _aggregate_run_tool_counts(run_dir: Path) -> dict[str, Any]:
    trials_root = run_dir / TRIALS_DIRNAME
    empty = {
        "tools": [],
        "total_calls": 0,
        "trials_counted": 0,
        "trials_with_tools": 0,
    }
    if not trials_root.is_dir():
        return empty
    trial_dirs = [meta.parent for meta in _iter_trial_meta_paths(trials_root)]
    if not trial_dirs:
        return empty

    totals: dict[str, int] = {}
    trials_with: dict[str, int] = {}
    trials_counted = 0
    trials_with_tools = 0
    workers = min(_HISTORY_SCAN_WORKERS, len(trial_dirs))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for counts in pool.map(trial_tool_counts, trial_dirs):
            trials_counted += 1
            if counts:
                trials_with_tools += 1
            for name, count in counts.items():
                totals[name] = totals.get(name, 0) + count
                trials_with[name] = trials_with.get(name, 0) + 1

    tools = [
        {"name": name, "count": count, "trials": trials_with.get(name, 0)}
        for name, count in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {
        "tools": tools,
        "total_calls": sum(totals.values()),
        "trials_counted": trials_counted,
        "trials_with_tools": trials_with_tools,
    }


def _entry_has_error_signal(entry: dict[str, Any]) -> bool:
    if entry.get("error") or entry.get("error_message") or entry.get("upstream_error"):
        return True
    status = str(entry.get("status") or "").strip().lower()
    if status and status != "success":
        return True
    response = entry.get("upstream_response")
    return isinstance(response, dict) and bool(response.get("error"))


def _trajectory_last_action(blocks: list[dict[str, Any]]) -> str:
    for block in reversed(blocks):
        btype = block.get("type")
        if btype == "tool_use":
            return f"tool: {block.get('name') or 'unknown'}"
        if btype == "thinking":
            return "reasoning"
        if btype == "text":
            return "assistant message"
    return ""


def load_step_context(proxy_jsonl: Path, step_index: int) -> dict[str, Any]:
    """Load detailed proxy context for a single step."""
    if not proxy_jsonl.exists():
        return {}
    entries = _load_success_entries(proxy_jsonl)
    if 0 <= step_index < len(entries):
        return _build_step_context(entries, step_index)
    return {}


def load_trial_step_context(trial_dir: Path, step_index: int) -> dict[str, Any]:
    """Load detailed context for one trajectory step from raw ``proxy.jsonl``."""

    return load_step_context(resolve_trial_proxy_jsonl(trial_dir), step_index)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _looks_like_trial_dir(trial_dir: Path) -> bool:
    return is_trial_dir(trial_dir)


def _resolve_indexed_trial_proxy_jsonl(trial_dir: Path) -> Path | None:
    """Resolve an indexed raw proxy log for a canonical trial path.

    The resolver intentionally uses ``ExperimentArtifactReader`` instead of
    opening guessed files below ``trials/``. Returning ``None`` means either the
    run is legacy, the canonical snapshot is incomplete, or the trial has no
    indexed raw proxy artifact; callers can then decide which compatibility
    fallback is appropriate.
    """

    for artifact in _indexed_trial_artifact_files(trial_dir):
        if artifact.kind not in _RAW_PROXY_ARTIFACT_KINDS:
            continue
        if artifact.path.is_file():
            return artifact.path
    return None


def _add_indexed_trial_file_entries(
    trial_dir: Path,
    entries: list[TrialFileEntry],
) -> None:
    """Append file-tree rows for indexed artifacts not seen in the legacy walk.

    Directory artifacts are intentionally represented by their indexed root
    row. The root is the canonical contract boundary; callers can offer a zip
    download without treating every child file as its own artifact ref.
    """

    base = trial_dir.expanduser().resolve()
    seen_paths = {entry.path.expanduser().resolve() for entry in entries}
    seen_relative = {entry.relative_path for entry in entries}
    for artifact in _indexed_trial_artifact_files(trial_dir):
        resolved = artifact.path.expanduser().resolve()
        if resolved in seen_paths:
            continue
        relative_path = _display_relative_artifact_path(
            base,
            artifact.ref_path,
            resolved,
        )
        if relative_path in seen_relative:
            continue
        is_dir = resolved.is_dir()
        size_bytes = 0
        size_label = ""
        if not is_dir:
            try:
                size_bytes = resolved.stat().st_size
            except OSError:
                continue
            size_label = _format_file_size(size_bytes)
        entries.append(
            TrialFileEntry(
                path=resolved,
                relative_path=relative_path,
                name=resolved.name,
                depth=max(0, len(Path(relative_path).parts) - 1),
                is_dir=is_dir,
                size_bytes=size_bytes,
                size_label=size_label,
                artifact_kind=artifact.kind,
            )
        )
        seen_paths.add(resolved)
        seen_relative.add(relative_path)


_run_snapshot_cache: dict[Path, tuple[tuple[int, int], Any]] = {}
_run_snapshot_cache_lock = threading.RLock()


def _cached_run_snapshot(run_dir: Path):
    """Memoized ``ExperimentArtifactReader(run_dir).try_load_snapshot()``.

    A canonical snapshot loads the run's experiment_record + artifact_index +
    EVERY trial record + all events (1000+ files for a 100-trial run). A single
    trial-detail render resolves several artifacts and otherwise reloads that
    whole snapshot ~8× — re-reading the same ~1700 files each time (≈8s on NAS).
    Cache it on the (artifact_index, experiment_record) mtimes so those calls
    share one load and a settled run is read once. (The proper long-term home is
    a cache inside ExperimentArtifactReader; kept here for now so reader.py — an
    active refactor surface — stays untouched.)
    """
    sig = (
        safe_mtime_ns(run_dir / "artifact_index.json"),
        safe_mtime_ns(run_dir / EXPERIMENT_RECORD_FILENAME),
    )
    with _run_snapshot_cache_lock:
        cached = _run_snapshot_cache.get(run_dir)
        if cached is not None and cached[0] == sig:
            return cached[1]
    snapshot = ExperimentArtifactReader(run_dir).try_load_snapshot()
    with _run_snapshot_cache_lock:
        _run_snapshot_cache[run_dir] = (sig, snapshot)
        if len(_run_snapshot_cache) > 64:
            for key in list(_run_snapshot_cache)[: len(_run_snapshot_cache) - 64]:
                _run_snapshot_cache.pop(key, None)
    return snapshot


def _indexed_trial_artifact_files(trial_dir: Path) -> list[ResolvedTrialArtifact]:
    """Return indexed artifacts attached to the trial represented by path.

    Maps the virtual trial path to its canonical ``TrialRecord`` (web concern),
    then delegates the "on record AND in ArtifactIndex, resolved" policy to
    :meth:`ExperimentArtifactReader.resolve_trial_artifacts` so web cannot
    accidentally bless unindexed path guesses.
    """

    context = _indexed_trial_record_context(trial_dir)
    if context is None:
        return []
    run_dir, trial_record, _, _ = context
    return ExperimentArtifactReader(run_dir).resolve_trial_artifacts(trial_record)


def _indexed_trial_record_row(trial_dir: Path) -> dict[str, Any]:
    """Return the canonical TrialRecord web row for a virtual trial path."""

    context = _indexed_trial_record_context(trial_dir)
    if context is None:
        return {}
    run_dir, trial_record, record_ref, trial_index = context
    return _canonical_trial_record_row_from_contract(
        run_dir=run_dir,
        trial_record=trial_record,
        record_ref=record_ref,
        trial_index=trial_index,
    )


def _indexed_trial_record_context(
    trial_dir: Path,
) -> tuple[Path, TrialRecord, str, int] | None:
    """Resolve a virtual ``.../trials/<id>`` path to its TrialRecord."""

    run_and_trial = _run_dir_and_trial_id_from_trial_path(trial_dir)
    if run_and_trial is None:
        return None
    run_dir, trial_id = run_and_trial
    snapshot = _cached_run_snapshot(run_dir)
    if snapshot is None:
        return None
    record_refs = {
        ref.trial_id: ref.record_ref
        for ref in snapshot.record.trials.records
    }
    for trial_index, trial_record in enumerate(snapshot.trial_records):
        if trial_record.trial_id == trial_id:
            return (
                run_dir,
                trial_record,
                record_refs.get(trial_id, ""),
                trial_index,
            )
    return None


def _display_relative_artifact_path(
    trial_dir: Path,
    artifact_ref: str,
    artifact_path: Path,
) -> str:
    """Return the file-tree label for an indexed artifact.

    Artifacts physically under the trial directory keep the familiar
    trial-relative label such as ``proxy/proxy.jsonl``. Artifacts elsewhere in
    the canonical run use their run-relative artifact ref so users can see
    where the durable record actually lives.
    """

    try:
        return artifact_path.relative_to(trial_dir).as_posix()
    except ValueError:
        return artifact_ref


def _run_dir_and_trial_id_from_trial_path(trial_dir: Path) -> tuple[Path, str] | None:
    """Infer ``run_dir`` and canonical ``trial_id`` from ``.../trials/<id>``.

    Canonical web rows can point at virtual trial paths even when that directory
    does not exist on disk. Splitting on the final ``trials`` path segment keeps
    nested trial ids such as ``pb-siyucms/pass_1`` intact while avoiding any
    assumptions about the run directory's parent layout.
    """

    resolved = trial_dir.expanduser().resolve()
    parts = resolved.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] != TRIALS_DIRNAME:
            continue
        trial_parts = parts[index + 1:]
        if not trial_parts:
            return None
        run_dir = Path(*parts[:index]).resolve()
        trial_id = "/".join(trial_parts)
        return run_dir, trial_id
    return None


def _load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_success_entries(proxy_jsonl: Path) -> list[dict[str, Any]]:
    """Load successful proxy entries with a small mtime/size keyed cache."""
    path = proxy_jsonl.resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return []
    signature = (stat.st_mtime_ns, stat.st_size)
    with _entry_cache_lock:
        cached = _entry_cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") != "success":
                continue
            entries.append(entry)

    with _entry_cache_lock:
        if len(_entry_cache) >= _ENTRY_CACHE_MAX:
            _entry_cache.pop(next(iter(_entry_cache)))
        _entry_cache[path] = (signature, entries)
    return entries


def _load_success_entries_until(
    proxy_jsonl: Path,
    *,
    stop_after: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Load successful entries until ``stop_after`` successes or EOF.

    This keeps ordinary paginated trajectory requests from materialising
    large ``proxy.jsonl`` files just to render the first screen.
    """
    if stop_after <= 0:
        return [], False
    entries: list[dict[str, Any]] = []
    reached_eof = True
    with proxy_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if entry.get("status") != "success":
                continue
            entries.append(entry)
            if len(entries) >= stop_after:
                reached_eof = False
                break
    return entries, reached_eof


def _collect_tool_results_by_id(entries: list[dict[str, Any]]) -> dict[str, str]:
    tool_results_by_id: dict[str, str] = {}

    for entry in entries:
        req, _ = _extract_request_body(entry)
        tool_results_by_id.update(_tool_results_from_body(req))

    for i in range(len(entries)):
        resp = entries[i].get("upstream_response") or {}
        choices = resp.get("choices", [])
        if not choices:
            continue
        resp_msg = choices[0].get("message", {})
        resp_tool_calls = resp_msg.get("tool_calls") or []
        if not resp_tool_calls:
            continue
        resp_ids = [tc.get("id", "") for tc in resp_tool_calls]
        if all(tid in tool_results_by_id for tid in resp_ids):
            continue
        n_tools = len(resp_ids)

        resp_fp = []
        resp_semantic = []
        for tc in resp_tool_calls:
            fn = tc.get("function", {})
            resp_fp.append((fn.get("name", ""), fn.get("arguments", "")[:200]))
            resp_semantic.append((
                fn.get("name", ""),
                _parse_tool_arguments(fn.get("arguments", tc.get("arguments", {}))),
            ))

        def _try_match(strict: bool) -> bool:
            for j in range(i + 1, min(i + 6, len(entries))):
                next_msgs = (entries[j].get("openai_request") or {}).get("messages", [])
                for mi, m in enumerate(next_msgs):
                    if m.get("role") != "assistant" or not m.get("tool_calls"):
                        continue
                    asst_tcs = m.get("tool_calls", [])
                    if len(asst_tcs) != n_tools:
                        continue
                    if strict:
                        asst_fp = [
                            (
                                tc.get("function", {}).get("name", ""),
                                tc.get("function", {}).get("arguments", "")[:200],
                            )
                            for tc in asst_tcs
                        ]
                        if asst_fp != resp_fp:
                            continue
                    else:
                        asst_semantic = [
                            (
                                tc.get("function", {}).get("name", ""),
                                _parse_tool_arguments(
                                    tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                                ),
                            )
                            for tc in asst_tcs
                        ]
                        if asst_semantic != resp_semantic:
                            continue
                    trail: list[str] = []
                    for mi2 in range(mi + 1, len(next_msgs)):
                        if next_msgs[mi2].get("role") == "tool":
                            trail.append(str(next_msgs[mi2].get("content", "")))
                        else:
                            break
                    if len(trail) == n_tools:
                        for tid, result in zip(resp_ids, trail):
                            if tid not in tool_results_by_id:
                                tool_results_by_id[tid] = result
                        return True
            return False

        if not _try_match(strict=True):
            _try_match(strict=False)

    return tool_results_by_id


def _build_step_context(entries: list[dict[str, Any]], step_index: int) -> dict[str, Any]:
    entry = entries[step_index]
    req_body, req_source = _extract_request_body(entry)
    req_messages = _extract_request_messages(req_body)
    tools = req_body.get("tools") or []
    system = _extract_system(req_body)
    usage = _extract_usage(entry)
    response = _extract_response_summary(entry)
    if not response.get("content"):
        tool_results = _collect_tool_results_by_id(entries)
        blocks = _extract_blocks(entry, tool_results)
        if not blocks:
            blocks = _infer_response_blocks_from_next_requests(
                entries,
                tool_results,
                start=step_index,
                end=step_index + 1,
            ).get(
                step_index,
                [],
            )
        if blocks:
            response["content"] = [_summary_content_from_block(block) for block in blocks]
            response["tool_names"] = [
                str(block.get("name") or "")
                for block in blocks
                if block.get("type") == "tool_use" and block.get("name")
            ]

    role_counts: dict[str, int] = {}
    for msg in req_messages:
        role = str(msg.get("role") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "step_index": step_index,
        "raw": entry,
        "summary": {
            "request_id": entry.get("request_id", ""),
            "trial_id": entry.get("trial_id", ""),
            "status": entry.get("status", ""),
            "model": req_body.get("model", ""),
            "ts_ms": entry.get("ts_ms", 0),
            "message_count": len(req_messages),
            "tool_count": len(tools) if isinstance(tools, list) else 0,
            "response_tool_count": len(response["tool_names"]),
            "system_chars": len(system),
            "tokens": usage,
            "stop_reason": response.get("stop_reason", ""),
        },
        "request": {
            "source": req_source,
            "body": req_body,
            "model": req_body.get("model", ""),
            "system": system,
            "system_chars": len(system),
            "messages": req_messages,
            "role_counts": dict(sorted(role_counts.items())),
            "tools": tools if isinstance(tools, list) else [],
            "params": _extract_request_params(req_body),
        },
        "response": response,
        "context_diff": _build_context_diff(entries[:step_index + 1], step_index),
    }


def _extract_request_body(entry: dict[str, Any]) -> tuple[dict[str, Any], str]:
    anthropic = entry.get("anthropic_request")
    if isinstance(anthropic, dict) and anthropic:
        return anthropic, "anthropic_request"
    openai = entry.get("openai_request")
    if isinstance(openai, dict) and openai:
        return openai, "openai_request"
    return {}, ""


def _extract_request_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]

    input_items = body.get("input")
    if not isinstance(input_items, list):
        return []
    result: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if not isinstance(role, str) or not role:
            continue
        if item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = [
                {**c, "type": "text"}
                if isinstance(c, dict) and c.get("type") in {"input_text", "output_text"}
                else c
                for c in content
            ]
        result.append({"role": role, "content": content})
    return result


def _extract_system(body: dict[str, Any]) -> str:
    system = body.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for item in system:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n\n".join(p for p in parts if p)
    instructions = body.get("instructions")
    return instructions if isinstance(instructions, str) else ""


def _extract_request_params(body: dict[str, Any]) -> dict[str, Any]:
    skip = {"messages", "system", "tools", "input", "instructions", "model"}
    return {key: value for key, value in body.items() if key not in skip}


def _extract_response_summary(entry: dict[str, Any]) -> dict[str, Any]:
    anthropic = entry.get("anthropic_response") or {}
    upstream = entry.get("upstream_response") or {}
    body = anthropic if isinstance(anthropic, dict) and anthropic else upstream

    blocks: list[dict[str, Any]] = []
    if isinstance(anthropic, dict):
        blocks.extend(_extract_response_blocks_from_body(anthropic, {}))
    upstream_blocks = (
        _extract_response_blocks_from_body(upstream, {})
        if isinstance(upstream, dict)
        else []
    )
    if not blocks:
        blocks = upstream_blocks
    elif upstream_blocks and not any(block.get("type") == "thinking" for block in blocks):
        thinking = [block for block in upstream_blocks if block.get("type") == "thinking"]
        blocks = thinking + blocks

    content = [_summary_content_from_block(block) for block in blocks]
    tool_names = [
        str(block.get("name") or "")
        for block in blocks
        if block.get("type") == "tool_use"
    ]

    return {
        "body": body if isinstance(body, dict) else {},
        "content": content,
        "tool_names": [n for n in tool_names if n],
        "usage": _extract_display_usage(entry),
        "stop_reason": _extract_response_stop_reason(anthropic, upstream),
        "error": entry.get("error", ""),
    }


def _summary_content_from_block(block: dict[str, Any]) -> dict[str, Any]:
    btype = block.get("type")
    if btype == "thinking":
        return {"type": "thinking", "text": str(block.get("content") or "")}
    if btype == "text":
        return {"type": "text", "text": str(block.get("content") or "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "name": block.get("name", "unknown"),
            "input": block.get("input", {}),
        }
    return {"type": str(btype or "json"), "value": block}


def _extract_response_stop_reason(*responses: Any) -> str:
    for response in responses:
        if not isinstance(response, dict):
            continue
        stop_reason = response.get("stop_reason")
        if stop_reason:
            return str(stop_reason)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = first.get("finish_reason")
            if finish_reason:
                return str(finish_reason)
        status = response.get("status")
        if status:
            return str(status)
    return ""


def _extract_display_usage(entry: dict[str, Any]) -> dict[str, int]:
    usage = _extract_usage(entry)
    return {
        "input_tokens": usage["in"],
        "output_tokens": usage["out"],
        "reasoning_tokens": usage["reasoning"],
    }


def _build_context_diff(entries: list[dict[str, Any]], step_index: int) -> dict[str, Any]:
    prev_idx, is_fallback = _find_previous_context(entries)
    current_body, _ = _extract_request_body(entries[-1])
    if prev_idx is None:
        diff = _empty_context_diff(current_body=current_body)
    else:
        old_body, _ = _extract_request_body(entries[prev_idx])
        diff = _structural_request_diff(old_body, current_body)
    diff["previous_step_index"] = prev_idx
    diff["current_step_index"] = step_index
    diff["is_fallback"] = is_fallback
    return diff


def _find_previous_context(entries: list[dict[str, Any]]) -> tuple[int | None, bool]:
    if len(entries) <= 1:
        return None, False
    current_body, _ = _extract_request_body(entries[-1])
    target_hashes = _message_hashes(current_body)
    best_idx: int | None = None
    best_len = 0
    for idx in range(len(entries) - 2, -1, -1):
        candidate_body, _ = _extract_request_body(entries[idx])
        candidate_hashes = _message_hashes(candidate_body)
        if (
            candidate_hashes
            and _is_prefix(candidate_hashes, target_hashes)
            and len(candidate_hashes) > best_len
        ):
            best_idx = idx
            best_len = len(candidate_hashes)
    if best_idx is not None:
        return best_idx, False

    model = current_body.get("model")
    for idx in range(len(entries) - 2, -1, -1):
        candidate_body, _ = _extract_request_body(entries[idx])
        if candidate_body.get("model") == model:
            return idx, True
    return len(entries) - 2, True


def _empty_context_diff(current_body: dict[str, Any]) -> dict[str, Any]:
    system = _extract_system(current_body)
    tools = current_body.get("tools") if isinstance(current_body.get("tools"), list) else []
    return {
        "previous_step_index": None,
        "current_step_index": None,
        "is_fallback": False,
        "messages": {
            "old_count": 0,
            "new_count": len(_extract_request_messages(current_body)),
            "unchanged_prefix": 0,
            "unchanged_suffix": 0,
            "added_count": 0,
            "removed_count": 0,
            "modified_count": 0,
            "added": [],
            "removed": [],
            "modified": [],
        },
        "system": {
            "changed": False,
            "old_chars": 0,
            "new_chars": len(system),
            "old_text": "",
            "new_text": system,
        },
        "tools": {
            "old_count": 0,
            "new_count": len(tools),
            "added": [],
            "removed": [],
            "changed": bool(tools),
        },
        "params": [],
    }


def _structural_request_diff(old_body: dict[str, Any], new_body: dict[str, Any]) -> dict[str, Any]:
    old_msgs = _extract_request_messages(old_body)
    new_msgs = _extract_request_messages(new_body)
    prefix = 0
    while prefix < min(len(old_msgs), len(new_msgs)) and _messages_equal(
        old_msgs[prefix],
        new_msgs[prefix],
    ):
        prefix += 1

    suffix = 0
    while (
        suffix < min(len(old_msgs) - prefix, len(new_msgs) - prefix)
        and _messages_equal(
            old_msgs[len(old_msgs) - 1 - suffix],
            new_msgs[len(new_msgs) - 1 - suffix],
        )
    ):
        suffix += 1

    old_tail = old_msgs[prefix:len(old_msgs) - suffix if suffix else len(old_msgs)]
    new_tail = new_msgs[prefix:len(new_msgs) - suffix if suffix else len(new_msgs)]
    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []
    old_idx = new_idx = 0
    while old_idx < len(old_tail) and new_idx < len(new_tail):
        old_msg = old_tail[old_idx]
        new_msg = new_tail[new_idx]
        if old_msg.get("role") == new_msg.get("role"):
            if not _messages_equal(old_msg, new_msg):
                modified.append({
                    "old": _compact_message(old_msg),
                    "new": _compact_message(new_msg),
                })
            old_idx += 1
            new_idx += 1
        elif (
            old_idx + 1 < len(old_tail)
            and old_tail[old_idx + 1].get("role") == new_msg.get("role")
        ):
            removed.append(_compact_message(old_msg))
            old_idx += 1
        elif (
            new_idx + 1 < len(new_tail)
            and old_msg.get("role") == new_tail[new_idx + 1].get("role")
        ):
            added.append(_compact_message(new_msg))
            new_idx += 1
        else:
            removed.append(_compact_message(old_msg))
            added.append(_compact_message(new_msg))
            old_idx += 1
            new_idx += 1
    while old_idx < len(old_tail):
        removed.append(_compact_message(old_tail[old_idx]))
        old_idx += 1
    while new_idx < len(new_tail):
        added.append(_compact_message(new_tail[new_idx]))
        new_idx += 1

    old_system = _extract_system(old_body)
    new_system = _extract_system(new_body)
    old_tool_names = _tool_names(old_body)
    new_tool_names = _tool_names(new_body)
    old_tool_set = set(old_tool_names)
    new_tool_set = set(new_tool_names)

    return {
        "previous_step_index": None,
        "current_step_index": None,
        "is_fallback": False,
        "messages": {
            "old_count": len(old_msgs),
            "new_count": len(new_msgs),
            "unchanged_prefix": prefix,
            "unchanged_suffix": suffix,
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
            "added": added,
            "removed": removed,
            "modified": modified,
        },
        "system": {
            "changed": old_system != new_system,
            "old_chars": len(old_system),
            "new_chars": len(new_system),
            "old_text": old_system,
            "new_text": new_system,
        },
        "tools": {
            "old_count": len(old_tool_names),
            "new_count": len(new_tool_names),
            "added": [name for name in new_tool_names if name not in old_tool_set],
            "removed": [name for name in old_tool_names if name not in new_tool_set],
            "changed": old_tool_names != new_tool_names,
        },
        "params": _field_changes(
            _extract_request_params(old_body),
            _extract_request_params(new_body),
        ),
    }


def _tool_names(body: dict[str, Any]) -> list[str]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [str(t.get("name") or "") for t in tools if isinstance(t, dict) and t.get("name")]


def _field_changes(old_params: dict[str, Any], new_params: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for key in sorted(set(old_params) | set(new_params)):
        old_value = old_params.get(key)
        new_value = new_params.get(key)
        if old_value != new_value:
            changes.append({"key": key, "old": old_value, "new": new_value})
    return changes


def _message_hashes(body: dict[str, Any]) -> list[str]:
    return [
        f"{msg.get('role', '')}:{_message_text(msg)[:500]}"
        for msg in _extract_request_messages(body)
    ]


def _is_prefix(shorter: list[str], longer: list[str]) -> bool:
    return len(shorter) <= len(longer) and shorter == longer[:len(shorter)]


def _messages_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("role") == right.get("role") and _message_text(left) == _message_text(right)


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": message.get("role", "unknown"),
        "content": message.get("content", ""),
        "text": _message_text(message),
    }


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        clean = {k: v for k, v in block.items() if k != "cache_control"}
        btype = clean.get("type")
        if btype in {"text", "input_text", "output_text"}:
            parts.append(str(clean.get("text") or ""))
        elif btype == "thinking":
            parts.append("[thinking]\n" + str(clean.get("thinking") or ""))
        elif btype == "tool_use":
            parts.append(
                f"[tool_use: {clean.get('name') or ''}]\n"
                + json.dumps(clean.get("input", {}), ensure_ascii=False, sort_keys=True)
            )
        elif btype == "tool_result":
            value = clean.get("content", "")
            if isinstance(value, list):
                value = "\n".join(
                    str(v.get("text") if isinstance(v, dict) and v.get("type") == "text" else v)
                    for v in value
                )
            parts.append("[tool_result]\n" + str(value))
        else:
            parts.append(json.dumps(clean, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _format_file_size(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _classify_trial_status(
    trial_dir: Path,
    info: dict[str, Any],
    *,
    run_status: str = "",
    live_check_enabled: bool | None = None,
) -> dict[str, str]:
    """Map an on-disk trial to a presentation chip (``status_kind`` + labels).

    This is deliberately richer than the canonical
    ``contracts.trial_status.classify_trial_status``: it overlays the
    live-check verdict (a clean "completed" with no live success becomes a
    "Max rounds" warning), timeout detection, and exit-code fallbacks that
    only the dashboard cares about. It shares the canonical *status sets*
    (so "what counts as completed/interrupted" stays single-sourced) but
    keeps its own classification flow — do not collapse it onto the shared
    classifier or the header/chip nuance is lost.

    ``live_check_enabled`` overrides the on-disk ``check_done_polls.jsonl``
    probe: pass ``True``/``False`` to classify without a per-trial filesystem
    stat (the settled-run overview path, which builds rows from the dashboard
    projection and accepts that the "Max rounds" nuance is only resolved on the
    trial-detail page). ``None`` keeps the authoritative disk probe.
    """

    if info.get("running"):
        # When the *run* was interrupted (Ctrl+C), the loop has stopped — nothing
        # is actually running anymore, even though this trial's meta.json was
        # never finalized past "running". Showing "Running" in the terminal
        # partial-results table (or a settled interrupted run) is misleading;
        # the trial was stopped mid-flight, so present it as Interrupted. The
        # live inspector passes no run_status, so in-flight trials still show
        # "Running" there.
        if str(run_status or "").lower() == "interrupted":
            return {
                "status_label": "Interrupted",
                "status_detail": "Stopped by user (Ctrl+C) while still running",
                "status_kind": "warning",
            }
        progress = info.get("progress") if isinstance(info.get("progress"), dict) else {}
        last_status = progress.get("last_status")
        detail = f"Last proxy status: {last_status}" if last_status else "Agent is still running"
        return {"status_label": "Running", "status_detail": detail, "status_kind": "running"}

    reason = str(info.get("termination_reason") or "").strip()
    detail = _normalize_status_detail(reason, info.get("termination_detail", "")).strip()
    if _is_resume_policy_repair_detail(info.get("termination_source"), detail):
        detail = ""
    if reason:
        # Reclassify "completed" against the live-check verdict for
        # task-mode runs. A clean agent exit with no live_success and
        # no scorer-confirmed success stopped at the round budget; don't
        # display it as a completed result.
        live_enabled = (
            _live_check_was_enabled(trial_dir, info)
            if live_check_enabled is None
            else live_check_enabled
        )
        if (
            reason.lower() == "completed"
            and live_enabled
            and not _live_check_passed(info)
        ):
            return {
                "status_label": "Max rounds",
                "status_detail": "Agent exited cleanly but never satisfied the live check",
                "status_kind": "warning",
            }
        return _status_from_reason(reason, detail=detail, exit_code=info.get("exit_code"))

    error = str(info.get("error") or "").strip()
    if error:
        if _looks_like_timeout(error):
            return {
                "status_label": "Timed out",
                "status_detail": error,
                "status_kind": "error",
            }
        return {"status_label": "Trial error", "status_detail": error, "status_kind": "error"}

    status = str(info.get("status") or "").strip().lower()
    if status in COMPLETED_TRIAL_STATUSES:
        return {
            "status_label": "Completed",
            "status_detail": "Task finished",
            "status_kind": "success",
        }
    if status in INTERRUPTED_TRIAL_STATUSES:
        return {
            "status_label": "Interrupted",
            "status_detail": "Stopped before completion",
            "status_kind": "warning",
        }
    if status == "failed":
        return {
            "status_label": "Trial failed",
            "status_detail": "Trial did not finish successfully",
            "status_kind": "error",
        }

    exit_code = info.get("exit_code")
    if str(run_status or "").lower() == "interrupted":
        return {
            "status_label": "Interrupted",
            "status_detail": "Stopped by user (Ctrl+C)",
            "status_kind": "warning",
        }

    if exit_code == 0:
        return {
            "status_label": "Completed",
            "status_detail": "Task finished",
            "status_kind": "success",
        }

    if exit_code is not None:
        return {
            "status_label": "Agent failed",
            "status_detail": f"Exited with code {exit_code}",
            "status_kind": "error",
        }
    return {"status_label": "Pending", "status_detail": "No result yet", "status_kind": "pending"}


# Mapping from canonical termination_reason → display chip.
#
# ``status_kind`` drives chip color in templates:
#   * "success"  — green: agent reached a normal stop (completed / max rounds)
#   * "warning"  — yellow: agent hit an expected budget (timeout / tool limit)
#   * "error"    — red: unexpected failure (upstream, OOM, target, exception)
#   * "pending"  — gray: never started / cancelled
#   * "live_success" — special green for the live-check fast-path
#
# Keep status_label short (≤ 18 chars) — it appears as a chip in dense lists.
_REASON_DISPLAY: dict[str, dict[str, str]] = {
    # ── Expected: agent reached a natural stop ─────────────────────────
    "completed": {
        "label": "Completed", "kind": "success",
        "default_detail": "Task finished",
    },
    "max_rounds_reached": {
        "label": "Max rounds", "kind": "warning",
        "default_detail": "Agent reached the configured max-rounds budget",
    },
    "max_input_tokens_reached": {
        "label": "Input tokens", "kind": "warning",
        "default_detail": "Agent reached the configured input-token budget",
    },
    "max_output_tokens_reached": {
        "label": "Output tokens", "kind": "warning",
        "default_detail": "Agent reached the configured output-token budget",
    },
    "max_cost_reached": {
        "label": "Max cost", "kind": "warning",
        "default_detail": "Agent reached the configured cost budget",
    },
    "execution_timeout": {
        "label": "Timed out", "kind": "warning",
        "default_detail": "Agent execution exceeded the configured trial timeout",
    },
    "tool_limit": {
        "label": "Tool limit", "kind": "warning",
        "default_detail": "Stopped after reaching the configured tool-call limit",
    },
    # ── Upstream model proxy errors (all unexpected) ───────────────────
    "model_quota_exhausted": {
        "label": "Quota exhausted", "kind": "error",
        "default_detail": "Upstream returned 429 usage_limit_reached",
    },
    "model_rate_limited": {
        "label": "Rate limited", "kind": "error",
        "default_detail": "Upstream returned 429 rate_limit",
    },
    "model_bad_gateway": {
        "label": "Bad gateway", "kind": "error",
        "default_detail": "Upstream returned 502/503/504 (or DNS/connection failure)",
    },
    "model_auth_error": {
        "label": "Auth error", "kind": "error",
        "default_detail": "Upstream rejected the request (401/403)",
    },
    "model_context_overflow": {
        "label": "Context overflow", "kind": "error",
        "default_detail": "Request exceeded the model's context window",
    },
    "model_timeout": {
        "label": "Model timeout", "kind": "error",
        "default_detail": "Upstream request timed out",
    },
    "model_error": {
        "label": "Model error", "kind": "error",
        "default_detail": "Upstream returned an HTTP error",
    },
    # ── Infra failures (all unexpected) ────────────────────────────────
    "oom_killed": {
        "label": "OOM killed", "kind": "error",
        "default_detail": "Process killed by SIGKILL (exit 137) — likely OOM",
    },
    "target_unavailable": {
        "label": "Target unavailable", "kind": "error",
        "default_detail": "Target stack failed to launch — agent was not invoked",
    },
    "trial_error": {
        "label": "Trial error", "kind": "error",
        "default_detail": "Trial raised an exception",
    },
    # ── State ──────────────────────────────────────────────────────────
    "user_interrupted": {
        # Rendered as a red "failed" chip so the run banner and home card
        # surface interrupted trials in the same bucket as model/infra
        # errors. They're not "successful runs" — even though the cause
        # is user input, the trial produced no usable result, and should
        # be visible in the failure count instead of buried in warnings.
        "label": "Interrupted", "kind": "error",
        "default_detail": "Stopped by user (Ctrl+C)",
    },
    "cancelled_before_start": {
        "label": "Cancelled", "kind": "pending",
        "default_detail": "Run stopped before this trial started",
    },
    "agent_exit_nonzero": {
        "label": "Agent failed", "kind": "error",
        "default_detail": "",  # populated with "Exited with code N"
    },
    "live_success": {
        "label": "Target passed", "kind": "live_success",
        "default_detail": "Target validator reported success.",
    },
}

# Legacy / synonym mappings — keep older meta.json files readable when the
# canonical reason name has moved or absorbed multiple historical variants.
_REASON_ALIASES: dict[str, str] = {
    "trial_timeout": "execution_timeout",
    "timeout": "execution_timeout",
    "proxy_timeout": "model_timeout",
    "upstream_timeout": "model_timeout",
    "interrupted": "user_interrupted",
    "ctrl_c": "user_interrupted",
    "cancelled": "cancelled_before_start",
    "terminated_by_limit": "tool_limit",
}


def _status_from_reason(reason: str, *, detail: str, exit_code: Any) -> dict[str, str]:
    normalized = (reason or "").lower()
    normalized = _REASON_ALIASES.get(normalized, normalized)
    config = _REASON_DISPLAY.get(normalized)
    if config is not None:
        if normalized == "target_unavailable":
            # ``detail`` may embed the full upstream response body (docker
            # compose log etc.). Keep the inline chip readable by picking
            # the first non-empty line; the full text still renders in the
            # Termination section via ``termination.detail``.
            first_line = next(
                (ln.strip() for ln in (detail or "").splitlines() if ln.strip()),
                "",
            )
            display_detail = first_line or config["default_detail"]
        elif normalized == "agent_exit_nonzero":
            display_detail = detail or f"Exited with code {exit_code}"
        elif normalized == "model_timeout":
            display_detail = detail or config["default_detail"]
        else:
            display_detail = detail or config["default_detail"]
        return {
            "status_label": config["label"],
            "status_detail": display_detail,
            "status_kind": config["kind"],
        }
    # Unknown reason — fall back to a humanized label so the UI still shows
    # something useful instead of a blank chip.
    return {
        "status_label": (reason or "Unknown").replace("_", " ").title(),
        "status_detail": detail or "Recorded by trial metadata",
        "status_kind": "warning",
    }


def _looks_like_timeout(text: str) -> bool:
    lowered = text.lower()
    return "timeout" in lowered or "timed out" in lowered


def _live_check_was_enabled(trial_dir: Path, info: dict[str, Any]) -> bool:
    """True if this trial actually ran a live-check poller.

    Detection is **artifact-based on disk only**. The orchestrator now
    writes the ``live_success`` / ``live_success_verdict`` keys to every
    trial's meta.json regardless of whether live-check was configured
    (see ``execute_trial`` storage.save_trial_meta call), so those
    keys are NOT a reliable signal — they made every completed trial in
    a benchmark with ``live_check.enabled: false`` get mis-labelled as
    "Max rounds" because the verdict is always empty.

    ``runtime/check_done_polls.jsonl`` is only created by the live-check
    service inside the agent container, so its presence is definitive.
    """
    polls = trial_dir / "runtime" / "check_done_polls.jsonl"
    try:
        return polls.is_file() and polls.stat().st_size > 0
    except OSError:
        return False


def _live_check_passed(info: dict[str, Any]) -> bool:
    """True if any live-check evidence in *info* fired positive."""
    if info.get("live_success") is True:
        return True
    verdict = info.get("live_success_verdict")
    if isinstance(verdict, dict) and verdict.get("success") is True:
        return True
    return False

def _extract_usage(entry: dict[str, Any]) -> dict[str, int]:
    """Extract token usage from a proxy entry."""
    usage = extract_entry_usage(entry)
    return {
        "in": usage["input_tokens"],
        "out": usage["output_tokens"],
        "reasoning": usage["reasoning_tokens"],
    }


def _infer_response_blocks_from_next_requests(
    entries: list[dict[str, Any]],
    tool_results: dict[str, str],
    *,
    start: int = 0,
    end: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    inferred: dict[int, list[dict[str, Any]]] = {}
    stop = min(len(entries) - 1, len(entries) if end is None else end)
    for idx in range(max(0, start), stop):
        current_body, _ = _extract_request_body(entries[idx])
        next_body, _ = _extract_request_body(entries[idx + 1])
        current_items = _extract_request_history_items(current_body)
        next_items = _extract_request_history_items(next_body)
        prefix = 0
        while (
            prefix < len(current_items)
            and prefix < len(next_items)
            and _history_items_equal(current_items[prefix], next_items[prefix])
        ):
            prefix += 1
        if prefix < len(current_items):
            continue

        appended = next_items[prefix:]
        local_results = _tool_results_from_history_items(appended)
        inferred_blocks = _blocks_from_responses_items(
            appended,
            {**tool_results, **local_results},
        )
        if inferred_blocks:
            inferred[idx] = inferred_blocks
    return inferred


def _extract_request_history_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    input_items = body.get("input")
    if isinstance(input_items, list):
        return [item for item in input_items if isinstance(item, dict)]
    messages = body.get("messages")
    if isinstance(messages, list):
        return [msg for msg in messages if isinstance(msg, dict)]
    return []


def _history_items_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("role") is not None or right.get("role") is not None:
        return _messages_equal(left, right)
    return _history_item_text(left) == _history_item_text(right)


def _history_item_text(item: dict[str, Any]) -> str:
    if item.get("type") == "function_call":
        return json.dumps(
            {
                "type": item.get("type"),
                "name": item.get("name"),
                "call_id": item.get("call_id") or item.get("id"),
                "arguments": item.get("arguments"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if item.get("type") == "function_call_output":
        return json.dumps(
            {
                "type": item.get("type"),
                "call_id": item.get("call_id") or item.get("id"),
                "output": item.get("output"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def _tool_results_from_body(body: dict[str, Any]) -> dict[str, str]:
    return _tool_results_from_history_items(_extract_request_history_items(body))


def _tool_results_from_history_items(items: list[dict[str, Any]]) -> dict[str, str]:
    results: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            call_id = str(item.get("call_id") or item.get("id") or "")
            if call_id:
                results[call_id] = _stringify_tool_result(
                    item.get("output", item.get("content", ""))
                )
            continue
        results.update(_tool_results_from_messages([item]))
    return results


def _tool_results_from_messages(messages: list[dict[str, Any]]) -> dict[str, str]:
    results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            tool_id = str(msg.get("tool_call_id") or msg.get("id") or "")
            if tool_id:
                results[tool_id] = _stringify_tool_result(msg.get("content", ""))
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_id = str(block.get("tool_use_id") or "")
            if tool_id:
                results[tool_id] = _stringify_tool_result(block.get("content", ""))
    return results


def _stringify_tool_result(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")
