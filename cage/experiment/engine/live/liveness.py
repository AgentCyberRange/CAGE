"""Run liveness detection — "is this ``cage run`` still ticking?"

A run is considered alive when its ``.cage_runs/<rid>/`` directory shows
evidence of recent activity:

  * No ``completed_at`` in ``dashboard.json``; AND
  * Some ``trials/.../proxy/progress.json`` was modified within the last
    ``LIVE_WINDOW`` seconds (5 minutes by default); OR
  * The dashboard reports pending work and the orchestrator only just
    started (no progress.json yet but ``planned_trials.json`` or the
    canonical ``experiment_record.json`` is fresh).
  * A canonical ``TrialRecord`` is in an active runtime status and its record
    file was updated within the live window.

The decision tree mirrors ``cage.web.data._build_run_info`` (which is
where the same logic powers the Web inspector's "running" badge). Both
``cage gc`` and the inspector should answer the same question the same
way; that's why the policy lives here in one place instead of being
duplicated.

Used by:
  * ``cage gc`` — to decide whether a run_id is reclaim-eligible.
  * ``cage/web/data/__init__.py`` — same answer in the inspector UI.

Reads ``.cage_runs/<rid>/`` only. Never writes. ``.cage_runs/`` is the
single source of truth for run history and must remain immutable from
this module's perspective.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from cage.experiment.engine.live.fs_signals import (
    RunFsSignals,
    dashboard_pending_count,
    is_recently_active,
    safe_mtime_ns,
    scan_run_signals,
)

_ACTIVE_TRIAL_RECORD_STATUSES = frozenset({
    "starting",
    "running",
    "agent_finished",
    "verifying",
    "tearing_down",
    "judging",
})
_TERMINAL_TRIAL_RECORD_STATUSES = frozenset({
    "completed",
    "interrupted",
    "failed",
    "cancelled",
    "not_scored",
    "scored",
})

__all__ = [
    "RunLiveness",
    "is_run_running",
    "iter_known_run_ids",
    "locate_run_dir",
]


@dataclass(frozen=True)
class RunLiveness:
    """Outcome of one liveness probe.

    ``running`` is the binary answer GC needs. ``reason`` is a short
    human string useful for ``cage gc`` dry-run output.
    """
    running: bool
    reason: str


def is_run_running(run_dir: Path) -> RunLiveness:
    """Decide whether the run at ``run_dir`` is still ticking.

    Mirrors ``cage.web.data._build_run_info``'s decision tree. Five
    branches:

      1. ``dashboard.json`` carries a ``completed_at`` → dead.
      2. Some trial's ``progress.json`` ticked within the live window → alive.
      3. Progress files exist but stalled (orchestrator died) → dead.
      4. Dashboard reports pending work, no progress.json yet
         (orchestrator just started) → alive.
      5. ``planned_trials.json`` or canonical ``experiment_record.json`` is
         fresh but no dashboard yet → alive.
      Anything else → dead.
    """
    if not run_dir.exists():
        return RunLiveness(False, "run_dir missing")

    dashboard_path = run_dir / "dashboard.json"
    planned_path = run_dir / "planned_trials.json"
    record_path = run_dir / "experiment_record.json"
    dashboard_mtime = safe_mtime_ns(dashboard_path)
    planned_mtime = safe_mtime_ns(planned_path)
    record_mtime = safe_mtime_ns(record_path)

    dashboard_data: dict = {}
    if dashboard_mtime >= 0:
        try:
            dashboard_data = json.loads(dashboard_path.read_text())
        except (OSError, json.JSONDecodeError):
            dashboard_data = {}
    record_data: dict = {}
    if record_mtime >= 0:
        try:
            record_data = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            record_data = {}

    record_completed_at = str(record_data.get("completed_at") or "").strip()
    if record_completed_at:
        return RunLiveness(False, f"experiment_record completed_at={record_completed_at}")
    record_status = str(record_data.get("status") or "").strip().lower()
    if record_status in {"completed", "interrupted", "failed", "cancelled"}:
        return RunLiveness(False, f"experiment_record status={record_status}")
    trial_record_liveness = _canonical_trial_record_liveness(run_dir)
    if trial_record_liveness is not None:
        return trial_record_liveness

    if (
        record_status in {"planned", "running"}
        and record_mtime > 0
        and is_recently_active(record_mtime)
    ):
        return RunLiveness(True, f"experiment_record status={record_status}, fresh")

    completed_at = str(dashboard_data.get("completed_at") or "").strip()
    if completed_at:
        return RunLiveness(False, f"completed_at={completed_at}")

    signals: RunFsSignals = scan_run_signals(run_dir)

    if signals.active_count > 0 and is_recently_active(signals.newest_progress_mtime_ns):
        return RunLiveness(True, f"{signals.active_count} active trial(s), recent progress.json tick")
    if signals.active_count > 0:
        return RunLiveness(False, f"{signals.active_count} active trial(s) but stalled")

    dashboard_pending = dashboard_pending_count(dashboard_data)
    if dashboard_mtime >= 0 and dashboard_pending > 0:
        return RunLiveness(True, f"dashboard pending={dashboard_pending}, pre-tick")

    if dashboard_mtime < 0 and planned_mtime > 0 and is_recently_active(planned_mtime):
        return RunLiveness(True, "planned_trials.json fresh, dashboard not yet written")
    if dashboard_mtime < 0 and record_mtime > 0 and is_recently_active(record_mtime):
        return RunLiveness(True, "experiment_record.json fresh, dashboard not yet written")

    return RunLiveness(False, "no activity signal")


def _canonical_trial_record_liveness(run_dir: Path) -> RunLiveness | None:
    """Return liveness from canonical trial records when they are decisive.

    Run-level ``experiment_record.json`` can be older than the trial records it
    references because runtime phases update individual ``TrialRecord`` files as
    the trial advances. GC should not require a legacy ``proxy/progress.json``
    tick when the canonical trial record itself says the trial is active and was
    recently written.

    Canonical terminal trial records are also decisive: once every loaded
    ``TrialRecord`` is terminal, a stale legacy progress file must not keep the
    run alive. Returning ``None`` means "the canonical snapshot is absent or not
    yet complete enough to answer"; callers should continue with
    dashboard/progress compatibility fallbacks.
    """

    from cage.artifacts.reader import ExperimentArtifactReader

    reader = ExperimentArtifactReader(run_dir)
    snapshot = reader.try_load_snapshot()
    if snapshot is None:
        return None

    record_ref_by_trial = {
        ref.trial_id: ref.record_ref
        for ref in snapshot.record.trials.records
    }
    saw_trial_record = False
    all_trial_records_terminal = True
    for trial_record in snapshot.trial_records:
        saw_trial_record = True
        status = trial_record.status.strip().lower()
        if status in _ACTIVE_TRIAL_RECORD_STATUSES:
            all_trial_records_terminal = False
            record_ref = record_ref_by_trial.get(trial_record.trial_id)
            if not record_ref:
                continue
            record_mtime = safe_mtime_ns(run_dir / record_ref)
            if record_mtime > 0 and is_recently_active(record_mtime):
                return RunLiveness(
                    True,
                    f"trial_record status={status}, fresh",
                )
            continue
        if status not in _TERMINAL_TRIAL_RECORD_STATUSES:
            all_trial_records_terminal = False
    if saw_trial_record and all_trial_records_terminal:
        return RunLiveness(False, "trial_record all terminal")
    return None


def locate_run_dir(rid: str, *, search_roots: list[Path]) -> Path | None:
    """Find the ``.cage_runs/<agent>/<run_id>/`` dir for a given ``rid``.

    The directory layout under ``.cage_runs/`` is
    ``<agent>:<model>:<mode>/<run_id>/``. We don't know the agent prefix
    a priori (a run_id only identifies the timestamped run dir, not
    which agent owns it), so we scan each search_root for a child whose
    name equals ``rid`` (any depth ≤ 2). First hit wins.
    """
    if not rid:
        return None
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            # Direct match: .cage_runs/<agent>/<rid>/
            for agent_dir in root.iterdir():
                if not agent_dir.is_dir():
                    continue
                candidate = agent_dir / rid
                if candidate.is_dir():
                    return candidate
        except OSError:
            continue
    return None


def iter_known_run_ids(search_roots: list[Path]) -> Iterator[tuple[str, Path]]:
    """Yield (run_id, run_dir) for every ``.cage_runs/<agent>/<run_id>/`` found.

    Used by ``cage gc`` to cross-reference docker resource owners against
    on-disk runs. A run with **no** ``.cage_runs/`` directory is an
    orphan and reclaimable; a run with a stalled directory is dead and
    reclaimable; a run whose directory is actively ticking is alive.
    """
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for agent_dir in root.iterdir():
                if not agent_dir.is_dir():
                    continue
                try:
                    for run_dir in agent_dir.iterdir():
                        if not run_dir.is_dir():
                            continue
                        rid = run_dir.name
                        if rid in seen:
                            continue
                        seen.add(rid)
                        yield rid, run_dir
                except OSError:
                    continue
        except OSError:
            continue
