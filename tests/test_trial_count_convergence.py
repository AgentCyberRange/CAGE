"""Convergence guard for the shared trial-status classification (P0).

Before the contracts.trial_status extraction, the canonical record, the
scoring summary and the resume summary each re-rolled "how many completed"
with a different definition — resume in particular counted ``scored`` /
``not_scored`` trials as zero. These tests pin the three paths that now share
``classify_trial_status`` / ``count_trials`` to the SAME completed/failed
numbers for one fixed mixed-status trial set, so the definitions cannot
silently drift apart again.
"""

from __future__ import annotations

from cage.contracts.trial_status import count_trials
from cage.experiment.engine.resume import _summary_from_trial_infos
from cage.experiment.model import TrialResult
from cage.sandbox.exec import Timing
from cage.scoring.lifecycle import _build_summary

# One fixed set covering every bucket and the easy-to-miss scoring states.
_MIXED_STATUSES = [
    "completed",
    "scored",      # post-completion scoring state → completed
    "not_scored",  # post-completion scoring state → completed
    "failed",
    "interrupted",
    "cancelled",   # interrupted bucket
]

# Canonical truth for this set.
_EXPECTED_COMPLETED = 3
_EXPECTED_FAILED = 1
_EXPECTED_INTERRUPTED = 2  # interrupted + cancelled


def _trial_result(status: str) -> TrialResult:
    return TrialResult(
        trial_id=f"task/{status}/pass_1",
        trial_index=0,
        trial_type="task",
        sample_id=status,
        output="",
        exit_code=0,
        timing=Timing(started_at_ms=0, ended_at_ms=1, duration_ms=1),
        metadata={"status": status},
    )


def test_canonical_count_trials_matches_expected():
    counts = count_trials({"status": s} for s in _MIXED_STATUSES)
    assert (counts.completed, counts.failed, counts.interrupted) == (
        _EXPECTED_COMPLETED,
        _EXPECTED_FAILED,
        _EXPECTED_INTERRUPTED,
    )


def test_scoring_summary_agrees_on_completed_failed():
    summary = _build_summary([_trial_result(s) for s in _MIXED_STATUSES])
    assert summary["completed"] == _EXPECTED_COMPLETED
    assert summary["failed"] == _EXPECTED_FAILED


def test_resume_summary_agrees_and_counts_scored_states():
    summary = _summary_from_trial_infos(
        [{"status": s} for s in _MIXED_STATUSES], fallback={}
    )
    assert summary["completed"] == _EXPECTED_COMPLETED  # scored/not_scored included
    assert summary["failed"] == _EXPECTED_FAILED
    # Resume keeps interrupted and cancelled split; together they equal the
    # canonical interrupted bucket.
    assert summary["interrupted"] + summary["cancelled"] == _EXPECTED_INTERRUPTED


def test_all_three_paths_report_identical_completed_failed():
    canonical = count_trials({"status": s} for s in _MIXED_STATUSES)
    scoring = _build_summary([_trial_result(s) for s in _MIXED_STATUSES])
    resume = _summary_from_trial_infos(
        [{"status": s} for s in _MIXED_STATUSES], fallback={}
    )
    completed = {canonical.completed, scoring["completed"], resume["completed"]}
    failed = {canonical.failed, scoring["failed"], resume["failed"]}
    assert completed == {_EXPECTED_COMPLETED}
    assert failed == {_EXPECTED_FAILED}
