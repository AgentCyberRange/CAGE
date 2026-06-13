"""Unit tests for the shared trial-status classifier and counter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cage.contracts.trial_status import (
    TrialCounts,
    classify_trial_status,
    count_trials,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", "completed"),
        ("scored", "completed"),
        ("not_scored", "completed"),
        ("failed", "failed"),
        ("interrupted", "interrupted"),
        ("cancelled", "interrupted"),
    ],
)
def test_status_set_takes_priority(status, expected):
    assert classify_trial_status(status=status) == expected


def test_case_and_whitespace_are_normalized():
    assert classify_trial_status(status="  COMPLETED  ") == "completed"
    assert classify_trial_status(status="Cancelled") == "interrupted"


def test_reason_used_when_status_absent():
    assert classify_trial_status(termination_reason="completed") == "completed"
    assert classify_trial_status(termination_reason="interrupted") == "interrupted"
    assert classify_trial_status(termination_reason="cancelled") == "interrupted"
    assert classify_trial_status(termination_reason="failed") == "failed"


def test_status_outranks_reason_and_error():
    # An explicit terminal status wins even if other signals disagree.
    assert (
        classify_trial_status(
            status="completed", error="boom", termination_reason="failed"
        )
        == "completed"
    )


def test_error_truthy_falls_to_failed():
    assert classify_trial_status(error="boom") == "failed"
    assert classify_trial_status(error=None) == "running"


def test_unknown_signals_fall_back_to_running():
    assert classify_trial_status() == "running"
    assert classify_trial_status(status="queued") == "running"


def test_default_override_for_finished_callers():
    # A post-run summary treats "no terminal signal" as a completion.
    assert classify_trial_status(default="completed") == "completed"
    assert classify_trial_status(status="queued", default="completed") == "completed"
    # An explicit terminal signal still wins over the default.
    assert classify_trial_status(error="boom", default="completed") == "failed"
    assert (
        classify_trial_status(termination_reason="interrupted", default="completed")
        == "interrupted"
    )


def test_count_trials_with_dicts():
    items = [
        {"status": "completed"},
        {"status": "scored"},
        {"status": "failed"},
        {"status": "cancelled"},
        {"status": "queued"},  # → running remainder
    ]
    counts = count_trials(items)
    assert counts == TrialCounts(
        total=5, completed=2, failed=1, interrupted=1, running=1
    )
    assert counts.pending() == 1


def test_count_trials_with_objects_and_error_field():
    @dataclass
    class _Result:
        status: str | None = None
        error: str | None = None
        termination_reason: str | None = None

    items = [
        _Result(status="completed"),
        _Result(error="boom"),  # no status → failed via error
        _Result(termination_reason="interrupted"),
    ]
    counts = count_trials(items)
    assert counts == TrialCounts(
        total=3, completed=1, failed=1, interrupted=1, running=0
    )


def test_count_trials_custom_keys():
    items = [{"st": "completed"}, {"st": "failed"}]
    counts = count_trials(items, status_key="st")
    assert (counts.completed, counts.failed) == (1, 1)


def test_running_never_negative():
    # Pathological double-classification cannot drive running below zero.
    counts = count_trials([{"status": "completed"}])
    assert counts.running == 0
