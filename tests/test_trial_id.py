"""Unit tests for the shared trial-id parse/format helpers."""

from __future__ import annotations

import pytest

from cage.experiment.model.trial_id import format_trial_id, parse_trial_id


@pytest.mark.parametrize(
    ("trial_id", "expected"),
    [
        ("range1/pass_2", ("range1", 2)),
        ("range1", ("range1", 1)),  # no suffix → pass 1
        ("range1/pass_bad", ("range1", 1)),  # non-int suffix keeps task, pass 1
        ("range1/pass_0", ("range1", 1)),  # clamped to >= 1
        ("a/b/c/pass_5", ("a/b/c", 5)),  # canonical 3-part id splits on last /pass_
        ("", ("", 1)),
    ],
)
def test_parse_trial_id(trial_id, expected):
    assert parse_trial_id(trial_id) == expected


def test_format_trial_id():
    assert format_trial_id("agent:model:stateless", "range1", 2) == (
        "agent:model:stateless/range1/pass_2"
    )
    assert format_trial_id("s", "t") == "s/t/pass_1"  # default pass


def test_parse_then_format_roundtrip_for_runtime_ids():
    task, pass_index = parse_trial_id("range1/pass_3")
    # A runtime id has no subject; formatting with an empty subject still
    # reconstructs the task/pass tail.
    assert format_trial_id("", task, pass_index).lstrip("/") == "range1/pass_3"
