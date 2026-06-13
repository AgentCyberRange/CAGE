"""Unit tests for the shared duration h/m/s split."""

from __future__ import annotations

import pytest

from cage.contracts.duration import split_duration_hms


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, (0, 0, 0)),
        (45, (0, 0, 45)),
        (83, (0, 1, 23)),
        (3600, (1, 0, 0)),
        (3723, (1, 2, 3)),
        (7322, (2, 2, 2)),
        (-5, (0, 0, 0)),  # clamped
    ],
)
def test_split_duration_hms(seconds, expected):
    assert split_duration_hms(seconds) == expected


def test_minutes_and_seconds_are_bounded():
    hours, minutes, secs = split_duration_hms(3600 * 3 + 59 * 60 + 59)
    assert (hours, minutes, secs) == (3, 59, 59)
