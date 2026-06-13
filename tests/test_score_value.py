"""Unit tests for the shared numeric score extractor."""

from __future__ import annotations

import pytest

from cage.contracts.scoring import Score, extract_numeric_score_value


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (0.75, 0.75),
        (1, 1.0),
        ({"value": 0.5, "answer": "ok"}, 0.5),
        (Score(value=0.25), 0.25),
        (True, None),  # bool is a verdict flag, not a magnitude
        (False, None),
        ({"value": True}, None),
        ({"answer": "ok"}, None),  # no value key
        ("0.75", None),  # numeric strings are not coerced here
        (None, None),
        ({}, None),
    ],
)
def test_extract_numeric_score_value(payload, expected):
    assert extract_numeric_score_value(payload) == expected
