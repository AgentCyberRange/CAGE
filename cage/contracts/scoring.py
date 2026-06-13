"""Score — the dependency-free verdict a ``Scorer`` produces.

This value type lives in ``contracts`` rather than ``scoring`` because the
experiment data model (``Trial``) stores a ``Score`` on every trial and must be
importable without dragging in the scoring runtime. ``Scorer.score()`` returns
it; ``cage.scoring`` re-exports it for the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Score:
    """One named scoring result for a trial or aggregate judgment."""

    value: float
    answer: str = ""
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_numeric_score_value(payload: Any) -> float | None:
    """Pull the numeric scalar out of a score payload, or ``None``.

    Accepts a :class:`Score`, a ``{"value": …}`` mapping, or a bare scalar —
    the three shapes a score can arrive in (live object, on-disk JSON, or an
    already-flattened number). Returns ``float`` for a real number and ``None``
    for anything non-numeric, including ``bool`` (a boolean is a verdict flag,
    not a score magnitude, so it must never be rendered or summed as 1.0).

    Centralizing this keeps every reader — web tables, summaries, exports —
    agreeing on what counts as a numeric score instead of each re-deciding how
    to unwrap ``value`` and whether a bool slips through.
    """

    if isinstance(payload, Score):
        value: Any = payload.value
    elif isinstance(payload, dict):
        value = payload.get("value")
    else:
        value = payload
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
