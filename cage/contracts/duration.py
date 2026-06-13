"""Duration arithmetic — the one place that splits seconds into h/m/s.

Several layers render elapsed time, each with its own *format* (the web tables
use ``"1h 2m 3s"``, the CLI banner uses the zero-padded ``"1h02m03s"``). Those
display choices legitimately differ per layer, so they stay where they are.
What they should NOT each re-derive is the ``divmod(…, 3600)`` / ``divmod(…,
60)`` decomposition. This module owns just that primitive; callers keep their
own formatting on top.

Deliberately minimal: callers whose output is *not* an hours/minutes/seconds
breakdown (single biggest exact unit, an always-seconds value, or a
total-minutes form that never rolls into hours) do their own arithmetic — a
shared h/m/s split would change their output.
"""

from __future__ import annotations


def split_duration_hms(total_seconds: int) -> tuple[int, int, int]:
    """Split a non-negative whole-second duration into ``(hours, minutes, seconds)``.

    Minutes and seconds are in ``0..59``; hours carry the remainder. Negative
    inputs are clamped to zero.
    """

    seconds = max(0, int(total_seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return hours, minutes, secs
