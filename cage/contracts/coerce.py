"""Scalar coercion primitives shared by the YAML- and display-facing surfaces.

Three deliberate families — pick by the semantics a call site needs, never
redefine these inline (inconsistent private copies of these helpers are how
the spec/sections parse drift happened):

- ``optional_int`` / ``optional_float`` — blank-tolerant, strict. ``None`` and
  ``""`` mean "not configured"; anything else must parse or the error
  propagates. **Zero and negative values are preserved** — callers that treat
  ``0`` as meaningful use these.
- ``positive_int_or_none`` / ``positive_float_or_none`` — blank-tolerant,
  strict, and **fold ``<= 0`` to ``None``** for knobs where zero/negative mean
  "unlimited" (token budgets, cost caps).
- ``int_or_zero`` — lenient display/accumulation coercion: garbage becomes
  ``0`` instead of an error. For counters and UI rollups, never for config.

The JSON-deserialization helpers in :mod:`cage.artifacts.records` are a
separate family on purpose: they raise on every malformed value (including
blanks) because canonical artifacts have a schema, while these primitives
parse hand-written YAML and ad-hoc telemetry.
"""

from __future__ import annotations

from typing import Any


def optional_int(value: Any) -> int | None:
    """Parse an optional integer, preserving zero/negative values."""

    if value in (None, ""):
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    """Parse an optional float, preserving zero/negative values."""

    if value in (None, ""):
        return None
    return float(value)


def positive_int_or_none(value: Any) -> int | None:
    """Parse an optional integer knob where ``<= 0`` means "unlimited"."""

    if value in (None, ""):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def positive_float_or_none(value: Any) -> float | None:
    """Parse an optional float knob where ``<= 0`` means "unlimited"."""

    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def int_or_zero(value: Any) -> int:
    """Coerce counter-ish telemetry values, treating garbage as zero."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
