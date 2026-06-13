"""Execution vocabulary: trial timing and the round-budget convention.

Dependency-free value types and pure functions. They live in ``contracts``
because the experiment data model (``Trial``) records timing and must stay
importable without the Docker/process substrate; ``cage.sandbox.exec``
re-exports ``Timing`` next to the ``ExecResult`` it accompanies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Timing:
    """Wall-clock timing captured for a trial or command execution."""

    started_at_ms: int = 0
    ended_at_ms: int = 0
    duration_ms: int = 0


# --- round budget -----------------------------------------------------------
#
# The CONFIG value space (what a user writes for ``max_rounds`` at the agent or
# runtime level) maps to a RESOLVED engine value (``-1`` == unlimited, ``0`` ==
# no rounds, ``N`` == exactly N rounds) like so:
#
#   absent / null / "unlimited"   -> UNLIMITED  (no round cap; the default)
#   -1 / "benchmark" / "default"  -> DEFER       (use the benchmark sample's
#                                                  declared default; unlimited if
#                                                  the benchmark declares none)
#   N >= 0                         -> exactly N
#
# Precedence is the same everywhere — the engine enforces it, preflight predicts
# it, the run banner displays it: an explicit per-agent value wins, then the
# runtime value, then (for DEFER) the benchmark sample default. When NEITHER
# agent nor runtime constrains the budget it is UNLIMITED — so a run with no
# round cap must be bounded by another termination condition (timeout / cost /
# token caps); see ``round_budget_is_unbounded`` and the run's termination check.

_UNLIMITED_TOKENS = frozenset({"unlimited", "inf", "infinite", "none", "null", "off"})
_DEFER_TOKENS = frozenset({"default", "benchmark", "defer", "sample"})


def classify_max_rounds(value: Any) -> tuple[str, int | None]:
    """Classify a config-level budget into (kind, count).

    kind ∈ {"inherit", "unlimited", "defer", "count"}; ``count`` is set only for
    "count". "inherit" means "say nothing, fall through to the next level".
    """

    if value is None:
        return ("inherit", None)
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "":
            return ("inherit", None)
        if token in _UNLIMITED_TOKENS:
            return ("unlimited", None)
        if token in _DEFER_TOKENS:
            return ("defer", None)
        try:
            value = int(token)
        except ValueError:
            return ("inherit", None)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return ("inherit", None)
    if parsed == -1:
        return ("defer", None)
    if parsed < -1:
        return ("inherit", None)
    return ("count", parsed)


def normalize_max_rounds_config(raw: Any, *, default: Any = None) -> int | str | None:
    """Canonicalize a raw config value to one of: ``"unlimited"`` | ``-1`` | N | None.

    ``default`` is substituted when ``raw`` is absent/None — pass ``"unlimited"``
    at the runtime level (absent ⇒ unlimited) and leave None at the agent level
    (absent ⇒ inherit the runtime value).
    """

    kind, count = classify_max_rounds(raw if raw is not None else default)
    if kind == "unlimited":
        return "unlimited"
    if kind == "defer":
        return -1
    if kind == "count":
        return count
    return None


def max_rounds_config_label(value: Any) -> str:
    """Human-readable label for a config-level round budget (for banners/dry-run)."""

    kind, count = classify_max_rounds(value)
    if kind == "unlimited":
        return "unlimited"
    if kind == "defer":
        return "benchmark default"
    if kind == "count":
        return str(count)
    return "inherit"  # None at the agent level ⇒ inherits the runtime budget


def sample_max_rounds(value: Any) -> int | None:
    """A benchmark sample's fallback round budget, when one is declared."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def round_budget_is_unbounded(agent_value: Any, execution_value: Any) -> bool:
    """Whether the agent+runtime budget imposes NO finite round cap.

    A DEFER (-1 / benchmark default) counts as bounded — the benchmark declares
    the cap. Only an explicit/implicit UNLIMITED is unbounded. Used by the
    termination-condition validation (a fully unbounded budget needs another
    stop condition).
    """

    saw_defer = False
    for level in (agent_value, execution_value):
        kind, _count = classify_max_rounds(level)
        if kind == "unlimited":
            return True
        if kind == "count":
            return False
        if kind == "defer":
            saw_defer = True  # defer to the benchmark default ⇒ assume bounded
        # "defer"/"inherit" -> consult the next (lower-precedence) level
    return not saw_defer  # a seen defer ⇒ bounded by benchmark; else unlimited


def run_lacks_termination_condition(
    *,
    agent_round_budgets: Any,
    execution_max_rounds: Any,
    timeout: float | None = 0.0,
    max_cost: float | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
) -> bool:
    """Whether a run has NO finite termination condition (so it could never stop).

    A run is bounded if any of these holds: a per-trial ``timeout`` > 0, a
    ``max_cost`` cap, a ``max_input_tokens``/``max_output_tokens`` cap, or a
    finite round budget for every agent (a DEFER/-1 counts as finite — the
    benchmark declares the cap). Returns True only when the round budget is
    unbounded for some agent AND none of the other caps is set.
    """

    if (timeout and timeout > 0) \
            or max_cost is not None \
            or max_input_tokens is not None \
            or max_output_tokens is not None:
        return False
    budgets = list(agent_round_budgets) or [None]
    return any(
        round_budget_is_unbounded(budget, execution_max_rounds)
        for budget in budgets
    )


def resolve_max_rounds(
    agent_value: Any,
    execution_value: Any,
    sample_value: Any,
) -> int:
    """Resolve one trial's round budget; ``-1`` == unlimited, ``0`` == no rounds.

    Precedence: a higher level's explicit COUNT/UNLIMITED wins immediately; a
    DEFER (-1) or INHERIT (absent) falls through to the next level. If a DEFER
    was seen and no level gave a concrete budget, use the benchmark sample
    default (unlimited if the benchmark declares none); if nothing was said at
    all, the budget is unlimited.
    """

    saw_defer = False
    for level in (agent_value, execution_value):
        kind, count = classify_max_rounds(level)
        if kind == "unlimited":
            return -1
        if kind == "count":
            return count if count is not None else -1
        if kind == "defer":
            saw_defer = True
        # "defer"/"inherit" -> fall through to the next (lower-precedence) level
    if saw_defer:
        sample_max = sample_max_rounds(sample_value)
        return sample_max if sample_max is not None else -1
    return -1  # nothing constrained the budget ⇒ unlimited
