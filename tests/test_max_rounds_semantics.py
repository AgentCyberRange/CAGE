"""Round-budget config vocabulary + the run termination-condition guard.

Config space (per level) -> resolved engine value (-1 unlimited, 0 no rounds, N):
  absent/null/"unlimited" -> unlimited ;  -1/"benchmark" -> benchmark default ;
  N -> N. Precedence: agent > runtime > sample. A run whose budget is unlimited
  must have another finite stop condition (timeout / cost / token cap).
"""

from __future__ import annotations

import pytest

from cage.contracts.execution import (
    classify_max_rounds,
    max_rounds_config_label,
    normalize_max_rounds_config,
    resolve_max_rounds,
    round_budget_is_unbounded,
    run_lacks_termination_condition,
)


# --------------------------------------------------------------------------- #
# resolve_max_rounds(agent, runtime, sample)  ->  -1 unlimited / 0 / N
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "agent, runtime, sample, expected",
    [
        # Unspecified runtime ⇒ unlimited (the new default), overriding the sample.
        (None, "unlimited", 150, -1),
        (None, None, 150, -1),
        # -1 ⇒ benchmark per-sample default (unlimited if the benchmark has none).
        (None, -1, 100, 100),
        (None, -1, None, -1),
        # Explicit counts win at their level.
        (None, 50, 150, 50),
        (30, "unlimited", 100, 30),
        (None, 0, 150, 0),  # 0 = no rounds
        # Agent -1 falls through to the runtime value (precedence preserved).
        (-1, 5, 150, 5),
        (-1, 0, 150, 0),
        (-1, -1, 150, 150),  # both defer ⇒ benchmark default
        # String forms.
        (None, "unlimited", None, -1),
        (None, "benchmark", 100, 100),
    ],
)
def test_resolve_max_rounds(agent, runtime, sample, expected):
    assert resolve_max_rounds(agent, runtime, sample) == expected


def test_classify_and_normalize():
    assert classify_max_rounds(None) == ("inherit", None)
    assert classify_max_rounds("unlimited") == ("unlimited", None)
    assert classify_max_rounds(-1) == ("defer", None)
    assert classify_max_rounds(7) == ("count", 7)
    assert normalize_max_rounds_config(None, default="unlimited") == "unlimited"
    assert normalize_max_rounds_config(-1) == -1
    assert normalize_max_rounds_config("5") == 5
    assert normalize_max_rounds_config(None) is None  # agent absent ⇒ inherit


def test_labels():
    assert max_rounds_config_label("unlimited") == "unlimited"
    assert max_rounds_config_label(-1) == "benchmark default"
    assert max_rounds_config_label(42) == "42"
    assert max_rounds_config_label(None) == "inherit"


# --------------------------------------------------------------------------- #
# round_budget_is_unbounded + the termination guard
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "agent, runtime, unbounded",
    [
        (None, "unlimited", True),
        (None, None, True),
        (None, -1, False),   # benchmark default ⇒ assume bounded
        (None, 50, False),
        (-1, 5, False),
        (-1, "unlimited", True),
    ],
)
def test_round_budget_is_unbounded(agent, runtime, unbounded):
    assert round_budget_is_unbounded(agent, runtime) is unbounded


def test_run_lacks_termination_condition():
    bad = run_lacks_termination_condition
    # Unlimited rounds and no other cap ⇒ no termination condition.
    assert bad(agent_round_budgets=[None], execution_max_rounds="unlimited") is True
    # Any other finite cap rescues it.
    assert bad(agent_round_budgets=[None], execution_max_rounds="unlimited", timeout=3600) is False
    assert bad(agent_round_budgets=[None], execution_max_rounds="unlimited", max_cost=5.0) is False
    assert bad(
        agent_round_budgets=[None], execution_max_rounds="unlimited", max_input_tokens=1000
    ) is False
    # A finite / benchmark-default round budget is itself a termination condition.
    assert bad(agent_round_budgets=[None], execution_max_rounds=-1) is False
    assert bad(agent_round_budgets=[None], execution_max_rounds=100) is False
    # One unbounded agent (no other cap) is enough to flag the run.
    assert bad(agent_round_budgets=["unlimited", 100], execution_max_rounds=-1) is True
