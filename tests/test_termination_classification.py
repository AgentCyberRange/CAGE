"""Tests for fine-grained termination_reason classification.

Goal: every failure mode the user might observe lands on a stable,
machine-readable reason — not the catch-all ``agent_exit_nonzero``. That
matters because resume's default retry set, the web UI's color coding,
and dashboard filters all key off ``termination_reason``.
"""

import json
from pathlib import Path

import pytest

from cage.experiment.engine.termination import (
    TerminationReason,
    TerminationStatus,
    classify_trial_termination,
    classify_upstream_error,
    max_cost_reached_termination,
    max_input_tokens_reached_termination,
    max_output_tokens_reached_termination,
    max_rounds_reached_termination,
)

# ── Tier 1: error.type substring (provider-portable) ───────────────────


@pytest.mark.parametrize("error_type, expected", [
    # Codex
    ("usage_limit_reached", TerminationReason.MODEL_QUOTA_EXHAUSTED),
    # OpenAI
    ("insufficient_quota", TerminationReason.MODEL_QUOTA_EXHAUSTED),
    ("rate_limit_exceeded", TerminationReason.MODEL_RATE_LIMITED),
    ("context_length_exceeded", TerminationReason.MODEL_CONTEXT_OVERFLOW),
    ("authentication_error", TerminationReason.MODEL_AUTH_ERROR),
    ("invalid_api_key", TerminationReason.MODEL_AUTH_ERROR),
    ("permission_denied", TerminationReason.MODEL_AUTH_ERROR),
    # Anthropic
    ("rate_limit_error", TerminationReason.MODEL_RATE_LIMITED),
    ("overloaded_error", TerminationReason.MODEL_BAD_GATEWAY),
    # Cage proxy
    ("bad_response_status_code", TerminationReason.MODEL_BAD_GATEWAY),
    ("upstream_error", TerminationReason.MODEL_BAD_GATEWAY),
    # Variants
    ("billing_hard_limit_reached", TerminationReason.MODEL_QUOTA_EXHAUSTED),
    ("string_too_long", TerminationReason.MODEL_CONTEXT_OVERFLOW),
    ("deadline_exceeded", TerminationReason.MODEL_TIMEOUT),
])
def test_classify_by_error_type(error_type: str, expected: TerminationReason) -> None:
    assert classify_upstream_error(error_type=error_type) is expected


# ── Tier 1: HTTP status code disambiguation ────────────────────────────


def test_classify_429_disambiguates_quota_vs_rate_limit() -> None:
    # 429 with quota-flavored message → quota
    assert classify_upstream_error(
        http_status=429,
        error_message="The usage limit has been reached. Try again later.",
    ) is TerminationReason.MODEL_QUOTA_EXHAUSTED
    assert classify_upstream_error(
        http_status=429,
        error_message="monthly limit exceeded",
    ) is TerminationReason.MODEL_QUOTA_EXHAUSTED
    # 429 without quota wording defaults to rate-limit (safer)
    assert classify_upstream_error(http_status=429) is TerminationReason.MODEL_RATE_LIMITED
    assert classify_upstream_error(
        http_status=429, error_message="too many requests"
    ) is TerminationReason.MODEL_RATE_LIMITED


@pytest.mark.parametrize("http_status, expected", [
    (401, TerminationReason.MODEL_AUTH_ERROR),
    (403, TerminationReason.MODEL_AUTH_ERROR),
    (408, TerminationReason.MODEL_TIMEOUT),
    (413, TerminationReason.MODEL_CONTEXT_OVERFLOW),
    (500, TerminationReason.MODEL_BAD_GATEWAY),
    (502, TerminationReason.MODEL_BAD_GATEWAY),
    (503, TerminationReason.MODEL_BAD_GATEWAY),
    (504, TerminationReason.MODEL_TIMEOUT),
    (599, TerminationReason.MODEL_BAD_GATEWAY),  # any 5xx
    (404, TerminationReason.MODEL_ERROR),         # other 4xx → generic
])
def test_classify_by_http_status(http_status: int, expected: TerminationReason) -> None:
    assert classify_upstream_error(http_status=http_status) is expected


# ── Tier 1 wins over Tier 2 ────────────────────────────────────────────


def test_error_type_beats_text_keyword() -> None:
    """If the upstream sets error.type, we trust that even when free text
    contains conflicting keywords. Reduces false-positive risk."""
    # Text says "rate limit" but error.type says quota → trust error.type
    result = classify_upstream_error(
        text="Hit the rate limit per minute",
        error_type="usage_limit_reached",
    )
    assert result is TerminationReason.MODEL_QUOTA_EXHAUSTED


# ── Tier 2: text fallback (conservative) ───────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("You've hit your usage limit. Try again later.", TerminationReason.MODEL_QUOTA_EXHAUSTED),
    ("Server returned: too many requests", TerminationReason.MODEL_RATE_LIMITED),
    ("API Error: bad gateway", TerminationReason.MODEL_BAD_GATEWAY),
    ("Name or service not known", TerminationReason.MODEL_BAD_GATEWAY),
    ("connection refused on upstream", TerminationReason.MODEL_BAD_GATEWAY),
    ("Maximum context length exceeded", TerminationReason.MODEL_CONTEXT_OVERFLOW),
    ("Read operation timed out after 600s", TerminationReason.MODEL_TIMEOUT),
])
def test_text_fallback_matches_only_safe_phrases(
    text: str, expected: TerminationReason
) -> None:
    assert classify_upstream_error(text=text) is expected


def test_classify_upstream_error_returns_none_on_unmatched() -> None:
    """No signals at all → None. Avoids false-positive on innocent text
    containing keyword-like substrings."""
    assert classify_upstream_error(text="just some agent stdout") is None
    assert classify_upstream_error(text="") is None
    # Innocent text containing partial keyword still returns None — Tier 2
    # patterns are full phrases, not bare substrings.
    assert classify_upstream_error(text="user mentioned context") is None
    assert classify_upstream_error(text="describes a rate") is None


# ── classify_trial_termination: full path ──────────────────────────────


def _write_proxy_jsonl(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "proxy.jsonl"
    p.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )
    return p


def test_exit_137_classified_as_oom(tmp_path: Path) -> None:
    info = classify_trial_termination(
        exit_code=137, timed_out=False, terminated_by_limit=False,
        output="The source block has not cleared...",
    )
    assert info.reason is TerminationReason.OOM_KILLED
    assert info.status is TerminationStatus.FAILED


def test_codex_quota_via_proxy_log(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [
        {"request_id": "req-1", "status": "success"},
        {
            "request_id": "req-2", "status": "error",
            "error_text": 'HTTP 429: {"error":{"message":"The usage limit has been reached","type":"usage_limit_reached"}}',
            "upstream_status_code": None,
        },
    ])
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        output="You've hit your usage limit. Try again later.",
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.MODEL_QUOTA_EXHAUSTED
    # Detail should include the upstream error text we recovered from the log
    assert "usage_limit_reached" in info.detail.lower() or "usage limit" in info.detail.lower()


def test_no_proxy_log_treats_failure_as_local(tmp_path: Path) -> None:
    """Per policy: if proxy.jsonl has no upstream error entry, the failure
    is local (agent_exit_nonzero) — even if the agent's stdout mentions
    phrases like "usage limit". This avoids false-positives from agent
    reasoning that incidentally contains error keywords (e.g. while
    debugging a target's web service)."""
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        output="You've hit your usage limit. Try again later.",
        proxy_jsonl_path=tmp_path / "does_not_exist.jsonl",
    )
    assert info.reason is TerminationReason.AGENT_EXIT_NONZERO


def test_bad_gateway_via_proxy_log(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "error", "upstream_status_code": 502,
         "error_text": "HTTP 502: upstream returned bad gateway"},
    ])
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        output="API Error: 502 Bad Gateway",
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.MODEL_BAD_GATEWAY


def test_rate_limit_distinct_from_quota(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "error", "upstream_status_code": 429,
         "error_text": 'HTTP 429: {"error":{"code":"rate_limit_exceeded"}}'},
    ])
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.MODEL_RATE_LIMITED


def test_auth_error_via_proxy_log(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "error", "upstream_status_code": 401,
         "error_text": "HTTP 401: invalid_api_key"},
    ])
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.MODEL_AUTH_ERROR


def test_context_overflow_via_proxy_log(tmp_path: Path) -> None:
    # Realistic OpenAI error body: error.type carries the canonical signal.
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "error",
         "error_text": 'HTTP 400: {"error":{"type":"context_length_exceeded","message":"context too large"}}'},
    ])
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.MODEL_CONTEXT_OVERFLOW


def test_unclassified_upstream_error_falls_back_to_model_error(tmp_path: Path) -> None:
    """Upstream HTTP error that doesn't match any specific pattern still
    counts as an infra-side failure — not ``agent_exit_nonzero`` — so resume
    will re-run it by default."""
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "error", "upstream_status_code": 418,
         "error_text": "HTTP 418: I'm a teapot"},
    ])
    info = classify_trial_termination(
        exit_code=1, timed_out=False, terminated_by_limit=False,
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.MODEL_ERROR


def test_no_upstream_error_and_no_specific_signal_stays_agent_exit_nonzero(tmp_path: Path) -> None:
    """If exit != 0 with no proxy error and no recognizable output pattern,
    we don't fabricate a category — keep the truthful catch-all."""
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "success"},  # successful upstream calls only
    ])
    info = classify_trial_termination(
        exit_code=2, timed_out=False, terminated_by_limit=False,
        output="some unrelated agent failure",
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.AGENT_EXIT_NONZERO


def test_recovered_upstream_error_does_not_override_later_success(tmp_path: Path) -> None:
    """Only the terminal proxy entry should drive model-error classification."""
    proxy = _write_proxy_jsonl(tmp_path, [
        {"request_id": "req-1", "status": "success"},
        {
            "request_id": "req-2",
            "status": "error",
            "upstream_status_code": 503,
            "error_text": "HTTP 503: upstream unavailable",
        },
        {"request_id": "req-3", "status": "success"},
    ])
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        output="some unrelated agent failure",
        proxy_jsonl_path=proxy,
        max_rounds=3,
    )

    assert info.reason is TerminationReason.AGENT_EXIT_NONZERO


def test_completed_unchanged_with_proxy_log(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [{"status": "success"}])
    info = classify_trial_termination(
        exit_code=0, timed_out=False, terminated_by_limit=False,
        output="done",
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.COMPLETED


def test_execution_timeout_wins_over_proxy_error(tmp_path: Path) -> None:
    """Wall-clock timeout from orchestrator is the truth — even if the proxy
    log also recorded an upstream error along the way."""
    proxy = _write_proxy_jsonl(tmp_path, [
        {"status": "error", "error_text": "HTTP 502 bad gateway"},
    ])
    info = classify_trial_termination(
        exit_code=-1, timed_out=True, terminated_by_limit=False,
        timeout_seconds=300,
        proxy_jsonl_path=proxy,
    )
    assert info.reason is TerminationReason.EXECUTION_TIMEOUT


def test_max_rounds_ignores_failed_upstream_records(tmp_path: Path) -> None:
    """Failed upstream calls should not spend the agent's round budget."""
    proxy = _write_proxy_jsonl(tmp_path, [
        {"request_id": "req-1", "status": "success"},
        {
            "request_id": "req-2",
            "status": "error",
            "upstream_status_code": 503,
            "error_text": "HTTP 503: upstream unavailable",
        },
    ])
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        proxy_jsonl_path=proxy,
        max_rounds=2,
    )

    assert info.reason is TerminationReason.MODEL_BAD_GATEWAY


def test_max_rounds_ignores_compact_rewritten_records(tmp_path: Path) -> None:
    """Context compaction calls are bookkeeping, not budgeted agent rounds."""
    proxy = _write_proxy_jsonl(tmp_path, [
        {
            "request_id": "req-1",
            "status": "success",
            "openai_request": {"_proxy_compact_rewritten": True},
        },
        {"request_id": "req-2", "status": "success"},
        {
            "request_id": "req-3",
            "status": "error",
            "upstream_status_code": 503,
            "error_text": "HTTP 503: upstream unavailable",
        },
    ])
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        proxy_jsonl_path=proxy,
        max_rounds=2,
    )

    assert info.reason is TerminationReason.MODEL_BAD_GATEWAY


def test_max_rounds_counts_successful_non_compact_rounds(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [
        {"request_id": "req-1", "status": "success"},
        {
            "request_id": "req-2",
            "status": "error",
            "upstream_status_code": 503,
            "error_text": "HTTP 503: upstream unavailable",
        },
        {"request_id": "req-3", "status": "success"},
    ])
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        proxy_jsonl_path=proxy,
        max_rounds=2,
    )

    assert info.reason is TerminationReason.MAX_ROUNDS_REACHED


def test_max_input_tokens_budget_wins_over_terminal_proxy_error(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [
        {
            "request_id": "req-1",
            "status": "success",
            "upstream_response": {
                "usage": {"prompt_tokens": 101, "completion_tokens": 7},
            },
        },
        {
            "request_id": "req-2",
            "status": "error",
            "upstream_status_code": 429,
            "error_text": (
                'HTTP 429: {"error":{"type":"rate_limit_error",'
                '"message":"budget reached"}}'
            ),
        },
    ])
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        proxy_jsonl_path=proxy,
        max_input_tokens=100,
    )

    assert info.reason is TerminationReason.MAX_INPUT_TOKENS_REACHED
    assert info.status is TerminationStatus.COMPLETED
    assert "101/100" in info.detail


def test_max_output_tokens_budget_detects_progress_snapshot(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [{"request_id": "req-1", "status": "success"}])
    (tmp_path / "progress.json").write_text(
        json.dumps({"tokens_in": 10, "tokens_out": 55, "cost_usd": 0.0}),
        encoding="utf-8",
    )
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        proxy_jsonl_path=proxy,
        max_output_tokens=50,
    )

    assert info.reason is TerminationReason.MAX_OUTPUT_TOKENS_REACHED
    assert info.status is TerminationStatus.COMPLETED
    assert "55/50" in info.detail


def test_max_cost_budget_detects_progress_snapshot(tmp_path: Path) -> None:
    proxy = _write_proxy_jsonl(tmp_path, [{"request_id": "req-1", "status": "success"}])
    (tmp_path / "progress.json").write_text(
        json.dumps({"tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0123}),
        encoding="utf-8",
    )
    info = classify_trial_termination(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        proxy_jsonl_path=proxy,
        max_cost=0.01,
    )

    assert info.reason is TerminationReason.MAX_COST_REACHED
    assert info.status is TerminationStatus.COMPLETED
    assert "$0.0123/$0.0100" in info.detail


# ── budget termination helpers ─────────────────────────────────────────


def test_max_rounds_reached_termination() -> None:
    info = max_rounds_reached_termination("Hit 100 turns")
    assert info.reason is TerminationReason.MAX_ROUNDS_REACHED
    assert info.status is TerminationStatus.COMPLETED
    meta = info.to_metadata()
    assert meta["status"] == "completed"
    assert meta["termination_reason"] == "max_rounds_reached"


@pytest.mark.parametrize(
    "factory, expected",
    [
        (max_input_tokens_reached_termination, "max_input_tokens_reached"),
        (max_output_tokens_reached_termination, "max_output_tokens_reached"),
        (max_cost_reached_termination, "max_cost_reached"),
    ],
)
def test_budget_reached_termination_helpers(factory, expected: str) -> None:
    info = factory("budget hit")
    assert info.status is TerminationStatus.COMPLETED
    assert info.reason.value == expected
    meta = info.to_metadata()
    assert meta["status"] == "completed"
    assert meta["termination_reason"] == expected


# ── Resume default retry set covers all unexpected reasons ─────────────


def test_default_resume_retry_covers_all_unexpected_reasons() -> None:
    """Anything in the 'unexpected' group of TerminationReason should be in
    the default retry set, and nothing from 'expected' should be."""
    from cage.experiment.engine.resume import _DEFAULT_RESUME_RETRY_REASONS

    expected_replayed = {
        TerminationReason.COMPLETED.value,
        TerminationReason.MAX_ROUNDS_REACHED.value,
        TerminationReason.MAX_INPUT_TOKENS_REACHED.value,
        TerminationReason.MAX_OUTPUT_TOKENS_REACHED.value,
        TerminationReason.MAX_COST_REACHED.value,
        TerminationReason.EXECUTION_TIMEOUT.value,
        TerminationReason.TOOL_LIMIT.value,
        TerminationReason.AGENT_EXIT_NONZERO.value,
    }
    expected_rerun = {
        TerminationReason.MODEL_QUOTA_EXHAUSTED.value,
        TerminationReason.MODEL_RATE_LIMITED.value,
        TerminationReason.MODEL_BAD_GATEWAY.value,
        TerminationReason.MODEL_AUTH_ERROR.value,
        TerminationReason.MODEL_CONTEXT_OVERFLOW.value,
        TerminationReason.MODEL_TIMEOUT.value,
        TerminationReason.MODEL_ERROR.value,
        TerminationReason.OOM_KILLED.value,
        TerminationReason.TARGET_UNAVAILABLE.value,
        TerminationReason.TRIAL_ERROR.value,
        TerminationReason.CANCELLED_BEFORE_START.value,
        TerminationReason.USER_INTERRUPTED.value,
    }
    assert expected_rerun.issubset(_DEFAULT_RESUME_RETRY_REASONS)
    assert _DEFAULT_RESUME_RETRY_REASONS.isdisjoint(expected_replayed)
