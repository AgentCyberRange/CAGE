"""Trial termination classification.

Maps a trial's exit conditions onto a stable ``TerminationInfo`` verdict
(status / reason / source). Value types live in :mod:`termination_model`;
proxy-log signals in :mod:`proxy_signals`.
"""
from __future__ import annotations

from cage.contracts.exceptions import (
    CageInterrupt,
    ProxyError,
    TrialError,
    TrialTimeout,
)
from cage.experiment.engine.termination_model import (  # re-exported
    TerminationInfo,
    TerminationReason,
    TerminationSource,
    TerminationStatus,
)
from cage.experiment.engine.proxy_signals import (  # classify_upstream_error re-exported
    _count_proxy_jsonl_budgeted_rounds,
    _last_proxy_error_signal,
    _proxy_budget_usage,
    classify_upstream_error,
)


__all__ = [
    "TerminationInfo",
    "TerminationReason",
    "TerminationSource",
    "TerminationStatus",
    "_token_cost_budget_termination",
    "cancelled_before_start_termination",
    "classify_trial_termination",
    "classify_upstream_error",
    "completed_termination",
    "looks_like_model_timeout",
    "max_cost_reached_termination",
    "max_input_tokens_reached_termination",
    "max_output_tokens_reached_termination",
    "max_rounds_reached_termination",
    "target_unavailable_termination",
    "termination_info_from_exception",
    "user_interrupted_termination",
]


def completed_termination() -> TerminationInfo:
    return TerminationInfo(
        status=TerminationStatus.COMPLETED,
        reason=TerminationReason.COMPLETED,
        detail="Task finished",
        source=TerminationSource.AGENT,
    )


def user_interrupted_termination() -> TerminationInfo:
    return TerminationInfo(
        status=TerminationStatus.INTERRUPTED,
        reason=TerminationReason.USER_INTERRUPTED,
        detail="Stopped by user (Ctrl+C)",
        source=TerminationSource.USER,
    )


def cancelled_before_start_termination() -> TerminationInfo:
    return TerminationInfo(
        status=TerminationStatus.CANCELLED,
        reason=TerminationReason.CANCELLED_BEFORE_START,
        detail="Run interrupted before this trial started",
        source=TerminationSource.USER,
    )


def target_unavailable_termination(detail: str) -> TerminationInfo:
    """The external target stack failed to launch — agent exec was skipped."""
    return TerminationInfo(
        status=TerminationStatus.FAILED,
        reason=TerminationReason.TARGET_UNAVAILABLE,
        detail=detail or "Target failed to launch",
        source=TerminationSource.ORCHESTRATOR,
    )


def max_rounds_reached_termination(detail: str = "") -> TerminationInfo:
    """Agent hit its configured per-trial round budget — natural stop.

    Treated as a *valid* outcome (status=completed) so resume keeps it.

    Two paths emit this:

    1. **Agent-side** — benchmark or per-agent output parser recognizes
       the agent's max-rounds marker (e.g. a sentinel line in stdout)
       and constructs this termination explicitly.

    2. **Classifier-side** — :func:`classify_trial_termination` detects
       it structurally from on-disk artifacts: successful non-compact
       proxy rounds >= max_rounds with a non-zero exit. The in-container
       proxy enforces the budget by 429-rejecting the next request, but
       the rejection short-circuits *before*
       ``ProxyRecorder.record()`` and is therefore not in proxy.jsonl
       itself — the successful non-compact round count is the structural
       signal. Upstream failures and context-compaction calls do not spend
       the agent's round budget.

    The classifier check runs before the upstream-error scan so a stray
    transient 5xx mid-run can't outvote a deterministic budget hit.
    """
    return TerminationInfo(
        status=TerminationStatus.COMPLETED,
        reason=TerminationReason.MAX_ROUNDS_REACHED,
        detail=detail or "Agent reached the configured max-rounds budget",
        source=TerminationSource.AGENT,
    )


def max_input_tokens_reached_termination(detail: str = "") -> TerminationInfo:
    return TerminationInfo(
        status=TerminationStatus.COMPLETED,
        reason=TerminationReason.MAX_INPUT_TOKENS_REACHED,
        detail=detail or "Agent reached the configured input-token budget",
        source=TerminationSource.AGENT,
    )


def max_output_tokens_reached_termination(detail: str = "") -> TerminationInfo:
    return TerminationInfo(
        status=TerminationStatus.COMPLETED,
        reason=TerminationReason.MAX_OUTPUT_TOKENS_REACHED,
        detail=detail or "Agent reached the configured output-token budget",
        source=TerminationSource.AGENT,
    )


def max_cost_reached_termination(detail: str = "") -> TerminationInfo:
    return TerminationInfo(
        status=TerminationStatus.COMPLETED,
        reason=TerminationReason.MAX_COST_REACHED,
        detail=detail or "Agent reached the configured cost budget",
        source=TerminationSource.AGENT,
    )


def _token_cost_budget_termination(
    proxy_jsonl_path: object,
    *,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_cost: float | None = None,
) -> TerminationInfo | None:
    usage = _proxy_budget_usage(proxy_jsonl_path)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cost_usd = float(usage.get("cost_usd", 0.0) or 0.0)

    if max_input_tokens is not None and max_input_tokens > 0 and input_tokens >= max_input_tokens:
        return max_input_tokens_reached_termination(
            f"Hit input-token budget - {input_tokens}/{max_input_tokens} tokens used; "
            "proxy 429-rejected the next call"
        )
    if (
        max_output_tokens is not None
        and max_output_tokens > 0
        and output_tokens >= max_output_tokens
    ):
        return max_output_tokens_reached_termination(
            f"Hit output-token budget - {output_tokens}/{max_output_tokens} tokens used; "
            "proxy 429-rejected the next call"
        )
    if max_cost is not None and max_cost > 0 and cost_usd >= max_cost:
        return max_cost_reached_termination(
            f"Hit cost budget - ${cost_usd:.4f}/${max_cost:.4f} used; "
            "proxy 429-rejected the next call"
        )
    return None


def classify_trial_termination(
    *,
    exit_code: int,
    timed_out: bool,
    terminated_by_limit: bool,
    terminated_by_max_rounds: bool = False,
    error: str = "",
    timeout_seconds: int | float = 0,
    output: str = "",
    exception: BaseException | None = None,
    proxy_jsonl_path: object = None,
    max_rounds: int = 0,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_cost: float | None = None,
    interrupted: bool = False,
) -> TerminationInfo:
    """Classify raw execution signals into stable trial termination metadata.

    Detection order is intentional: orchestrator-level signals (exception,
    timeout, tool-limit, interrupt, proxy-enforced max-rounds stop, error)
    win over output parsing, then OOM detection from exit code, then
    max-rounds budget check, then upstream-error classification from
    proxy.jsonl + output, then the catch-all ``agent_exit_nonzero``.

    ``interrupted`` is the structural Ctrl+C signal: the SIGINT handler
    force-removed the container while the agent was mid-trial. Caller
    derives it from process-level state (``_RUN_STOP_EVENT``) and/or
    ``StateSnapshot.has_failures`` (container disappeared before
    snapshot). We trust the bool — never re-derive it from exit_code,
    because ``docker exec`` returns 0 in some Docker versions when the
    underlying container is force-removed mid-stream, which would
    otherwise be misread as a clean ``completed``.
    """
    if exception is not None:
        return termination_info_from_exception(exception)
    if timed_out:
        timeout_label = f"{timeout_seconds:g}s" if timeout_seconds else "the configured timeout"
        return TerminationInfo(
            status=TerminationStatus.FAILED,
            reason=TerminationReason.EXECUTION_TIMEOUT,
            detail=f"Agent execution exceeded {timeout_label}",
            source=TerminationSource.ORCHESTRATOR,
        )
    if terminated_by_limit:
        return TerminationInfo(
            status=TerminationStatus.INTERRUPTED,
            reason=TerminationReason.TOOL_LIMIT,
            detail="Stopped after reaching the configured tool-call limit",
            source=TerminationSource.ORCHESTRATOR,
        )
    if interrupted:
        return user_interrupted_termination()
    if terminated_by_max_rounds:
        round_count = _count_proxy_jsonl_budgeted_rounds(proxy_jsonl_path)
        if max_rounds > 0 and round_count > 0:
            detail = (
                f"Hit round budget - {round_count}/{max_rounds} successful "
                "agent rounds; orchestrator stopped the agent after proxy "
                "enforced the limit"
            )
        else:
            detail = "Orchestrator stopped the agent after proxy enforced max_rounds"
        return max_rounds_reached_termination(detail=detail)
    if error:
        return TerminationInfo(
            status=TerminationStatus.FAILED,
            reason=TerminationReason.TRIAL_ERROR,
            detail=error,
            source=TerminationSource.ORCHESTRATOR,
        )

    # Process killed by OOM-killer / external SIGKILL → exit 128+9 = 137.
    # Containers also bubble up 137 when the docker daemon kills the process
    # (resource constraint, hostmem squeeze). Treat as infra failure.
    if exit_code == 137:
        return TerminationInfo(
            status=TerminationStatus.FAILED,
            reason=TerminationReason.OOM_KILLED,
            detail=(str(output or "").strip()[:500] or
                    "Process killed by SIGKILL (exit 137) — likely OOM"),
            source=TerminationSource.ORCHESTRATOR,
        )

    # Max rounds reached — structural signal from proxy.jsonl + max_rounds.
    #
    # The in-container proxy enforces ``max_rounds`` by 429-rejecting the
    # next request after the successful non-compact round count reaches the
    # configured budget. It does NOT call ``record()`` for the rejection
    # (see ``container_proxy.py`` ``_send_json(429)`` branch — it ``return``s
    # before recording). Failed upstream requests and compact rewrites remain
    # in proxy.jsonl for audit but do not spend the agent's round budget.
    #
    # We check this BEFORE the proxy-error scan so a stray 502 / 503 in
    # the middle doesn't outvote the budget hit. A non-zero exit is the
    # gate: a clean exit at exactly max_rounds means the agent finished
    # the task within budget and shouldn't be relabelled.
    if exit_code != 0 and max_rounds > 0:
        round_count = _count_proxy_jsonl_budgeted_rounds(proxy_jsonl_path)
        if round_count >= max_rounds:
            return max_rounds_reached_termination(
                detail=(
                    f"Hit round budget — {round_count}/{max_rounds} successful "
                    f"agent rounds; proxy 429-rejected the next call"
                ),
            )

    if exit_code != 0:
        budget_termination = _token_cost_budget_termination(
            proxy_jsonl_path,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_cost=max_cost,
        )
        if budget_termination is not None:
            return budget_termination

    # Upstream model error detection — structural only, single source of truth.
    #
    # Rule: a trial is a "model error" iff its final proxy.jsonl entry is
    # a non-200 upstream response. Earlier transient upstream errors followed
    # by later successes are recovered calls, not the terminal cause.
    # Subtype refinement uses ``error.type`` substring + HTTP status code;
    # when neither pins it to a specific category, we still mark it
    # ``MODEL_ERROR`` (generic) so resume re-runs it by default.
    #
    # We do NOT scan agent stdout for "502" / "quota" / "rate limit" — those
    # phrases can appear in normal agent reasoning (e.g. while debugging a
    # target's web service) and would falsely classify a real agent failure
    # as a model error. If the proxy log shows no upstream error, the failure
    # is local → falls through to AGENT_EXIT_NONZERO below.
    if exit_code != 0:
        signal = _last_proxy_error_signal(proxy_jsonl_path)
        if signal is not None:
            upstream_reason = classify_upstream_error(
                http_status=signal.http_status,
                error_type=signal.error_type,
                error_message=signal.error_message,
            )
            if upstream_reason is None:
                # Proxy saw a non-200 but we couldn't pin the subtype — still
                # an upstream failure, still retry-worthy.
                upstream_reason = TerminationReason.MODEL_ERROR
            return TerminationInfo(
                status=TerminationStatus.FAILED,
                reason=upstream_reason,
                detail=(signal.error_message or signal.raw_text or "").strip()[:500]
                       or f"Upstream returned HTTP {signal.http_status}",
                source=TerminationSource.MODEL_PROXY,
            )

    if exit_code == 0:
        return completed_termination()
    return TerminationInfo(
        status=TerminationStatus.FAILED,
        reason=TerminationReason.AGENT_EXIT_NONZERO,
        detail=f"Agent exited with code {exit_code}",
        source=TerminationSource.AGENT,
    )


def looks_like_model_timeout(text: str) -> bool:
    lowered = str(text or "").lower()
    timeout_markers = ("timed out", "timeout", "read operation timed out")
    upstream_markers = ("api error", "502", "proxy_error", "model", "upstream", "server-side")
    return any(marker in lowered for marker in timeout_markers) and any(
        marker in lowered for marker in upstream_markers
    )


def termination_info_from_exception(exc: BaseException) -> TerminationInfo:
    """Map Cage exceptions and common interrupts into TerminationInfo."""
    if isinstance(exc, (CageInterrupt, KeyboardInterrupt)):
        return user_interrupted_termination()
    if isinstance(exc, TrialTimeout):
        return TerminationInfo(
            status=TerminationStatus.FAILED,
            reason=TerminationReason.EXECUTION_TIMEOUT,
            detail=str(exc),
            source=TerminationSource.ORCHESTRATOR,
        )
    if isinstance(exc, ProxyError) and looks_like_model_timeout(str(exc)):
        return TerminationInfo(
            status=TerminationStatus.FAILED,
            reason=TerminationReason.MODEL_TIMEOUT,
            detail=str(exc),
            source=TerminationSource.MODEL_PROXY,
        )
    if isinstance(exc, TrialError):
        return TerminationInfo(
            status=TerminationStatus.FAILED,
            reason=TerminationReason.TRIAL_ERROR,
            detail=str(exc),
            source=TerminationSource.ORCHESTRATOR,
        )
    return TerminationInfo(
        status=TerminationStatus.FAILED,
        reason=TerminationReason.TRIAL_ERROR,
        detail=str(exc),
        source=TerminationSource.ORCHESTRATOR,
    )

