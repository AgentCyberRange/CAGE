"""Trial termination vocabulary: status / reason / source / info value types."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerminationStatus(str, Enum):
    """Stable coarse-grained trial lifecycle status."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class TerminationReason(str, Enum):
    """Stable machine-readable reason a trial stopped.

    Reasons partition into three groups by how resume should treat them:

    1. **Expected** (resume keeps the result, does NOT re-run): the agent
       reached a natural stopping point.
       - ``COMPLETED``           agent exited 0 with no errors
       - ``MAX_ROUNDS_REACHED``  agent hit its per-trial round budget
                                 (detected structurally from successful
                                 non-compact proxy rounds —
                                 see ``max_rounds_reached_termination``)
       - ``MAX_INPUT_TOKENS_REACHED``  proxy-observed input token budget hit
       - ``MAX_OUTPUT_TOKENS_REACHED`` proxy-observed output token budget hit
       - ``MAX_COST_REACHED``          proxy-observed cost budget hit
       - ``EXECUTION_TIMEOUT``   wall-clock orchestrator cap
       - ``TOOL_LIMIT``          tool-call budget exhausted (legacy)

    2. **Unexpected / infra** (resume re-runs by default): something outside
       the agent's control prevented a real result.
       - ``MODEL_QUOTA_EXHAUSTED``  upstream returned 429 usage_limit_reached
       - ``MODEL_RATE_LIMITED``     upstream 429 rate-limit
       - ``MODEL_BAD_GATEWAY``      upstream 502/503/504
       - ``MODEL_AUTH_ERROR``       upstream 401/403
       - ``MODEL_CONTEXT_OVERFLOW`` request exceeded model context window
       - ``MODEL_TIMEOUT``          upstream request timed out
       - ``MODEL_ERROR``            other unclassified upstream HTTP error
       - ``OOM_KILLED``             container killed (SIGKILL / exit 137)
       - ``TARGET_UNAVAILABLE``     target stack failed to launch
       - ``TRIAL_ERROR``            orchestrator / container raised an exception

    3. **State** (resume policy depends on user intent):
       - ``USER_INTERRUPTED``       Ctrl+C while running
       - ``CANCELLED_BEFORE_START`` Ctrl+C before trial started (rerun)
       - ``AGENT_EXIT_NONZERO``     truly unclassified agent failure (keep)
    """

    # Expected
    COMPLETED = "completed"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    MAX_INPUT_TOKENS_REACHED = "max_input_tokens_reached"
    MAX_OUTPUT_TOKENS_REACHED = "max_output_tokens_reached"
    MAX_COST_REACHED = "max_cost_reached"
    EXECUTION_TIMEOUT = "execution_timeout"
    TOOL_LIMIT = "tool_limit"

    # Unexpected — upstream model proxy
    MODEL_QUOTA_EXHAUSTED = "model_quota_exhausted"
    MODEL_RATE_LIMITED = "model_rate_limited"
    MODEL_BAD_GATEWAY = "model_bad_gateway"
    MODEL_AUTH_ERROR = "model_auth_error"
    MODEL_CONTEXT_OVERFLOW = "model_context_overflow"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_ERROR = "model_error"

    # Unexpected — infra
    OOM_KILLED = "oom_killed"
    TARGET_UNAVAILABLE = "target_unavailable"
    TRIAL_ERROR = "trial_error"

    # State
    USER_INTERRUPTED = "user_interrupted"
    CANCELLED_BEFORE_START = "cancelled_before_start"
    AGENT_EXIT_NONZERO = "agent_exit_nonzero"


class TerminationSource(str, Enum):
    """Component that produced the terminal condition."""

    AGENT = "agent"
    MODEL_PROXY = "model_proxy"
    ORCHESTRATOR = "orchestrator"
    USER = "user"


@dataclass(frozen=True)
class TerminationInfo:
    """Structured trial termination metadata shared by storage and UI."""

    status: TerminationStatus
    reason: TerminationReason
    detail: str
    source: TerminationSource

    def to_metadata(self) -> dict[str, str]:
        return {
            "status": self.status.value,
            "termination_reason": self.reason.value,
            "termination_detail": self.detail,
            "termination_source": self.source.value,
        }

