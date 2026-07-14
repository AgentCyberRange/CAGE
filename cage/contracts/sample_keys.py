"""The framework-owned keys of the open ``Trial.sample`` bag.

``Trial.sample`` is deliberately an untyped mapping — Layer-2 benchmarks
choose what their samples look like. But the framework itself also writes
into and interprets that bag, and every such key is a cross-layer contract:
prompt templates, resume bridging, and pass@k expansion all depend on the
exact spelling. This module declares those keys so they are named, greppable
channels instead of scattered string literals.

Declared elsewhere (richer docstrings, same idea):

- :data:`cage.contracts.runtime_state.RUNTIME_STATE_KEY` — the target
  runtime-state channel.
- :data:`cage.contracts.runtime_state.CHECK_SUPPORTED_KEY` — the live-check
  availability flag prompt templates branch on.

Benchmark obligations (``sample["id"]`` / ``sample["content"]``) are part of
the :class:`cage.benchmarks.Benchmark` ABC contract and stay documented there.
"""

from __future__ import annotations

from typing import Any

#: Structured pass@k index. Written by the conductor's (and resume preview's)
#: pass@k expansion; wins over the ``…/pass_N`` trial-id suffix everywhere a
#: pass index is recovered, because it is stored before execution.
SAMPLE_PASS_INDEX_KEY = "pass_index"

#: Benchmark-declared variant (e.g. a prompt profile). The default trial
#: sequence nests it into the trial directory name.
SAMPLE_VARIANT_KEY = "variant"

#: Benchmark-declared fallback round budget for one sample. Lowest precedence
#: in :func:`cage.contracts.execution.resolve_max_rounds`.
SAMPLE_MAX_ROUNDS_KEY = "max_rounds"

#: Benchmark-declared wrap-up window: how many rounds before the hard round cap
#: the in-container proxy should start injecting the wrap-up reminder (a graceful
#: pre-cap flush, so the agent finalizes its deliverable instead of being hard-
#: stopped). Offset, not absolute, so it tracks whatever ``max_rounds`` resolves
#: to. Lowest precedence in :func:`cage.contracts.execution.resolve_wrapup`.
SAMPLE_WRAPUP_BEFORE_KEY = "wrapup_before"

#: Benchmark-declared text of the wrap-up reminder. Benchmark-specific content
#: (what "finalize your deliverable" means for this domain); the framework only
#: forwards it verbatim into the request. Empty ⇒ wrap-up disabled.
SAMPLE_WRAPUP_MESSAGE_KEY = "wrapup_message"

#: Benchmark-declared per-sample target runtime overrides, merged into the
#: target launch request by Layer-1 provisioning.
SAMPLE_RUNTIME_ARGS_KEY = "runtime_args"

#: Written by Layer-1 provisioning when a target stack is attached:
#: the resolved service descriptor and the agent-facing network identity.
SAMPLE_TARGET_INFO_KEY = "target_info"
SAMPLE_NETWORK_NAME_KEY = "network_name"
SAMPLE_NETWORK_SUBNET_KEY = "network_subnet"


def sample_pass_index(sample: Any) -> int | None:
    """The structured pass@k index stored on a sample bag, if declared.

    Returns ``None`` when the bag is not a mapping, the key is absent, or the
    value is malformed — callers then fall back to parsing the ``…/pass_N``
    trial-id suffix via :func:`cage.experiment.model.trial_id.parse_trial_id`.
    The index is clamped to ``>= 1``.
    """

    if not isinstance(sample, dict):
        return None
    value = sample.get(SAMPLE_PASS_INDEX_KEY)
    if value is None:
        return None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return None
