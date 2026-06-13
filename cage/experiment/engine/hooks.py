"""Hook system — user-defined Python functions at lifecycle points.

Hook points:
  pre_run       — before any trial runs; produces the trial sequence
  pre_trial     — before each trial
  post_trial    — after each trial
  pre_chunk     — at chunk boundary (before)
  post_chunk    — at chunk boundary (after); can append reflection trials
  post_run      — after all trials complete
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

from cage.contracts.sample_keys import SAMPLE_PASS_INDEX_KEY, SAMPLE_VARIANT_KEY
from cage.experiment.model import Trial, TrialType

logger = logging.getLogger(__name__)


class HookContext:
    """Read-only context passed to hook functions."""

    def __init__(
        self,
        *,
        experiment_config: dict[str, Any],
        samples: list[dict[str, Any]],
        trials_completed: list[Trial],
        trials_pending: list[Trial],
        run_artifacts_dir: str | None = None,
        current_trial: Trial | None = None,
    ):
        self.experiment_config = experiment_config
        self.samples = samples
        self.trials_completed = list(trials_completed)
        self.trials_pending = list(trials_pending)
        self.run_artifacts_dir = run_artifacts_dir
        self.current_trial = current_trial


@dataclass(frozen=True)
class HookRef:
    """Reference to a hook function."""

    module: str
    function: str
    params: dict[str, Any] = field(default_factory=dict)

    def load(self) -> Any:
        mod = importlib.import_module(self.module)
        return getattr(mod, self.function)

    def call(self, ctx: HookContext) -> Any:
        fn = self.load()
        return fn(ctx, **self.params)


@dataclass
class HookRegistry:
    """All hooks for an experiment, organized by trigger point."""

    pre_run: list[HookRef] = field(default_factory=list)
    pre_trial: list[HookRef] = field(default_factory=list)
    post_trial: list[HookRef] = field(default_factory=list)
    pre_chunk: list[HookRef] = field(default_factory=list)
    post_chunk: list[HookRef] = field(default_factory=list)
    post_run: list[HookRef] = field(default_factory=list)

    def fire(self, point: str, ctx: HookContext) -> Any:
        hooks = getattr(self, point, [])
        results = []
        for hook in hooks:
            try:
                results.append(hook.call(ctx))
            except Exception:
                logger.exception("Hook %s.%s failed", hook.module, hook.function)
        return results


def _sanitize_trial_id(name: str) -> str:
    """Make a string safe for use as a directory/filename."""
    import re
    # Replace non-alphanumeric (except dash/underscore/dot) with underscore
    sanitized = re.sub(r'[^\w\-.]', '_', name)
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    return sanitized.strip('_')


def default_trial_sequence(ctx: HookContext) -> list[Trial]:
    """Default pre_run hook: samples → trials in order."""
    trials = []
    for i, sample in enumerate(ctx.samples):
        # Use sample id or name as trial directory name
        sample_name = sample.get("id") or sample.get("name") or f"sample_{i:04d}"
        trial_id = _sanitize_trial_id(str(sample_name))
        # If the sample carries a variant key (e.g. one_day / zero_day),
        # nest the variant under the challenge directory so trials for different
        # variants of the same challenge live in sibling subdirectories.
        variant = sample.get(SAMPLE_VARIANT_KEY)
        if variant:
            variant_segment = _sanitize_trial_id(str(variant))
            if variant_segment:
                trial_id = f"{trial_id}/{variant_segment}"
        trials.append(Trial(
            id=trial_id,
            index=i,
            type=TrialType.TASK,
            sample=sample,
        ))
    return trials


def expand_trials_for_passk(trials: list[Trial], passk: int) -> list[Trial]:
    """Replay the trial sequence ``passk`` times, stamping each pass.

    The order is ``[pass1_all_samples, pass2_all_samples, …]`` — also the
    worker-pool admission order under parallelism. Each expanded trial gets a
    runtime 2-part ``<task>/pass_<n>`` id (the 3-part canonical id is built
    later via ``format_trial_id``) and the structured
    ``sample["pass_index"]`` the rest of the system treats as authoritative.

    The conductor builds the real sequence with this and the resume preview
    mirrors it — one implementation so the preview cannot drift. Indices are
    not assigned here; both callers re-enumerate afterwards.
    """

    if passk <= 1:
        return trials
    expanded: list[Trial] = []
    for pass_idx in range(1, passk + 1):
        for t in trials:
            sample_copy = dict(t.sample)
            sample_copy[SAMPLE_PASS_INDEX_KEY] = pass_idx
            expanded.append(Trial(
                id=f"{t.id}/pass_{pass_idx}",
                index=0,
                type=t.type,
                sample=sample_copy,
            ))
    return expanded


def load_hooks(cfg: dict[str, Any]) -> HookRegistry:
    """Build HookRegistry from YAML config."""
    registry = HookRegistry()
    for point in ("pre_run", "pre_trial", "post_trial",
                  "pre_chunk", "post_chunk", "post_run"):
        raw_hooks = cfg.get(point, [])
        for h in raw_hooks:
            ref = HookRef(
                module=h["module"],
                function=h["function"],
                params=h.get("params", {}),
            )
            getattr(registry, point).append(ref)
    return registry
