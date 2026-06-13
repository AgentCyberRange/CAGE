"""Runtime configuration for one agent in one experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

from cage.agents.base.definition import AgentType
from cage.models import ModelConfig


@dataclass
class AgentInstance:
    """Resolved agent configuration used by the experiment runner."""

    agent_type: AgentType
    model: ModelConfig
    # Multi-source rotation pool. When non-empty, this run round-robins across
    # these registered endpoints per trial (see ``_trial_model_for_agent``);
    # ``model`` is the logical representative whose ``id`` is the stable run key,
    # with endpoint fields from the first source. Empty means single-endpoint:
    # the run always uses ``model``.
    model_sources: list[ModelConfig] = field(default_factory=list)
    # Per-source concurrency caps (source model id → max in-flight trials). A
    # source absent here takes the default even share of the run's total. Only
    # meaningful with model_sources.
    source_concurrency: dict[str, int] = field(default_factory=dict)
    id: str = ""
    home: str = "/home/agent/workspace"
    session_args: list[str] = field(default_factory=list)
    shared_paths: list[str] = field(default_factory=list)
    skill: str = ""
    plugins: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    version: str = "latest"
    image: str = ""
    # Config space: None ⇒ inherit the runtime budget; "unlimited"; -1 ⇒
    # benchmark default; N ⇒ N rounds. Resolved by resolve_max_rounds().
    max_rounds: int | str | None = None
    # None ⇒ user did not set a threshold; each agent falls back to its own
    # CLI default instead of Cage imposing one. See AgentType.env_vars.
    context_compaction_threshold: float | None = None
    max_concurrent: int = 0

    @property
    def stateful(self) -> bool:
        """Return true when state paths persist across trials."""

        return bool(self.shared_paths)

    @property
    def effective_image(self) -> str:
        """Return the Docker image selected for this agent instance."""

        if self.image:
            return self.image
        for plugin in self.plugins:
            variant = self.agent_type.plugin_images.get(plugin)
            if variant:
                return self.agent_type.image_for_variant(variant)
        return self.agent_type.default_image

    @property
    def effective_state_paths(self) -> list[str]:
        """Paths to snapshot and restore between stateful trials."""

        if self.shared_paths:
            return list(self.shared_paths)
        return list(self.agent_type.state_paths)

    def label(self) -> str:
        """Human-readable identity used in artifacts and logs."""

        agent_id = self.id or self.agent_type.name
        mode = "stateful" if self.stateful else "stateless"
        return f"{agent_id}:{self.model.id}:{mode}"

    @property
    def subject_plan_id(self) -> str:
        """Canonical ``SubjectPlan`` id for this agent (``agent:model:mode``).

        The id every canonical ``TrialPlan`` and resource-ledger row carries.
        The profile leg mirrors :meth:`label` — ``stateful`` / ``stateless`` —
        so the canonical subject id, the per-agent run directory, and the
        human-readable label all agree (no ``:default`` vs ``:stateless`` skew).
        Real user-selectable profiles can replace the mode leg later.
        """

        agent_id = self.id or self.agent_type.name
        mode = "stateful" if self.stateful else "stateless"
        return f"{agent_id}:{self.model.id}:{mode}"
