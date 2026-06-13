"""Model endpoint declarations consumed by agents, proxy, and judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """A registered model endpoint that Cage may call through a protocol proxy.

    The framework uses this to configure proxy upstream routing, inject API
    credentials into isolated agent runtimes, and decide whether a request
    needs protocol translation. Concurrency is not model-owned; each
    ``AgentInstance`` declares how many trials it wants to run against an
    endpoint in the current experiment.
    """

    id: str
    provider: str
    model: str
    agent_model_names: dict[str, str] = field(default_factory=dict)
    base_url: str = ""
    api_key: str = ""
    auth_source: str = ""
    timeout: int = 360
    max_retries: int = 2
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    api_keys: list[str] = field(default_factory=list)
    # The endpoint's real context window, in tokens, and the headroom an agent
    # should reserve for its next response. Both are model capabilities, so they
    # live here (declared per-model in config/models.yml) rather than as an
    # ``extra`` magic key. ``None`` ⇒ undeclared: each agent falls back to its
    # CLI's own default instead of Cage inventing a number. NOTE: the Claude
    # Code CLI has no knob to accept a custom window, so it cannot honour these.
    max_context_size: int | None = None
    reserved_context_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key_pool(self) -> list[str]:
        """All API keys available for this endpoint."""

        if self.api_keys:
            return list(self.api_keys)
        return [self.api_key] if self.api_key else []

    @property
    def protocol(self) -> str:
        """Wire protocol spoken by this endpoint."""

        if self.provider in ("openai", "vllm", "sglang"):
            return "openai"
        return "anthropic"

    @property
    def is_local_endpoint(self) -> bool:
        """True for self-hosted inference frameworks (vLLM / SGLang).

        Only these are eligible for ``--wait-for-model`` readiness polling: a
        remotely-launched local server has an unknown boot time, whereas managed
        SaaS providers (anthropic / openai / zai / …) are assumed always up.
        """
        return self.provider in ("vllm", "sglang")

    def needs_translation(self, agent_protocol: str) -> bool:
        """Return true when proxy translation is required for the agent."""

        return agent_protocol != self.protocol

    def model_name_for_agent(self, agent_name: str) -> str:
        """Return the model string a specific agent CLI should receive."""

        name = str(agent_name or "").strip()
        if not name:
            return self.model
        return str(self.agent_model_names.get(name) or self.model)
