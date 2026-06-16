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
    # RL training integration. When ``rl_reward_sink`` is set, this model is in
    # "RL mode": every LLM call it drives carries an ``X-Trial-Id`` header (so an
    # external trainer can group one trajectory's calls), and each finished
    # trial's reward is POSTed to this URL. Empty ⇒ an ordinary model, behaviour
    # identical to before. It's a single typed knob so the whole feature toggles
    # from one place in the model definition (registry / ``cage model set``).
    rl_reward_sink: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rl_enabled(self) -> bool:
        """RL mode is on iff this model declares a reward sink."""

        return bool(self.rl_reward_sink)

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
    def upstream_extra_body(self) -> dict[str, Any]:
        """Per-request body fields to merge into an OpenAI/vLLM upstream call.

        Lets a registry entry pin inference knobs the agent CLI cannot express
        itself — e.g. Qwen's ``enable_thinking`` (only switchable via the
        per-request ``chat_template_kwargs`` flag) or recommended sampling.
        The proxy forwards these on the anthropic→openai translation path, so a
        Claude-Code-driven run can talk to a vLLM model with a fixed inference
        config. All keys are read from ``extra`` (where the registry routes any
        field not in ``_KNOWN_FIELDS``); ``{}`` ⇒ nothing injected.

        Sources, in increasing precedence:
          - ``extra['extra_body']``        — raw passthrough dict
          - sampling keys in ``extra``     — temperature/top_p/top_k/
                                             presence_penalty/frequency_penalty
          - ``extra['chat_template_kwargs']`` merged with the ``enable_thinking``
            top-level shorthand (same key qwen-code honours).
        """

        extra = self.extra if isinstance(self.extra, dict) else {}
        body: dict[str, Any] = {}
        raw = extra.get("extra_body")
        if isinstance(raw, dict):
            body.update(raw)
        for key in (
            "temperature", "top_p", "top_k",
            "presence_penalty", "frequency_penalty",
        ):
            if extra.get(key) is not None:
                body[key] = extra[key]
        template_kwargs: dict[str, Any] = {}
        raw_kwargs = extra.get("chat_template_kwargs")
        if isinstance(raw_kwargs, dict):
            template_kwargs.update(raw_kwargs)
        if extra.get("enable_thinking") is not None:
            template_kwargs["enable_thinking"] = extra["enable_thinking"]
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        return body

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
