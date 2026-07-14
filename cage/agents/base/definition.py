"""Static contract implemented by each supported agent CLI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cage.agents.base.resources import AgentContainerResources, HostRunService
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult


class AgentType(ABC):
    """Static definition of how an installed agent works inside Cage."""

    name: str = ""
    state_paths: list[str] = []
    default_image: str = ""
    dockerfile: str = ""
    plugin_images: dict[str, str] = {}

    def image_for_variant(self, variant: str) -> str:
        """Return the local image ref for a build/runtime ``variant``.

        Each agent has its own ``cage/<agent>`` repo (the part of
        ``default_image`` before the colon) and the variant is the tag, e.g.
        ``cage/claude-code:openviking``. These images are built and run
        locally; only the base images a Dockerfile ``FROM``-s (e.g.
        ``pursu1ng/cage-images:pentest-env``) are pulled from the registry.
        """

        repo = self.default_image.split(":", 1)[0]
        return f"{repo}:{variant}"

    @abstractmethod
    def install_command(self, version: str = "latest") -> str:
        """Return the shell command that installs this agent CLI."""

    @abstractmethod
    def build_launch_command(
        self,
        prompt: str,
        *,
        model: ModelConfig,
        max_rounds: int = -1,
        proxy_url: str = "",
    ) -> str:
        """Build the command used to launch one agent session."""

    @abstractmethod
    def parse_output(self, result: ExecResult) -> str:
        """Extract the task answer from an agent process result."""

    def env_vars(
        self,
        *,
        proxy_url: str,
        model: ModelConfig,
        container: Any = None,
        home_dir: str = "/home/agent",
        workspace_dir: str = "",
        max_rounds: int = -1,
        context_compaction_threshold: float | None = None,
    ) -> dict[str, str]:
        """Build environment variables exposed to the agent CLI.

        ``context_compaction_threshold`` is ``None`` unless the user set it in
        the experiment YAML. ``None`` means "do not impose a threshold" — the
        agent's own CLI default stands. Implementations that map it to a CLI
        knob must skip the knob (or substitute their CLI's documented default)
        when it is ``None``.

        The keyword set IS the contract the engine calls with (see
        ``execute_trial``) — implementations accept all of it and use what
        they need, instead of hiding the channel behind ``**kwargs``.
        """

        return {}

    def version_command(self) -> str:
        """Return a command that prints the installed agent version."""

        return "echo unknown"

    def artifact_files(self) -> list[tuple[str, str]]:
        """Container files to pull into the trial dir after the agent finishes.

        Returns ``(container_path, artifact_filename)`` pairs. Collection is
        best-effort — a missing file is skipped silently. The default is none;
        agents that emit their own logs/traces (e.g. a custom agent's node
        trace) override this so the runner needs no agent-specific knowledge.
        """

        return []

    def setup_container(
        self,
        container: Any,
        *,
        home_dir: str,
        model: ModelConfig | None = None,
    ) -> None:
        """Prepare a started container before the first agent run.

        Subclasses may widen the signature with additional defaulted
        keywords (kimi/qwen take a compaction threshold) — callers pass
        only this named contract.
        """

        return None

    def container_resources(
        self,
        *,
        home_dir: str,
        model: ModelConfig,
    ) -> AgentContainerResources:
        """Return pre-start Docker resources needed by this agent."""

        del home_dir, model
        return AgentContainerResources()

    def validate_auth(self, model: ModelConfig) -> None:
        """Validate local authentication preconditions for this agent/model."""

        return None

    @property
    def subscription_upstream_base_url(self) -> str:
        """Real provider API the proxy forwards to in subscription/OAuth mode.

        When a model declares an ``auth_source`` (host OAuth credentials) and
        no explicit ``base_url``, Cage runs the agent in subscription mode: the
        CLI authenticates with its own OAuth bearer, and the in-container proxy
        forwards that request — bearer passed through verbatim — to the
        provider's real backend. That backend is agent-specific (Anthropic for
        Claude Code, the ChatGPT Codex backend for Codex), so each agent that
        supports OAuth names its upstream here. Empty ⇒ the agent has no
        subscription backend (OAuth mode unsupported).
        """

        return ""

    def host_run_services(
        self,
        model: ModelConfig,
        *,
        http_proxy: str = "",
    ) -> list[HostRunService]:
        """Return host-side processes required while a run is active."""

        del model, http_proxy
        return []

    def install_plugin(
        self,
        container: Any,
        *,
        name: str,
        home_dir: str,
        agent_id: str = "",
    ) -> None:
        """Install a plugin from a mounted marketplace directory."""

        if self._plugin_installed(container, name=name, home_dir=home_dir):
            return
        self._do_install_plugin(
            container,
            name=name,
            home_dir=home_dir,
            agent_id=agent_id,
        )

    def _plugin_installed(
        self,
        container: Any,
        *,
        name: str,
        home_dir: str,
    ) -> bool:
        """Return true if ``name`` is already installed in the container."""

        return False

    def _do_install_plugin(
        self,
        container: Any,
        *,
        name: str,
        home_dir: str,
        agent_id: str = "",
    ) -> None:
        """Perform the agent-specific plugin installation."""
