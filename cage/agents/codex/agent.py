"""Codex CLI agent type.

Codex speaks OpenAI protocol natively.
"""

from __future__ import annotations

import json
from typing import Any

from cage.agents.base import openviking
from cage.agents.base import AgentType, register_agent_type
from cage.agents.codex.output import parse_codex_event_stream
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult


@register_agent_type
class CodexAgent(AgentType):
    name = "codex"
    state_paths = [".codex"]
    default_image = "cage/codex:pentestenv"
    dockerfile = "docker/codex/pentestenv.Dockerfile"

    def install_command(self, version: str = "latest") -> str:
        return f"npm install -g @openai/codex@{version}"

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        escaped = prompt.replace("'", "'\\''")
        # Static codex config (model_provider, sandbox, auto-compact disable)
        # lives in ~/.codex/config.toml, seeded by setup_container. The only
        # per-trial dynamic value is the proxy base URL, since the in-container
        # proxy binds a fresh port each trial.
        base_url = (proxy_url.rstrip("/") + "/v1") if proxy_url else (model.base_url or "")
        base_flag = (
            f' -c model_providers.cage.base_url="{base_url}"' if base_url else ""
        )
        return (
            f"codex exec '{escaped}' --model {model.model}"
            f"{base_flag}"
            f" --dangerously-bypass-approvals-and-sandbox"
            f" --skip-git-repo-check"
            f" --cd /home/agent/workspace"
            f" --json"
        )

    def parse_output(self, result: ExecResult) -> str:
        if result.exit_code != 0 and not result.stdout.strip():
            return f"[Agent exited with code {result.exit_code}]\n{result.stderr[:1000]}"
        summary = parse_codex_event_stream(result.stdout)
        if summary.is_event_stream:
            parsed = summary.final_output()
            if parsed:
                return parsed
        return result.stdout.strip()

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
        env: dict[str, str] = {}
        if proxy_url:
            env["OPENAI_BASE_URL"] = proxy_url
        elif model.base_url:
            env["OPENAI_BASE_URL"] = model.base_url
        if model.api_key:
            env["OPENAI_API_KEY"] = model.api_key
        return env

    def setup_container(
        self, container: Any, *, home_dir: str,
        model: ModelConfig | None = None,
    ) -> None:
        """Seed ``~/.codex/auth.json`` and ``~/.codex/config.toml``.

        Putting the static codex configuration in ``config.toml`` (the
        supported upstream path) keeps `codex exec` invocations short and
        avoids re-passing the same ``-c`` flags every trial. The shape
        mirrors a known-working user config: ``model``, ``model_provider``,
        and a ``[model_providers.cage]`` block with ``wire_api = "responses"``.

        The proxy ``base_url`` is set per-trial via ``-c`` in
        ``build_launch_command`` because the in-container proxy chooses a
        free port at each trial. Per-experiment knobs such as
        ``model_reasoning_effort`` belong in ``project.yml`` under
        ``agents[].session_args`` rather than here.
        """
        codex_dir = f"{home_dir.rstrip('/')}/.codex"
        container.exec(f"mkdir -p {codex_dir}", timeout=5.0)

        # Auth — real API key when known, placeholder otherwise. codex exec
        # tolerates a placeholder when OPENAI_API_KEY is set in env, but the
        # interactive TUI insists on a non-empty auth.json.
        api_key = (model.api_key if model is not None else "") or "placeholder"
        container.write_file(
            f"{codex_dir}/auth.json", json.dumps({"OPENAI_API_KEY": api_key}),
        )

        model_name = model.model if model is not None else ""
        model_line = f'model = "{model_name}"\n' if model_name else ""
        config_toml = (
            f'{model_line}'
            f'model_provider = "cage"\n'
            f'approval_policy = "never"\n'
            f'sandbox_mode = "danger-full-access"\n'
            f'\n'
            f'[model_providers.cage]\n'
            f'name = "Cage Proxy"\n'
            f'env_key = "OPENAI_API_KEY"\n'
            f'wire_api = "responses"\n'
        )
        container.write_file(f"{codex_dir}/config.toml", config_toml)
        container.exec(f"chown -R agent:agent {codex_dir}", timeout=5.0)

    def version_command(self) -> str:
        # NOTE: do NOT run 'codex --version' — it initialises a Landlock
        # sandbox as root, which permanently blocks later user-level execs.
        return "which codex >/dev/null 2>&1 && dpkg-query -W codex 2>/dev/null || (ls /usr/bin/codex >/dev/null 2>&1 && echo 'codex installed') || echo unknown"

    # ------------------------------------------------------------------ #
    # Plugin installation
    # ------------------------------------------------------------------ #

    def _plugin_installed(
        self, container: Any, *, name: str, home_dir: str,
    ) -> bool:
        """Check if MCP server is already registered in config.toml."""
        config = f"{home_dir}/.codex/config.toml"
        result = container.exec(
            f"grep -q '{name}' {config} 2>/dev/null", timeout=5.0,
        )
        return result.exit_code == 0

    def _do_install_plugin(
        self, container: Any, *, name: str, home_dir: str,
        agent_id: str = "",
    ) -> None:
        """Register the plugin's MCP server via ``codex mcp add``.

        The server binary is read from the mounted marketplace directory
        at ``/opt/cage-plugins/{name}-marketplace/plugins/{name}/``.
        """
        server = (
            f"/opt/cage-plugins/{name}-marketplace"
            f"/plugins/{name}/servers/memory-server.js"
        )
        container.exec(
            f"codex mcp add {name} -- node {server}",
            user="agent", timeout=10.0,
        )

        if name == "openviking-memory":
            openviking.seed_conf(container, home_dir=home_dir)

    @property
    def protocol(self) -> str:
        return "openai"
