"""Hermes agent type.

Hermes speaks Anthropic protocol when configured with ``api_mode:
anthropic_messages`` against a custom provider. The in-container proxy
already exposes an Anthropic-compatible endpoint, so we point Hermes
at ``http://localhost:<proxy_port>`` and let it talk natively.

State paths: ``.hermes/`` contains ``config.yaml`` and any session/cache
data the CLI persists.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

from cage.agents.base.output import extract_stream_json_text, failure_banner
from cage.agents.base import openviking
from cage.agents.base import AgentType, register_agent_type
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "http://localhost:8877"
_PROVIDER_NAME = "local-anthropic-proxy"


def _initial_config(model_name: str) -> dict[str, Any]:
    """Build the initial ``~/.hermes/config.yaml`` payload (no proxy yet).

    The provider entry's ``base_url`` and the model's ``default`` are
    refreshed per-trial inside :meth:`HermesAgent.env_vars` once both
    proxy port and ``ModelConfig`` are known. We still seed something
    valid so the CLI can boot at setup time.
    """
    return {
        "custom_providers": [
            {
                "name": _PROVIDER_NAME,
                "base_url": _DEFAULT_BASE_URL,
                "api_mode": "anthropic_messages",
            }
        ],
        "model": {
            "provider": f"custom:{_PROVIDER_NAME}",
            "default": model_name or "",
        },
    }


@register_agent_type
class HermesAgent(AgentType):
    name = "hermes"
    state_paths = [".hermes"]
    default_image = "cage/hermes:pentestenv"
    dockerfile = "docker/hermes/pentestenv.Dockerfile"
    plugin_images = {"openviking-memory": "openviking"}

    def install_command(self, version: str = "latest") -> str:
        # Hermes is installed via the upstream install script; version is
        # not currently selectable through that script, so it is ignored.
        return (
            "curl -fsSL "
            "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh "
            "| bash -s -- --skip-setup"
        )

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        """Launch Hermes in non-interactive mode with the task prompt.

        Per-trial config rewriting is handled by :meth:`env_vars` (which
        receives the container handle); this method only emits the CLI
        invocation. ``max_rounds`` / ``proxy_url`` are accepted for
        signature compatibility but not used — Hermes reads everything it
        needs from ``~/.hermes/config.yaml``.
        """
        escaped = prompt.replace("'", "'\\''")
        return f"hermes chat -q '{escaped}'"

    def parse_output(self, result: ExecResult) -> str:
        """Best-effort output extraction.

        Hermes can emit either plain text or NDJSON depending on flags.
        We try NDJSON first (looking for the final assistant message) and
        fall back to raw stdout.
        """
        banner = failure_banner(result)
        if banner is not None:
            return banner
        last_text, saw_json = extract_stream_json_text(result.stdout)
        if saw_json and last_text:
            return last_text
        return result.stdout.strip()[:4000] or last_text

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
        """Refresh ``~/.hermes/config.yaml`` with per-trial values, return env.

        Hermes is configured by file, so we patch the YAML in place each
        trial: load → mutate ``custom_providers[*].base_url`` for our
        provider, ``model.default``, ``agent.max_turns``, and
        ``skills.external_dirs`` (so benchmarks like skill_inject that
        drop a SKILL.md into ``<workspace>/.claude/skills/`` get picked
        up — Hermes' default scan only looks at ``~/.hermes/skills``).
        Returned env carries only what we cannot put in the file.
        """
        if container is not None:
            self._patch_config(
                container,
                home_dir=home_dir,
                base_url=proxy_url or _DEFAULT_BASE_URL,
                model_name=model.model,
                max_turns=max_rounds,
                workspace_dir=workspace_dir,
            )

        env: dict[str, str] = {"DISABLE_TELEMETRY": "1"}
        if model.api_key:
            env["ANTHROPIC_API_KEY"] = model.api_key
        return env

    def _patch_config(
        self, container: Any, *, home_dir: str, base_url: str, model_name: str,
        max_turns: int = 0, workspace_dir: str = "",
    ) -> None:
        """Load → mutate → write ``~/.hermes/config.yaml`` via the yaml lib."""
        config_path = f"{home_dir.rstrip('/')}/.hermes/config.yaml"
        read = container.exec(f"cat {config_path} 2>/dev/null", timeout=5.0)

        data: dict[str, Any]
        if read.exit_code == 0 and read.stdout.strip():
            try:
                loaded = yaml.safe_load(read.stdout)
                data = loaded if isinstance(loaded, dict) else {}
            except yaml.YAMLError as exc:
                logger.warning(
                    "hermes config.yaml unparseable, reseeding: %s", exc,
                )
                data = {}
        else:
            data = {}

        if not data:
            data = _initial_config(model_name)

        providers = data.setdefault("custom_providers", [])
        if not isinstance(providers, list):
            providers = []
            data["custom_providers"] = providers

        target = next(
            (p for p in providers if isinstance(p, dict) and p.get("name") == _PROVIDER_NAME),
            None,
        )
        if target is None:
            target = {
                "name": _PROVIDER_NAME,
                "base_url": base_url,
                "api_mode": "anthropic_messages",
            }
            providers.append(target)
        else:
            target["base_url"] = base_url
            target.setdefault("api_mode", "anthropic_messages")

        model_block = data.setdefault("model", {})
        if not isinstance(model_block, dict):
            model_block = {}
            data["model"] = model_block
        model_block["provider"] = f"custom:{_PROVIDER_NAME}"
        if model_name:
            model_block["default"] = model_name

        # max_turns is sourced from the orchestrator's effective max_rounds.
        # Negative values mean unset; in that case Hermes uses its own default.
        agent_block = data.setdefault("agent", {})
        if not isinstance(agent_block, dict):
            agent_block = {}
            data["agent"] = agent_block
        if max_turns and max_turns > 0:
            agent_block["max_turns"] = int(max_turns)
        else:
            agent_block.pop("max_turns", None)

        # skills.external_dirs — surface benchmarks that drop their own
        # skill catalogues into the workspace (skill_inject's
        # ``<workspace>/.claude/skills/<name>/SKILL.md`` layout, mirroring
        # claude_code's defaults). Without this, Hermes only scans
        # ``~/.hermes/skills`` and the injected skill is invisible.
        if workspace_dir:
            skills_block = data.setdefault("skills", {})
            if not isinstance(skills_block, dict):
                skills_block = {}
                data["skills"] = skills_block
            ext_dirs = [
                f"{workspace_dir.rstrip('/')}/.claude/skills",
                f"{workspace_dir.rstrip('/')}/.hermes/skills",
            ]
            skills_block["external_dirs"] = ext_dirs

        rendered = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        container.write_file(config_path, rendered)

    def version_command(self) -> str:
        # Cheap presence check — actually running `hermes --version` takes
        # 5-15s on first cold start (loads OpenAI/Anthropic SDKs + tool
        # schemas) which trips cage's 10s exec timeout and triggers a bogus
        # reinstall. A path lookup tells us everything we need: the image
        # builder put hermes here, and cage doesn't pin a specific version.
        return "command -v hermes >/dev/null 2>&1 && echo hermes-installed || echo unknown"

    def setup_container(
        self, container: Any, *, home_dir: str,
        model: ModelConfig | None = None,
    ) -> None:
        """Seed ``~/.hermes/config.yaml`` so the CLI has a valid config to read.

        The dynamic ``base_url`` is filled in per trial by :meth:`env_vars`
        once the proxy port is known.
        """
        config_dir = f"{home_dir.rstrip('/')}/.hermes"
        container.exec(f"mkdir -p {config_dir}", timeout=5.0)

        model_name = model.model if model is not None else ""
        config_yaml = yaml.safe_dump(
            _initial_config(model_name),
            sort_keys=False, default_flow_style=False,
        )
        container.write_file(f"{config_dir}/config.yaml", config_yaml)
        container.exec(f"chown -R agent:agent {config_dir}", timeout=5.0)

    # ------------------------------------------------------------------ #
    # Plugin installation
    # ------------------------------------------------------------------ #

    def _plugin_installed(
        self, container: Any, *, name: str, home_dir: str,
    ) -> bool:
        """Detect installation by presence of the plugin directory."""
        path = f"{home_dir}/.hermes/plugins/{name}"
        result = container.exec(f"test -d {path}", timeout=5.0)
        return result.exit_code == 0

    def _do_install_plugin(
        self, container: Any, *, name: str, home_dir: str,
        agent_id: str = "",
    ) -> None:
        """Copy the plugin directory from the mounted marketplace into ~/.hermes.

        The marketplace tree at ``/opt/cage-plugins/{name}-marketplace`` has a
        ``plugins/`` subdirectory containing the plugin payload. We copy its
        contents into ``~/.hermes/plugins/`` so the Hermes CLI picks them up.
        """
        src = f"/opt/cage-plugins/{name}-marketplace/plugins"
        dst = f"{home_dir.rstrip('/')}/.hermes/plugins"
        container.exec(f"mkdir -p {dst}", user="agent", timeout=10.0)
        # Use cp -rT-style semantics: copy the *contents* of src into dst.
        container.exec(
            f"cp -a {src}/. {dst}/",
            user="agent", timeout=300.0,
        )
        container.exec(
            f"chown -R agent:agent {home_dir.rstrip('/')}/.hermes/plugins",
            timeout=10.0,
        )

        if name == "openviking-memory":
            openviking.seed_conf(
                container, home_dir=home_dir, agent_id=agent_id,
                namespace_key="hermes",
            )

    # ------------------------------------------------------------------ #
    # OpenViking server lifecycle (same shape as claude_code)
    # ------------------------------------------------------------------ #

    def start_openviking_server(
        self, container: Any, *, home_dir: str,
    ) -> str:
        """Start ``openviking-server`` (shared lifecycle in base.openviking)."""
        return openviking.start_server(
            container, home_dir=home_dir,
            image_hint="pursu1ng/cage-images:hermes-openviking",
        )

    @property
    def protocol(self) -> str:
        """Hermes is configured to speak Anthropic protocol via the proxy."""
        return "anthropic"
