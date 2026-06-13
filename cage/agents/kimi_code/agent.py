"""Kimi Code CLI agent type.

Upstream: https://github.com/MoonshotAI/kimi-cli (binary: ``kimi``;
PyPI package ``kimi-cli`` installed via ``uv tool install``). The CLI
supports several upstream wire formats — ``kimi`` (proprietary
Moonshot), ``openai_legacy`` (Chat Completions), ``openai_responses``,
``anthropic``, ``gemini``, ``vertexai`` — selected by the
``[providers.<name>].type`` key in ``~/.kimi/config.toml``.

For Cage we need ``openai_legacy`` because the model endpoints we
target (vLLM / sglang) speak OpenAI Chat Completions, and the
in-container proxy passes those through transparently. The
``KIMI_BASE_URL`` env-var path is *not* used because it configures the
implicit Moonshot-proprietary "kimi" provider, which would fail against
an OpenAI-compatible upstream.

Per-trial proxy port rewriting is handled the same way Hermes handles
``~/.hermes/config.yaml``: ``setup_container`` seeds a baseline TOML,
and ``env_vars`` rewrites the provider ``base_url`` on every trial
(``env_vars`` is the only AgentType hook that receives both the
container handle and the live ``proxy_url``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cage.agents.base.output import failure_banner
from cage.agents.base import AgentType, register_agent_type
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult

logger = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "http://localhost:8877/v1"
_PROVIDER_NAME = "cage"
_MODEL_ALIAS = "cage-model"
# Fallback compaction_trigger_ratio when the user did not opt into a Cage-level
# threshold. Matches kimi-cli's documented default; clamped to [0.5, 0.99].
_KIMI_DEFAULT_RATIO = 0.85


def _toml_string(value: Any) -> str:
    """Render a TOML basic string with JSON-compatible escaping."""
    return json.dumps(str(value), ensure_ascii=False)


def _provider_extra_options(extra: dict[str, Any]) -> dict[str, Any]:
    reasoning_key = extra.get("reasoning_key")
    custom_headers = extra.get("custom_headers")
    return {
        "reasoning_key": str(reasoning_key) if reasoning_key is not None else None,
        "custom_headers": custom_headers if isinstance(custom_headers, dict) else None,
        "display_name": str(extra.get("display_name", "Cage proxied model")),
    }


def _render_config_toml(
    *, base_url: str, api_key: str, model_name: str,
    compaction_ratio: float = 0.85,
    max_context_size: int | None = None,
    reserved_context_size: int | None = None,
    reasoning_key: str | None = None,
    custom_headers: dict[str, Any] | None = None,
    display_name: str = "Cage proxied model",
) -> str:
    """Render a minimal ``~/.kimi/config.toml`` payload.

    ``openai_legacy`` is the OpenAI Chat Completions wire format; the
    proxy passes those through to vLLM unchanged. ``default_model``
    pins the alias used by ``kimi -m`` to the per-trial model.

    ``compaction_ratio`` → ``[loop_control].compaction_trigger_ratio``.
    kimi-cli clamps to [0.5, 0.99] (``src/kimi_cli/config.py``).
    ``reserved_context_size`` → ``[loop_control].reserved_context_size``;
    kimi-cli enforces ≥1000. Auto-compaction fires when either
    ``context_tokens >= max_context_size * compaction_trigger_ratio``
    **or** ``context_tokens + reserved_context_size >= max_context_size``
    (the unconditional trigger). To force compaction early during
    integration tests, lower ``max_context_size`` or raise
    ``reserved_context_size`` so the second condition fires sooner.

    ``max_context_size``/``reserved_context_size`` are ``None`` when the model
    did not declare them; we then omit those keys so kimi-cli applies its own
    built-in defaults instead of Cage inventing a window.
    """
    safe_ratio = max(0.5, min(0.99, float(compaction_ratio)))
    safe_max_ctx = (
        max(2000, int(max_context_size)) if max_context_size is not None else None
    )
    if reserved_context_size is None:
        safe_reserved: int | None = None
    elif safe_max_ctx is not None:
        safe_reserved = max(1000, min(safe_max_ctx - 1000, int(reserved_context_size)))
    else:
        safe_reserved = max(1000, int(reserved_context_size))
    provider_lines = [
        f"[providers.{_PROVIDER_NAME}]",
        'type = "openai_legacy"',
        f"base_url = {_toml_string(base_url)}",
        f"api_key = {_toml_string(api_key or 'EMPTY')}",
    ]
    if reasoning_key is not None:
        provider_lines.append(f"reasoning_key = {_toml_string(reasoning_key)}")

    header_lines: list[str] = []
    if isinstance(custom_headers, dict) and custom_headers:
        header_lines.append(f"[providers.{_PROVIDER_NAME}.custom_headers]")
        for key, value in sorted(custom_headers.items()):
            if value is None:
                continue
            header_lines.append(f"{_toml_string(key)} = {_toml_string(value)}")

    sections = [
        f"default_model = {_toml_string(_MODEL_ALIAS)}",
        "\n".join(provider_lines),
    ]
    if header_lines:
        sections.append("\n".join(header_lines))
    model_lines = [
        f'[models.{_MODEL_ALIAS}]',
        f"provider = {_toml_string(_PROVIDER_NAME)}",
        f"model = {_toml_string(model_name or '')}",
    ]
    if safe_max_ctx is not None:
        model_lines.append(f'max_context_size = {safe_max_ctx}')
    model_lines.append(
        f"display_name = {_toml_string(display_name or 'Cage proxied model')}"
    )
    sections.append("\n".join(model_lines))

    loop_lines = [
        '[loop_control]',
        f'compaction_trigger_ratio = {safe_ratio:.4f}',
    ]
    if safe_reserved is not None:
        loop_lines.append(f'reserved_context_size = {safe_reserved}')
    sections.append("\n".join(loop_lines))
    return "\n\n".join(sections) + "\n"


@register_agent_type
class KimiCodeAgent(AgentType):
    name = "kimi_code"
    state_paths = [".kimi"]
    default_image = "cage/kimi-code:pentestenv"
    dockerfile = "docker/kimi_code_pentestenv.Dockerfile"

    def install_command(self, version: str = "latest") -> str:
        """Install kimi-cli via uv (the documented path).

        The OSS one-liner (``curl -L code.kimi.com/install.sh | bash``)
        also installs uv first if absent. In our image uv is already
        present, so we collapse to the final ``uv tool install`` step.
        Version pinning uses uv's spec syntax (``kimi-cli==X.Y.Z``).
        """
        spec = "kimi-cli" if version in ("", "latest") else f"kimi-cli=={version}"
        return f"uv tool install --python 3.13 {spec}"

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        """Launch ``kimi`` in non-interactive print mode.

        ``--print`` enables headless mode (implicitly enabling
        ``--afk`` to auto-approve tool calls and dismiss
        ``AskUserQuestion``); ``--output-format stream-json`` emits
        JSONL Wire events on stdout so trajectories can be rebuilt;
        ``--max-steps-per-turn`` enforces Cage's round budget.

        ``-m <alias>`` points at the ``[models.<alias>]`` block we
        seed in ``setup_container``; the alias decouples Cage's
        per-trial model name from the user-facing CLI flag.
        """
        escaped = prompt.replace("'", "'\\''")
        max_flag = (
            f" --max-steps-per-turn {int(max_rounds)}" if max_rounds and max_rounds > 0 else ""
        )
        return (
            f"kimi --print --output-format stream-json"
            f"{max_flag}"
            f" -m {_MODEL_ALIAS}"
            f" -p '{escaped}'"
        )

    def parse_output(self, result: ExecResult) -> str:
        """Extract the final assistant text from ``stream-json`` stdout.

        Wire-format events are JSONL; the final assistant message has
        ``type == "assistant"`` (or ``role == "assistant"``) with a
        text/content field. ``--final-message-only`` would be cleaner
        but discards intermediate tool calls we want in trajectories.
        """
        banner = failure_banner(result)
        if banner is not None:
            return banner

        last_text = ""
        saw_json = False
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            saw_json = True
            if not isinstance(obj, dict):
                continue

            # Wire-format final result event
            if obj.get("type") == "result":
                value = obj.get("result") or obj.get("text") or ""
                if isinstance(value, str) and value.strip():
                    last_text = value
                continue

            # Assistant message events (shape varies across CLI versions)
            if obj.get("type") in ("assistant", "message") or obj.get("role") == "assistant":
                msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    last_text = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                last_text = text
                else:
                    text = msg.get("text", "")
                    if isinstance(text, str) and text.strip():
                        last_text = text

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
        """Refresh ``~/.kimi/config.toml`` for this trial, return env.

        The proxy port is per-trial, so we rewrite the provider
        ``base_url`` here every time. ``context_compaction_threshold``
        is forwarded to ``[loop_control].compaction_trigger_ratio`` so
        the Cage-level knob propagates to kimi's auto-compactor.
        ``KIMI_CLI_NO_AUTO_UPDATE`` suppresses the version-check
        network call (we don't want leaks past the proxy);
        ``KIMI_SHARE_DIR`` is unset so the CLI keeps using
        ``$HOME/.kimi``.
        """
        if container is not None:
            base_url = (
                proxy_url.rstrip("/") + "/v1" if proxy_url else (model.base_url or _DEFAULT_BASE_URL)
            )
            extra = model.extra if isinstance(model.extra, dict) else {}
            self._patch_config(
                container,
                home_dir=home_dir,
                base_url=base_url,
                api_key=model.api_key,
                model_name=model.model,
                compaction_ratio=(
                    _KIMI_DEFAULT_RATIO
                    if context_compaction_threshold is None
                    else context_compaction_threshold
                ),
                max_context_size=model.max_context_size,
                reserved_context_size=model.reserved_context_size,
                **_provider_extra_options(extra),
            )

        env: dict[str, str] = {"KIMI_CLI_NO_AUTO_UPDATE": "1"}
        return env

    def _patch_config(
        self, container: Any, *, home_dir: str, base_url: str,
        api_key: str, model_name: str, compaction_ratio: float = 0.85,
        max_context_size: int | None = None,
        reserved_context_size: int | None = None,
        reasoning_key: str | None = None,
        custom_headers: dict[str, Any] | None = None,
        display_name: str = "Cage proxied model",
    ) -> None:
        """Write a fresh ``~/.kimi/config.toml`` with this trial's values."""
        config_dir = f"{home_dir.rstrip('/')}/.kimi"
        config_path = f"{config_dir}/config.toml"
        container.exec(f"mkdir -p {config_dir}", timeout=5.0)
        container.write_file(
            config_path,
            _render_config_toml(
                base_url=base_url, api_key=api_key, model_name=model_name,
                compaction_ratio=compaction_ratio,
                max_context_size=max_context_size,
                reserved_context_size=reserved_context_size,
                reasoning_key=reasoning_key,
                custom_headers=custom_headers,
                display_name=display_name,
            ),
        )
        container.exec(f"chown -R agent:agent {config_dir}", timeout=5.0)

    def version_command(self) -> str:
        return "kimi --version 2>/dev/null || echo unknown"

    def setup_container(
        self, container: Any, *, home_dir: str,
        model: ModelConfig | None = None,
        context_compaction_threshold: float | None = None,
    ) -> None:
        """Seed ``~/.kimi/config.toml`` and disable auto-update side effects.

        The dynamic ``base_url`` (and the same compaction ratio) is
        overwritten per-trial in :meth:`env_vars`; we still seed a
        valid file at setup so the CLI can boot during the version
        probe and during ``cage debug``.
        """
        api_key = (model.api_key if model is not None else "") or "EMPTY"
        model_name = model.model if model is not None else ""
        extra = (model.extra if (model is not None and isinstance(model.extra, dict)) else {})
        config_dir = f"{home_dir.rstrip('/')}/.kimi"
        container.exec(f"mkdir -p {config_dir}", timeout=5.0)
        container.write_file(
            f"{config_dir}/config.toml",
            _render_config_toml(
                base_url=_DEFAULT_BASE_URL,
                api_key=api_key,
                model_name=model_name,
                compaction_ratio=(
                    _KIMI_DEFAULT_RATIO
                    if context_compaction_threshold is None
                    else context_compaction_threshold
                ),
                max_context_size=(model.max_context_size if model is not None else None),
                reserved_context_size=(
                    model.reserved_context_size if model is not None else None
                ),
                **_provider_extra_options(extra),
            ),
        )
        # Mark the no-auto-update flag in the kimi.json runtime metadata
        # so the CLI never probes upstream for new releases during runs.
        kimi_json = json.dumps({"auto_update": False}, indent=2)
        container.write_file(f"{config_dir}/kimi.json", kimi_json)
        container.exec(f"chown -R agent:agent {config_dir}", timeout=5.0)

    @property
    def protocol(self) -> str:
        return "openai"
