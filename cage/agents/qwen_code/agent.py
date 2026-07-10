"""Qwen Code CLI agent type.

Upstream: https://github.com/QwenLM/qwen-code (binary: ``qwen``;
npm package ``@qwen-code/qwen-code``). Speaks OpenAI Chat Completions
wire format, configured via the ``OPENAI_*`` env vars or ``--openai-*``
CLI flags. We use the env-var path because it is the officially
documented configuration surface for headless / non-interactive runs
and matches Cage's existing pattern for proxy-routed OpenAI agents.

The official OSS install script (https://qwen-code-assets.oss-cn-hangzhou
.aliyuncs.com/installation/install-qwen.sh) installs NVM + Node + the
npm package and mutates the user's shell rc files; that's overkill for
a Docker layer with a deterministic Node base, so we shortcut to a
plain ``npm install -g``.
"""

from __future__ import annotations

import json
from typing import Any

from cage.agents.base.output import failure_banner
from cage.agents.base import AgentType, register_agent_type
from cage.models import ModelConfig
from cage.sandbox.exec import ExecResult


_API_KEY_ENV = "QWEN_CODE_API_KEY"
_DEFAULT_BASE_URL = "http://localhost:8877/v1"


def _qwen_generation_config(model: ModelConfig) -> dict[str, Any]:
    """Build Qwen Code generationConfig from model registry extras."""
    extra = model.extra if isinstance(model.extra, dict) else {}
    generation = extra.get("generation_config")
    if not isinstance(generation, dict):
        generation = extra.get("generationConfig")
    config = dict(generation) if isinstance(generation, dict) else {}

    # Declared window now lives on the typed ModelConfig field (the registry
    # accepts ``context_window_size`` as an alias and promotes it). Unset ⇒
    # leave qwen-code's own default contextWindowSize in place.
    if model.max_context_size is not None:
        config["contextWindowSize"] = int(model.max_context_size)

    custom_headers = extra.get("custom_headers", extra.get("customHeaders"))
    if isinstance(custom_headers, dict) and custom_headers:
        config["customHeaders"] = {
            str(key): str(value)
            for key, value in custom_headers.items()
            if value is not None
        }

    extra_body = extra.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = extra.get("extraBody")
    body = dict(extra_body) if isinstance(extra_body, dict) else {}
    for key in ("enable_thinking", "thinking_budget", "preserve_thinking"):
        if key in extra:
            body[key] = extra[key]
    if body:
        config["extra_body"] = body

    return config


def _qwen_settings(
    *,
    model: ModelConfig | None,
    base_url: str,
    context_compaction_threshold: float | None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "security": {"auth": {"selectedType": "openai"}},
        "telemetry": {"enabled": False},
        "general": {"disableUpdateNag": True},
        "model": {},
    }
    # Only pin qwen-code's chat-compression threshold when the user opted in;
    # unset ⇒ leave qwen-code's own built-in default in place.
    if context_compaction_threshold is not None:
        threshold = max(0.0, min(1.0, float(context_compaction_threshold)))
        settings["model"]["chatCompression"] = {
            "contextPercentageThreshold": threshold,
        }
    if model is None:
        return settings

    generation_config = _qwen_generation_config(model)
    provider: dict[str, Any] = {
        "id": model.model,
        "name": model.model,
        "baseUrl": base_url,
        "description": f"{model.model} via Cage",
        "envKey": _API_KEY_ENV,
    }
    if generation_config:
        provider["generationConfig"] = generation_config

    settings["model"]["name"] = model.model
    settings["modelProviders"] = {"openai": [provider]}
    return settings


@register_agent_type
class QwenCodeAgent(AgentType):
    name = "qwen_code"
    state_paths = [".qwen"]
    default_image = "cage/qwen-code:pentestenv"
    dockerfile = "docker/qwen_code/pentestenv.Dockerfile"

    def install_command(self, version: str = "latest") -> str:
        return f"npm install -g @qwen-code/qwen-code@{version}"

    def build_launch_command(
        self, prompt: str, *, model: ModelConfig, max_rounds: int = -1, proxy_url: str = "",
    ) -> str:
        """Launch ``qwen`` in non-interactive single-turn mode.

        ``--yolo`` auto-approves tool calls (equivalent to
        ``--approval-mode=yolo``); ``--output-format json`` produces a
        structured stdout we can parse; ``--max-session-turns`` enforces
        Cage's per-trial round budget (the CLI defaults to ``-1``,
        i.e. unbounded, so passing it conditionally keeps the framework
        contract that negative ``max_rounds`` means unset).

        Caveat: qwen-code's "session turn" counter is **not** strictly
        1:1 with LLM API calls — a single session turn can issue
        multiple LLM requests when the model emits parallel tool calls
        that fan back through the loop. The proxy's ``max_requests``
        backstop (set to the same ``max_rounds`` value by the
        orchestrator) is therefore the hard cap; qwen-code retries
        aggressively on HTTP 429, so prefer raising both limits together
        rather than leaving the agent CLI's limit higher than the
        proxy's.
        """
        escaped = prompt.replace("'", "'\\''")
        max_flag = (
            f" --max-session-turns {int(max_rounds)}" if max_rounds and max_rounds > 0 else ""
        )
        return (
            f"qwen --yolo --output-format json"
            f"{max_flag}"
            f" -p '{escaped}'"
        )

    def parse_output(self, result: ExecResult) -> str:
        """Extract final assistant text from ``--output-format json`` stdout.

        The JSON shape for ``--output-format json`` is a single object
        with a ``response`` (or similar) field. Schemas have churned
        across versions, so we try a list of known fields, then fall
        back to scanning JSON lines for assistant message text, then to
        raw stdout.
        """
        banner = failure_banner(result)
        if banner is not None:
            return banner

        stdout = result.stdout.strip()
        if not stdout:
            return ""

        # Single-object JSON (the documented shape for --output-format json)
        try:
            obj = json.loads(stdout)
        except json.JSONDecodeError:
            obj = None

        if isinstance(obj, dict):
            for key in ("response", "result", "final_response", "output", "text"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            messages = obj.get("messages") or obj.get("history")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    return text

        # Fallback: NDJSON scan (in case the user passed stream-json or the
        # CLI emits multiple lines)
        last_text = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("type") in ("assistant", "message"):
                msg = ev.get("message") or ev
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if isinstance(content, str) and content.strip():
                    last_text = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                last_text = text
            elif ev.get("type") == "result":
                value = ev.get("result", "")
                if value:
                    last_text = value
        if last_text:
            return last_text

        return stdout[:4000]

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
        """Wire qwen-code's OpenAI client at the in-container proxy.

        ``proxy_url`` lacks a path; the OpenAI client expects ``/v1`` so
        we append it. Telemetry env vars match qwen-code defaults
        (the CLI doesn't ship a single global telemetry kill, but env
        ``QWEN_DEBUG=0`` keeps logs quiet).
        """
        env: dict[str, str] = {}
        base = ""
        if proxy_url:
            base = proxy_url.rstrip("/") + "/v1"
        elif model.base_url:
            base = model.base_url
        if container is not None:
            self._patch_settings(
                container,
                home_dir=home_dir,
                base_url=base or _DEFAULT_BASE_URL,
                model=model,
                context_compaction_threshold=context_compaction_threshold,
            )
        if base:
            env["OPENAI_BASE_URL"] = base
        if model.api_key:
            env["OPENAI_API_KEY"] = model.api_key
            env[_API_KEY_ENV] = model.api_key
        if model.model:
            env["OPENAI_MODEL"] = model.model
        return env

    def _patch_settings(
        self,
        container: Any,
        *,
        home_dir: str,
        base_url: str,
        model: ModelConfig | None,
        context_compaction_threshold: float | None,
    ) -> None:
        """Write a fresh ``~/.qwen/settings.json`` for this trial."""
        qwen_dir = f"{home_dir.rstrip('/')}/.qwen"
        container.exec(f"mkdir -p {qwen_dir}", timeout=5.0)
        container.write_file(
            f"{qwen_dir}/settings.json",
            json.dumps(
                _qwen_settings(
                    model=model,
                    base_url=base_url,
                    context_compaction_threshold=context_compaction_threshold,
                ),
                indent=2,
            ),
        )
        container.exec(f"chown -R agent:agent {qwen_dir}", timeout=5.0)

    def setup_container(
        self, container: Any, *, home_dir: str,
        model: ModelConfig | None = None,
        context_compaction_threshold: float | None = None,
    ) -> None:
        """Seed ``~/.qwen/settings.json`` so the CLI skips first-launch UX.

        Without this, ``qwen -p ...`` may print a one-time
        "telemetry consent" / "theme selection" prompt on stdout that
        contaminates ``--output-format json`` parsing. The settings file
        also lets us pre-select OpenAI auth so non-interactive mode
        doesn't bounce to the OAuth flow when ``OPENAI_API_KEY`` is
        present.

        ``context_compaction_threshold`` (0.0-1.0) is forwarded to
        qwen-code's ``model.chatCompression.contextPercentageThreshold``
        — the auto-compaction trigger documented at
        https://qwenlm.github.io/qwen-code-docs/en/users/configuration/settings/
        (default 0.7). Pass 0 to disable auto-compaction entirely;
        ``/compress`` still works manually because it passes
        ``force=true`` internally.
        """
        self._patch_settings(
            container,
            home_dir=home_dir,
            base_url=model.base_url if model is not None and model.base_url else _DEFAULT_BASE_URL,
            model=model,
            context_compaction_threshold=context_compaction_threshold,
        )

    def version_command(self) -> str:
        return "qwen --version 2>/dev/null || echo unknown"

    @property
    def protocol(self) -> str:
        return "openai"
