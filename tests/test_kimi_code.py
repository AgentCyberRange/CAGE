"""Tests for the Kimi Code agent config bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cage.agents.kimi_code import KimiCodeAgent
from cage.models import load_models


class _FakeContainer:
    """Minimal container double that records writes."""

    def __init__(self) -> None:
        self.exec_calls: list[str] = []
        self.files: dict[str, str] = {}

    def exec(self, cmd: str, *_: Any, **__: Any) -> Any:
        self.exec_calls.append(cmd)

        class _Result:
            exit_code = 0
            stdout = ""
            stderr = ""

        return _Result()

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


def test_kimi_agent_renders_provider_extras_from_models_yml(tmp_path: Path) -> None:
    container = _FakeContainer()
    models_yml = tmp_path / "models.yml"
    models_yml.write_text(
        """
models:
  kimi-k2.6:
    provider: vllm
    model: Kimi-K2.6
    base_url: https://example.invalid/v1
    api_key: sk-test
    max_context_size: 262144
    reserved_context_size: 50000
    reasoning_key: reasoning
    display_name: Kimi K2.6 via Cage
    custom_headers:
      X-Cage-Test: enabled
      X-Quoted: value "with quotes"
""",
        encoding="utf-8",
    )
    model = load_models(models_yml)["kimi-k2.6"]

    env = KimiCodeAgent().env_vars(
        proxy_url="http://127.0.0.1:8877",
        model=model,
        container=container,
        home_dir="/home/agent",
    )

    cfg = container.files["/home/agent/.kimi/config.toml"]
    assert env == {"KIMI_CLI_NO_AUTO_UPDATE": "1"}
    assert 'base_url = "http://127.0.0.1:8877/v1"' in cfg
    assert 'model = "Kimi-K2.6"' in cfg
    assert 'max_context_size = 262144' in cfg
    assert 'display_name = "Kimi K2.6 via Cage"' in cfg
    assert 'reasoning_key = "reasoning"' in cfg
    assert "[providers.cage.custom_headers]" in cfg
    assert '"X-Cage-Test" = "enabled"' in cfg
    assert '"X-Quoted" = "value \\"with quotes\\""' in cfg
