"""Tests for the Qwen Code agent config bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cage.agents.qwen_code import QwenCodeAgent
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


def test_qwen_agent_renders_generation_config_from_models_yml(tmp_path: Path) -> None:
    container = _FakeContainer()
    models_yml = tmp_path / "models.yml"
    models_yml.write_text(
        """
models:
  qwen3.6-max-preview:
    provider: openai
    model: qwen3.6-max-preview
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: sk-test
    max_context_size: 262144
    enable_thinking: true
    thinking_budget: 81920
    preserve_thinking: true
    custom_headers:
      X-Cage-Test: enabled
""",
        encoding="utf-8",
    )
    model = load_models(models_yml)["qwen3.6-max-preview"]

    env = QwenCodeAgent().env_vars(
        proxy_url="http://127.0.0.1:8877",
        model=model,
        container=container,
        home_dir="/home/agent",
    )

    settings = json.loads(container.files["/home/agent/.qwen/settings.json"])
    provider = settings["modelProviders"]["openai"][0]
    generation = provider["generationConfig"]

    assert env["QWEN_CODE_API_KEY"] == "sk-test"
    assert env["OPENAI_MODEL"] == "qwen3.6-max-preview"
    assert provider["id"] == "qwen3.6-max-preview"
    assert provider["baseUrl"] == "http://127.0.0.1:8877/v1"
    assert provider["envKey"] == "QWEN_CODE_API_KEY"
    assert settings["model"]["name"] == "qwen3.6-max-preview"
    assert generation["contextWindowSize"] == 262144
    assert generation["customHeaders"] == {"X-Cage-Test": "enabled"}
    assert generation["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 81920,
        "preserve_thinking": True,
    }
