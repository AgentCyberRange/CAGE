from __future__ import annotations

import yaml
from click.testing import CliRunner

from cage.cli import main


def test_model_set_uses_repo_config_default_models_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        "models_file: config/models.yml\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "model",
            "set",
            "local-test",
            "--provider",
            "openai",
            "--model",
            "local-model",
            "--endpoint",
            "http://127.0.0.1:8000/v1",
            "--api-key",
            "sk-local",
            "--input-cost-per-1m",
            "1.25",
            "--output-cost-per-1m",
            "3.5",
        ],
    )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load((config_dir / "models.yml").read_text(encoding="utf-8"))
    entry = data["models"]["local-test"]
    assert entry["provider"] == "openai"
    assert entry["model"] == "local-model"
    assert entry["agent_model_names"] == {}
    assert entry["base_url"] == "http://127.0.0.1:8000/v1"
    assert entry["api_key"] == "sk-local"
    assert entry["auth_source"] == ""
    assert entry["api_keys"] == []
    assert entry["input_cost_per_1m"] == 1.25
    assert entry["output_cost_per_1m"] == 3.5
    assert entry["timeout"] == 360
    assert entry["max_retries"] == 2
    assert entry["extra_headers"] == {}


def test_model_list_masks_api_key_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        "models_file: config/models.yml\n",
        encoding="utf-8",
    )
    (config_dir / "models.yml").write_text(
        """
models:
  demo:
    provider: openai
    model: demo-model
    base_url: https://api.example.test/v1
    api_key: sk-secret-value
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["model", "list"])

    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert "openai" in result.output
    assert "sk-secret-value" not in result.output
    assert "sk-...alue" in result.output


def test_model_set_can_write_agent_specific_model_aliases(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        "models_file: config/models.yml\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "model",
            "set",
            "deepseek-v4-pro",
            "--provider",
            "anthropic",
            "--model",
            "deepseek-v4-pro",
            "--endpoint",
            "https://api.deepseek.com/anthropic",
            "--api-key",
            "${DEEPSEEK_API_KEY}",
            "--agent-model-name",
            "claude_code=deepseek-v4-pro[1m]",
        ],
    )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load((config_dir / "models.yml").read_text(encoding="utf-8"))
    entry = data["models"]["deepseek-v4-pro"]
    assert entry["model"] == "deepseek-v4-pro"
    assert entry["agent_model_names"] == {
        "claude_code": "deepseek-v4-pro[1m]",
    }
