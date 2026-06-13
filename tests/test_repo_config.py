from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from cage.cli import main
from cage.config import load_repo_config
from cage.models import load_models

ROOT = Path(__file__).resolve().parents[1]


def test_load_repo_config_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_repo_config(tmp_path)

    assert config.models_file == "config/models.yml"
    assert config.web_inspector.host == "0.0.0.0"
    # The shared inspector port defaults to 7777 (not an ephemeral 0) so a
    # managed board started without a local config/cage.yml still lands there.
    assert config.web_inspector.port == 7777
    assert config.web_inspector.open_browser is True
    assert config.web_inspector.ui.run_filters_open is True
    assert config.web_inspector.ui.trial_filters_open is True
    assert config.web_inspector.ui.default_min_run_duration_ms == 0
    assert config.web_inspector.ui.default_min_trial_duration_ms == 0
    assert config.web_inspector.auth.enabled is False
    assert config.web_inspector.auth.token == ""


def test_tracked_models_example_is_complete_and_public_safe() -> None:
    example = ROOT / "config" / "models.example.yml"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    models = raw["models"]
    expected_fields = {
        "provider",
        "model",
        "agent_model_names",
        "base_url",
        "api_key",
        "auth_source",
        "api_keys",
        "input_cost_per_1m",
        "output_cost_per_1m",
        "timeout",
        "max_retries",
        "extra_headers",
    }

    assert "gpt-5.5" in models
    assert "deepseek-v4-pro" in models
    assert "qwen3.7-max" in models
    for model_id, entry in models.items():
        assert expected_fields.issubset(entry), model_id
        assert entry["provider"] in {"openai", "anthropic", "vllm"}
        assert entry["model"]
        assert isinstance(entry["api_keys"], list), model_id
        assert isinstance(entry["agent_model_names"], dict), model_id
        assert isinstance(entry["extra_headers"], dict), model_id
        if not entry["auth_source"]:
            assert entry["base_url"], model_id
            assert str(entry["api_key"]).startswith("${"), model_id

    text = example.read_text(encoding="utf-8")
    assert "fbpuQ/" not in text
    assert "stpmj/" not in text


def test_load_repo_config_reads_web_inspector_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAGE_INSPECTOR_TOKEN", "local-secret")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        """
web_inspector:
  host: 127.0.0.1
  port: 7777
  open_browser: false
  ui:
    run_filters_open: false
    trial_filters_open: true
    default_min_run_duration_ms: 1800000
    default_min_trial_duration_ms: 480000
  auth:
    enabled: true
    token: ${CAGE_INSPECTOR_TOKEN}
""",
        encoding="utf-8",
    )

    config = load_repo_config(tmp_path)

    assert config.web_inspector.host == "127.0.0.1"
    assert config.web_inspector.port == 7777
    assert config.web_inspector.open_browser is False
    assert config.web_inspector.ui.run_filters_open is False
    assert config.web_inspector.ui.trial_filters_open is True
    assert config.web_inspector.ui.default_min_run_duration_ms == 1800000
    assert config.web_inspector.ui.default_min_trial_duration_ms == 480000
    assert config.web_inspector.auth.enabled is True
    assert config.web_inspector.auth.token == "local-secret"


def test_load_repo_config_reads_default_models_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        """
models_file: config/models.local.yml
""",
        encoding="utf-8",
    )

    config = load_repo_config(tmp_path)

    assert config.models_file == "config/models.local.yml"


def test_load_models_expands_environment_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAGE_TEST_MODEL_KEY", "model-secret")
    models_file = tmp_path / "models.yml"
    models_file.write_text(
        """
models:
  demo:
    provider: openai
    model: demo-model
    base_url: https://example.test/v1
    api_key: ${CAGE_TEST_MODEL_KEY}
""",
        encoding="utf-8",
    )

    models = load_models(models_file)

    assert models["demo"].api_key == "model-secret"


def test_load_models_separates_endpoint_model_from_agent_specific_cli_names(
    tmp_path: Path,
) -> None:
    models_file = tmp_path / "models.yml"
    models_file.write_text(
        """
models:
  deepseek-v4-pro:
    provider: anthropic
    model: deepseek-v4-pro
    base_url: https://api.deepseek.com/anthropic
    api_key: sk-test
    agent_model_names:
      claude_code: deepseek-v4-pro[1m]
""",
        encoding="utf-8",
    )

    model = load_models(models_file)["deepseek-v4-pro"]

    assert model.model == "deepseek-v4-pro"
    assert model.model_name_for_agent("claude_code") == "deepseek-v4-pro[1m]"
    assert model.model_name_for_agent("codex") == "deepseek-v4-pro"


def test_strongreject_uses_repo_local_models_config() -> None:
    strongreject = ROOT / "examples" / "strongreject"

    assert not (strongreject / "models.yaml").exists()
    assert not (strongreject / "models.yml").exists()

    for name in ("default_strongreject.yml",):
        text = (strongreject / name).read_text(encoding="utf-8")
        assert "models_file:" not in text
        assert "glm-5.1-sii" not in text


def test_inspect_command_uses_repo_local_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAGE_INSPECTOR_TOKEN", "local-secret")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        """
web_inspector:
  host: 127.0.0.1
  port: 7777
  open_browser: false
  auth:
    enabled: true
    token: ${CAGE_INSPECTOR_TOKEN}
""",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeApp:
        def run(self, *, host: str, port: int, debug: bool, threaded: bool = False) -> None:
            calls["host"] = host
            calls["port"] = port
            calls["debug"] = debug

    def fake_create_app(root: Path, *, auth, ui):
        calls["root"] = root
        calls["auth_enabled"] = auth.enabled
        calls["auth_token"] = auth.token
        calls["default_min_run_duration_ms"] = ui.default_min_run_duration_ms
        calls["default_min_trial_duration_ms"] = ui.default_min_trial_duration_ms
        return FakeApp()

    monkeypatch.setattr("cage.web.app.create_app", fake_create_app)
    # Single-port policy: serve_inspector now refuses a busy port. Keep the
    # config-wiring assertions independent of whatever holds 7777 locally.
    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda *a, **k: True)

    result = CliRunner().invoke(main, ["inspect", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 7777
    assert calls["debug"] is False
    assert calls["root"] == tmp_path.resolve()
    assert calls["auth_enabled"] is True
    assert calls["auth_token"] == "local-secret"
    assert calls["default_min_run_duration_ms"] == 0
    assert calls["default_min_trial_duration_ms"] == 0


def test_repo_default_config_binds_inspector_public_without_auth() -> None:
    config = load_repo_config(ROOT)

    assert config.web_inspector.host == "0.0.0.0"
    assert config.web_inspector.ui.run_filters_open is True
    assert config.web_inspector.ui.trial_filters_open is True
    assert config.web_inspector.ui.default_min_run_duration_ms == 0
    assert config.web_inspector.ui.default_min_trial_duration_ms == 0
    assert config.web_inspector.auth.enabled is False
    assert config.web_inspector.auth.token == ""


def test_inspect_command_allows_public_bind_without_auth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "cage.yml").write_text(
        """
web_inspector:
  host: 0.0.0.0
  port: 7777
  open_browser: false
  auth:
    enabled: false
    token: ""
""",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeApp:
        def run(self, *, host: str, port: int, debug: bool, threaded: bool = False) -> None:
            calls["host"] = host
            calls["port"] = port
            calls["debug"] = debug

    def fake_create_app(root: Path, *, auth, ui):
        calls["root"] = root
        calls["auth_enabled"] = auth.enabled
        calls["auth_token"] = auth.token
        calls["default_min_run_duration_ms"] = ui.default_min_run_duration_ms
        calls["default_min_trial_duration_ms"] = ui.default_min_trial_duration_ms
        return FakeApp()

    monkeypatch.setattr("cage.web.app.create_app", fake_create_app)
    # Single-port policy: serve_inspector now refuses a busy port. Keep the
    # config-wiring assertions independent of whatever holds 7777 locally.
    monkeypatch.setattr("cage.web.inspect_board.port_is_free", lambda *a, **k: True)

    result = CliRunner().invoke(main, ["inspect", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 7777
    assert calls["debug"] is False
    assert calls["root"] == tmp_path.resolve()
    assert calls["auth_enabled"] is False
    assert calls["auth_token"] == ""
    assert calls["default_min_run_duration_ms"] == 0
    assert calls["default_min_trial_duration_ms"] == 0
