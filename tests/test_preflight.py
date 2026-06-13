from __future__ import annotations

from types import SimpleNamespace

from cage.contracts.logging import LoggingConfig
from cage.experiment.engine import preflight


def test_run_preflight_terminal_ui_prints_plain_status(capsys):
    config = SimpleNamespace(
        agents=[],
        metadata={},
        logging=LoggingConfig(terminal_ui=True),
    )

    result = preflight.run_preflight(config)

    output = capsys.readouterr().err
    assert result.passed == 0
    assert "Pre-flight checks" in output
    assert "Ready: 0 passed, 0 warnings" in output
    assert "[info" not in output
    assert "PRE-FLIGHT CHECKS" not in output


def test_run_preflight_terminal_ui_can_emit_ansi_color(capsys, monkeypatch):
    monkeypatch.setenv("CAGE_COLOR", "always")
    config = SimpleNamespace(
        agents=[],
        metadata={},
        logging=LoggingConfig(terminal_ui=True),
    )

    preflight.run_preflight(config)

    output = capsys.readouterr().err
    assert "\033[" in output
    assert "Pre-flight checks" in output
    assert "Ready: 0 passed, 0 warnings" in output


def test_run_preflight_skips_model_chain_when_zero_rounds(monkeypatch):
    calls: list[str] = []

    def fake_check_image(_image, result):
        calls.append("image")
        preflight._pass("Docker image", result)

    def fake_check_container_boot(_image, _network_mode, result):
        calls.append("boot")
        preflight._pass("Container boot", result)

    def fail_model_chain(*_args, **_kwargs):
        raise AssertionError("zero-round runs must not probe the model")

    model = SimpleNamespace(
        base_url="http://127.0.0.1:1/v1",
        api_key="test-key",
        protocol="openai",
        model="demo-model",
        id="demo",
        auth_source="",
    )
    agent = SimpleNamespace(
        effective_image="demo-image",
        model=model,
        max_rounds=-1,
        agent_type=SimpleNamespace(name="codex"),
    )
    config = SimpleNamespace(
        agents=[agent],
        metadata={},
        logging=LoggingConfig(terminal_ui=True),
        execution=SimpleNamespace(agent_network_mode="bridge", max_rounds=0),
        proxy=SimpleNamespace(enabled=True, upstream_http_proxy=""),
    )

    monkeypatch.setattr(preflight, "_check_image", fake_check_image)
    monkeypatch.setattr(preflight, "_check_container_boot", fake_check_container_boot)
    monkeypatch.setattr(preflight, "_check_proxy_chain", fail_model_chain)
    monkeypatch.setattr(preflight, "_check_model_api", fail_model_chain)

    result = preflight.run_preflight(config, samples=[{"id": "sample", "max_rounds": 150}])

    assert calls == ["image", "boot"]
    assert result.failed == 0
    assert result.warnings == 1
    assert any("max_rounds=0" in detail for detail in result.details)


def test_proxy_chain_uses_configured_upstream_http_proxy(monkeypatch):
    docker_runs: list[list[str]] = []
    copied_config: dict[str, object] = {}
    container_name = ""

    def fake_run_cmd(cmd, timeout=None):
        nonlocal container_name
        if cmd[:3] == ["docker", "run", "-d"]:
            docker_runs.append(cmd)
            container_name = cmd[cmd.index("--name") + 1]
            return 0, "container-id", ""
        if cmd[:3] == ["docker", "rm", "-f"]:
            return 0, "", ""
        if cmd[:2] == ["docker", "cp"] and cmd[3] == f"{container_name}:/tmp/pc.json":
            import json
            from pathlib import Path

            copied_config.update(json.loads(Path(cmd[2]).read_text(encoding="utf-8")))
            return 0, "", ""
        if cmd[:3] == ["docker", "exec", container_name]:
            if cmd[-1].endswith("/healthz"):
                return 0, "ok", ""
            return 0, "", ""
        return 0, "", ""

    def fake_sleep(_seconds):
        return None

    model = SimpleNamespace(
        base_url="http://example.invalid/v1",
        api_key="test-key",
        protocol="openai",
        model="demo-model",
        id="demo",
    )
    result = preflight.PreflightResult()

    monkeypatch.setattr(preflight.time, "time", lambda: 12345)
    monkeypatch.setattr(preflight, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(preflight.time, "sleep", fake_sleep)

    preflight._check_proxy_chain(
        "demo-image",
        model,
        "bridge",
        result,
        upstream_http_proxy="http://host.docker.internal:7890",
    )

    assert docker_runs, "expected docker run to be invoked"
    assert "--add-host" in docker_runs[0]
    assert "host.docker.internal:host-gateway" in docker_runs[0]
    assert copied_config["http_proxy"] == "http://host.docker.internal:7890"


def test_validate_proxy_response_detects_proxy_error_body():
    # Real failure body from container_proxy.py on DNS / upstream failure.
    body = (
        '{"error": {"message": "[Errno -2] Name or service not known", '
        '"type": "proxy_error"}}'
    )
    ok, reason = preflight._validate_proxy_response(body, curl_exit=0)
    assert ok is False
    assert "Name or service not known" in reason


def test_validate_proxy_response_detects_upstream_error_body():
    # Real failure body from zai/bigmodel.cn upstream returning HTTP 400.
    body = '{"error": {"message": "Bad request", "code": 400}}'
    ok, reason = preflight._validate_proxy_response(body, curl_exit=0)
    assert ok is False
    assert "Bad request" in reason


def test_validate_proxy_response_accepts_real_completion():
    body = (
        '{"id": "cmpl-1", "choices": [{"message": '
        '{"role": "assistant", "content": "hi"}}], "model": "x"}'
    )
    ok, reason = preflight._validate_proxy_response(body, curl_exit=0)
    assert ok is True
    assert reason == ""


def test_validate_proxy_response_rejects_curl_failure():
    ok, reason = preflight._validate_proxy_response("", curl_exit=7)
    assert ok is False
    assert "curl" in reason


def test_validate_proxy_response_rejects_non_json():
    ok, reason = preflight._validate_proxy_response("upstream is down", curl_exit=0)
    assert ok is False
    assert "Non-JSON" in reason or "Empty" in reason


def test_validate_proxy_response_accepts_sse_stream():
    # Real /v1/responses SSE shape from a codex-relay endpoint.
    body = (
        'data: {"type": "response.created", "response": {"id": "resp_1"}}\n\n'
        'data: {"type": "response.output_text.delta", "delta": "hi"}\n\n'
        'data: {"type": "response.completed"}\n\n'
        'data: [DONE]\n\n'
    )
    ok, reason = preflight._validate_proxy_response(body, curl_exit=0)
    assert ok is True, reason


def test_validate_proxy_response_rejects_sse_error_event():
    body = (
        'data: {"type": "response.created", "response": {"id": "resp_1"}}\n\n'
        'data: {"type": "error", "error": '
        '{"message": "Stream must be set to true", "code": 400}}\n\n'
    )
    ok, reason = preflight._validate_proxy_response(body, curl_exit=0)
    assert ok is False
    assert "Stream must be set to true" in reason


def test_validate_proxy_response_rejects_sse_with_no_events():
    body = "data: \n\ndata: [DONE]\n\n"
    ok, reason = preflight._validate_proxy_response(body, curl_exit=0)
    assert ok is False
    assert "no parseable events" in reason
