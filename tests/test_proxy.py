"""Tests for the proxy translation layer."""

import gzip
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from cage.proxy import sidecar as container_proxy
from cage.proxy.host import (
    ContainerProxyInstance,
    ProxyInstance,
    ProxyInstanceConfig,
    ProxyModifyRule,
    _apply_modify_rules,
    _build_openai_request,
    _translate_messages_anthropic_to_openai,
    _translate_response_openai_to_anthropic,
    _translate_tool_choice,
    _translate_tools_anthropic_to_openai,
    start_container_proxy,
)
from cage.sandbox.containers import Container
from cage.sandbox.exec import ExecResult


class TestToolTranslation:
    def test_basic_tool(self):
        tools = [{"name": "bash", "description": "Run bash", "input_schema": {"type": "object"}}]
        result = _translate_tools_anthropic_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"

    def test_multiple_tools(self):
        tools = [
            {"name": "read", "description": "Read file", "input_schema": {"type": "object"}},
            {"name": "write", "description": "Write file", "input_schema": {"type": "object"}},
        ]
        result = _translate_tools_anthropic_to_openai(tools)
        assert len(result) == 2


class TestMessageTranslation:
    def test_simple_text(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_tool_use_blocks(self):
        msgs = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "ls"}},
            ],
        }]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Let me check"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "bash"

    def test_tool_result_blocks(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt"},
            ],
        }]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tu_1"

    def test_tool_result_and_text_ordering(self):
        # A `tool` message must directly follow the assistant tool_calls turn;
        # any accompanying text (e.g. a system-reminder Claude Code injects)
        # goes *after* it, never wedged in between.
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "out"},
                {"type": "text", "text": "<system-reminder>keep going</system-reminder>"},
            ]},
        ]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert [m["role"] for m in result] == ["assistant", "tool", "user"]
        assert result[1]["tool_call_id"] == "tu_1"
        assert "system-reminder" in result[2]["content"]

    def test_document_block_gets_marker_not_dropped(self):
        # A base64 PDF the agent Read must not vanish silently.
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "PDF read"},
            {"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf", "data": "JVBERi0x" * 100}},
        ]}]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert [m["role"] for m in result] == ["tool", "user"]
        assert "application/pdf" in result[1]["content"]
        assert "omitted" in result[1]["content"]

    def test_image_in_tool_result_gets_marker(self):
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}},
            ]},
        ]}]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert result[0]["role"] == "tool"
        assert "image/png" in result[0]["content"]

    def test_inline_text_document_is_unwrapped(self):
        msgs = [{"role": "user", "content": [
            {"type": "document", "source": {
                "type": "text", "media_type": "text/plain", "data": "hello from doc"}},
        ]}]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert result[0]["content"] == "hello from doc"

    def test_thinking_block_is_dropped(self):
        msgs = [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "secret reasoning"},
            {"type": "text", "text": "answer"},
        ]}]
        result = _translate_messages_anthropic_to_openai(msgs)
        assert result == [{"role": "assistant", "content": "answer"}]


class TestResponseTranslation:
    def test_text_response(self):
        resp = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = _translate_response_openai_to_anthropic(
            request_id="req-1", model="llama", response=resp
        )
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello"
        assert result["stop_reason"] == "end_turn"

    def test_tool_calls_response(self):
        resp = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "tc_1",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = _translate_response_openai_to_anthropic(
            request_id="req-1", model="llama", response=resp
        )
        assert result["stop_reason"] == "tool_use"
        tool_use = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_use) == 1
        assert tool_use[0]["name"] == "bash"
        assert tool_use[0]["input"] == {"command": "ls"}

    def test_tool_call_in_reasoning_content_is_recovered(self):
        # Thinking models w/o a tool parser emit <tool_call> XML inside
        # reasoning_content with an empty content field — must not be lost.
        resp = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content":
                        '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>',
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = _translate_response_openai_to_anthropic(
            request_id="req-1", model="qwen", response=resp
        )
        assert result["stop_reason"] == "tool_use"
        tool_use = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_use) == 1
        assert tool_use[0]["name"] == "bash"
        assert tool_use[0]["input"] == {"command": "ls"}

    def test_usage_cache_read_tokens_passthrough(self):
        resp = {
            "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }
        result = _translate_response_openai_to_anthropic(
            request_id="req-1", model="llama", response=resp
        )
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 20
        assert result["usage"]["cache_read_input_tokens"] == 80

    def test_string_tool_call_response_becomes_tool_use(self):
        resp = {
            "choices": [{
                "message": {
                    "content": (
                        "Let me inspect the workspace.\n\n"
                        "<tool_call>"
                        '{"name": "Bash", "arguments": {"command": "pwd"}}'
                        "</tool_call>"
                    ),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        for translator in (
            _translate_response_openai_to_anthropic,
            container_proxy._translate_response_openai_to_anthropic,
        ):
            result = translator(request_id="req-1", model="llama", response=resp)

            assert result["stop_reason"] == "tool_use"
            assert result["content"][0] == {
                "type": "text",
                "text": "Let me inspect the workspace.",
            }
            assert result["content"][1]["type"] == "tool_use"
            assert result["content"][1]["name"] == "Bash"
            assert result["content"][1]["input"] == {"command": "pwd"}

    def test_qwen_xml_tool_call_response_becomes_tool_use(self):
        """Qwen/Hermes-served models emit tool calls as ``<function=…><parameter=…>``
        XML inside the ``<tool_call>`` wrapper rather than JSON. The parser
        must handle this or every agentic trial against such a model
        terminates after round 1 with an empty tool_use list (real failure
        captured on qwen36-27b + cvebench).
        """
        resp = {
            "choices": [{
                "message": {
                    "content": (
                        "Reconnaissance time.\n"
                        "<tool_call>\n"
                        "<function=Bash>\n"
                        "<parameter=command>\n"
                        "curl -s http://target:9090/ | head -200\n"
                        "</parameter>\n"
                        "<parameter=timeout>\n"
                        "15000\n"
                        "</parameter>\n"
                        "<parameter=description>\n"
                        "Initial recon\n"
                        "</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    ),
                },
                "finish_reason": "stop",
            }],
        }

        for translator in (
            _translate_response_openai_to_anthropic,
            container_proxy._translate_response_openai_to_anthropic,
        ):
            result = translator(request_id="req-1", model="qwen36-27b", response=resp)

            assert result["stop_reason"] == "tool_use"
            assert result["content"][0] == {
                "type": "text",
                "text": "Reconnaissance time.",
            }
            assert result["content"][1]["type"] == "tool_use"
            assert result["content"][1]["name"] == "Bash"
            cmd = result["content"][1]["input"]["command"]
            # Command preserved verbatim (string), surrounding whitespace stripped.
            assert cmd.strip() == "curl -s http://target:9090/ | head -200"
            # Numeric param coerced to int so Claude Code's schema validates.
            assert result["content"][1]["input"]["timeout"] == 15000
            assert result["content"][1]["input"]["description"].strip() == "Initial recon"

    def test_qwen_xml_tool_call_multiple_parallel_calls(self):
        """Qwen routinely emits multiple back-to-back ``<tool_call>`` blocks
        for parallel work — all must survive translation.
        """
        resp = {
            "choices": [{
                "message": {
                    "content": (
                        "<tool_call>\n<function=Bash>\n"
                        "<parameter=command>\nls /tmp\n</parameter>\n"
                        "</function>\n</tool_call>\n"
                        "<tool_call>\n<function=Read>\n"
                        "<parameter=file_path>\n/etc/hostname\n</parameter>\n"
                        "<parameter=limit>\n10\n</parameter>\n"
                        "</function>\n</tool_call>"
                    ),
                },
                "finish_reason": "stop",
            }],
        }
        result = container_proxy._translate_response_openai_to_anthropic(
            request_id="req-2", model="qwen36-27b", response=resp,
        )
        tool_uses = [b for b in result["content"] if b["type"] == "tool_use"]
        assert [tu["name"] for tu in tool_uses] == ["Bash", "Read"]
        assert tool_uses[1]["input"]["limit"] == 10  # int coercion

    def test_string_tool_call_response_tolerates_extra_trailing_brace(self):
        resp = {
            "choices": [{
                "message": {
                    "content": (
                        "<tool_call>\n"
                        '{"name": "Bash", "arguments": '
                        "{\"command\": \"r2 ./maze_public -c 'aaa'\"}, "
                        '"description": "Analyze all in radare2"}}\n'
                        "</tool_call>"
                    ),
                },
                "finish_reason": "stop",
            }],
        }

        for translator in (
            _translate_response_openai_to_anthropic,
            container_proxy._translate_response_openai_to_anthropic,
        ):
            result = translator(request_id="req-1", model="llama", response=resp)

            assert result["stop_reason"] == "tool_use"
            assert result["content"] == [{
                "type": "tool_use",
                "id": "req-1-text-tool-0",
                "name": "Bash",
                "input": {"command": "r2 ./maze_public -c 'aaa'"},
            }]


class TestModifyRules:
    def test_append(self):
        orig, mod = _apply_modify_rules("original system", [
            ProxyModifyRule(target="system_prompt", rule="append", content="Be helpful."),
        ])
        assert orig == "original system"
        assert mod == "original system\n\nBe helpful."

    def test_prepend(self):
        _, mod = _apply_modify_rules("original", [
            ProxyModifyRule(target="system_prompt", rule="prepend", content="PREFIX"),
        ])
        assert mod == "PREFIX\n\noriginal"

    def test_replace(self):
        _, mod = _apply_modify_rules("original", [
            ProxyModifyRule(target="system_prompt", rule="replace", content="new system"),
        ])
        assert mod == "new system"

    def test_non_system_target_ignored(self):
        orig, mod = _apply_modify_rules("original", [
            ProxyModifyRule(target="other_field", rule="append", content="ignored"),
        ])
        assert mod == "original"


class TestBuildOpenaiRequest:
    def test_basic(self):
        anthropic_req = {
            "model": "claude-3",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1024,
        }
        payload, orig, mod = _build_openai_request(anthropic_req, modify_rules=[])
        assert payload["model"] == "claude-3"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["max_tokens"] == 1024

    def test_with_modify_rules(self):
        anthropic_req = {
            "model": "claude-3",
            "system": "Original",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        rules = [ProxyModifyRule(target="system_prompt", rule="append", content="Extra.")]
        payload, orig, mod = _build_openai_request(anthropic_req, modify_rules=rules)
        assert orig == "Original"
        assert mod == "Original\n\nExtra."
        assert payload["messages"][0]["content"] == "Original\n\nExtra."

    def test_with_tools(self):
        anthropic_req = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "bash", "description": "Run bash", "input_schema": {"type": "object"}}],
        }
        payload, _, _ = _build_openai_request(anthropic_req, modify_rules=[])
        assert "tools" in payload
        assert payload["tools"][0]["type"] == "function"

    def test_tool_choice_and_stop_sequences_translated(self):
        anthropic_req = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "bash", "description": "", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "bash"},
            "stop_sequences": ["\n\nHuman:"],
        }
        payload, _, _ = _build_openai_request(anthropic_req, modify_rules=[])
        assert payload["tool_choice"] == {"type": "function", "function": {"name": "bash"}}
        assert payload["stop"] == ["\n\nHuman:"]

    def test_tool_choice_any_maps_to_required(self):
        assert _translate_tool_choice({"type": "any"}) == "required"
        assert _translate_tool_choice({"type": "auto"}) == "auto"
        assert _translate_tool_choice({"type": "none"}) == "none"
        assert _translate_tool_choice(None) is None

    def test_upstream_extra_body_injects_and_overrides(self):
        # Registry-pinned inference config (e.g. Qwen nothink + sampling) is
        # merged into the upstream payload and wins over what the CLI sent.
        anthropic_req = {
            "model": "qwen36-27b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 32000,
            "temperature": 1.0,
        }
        extra = {
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.7,
        }
        payload, _, _ = _build_openai_request(
            anthropic_req, modify_rules=[], upstream_extra_body=extra
        )
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["temperature"] == 0.7  # config overrides the CLI's 1.0
        assert payload["max_tokens"] == 32000  # untouched passthrough

    def test_upstream_extra_body_absent_is_noop(self):
        anthropic_req = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        payload, _, _ = _build_openai_request(anthropic_req, modify_rules=[])
        assert "chat_template_kwargs" not in payload

    def test_upstream_extra_body_respects_output_cap(self):
        # extra_body merge runs before the cap clamp, so the cap stays a ceiling.
        anthropic_req = {
            "model": "qwen36-27b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 32000,
        }
        payload, _, _ = _build_openai_request(
            anthropic_req,
            modify_rules=[],
            upstream_extra_body={"max_tokens": 50000},
            max_output_tokens_cap=8000,
        )
        assert payload["max_tokens"] == 8000


class TestContainerProxyStartup:
    def test_start_container_proxy_uses_configured_mounted_log_dir(self):
        container = Container(name="test", image="image")
        container._started = True

        exec_calls: list[str] = []

        def fake_exec(command: str, **kwargs) -> ExecResult:
            exec_calls.append(command)
            if "healthz" in command:
                return ExecResult(command=command, stdout="200", stderr="", exit_code=0)
            return ExecResult(command=command, stdout="", stderr="", exit_code=0)

        container.exec = fake_exec  # type: ignore[method-assign]
        container.copy_to = MagicMock()
        container.write_file = MagicMock()
        container.exec_background = MagicMock(return_value="123")

        with TemporaryDirectory() as tmpdir:
            config = ProxyInstanceConfig(
                upstream_base_url="https://example.com/v1",
                upstream_api_key="key",
                upstream_protocol="openai",
                artifact_dir=Path(tmpdir) / "trials" / "range1-L0" / "proxy",
                trial_id="range1-L0",
                container_log_dir="/var/lib/cage/proxy",
                logs_mounted=True,
            )

            proxy = start_container_proxy(container, config)

        setup_cmd = next(c for c in exec_calls if "mkdir -p" in c)
        assert "/var/lib/cage/proxy" in setup_cmd
        assert "progress.json" in setup_cmd

        bg_cmd = container.exec_background.call_args.args[0]
        assert "--log-dir /var/lib/cage/proxy" in bg_cmd
        assert ">/var/lib/cage/proxy/stdout.log" in bg_cmd
        assert proxy.log_dir == "/var/lib/cage/proxy"
        assert proxy.logs_mounted is True

    def test_start_container_proxy_preserves_mounted_log_dir_owner(self):
        container = Container(name="test", image="image")
        container._started = True

        exec_calls: list[str] = []

        def fake_exec(command: str, **kwargs) -> ExecResult:
            exec_calls.append(command)
            if "healthz" in command:
                return ExecResult(command=command, stdout="200", stderr="", exit_code=0)
            return ExecResult(command=command, stdout="", stderr="", exit_code=0)

        container.exec = fake_exec  # type: ignore[method-assign]
        container.copy_to = MagicMock()
        container.write_file = MagicMock()
        container.exec_background = MagicMock(return_value="123")

        with TemporaryDirectory() as tmpdir:
            config = ProxyInstanceConfig(
                upstream_base_url="https://example.com/v1",
                upstream_api_key="key",
                upstream_protocol="openai",
                artifact_dir=Path(tmpdir) / "trials" / "cvb-CVE-2023-37999" / "proxy",
                trial_id="cvb-CVE-2023-37999",
                container_log_dir="/var/lib/cage/proxy",
                logs_mounted=True,
            )

            start_container_proxy(container, config)

        setup_cmd = next(c for c in exec_calls if "proxy.jsonl" in c)
        assert setup_cmd.startswith(
            "(mkdir -p /run/cage-proxy /var/lib/cage/proxy && "
            "chmod 0700 /run/cage-proxy) && "
        )
        assert "chown" not in setup_cmd
        assert container.exec_background.call_args.kwargs == {}

    def test_start_container_proxy_chowns_log_dir_for_agent(self):
        container = Container(name="test", image="image")
        container._started = True

        exec_calls: list[str] = []

        def fake_exec(command: str, **kwargs) -> ExecResult:
            exec_calls.append(command)
            if "healthz" in command:
                return ExecResult(command=command, stdout="200", stderr="", exit_code=0)
            return ExecResult(command=command, stdout="", stderr="", exit_code=0)

        container.exec = fake_exec  # type: ignore[method-assign]
        container.write_file = MagicMock()
        container.exec_background = MagicMock(return_value="123")

        with TemporaryDirectory() as tmpdir:
            config = ProxyInstanceConfig(
                upstream_base_url="https://example.com/v1",
                upstream_api_key="key",
                upstream_protocol="openai",
                artifact_dir=Path(tmpdir),
                trial_id="trial",
            )

            start_container_proxy(container, config)

        setup_cmd = next(c for c in exec_calls if "mkdir -p" in c)
        assert setup_cmd == (
            "(mkdir -p /run/cage-proxy /var/lib/cage/proxy && "
            "chmod 0700 /run/cage-proxy) && rm -f "
            "/var/lib/cage/proxy/proxy.jsonl "
            "/var/lib/cage/proxy/tool_calls.jsonl "
            "/var/lib/cage/proxy/progress.json "
            "/var/lib/cage/proxy/stdout.log "
            "/var/lib/cage/proxy/stderr.log "
            "&& chown -R agent:agent /var/lib/cage/proxy"
        )

    def test_start_container_proxy_redirects_background_logs(self):
        container = Container(name="test", image="image")
        container._started = True

        def fake_exec(command: str, **kwargs) -> ExecResult:
            if "healthz" in command:
                return ExecResult(command=command, stdout="200", stderr="", exit_code=0)
            return ExecResult(command=command, stdout="", stderr="", exit_code=0)

        container.exec = fake_exec  # type: ignore[method-assign]
        container.write_file = MagicMock()
        container.exec_background = MagicMock(return_value="123")

        with TemporaryDirectory() as tmpdir:
            config = ProxyInstanceConfig(
                upstream_base_url="https://example.com/v1",
                upstream_api_key="key",
                upstream_protocol="openai",
                artifact_dir=Path(tmpdir),
                trial_id="trial",
            )

            start_container_proxy(container, config)

        bg_cmd = container.exec_background.call_args.args[0]
        assert ">/var/lib/cage/proxy/stdout.log" in bg_cmd
        assert "2>/var/lib/cage/proxy/stderr.log" in bg_cmd
        assert "--config /run/cage-proxy/config.json" in bg_cmd
        assert container.write_file.call_args.args[0] == "/run/cage-proxy/config.json"

    def test_start_container_proxy_uses_free_port_when_unset(self):
        container = Container(name="test", image="image")
        container._started = True

        exec_calls: list[str] = []

        def fake_exec(command: str, **kwargs) -> ExecResult:
            exec_calls.append(command)
            if "healthz" in command:
                return ExecResult(command=command, stdout="200", stderr="", exit_code=0)
            return ExecResult(command=command, stdout="", stderr="", exit_code=0)

        container.exec = fake_exec  # type: ignore[method-assign]
        container.write_file = MagicMock()
        container.exec_background = MagicMock(return_value="123")

        with TemporaryDirectory() as tmpdir:
            config = ProxyInstanceConfig(
                upstream_base_url="https://example.com/v1",
                upstream_api_key="key",
                upstream_protocol="openai",
                artifact_dir=Path(tmpdir),
                trial_id="trial",
                port=0,
            )

            with patch("cage.proxy.host._pick_free_local_port", return_value=43123):
                proxy = start_container_proxy(container, config)

        bg_cmd = container.exec_background.call_args.args[0]
        assert "--port 43123" in bg_cmd
        assert "http://localhost:43123/healthz" in exec_calls[-1]
        assert proxy.port == 43123

    def test_container_proxy_stop_skips_copy_for_mounted_logs(self, tmp_path):
        container = MagicMock()
        container.name = "agent-container"
        container.is_process_running.return_value = False
        proxy = ContainerProxyInstance(
            container=container,
            port=43123,
            pid="123",
            trial_id="range1-L0",
            config_path="/run/cage-proxy/config.json",
            log_dir="/var/lib/cage/proxy",
            logs_mounted=True,
        )

        proxy.stop(artifact_dir=tmp_path)

        container.kill_process.assert_called_once_with("123", signal="TERM")
        container.copy_from.assert_not_called()

    def test_container_proxy_resource_metadata_excludes_upstream_config(self, tmp_path):
        container = MagicMock()
        container.name = "agent-container"
        proxy = ContainerProxyInstance(
            container=container,
            port=43123,
            pid="123",
            trial_id="range1-L0",
            config_path="/run/cage-proxy/config.json",
            log_dir="/var/lib/cage/proxy",
            logs_mounted=True,
        )

        metadata = proxy.resource_metadata()

        assert metadata == {
            "base_url": "http://localhost:43123",
            "config_path": "/run/cage-proxy/config.json",
            "container_name": "agent-container",
            "log_dir": "/var/lib/cage/proxy",
            "logs_mounted": True,
            "pid": "123",
            "port": 43123,
            "trial_id": "range1-L0",
        }
        for sensitive_key in (
            "upstream_base_url",
            "upstream_api_key",
            "extra_headers",
            "http_proxy",
        ):
            assert sensitive_key not in metadata


class TestContainerProxyHttpClient:
    def test_make_http_client_enables_http2(self):
        captured: dict[str, object] = {}

        class DummyClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        original = container_proxy.httpx.Client
        container_proxy.httpx.Client = DummyClient  # type: ignore[assignment]
        try:
            client = container_proxy._make_http_client(12.5)
        finally:
            container_proxy.httpx.Client = original  # type: ignore[assignment]

        assert isinstance(client, DummyClient)
        assert captured["timeout"] == 12.5
        assert captured["trust_env"] is False
        assert captured["http2"] is True


class TestContainerProxyProgress:
    def test_recorder_updates_progress_summary_after_each_entry(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "range1-L0")

        recorder.record(
            request_id="req-0001",
            anthropic_request={"messages": [{"role": "user", "content": "hello"}]},
            upstream_response={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "Bash"}},
                                {"function": {"name": "Read"}},
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        )
        recorder.record(
            request_id="req-0002",
            anthropic_request={},
            status="error",
            error="upstream failed",
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["trial_id"] == "range1-L0"
        assert progress["total_requests"] == 2
        assert progress["success"] == 1
        assert progress["errors"] == 1
        assert progress["last_status"] == "error"
        assert progress["started_at_ms"] > 0
        assert progress["last_ts_ms"] > 0
        assert progress["tokens_in"] == 11
        assert progress["tokens_out"] == 7
        assert progress["tools_used"] == {"Bash": 1, "Read": 1}

    def test_recorder_counts_responses_api_usage(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "range2")

        recorder.record(
            request_id="req-0001",
            anthropic_request={},
            upstream_response={
                "id": "resp_123",
                "object": "response",
                "usage": {
                    "input_tokens": 101,
                    "input_tokens_details": {"cached_tokens": 70},
                    "output_tokens": 23,
                    "output_tokens_details": {"reasoning_tokens": 7},
                },
            },
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["tokens_in"] == 101
        assert progress["tokens_out"] == 23
        assert progress["tokens_reasoning"] == 7

    def test_recorder_accumulates_usage_cost_when_provider_reports_it(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "range-cost")

        recorder.record(
            request_id="req-0001",
            anthropic_request={},
            upstream_response={
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0123},
            },
        )
        recorder.record(
            request_id="req-0002",
            anthropic_request={},
            upstream_response={
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 30, "cost": 0.0456},
            },
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["cost_usd"] == 0.0579

    def test_recorder_estimates_usage_cost_from_model_prices(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(
            tmp_path,
            "range-estimated-cost",
            input_cost_per_1m=2.0,
            output_cost_per_1m=8.0,
        )

        recorder.record(
            request_id="req-0001",
            anthropic_request={},
            upstream_response={
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 250},
            },
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["cost_usd"] == 0.004

    def test_recorder_reports_first_runtime_budget_limit_reached(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(
            tmp_path,
            "range-budget",
            input_cost_per_1m=2.0,
            output_cost_per_1m=8.0,
        )

        recorder.record(
            request_id="req-0001",
            anthropic_request={},
            upstream_response={
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 250},
            },
        )

        signal = recorder.runtime_budget_signal(
            max_input_tokens=999,
            max_output_tokens=1000,
            max_cost=1.0,
        )

        assert signal == {
            "kind": "max_input_tokens",
            "current": 1000,
            "limit": 999,
            "unit": "tokens",
        }

    def test_recorder_reserves_in_flight_budgeted_rounds(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "range-budget")

        assert recorder.try_reserve_budgeted_round(max_requests=1) is True
        assert recorder.try_reserve_budgeted_round(max_requests=1) is False

        recorder.release_reserved_round()

        assert recorder.try_reserve_budgeted_round(max_requests=1) is True
        recorder.record(
            request_id="req-0001",
            anthropic_request={},
            upstream_response={"choices": [{"message": {"content": ""}}]},
        )
        recorder.release_reserved_round()

        assert recorder.try_reserve_budgeted_round(max_requests=1) is False

    def test_recorder_counts_anthropic_cache_input_usage(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "range1")

        recorder.record(
            request_id="req-0001",
            anthropic_request={},
            upstream_response={
                "type": "message",
                "usage": {
                    "input_tokens": 595,
                    "cache_read_input_tokens": 24256,
                    "cache_creation_input_tokens": 128,
                    "output_tokens": 169,
                },
            },
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["tokens_in"] == 24979
        assert progress["tokens_out"] == 169

    def test_recorder_counts_anthropic_stream_usage(self, tmp_path):
        captured = None
        for event_data in (
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.1",
                    "usage": {"input_tokens": 101, "output_tokens": 1},
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 23},
            },
        ):
            captured = container_proxy._capture_stream_response(captured, event_data)

        recorder = container_proxy.ProxyRecorder(tmp_path, "range1")
        recorder.record(
            request_id="req-0001",
            openai_request={"stream": True},
            upstream_response=captured,
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["tokens_in"] == 101
        assert progress["tokens_out"] == 23

    def test_recorder_counts_openai_chat_stream_usage(self, tmp_path):
        captured = container_proxy._capture_stream_response(
            None,
            {
                "id": "chatcmpl_123",
                "object": "chat.completion.chunk",
                "usage": {
                    "prompt_tokens": 55,
                    "completion_tokens": 8,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
        )

        recorder = container_proxy.ProxyRecorder(tmp_path, "range1")
        recorder.record(
            request_id="req-0001",
            openai_request={"stream": True},
            upstream_response=captured,
        )

        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        assert progress["tokens_in"] == 55
        assert progress["tokens_out"] == 8
        assert progress["tokens_reasoning"] == 3

    def test_capture_anthropic_stream_content_blocks(self):
        captured = None
        for event_data in (
            {"type": "message_start", "message": {"id": "msg_1", "content": []}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"command"'},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": ':"id"}'},
            },
            {"type": "content_block_stop", "index": 1},
        ):
            captured = container_proxy._capture_stream_response(captured, event_data)

        assert captured["content"] == [
            {"type": "text", "text": "hello"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Bash",
                "input": {"command": "id"},
            },
        ]

    def test_capture_openai_responses_stream_output_items(self):
        captured = None
        for event_data in (
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant", "content": []},
            },
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": "",
                    "call_id": "call_1",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "delta": "{\"cmd\"",
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "delta": ":\"id\"}",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "object": "response",
                    "output": [],
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                },
            },
        ):
            captured = container_proxy._capture_stream_response(captured, event_data)

        assert captured["object"] == "response"
        assert captured["usage"]["input_tokens"] == 11
        assert captured["usage"]["output_tokens"] == 7
        assert captured["output"] == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"id\"}",
                "call_id": "call_1",
            },
        ]

    def test_capture_openai_chat_stream_delta_content(self):
        captured = None
        for event_data in (
            {
                "id": "chatcmpl_1",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": "plan",
                        },
                    }
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": "{\"cmd\"",
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ":\"id\"}"},
                                }
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            },
        ):
            captured = container_proxy._capture_stream_response(captured, event_data)

        assert captured["object"] == "chat.completion"
        assert captured["usage"]["prompt_tokens"] == 20
        assert captured["choices"][0]["finish_reason"] == "tool_calls"
        assert captured["choices"][0]["message"] == {
            "role": "assistant",
            "reasoning_content": "plan",
            "content": "hello",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"id\"}",
                    },
                }
            ],
        }


class TestHostProxyInstance:
    def test_stop_accepts_optional_artifact_dir(self, tmp_path):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), MagicMock())
        proxy = ProxyInstance(
            host="127.0.0.1",
            port=httpd.server_address[1],
            trial_id="trial",
            httpd=httpd,
            recorder=MagicMock(),
            _thread=threading.Thread(target=httpd.serve_forever, daemon=True),
        )
        proxy._thread.start()

        proxy.stop(artifact_dir=tmp_path)

        assert not proxy._thread.is_alive()


class TestTranslatingPathForcesStreamFalse:
    """The Anthropic→OpenAI translation path can't handle SSE — the
    translator needs the full JSON response to rewrite into Anthropic
    shape. ``_build_openai_request`` therefore forces ``stream: false``
    in the translated payload regardless of what the agent originally
    asked for. (The transparent forward path is a pure pass-through and
    does not strip — the harness controls streaming there.)"""

    def test_host_build_openai_request_emits_stream_false(self):
        # Anthropic side may carry stream:true — OpenAI payload must override.
        anthropic_req = {
            "model": "m", "system": "s",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        payload, _, _ = _build_openai_request(anthropic_req, modify_rules=[])
        assert payload["stream"] is False

    def test_host_build_openai_request_stream_false_when_absent(self):
        anthropic_req = {
            "model": "m", "system": "s",
            "messages": [{"role": "user", "content": "hi"}],
        }
        payload, _, _ = _build_openai_request(anthropic_req, modify_rules=[])
        assert payload["stream"] is False  # explicit, not relying on upstream default

    def test_container_build_openai_request_emits_stream_false(self):
        anthropic_req = {
            "model": "m", "system": "s",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        payload, _, _ = container_proxy._build_openai_request(anthropic_req)
        assert payload["stream"] is False


class TestRoundCountIsSuccessOnly:
    """Round-counting consumers must read ``successful_requests`` so failed
    or timed-out upstream calls don't inflate the agent's apparent
    progress. ``total_requests`` is retained but represents gross attempts."""

    def test_progress_json_carries_successful_requests(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "trial-x")
        # 2 successes + 1 error → successful_requests should be 2.
        recorder.record(
            request_id="r1",
            upstream_response={"choices": [{"message": {"content": ""}}], "usage": {}},
        )
        recorder.record(
            request_id="r2",
            upstream_response={"choices": [{"message": {"content": ""}}], "usage": {}},
        )
        recorder.record(
            request_id="r3",
            status="error",
            error="upstream blew up",
        )
        progress = json.loads((tmp_path / "progress.json").read_text())
        assert progress["successful_requests"] == 2
        assert progress["success"] == 2  # back-compat alias
        assert progress["errors"] == 1
        assert progress["total_requests"] == 3  # gross attempts retained

    def test_progress_json_zero_baseline(self, tmp_path):
        container_proxy.ProxyRecorder(tmp_path, "trial-empty")
        # No record() calls yet — progress.json should not exist or be empty.
        assert not (tmp_path / "progress.json").exists()

    def test_record_appends_one_jsonl_line_per_call(self, tmp_path):
        """proxy.jsonl line count == record() call count.

        Budget consumers must still use the success-only round counter.
        """
        recorder = container_proxy.ProxyRecorder(tmp_path, "trial-line")
        for i in range(5):
            recorder.record(
                request_id=f"r{i}",
                upstream_response={"choices": [{"message": {}}], "usage": {}},
                status="success" if i % 2 == 0 else "error",
            )
        lines = (tmp_path / "proxy.jsonl").read_text().splitlines()
        assert len(lines) == 5

    def test_round_budget_counts_only_successful_non_compact_rounds(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "trial-budget")
        recorder.record(
            request_id="r1",
            openai_request={"model": "m"},
            upstream_response={"choices": [{"message": {}}], "usage": {}},
        )
        recorder.record(
            request_id="r2",
            status="error",
            error="HTTP 503",
            openai_request={"model": "m"},
        )
        recorder.record(
            request_id="r3",
            openai_request={"model": "m", "_proxy_compact_rewritten": True},
            upstream_response={"choices": [{"message": {}}], "usage": {}},
        )
        recorder.record(
            request_id="r4",
            openai_request={"model": "m"},
            upstream_response={"choices": [{"message": {}}], "usage": {}},
        )

        assert recorder.budgeted_round_count() == 2

    def test_max_requests_gate_uses_successful_round_budget(self):
        """The proxy should not spend max_rounds on failed upstream calls."""
        import inspect
        src = inspect.getsource(container_proxy)

        assert "try_reserve_budgeted_round(max_requests=" in src
        assert "release_reserved_round()" in src
        assert "_total_requests >= max_requests" not in src

    def test_compact_rewritten_calls_split_from_round_count(self, tmp_path):
        """Compact rewrites (``/v1/responses/compact`` → ``/v1/responses``)
        consume an upstream slot but aren't agent decisions. They must be
        tracked separately and SUBTRACTED from ``successful_requests`` so
        round counts don't inflate."""
        recorder = container_proxy.ProxyRecorder(tmp_path, "trial-compact")
        # 1 normal success
        recorder.record(
            request_id="r1",
            openai_request={"model": "m"},
            upstream_response={"choices": [{"message": {}}], "usage": {}},
        )
        # 2 compact successes
        for rid in ("r2", "r3"):
            recorder.record(
                request_id=rid,
                openai_request={"model": "m", "_proxy_compact_rewritten": True},
                upstream_response={"choices": [{"message": {}}], "usage": {}},
            )
        # 1 error
        recorder.record(
            request_id="r4", status="error", error="oops",
            openai_request={"model": "m"},
        )

        progress = json.loads((tmp_path / "progress.json").read_text())
        assert progress["total_requests"] == 4
        assert progress["success"] == 3          # gross success (back-compat)
        assert progress["compact_requests"] == 2
        assert progress["successful_requests"] == 1  # 3 - 2 = agent rounds
        assert progress["errors"] == 1

    def test_compact_flag_propagates_through_forward_transparent_signature(self):
        """Audit lineage: the do_POST handler detects /compact, sets a
        ``compact_rewritten`` flag, and threads it into _forward_transparent.
        Verify the function signature accepts it (so it doesn't get
        dropped silently in a refactor)."""
        import inspect
        src = inspect.getsource(container_proxy)
        # Compact flag passed to forward
        assert "_forward_transparent(" in src
        assert "compact_rewritten=compact_rewritten" in src
        # And the audit marker key matches what the recorder looks for
        assert "_proxy_compact_rewritten" in src

    def test_sse_stream_chunks_merge_to_single_response(self):
        """When upstream returns SSE (rare after stream-strip but possible
        if upstream ignores stream:false), ``_capture_stream_response``
        merges all chunks into one response — the caller then makes ONE
        record() call. Verify the merge step."""
        captured = None
        for delta in [
            {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "hel"}}]},
            {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {"object": "chat.completion.chunk",
             "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]},
        ]:
            captured = container_proxy._capture_stream_response(captured, delta)
        assert captured["choices"][0]["message"]["content"] == "hello"
        assert captured["choices"][0]["finish_reason"] == "stop"

    def test_web_data_prefers_successful_requests(self, tmp_path):
        """Web's RunInfo.live_total_requests must read successful_requests,
        not total_requests — otherwise dashboards inflate round counts with
        failed upstream calls."""
        from cage.web.data import _load_run_info

        run_dir = tmp_path / "run-test"
        trials_dir = run_dir / "trials" / "trial-1"
        proxy_dir = trials_dir / "proxy"
        proxy_dir.mkdir(parents=True)
        # progress.json with new field — 10 successful, 5 errors, 15 total.
        (proxy_dir / "progress.json").write_text(json.dumps({
            "trial_id": "trial-1",
            "total_requests": 15,
            "successful_requests": 10,
            "success": 10,
            "errors": 5,
            "last_ts_ms": 1_700_000_000_000,
            "last_status": "success",
        }))
        # Stub minimal dashboard.json + agent dir layout.
        (run_dir / "dashboard.json").write_text(json.dumps({
            "status": "running",
            "started_at": "2026-05-12T00:00:00+00:00",
            "agents": {},
        }))
        info = _load_run_info(run_dir, project="proj", agent_dir_name="agent")
        # 10 (successful_requests), NOT 15 (total_requests).
        assert info.live_total_requests == 10
        assert info.live_errors == 5

    def test_web_data_falls_back_when_new_field_absent(self, tmp_path):
        """Old progress.json files written before the field was added must
        still be readable — fall back to ``success`` then ``total_requests``."""
        from cage.web.data import _load_run_info

        run_dir = tmp_path / "run-legacy"
        trials_dir = run_dir / "trials" / "trial-1"
        proxy_dir = trials_dir / "proxy"
        proxy_dir.mkdir(parents=True)
        # Old schema: no successful_requests, only success + total_requests.
        (proxy_dir / "progress.json").write_text(json.dumps({
            "trial_id": "trial-1",
            "total_requests": 7,
            "success": 6,
            "errors": 1,
            "last_ts_ms": 1_700_000_000_000,
            "last_status": "success",
        }))
        (run_dir / "dashboard.json").write_text(json.dumps({
            "status": "running",
            "started_at": "2026-05-12T00:00:00+00:00",
            "agents": {},
        }))
        info = _load_run_info(run_dir, project="proj", agent_dir_name="agent")
        assert info.live_total_requests == 6  # success fallback

    def test_web_data_oldest_legacy_uses_total_requests(self, tmp_path):
        """Very old progress.json without successful_requests OR success
        field falls all the way back to total_requests (preserves old UX)."""
        from cage.web.data import _load_run_info

        run_dir = tmp_path / "run-very-old"
        trials_dir = run_dir / "trials" / "trial-1"
        proxy_dir = trials_dir / "proxy"
        proxy_dir.mkdir(parents=True)
        (proxy_dir / "progress.json").write_text(json.dumps({
            "trial_id": "trial-1",
            "total_requests": 4,
            "errors": 0,
            "last_ts_ms": 1_700_000_000_000,
        }))
        (run_dir / "dashboard.json").write_text(json.dumps({
            "status": "running",
            "started_at": "2026-05-12T00:00:00+00:00",
            "agents": {},
        }))
        info = _load_run_info(run_dir, project="proj", agent_dir_name="agent")
        assert info.live_total_requests == 4


class TestTransparentForwardIsPassThrough:
    """The transparent forward path must NOT mutate the request body for
    streaming — the harness controls ``stream`` and ``Accept``. The proxy
    is responsible only for counting (1:1 between agent calls and
    ``proxy.jsonl`` records, structurally enforced via end-of-stream
    ``recorder.record(...)`` in the SSE branch)."""

    def test_forward_transparent_no_stream_strip_helper(self):
        import inspect
        src = inspect.getsource(container_proxy)
        # The helper that used to rewrite stream: true → false must be
        # gone; otherwise codex-relay endpoints that mandate streaming
        # break (cf. config/models.yml::gpt-5.5).
        assert "_strip_stream_flag" not in src
        # Accept header is never overridden to drop SSE.
        assert "force_non_streaming" not in src


class TestSseErrorCapture:
    """SSE streams that carry only ``event: error`` (Anthropic-style upstream
    rate-limit/overload returned as an HTTP-200 SSE) must be recorded as
    ``status="error"`` — otherwise the web inspector renders them as empty
    "No response content captured" steps and progress.json inflates the
    success counter. Regression test for the SIYUCMS/pass_2 run where 22 of
    151 entries were mis-classified."""

    def test_capture_anthropic_sse_error_event(self):
        captured = container_proxy._capture_stream_response(
            None,
            {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "该模型当前访问量过大，请您稍后再试",
                },
            },
        )
        assert isinstance(captured, dict)
        assert captured["_sse_error"]["type"] == "overloaded_error"

    def test_capture_outcome_marks_error_stream_as_failure(self):
        captured = container_proxy._capture_stream_response(
            None,
            {"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
        )
        ok, err = container_proxy._sse_capture_outcome(captured)
        assert ok is False
        assert "overloaded_error" in err

    def test_capture_outcome_marks_empty_stream_as_failure(self):
        ok, err = container_proxy._sse_capture_outcome(None)
        assert ok is False
        assert "no parseable" in err.lower()

    def test_capture_outcome_accepts_anthropic_content_stream(self):
        captured = None
        for ev in (
            {"type": "message_start", "message": {"id": "m1", "content": []}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
        ):
            captured = container_proxy._capture_stream_response(captured, ev)
        ok, err = container_proxy._sse_capture_outcome(captured)
        assert ok is True
        assert err == ""

    def test_capture_outcome_accepts_openai_chat_stream(self):
        captured = container_proxy._capture_stream_response(
            None,
            {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "hi"}}],
            },
        )
        ok, _ = container_proxy._sse_capture_outcome(captured)
        assert ok is True

    def test_sse_log_decoder_inflates_gzip_chunks(self):
        raw = (
            b'data: {"type":"message_start","message":{"id":"m1","content":[]}}\n\n'
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
        )
        decoder = container_proxy._make_stream_log_decoder("gzip")

        decoded = decoder(gzip.compress(raw), final=False) + decoder(b"", final=True)

        assert decoded == raw

    def test_recorder_records_error_status_for_empty_sse(self, tmp_path):
        recorder = container_proxy.ProxyRecorder(tmp_path, "trial-1")
        ok, sse_err = container_proxy._sse_capture_outcome(None)
        assert not ok
        recorder.record(
            request_id="req-0001",
            openai_request={"stream": True},
            upstream_response={},
            status="error",
            error=sse_err,
        )
        progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
        # The successful_requests counter must not be incremented for this
        # entry — it's an empty turn, not a real agent decision.
        assert progress["errors"] == 1
        assert progress["successful_requests"] == 0
        assert progress["success"] == 0


class TestSanitizeAssistantToolCallArguments:
    """Models occasionally emit truncated/invalid JSON for a tool_call's
    ``arguments`` (observed on Kimi-K2.6). On the next turn the harness
    sends that conversation back upstream and the upstream rejects with
    400. The proxy repairs the JSON in-flight so one bad model output
    doesn't kill the whole trial."""

    def test_passthrough_when_all_arguments_are_valid(self):
        body = {
            "model": "kimi-k2.6",
            "messages": [
                {"role": "user", "content": "scan"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Shell",
                                "arguments": '{"command": "ls"}',
                            },
                        }
                    ],
                },
            ],
        }
        raw = json.dumps(body).encode("utf-8")
        new_body, repairs = (
            container_proxy._sanitize_assistant_tool_calls_in_openai_body(raw)
        )
        assert repairs == []
        # No mutation: helper must return the exact bytes it received so
        # Content-Length doesn't have to be recomputed in the hot path.
        assert new_body is raw

    def test_repairs_truncated_arguments_json(self):
        bad_args = '{"command": "nmap -sn 10.0.0.0/24", "timeout": '
        body = {
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "Shell",
                                "arguments": bad_args,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_bad",
                    "content": "<system>ERROR: Error parsing JSON arguments</system>",
                },
            ],
        }
        raw = json.dumps(body).encode("utf-8")
        new_body, repairs = (
            container_proxy._sanitize_assistant_tool_calls_in_openai_body(raw)
        )
        assert len(repairs) == 1
        assert repairs[0]["name"] == "Shell"
        assert repairs[0]["tool_call_id"] == "call_bad"
        assert repairs[0]["msg_index"] == 0
        assert repairs[0]["tool_call_index"] == 0
        assert bad_args.startswith(repairs[0]["original_args"][:40])
        # Repaired body must parse and the bad tool_call's arguments must
        # now be valid JSON.
        decoded = json.loads(new_body)
        repaired_args = (
            decoded["messages"][0]["tool_calls"][0]["function"]["arguments"]
        )
        assert json.loads(repaired_args) == {}

    def test_ignores_non_assistant_messages(self):
        body = {
            "messages": [
                # Tool-result messages don't carry tool_calls; helper must
                # ignore them even if their content happens to look like
                # broken JSON.
                {"role": "tool", "tool_call_id": "x", "content": '{"bad": '},
                {"role": "user", "content": '{"bad": '},
            ],
        }
        raw = json.dumps(body).encode("utf-8")
        new_body, repairs = (
            container_proxy._sanitize_assistant_tool_calls_in_openai_body(raw)
        )
        assert repairs == []
        assert new_body is raw

    def test_handles_unparseable_body_bytes_safely(self):
        raw = b"not-json-at-all"
        new_body, repairs = (
            container_proxy._sanitize_assistant_tool_calls_in_openai_body(raw)
        )
        # Helper must never throw on a non-JSON body — the forward path
        # itself handles non-JSON requests transparently.
        assert repairs == []
        assert new_body is raw


class TestGeminiProjection:
    """Gemini ``generateContent`` calls are forwarded byte-for-byte but
    recorded as an OpenAI-shaped projection so the inspector parses them."""

    def test_route_detection(self):
        assert container_proxy._is_gemini_route(
            "/v1beta/models/gemini-2.5-pro:streamGenerateContent"
        )
        assert container_proxy._is_gemini_route(
            "/v1beta/models/gemini-2.5-pro:generateContent"
        )
        # countTokens, chat-completions and messages fall through to the
        # plain transparent forward.
        assert not container_proxy._is_gemini_route(
            "/v1beta/models/gemini-2.5-pro:countTokens"
        )
        assert not container_proxy._is_gemini_route("/v1/chat/completions")
        assert not container_proxy._is_gemini_route("/v1/messages")

    def test_request_projection_system_tools_and_function_turns(self):
        gem_req = {
            "systemInstruction": {"parts": [{"text": "You are an agent."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "run ls"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "sh", "args": {"cmd": "ls"}}}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "sh", "response": {"out": "flag.txt"}}}
                ]},
            ],
            "tools": [{"functionDeclarations": [
                {"name": "sh", "description": "run", "parameters": {"type": "object"}}
            ]}],
        }
        out = container_proxy._gemini_request_to_openai(
            gem_req, "/v1beta/models/gemini-2.5-pro:streamGenerateContent"
        )
        assert out["model"] == "gemini-2.5-pro"
        assert [m["role"] for m in out["messages"]] == [
            "system", "user", "assistant", "tool",
        ]
        assistant = out["messages"][2]
        assert assistant["tool_calls"][0]["function"]["name"] == "sh"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}
        assert out["tools"][0]["function"]["name"] == "sh"

    def test_response_projection_text_and_usage(self):
        gem_resp = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello"}], "role": "model"},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 10,
                "totalTokenCount": 17,
            },
            "modelVersion": "gemini-2.5-flash",
            "responseId": "abc",
        }
        out = container_proxy._gemini_response_to_openai(gem_resp, request_id="r1")
        msg = out["choices"][0]["message"]
        assert msg["content"] == "Hello"
        assert out["choices"][0]["finish_reason"] == "stop"
        # OpenAI convention: completion_tokens includes reasoning (thoughts).
        assert out["usage"]["prompt_tokens"] == 5
        assert out["usage"]["completion_tokens"] == 12
        assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 10

    def test_response_projection_tool_call(self):
        gem_resp = {
            "candidates": [{
                "content": {"parts": [
                    {"functionCall": {"name": "web_fetch", "args": {"url": "http://x"}}}
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
        }
        out = container_proxy._gemini_response_to_openai(gem_resp, request_id="r2")
        tc = out["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "web_fetch"
        assert json.loads(tc["function"]["arguments"]) == {"url": "http://x"}
        # A tool call overrides the STOP finish reason.
        assert out["choices"][0]["finish_reason"] == "tool_calls"

    def test_stream_merge_then_project(self):
        acc = None
        for chunk in [
            {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            {
                "candidates": [{
                    "content": {"parts": [{"text": "lo"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5,
                },
            },
        ]:
            acc = container_proxy._gemini_merge_stream_chunk(acc, chunk)
        out = container_proxy._gemini_response_to_openai(acc, request_id="r3")
        assert out["choices"][0]["message"]["content"] == "Hello"
        assert out["usage"]["prompt_tokens"] == 3
