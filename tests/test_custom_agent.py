"""Custom (manifest-driven) agent: manifest parse, token/param substitution,
trace env, artifact declaration, and the --param overlay merge."""

from __future__ import annotations

from pathlib import Path

import shlex
import pytest

from cage.agents.custom import load_custom_agent, load_manifest
from cage.experiment.engine.overlays import (
    merge_selected_agent_params,
    parse_set_expression,
)
from cage.models import ModelConfig


def _write_agent(tmp_path: Path, *, command: str, params: dict | None = None) -> Path:
    body = [
        "name: demo",
        "image: cage/custom-langgraph:base",
        f"command: {command}",
        "env:",
        '  OPENAI_BASE_URL: "{base_url}"',
        '  OPENAI_MODEL: "{model_name}"',
    ]
    if params is not None:
        body.append("params:")
        for key, value in params.items():
            body.append(f"  {key}: {value}")
    (tmp_path / "agent.yml").write_text("\n".join(body) + "\n", encoding="utf-8")
    return tmp_path


def _model() -> ModelConfig:
    return ModelConfig(id="nex-n2", provider="openai", model="nex-n2", api_key="sk-x")


def _cmd(agent) -> str:
    full = agent.build_launch_command(
        "solve it", model=_model(), max_rounds=-1, proxy_url="http://127.0.0.1:9000"
    )
    # build_launch_command wraps the real command in `bash -c <inner>` so Cage's
    # env exports reach the agent process; unwrap to assert on the inner command.
    parts = shlex.split(full)
    return parts[2] if parts[:2] == ["bash", "-c"] else full


def test_params_may_not_shadow_a_reserved_token(tmp_path: Path):
    # A param that duplicates a Cage-owned concept (rounds/model/...) is a second
    # knob for the same thing — rejected at load, not silently lost. Map the
    # reserved {token} in `command` instead (e.g. {max_rounds}).
    src = _write_agent(tmp_path, command="run --iters {max_rounds}", params={"max_rounds": 5})
    with pytest.raises(ValueError, match="reserved"):
        load_manifest(str(src), tmp_path)
    src2 = _write_agent(tmp_path, command="run", params={"model.foo": "x"})
    with pytest.raises(ValueError, match="reserved"):
        load_manifest(str(src2), tmp_path)


def test_manifest_requires_image_and_command(tmp_path: Path):
    (tmp_path / "agent.yml").write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_manifest(str(tmp_path), tmp_path)


def test_reserved_tokens_fill_command_and_env(tmp_path: Path):
    src = _write_agent(tmp_path, command="run {workspace_dir} --model {model_name} -i {task_instruction}")
    agent = load_custom_agent(str(src), tmp_path)
    cmd = _cmd(agent)
    assert "/home/agent/workspace" in cmd
    assert "--model nex-n2" in cmd
    assert "-i 'solve it'" in cmd  # shell-quoted
    env = agent.env_vars(proxy_url="http://127.0.0.1:9000", model=_model())
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"  # /v1 for openai
    assert env["OPENAI_MODEL"] == "nex-n2"


def test_param_default_and_override(tmp_path: Path):
    src = _write_agent(
        tmp_path, command="run --iters {max_iterations}", params={"max_iterations": 10}
    )
    assert "--iters 10" in _cmd(load_custom_agent(str(src), tmp_path))
    assert "--iters 25" in _cmd(
        load_custom_agent(str(src), tmp_path, {"max_iterations": "25"})
    )


def test_reserved_token_beats_same_named_param(tmp_path: Path):
    src = _write_agent(tmp_path, command="run --model {model_name}")
    agent = load_custom_agent(str(src), tmp_path, {"model_name": "HIJACK"})
    cmd = _cmd(agent)
    assert "--model nex-n2" in cmd and "HIJACK" not in cmd


def test_unknown_placeholder_with_no_value_raises(tmp_path: Path):
    src = _write_agent(tmp_path, command="run {nonexistent}")
    agent = load_custom_agent(str(src), tmp_path)
    with pytest.raises(ValueError, match="unknown placeholder"):
        _cmd(agent)


def test_trace_env_enabled(tmp_path: Path):
    # CAGE_TRACE turns on the base image's hook, which stamps the LangGraph node
    # on each model request as X-Cage-* headers (recorded by the proxy).
    src = _write_agent(tmp_path, command="run")
    agent = load_custom_agent(str(src), tmp_path)
    env = agent.env_vars(proxy_url="http://127.0.0.1:9000", model=_model())
    assert env["CAGE_TRACE"] == "1"


def test_privileged_defaults_off_and_opts_in(tmp_path: Path):
    # Default: no privileged container.
    src = _write_agent(tmp_path, command="run")
    agent = load_custom_agent(str(src), tmp_path)
    assert agent.manifest.privileged is False
    assert agent.container_resources(home_dir="/h", model=_model()).privileged is False

    # Opt in via the manifest (Docker-in-Docker agents like Cairn).
    (tmp_path / "agent.yml").write_text(
        "name: dind\nimage: cage/cairn:latest\ncommand: run\nprivileged: true\n",
        encoding="utf-8",
    )
    agent2 = load_custom_agent(str(tmp_path), tmp_path)
    assert agent2.manifest.privileged is True
    assert agent2.container_resources(home_dir="/h", model=_model()).privileged is True


def test_merge_selected_agent_params_overlay():
    raw = {"agents": [{"id": "demo", "source": "x", "params": {"a": 1, "b": 2}}]}
    key, value = parse_set_expression("a=9")
    merge_selected_agent_params(raw, agent_ids=("demo",), params={key: value})
    assert raw["agents"][0]["params"] == {"a": 9, "b": 2}


def test_merge_params_ambiguous_agent_raises():
    raw = {"agents": [{"id": "a", "source": "x"}, {"id": "b", "source": "y"}]}
    with pytest.raises(ValueError, match="exactly one"):
        merge_selected_agent_params(raw, agent_ids=(), params={"k": "v"})
