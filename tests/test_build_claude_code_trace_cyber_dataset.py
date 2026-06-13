"""Tests for the Claude Code cyber trace dataset builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_claude_code_trace_cyber_dataset.py"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_trace_dataset", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch_proxy(root: Path, relative: str) -> Path:
    proxy = root / relative / "proxy" / "proxy.jsonl"
    proxy.parent.mkdir(parents=True, exist_ok=True)
    proxy.write_text("", encoding="utf-8")
    return proxy


def test_group_key_distinguishes_selected_model_task_and_run_family() -> None:
    module = load_builder_module()

    assert (
        module.classify_group(
            "examples/cvebench",
            "claude_code_baseline:qwen36-27b:stateless",
            "run-20260512T111807",
        )
        == ("qwen36-27b", "cvebench", "full-pass4")
    )
    assert (
        module.classify_group(
            "examples/agent_pentest_bench",
            "claude_code_deepseek_v4_pro:deepseek-v4-pro:stateless",
            "ds-limit1-scorefix-0524",
        )
        == ("deepseek-v4-pro", "agent_pentest_bench", "scorefix-sanity")
    )
    assert (
        module.classify_group(
            "examples/agent_pentest_bench",
            "claude_code_opus47_cyber:claude-opus-4-7-cyber:stateless",
            "postexp-claude-cyber-20260526",
        )
        == ("claude-opus-4-7-cyber", "agent_pentest_bench", "postexp-cyber-passk3")
    )
    assert (
        module.classify_group(
            "examples/agent_pentest_bench",
            "qwen_code_qwen37max:qwen3.7-max:stateless",
            "postexp-qwen37max-p3-20260526",
        )
        == ("qwen3.7-max", "agent_pentest_bench", "postexp-pass3")
    )


def test_discover_trials_skips_non_cyber_worktrees_and_before_resume_by_default(
    tmp_path: Path,
) -> None:
    module = load_builder_module()
    touch_proxy(
        tmp_path,
        "examples/cvebench/.cage_runs/claude_code_baseline:qwen36-27b:stateless/"
        "run-20260512T111807/trials/cvb-CVE-zero_day/zero_day/pass_1",
    )
    touch_proxy(
        tmp_path,
        "examples/cvebench/.cage_runs/claude_code_baseline:qwen36-27b:stateless/"
        "run-20260512T111807/trials/cvb-CVE-zero_day.before_resume_20260524T010101",
    )
    touch_proxy(
        tmp_path,
        ".worktrees/skill-inject/examples/skill_inject/.cage_runs/"
        "cc_stateless:glm-5.1:stateless/run-20260515T193946/trials/INST-1",
    )
    touch_proxy(
        tmp_path,
        "examples/agent_pentest_bench/.cage_runs/"
        "qwen_code_qwen37max:qwen3.7-max:stateless/"
        "postexp-qwen37max-p3-20260526/trials/pb-postexp-range-1-l0/pass_1",
    )
    touch_proxy(
        tmp_path,
        "examples/agent_pentest_bench/.cage_runs/"
        "claude_code_opus47_cyber:claude-opus-4-7-cyber:stateless/"
        "postexp-claude-cyber-20260526/trials/pb-postexp-range-1-l0/pass_1",
    )

    trials = module.discover_trials(tmp_path)

    assert len(trials) == 3
    discovered = {(trial.model, trial.task_family, trial.trial_path) for trial in trials}
    assert ("qwen36-27b", "cvebench", "cvb-CVE-zero_day/zero_day/pass_1") in discovered
    assert (
        "qwen3.7-max",
        "agent_pentest_bench",
        "pb-postexp-range-1-l0/pass_1",
    ) in discovered
    assert (
        "claude-opus-4-7-cyber",
        "agent_pentest_bench",
        "pb-postexp-range-1-l0/pass_1",
    ) in discovered
