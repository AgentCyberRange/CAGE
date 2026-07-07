from __future__ import annotations

import importlib
import io
import json
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from cage.cli import main
from cage.contracts.logging import LoggingConfig

CLI_MAIN = importlib.import_module("cage.cli.main")


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        benchmark=SimpleNamespace(),
        agents=[],
        # timeout gives the run a finite termination condition (now required).
        execution=SimpleNamespace(max_trial=None, timeout=600, max_rounds="unlimited"),
        logging=LoggingConfig(),
        metadata={},
        resume=False,
        run_id="",
    )


def _run_summary() -> dict:
    return {
        "experiment": "demo",
        "run_id": "run-1",
        "run_dir": "",
        "dashboard_path": "",
        "agents": {},
    }


def test_logging_config_defaults_to_plain_progress_and_inspect_on() -> None:
    config = LoggingConfig()

    assert config.terminal_ui is True
    assert config.inspect_mode == "on"


def test_run_command_keeps_terminal_ui_enabled_by_default(
    tmp_path: Path, monkeypatch,
) -> None:
    project_file = tmp_path / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    config = _fake_config()
    captured: dict[str, object] = {}

    monkeypatch.setattr(CLI_MAIN, "setup_logging", lambda _config: None)
    monkeypatch.setattr("cage.config.experiment.resolve", lambda _path: config)

    def fake_run_experiment(run_config, **_kwargs):
        captured["terminal_ui"] = run_config.logging.terminal_ui
        captured["inspect_mode"] = run_config.logging.inspect_mode
        return _run_summary()

    fake_orchestrator = types.SimpleNamespace(
        ResumeCompatibilityError=RuntimeError,
        analyze_resume_plan=lambda _config: [],
        run_experiment=fake_run_experiment,
    )
    monkeypatch.setitem(sys.modules, "cage.experiment.engine.conductor", fake_orchestrator)

    result = CliRunner().invoke(main, ["run", str(project_file)])

    assert result.exit_code == 0, result.output
    assert captured["terminal_ui"] is True
    assert captured["inspect_mode"] == "on"
    assert result.output.startswith("   _________   ____________")
    assert "Experiment:" not in result.output
    assert "View with: cage inspect" not in result.output


def test_run_command_prints_banner_before_run_starts(
    tmp_path: Path, monkeypatch,
) -> None:
    project_file = tmp_path / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    config = _fake_config()

    monkeypatch.setattr(CLI_MAIN, "setup_logging", lambda _config: None)
    monkeypatch.setattr("cage.config.experiment.resolve", lambda _path: config)

    def fake_run_experiment(_run_config, **_kwargs):
        print("PREFLIGHT MARKER")
        return _run_summary()

    fake_orchestrator = types.SimpleNamespace(
        ResumeCompatibilityError=RuntimeError,
        analyze_resume_plan=lambda _config: [],
        run_experiment=fake_run_experiment,
    )
    monkeypatch.setitem(sys.modules, "cage.experiment.engine.conductor", fake_orchestrator)

    result = CliRunner().invoke(main, ["run", str(project_file)])

    assert result.exit_code == 0, result.output
    assert result.output.index("_________   ____________") < result.output.index(
        "PREFLIGHT MARKER"
    )


def test_run_command_rejects_removed_display_flags(
    tmp_path: Path, monkeypatch,
) -> None:
    project_file = tmp_path / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    config = _fake_config()

    monkeypatch.setattr(CLI_MAIN, "setup_logging", lambda _config: None)
    monkeypatch.setattr("cage.config.experiment.resolve", lambda _path: config)

    fake_orchestrator = types.SimpleNamespace(
        ResumeCompatibilityError=RuntimeError,
        analyze_resume_plan=lambda _config: [],
        run_experiment=lambda _config: _run_summary(),
    )
    monkeypatch.setitem(sys.modules, "cage.experiment.engine.conductor", fake_orchestrator)

    runner = CliRunner()

    for args in (
        ["--no-terminal-ui"],
        ["--inspect", "off"],
        ["--debug-log"],
    ):
        result = runner.invoke(main, ["run", str(project_file), *args])
        assert result.exit_code != 0, result.output
        assert "Unknown run option" in result.output


def test_run_help_hides_removed_display_and_debug_flags() -> None:
    result = CliRunner().invoke(main, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "--inspect" not in result.output
    assert "--no-terminal-ui" not in result.output
    assert "--debug-log" not in result.output
    assert "--max-sample-num" in result.output
    assert "--max-trial-num" in result.output


def test_run_banner_formats_ascii_art_with_optional_color() -> None:
    from cage.cli.ui.run import format_run_banner

    plain = format_run_banner()
    colored = format_run_banner(color=True)

    assert plain.startswith("   _________   ____________")
    assert "\\____/_/  |_\\____/_____/" in plain
    assert "\033[" not in plain
    assert "\033[" in colored
    assert "_________   ____________" in colored


def test_run_progress_reporter_tracks_trial_lifecycle() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(total_trials=3, enabled=False)

    reporter.agent_started("agent-a", total_trials=3)
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-1",
        status="completed",
        duration_ms=1500,
        exit_code=0,
    )
    reporter.trial_replayed(
        agent_label="agent-a",
        trial_id="trial-2",
        sample_id="sample-2",
        trial_index=1,
        status="completed",
    )
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-3",
        sample_id="sample-3",
        trial_index=2,
    )
    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-3",
        status="failed",
        duration_ms=500,
        exit_code=1,
    )

    snapshot = reporter.snapshot()

    assert snapshot["completed"] == 1
    assert snapshot["failed"] == 1
    assert snapshot["replayed"] == 1
    assert snapshot["active"] == []
    assert snapshot["by_status"] == {
        "completed": 1,
        "failed": 1,
        "replayed": 1,
    }


def test_resume_progress_counts_only_plan_to_run_not_replayed() -> None:
    """On --resume the live bar tracks plan-to-run, excluding kept-from-disk.

    Regression for the "33/60 resolved" confusion: a resume of 60 trials where
    33 are replayed and 27 re-run must show the live denominator as 27 and not
    pre-fill ``done`` with the 33 replayed (those land in ``replayed``).
    """
    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(
        total_trials=60, runnable_trials=27, resume_replayed_trials=33, enabled=False
    )
    for i in range(33):
        reporter.trial_replayed(
            agent_label="a", trial_id=f"r{i}", sample_id=f"s{i}", trial_index=i
        )

    snap = reporter.snapshot()
    assert snap["total"] == 27, snap
    assert snap["done"] == 0, snap
    assert snap["replayed"] == 33, snap

    reporter.trial_started(
        agent_label="a", trial_id="n0", sample_id="ns0", trial_index=33
    )
    reporter.trial_finished(
        agent_label="a", trial_id="n0", status="completed", duration_ms=10, exit_code=0
    )
    snap = reporter.snapshot()
    assert snap["total"] == 27 and snap["done"] == 1, snap


def test_print_interrupt_banner_emits_final_line_lock_free() -> None:
    """The forced-exit (Ctrl+C×2) banner always prints a final status line."""
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(
        total_trials=27, runnable_trials=27, enabled=True, stream=io.StringIO()
    )
    reporter.start()
    reporter._print_progress()
    reporter.print_interrupt_banner()
    output = reporter.stream.getvalue()
    # Immediate "second Ctrl+C registered, force-quitting now" acknowledgment so
    # the user gets feedback before the slower table render + teardown latency.
    assert "force-quitting now" in output
    assert "0/27 done" in output
    assert "force-interrupted" in output


def test_print_interrupt_banner_shows_live_inspector_url_not_placeholder() -> None:
    """Ctrl+C×2 banner prints the live inspector URL + run dir, not a template."""
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter
    from cage.cli.ui.run import RunContract

    contract = RunContract(
        run_id="run-x",
        run_dir="/runs/agent/run-x",
        board_url="http://127.0.0.1:7777",
        run_url="http://127.0.0.1:7777/run/agent/run-x",
    )
    reporter = RunProgressReporter(
        total_trials=9,
        runnable_trials=9,
        enabled=True,
        contract=contract,
        stream=io.StringIO(),
    )
    reporter.start()
    reporter._print_progress()
    reporter.print_interrupt_banner()
    output = reporter.stream.getvalue()
    # The live inspector URL appears (board outlives the run) ...
    assert "http://127.0.0.1:7777/run/agent/run-x" in output
    # ... and the durable run dir for `cage inspect`, never the old placeholder.
    assert "/runs/agent/run-x" in output
    assert "<run-dir>" not in output


def test_print_graceful_stop_notice_explains_drain_and_force_quit() -> None:
    """First Ctrl+C prints the graceful-stop contract (drain + force-quit)."""
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(
        total_trials=9, runnable_trials=9, enabled=True, stream=io.StringIO()
    )
    reporter.start()
    reporter.print_graceful_stop_notice()
    output = reporter.stream.getvalue()
    assert "stopping gracefully" in output
    assert "in-flight" in output
    assert "cancelled" in output
    assert "again" in output


def test_run_contract_plain_text_contains_static_run_configuration() -> None:
    from cage.cli.ui.run import RunAgentContract, RunContract

    contract = RunContract(
        project_name="agent-pentest-bench-smoke-deepseek-v4-pro-web",
        benchmark_name="AgentPentestBench",
        benchmark_path="examples/agent_pentest_bench",
        project_file="examples/agent_pentest_bench/default_web_exploit.yml",
        run_id="run-ui-smoke",
        run_dir="examples/agent_pentest_bench/.cage_runs/agent/run-ui-smoke",
        board_url="http://127.0.0.1:7777",
        run_url="http://127.0.0.1:7777/run/agent/run-ui-smoke",
        dashboard_url="http://127.0.0.1:7777/run/agent/run-ui-smoke/dashboard",
        inspect_command="cage inspect examples/agent_pentest_bench",
        run_log_path="examples/agent_pentest_bench/.cage_runs/agent/run-ui-smoke/.cage.runlog",
        debug_log_path="examples/agent_pentest_bench/.cage_runs/agent/run-ui-smoke/.cage.debuglog",
        planned_trials=3,
        runnable_trials=1,
        passk=3,
        levels="l0,l1",
        samples="pb-siyucms",
        max_trials_global=2,
        max_target_setups=1,
        trial_timeout_s=1200,
        request_timeout_s=1200,
        target_startup_timeout_s=600,
        target_compose_timeout_s=1200,
        effective_max_rounds="150",
        judge_max_tokens="4096",
        max_input_tokens="100000",
        max_output_tokens="20000",
        max_cost="$12.34",
        agents=[
            RunAgentContract(
                agent_id="claude_code_deepseek_v4_pro_smoke",
                label="claude_code_deepseek_v4_pro_smoke:deepseek-v4-pro:stateless",
                kind="claude_code",
                model_id="deepseek-v4-pro",
                provider="openai",
                model_name="deepseek-chat",
                agent_model_name="deepseek-chat[1m]",
                image="cage/claude-code:pentestenv",
                max_concurrent=1,
                max_rounds="benchmark default",
                session_args="--permission-mode bypassPermissions --verbose",
            )
        ],
    )

    output = contract.to_plain_text()

    assert output.startswith("Pre-flight checks passed.")
    assert "_________   ____________" not in output
    assert not output.startswith("=" * 72)
    assert "Pre-flight checks passed. Cage is entering the benchmark run." in output
    assert "Open a browser URL below for live trials, logs, artifacts, and scores." in output
    assert "Browser inspector" in output
    assert "Run selection" in output
    assert "http://127.0.0.1:7777" in output
    assert "benchmark: AgentPentestBench" in output
    assert "project: agent-pentest-bench-smoke-deepseek-v4-pro-web" in output
    assert "run id: run-ui-smoke" in output
    assert "samples: pb-siyucms" in output
    assert "planned trials: 3 total before filters and caps" in output
    assert "trials to run: 1 runnable after filters and caps" in output
    assert "pass@k attempts: 3 per sample" in output
    assert "prompt/hint levels: l0,l1" in output
    assert "max rounds: 150" in output
    assert "max input tokens: 100000" in output
    assert "max output tokens: 20000" in output
    assert "judge=4096" not in output
    assert "max cost: $12.34" in output
    assert "agent: claude_code_deepseek_v4_pro_smoke" in output
    assert "model: deepseek-v4-pro" in output
    assert "endpoint model: deepseek-chat" in output
    assert "agent model: deepseek-chat[1m]" in output
    assert "run log: examples/agent_pentest_bench/.cage_runs/agent/run-ui-smoke/.cage.runlog" in output
    assert "debug log: examples/agent_pentest_bench/.cage_runs/agent/run-ui-smoke/.cage.debuglog" in output
    assert "--permission-mode bypassPermissions --verbose" in output
    assert "local url: http://127.0.0.1:7777/run/agent/run-ui-smoke" in output
    assert "/dashboard" not in output
    assert "\033[" not in output


def test_run_contract_plain_text_can_emit_ansi_color() -> None:
    from cage.cli.ui.run import RunContract, RunViewLink

    output = RunContract(
        project_name="web-exploit-bench",
        benchmark_id="web_exploit_bench",
        benchmark_name="agent_pentest_bench",
        run_id="run-1",
        view_links=[RunViewLink(base_url="http://127.0.0.1:7777")],
        planned_trials=1,
        runnable_trials=1,
        passk=1,
        levels="l2",
        samples="pb-siyucms",
    ).to_plain_text(color=True)

    assert "\033[" in output
    assert "_________   ____________" not in output
    assert "Browser inspector" in output
    assert "http://127.0.0.1:7777" in output


def test_run_contract_prefers_cli_benchmark_id_over_suite_name() -> None:
    from cage.cli.ui.run import build_run_contract

    config = SimpleNamespace(
        name="web-exploit-bench",
        project_file=Path("project.yml"),
        benchmark=SimpleNamespace(
            name="agent_pentest_bench",
        ),
        sample_ids=("pb-siyucms",),
        benchmark_dir=Path("examples/agent_pentest_bench"),
        metadata={"benchmark_id": "web_exploit_bench"},
        agents=[],
        execution=SimpleNamespace(
            passk=1,
            max_trials_global=1,
            max_target_setups=1,
            timeout=7200,
            max_rounds=-1,
            max_input_tokens=None,
            max_output_tokens=None,
            max_cost=None,
        ),
        proxy=SimpleNamespace(request_timeout=3600),
        target=SimpleNamespace(startup_timeout=1800, compose_up_timeout=3600),
        logging=LoggingConfig(terminal_ui=True),
    )

    contract = build_run_contract(
        config=config,
        run_id="run-1",
        run_dir=Path(".cage_runs/agent/run-1"),
        planned_trials=1,
        runnable_trials=1,
        samples=[{"id": "pb-siyucms", "max_rounds": 150}],
    )
    output = contract.to_plain_text()

    assert "benchmark: web_exploit_bench" in output
    assert "suite: agent_pentest_bench" in output


def test_run_contract_displays_post_exploit_hint_levels() -> None:
    from cage.cli.ui.run import build_run_contract

    config = SimpleNamespace(
        name="post-exploit-bench",
        project_file=Path("project.yml"),
        benchmark=SimpleNamespace(
            name="agent_pentest_bench",
            variant_display_axes=lambda: {"prompt": ("l0",), "hint": ("2",)},
        ),
        benchmark_dir=Path("examples/agent_pentest_bench"),
        metadata={"benchmark_id": "post_exploit_bench"},
        agents=[],
        execution=SimpleNamespace(
            passk=2,
            max_trials_global=2,
            max_target_setups=1,
            timeout=7200,
            max_rounds=-1,
            max_input_tokens=None,
            max_output_tokens=None,
            max_cost=None,
        ),
        proxy=SimpleNamespace(request_timeout=3600),
        target=SimpleNamespace(startup_timeout=1800, compose_up_timeout=3600),
        logging=LoggingConfig(terminal_ui=True),
    )

    contract = build_run_contract(
        config=config,
        run_id="postexp-opus46-authp-l2-p2-20260603",
        run_dir=Path(".cage_runs/agent/run-1"),
        planned_trials=16,
        runnable_trials=14,
        samples=[],
    )
    output = contract.to_plain_text()

    assert contract.levels == "l2"
    assert "prompt/hint levels: l2" in output
    assert "prompt/hint levels: l0" not in output


def test_run_contract_displays_launch_build_policy_and_zero_round_mode() -> None:
    from cage.cli.ui.run import build_run_contract

    config = SimpleNamespace(
        name="web-exploit-bench",
        project_file=Path("project.yml"),
        benchmark=SimpleNamespace(
            name="agent_pentest_bench",
        ),
        sample_ids=("pb-siyucms",),
        benchmark_dir=Path("examples/agent_pentest_bench"),
        metadata={"benchmark_id": "web_exploit_bench", "launch_build": "disabled"},
        agents=[],
        execution=SimpleNamespace(
            passk=1,
            max_trials_global=1,
            max_target_setups=2,
            timeout=7200,
            max_rounds=0,
            max_input_tokens=None,
            max_output_tokens=None,
            max_cost=None,
        ),
        proxy=SimpleNamespace(request_timeout=3600),
        target=SimpleNamespace(
            enabled=True,
            startup_timeout=1800,
            compose_up_timeout=3600,
        ),
        logging=LoggingConfig(terminal_ui=True),
    )

    contract = build_run_contract(
        config=config,
        run_id="run-zero",
        run_dir=Path(".cage_runs/agent/run-zero"),
        planned_trials=1,
        runnable_trials=1,
        samples=[{"id": "pb-siyucms", "max_rounds": 150}],
    )

    output = contract.to_plain_text()

    assert "Target lifecycle" in output
    assert "target server: enabled" in output
    assert "launch build: disabled; run cage benchmark build before cage run" in output
    assert "target setup cap: max_target_setups=2" in output
    assert "Zero-round mode" in output
    assert "target setup: enabled; targets still launch and tear down" in output
    assert "agent/proxy/model: skipped; no model calls" in output


def test_run_contract_parameter_text_is_web_first_and_operator_friendly() -> None:
    from cage.cli.ui.run import RunAgentContract, RunContract, RunViewLink

    contract = RunContract(
        project_name="agent-pentest-bench-web",
        benchmark_name="AgentPentestBench",
        benchmark_path="examples/agent_pentest_bench",
        project_file="examples/agent_pentest_bench/default_web_exploit.yml",
        run_id="web-claude-deepseek-smoke-001",
        run_dir="examples/agent_pentest_bench/.cage_runs/agent/run-1",
        board_url="http://0.0.0.0:7777",
        run_url="http://127.0.0.1:7777/run/agent/run-1",
        dashboard_url="http://127.0.0.1:7777/run/agent/run-1/dashboard",
        inspect_command="cage inspect examples/agent_pentest_bench",
        view_links=[
            RunViewLink(
                label="localhost",
                base_url="http://127.0.0.1:7777",
                run_url="http://127.0.0.1:7777/run/agent/run-1",
                dashboard_url="http://127.0.0.1:7777/run/agent/run-1/dashboard",
            ),
            RunViewLink(
                label="bind address",
                base_url="http://0.0.0.0:7777",
                run_url="http://0.0.0.0:7777/run/agent/run-1",
                dashboard_url="http://0.0.0.0:7777/run/agent/run-1/dashboard",
            ),
            RunViewLink(
                label="LAN 10.1.2.3",
                base_url="http://10.1.2.3:7777",
                run_url="http://10.1.2.3:7777/run/agent/run-1",
                dashboard_url="http://10.1.2.3:7777/run/agent/run-1/dashboard",
            ),
        ],
        planned_trials=3,
        runnable_trials=1,
        passk=1,
        levels="l0",
        samples="pb-siyucms",
        max_trials_global=1,
        max_target_setups=1,
        trial_timeout_s="7200s",
        request_timeout_s="3600s",
        effective_max_rounds="150",
        max_input_tokens="unlimited",
        max_output_tokens="unlimited",
        max_cost="unlimited",
        agents=[
            RunAgentContract(
                agent_id="claude_code",
                label="claude_code:deepseek-v4-pro:stateless",
                kind="claude_code",
                model_id="deepseek-v4-pro",
                provider="openai",
                max_concurrent=1,
                session_args="--permission-mode bypassPermissions",
            )
        ],
    )

    text = contract.to_plain_text()

    assert "Browser inspector" in text
    assert "Run selection" in text
    assert "Stop conditions" in text
    assert "Agent / model" in text
    assert "Logs" in text
    assert "open browser: live trials, logs, artifacts, and scores" in text
    assert "terminal output: summary plus one progress line; details stay in web/logs" in text
    assert "network url: http://10.1.2.3:7777/run/agent/run-1" in text
    assert "local url: http://127.0.0.1:7777/run/agent/run-1" in text
    # 0.0.0.0 is a bind-only wildcard — never shown as a clickable URL.
    assert "0.0.0.0:7777/run/agent/run-1" not in text
    assert "/dashboard" not in text


def test_run_contract_formats_unset_runtime_budgets_as_unlimited() -> None:
    from cage.cli.ui.run import build_run_contract

    config = SimpleNamespace(
        name="demo",
        project_file=Path("project.yml"),
        benchmark=SimpleNamespace(
            name="AgentPentestBench",
        ),
        sample_ids=("pb-siyucms",),
        benchmark_dir=Path("examples/agent_pentest_bench"),
        agents=[],
        execution=SimpleNamespace(
            passk=1,
            max_trials_global=1,
            max_target_setups=1,
            timeout=7200,
            max_rounds=-1,
            max_input_tokens=None,
            max_output_tokens=None,
            max_cost=None,
        ),
        proxy=SimpleNamespace(request_timeout=3600),
        target=SimpleNamespace(startup_timeout=1800, compose_up_timeout=3600),
        logging=LoggingConfig(terminal_ui=True),
    )

    contract = build_run_contract(
        config=config,
        run_id="run-1",
        run_dir=Path(".cage_runs/agent/run-1"),
        planned_trials=1,
        runnable_trials=1,
        samples=[{"id": "pb-siyucms", "max_rounds": 150}],
    )

    assert contract.max_input_tokens == "unlimited"
    assert contract.max_output_tokens == "unlimited"
    assert contract.max_cost == "unlimited"


def test_run_contract_plain_text_marks_inspector_disabled_without_urls() -> None:
    from cage.cli.ui.run import RunContract

    text = RunContract(
        project_name="demo",
        benchmark_name="AgentPentestBench",
        run_id="run-1",
    ).to_plain_text()

    assert "web: disabled" in text
    assert "127.0.0.1:7777" not in text
    assert "0.0.0.0:7777" not in text


def test_run_contract_does_not_invent_default_inspector_port() -> None:
    from cage.cli.ui.run import RunContract

    text = RunContract(
        project_name="demo",
        benchmark_name="AgentPentestBench",
        run_id="run-1",
        board_url="http://inspector.internal",
    ).to_plain_text()

    assert "http://inspector.internal" in text
    assert ":7777" not in text


def test_run_contract_limits_browser_urls_to_one_per_reachability_kind() -> None:
    from cage.cli.ui.run import RunContract, RunViewLink

    text = RunContract(
        project_name="demo",
        benchmark_name="AgentPentestBench",
        run_id="run-1",
        view_links=[
            RunViewLink(base_url="http://127.0.0.1:8090"),
            RunViewLink(base_url="http://0.0.0.0:8090"),
            RunViewLink(base_url="http://192.0.2.10:8090"),
            RunViewLink(base_url="http://172.17.0.1:8090"),
            RunViewLink(base_url="http://172.18.0.1:8090"),
        ],
    ).to_plain_text()

    assert "network url: http://192.0.2.10:8090" in text
    assert "local url: http://127.0.0.1:8090" in text
    # The wildcard bind address is never connectable from a browser, so it must
    # not be offered as a clickable URL.
    assert "0.0.0.0:8090" not in text
    assert "http://172.17.0.1:8090" not in text
    assert "http://172.18.0.1:8090" not in text


def test_effective_max_rounds_prefers_runtime_override_over_sample_default() -> None:
    from cage.cli.ui.run import _effective_max_rounds

    config = SimpleNamespace(
        execution=SimpleNamespace(max_rounds=5),
        agents=[SimpleNamespace(max_rounds=-1)],
    )
    samples = [{"id": "pb-siyucms", "max_rounds": 150}]

    assert _effective_max_rounds(config, samples) == "5"


def test_effective_max_rounds_displays_zero_runtime_override() -> None:
    from cage.cli.ui.run import _effective_max_rounds

    config = SimpleNamespace(
        execution=SimpleNamespace(max_rounds=0),
        agents=[SimpleNamespace(max_rounds=-1)],
    )
    samples = [{"id": "pb-siyucms", "max_rounds": 150}]

    assert _effective_max_rounds(config, samples) == "0"


def test_effective_max_rounds_negative_runtime_defers_to_sample_default() -> None:
    from cage.cli.ui.run import _effective_max_rounds

    config = SimpleNamespace(
        execution=SimpleNamespace(max_rounds=-1),
        agents=[SimpleNamespace(max_rounds=-1)],
    )
    samples = [{"id": "pb-siyucms", "max_rounds": 150}]

    assert _effective_max_rounds(config, samples) == "150"


def test_run_progress_reporter_prints_plain_start_banner_and_progress() -> None:
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter
    from cage.cli.ui.run import (
        RunAgentContract,
        RunContract,
        RunViewLink,
    )
    from cage.contracts.telemetry import ModelRequestEvent

    contract = RunContract(
        project_name="demo",
        benchmark_name="AgentPentestBench",
        project_file="project.yml",
        run_id="run-1",
        run_dir=".cage_runs/agent/run-1",
        board_url="http://127.0.0.1:7777",
        run_url="http://127.0.0.1:7777/run/agent/run-1",
        dashboard_url="http://127.0.0.1:7777/run/agent/run-1/dashboard",
        inspect_command="cage inspect examples/agent_pentest_bench",
        view_links=[
            RunViewLink(base_url="http://127.0.0.1:7777"),
            RunViewLink(base_url="http://0.0.0.0:7777"),
            RunViewLink(base_url="http://10.1.2.3:7777"),
        ],
        planned_trials=1,
        runnable_trials=1,
        passk=1,
        agents=[
            RunAgentContract(
                agent_id="agent-a",
                label="agent-a:model:stateless",
                kind="claude_code",
                model_id="model-a",
            )
        ],
    )
    buffer = io.StringIO()
    reporter = RunProgressReporter(total_trials=1, enabled=True, contract=contract, stream=buffer)

    with reporter.live():
        reporter.trial_started(
            agent_label="agent-a:model:stateless",
            trial_id="pb-siyucms/pass_1",
            sample_id="pb-siyucms",
            trial_index=0,
        )
        reporter.update_trial_progress(
            agent_label="agent-a:model:stateless",
            trial_id="pb-siyucms/pass_1",
            progress={
                "successful_requests": 7,
                "tokens_in": 123_000,
                "tokens_out": 4_500,
                "tokens_reasoning": 700,
                "errors": 1,
                "cost_usd": 0.42,
            },
        )
        reporter.record_model_request(
            ModelRequestEvent(
                trial_id="pb-siyucms/pass_1",
                step=7,
                status="success",
                latency_s=8.4,
                input_tokens=9000,
                output_tokens=612,
                reasoning_tokens=0,
                cost_usd=0.03,
            )
        )
        reporter.trial_finished(
            agent_label="agent-a:model:stateless",
            trial_id="pb-siyucms/pass_1",
            status="completed",
            duration_ms=1000,
            exit_code=0,
        )
    output = buffer.getvalue()

    assert "_________   ____________" not in output
    assert "Browser inspector" in output
    assert "AgentPentestBench" in output
    assert "http://127.0.0.1:7777" in output
    # 0.0.0.0 is bind-only; it is never offered as a clickable URL.
    assert "http://0.0.0.0:7777" not in output
    assert "http://10.1.2.3:7777" in output
    assert "progress [" in output
    assert "1/1 done" in output
    assert "llm_calls=7" in output
    assert "tokens=9,000/612" in output
    assert "cost=$0.03" in output


def test_run_progress_line_uses_compact_single_line_copy() -> None:
    import time

    from cage.cli.ui.progress_reporter import RunProgressReporter
    from cage.contracts.telemetry import ModelRequestEvent

    reporter = RunProgressReporter(total_trials=3, enabled=False)
    reporter._start_time = time.time() - 83
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-2",
        sample_id="sample-2",
        trial_index=1,
    )
    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-3",
        status="completed",
        duration_ms=1000,
        exit_code=0,
    )
    for index in range(12):
        reporter.record_model_request(
            ModelRequestEvent(
                trial_id=f"trial-{index}",
                step=index + 1,
                status="success",
                input_tokens=36_829 if index < 11 else 36_835,
                output_tokens=416 if index < 11 else 424,
                cost_usd=0.0,
            )
        )

    line = reporter._progress_line()

    assert line == (
        "progress [########----------------] 33% 1/3 done, running=2, "
        "failed=0, llm_calls=12, tokens=441,954/5,000, elapsed=1m23s"
    )


def test_run_progress_reporter_counts_model_calls_from_progress() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter
    from cage.contracts.telemetry import ModelRequestEvent

    reporter = RunProgressReporter(total_trials=1, enabled=False)
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.update_trial_progress(
        agent_label="agent-a",
        trial_id="trial-1",
        progress={
            "successful_requests": 10,
            "tokens_in": 1_000,
            "tokens_out": 200,
            "errors": 0,
        },
    )
    reporter.record_model_request(
        ModelRequestEvent(
            trial_id="trial-1",
            step=10,
            status="success",
            input_tokens=1_000,
            output_tokens=200,
        )
    )

    assert reporter.snapshot()["model_calls"] == 10


def test_run_progress_reporter_preserves_progress_calls_after_finish() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(total_trials=1, enabled=False)
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.update_trial_progress(
        agent_label="agent-a",
        trial_id="trial-1",
        progress={
            "successful_requests": 10,
            "tokens_in": 1_000,
            "tokens_out": 200,
            "errors": 0,
        },
    )

    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-1",
        status="completed",
        duration_ms=1000,
        exit_code=0,
    )

    assert reporter.snapshot()["model_calls"] == 10


def test_run_progress_line_handles_zero_trial_smoke() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(total_trials=0, enabled=False)

    assert reporter._progress_line() == (
        "progress [------------------------] no trials to run, llm_calls=0, "
        "tokens=0/0, elapsed=0s"
    )


def test_run_progress_line_can_emit_ansi_color(monkeypatch) -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter

    monkeypatch.setenv("CAGE_COLOR", "always")
    reporter = RunProgressReporter(total_trials=0, enabled=False)

    line = reporter._progress_line()

    assert "\033[" in line
    assert "no trials to run" in line


def test_run_progress_reporter_clears_and_redraws_single_tty_line() -> None:
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter

    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = TtyBuffer()
    reporter = RunProgressReporter(total_trials=1, enabled=True, stream=stream)

    reporter.start()
    reporter.clear_for_external_write()
    stream.write("log message\n")
    reporter.redraw_after_external_write()
    reporter.stop()
    output = stream.getvalue()

    assert "\rprogress [" in output
    expected_line = (
        "progress [------------------------] 0% 0/1 done, running=0, "
        "failed=0, llm_calls=0, tokens=0/0, elapsed=0s"
    )
    assert "\r" + (" " * len(expected_line)) + "\r" in output
    assert "log message\n\rprogress [" in output


def test_run_progress_reporter_rewrites_one_line_for_plain_streams() -> None:
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter

    stream = io.StringIO()
    reporter = RunProgressReporter(total_trials=1, enabled=True, stream=stream)

    reporter.start()
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-1",
        status="completed",
        duration_ms=1000,
        exit_code=0,
    )
    reporter.stop()

    output = stream.getvalue()
    assert "\rprogress [" in output
    assert output.count("\n") == 1
    assert output.endswith("\n")


def test_run_progress_reporter_skips_duplicate_redraws() -> None:
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter

    stream = io.StringIO()
    reporter = RunProgressReporter(total_trials=0, enabled=True, stream=stream)

    reporter.start()
    reporter.agent_started("agent-a", total_trials=0)
    reporter.stop()

    output = stream.getvalue()
    assert output.count("\rprogress [") == 1
    assert output.endswith("\n")


def test_progress_aware_console_handler_keeps_logs_off_progress_line() -> None:
    import io

    from cage.cli.ui.progress_reporter import RunProgressReporter
    from cage.contracts.logging import _ProgressAwareConsoleHandler

    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = TtyBuffer()
    reporter = RunProgressReporter(total_trials=1, enabled=True, stream=stream)
    handler = _ProgressAwareConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="container started",
        args=(),
        exc_info=None,
    )

    reporter.start()
    try:
        handler.emit(record)
    finally:
        reporter.stop()
        handler.close()

    output = stream.getvalue()
    assert "progress [------------------------] 0% 0/1 done" in output
    assert "container started\n\rprogress [" in output
    assert "progress [------------------------] 0/1 donecontainer started" not in output


def _long_path_dashboard_contract() -> object:
    from cage.cli.ui.run import RunAgentContract, RunContract, RunViewLink

    long_run_dir = (
        "/path/to/cage/.worktrees/open-source-release-prep/"
        ".cage_runs/claude_code:deepseek-v4-pro:stateless/"
        "web-claude-deepseek-smoke-001"
    )
    long_run_url = (
        "http://127.0.0.1:8090/run/"
        "claude_code:deepseek-v4-pro:stateless/"
        "web-claude-deepseek-smoke-001"
    )
    long_dashboard_url = f"{long_run_url}/dashboard"
    return RunContract(
        project_name="agent-pentest-bench-smoke",
        benchmark_name="AgentPentestBench",
        benchmark_path=(
            "/path/to/cage/.worktrees/open-source-release-prep/"
            "examples/agent_pentest_bench"
        ),
        project_file=(
            "/path/to/cage/.worktrees/open-source-release-prep/"
            "examples/agent_pentest_bench/default_web_exploit.yml"
        ),
        run_id="web-claude-deepseek-smoke-001",
        run_dir=long_run_dir,
        board_url="http://127.0.0.1:8090",
        run_url=long_run_url,
        dashboard_url=long_dashboard_url,
        inspect_command=(
            "cage inspect "
            "/path/to/cage/.worktrees/open-source-release-prep/"
            "examples/agent_pentest_bench"
        ),
        view_links=[
            RunViewLink(base_url="http://127.0.0.1:8090"),
            RunViewLink(base_url="http://0.0.0.0:8090"),
            RunViewLink(base_url="http://10.1.2.3:8090"),
        ],
        planned_trials=1,
        runnable_trials=1,
        passk=1,
        levels="l0",
        samples="pb-siyucms",
        max_trials_global=1,
        max_target_setups=1,
        trial_timeout_s="7200s",
        request_timeout_s="3600s",
        effective_max_rounds="150",
        max_input_tokens="unlimited",
        max_output_tokens="unlimited",
        max_cost="unlimited",
        agents=[
            RunAgentContract(
                agent_id="claude_code",
                label="claude_code:deepseek-v4-pro:stateless",
                kind="claude_code",
                model_id="deepseek-v4-pro",
                provider="openai",
                model_name="deepseek-chat",
                image="cage/claude-code:pentestenv",
                max_concurrent=1,
                session_args="--permission-mode bypassPermissions --verbose",
            )
        ],
    )


def test_run_parameter_text_right_aligns_field_labels() -> None:
    from cage.cli.ui.run import format_run_parameter_text

    text = format_run_parameter_text(_long_path_dashboard_contract())
    lines = text.splitlines()
    field_lines = [line for line in lines if ": " in line]
    colon_positions = {line.index(":") for line in field_lines}

    assert len(colon_positions) == 1
    assert "local url: http://127.0.0.1:8090" in text
    assert "     max rounds: 150" in text
    assert "  trial timeout: 7200s" in text


def test_run_parameter_text_groups_operator_questions() -> None:
    from cage.cli.ui.run import format_run_parameter_text

    text = format_run_parameter_text(_long_path_dashboard_contract())

    assert text.index("Browser inspector\n") < text.index("Run selection\n")
    assert text.index("Run selection\n") < text.index("Stop conditions\n")
    assert text.index("Stop conditions\n") < text.index("Agent / model\n")
    assert text.index("Agent / model\n") < text.index("Logs\n")

    view_section = text[
        text.index("Browser inspector\n"):text.index("Run selection\n")
    ]
    assert (
        "local url: http://127.0.0.1:8090/run/"
        "claude_code:deepseek-v4-pro:stateless/web-claude-deepseek-smoke-001"
    ) in view_section
    # 0.0.0.0 is bind-only — never offered as a clickable URL.
    assert "0.0.0.0:8090" not in view_section
    assert (
        "network url: http://10.1.2.3:8090/run/"
        "claude_code:deepseek-v4-pro:stateless/web-claude-deepseek-smoke-001"
    ) in view_section
    assert "open browser: live trials, logs, artifacts, and scores" in view_section
    assert "/dashboard" not in view_section

    stop_section = text[text.index("Stop conditions\n"):text.index("Agent / model\n")]
    assert "     max rounds: 150" in stop_section
    assert "  trial timeout: 7200s" in stop_section
    assert "max input tokens: unlimited" in stop_section
    assert "judge max tokens" not in stop_section
    assert "       max cost: unlimited" in stop_section


def test_run_ui_module_has_no_dashboard_or_rich_surface() -> None:
    import cage.cli.ui.run as run_ui

    assert not hasattr(run_ui, "CageRunDashboardApp")
    assert not hasattr(run_ui, "RunParameterPanels")
    assert not hasattr(run_ui, "Live")
    assert not hasattr(run_ui, "Progress")
    assert not hasattr(run_ui, "Table")


def test_create_run_reporter_returns_plain_progress_reporter() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter, create_run_reporter

    reporter = create_run_reporter(
        enabled=True,
        total_trials=1,
    )

    assert isinstance(reporter, RunProgressReporter)
    assert reporter.enabled is True


def test_run_progress_reporter_shows_most_recent_finished_trials() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(total_trials=3, enabled=False)

    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-1",
        status="completed",
        duration_ms=1500,
        exit_code=0,
    )
    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-2",
        sample_id="sample-2",
        trial_index=1,
    )
    reporter.trial_finished(
        agent_label="agent-a",
        trial_id="trial-2",
        status="failed",
        duration_ms=500,
        exit_code=1,
    )

    assert reporter.recent_trials() == ["failed: trial-2", "completed: trial-1"]


def test_run_progress_reporter_updates_active_trial_status() -> None:
    from cage.cli.ui.progress_reporter import RunProgressReporter

    reporter = RunProgressReporter(total_trials=1, enabled=False)

    reporter.trial_started(
        agent_label="agent-a",
        trial_id="trial-1",
        sample_id="sample-1",
        trial_index=0,
    )
    reporter.update_trial_status(
        agent_label="agent-a",
        trial_id="trial-1",
        message="scoring",
    )

    assert reporter.snapshot()["active"][0]["status"] == "scoring"


def test_create_run_reporter_keeps_progress_enabled_for_normal_log_streams() -> None:
    from cage.cli.ui.progress_reporter import create_run_reporter

    class NonTty:
        def isatty(self) -> bool:
            return False

    reporter = create_run_reporter(
        enabled=True,
        total_trials=1,
        stream=NonTty(),
    )

    assert reporter.enabled is True


def test_create_run_reporter_can_be_explicitly_disabled() -> None:
    from cage.cli.ui.progress_reporter import create_run_reporter

    reporter = create_run_reporter(enabled=False, total_trials=1)

    assert reporter.enabled is False


def test_quiet_console_logging_suppresses_console_handlers_but_keeps_files(
    tmp_path: Path,
) -> None:
    from cage.contracts.logging import quiet_console_logging

    root = logging.getLogger()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(tmp_path / "run.log")
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    try:
        with quiet_console_logging(enabled=True):
            assert stream_handler.level > logging.CRITICAL
            assert file_handler.level == logging.DEBUG
        assert stream_handler.level == logging.INFO
        assert file_handler.level == logging.DEBUG
    finally:
        root.removeHandler(stream_handler)
        root.removeHandler(file_handler)
        stream_handler.close()
        file_handler.close()


def test_quiet_console_logging_disabled_keeps_console_handlers_visible(
    tmp_path: Path,
) -> None:
    from cage.contracts.logging import quiet_console_logging

    root = logging.getLogger()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(tmp_path / "run.log")
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    try:
        with quiet_console_logging(enabled=False):
            assert stream_handler.level == logging.INFO
            assert file_handler.level == logging.DEBUG
        assert stream_handler.level == logging.INFO
        assert file_handler.level == logging.DEBUG
    finally:
        root.removeHandler(stream_handler)
        root.removeHandler(file_handler)
        stream_handler.close()
        file_handler.close()


def test_parallel_trial_runner_reports_trial_start_and_finish(
    tmp_path: Path, monkeypatch,
) -> None:
    import cage.experiment.engine.conductor as orchestrator
    from cage.experiment.model import Trial, TrialResult, TrialType
    from cage.sandbox.exec import Timing

    trial = Trial(
        id="trial-1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-1"},
    )
    storage = SimpleNamespace(
        run_dir=tmp_path / ".cage_runs" / "agent-a:model:stateless" / "run-1",
    )
    storage.run_dir.mkdir(parents=True)
    agent = SimpleNamespace(label=lambda: "agent-a:model:stateless")
    config = SimpleNamespace(scheduler=None, cleanup=None)
    hook_ctx = SimpleNamespace()
    events: list[tuple[str, str, str]] = []

    class Recorder:
        def trial_started(self, *, agent_label, trial_id, sample_id, trial_index):
            events.append(("started", agent_label, trial_id))

        def trial_finished(self, *, agent_label, trial_id, status, duration_ms, exit_code):
            events.append((status, agent_label, trial_id))

    def fake_run_trial_isolated(
        run_config, run_agent, run_trial, cage_runs, run_id, reporter=None, **_kwargs
    ):
        assert run_trial is trial
        assert reporter is not None
        return TrialResult(
            trial_id=run_trial.id,
            trial_index=run_trial.index,
            trial_type=run_trial.type.value,
            sample_id=run_trial.sample_id,
            output="ok",
            exit_code=0,
            timing=Timing(started_at_ms=0, ended_at_ms=100, duration_ms=100),
            metadata={"status": "completed"},
        )

    monkeypatch.setattr(orchestrator, "run_trial_isolated", fake_run_trial_isolated)

    results = orchestrator._run_agent_trials_parallel(
        config,
        agent,
        storage,
        [trial],
        hook_ctx,
        max_workers=1,
        passk=1,
        reporter=Recorder(),
    )

    assert [result.trial_id for result in results] == ["trial-1"]
    assert events == [
        ("started", "agent-a:model:stateless", "trial-1"),
        ("completed", "agent-a:model:stateless", "trial-1"),
    ]


def test_parallel_trial_runner_cancels_queued_trials_when_stopped(
    tmp_path: Path, monkeypatch
) -> None:
    """Graceful stop: a not-yet-started trial is cancelled before any launch."""
    import cage.experiment.engine.conductor as orchestrator
    from cage.experiment.model import Trial, TrialType

    trial = Trial(id="trial-1", index=0, type=TrialType.TASK, sample={"id": "sample-1"})
    storage = SimpleNamespace(
        run_dir=tmp_path / ".cage_runs" / "agent-a:model:stateless" / "run-1",
    )
    storage.run_dir.mkdir(parents=True)
    agent = SimpleNamespace(label=lambda: "agent-a:model:stateless")
    # Scheduler reports a graceful stop was requested.
    config = SimpleNamespace(
        scheduler=SimpleNamespace(is_stopped=lambda: True), cleanup=None
    )
    hook_ctx = SimpleNamespace()
    launched: list[str] = []

    def fake_run_trial_isolated(*_args, **_kwargs):
        launched.append("ran")
        raise AssertionError("a stopped run must not launch queued trials")

    monkeypatch.setattr(orchestrator, "run_trial_isolated", fake_run_trial_isolated)

    results = orchestrator._run_agent_trials_parallel(
        config, agent, storage, [trial], hook_ctx,
        max_workers=1, passk=1, reporter=None,
    )

    # Cancelled, not failed: no result row, and the trial never launched.
    assert results == []
    assert launched == []


def test_run_experiment_creates_and_passes_reporter(tmp_path: Path, monkeypatch) -> None:
    import cage.experiment.engine.conductor as orchestrator
    from cage.config.sections import ExecutionConfig, TargetConfig
    from cage.experiment.engine.run_context import ExperimentRun
    from cage.experiment.engine.hooks import HookRegistry

    class OneSampleBenchmark:
        name = "demo"

        def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
            return iter([{"id": "sample-1"}])

        def teardown(self) -> None:
            pass

    class PreflightOk:
        def to_dict(self):
            return {"ok": True}

    project_dir = tmp_path / "bench"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    agent = SimpleNamespace(label=lambda: "agent-a:model:stateless")
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(terminal_ui=True),
        execution=ExecutionConfig(passk=2),
        target=TargetConfig(enabled=False),
    )
    class Reporter:
        enabled = True

        def __init__(self) -> None:
            self.used_live_context = False

        @contextmanager
        def live(self):
            self.used_live_context = True
            yield self

    reporter = Reporter()
    created: dict[str, object] = {}
    passed: list[object] = []

    def fake_create_run_reporter(*, enabled, total_trials, **kwargs):
        created["enabled"] = enabled
        created["total_trials"] = total_trials
        created.update(kwargs)
        return reporter

    def fake_run_single_agent(run, run_agent):
        passed.append(run.reporter)
        return []

    monkeypatch.setattr(orchestrator, "create_run_id", lambda: "run-1")
    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: PreflightOk())
    monkeypatch.setattr(orchestrator, "_run_single_agent", fake_run_single_agent)
    monkeypatch.setattr(orchestrator, "_score_trials", lambda *_args, **_kwargs: None)

    orchestrator.run_experiment(config, make_reporter=fake_create_run_reporter)

    assert created["enabled"] is True
    assert created["total_trials"] == 2
    assert passed == [reporter]
    assert reporter.used_live_context is True


def test_run_experiment_quiets_console_logs_after_run_ui_starts(
    tmp_path: Path, monkeypatch,
) -> None:
    import cage.experiment.engine.conductor as orchestrator
    from cage.config.sections import ExecutionConfig, TargetConfig
    from cage.experiment.engine.run_context import ExperimentRun
    from cage.experiment.engine.hooks import HookRegistry

    class OneSampleBenchmark:
        name = "demo"

        def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
            return iter([{"id": "sample-1"}])

        def teardown(self) -> None:
            pass

    class PreflightOk:
        def to_dict(self):
            return {"ok": True}

    class Reporter:
        enabled = True

        @contextmanager
        def live(self):
            yield self

    project_dir = tmp_path / "bench"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    agent = SimpleNamespace(label=lambda: "agent-a:model:stateless")
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(terminal_ui=True),
        execution=ExecutionConfig(passk=1),
        target=TargetConfig(enabled=False),
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        monkeypatch.setattr(orchestrator, "create_run_id", lambda: "run-1")
        monkeypatch.setattr(
            "cage.experiment.engine.preflight.run_preflight",
            lambda *_args, **_kwargs: PreflightOk(),
        )
        monkeypatch.setattr(orchestrator, "_run_single_agent", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(orchestrator, "_score_trials", lambda *_args, **_kwargs: None)

        orchestrator.run_experiment(config, make_reporter=lambda **_kwargs: Reporter())
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    output = stream.getvalue()
    assert "Dashboard written" not in output
    assert "Experiment complete" not in output


def test_run_experiment_caps_reporter_total_by_max_trial(
    tmp_path: Path, monkeypatch,
) -> None:
    import cage.experiment.engine.conductor as orchestrator
    from cage.config.sections import ExecutionConfig, TargetConfig
    from cage.experiment.engine.run_context import ExperimentRun
    from cage.experiment.engine.hooks import HookRegistry

    class FourSampleBenchmark:
        name = "demo"

        def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
            return iter(
                [
                    {"id": "sample-1"},
                    {"id": "sample-2"},
                    {"id": "sample-3"},
                    {"id": "sample-4"},
                ]
            )

        def teardown(self) -> None:
            pass

    class PreflightOk:
        def to_dict(self):
            return {"ok": True}

    project_dir = tmp_path / "bench"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    agent = SimpleNamespace(label=lambda: "agent-a:model:stateless")
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=FourSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(terminal_ui=True),
        execution=ExecutionConfig(passk=3, max_trial=2),
        target=TargetConfig(enabled=False),
    )
    created: dict[str, object] = {}

    def fake_create_run_reporter(*, enabled, total_trials, **kwargs):
        created["enabled"] = enabled
        created["total_trials"] = total_trials
        created.update(kwargs)
        return object()

    monkeypatch.setattr(orchestrator, "create_run_id", lambda: "run-1")
    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: PreflightOk())
    monkeypatch.setattr(orchestrator, "_run_single_agent", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(orchestrator, "_score_trials", lambda *_args, **_kwargs: None)

    orchestrator.run_experiment(config, make_reporter=fake_create_run_reporter)

    assert created["enabled"] is True
    assert created["total_trials"] == 2


def test_run_experiment_run_contract_counts_resume_reruns(
    tmp_path: Path, monkeypatch,
) -> None:
    import cage.experiment.engine.conductor as orchestrator
    from cage.agents.base import AgentInstance, AgentType
    from cage.config.sections import ExecutionConfig, TargetConfig
    from cage.experiment.engine.run_context import ExperimentRun
    from cage.experiment.engine.hooks import HookRegistry
    from cage.models import ModelConfig

    class FakeAgentType(AgentType):
        name = "claude_code"

        def install_command(self, version: str = "latest") -> str:
            return "true"

        def build_launch_command(self, prompt, *, model, max_rounds=-1, proxy_url=""):
            return "true"

        def parse_output(self, result):
            return ""

    class TwoSampleBenchmark:
        name = "demo"

        def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
            return iter([
                {"id": "sample-1"},
                {"id": "sample-2"},
            ])

        def teardown(self) -> None:
            pass

    class PreflightOk:
        def to_dict(self):
            return {"ok": True}

    project_dir = tmp_path / "bench"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    model = ModelConfig(id="demo-model", provider="openai", model="demo")
    agent = AgentInstance(
        agent_type=FakeAgentType(),
        model=model,
        id="agent-a",
    )
    run_dir = project_dir / ".cage_runs" / agent.label() / "run-1"
    (run_dir / "trials" / "sample-1").mkdir(parents=True)
    (run_dir / "trials" / "sample-1" / "meta.json").write_text(
        json.dumps({
            "status": "failed",
            "termination_reason": "execution_timeout",
            "trial_index": 0,
            "trial_type": "task",
        }),
        encoding="utf-8",
    )
    (run_dir / "planned_trials.json").write_text(
        json.dumps([
            {
                "trial_id": "sample-1",
                "trial_index": 0,
                "trial_type": "task",
                "sample_id": "sample-1",
            },
            {
                "trial_id": "sample-2",
                "trial_index": 1,
                "trial_type": "task",
                "sample_id": "sample-2",
            },
        ]),
        encoding="utf-8",
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=TwoSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={"demo-model": model},
        hooks=HookRegistry(),
        logging=LoggingConfig(terminal_ui=True, inspect_mode="off"),
        execution=ExecutionConfig(passk=1),
        target=TargetConfig(enabled=False),
        run_id="run-1",
        resume=True,
    )
    from cage.cli.ui.progress_reporter import create_run_reporter_with_contract

    created: dict[str, object] = {}

    def fake_make_reporter(**kwargs):
        reporter = create_run_reporter_with_contract(**kwargs)
        created["enabled"] = kwargs["enabled"]
        created["total_trials"] = kwargs["total_trials"]
        created["contract"] = reporter.contract
        return reporter

    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: PreflightOk())
    monkeypatch.setattr(orchestrator, "_run_single_agent", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(orchestrator, "_score_trials", lambda *_args, **_kwargs: None)

    orchestrator.run_experiment(config, make_reporter=fake_make_reporter)

    contract = created["contract"]
    assert contract is not None
    assert contract.planned_trials == 2
    assert contract.runnable_trials == 1
    assert contract.resume_replayed_trials == 1
    assert created["total_trials"] == 2
    output = contract.to_plain_text()
    assert "trials to run: 1 runnable after filters, caps, and resume replay" in output
    assert "resume replayed: 1 kept from disk" in output


def test_run_experiment_starts_inspector_and_passes_run_contract(
    tmp_path: Path, monkeypatch,
) -> None:
    import cage.experiment.engine.conductor as orchestrator
    from cage.agents.base import AgentInstance, AgentType
    from cage.config.sections import (
        ExecutionConfig,
        JudgeConfig,
        ProxyConfig,
        TargetConfig,
    )
    from cage.experiment.engine.run_context import ExperimentRun
    from cage.experiment.engine.hooks import HookRegistry
    from cage.models import ModelConfig

    class FakeAgentType(AgentType):
        name = "claude_code"

        def install_command(self, version: str = "latest") -> str:
            return "true"

        def build_launch_command(self, prompt, *, model, max_rounds=-1, proxy_url=""):
            return "true"

        def parse_output(self, result):
            return ""

    class OneSampleBenchmark:
        name = "AgentPentestBench"

        def variant_display_axes(self):
            return {"prompt": ("l0",)}

        def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
            return iter([{"id": "pb-siyucms", "max_rounds": 150}])

        def teardown(self) -> None:
            pass

    class PreflightOk:
        def to_dict(self):
            return {"ok": True}

    project_dir = tmp_path / "bench"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\nruntime:\n  timeout: 600\n", encoding="utf-8")
    model = ModelConfig(
        id="deepseek-v4-pro",
        provider="openai",
        model="deepseek-chat",
        agent_model_names={"claude_code": "deepseek-chat[1m]"},
        base_url="http://model.local/v1",
    )
    agent = AgentInstance(
        agent_type=FakeAgentType(),
        model=model,
        id="agent-a",
        image="cage/claude-code:pentestenv",
        max_concurrent=1,
        session_args=["--permission-mode", "bypassPermissions"],
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=OneSampleBenchmark(),
        benchmark_dir=project_dir,
        sample_ids=("pb-siyucms",),
        agents=[agent],
        models={"deepseek-v4-pro": model},
        hooks=HookRegistry(),
        logging=LoggingConfig(terminal_ui=True, inspect_mode="on"),
        execution=ExecutionConfig(
            passk=1,
            timeout=1200,
            max_trials_global=1,
            max_target_setups=1,
            max_input_tokens=100000,
            max_output_tokens=20000,
            max_cost=12.34,
        ),
        proxy=ProxyConfig(request_timeout=1200),
        judge=JudgeConfig(model=model, max_tokens=4096),
        target=TargetConfig(enabled=False, startup_timeout=600, compose_up_timeout=1200),
    )
    created: dict[str, object] = {}
    inspect_calls: list[Path] = []

    from cage.cli.ui.progress_reporter import create_run_reporter_with_contract

    def fake_make_reporter(**kwargs):
        reporter = create_run_reporter_with_contract(**kwargs)
        created["enabled"] = kwargs["enabled"]
        created["total_trials"] = kwargs["total_trials"]
        created["contract"] = reporter.contract
        return reporter

    def fake_ensure_inspector_board(root, web_config, mode, interactive):
        inspect_calls.append(Path(root))
        return SimpleNamespace(
            enabled=True,
            started=True,
            url="http://127.0.0.1:7777",
            root=Path(root),
            pid=123,
            log_path=Path(root) / ".cage" / "inspect.log",
        )

    monkeypatch.setattr(orchestrator, "create_run_id", lambda: "run-1")
    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: PreflightOk())
    monkeypatch.setattr("cage.web.inspect_board.ensure_inspector_board", fake_ensure_inspector_board)
    monkeypatch.setattr(
        "cage.config.load_repo_config",
        lambda _root=None: SimpleNamespace(web_inspector=SimpleNamespace()),
    )
    monkeypatch.setattr(orchestrator, "_run_single_agent", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(orchestrator, "_score_trials", lambda *_args, **_kwargs: None)

    orchestrator.run_experiment(config, make_reporter=fake_make_reporter)

    assert inspect_calls == [project_dir]
    contract = created["contract"]
    assert contract is not None
    assert contract.project_name == "demo"
    assert contract.benchmark_name == "AgentPentestBench"
    assert contract.board_url == "http://127.0.0.1:7777"
    assert contract.run_url.startswith("http://127.0.0.1:7777/run/")
    assert not contract.run_url.endswith("/dashboard")
    assert contract.dashboard_url == f"{contract.run_url}/dashboard"
    assert contract.planned_trials == 1
    assert contract.runnable_trials == 1
    assert contract.passk == 1
    assert contract.levels == "l0"
    assert contract.samples == "pb-siyucms"
    assert contract.effective_max_rounds == "150"
    assert contract.judge_max_tokens == "4096"
    assert contract.max_input_tokens == "100000"
    assert contract.max_output_tokens == "20000"
    assert contract.max_cost == "$12.34"
    assert contract.agents[0].agent_id == "agent-a"
    assert contract.agents[0].model_id == "deepseek-v4-pro"
    assert contract.agents[0].model_name == "deepseek-chat"
    assert contract.agents[0].agent_model_name == "deepseek-chat[1m]"


def test_proxy_monitor_reports_progress_and_model_request_metadata(tmp_path: Path) -> None:
    import json

    from cage.proxy.monitor import _ProxyMonitor

    class NoExecContainer:
        def exec(self, _cmd, timeout=0):
            raise AssertionError("proxy progress monitor should not docker exec")

    class Recorder:
        def __init__(self):
            self.progress = []
            self.requests = []

        def update_trial_progress(self, **kwargs):
            self.progress.append(kwargs)

        def record_model_request(self, event):
            self.requests.append(event)

    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    (proxy_dir / "progress.json").write_text(
        json.dumps(
            {
                "total_requests": 3,
                "successful_requests": 2,
                "last_status": "success",
                "tokens_in": 10_000,
                "tokens_out": 500,
                "tokens_reasoning": 25,
                "errors": 1,
                "cost_usd": 0.12,
            }
        ),
        encoding="utf-8",
    )

    reporter = Recorder()
    monitor = _ProxyMonitor(
        container=NoExecContainer(),
        log_dir="/opt/cage-proxy/logs",
        trial_id="trial-1",
        artifact_dir=proxy_dir,
        reporter=reporter,
        agent_label="agent-a:model:stateless",
        poll_interval=0.1,
    )

    monitor._report()

    assert reporter.progress == [
        {
            "agent_label": "agent-a:model:stateless",
            "trial_id": "trial-1",
            "progress": {
                "total_requests": 3,
                "successful_requests": 2,
                "last_status": "success",
                "tokens_in": 10_000,
                "tokens_out": 500,
                "tokens_reasoning": 25,
                "errors": 1,
                "cost_usd": 0.12,
            },
        }
    ]
    assert len(reporter.requests) == 1
    event = reporter.requests[0]
    assert event.trial_id == "trial-1"
    assert event.step == 2
    assert event.status == "success"
    assert event.input_tokens == 10_000
    assert event.output_tokens == 500
    assert event.reasoning_tokens == 25
    assert event.cost_usd == 0.12
