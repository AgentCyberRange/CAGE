"""Tests for orchestrator live check lifecycle integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cage.config.sections import ExecutionConfig, LiveCheckConfig
from cage.experiment.engine.hooks import HookContext, HookRegistry
from cage.experiment.model import Trial, TrialType
from cage.contracts.logging import LoggingConfig
from cage.experiment.engine.conductor import _run_agent_trials_serial
from cage.experiment.engine.trial_runner import execute_trial
from cage.target.provisioning import create_check_network
from cage.target.services.submit.service import SubmitServiceHandle, needs_submit_service

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trial(sample_id: str = "sample-1", flag: str = "flag{test}") -> Trial:
    return Trial(
        id=f"trial-{sample_id}",
        index=0,
        type=TrialType.TASK,
        sample={"id": sample_id, "flag": flag},
    )


def _make_agent() -> SimpleNamespace:
    """Build a minimal agent object for execute_trial."""
    return SimpleNamespace(
        agent_type=SimpleNamespace(
            install_command=lambda v: "echo install",
            build_launch_command=lambda prompt, model=None, **kwargs: "echo run",
            parse_output=lambda r: r.stdout,
            env_vars=lambda proxy_url="", model=None, **kwargs: {},
            setup_container=lambda *a, **kw: None,
            version_command=lambda: "echo 1.0",
        ),
        model=SimpleNamespace(
            id="test-model", base_url="http://localhost",
            api_key="k", protocol="openai",
        ),
        home="/home/agent/workspace",
        shared_paths=[],
        stateful=False,
        effective_state_paths=[],
        effective_image="test:latest",
        label=lambda: "test-agent:test-model:stateless",
        session_timeout=60,
        session_args=[],
        max_rounds=None,
        context_compaction_threshold=None,
        extra_env={},
        plugins=[],
        version="latest",
    )


def _make_config(
    benchmark_name: str = "nyu_ctf",
    live_check_enabled: bool = True,
    live_check_max_calls: int = 3,
) -> SimpleNamespace:
    """Build a minimal ExperimentRun-like object for execute_trial."""
    benchmark = SimpleNamespace(
        name=benchmark_name,
        # Declarative live-check capabilities, mirroring the real example
        # benchmarks (nyu_ctf / autopenbench use the submit+check services;
        # cvebench has a built-in check).
        needs_submit_service=benchmark_name in {"nyu_ctf", "autopenbench"},
        needs_check_service=benchmark_name in {"nyu_ctf", "autopenbench"},
        uses_builtin_check=benchmark_name == "cvebench",
        prepare_trial=lambda container, sample, workspace_dir: None,
        build_prompt=lambda sample: "test prompt",
        scorer=lambda: SimpleNamespace(strategy="post_run"),
    )
    live_check = LiveCheckConfig(enabled=live_check_enabled, max_calls=live_check_max_calls)
    execution = ExecutionConfig(live_check=live_check, agent_network_mode="host")

    return SimpleNamespace(
        benchmark=benchmark,
        execution=execution,
        proxy=SimpleNamespace(enabled=False),
        ctf=SimpleNamespace(enabled=False),
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        name="test",
        project_file=Path("/tmp/test"),
    )


def _make_cvebench_config(gather) -> SimpleNamespace:
    config = _make_config(benchmark_name="cvebench", live_check_enabled=True)
    # Live gather is the scorer's job now: benchmark.scorer().gather(...). Return a
    # stable scorer object so the mock records every call across the trial.
    scorer_obj = SimpleNamespace(strategy="post_run", gather=gather)
    config.benchmark.scorer = lambda: scorer_obj
    return config


def _make_container() -> MagicMock:
    container = MagicMock()
    container.name = "test-ctr"
    container.is_running = True
    container.volumes = {"/host/trials/trial-sample-1/proxy": "/var/lib/cage/proxy"}
    container._runtime_network_name = None
    container.exec.return_value = SimpleNamespace(exit_code=0, stdout="", stderr="")
    container.exec_async.return_value = MagicMock(
        communicate=lambda timeout=60: ("output", ""),
        returncode=0,
    )
    container.get_version.return_value = "1.0"
    return container


def _make_storage(tmp_path: Path) -> SimpleNamespace:
    trial_dir = tmp_path / "trials" / "trial-sample-1"
    trial_dir.mkdir(parents=True, exist_ok=True)

    storage = SimpleNamespace(
        trial_dir=lambda tid: tmp_path / "trials" / tid,
        trial_state_pre_dir=lambda tid: tmp_path / "state_pre" / tid,
        trial_state_post_dir=lambda tid: tmp_path / "state_post" / tid,
        trial_proxy_dir=lambda tid: tmp_path / "proxy" / tid,
        save_trial_prompt=lambda tid, p: None,
        save_trial_meta=lambda tid, m: None,
        save_trial_output=lambda tid, m: None,
        initial_state_dir=lambda: tmp_path / "initial_state",
        log_file_path=lambda: tmp_path / "run.log",
        debug_log_path=lambda: None,
        run_dir=tmp_path,
    )
    # Ensure subdirectories exist
    for d in [
        "state_pre/trial-sample-1",
        "state_post/trial-sample-1",
        "proxy/trial-sample-1",
        "initial_state",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    return storage


def _make_hook_ctx(tmp_path: Path, sample: dict) -> HookContext:
    return HookContext(
        experiment_config={"name": "test"},
        samples=[sample],
        trials_completed=[],
        trials_pending=[],
        run_artifacts_dir=str(tmp_path),
    )


def _run_execute_trial(
    *,
    trial,
    config,
    container,
    storage,
    hook_ctx,
    run_id="run-1",
    agent=None,
):
    """Thin wrapper around execute_trial with a default agent."""
    from cage.experiment.engine.scheduler import RunScheduler

    return execute_trial(
        trial=trial,
        agent=agent or _make_agent(),
        run=config,
        container=container,
        storage=storage,
        hook_ctx=hook_ctx,
        scheduler=RunScheduler.inactive(),
        run_id=run_id,
    )


def test_create_check_network_uses_problem_check_net_prefix():
    completed = SimpleNamespace(returncode=0, stderr="")

    with patch("cage.target.provisioning.subprocess.run", return_value=completed) as mock_run:
        network_name = create_check_network(
            run_id="run:abc",
            trial_id="trial/sample",
        )

    assert network_name.startswith("problem-check-net-")
    cmd = mock_run.call_args[0][0]
    assert cmd[:4] == ["docker", "network", "create", "--driver"]
    assert cmd[-1] == network_name


@patch("cage.experiment.engine.conductor._run_agent_trials_parallel")
def test_serial_runner_uses_per_trial_isolated_executor(mock_parallel, tmp_path):
    mock_parallel.return_value = ["result"]
    config = _make_config(live_check_enabled=False)
    agent = _make_agent()
    storage = _make_storage(tmp_path)
    trial = _make_trial()
    hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

    result = _run_agent_trials_serial(
        config,
        agent,
        storage,
        [trial],
        hook_ctx,
        run_id="run-serial",
        passk=2,
    )

    assert result == ["result"]
    mock_parallel.assert_called_once_with(
        config,
        agent,
        storage,
        [trial],
        hook_ctx,
        max_workers=1,
        passk=2,
    )


# ---------------------------------------------------------------------------
# Tests: execute_trial starts check container when needed
# ---------------------------------------------------------------------------


class TestExecuteTrialSubmitService:
    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_starts_submit_service_for_nyu_ctf(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
            has_failures=False,
        )
        handle = MagicMock(spec=SubmitServiceHandle)
        mock_start.return_value = handle

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{secret}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        with patch("cage.target.provisioning.create_check_network") as mock_create:
            _run_execute_trial(
                trial=trial, config=config, container=container,
                storage=storage, hook_ctx=hook_ctx, run_id="run-123",
            )

        mock_start.assert_called_once()
        call_kwargs = mock_start.call_args[1]
        assert call_kwargs["question_id"] == "sample-1"
        assert call_kwargs["expected_answer"] == "flag{secret}"
        assert call_kwargs["max_checks"] == 3
        assert call_kwargs["benchmark"] == "nyu_ctf"
        assert call_kwargs["trial_id"] == "trial-sample-1"
        assert call_kwargs["container_artifact_path"] == ""
        mock_create.assert_not_called()
        container.connect_network.assert_not_called()

        # sample metadata set for prompt rendering
        assert trial.sample["check_supported"] is True
        assert trial.sample.get("check_endpoint") is None

    @patch("cage.experiment.engine.trial_runner.ReactiveLiveCheckMonitor")
    @patch("cage.experiment.engine.trial_runner.CheckDonePoller")
    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_starts_reactive_monitor_without_polling_by_default(
        self, mock_find_pid, mock_snapshot, mock_start, mock_poller, mock_reactive, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        mock_start.return_value = MagicMock(spec=SubmitServiceHandle)
        reactive = mock_reactive.return_value
        reactive.success_event.is_set.return_value = False
        reactive.verdict = None

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{secret}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-123",
        )

        mock_reactive.assert_called_once()
        reactive.start.assert_called_once()
        reactive.stop.assert_called_once()
        mock_poller.assert_not_called()

    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_launches_agent_from_agent_home(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        mock_start.return_value = MagicMock(spec=SubmitServiceHandle)

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{secret}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-123",
        )

        command = container.exec_async.call_args.args[0]
        assert command.startswith("cd /home/agent/workspace && ")
        assert command.endswith("echo run")

    @patch("cage.experiment.engine.trial_runner.CheckDonePoller")
    @patch("cage.experiment.engine.trial_runner.ReactiveLiveCheckMonitor")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_polling_monitor_is_separate_configured_mode(
        self, mock_find_pid, mock_snapshot, mock_reactive, mock_poller, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        config = _make_cvebench_config(MagicMock(return_value='{"status": true}'))
        config.execution.live_check.polling.enabled = True
        poller = mock_poller.return_value
        poller.success_event.is_set.return_value = False
        poller.verdict = None
        reactive = mock_reactive.return_value
        reactive.success_event.is_set.return_value = False
        reactive.verdict = None
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="CVE-2024-1234")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-123",
        )

        mock_reactive.assert_called_once()
        mock_poller.assert_called_once()
        poller.start.assert_called_once()
        poller.stop.assert_called_once()

    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_does_not_create_check_network_for_submit_service(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        handle = MagicMock(spec=SubmitServiceHandle)
        mock_start.return_value = handle

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{x}")
        trial.sample["network_name"] = "sample-net"
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        with (
            patch("cage.target.provisioning.create_check_network") as mock_create,
            patch("cage.target.provisioning.remove_check_network") as mock_remove,
        ):
            _run_execute_trial(
                trial=trial, config=config, container=container,
                storage=storage, hook_ctx=hook_ctx, run_id="run-123",
            )

        mock_start.assert_called_once()
        mock_create.assert_not_called()
        mock_remove.assert_not_called()
        container.connect_network.assert_not_called()

    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_autopenbench_uses_submit_without_runtime_network_changes(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        handle = MagicMock(spec=SubmitServiceHandle)
        mock_start.return_value = handle

        config = _make_config(benchmark_name="autopenbench", live_check_enabled=True)
        container = _make_container()
        container._runtime_network_name = "existing-net"
        storage = _make_storage(tmp_path)
        trial = _make_trial(sample_id="apb-1", flag="Ey8C7gOdzaKxTNqp")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        with patch("cage.target.provisioning.create_check_network") as mock_create:
            _run_execute_trial(
                trial=trial, config=config, container=container,
                storage=storage, hook_ctx=hook_ctx, run_id="run-456",
            )

        mock_create.assert_not_called()
        mock_start.assert_called_once()
        assert mock_start.call_args[1]["question_id"] == "apb-1"
        assert mock_start.call_args[1]["expected_answer"] == "Ey8C7gOdzaKxTNqp"
        container.connect_network.assert_not_called()
        container.disconnect_network.assert_not_called()

    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_no_submit_service_for_cvebench(
        self, mock_find_pid, mock_snapshot, tmp_path,
    ):
        """cvebench has builtin check — no submit service started."""
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )

        config = _make_config(benchmark_name="cvebench", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="CVE-2024-1234")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        with patch("cage.experiment.engine.trial_runner.start_submit_service") as mock_start:
            _run_execute_trial(
                trial=trial, config=config, container=container,
                storage=storage, hook_ctx=hook_ctx, run_id="run-789",
            )
            mock_start.assert_not_called()

        # No check metadata on sample
        assert trial.sample.get("check_endpoint") is None
        assert trial.sample.get("check_supported") is None

    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_cvebench_check_done_runs_with_agent_container_and_is_saved(
        self, mock_find_pid, mock_snapshot, tmp_path,
    ):
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        check_done = MagicMock(return_value='{"status": true, "message": "ok"}')

        config = _make_cvebench_config(check_done)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="CVE-2024-1234")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-789",
        )

        # gather is now called with a GatherRuntime carrying the live container + sample.
        check_done.assert_called_once()
        (runtime_arg,), _ = check_done.call_args
        assert runtime_arg.container is container
        assert runtime_arg.sample == trial.sample
        output_path = tmp_path / "trials" / "trial-sample-1" / "runtime" / "check_done_output.txt"
        assert output_path.read_text(encoding="utf-8") == '{"status": true, "message": "ok"}'

    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_no_submit_service_when_live_check_disabled(
        self, mock_find_pid, mock_snapshot, tmp_path,
    ):
        """No submit service when live_check.enabled=False."""
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=False)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{test}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        with patch("cage.experiment.engine.trial_runner.start_submit_service") as mock_start:
            _run_execute_trial(
                trial=trial, config=config, container=container,
                storage=storage, hook_ctx=hook_ctx, run_id="run-000",
            )
            mock_start.assert_not_called()


class TestSubmitServiceCleanup:
    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_submit_service_stopped_after_trial(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        """Submit service is stopped in the finally block."""
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        handle = MagicMock(spec=SubmitServiceHandle)
        mock_start.return_value = handle

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{test}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-cleanup",
        )

        handle.stop.assert_called_once()

    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_submit_service_stopped_on_trial_error(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        """Submit service is cleaned up even when trial execution fails."""
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )
        handle = MagicMock(spec=SubmitServiceHandle)
        mock_start.return_value = handle

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        # Make the container exec raise during agent execution
        container.exec_async.side_effect = RuntimeError("Agent crashed")
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{test}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        result = _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-err",
        )

        # Trial failed but submit service still cleaned up
        assert result.error is not None
        handle.stop.assert_called_once()

    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_no_flag_skips_submit_service(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        """When sample has no flag, submit service is not started."""
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
        )

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="")  # empty flag
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        with (
            patch("cage.experiment.engine.trial_runner.start_submit_service") as mock_start_inner,
            patch("cage.target.provisioning.create_check_network") as mock_create,
        ):
            _run_execute_trial(
                trial=trial, config=config, container=container,
                storage=storage, hook_ctx=hook_ctx, run_id="run-noflag",
            )
            mock_start_inner.assert_not_called()
            mock_create.assert_not_called()

    @patch("cage.experiment.engine.trial_runner.start_submit_service")
    @patch("cage.experiment.engine.trial_runner.snapshot_state")
    @patch("cage.experiment.engine.trial_runner._find_agent_pid", return_value="")
    def test_submit_start_failure_does_not_break_trial(
        self, mock_find_pid, mock_snapshot, mock_start, tmp_path,
    ):
        """If submit service fails to start, trial continues without check."""
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir(parents=True, exist_ok=True)
        mock_snapshot.return_value = SimpleNamespace(
            snapshot_dir=snap_dir, timestamp_ms=0, state_paths=(),
            has_failures=False,
        )
        mock_start.side_effect = RuntimeError("Docker unavailable")

        config = _make_config(benchmark_name="nyu_ctf", live_check_enabled=True)
        container = _make_container()
        storage = _make_storage(tmp_path)
        trial = _make_trial(flag="flag{test}")
        hook_ctx = _make_hook_ctx(tmp_path, trial.sample)

        result = _run_execute_trial(
            trial=trial, config=config, container=container,
            storage=storage, hook_ctx=hook_ctx, run_id="run-fail",
        )

        # Trial completed (no error from submit failure)
        assert result.error is None
        # Check metadata not set since start failed
        assert trial.sample.get("check_endpoint") is None
        assert trial.sample.get("check_supported") is None


class TestNeedsSubmitServiceIntegration:
    """Verify the benchmark->submit mapping the orchestrator branches on.

    The mapping is declarative: a benchmark sets ``needs_submit_service`` and the
    framework reads it. These assert a real benchmark that declares it (nyu_ctf)
    and the live-check gating, never a hardcoded name set.
    """

    @staticmethod
    def _load_nyu_class():
        import importlib.util

        module_path = Path(__file__).resolve().parents[1] / "examples" / "nyu" / "benchmark.py"
        spec = importlib.util.spec_from_file_location("nyu_benchmark_caps", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NYUCTF

    def test_declaring_benchmark_needs_submit(self):
        nyu = self._load_nyu_class()
        assert nyu.needs_submit_service is True
        assert needs_submit_service(SimpleNamespace(needs_submit_service=True), True) is True

    def test_non_declaring_benchmark_does_not(self):
        assert needs_submit_service(SimpleNamespace(needs_submit_service=False), True) is False

    def test_disabled_always_false(self):
        assert needs_submit_service(SimpleNamespace(needs_submit_service=True), False) is False
