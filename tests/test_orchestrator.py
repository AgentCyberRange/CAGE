"""Tests for orchestrator run output paths."""

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cage.agents.base import AgentContainerResources
from dataclasses import replace
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.artifacts.resources import ResourceLedgerWriter
from cage.artifacts.run_storage import RunStorage
from cage.artifacts.dashboard import Dashboard
from cage.config.sections import ExecutionConfig, TargetConfig
from cage.experiment.model import TrialTermination
from cage.experiment.engine.hooks import HookContext, HookRegistry
from cage.experiment.model import Trial, TrialResult, TrialStatus, TrialType
from cage.contracts.logging import LoggingConfig
from cage.experiment.engine.scheduler import RunScheduler
from cage.artifacts.canonical_marks import (
    _mark_canonical_trial_final_evidence_artifact,
    _mark_canonical_trial_finished,
    _mark_canonical_trial_live_evidence_artifact,
    _mark_canonical_trial_output_artifact,
    _mark_canonical_trial_started,
    _mark_canonical_trial_state_artifact,
    _mark_canonical_trial_trajectory_artifact,
    _save_canonical_experiment_snapshot,
)
from cage.artifacts.record_snapshots import _build_run_manifest, _json_fingerprint
from cage.experiment.engine.reporting import _write_dashboards
from cage.experiment.engine.resume import _assert_resume_compatible, _cage_runs_root, _cap_trials_for_execution, _load_planned_trials
from cage.experiment.engine.conductor import _resolve_run_id, _run_agent_trials_parallel, _run_single_agent, run_experiment
from cage.experiment.engine.run_context import ExperimentRun
from cage.proxy.monitor import _parse_proxy_stats
from cage.experiment.engine.resource_recorder import _record_agent_container_resource, _record_agent_isolation_network_resource, _record_container_proxy_resource, _record_target_runtime_resource, _stop_container_proxy_resource, _target_teardown_resource_status
from cage.experiment.engine.trial_runner import _append_session_args, _create_container, _effective_trial_max_rounds, execute_trial, _trial_termination_metadata
from cage.scoring.lifecycle import _hydrate_scores_from_disk, _score_one_trial
from cage.target.provisioning import AgentIsolationNetwork, target_launch_request_timeout_s, target_server_timeout_env
from cage.contracts.exceptions import TrialTimeout
from cage.proxy.host import ContainerProxyInstance
from cage.sandbox.exec import Timing
from cage.experiment.engine.termination import (
    TerminationReason,
    target_unavailable_termination,
    termination_info_from_exception,
)
from cage.scoring import Score
from cage.target.client import TargetTeardownResult


class _DemoBenchmark:
    name = "demo"

    def iter_samples(self):
        return iter([])

    def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
        return iter([])

    def teardown(self) -> None:
        pass


class _OneSampleBenchmark(_DemoBenchmark):
    def iter_samples_limited(self, limit=None, sample_ids=None, slice_spec=None):
        return iter([{"id": "range1", "content": "task"}])


class _DashboardBenchmark(_OneSampleBenchmark):
    def build_dashboard(self, run_dir: Path) -> Dashboard:
        return Dashboard(title="Demo dashboard")


class _PreflightOk:
    def to_dict(self):
        return {"ok": True}


def _passk_trials(n_samples: int, passk: int) -> list[Trial]:
    """Build a pass-major trial list mirroring the orchestrator's expansion."""
    trials: list[Trial] = []
    for pass_idx in range(1, passk + 1):
        for s in range(n_samples):
            trials.append(Trial(
                id=f"range{s}/pass_{pass_idx}", index=0, type=TrialType.TASK,
                sample={"id": f"range{s}", "pass_index": pass_idx},
            ))
    for index, t in enumerate(trials):
        t.index = index
    return trials


def test_cap_trials_for_execution_none_is_noop():
    trials = _passk_trials(24, 3)
    assert _cap_trials_for_execution(trials, None) is trials
    assert _cap_trials_for_execution(trials, -1) is trials


def test_cap_trials_for_execution_defers_last_pass():
    # 24 samples × pass@3: pass-major order means pass_3 is index 48..71.
    trials = _passk_trials(24, 3)
    capped = _cap_trials_for_execution(trials, 48)
    assert len(capped) == 48
    assert all(t.index < 48 for t in capped)
    # Everything kept is pass_1/pass_2; nothing from pass_3 leaks through.
    assert all(t.sample["pass_index"] in (1, 2) for t in capped)
    assert not any(t.sample["pass_index"] == 3 for t in capped)


def test_cap_trials_for_execution_throttle_partial_pass():
    trials = _passk_trials(24, 3)
    capped = _cap_trials_for_execution(trials, 30)
    assert [t.index for t in capped] == list(range(30))


def test_force_run_id_archives_existing_run_dir_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    cage_runs = tmp_path / ".cage_runs"
    run_dir = cage_runs / "agent:model:stateless" / "run-fixed"
    run_dir.mkdir(parents=True)
    (run_dir / "dashboard.json").write_text('{"run_id": "run-fixed"}', encoding="utf-8")

    config = ExperimentRun(
        name="demo",
        project_file=tmp_path / "project.yml",
        benchmark=_DemoBenchmark(),
        agents=[],
        models={},
        force=True,
        run_id="run-fixed",
    )
    monkeypatch.setattr("cage.experiment.engine.conductor.time.strftime", lambda _fmt: "20260602T120000")
    caplog.set_level(logging.DEBUG, logger="cage.experiment.engine.conductor")

    resolved = _resolve_run_id(config, cage_runs)

    archive = cage_runs / "agent:model:stateless" / "run-fixed.previous_20260602T120000"
    marker = json.loads((archive / "force_archive.json").read_text(encoding="utf-8"))
    dashboard = json.loads((archive / "dashboard.json").read_text(encoding="utf-8"))

    assert resolved == "run-fixed"
    assert not run_dir.exists()
    assert dashboard["run_id"] == "run-fixed.previous_20260602T120000"
    assert dashboard["archived_from_run_id"] == "run-fixed"
    assert marker == {
        "archived_from_run_id": "run-fixed",
        "archived_at": "20260602T120000",
        "archive_reason": "force_run_id_reuse",
    }
    archive_logs = [
        record
        for record in caplog.records
        if record.name == "cage.experiment.engine.conductor" and "force: archived existing run" in record.message
    ]
    assert archive_logs
    assert all(record.levelno == logging.DEBUG for record in archive_logs)


def test_trial_termination_metadata_distinguishes_execution_timeout():
    meta = _trial_termination_metadata(
        exit_code=-1,
        timed_out=True,
        terminated_by_limit=False,
        error="",
        timeout_seconds=300,
    )

    assert meta == {
        "status": "failed",
        "termination_reason": "execution_timeout",
        "termination_detail": "Agent execution exceeded 300s",
        "termination_source": "orchestrator",
    }


def test_trial_termination_metadata_distinguishes_tool_limit():
    meta = _trial_termination_metadata(
        exit_code=0,
        timed_out=False,
        terminated_by_limit=True,
        error="",
        timeout_seconds=0,
    )

    assert meta["termination_reason"] == "tool_limit"


def test_trial_termination_metadata_distinguishes_proxy_max_rounds_stop(
    tmp_path: Path,
):
    proxy_jsonl = tmp_path / "proxy.jsonl"
    proxy_jsonl.write_text(
        "".join(
            json.dumps({"status": "success", "request_id": f"req-{idx:04d}"}) + "\n"
            for idx in range(1, 4)
        ),
        encoding="utf-8",
    )

    meta = _trial_termination_metadata(
        exit_code=0,
        timed_out=False,
        terminated_by_limit=False,
        terminated_by_max_rounds=True,
        error="",
        timeout_seconds=0,
        proxy_jsonl_path=proxy_jsonl,
        max_rounds=3,
    )

    assert meta["status"] == "completed"
    assert meta["termination_reason"] == "max_rounds_reached"


def test_trial_termination_metadata_distinguishes_nonzero_agent_exit():
    meta = _trial_termination_metadata(
        exit_code=2,
        timed_out=False,
        terminated_by_limit=False,
        error="",
        timeout_seconds=0,
    )

    assert meta == {
        "status": "failed",
        "termination_reason": "agent_exit_nonzero",
        "termination_detail": "Agent exited with code 2",
        "termination_source": "agent",
    }


def test_trial_termination_metadata_distinguishes_model_timeout_via_proxy_log(
    tmp_path: Path,
):
    proxy_jsonl = tmp_path / "proxy.jsonl"
    proxy_jsonl.write_text(
        json.dumps({
            "status": "error",
            "upstream_status_code": 408,
            "error_text": "HTTP 408: The read operation timed out.",
        }) + "\n",
        encoding="utf-8",
    )

    meta = _trial_termination_metadata(
        exit_code=1,
        timed_out=False,
        terminated_by_limit=False,
        error="",
        timeout_seconds=0,
        output="API Error: 502 The read operation timed out.",
        proxy_jsonl_path=proxy_jsonl,
    )

    assert meta == {
        "status": "failed",
        "termination_reason": "model_timeout",
        "termination_detail": "HTTP 408: The read operation timed out.",
        "termination_source": "model_proxy",
    }


def test_trial_exception_exposes_shared_termination_metadata():
    info = termination_info_from_exception(TrialTimeout("range1-L0", 300))

    assert info.to_metadata() == {
        "status": "failed",
        "termination_reason": "execution_timeout",
        "termination_detail": "Trial range1-L0: timed out after 300s",
        "termination_source": "orchestrator",
    }


def test_target_unavailable_termination_has_stable_metadata():
    """Orchestrator's fail-fast on target launch failure must produce a stable
    ``termination_reason`` so the inspector / dashboard can distinguish a trial
    that never started from one that ran and lost. Benchmarks without target
    (swebench, cybergym, …) never hit this path because the orchestrator only
    invokes it when ``challenge_client is not None`` AND ``chal_id`` is set.
    """
    info = target_unavailable_termination(
        "target_status=stopped — server-side launch failed"
    )

    assert info.reason is TerminationReason.TARGET_UNAVAILABLE
    assert info.to_metadata() == {
        "status": "failed",
        "termination_reason": "target_unavailable",
        "termination_detail": "target_status=stopped — server-side launch failed",
        "termination_source": "orchestrator",
    }


def test_setup_interrupted_during_stop_records_interrupted_not_error(tmp_path):
    """A pre-agent setup exception while the run is stopping (e.g. the SIGINT
    teardown force-removed the container → ``No such container``) must record an
    INTERRUPTED trial, not a spurious ``trial_error`` — so resume re-runs it and
    the run summary doesn't inflate the failure count with Ctrl+C casualties.
    """
    from cage.experiment.engine.trial_runner import _trial_setup_interrupted_result
    from cage.contracts.trial_status import classify_trial_status, FAILED, INTERRUPTED

    storage = RunStorage(run_dir=tmp_path / ".cage_runs" / "agent" / "run-1")
    trial = Trial(id="pb-x-l0/pass_1", index=2, type=TrialType.TASK, sample={"id": "pb-x-l0"})

    result = _trial_setup_interrupted_result(
        started=1000, storage=storage, trial=trial, trial_id=trial.id,
    )

    meta = result.metadata
    assert meta["status"] == "interrupted"
    assert meta["termination_reason"] == "user_interrupted"
    assert classify_trial_status(
        status=meta["status"], termination_reason=meta["termination_reason"]
    ) == INTERRUPTED
    assert classify_trial_status(status=meta["status"]) != FAILED

    on_disk = json.loads(
        (storage.run_dir / "trials" / "pb-x-l0" / "pass_1" / "meta.json").read_text()
    )
    assert on_disk["status"] == "interrupted"
    assert on_disk["termination_reason"] == "user_interrupted"


def test_execute_trial_skips_setup_when_already_stopped(tmp_path, monkeypatch):
    """Once a stop is in flight, ``execute_trial`` must not drive the trial deeper
    into the pre-agent phase (snapshot → reset → proxy) only to crash on a
    force-removed container; it returns an interrupted result up front.
    """
    import cage.experiment.engine.trial_runner as tr
    from cage.contracts.trial_status import classify_trial_status, INTERRUPTED

    def _must_not_run(*args, **kwargs):
        raise AssertionError("snapshot_state must not run once the run is stopped")

    monkeypatch.setattr(tr, "snapshot_state", _must_not_run)
    monkeypatch.setattr(tr, "_cleanup_trial_resources", lambda **kwargs: None)

    storage = RunStorage(run_dir=tmp_path / ".cage_runs" / "agent" / "run-1")
    trial = Trial(id="pb-x-l0/pass_1", index=0, type=TrialType.TASK, sample={"id": "pb-x-l0"})
    hook_ctx = HookContext(
        experiment_config={"name": "x"}, samples=[trial.sample],
        trials_completed=[], trials_pending=[], run_artifacts_dir=str(storage.run_dir),
    )

    result = tr.execute_trial(
        trial=trial,
        agent=SimpleNamespace(home="/home/agent"),
        run=SimpleNamespace(),
        container=SimpleNamespace(),
        storage=storage,
        hook_ctx=hook_ctx,
        scheduler=SimpleNamespace(is_stopped=lambda: True),
        challenge_client=None,
        run_id="run-1",
        reporter=None,
    )

    assert result.metadata["termination_reason"] == "user_interrupted"
    assert classify_trial_status(
        status=result.metadata["status"],
        termination_reason=result.metadata["termination_reason"],
    ) == INTERRUPTED


def test_target_setup_gate_serializes_target_launches():
    from cage.experiment.engine.scheduler import RunScheduler

    scheduler = RunScheduler(target_setup_cap=1)
    start = threading.Barrier(3)
    lock = threading.Lock()
    active = 0
    max_active = 0
    errors: list[BaseException] = []

    def worker(idx: int) -> None:
        nonlocal active, max_active
        try:
            start.wait(timeout=2)
            with scheduler.target_setup_gate(f"chal-{idx}", f"trial-{idx}"):
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert max_active == 1


def test_target_timeout_config_feeds_embedded_server_and_launch_client():
    target = TargetConfig(startup_timeout=1800, compose_up_timeout=3600)

    assert target_server_timeout_env(target) == {
        "TARGET_SERVER_COMPOSE_UP_TIMEOUT_S": "3600.0",
        "TARGET_SERVER_STARTUP_TIMEOUT_S": "1800.0",
        "TARGET_SERVER_BUILD_IF_MISSING": "0",
    }
    assert target_launch_request_timeout_s(target) == 12660.0
    assert target_launch_request_timeout_s(TargetConfig()) == 300.0


def test_run_experiment_disables_embedded_target_server_builds_by_default(
    tmp_path,
    monkeypatch,
):
    import cage.experiment.engine.conductor as orchestrator

    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")

    agent = SimpleNamespace(
        id="demo_agent",
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_DemoBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1),
        target=TargetConfig(enabled=True),
    )
    config.metadata["benchmark_id"] = "web_exploit_bench"
    captured: dict[str, object] = {}

    def fake_spawn_embedded_target_server(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            server_url="http://127.0.0.1:9999",
            shutdown=lambda: None,
        )

    monkeypatch.setattr(orchestrator, "create_run_id", lambda: "run-fixed")
    monkeypatch.setattr(
        orchestrator,
        "spawn_embedded_target_server",
        fake_spawn_embedded_target_server,
    )
    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: _PreflightOk())
    monkeypatch.setattr(orchestrator, "_run_single_agent", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(orchestrator, "_score_trials", lambda *_args, **_kwargs: None)

    run_experiment(config)

    assert captured["extra_env"] == {
        "TARGET_SERVER_BUILD_IF_MISSING": "0",
        "CAGE_BENCHMARK_ID": "web_exploit_bench",
    }


def test_parse_proxy_stats_counts_official_responses_api_usage(tmp_path):
    proxy_log = tmp_path / "proxy.jsonl"
    proxy_log.write_text(
        json.dumps(
            {
                "status": "success",
                "upstream_response": {
                    "id": "resp_123",
                    "object": "response",
                    "usage": {
                        "input_tokens": 101,
                        "output_tokens": 23,
                        "output_tokens_details": {"reasoning_tokens": 7},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = _parse_proxy_stats(proxy_log)

    assert stats["input_tokens"] == 101
    assert stats["output_tokens"] == 23
    assert stats["reasoning_tokens"] == 7
    assert stats["num_requests"] == 1


def test_parse_proxy_stats_counts_official_anthropic_usage(tmp_path):
    proxy_log = tmp_path / "proxy.jsonl"
    proxy_log.write_text(
        json.dumps(
            {
                "status": "success",
                "anthropic_response": {
                    "usage": {
                        "input_tokens": 88,
                        "output_tokens": 13,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = _parse_proxy_stats(proxy_log)

    assert stats["input_tokens"] == 88
    assert stats["output_tokens"] == 13
    assert stats["num_requests"] == 1


def test_parse_proxy_stats_counts_anthropic_cache_input_usage(tmp_path):
    proxy_log = tmp_path / "proxy.jsonl"
    proxy_log.write_text(
        json.dumps(
            {
                "status": "success",
                "upstream_response": {
                    "type": "message",
                    "usage": {
                        "input_tokens": 595,
                        "cache_read_input_tokens": 24256,
                        "cache_creation_input_tokens": 128,
                        "output_tokens": 169,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats = _parse_proxy_stats(proxy_log)

    assert stats["input_tokens"] == 24979
    assert stats["output_tokens"] == 169
    assert stats["num_requests"] == 1


def test_proxy_monitor_reads_host_progress_without_container_exec(tmp_path):
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
    assert reporter.requests[0].step == 2
    assert reporter.requests[0].input_tokens == 10_000
    assert reporter.requests[0].output_tokens == 500
    assert reporter.requests[0].reasoning_tokens == 25
    assert reporter.requests[0].cost_usd == 0.12


def test_proxy_monitor_skips_when_host_progress_is_missing(tmp_path):
    from cage.proxy.monitor import _ProxyMonitor

    class NoExecContainer:
        def exec(self, _cmd, timeout=0):
            raise AssertionError("proxy progress monitor should not fallback to docker exec")

    class Recorder:
        def __init__(self):
            self.progress = []
            self.requests = []

        def update_trial_progress(self, **kwargs):
            self.progress.append(kwargs)

        def record_model_request(self, event):
            self.requests.append(event)

    reporter = Recorder()
    monitor = _ProxyMonitor(
        container=NoExecContainer(),
        log_dir="/opt/cage-proxy/logs",
        trial_id="trial-1",
        artifact_dir=tmp_path / "proxy",
        reporter=reporter,
        agent_label="agent-a:model:stateless",
    )

    monitor._report()

    assert reporter.progress == []
    assert reporter.requests == []


def test_proxy_monitor_stop_flushes_latest_host_progress(tmp_path):
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
                "total_requests": 150,
                "successful_requests": 150,
                "last_status": "success",
                "tokens_in": 10_000,
                "tokens_out": 500,
                "errors": 0,
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
    )

    monitor.stop()

    assert len(reporter.progress) == 1
    assert reporter.progress[0]["progress"]["successful_requests"] == 150
    assert len(reporter.requests) == 1
    assert reporter.requests[0].step == 150


def test_proxy_monitor_terminates_agent_when_max_rounds_reached(tmp_path):
    from cage.proxy.monitor import _ProxyMonitor

    class NoExecContainer:
        def exec(self, _cmd, timeout=0):
            raise AssertionError("proxy progress monitor should not docker exec")

    class Process:
        def __init__(self):
            self.terminate_calls = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminate_calls += 1

    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    (proxy_dir / "progress.json").write_text(
        json.dumps(
            {
                "total_requests": 153,
                "successful_requests": 153,
                "last_status": "success",
                "tokens_in": 10_000,
                "tokens_out": 500,
                "errors": 0,
            }
        ),
        encoding="utf-8",
    )
    process = Process()
    monitor = _ProxyMonitor(
        container=NoExecContainer(),
        log_dir="/opt/cage-proxy/logs",
        trial_id="trial-1",
        artifact_dir=proxy_dir,
        process=process,
        max_rounds=150,
    )

    monitor._report()
    monitor._report()

    assert process.terminate_calls == 1
    assert monitor.terminated_by_max_rounds is True


def test_interrupted_dashboard_uses_planned_trials_for_precise_status(tmp_path):
    cage_runs = tmp_path / ".cage_runs"
    run_dir = cage_runs / "agent:model:stateless" / "run-fixed"
    run_dir.mkdir(parents=True)
    (run_dir / "planned_trials.json").write_text(
        json.dumps(
            [
                {
                    "trial_id": "range1-L0",
                    "trial_index": 0,
                    "trial_type": "task",
                    "sample_id": "range1-L0",
                },
                {
                    "trial_id": "range1-L1",
                    "trial_index": 1,
                    "trial_type": "task",
                    "sample_id": "range1-L1",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    started_meta = {
        "trial_id": "range1-L0",
        "trial_index": 0,
        "trial_type": "task",
        "sample_id": "range1-L0",
        "exit_code": 0,
    }
    meta_path = run_dir / "trials" / "range1-L0" / "meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps(started_meta), encoding="utf-8")

    config = SimpleNamespace(
        name="demo",
        output=SimpleNamespace(
            dashboard_prompt=False,
            dashboard_output=False,
            dashboard_reasoning=False,
            csv_prompt=False,
            csv_output=False,
            csv_reasoning=False,
        ),
    )

    _write_dashboards(
        config,
        {"agent:model:stateless": []},
        cage_runs,
        "run-fixed",
        "2026-04-28T00:00:00",
        "2026-04-28T00:01:00",
        interrupted=True,
    )

    dashboard = json.loads((run_dir / "dashboard.json").read_text(encoding="utf-8"))
    agent_summary = dashboard["agents"]["agent:model:stateless"]
    trials = agent_summary["trials"]
    by_id = {trial["trial_id"]: trial for trial in trials}

    assert agent_summary["total"] == 2
    assert agent_summary["interrupted"] == 1
    assert agent_summary["cancelled"] == 1
    assert by_id["range1-L0"]["status"] == "interrupted"
    assert by_id["range1-L0"]["termination_reason"] == "user_interrupted"
    assert by_id["range1-L0"]["termination_source"] == "user"
    assert by_id["range1-L1"]["status"] == "cancelled"
    assert by_id["range1-L1"]["termination_reason"] == "cancelled_before_start"

    persisted_started = json.loads(meta_path.read_text(encoding="utf-8"))
    assert persisted_started["termination_reason"] == "user_interrupted"
    pending_meta = json.loads(
        (run_dir / "trials" / "range1-L1" / "meta.json").read_text(encoding="utf-8")
    )
    assert pending_meta["termination_reason"] == "cancelled_before_start"


def test_write_dashboards_appends_run_history_for_reused_run_id(tmp_path: Path) -> None:
    cage_runs = tmp_path / ".cage_runs"
    run_dir = cage_runs / "agent:model:stateless" / "run-fixed"

    config = SimpleNamespace(
        name="demo",
        output=SimpleNamespace(
            dashboard_prompt=False,
            dashboard_output=False,
            dashboard_reasoning=False,
            csv_prompt=False,
            csv_output=False,
            csv_reasoning=False,
        ),
    )

    _write_dashboards(
        config,
        {"agent:model:stateless": []},
        cage_runs,
        "run-fixed",
        "2026-04-28T10:00:00",
        "2026-04-28T10:20:00",
    )
    _write_dashboards(
        config,
        {"agent:model:stateless": []},
        cage_runs,
        "run-fixed",
        "2026-04-28T11:00:00",
        "2026-04-28T11:05:00",
    )

    history = json.loads((run_dir / "run_history.json").read_text(encoding="utf-8"))
    assert [entry["label"] for entry in history["attempts"]] == [
        "Initial run",
        "Resume #1",
    ]
    assert history["attempts"][0]["started_at"] == "2026-04-28T10:00:00"
    assert history["attempts"][0]["completed_at"] == "2026-04-28T10:20:00"
    assert history["attempts"][1]["started_at"] == "2026-04-28T11:00:00"
    assert history["attempts"][1]["completed_at"] == "2026-04-28T11:05:00"
    assert history["attempts"][1]["is_latest"] is True

    dashboard = json.loads((run_dir / "dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["started_at"] == "2026-04-28T11:00:00"
    assert (
        dashboard["run_history"]["attempts"][0]["started_at"]
        == "2026-04-28T10:00:00"
    )
    assert (
        dashboard["run_history"]["attempts"][1]["started_at"]
        == "2026-04-28T11:00:00"
    )


def test_cage_runs_root_uses_benchmark_dir_not_project_file_dir(tmp_path):
    benchmark_dir = tmp_path / "examples" / "agent_pentest_bench"
    config_dir = benchmark_dir / "configs" / "campaigns"
    config_dir.mkdir(parents=True)
    project_file = config_dir / "project_internal.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")

    config = SimpleNamespace(
        project_file=project_file,
        benchmark_dir=benchmark_dir,
    )

    assert _cage_runs_root(config) == benchmark_dir / ".cage_runs"


def test_run_experiment_writes_under_benchmark_root_cage_runs(tmp_path, monkeypatch):
    project_dir = tmp_path / "examples" / "strongreject"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")

    agent = SimpleNamespace(
        id="demo_agent",
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
    )
    other_agent = SimpleNamespace(
        id="other_agent",
        agent_type=SimpleNamespace(name="other_agent"),
        subject_plan_id="other_agent:other-model:stateless",
        max_concurrent=0,
        label=lambda: "other_agent:other-model:stateless",
        model=SimpleNamespace(id="other-model"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_DemoBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent, other_agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
    )

    monkeypatch.setattr("cage.experiment.engine.conductor.create_run_id", lambda: "run-fixed")
    monkeypatch.setattr(
        "cage.experiment.engine.preflight.run_preflight",
        lambda *_args, **_kwargs: _PreflightOk(),
    )
    monkeypatch.setattr(
        "cage.experiment.engine.conductor._run_agent_trials_serial",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "cage.experiment.engine.conductor._score_trials",
        lambda *_args, **_kwargs: None,
    )

    summary = run_experiment(config)

    # Layout: .cage_runs/{agent_id}:{model_id}:{mode}/run-{timestamp}/
    cage_runs = project_dir / ".cage_runs"
    run_dir = cage_runs / "demo_agent:demo-model:stateless" / "run-fixed"

    summary_run_dir = Path(summary["run_dir"])
    assert summary_run_dir in {
        run_dir,
        cage_runs / "other_agent:other-model:stateless" / "run-fixed",
    }
    assert Path(summary["dashboard_path"]) == summary_run_dir / "dashboard.json"
    assert (run_dir / "dashboard.json").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "config.yaml").exists()
    other_run_dir = cage_runs / "other_agent:other-model:stateless" / "run-fixed"
    assert (other_run_dir / "dashboard.json").exists()
    assert (other_run_dir / "results.csv").exists()
    assert not (cage_runs / "run-fixed" / "dashboard.json").exists()


def test_run_experiment_writes_canonical_experiment_artifacts(tmp_path, monkeypatch):
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: demo
runtime:
  max_rounds: 5
agents:
  - id: demo_agent
    kind: demo_agent
    model: demo-model
""".lstrip(),
        encoding="utf-8",
    )

    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_DashboardBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )

    monkeypatch.setattr("cage.experiment.engine.conductor.create_run_id", lambda: "run-fixed")
    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: _PreflightOk())
    monkeypatch.setattr("cage.experiment.engine.conductor._run_agent_trials_serial", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("cage.experiment.engine.conductor._score_trials", lambda *_args, **_kwargs: None)

    run_experiment(config)

    run_dir = project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-fixed"
    record = json.loads(
        (run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    plan = json.loads((run_dir / "experiment_plan.json").read_text(encoding="utf-8"))
    artifact_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )

    assert record["run_id"] == "run-fixed"
    assert record["status"] == "completed"
    assert record["started_at"]
    assert record["completed_at"]
    assert record["trials"]["total"] == 1
    assert plan["subjects"][0]["subject_id"] == "demo_agent:demo-model:stateless"
    assert plan["trials"][0]["trial_id"] == "range1"
    assert record["trials"]["records"][0]["trial_id"] == plan["trials"][0]["trial_id"]
    artifact_paths = {artifact["path"] for artifact in artifact_index["artifacts"]}
    assert "experiment_plan.json" in artifact_paths
    assert "planned_trials.json" in artifact_paths
    assert "run_manifest.json" in artifact_paths
    assert "dashboard.json" in artifact_paths
    assert "dashboard_view.json" in artifact_paths
    assert "results.csv" in artifact_paths
    assert "config.yaml" in artifact_paths
    assert "project.yml" in artifact_paths
    assert "preflight.json" in artifact_paths
    assert "run_history.json" in artifact_paths
    assert "summary.json" in artifact_paths
    assert record["trials"]["records"][0]["record_ref"] in artifact_paths
    planned_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "planned_trials.json"
    )
    assert planned_artifact["kind"] == "compat_planned_trials"
    manifest_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "run_manifest.json"
    )
    assert manifest_artifact["kind"] == "compat_run_manifest"
    dashboard_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "dashboard.json"
    )
    assert dashboard_artifact["kind"] == "compat_dashboard"
    dashboard_view_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "dashboard_view.json"
    )
    assert dashboard_view_artifact["kind"] == "dashboard_view"
    results_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "results.csv"
    )
    assert results_artifact["kind"] == "compat_results_csv"
    config_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "config.yaml"
    )
    assert config_artifact["kind"] == "compat_run_config"
    project_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "project.yml"
    )
    assert project_artifact["kind"] == "compat_project_yml"
    preflight_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "preflight.json"
    )
    assert preflight_artifact["kind"] == "preflight_result"
    run_history_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "run_history.json"
    )
    assert run_history_artifact["kind"] == "run_history"
    summary_artifact = next(
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "summary.json"
    )
    assert summary_artifact["kind"] == "compat_run_summary"


def test_trial_runner_updates_canonical_trial_records(tmp_path, monkeypatch):
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: demo
runtime:
  max_rounds: 5
agents:
  - id: demo_agent
    kind: demo_agent
    model: demo-model
""".lstrip(),
        encoding="utf-8",
    )
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    run_id = "run-fixed"
    storage = RunStorage(
        project_dir
        / ".cage_runs"
        / "demo_agent:demo-model:stateless"
        / run_id,
        agent_label=agent.label(),
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], run_id)

    def _fake_run_trial(*_args, **_kwargs):
        return TrialResult(
            trial_id=trial.id,
            trial_index=trial.index,
            trial_type=trial.type.value,
            sample_id=trial.sample_id,
            output="",
            exit_code=2,
            timing=Timing(started_at_ms=1, ended_at_ms=3, duration_ms=2),
            error="agent failed",
            metadata={
                "status": "failed",
                "termination_reason": "agent_exit_nonzero",
            },
        )

    monkeypatch.setattr("cage.experiment.engine.conductor.run_trial_isolated", _fake_run_trial)

    _run_agent_trials_parallel(
        config,
        agent,
        storage,
        [trial],
        HookContext(
            experiment_config={"name": config.name},
            samples=[trial.sample],
            trials_completed=[],
            trials_pending=[],
            run_artifacts_dir=str(storage.run_dir),
        ),
        max_workers=1,
    )

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))

    assert record["trials"]["failed"] == 1
    assert trial_record["status"] == "failed"
    assert trial_record["status_reason"] == "agent_exit_nonzero"
    assert trial_record["started_at"]
    assert trial_record["completed_at"]
    assert trial_record["termination"]["reason"] == "agent_exit_nonzero"
    assert trial_record["termination"]["exit_code"] == 2


def test_run_single_agent_records_canonical_trial_scoring_refs(tmp_path, monkeypatch):
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: demo
runtime:
  max_rounds: 5
agents:
  - id: demo_agent
    kind: demo_agent
    model: demo-model
""".lstrip(),
        encoding="utf-8",
    )
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    cage_runs = project_dir / ".cage_runs"

    def _fake_run_trials(_config, _agent, storage, trials, *_args, **_kwargs):
        [trial] = trials
        return [
            TrialResult(
                trial_id=trial.id,
                trial_index=trial.index,
                trial_type=trial.type.value,
                sample_id=trial.sample_id,
                output="done",
                exit_code=0,
                timing=Timing(started_at_ms=1, ended_at_ms=3, duration_ms=2),
                metadata={
                    "sample": trial.sample,
                    "status": "completed",
                    "termination_reason": "completed",
                },
            )
        ]

    def _fake_score_trials(results, _benchmark, storage):
        [result] = results
        result.scores["demo"] = Score(value=1.0, answer="ok", explanation="")
        storage.save_trial_scores(
            result.trial_id,
            "demo",
            {"demo": {"value": 1.0, "answer": "ok", "explanation": ""}},
        )

    monkeypatch.setattr("cage.experiment.engine.conductor._run_agent_trials_serial", _fake_run_trials)
    monkeypatch.setattr("cage.experiment.engine.conductor._score_trials", _fake_score_trials)

    _run_single_agent(replace(config, cage_runs=cage_runs, run_id="run-fixed"), agent)

    run_dir = cage_runs / "demo_agent:demo-model:stateless" / "run-fixed"
    record = json.loads(
        (run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((run_dir / trial_ref).read_text(encoding="utf-8"))

    assert trial_record["scoring_id"] == "demo"
    assert trial_record["scoring"]["status"] == "scored"
    assert trial_record["scoring"]["score_ref"] == "trials/range1/scores/demo.json"
    assert record["score_summary"]["status"] == "scored"
    assert record["score_summary"]["summary_ref"] == "summary.json"
    artifact_index = json.loads(
        (run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    trial_score_artifacts = [
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "trials/range1/scores/demo.json"
    ]
    summary_artifacts = [
        artifact
        for artifact in artifact_index["artifacts"]
        if artifact["path"] == "summary.json"
    ]
    assert trial_score_artifacts
    assert trial_score_artifacts[0]["kind"] == "trial_score"
    assert summary_artifacts
    assert summary_artifacts[0]["kind"] == "score_summary"


def test_hydrate_scores_uses_indexed_canonical_score_ref(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="agent:model:stateless",
        id="agent",
        label=lambda: "agent:model:stateless",
        model=SimpleNamespace(
            id="model",
            provider="anthropic",
            model="model",
            base_url="",
            auth_source="",
            api_key="",
            timeout=360,
            max_retries=2,
            extra={},
        ),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        session_args=[],
        plugins=[],
        skill="",
        version="latest",
        image="",
        home="/home/agent/workspace",
        max_rounds=-1,
        max_concurrent=1,
        context_compaction_threshold=0.5,
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1),
        target=TargetConfig(enabled=False),
        run_id="run-fixed",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    run_dir = project_dir / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    storage = RunStorage(run_dir, agent_label=agent.label())
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-fixed")
    record = json.loads(
        (run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    canonical_trial_id = record["trials"]["records"][0]["trial_id"]
    score_path = run_dir / "indexed_scores" / "demo.json"
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        json.dumps({
            "demo": {
                "value": 0.75,
                "answer": "ok",
                "explanation": "loaded from canonical ref",
                "metadata": {"source": "artifact_index"},
            }
        }),
        encoding="utf-8",
    )
    writer = ExperimentArtifactWriter(run_dir)
    writer.mark_trial_artifact(
        canonical_trial_id,
        artifact_id=f"trial.{canonical_trial_id}.score.demo",
        path=score_path,
        kind="trial_score",
        schema_version="trial_score.v1",
        producer="test",
        replayability="replayable",
    )
    writer.mark_trial_scored(
        canonical_trial_id,
        score_ref="indexed_scores/demo.json",
        scoring_id="demo",
    )
    result = TrialResult(
        trial_id="range1",
        trial_index=0,
        trial_type="task",
        sample_id="range1",
        output="done",
        exit_code=0,
        timing=Timing(started_at_ms=1, ended_at_ms=2, duration_ms=1),
    )

    assert _hydrate_scores_from_disk(result, _DemoBenchmark(), storage)
    assert result.scores["demo"].value == 0.75
    assert result.scores["demo"].metadata == {"source": "artifact_index"}


def test_score_one_trial_uses_canonical_scoring_context_artifacts(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )

    class _CanonicalEvidenceScorer:
        name = "canonical"

        def score(self, ctx):
            if ctx.check_done_output != "canonical evidence":
                raise AssertionError("missing canonical final evidence")
            if not ctx.canonical_trial_id:
                raise AssertionError("missing canonical trial id")
            return {
                "canonical": Score(
                    value=1.0,
                    answer=ctx.canonical_trial_id,
                    explanation=ctx.check_done_output,
                )
            }

    class _CanonicalEvidenceBenchmark(_OneSampleBenchmark):
        name = "demo"

        def scorer(self):
            return _CanonicalEvidenceScorer()

    benchmark = _CanonicalEvidenceBenchmark()
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=benchmark,
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    storage.save_trial_output(
        trial.id,
        {
            "trial_id": trial.id,
            "trial_index": trial.index,
            "sample": trial.sample,
            "output": "done",
            "exit_code": 0,
        },
    )
    _mark_canonical_trial_output_artifact(storage=storage, agent=agent, trial=trial)
    canonical_trial_id = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )["trials"]["records"][0]["trial_id"]
    evidence_path = storage.run_dir / "canonical_evidence" / "check_done_output.txt"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("canonical evidence", encoding="utf-8")
    ExperimentArtifactWriter(storage.run_dir).mark_trial_artifact(
        canonical_trial_id,
        artifact_id=f"trial.{canonical_trial_id}.final_evidence",
        path=evidence_path,
        kind="final_evidence",
        schema_version="check_done_output.txt.v1",
        producer="test",
        replayability="audit",
    )
    result = TrialResult(
        trial_id=trial.id,
        trial_index=trial.index,
        trial_type=trial.type.value,
        sample_id=trial.sample_id,
        output="done",
        exit_code=0,
        timing=Timing(started_at_ms=1, ended_at_ms=2, duration_ms=1),
        metadata={"sample": trial.sample},
    )

    assert _score_one_trial(result, benchmark, storage)
    assert result.scores["canonical"].value == 1.0
    assert result.scores["canonical"].answer == canonical_trial_id


def test_orchestrator_records_canonical_task_output_ref(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    storage.save_trial_output(
        trial.id,
        {
            "trial_id": trial.id,
            "trial_index": trial.index,
            "sample": trial.sample,
            "output": "done",
            "exit_code": 0,
        },
    )

    _mark_canonical_trial_output_artifact(storage=storage, agent=agent, trial=trial)

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in trial_record["artifacts"]
        if item["kind"] == "task_output"
    )

    assert artifact["kind"] == "task_output"
    assert artifact["path"] == "trials/range1/task_output.json"
    assert artifact["sha256"]
    saved_index = json.loads(
        (storage.run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert "trials/range1/task_output.json" in {
        item["path"] for item in saved_index["artifacts"]
    }


def test_orchestrator_records_canonical_final_evidence_ref(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    evidence_path = storage.trial_dir(trial.id) / "runtime" / "check_done_output.txt"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text("success: true\n", encoding="utf-8")

    _mark_canonical_trial_final_evidence_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        evidence_path=evidence_path,
    )

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))
    artifact = next(
        item for item in trial_record["artifacts"] if item["kind"] == "final_evidence"
    )

    assert artifact["path"] == "trials/range1/runtime/check_done_output.txt"
    assert artifact["content_type"] == "text/plain"
    assert artifact["privacy"] == "scorer_private"
    assert trial_record["scoring"]["final_evidence_ref"] == artifact["path"]
    saved_index = json.loads(
        (storage.run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert artifact["path"] in {item["path"] for item in saved_index["artifacts"]}


def test_orchestrator_records_canonical_live_evidence_ref(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    evidence_path = storage.trial_dir(trial.id) / "runtime" / "live_success.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps({"success": True, "source": "poller", "evidence": {"ok": True}}),
        encoding="utf-8",
    )

    _mark_canonical_trial_live_evidence_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        evidence_path=evidence_path,
    )

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))
    artifact = next(
        item for item in trial_record["artifacts"] if item["kind"] == "live_evidence"
    )

    assert artifact["path"] == "trials/range1/runtime/live_success.json"
    assert artifact["content_type"] == "application/json"
    assert artifact["privacy"] == "scorer_private"
    assert trial_record["scoring"]["live_evidence_ref"] == artifact["path"]
    saved_index = json.loads(
        (storage.run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert artifact["path"] in {item["path"] for item in saved_index["artifacts"]}


def test_orchestrator_appends_canonical_trial_lifecycle_events(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    result = TrialResult(
        trial_id=trial.id,
        trial_index=trial.index,
        trial_type=trial.type.value,
        sample_id=trial.sample_id,
        output="done",
        exit_code=0,
        timing=Timing(started_at_ms=1, ended_at_ms=3, duration_ms=2),
        metadata={"status": "completed", "termination_reason": "completed"},
    )

    _mark_canonical_trial_started(storage=storage, agent=agent, trial=trial)
    _mark_canonical_trial_finished(
        storage=storage,
        agent=agent,
        trial=trial,
        result=result,
    )

    events = [
        json.loads(line)
        for line in (storage.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_id"] for event in events] == ["evt_000001", "evt_000002"]
    assert [event["type"] for event in events] == ["trial_started", "trial_finished"]
    assert events[0]["phase"] == "running"
    assert events[1]["phase"] == "completed"
    assert events[1]["payload"]["termination_reason"] == "completed"


def test_orchestrator_records_agent_container_resource_states(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    agent = SimpleNamespace(
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
    )
    container = SimpleNamespace(
        name="cage-demo-agent-run-1-t0",
        image="cage/claude-code:pentestenv",
        labels={"cage.run_id": "run-1", "cage.component": "agent"},
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )

    for status in ("created", "started", "released"):
        _record_agent_container_resource(
            storage=storage,
            run_id="run-1",
            agent=agent,
            trial=trial,
            container=container,
            status=status,
        )

    records = [
        json.loads(line)
        for line in (storage.run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_id"] for record in records] == [
        "res_000001",
        "res_000002",
        "res_000003",
    ]
    assert [record["status"] for record in records] == [
        "created",
        "started",
        "released",
    ]
    assert records[0]["kind"] == "docker_container"
    assert records[0]["trial_id"] == "range1"
    assert records[0]["resource_id"] == "docker_container:cage-demo-agent-run-1-t0"
    assert records[0]["cleanup_action"] == "docker rm -f cage-demo-agent-run-1-t0"
    assert records[0]["metadata"]["image"] == "cage/claude-code:pentestenv"


def test_orchestrator_records_agent_isolation_network_resource_states(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    agent = SimpleNamespace(
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    isolation = AgentIsolationNetwork(
        name="cage-trial-range1-abc12345",
        connected_targets=["target-web-1", "target-api-1"],
        subnet="10.201.7.64/26",
    )

    _record_agent_isolation_network_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        isolation=isolation,
        status="created",
    )
    _record_agent_isolation_network_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        isolation=isolation,
        status="released",
    )
    _record_agent_isolation_network_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        isolation=isolation,
        status="cleanup_failed",
        cleanup_error="docker network rm returned busy",
    )

    records = [
        json.loads(line)
        for line in (storage.run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [record["status"] for record in records] == [
        "created",
        "released",
        "cleanup_failed",
    ]
    assert records[0]["kind"] == "docker_network"
    assert records[0]["trial_id"] == "range1"
    assert records[0]["resource_id"] == "docker_network:cage-trial-range1-abc12345"
    assert records[0]["cleanup_action"] == "docker network rm cage-trial-range1-abc12345"
    assert records[0]["metadata"] == {
        "connected_targets": ["target-web-1", "target-api-1"],
        "subnet": "10.201.7.64/26",
    }
    assert records[2]["cleanup_error"] == "docker network rm returned busy"


def test_orchestrator_records_container_proxy_resource_states(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    agent = SimpleNamespace(
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    proxy = ContainerProxyInstance(
        container=SimpleNamespace(name="agent-container"),
        port=43123,
        pid="4242",
        trial_id="range1",
        config_path="/run/cage-proxy/config.json",
        log_dir="/var/lib/cage/proxy",
        logs_mounted=True,
    )

    _record_container_proxy_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        proxy=proxy,
        status="started",
    )
    _record_container_proxy_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        proxy=proxy,
        status="released",
    )
    _record_container_proxy_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        proxy=proxy,
        status="cleanup_failed",
        cleanup_error="proxy stop failed",
    )

    records = [
        json.loads(line)
        for line in (storage.run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [record["status"] for record in records] == [
        "started",
        "released",
        "cleanup_failed",
    ]
    assert records[0]["kind"] == "container_proxy"
    assert records[0]["provider"] == "docker_exec"
    assert records[0]["trial_id"] == "range1"
    assert records[0]["resource_id"] == "container_proxy:range1"
    assert records[0]["external_id"] == "agent-container:4242"
    assert records[0]["cleanup_action"] == "docker exec agent-container kill -TERM 4242"
    assert records[0]["metadata"] == {
        "base_url": "http://localhost:43123",
        "config_path": "/run/cage-proxy/config.json",
        "container_name": "agent-container",
        "log_dir": "/var/lib/cage/proxy",
        "logs_mounted": True,
        "pid": "4242",
        "port": 43123,
        "trial_id": "range1",
    }
    assert records[2]["cleanup_error"] == "proxy stop failed"
    for sensitive_key in (
        "upstream_base_url",
        "upstream_api_key",
        "extra_headers",
        "http_proxy",
    ):
        assert sensitive_key not in records[0]["metadata"]


def test_orchestrator_records_container_proxy_cleanup_failure_before_reraising(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    agent = SimpleNamespace(
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    proxy = ContainerProxyInstance(
        container=SimpleNamespace(name="agent-container"),
        port=43123,
        pid="4242",
        trial_id="range1",
        config_path="/run/cage-proxy/config.json",
        log_dir="/var/lib/cage/proxy",
        logs_mounted=True,
    )

    def fail_stop(*, artifact_dir: Path | None = None) -> None:
        raise RuntimeError("proxy stop failed")

    proxy.stop = fail_stop  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="proxy stop failed"):
        _stop_container_proxy_resource(
            storage=storage,
            run_id="run-1",
            agent=agent,
            trial=trial,
            proxy=proxy,
            artifact_dir=storage.trial_proxy_dir(trial.id),
        )

    records = [
        json.loads(line)
        for line in (storage.run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["status"] == "cleanup_failed"
    assert records[0]["cleanup_error"] == "proxy stop failed"
    assert records[0]["metadata"]["pid"] == "4242"


def test_orchestrator_records_target_runtime_resource_states(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    agent = SimpleNamespace(
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    target_data = {
        "target_status": "running",
        "target_info": {
            "web": {
                "external_port": 18080,
                "host": "web",
                "port": 80,
                "url": "http://web:80",
                "netcat": "nc web 80",
            },
            "db": {
                "host": "db",
                "port": 3306,
                "url": "mysql://db:3306",
            },
        },
        "runtime": {
            "run_id": "target-run-123",
            "project_name": "cage_pb_siyucms_target-run-123",
            "network_name": "cage_pb_siyucms_default",
            "network_subnet": "10.201.8.0/26",
            "network_gateway": "10.201.8.1",
            "scoring": {"token": "do-not-copy-scoring-config"},
            "debug": {"network": {"services": [{"container_name": "target-web-1"}]}},
        },
    }

    _record_target_runtime_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        chal_id="pb-siyucms",
        target_data=target_data,
        status="created",
    )
    _record_target_runtime_resource(
        storage=storage,
        run_id="run-1",
        agent=agent,
        trial=trial,
        chal_id="pb-siyucms",
        target_data=target_data,
        status="cleanup_requested",
    )

    records = [
        json.loads(line)
        for line in (storage.run_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [record["status"] for record in records] == ["created", "cleanup_requested"]
    assert records[0]["kind"] == "target_runtime"
    assert records[0]["provider"] == "target_server"
    assert records[0]["trial_id"] == "range1"
    assert records[0]["resource_id"] == (
        "target_runtime:range1:pb-siyucms"
    )
    assert records[0]["external_id"] == "cage_pb_siyucms_target-run-123"
    assert records[0]["cleanup_action"] == "target_server DELETE /launch/pb-siyucms?run_id=target-run-123"
    assert records[0]["metadata"] == {
        "challenge_id": "pb-siyucms",
        "network_gateway": "10.201.8.1",
        "network_name": "cage_pb_siyucms_default",
        "network_subnet": "10.201.8.0/26",
        "project_name": "cage_pb_siyucms_target-run-123",
        "public_service_names": ["web"],
        "service_names": ["db", "web"],
        "target_run_id": "target-run-123",
        "target_status": "running",
    }
    for forbidden in ("url", "netcat", "host", "port", "scoring", "debug"):
        assert forbidden not in records[0]["metadata"]


def test_target_teardown_result_maps_to_resource_status() -> None:
    """Target cleanup ledgers should reflect proof, not just method calls."""
    released = TargetTeardownResult(
        challenge_id="pb-siyucms",
        run_id="target-run-123",
        requested=True,
        succeeded=True,
    )
    failed = TargetTeardownResult(
        challenge_id="pb-siyucms",
        run_id="target-run-123",
        requested=True,
        succeeded=False,
        error="delete failed",
    )
    unknown = TargetTeardownResult(
        challenge_id="pb-siyucms",
        run_id="target-run-123",
        requested=True,
        succeeded=None,
    )

    assert _target_teardown_resource_status(released) == ("released", None)
    assert _target_teardown_resource_status(failed) == ("cleanup_failed", "delete failed")
    assert _target_teardown_resource_status(unknown) == ("cleanup_requested", None)


def test_orchestrator_records_canonical_legacy_trajectory_artifact(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    traj_path = storage.trial_dir(trial.id) / "range1.traj"
    traj_path.write_text("Step 0\n", encoding="utf-8")

    _mark_canonical_trial_trajectory_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        traj_path=traj_path,
    )

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in trial_record["artifacts"]
        if item["kind"] == "compat_trajectory"
    )

    assert artifact["path"] == "trials/range1/range1.traj"
    assert artifact["schema_version"] == "traj.compat.v1"
    assert artifact["content_type"] == "text/plain"
    saved_index = json.loads(
        (storage.run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    assert artifact["path"] in {item["path"] for item in saved_index["artifacts"]}


def test_orchestrator_records_canonical_state_snapshot_artifacts(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        max_concurrent=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")
    pre_dir = storage.trial_state_pre_dir(trial.id)
    post_dir = storage.trial_state_post_dir(trial.id)
    (pre_dir / "home" / "agent").mkdir(parents=True)
    (post_dir / "home" / "agent").mkdir(parents=True)
    (pre_dir / "home" / "agent" / "before.txt").write_text(
        "before\n",
        encoding="utf-8",
    )
    (post_dir / "home" / "agent" / "after.txt").write_text(
        "after\n",
        encoding="utf-8",
    )

    _mark_canonical_trial_state_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        state_dir=pre_dir,
        phase="pre",
    )
    _mark_canonical_trial_state_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        state_dir=post_dir,
        phase="post",
    )

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))
    artifacts_by_kind = {
        item["kind"]: item
        for item in trial_record["artifacts"]
        if item["kind"].startswith("state_snapshot_")
    }

    assert artifacts_by_kind["state_snapshot_pre"]["path"] == "trials/range1/state_pre"
    assert artifacts_by_kind["state_snapshot_post"]["path"] == "trials/range1/state_post"
    assert artifacts_by_kind["state_snapshot_pre"]["content_type"] == "inode/directory"
    assert artifacts_by_kind["state_snapshot_post"]["content_type"] == "inode/directory"
    assert artifacts_by_kind["state_snapshot_pre"]["sha256"]
    assert artifacts_by_kind["state_snapshot_post"]["sha256"]
    saved_index = json.loads(
        (storage.run_dir / "artifact_index.json").read_text(encoding="utf-8")
    )
    indexed_paths = {item["path"] for item in saved_index["artifacts"]}
    assert "trials/range1/state_pre" in indexed_paths
    assert "trials/range1/state_post" in indexed_paths


def test_run_experiment_passes_run_artifacts_dir_into_hook_context(tmp_path, monkeypatch):
    project_dir = tmp_path / "examples" / "cvebench"
    config_dir = project_dir / "configs" / "smoke"
    config_dir.mkdir(parents=True)
    project_file = config_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")

    agent = SimpleNamespace(
        id="demo_agent",
        agent_type=SimpleNamespace(name="demo_agent"),
        subject_plan_id="demo_agent:demo-model:stateless",
        max_concurrent=0,
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        stateful=False,
        shared_paths=[],
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_DemoBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
    )

    captured = {}

    def fake_run_agent_trials(run_config, run_agent, storage, trials, hook_ctx, run_id=""):
        captured["run_artifacts_dir"] = storage.run_dir
        return []

    monkeypatch.setattr("cage.experiment.engine.conductor.create_run_id", lambda: "run-fixed")
    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: _PreflightOk())
    monkeypatch.setattr("cage.experiment.engine.conductor._run_agent_trials_serial", fake_run_agent_trials)
    monkeypatch.setattr("cage.experiment.engine.conductor._score_trials", lambda *_args, **_kwargs: None)

    with patch("cage.experiment.engine.conductor.default_trial_sequence") as default_trial_sequence:
        def _capture(ctx):
            captured["hook_ctx_run_artifacts_dir"] = ctx.run_artifacts_dir
            return []

        default_trial_sequence.side_effect = _capture
        run_experiment(config)

    expected_run_dir = (
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-fixed"
    )
    assert captured["hook_ctx_run_artifacts_dir"] == str(expected_run_dir)


def test_resume_rejects_changed_trial_plan_before_overwriting_metadata(tmp_path, monkeypatch):
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")

    agent = SimpleNamespace(
        subject_plan_id="agent:model:stateless",
        id="agent",
        label=lambda: "agent:model:stateless",
        model=SimpleNamespace(
            id="model",
            provider="anthropic",
            model="model",
            base_url="",
            auth_source="",
            api_key="",
            timeout=360,
            max_retries=2,
            extra={},
        ),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        session_args=[],
        plugins=[],
        skill="",
        version="latest",
        image="",
        home="/home/agent/workspace",
        max_rounds=-1,
        max_concurrent=1,
        context_compaction_threshold=0.5,
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1),
        target=TargetConfig(enabled=False),
        resume=True,
        run_id="run-fixed",
    )

    run_dir = project_dir / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    run_dir.mkdir(parents=True)
    previous_plan = [
        {
            "trial_id": "range1/pass_1",
            "trial_index": 0,
            "trial_type": "task",
            "sample_id": "range1",
        },
        {
            "trial_id": "range1/pass_2",
            "trial_index": 1,
            "trial_type": "task",
            "sample_id": "range1",
        },
    ]
    planned_path = run_dir / "planned_trials.json"
    planned_path.write_text(json.dumps(previous_plan, indent=2), encoding="utf-8")
    (run_dir / "project.yml").write_text(project_file.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "config.yaml").write_text("old: config\n", encoding="utf-8")

    monkeypatch.setattr("cage.experiment.engine.preflight.run_preflight", lambda *_args, **_kwargs: _PreflightOk())
    monkeypatch.setattr("cage.experiment.engine.conductor._run_agent_trials_serial", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("cage.experiment.engine.conductor._score_trials", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="trial plan changed"):
        run_experiment(config)

    assert json.loads(planned_path.read_text(encoding="utf-8")) == previous_plan
    assert (run_dir / "config.yaml").read_text(encoding="utf-8") == "old: config\n"


def test_run_single_agent_resume_preserves_prior_canonical_truth_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real resume must classify replay before writing a new initial snapshot."""

    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text(
        """
project:
  name: demo
runtime:
  max_rounds: 5
agents:
  - id: demo_agent
    kind: demo_agent
    model: demo-model
""".lstrip(),
        encoding="utf-8",
    )
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        id="demo_agent",
        label=lambda: "demo_agent:demo-model:stateless",
        model=SimpleNamespace(id="demo-model"),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        session_args=[],
        plugins=[],
        skill="",
        version="latest",
        image="",
        home="/home/agent/workspace",
        max_rounds=-1,
        max_concurrent=1,
        context_compaction_threshold=0.5,
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=5),
        target=TargetConfig(enabled=False),
        resume=True,
        run_id="run-fixed",
    )
    cage_runs = project_dir / ".cage_runs"
    storage = RunStorage(
        cage_runs / "demo_agent:demo-model:stateless" / "run-fixed",
        agent_label=agent.label(),
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    _save_canonical_experiment_snapshot(
        config,
        agent,
        storage,
        [trial],
        "run-fixed",
    )
    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    canonical_trial_id = record["trials"]["records"][0]["trial_id"]
    task_output_ref = "canonical_outputs/task_output.json"
    task_output_path = storage.run_dir / task_output_ref
    task_output_path.parent.mkdir(parents=True)
    task_output_path.write_text(
        json.dumps(
            {
                "trial_id": canonical_trial_id,
                "trial_index": 0,
                "output": "prior canonical output",
                "exit_code": 2,
                "sample": {"id": "range1"},
            }
        ),
        encoding="utf-8",
    )
    writer = ExperimentArtifactWriter(storage.run_dir)
    writer.mark_trial_artifact(
        canonical_trial_id,
        artifact_id=f"trial.{canonical_trial_id}.task_output",
        path=task_output_ref,
        kind="task_output",
        schema_version="task_output.v1",
        producer="test",
        replayability="replayable",
    )
    writer.mark_trial_finished(
        canonical_trial_id,
        status="failed",
        completed_at="2026-06-05T00:01:00Z",
        status_reason="agent_exit_nonzero",
        termination=TrialTermination(reason="agent_exit_nonzero", exit_code=2),
    )
    ResourceLedgerWriter(storage.run_dir).append_resource(
        run_id="run-fixed",
        resource_id="container:prior",
        kind="docker_container",
        provider="docker",
        external_id="prior-container",
        status="released",
        cleanup_action="docker rm -f prior-container",
        timestamp="2026-06-05T00:01:05Z",
        trial_id=canonical_trial_id,
    )

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("resume should replay the prior canonical trial")

    monkeypatch.setattr("cage.experiment.engine.conductor._run_agent_trials_serial", _unexpected_run)
    monkeypatch.setattr("cage.experiment.engine.conductor._score_trials", lambda *_args, **_kwargs: None)

    results = _run_single_agent(replace(config, cage_runs=cage_runs, run_id="run-fixed"), agent)

    assert [result.output for result in results] == ["prior canonical output"]
    resources = [
        json.loads(line)
        for line in (storage.run_dir / "resources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [resource["resource_id"] for resource in resources] == ["container:prior"]


def test_resume_allows_config_drift_when_trial_plan_is_unchanged(tmp_path, caplog):
    run_dir = tmp_path / "run-fixed"
    run_dir.mkdir()
    planned_trials = [
        {
            "trial_id": "range1/pass_1",
            "trial_index": 0,
            "trial_type": "task",
            "sample_id": "range1",
        },
    ]
    trial_plan_fingerprint = _json_fingerprint(planned_trials)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_yml_sha256": "old-project-hash",
                "semantic_config_fingerprint": "old-config-hash",
                "trial_plan_fingerprint": trial_plan_fingerprint,
                "planned_trials": planned_trials,
            }
        ),
        encoding="utf-8",
    )
    current_manifest = {
        "schema_version": 1,
        "project_yml_sha256": "new-project-hash",
        "semantic_config_fingerprint": "new-config-hash",
        "trial_plan_fingerprint": trial_plan_fingerprint,
        "planned_trials": planned_trials,
    }

    caplog.set_level("WARNING", logger="cage.experiment.engine.conductor")

    _assert_resume_compatible(run_dir, current_manifest)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "project.yml changed" in messages
    assert "configuration changed" in messages


def test_resume_allows_legacy_sample_id_casing_in_trial_plan(tmp_path):
    run_dir = tmp_path / "run-fixed"
    run_dir.mkdir()
    previous_trials = [
        {
            "trial_id": "pb-SIYUCMS",
            "trial_index": 0,
            "trial_type": "task",
            "sample_id": "pb-SIYUCMS",
        },
    ]
    current_trials = [
        {
            "trial_id": "pb-siyucms",
            "trial_index": 0,
            "trial_type": "task",
            "sample_id": "pb-siyucms",
        },
    ]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_yml_sha256": "same-project",
                "semantic_config_fingerprint": "same-config",
                "trial_plan_fingerprint": _json_fingerprint(previous_trials),
                "planned_trials": previous_trials,
            }
        ),
        encoding="utf-8",
    )
    current_manifest = {
        "schema_version": 1,
        "project_yml_sha256": "same-project",
        "semantic_config_fingerprint": "same-config",
        "trial_plan_fingerprint": _json_fingerprint(current_trials),
        "planned_trials": current_trials,
    }

    _assert_resume_compatible(run_dir, current_manifest)


def test_resume_uses_canonical_plan_when_legacy_plan_files_are_missing(tmp_path):
    """Canonical ExperimentPlan should be enough to verify resume compatibility."""
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="agent:model:stateless",
        id="agent",
        label=lambda: "agent:model:stateless",
        model=SimpleNamespace(
            id="model",
            provider="anthropic",
            model="model",
            base_url="",
            auth_source="",
            api_key="",
            timeout=360,
            max_retries=2,
            extra={},
        ),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        session_args=[],
        plugins=[],
        skill="",
        version="latest",
        image="",
        home="/home/agent/workspace",
        max_rounds=-1,
        max_concurrent=1,
        context_compaction_threshold=0.5,
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1),
        target=TargetConfig(enabled=False),
        run_id="run-fixed",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    run_dir = project_dir / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    storage = RunStorage(run_dir, agent_label=agent.label())
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-fixed")
    assert not (run_dir / "run_manifest.json").exists()
    assert not (run_dir / "planned_trials.json").exists()

    _assert_resume_compatible(run_dir, _build_run_manifest(config, agent, [trial]))


def test_resume_prefers_canonical_plan_over_stale_manifest_trials(tmp_path):
    """Canonical ExperimentPlan is the trial-plan truth when both artifacts exist."""
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="agent:model:stateless",
        id="agent",
        label=lambda: "agent:model:stateless",
        model=SimpleNamespace(
            id="model",
            provider="anthropic",
            model="model",
            base_url="",
            auth_source="",
            api_key="",
            timeout=360,
            max_retries=2,
            extra={},
        ),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        session_args=[],
        plugins=[],
        skill="",
        version="latest",
        image="",
        home="/home/agent/workspace",
        max_rounds=-1,
        max_concurrent=1,
        context_compaction_threshold=0.5,
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1),
        target=TargetConfig(enabled=False),
        run_id="run-fixed",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    run_dir = project_dir / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    storage = RunStorage(run_dir, agent_label=agent.label())
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-fixed")
    stale_trials = [
        {
            "trial_id": "stale-range",
            "trial_index": 0,
            "trial_type": "task",
            "sample_id": "stale-range",
        }
    ]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_yml_sha256": "stale-project",
                "semantic_config_fingerprint": "stale-config",
                "trial_plan_fingerprint": _json_fingerprint(stale_trials),
                "planned_trials": stale_trials,
            }
        ),
        encoding="utf-8",
    )

    _assert_resume_compatible(run_dir, _build_run_manifest(config, agent, [trial]))


def test_load_planned_trials_falls_back_to_canonical_plan(tmp_path):
    """Resume planning should not require legacy planned_trials.json."""
    project_dir = tmp_path / "examples" / "demo"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="agent:model:stateless",
        id="agent",
        label=lambda: "agent:model:stateless",
        model=SimpleNamespace(
            id="model",
            provider="anthropic",
            model="model",
            base_url="",
            auth_source="",
            api_key="",
            timeout=360,
            max_retries=2,
            extra={},
        ),
        agent_type=SimpleNamespace(name="demo_agent"),
        stateful=False,
        shared_paths=[],
        session_args=[],
        plugins=[],
        skill="",
        version="latest",
        image="",
        home="/home/agent/workspace",
        max_rounds=-1,
        max_concurrent=1,
        context_compaction_threshold=0.5,
        trial_max_workers=1,
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=_OneSampleBenchmark(),
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1),
        target=TargetConfig(enabled=False),
        run_id="run-fixed",
    )
    trial = Trial(
        id="range1",
        index=0,
        type=TrialType.TASK,
        sample={"id": "range1", "content": "task"},
    )
    run_dir = project_dir / ".cage_runs" / "agent:model:stateless" / "run-fixed"
    storage = RunStorage(run_dir, agent_label=agent.label())
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-fixed")
    assert not (run_dir / "planned_trials.json").exists()

    planned = _load_planned_trials(run_dir)

    assert planned == [
        {
            "trial_id": "range1",
            "trial_index": 0,
            "trial_type": "task",
            "sample_id": "range1",
        }
    ]


def test_runtime_max_rounds_overrides_sample_default_for_smoke_runs():
    agent = SimpleNamespace(max_rounds=-1)
    config = SimpleNamespace(execution=SimpleNamespace(max_rounds=5))
    sample = {"id": "pb-siyucms", "max_rounds": 150}

    assert _effective_trial_max_rounds(agent, sample, config) == 5


def test_runtime_zero_max_rounds_overrides_sample_default():
    agent = SimpleNamespace(max_rounds=-1)
    config = SimpleNamespace(execution=SimpleNamespace(max_rounds=0))
    sample = {"id": "pb-siyucms", "max_rounds": 150}

    assert _effective_trial_max_rounds(agent, sample, config) == 0


def test_negative_max_rounds_defers_to_sample_default():
    agent = SimpleNamespace(max_rounds=-1)
    config = SimpleNamespace(execution=SimpleNamespace(max_rounds=-1))
    sample = {"id": "pb-siyucms", "max_rounds": 150}

    assert _effective_trial_max_rounds(agent, sample, config) == 150


def test_zero_max_rounds_writes_terminal_output_without_agent_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoExecContainer:
        is_running = True

        def reset_directory(self, _path: str) -> None:
            return None

        def exec_async(self, *_args, **_kwargs):
            raise AssertionError("max_rounds=0 must not execute the agent")

        def exec(self, *_args, **_kwargs):
            raise AssertionError("max_rounds=0 must not chown/snapshot post-state")

    class NoExecAgentType:
        def env_vars(self, **_kwargs):
            raise AssertionError("max_rounds=0 must not build agent env")

        def build_launch_command(self, *_args, **_kwargs):
            raise AssertionError("max_rounds=0 must not build launch command")

    class ZeroRoundBenchmark(_DemoBenchmark):
        name = "demo"

        def prepare_trial(self, _container, sample, workspace_dir):
            sample["prepared_workspace"] = workspace_dir

        def build_prompt(self, sample):
            return f"prompt for {sample['id']}"

        def scorer(self):
            return SimpleNamespace(strategy="post_run")

    storage = RunStorage(tmp_path / "run", agent_label="agent:model:stateless")
    monkeypatch.setattr(
        "cage.experiment.engine.trial_runner.snapshot_state",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot_dir=tmp_path / "snapshot",
            timestamp_ms=123,
            has_failures=False,
        ),
    )

    trial = Trial(
        id="trial-zero",
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-zero", "content": "task", "max_rounds": 150},
    )
    agent = SimpleNamespace(
        id="agent",
        subject_plan_id="agent:model:stateless",
        max_concurrent=0,
        home="/workspace",
        effective_state_paths=[],
        shared_paths=[],
        max_rounds=-1,
        model=SimpleNamespace(),
        agent_type=NoExecAgentType(),
        session_args=[],
        label=lambda: "agent:model:stateless",
    )
    config = SimpleNamespace(
        benchmark=ZeroRoundBenchmark(),
        execution=SimpleNamespace(
            max_rounds=0,
            live_check=SimpleNamespace(enabled=False),
            timeout=60,
        ),
        proxy=SimpleNamespace(enabled=True),
    )

    result = execute_trial(
        trial=trial,
        agent=agent,
        run=config,
        container=NoExecContainer(),
        storage=storage,
        hook_ctx=HookRegistry(),
        scheduler=RunScheduler.inactive(),
    )

    assert result.exit_code == 0
    assert result.output == "Skipped agent execution because max_rounds=0; no model-call rounds were requested."
    assert trial.status == TrialStatus.COMPLETED
    assert trial.exit_code == 0
    assert (storage.trial_dir("trial-zero") / "prompt.txt").read_text(
        encoding="utf-8"
    ) == "prompt for sample-zero"
    task_output = json.loads(
        (storage.trial_dir("trial-zero") / "task_output.json").read_text(
            encoding="utf-8"
        )
    )
    assert task_output["exit_code"] == 0
    assert task_output["output"] == result.output
    meta = json.loads(
        (storage.trial_dir("trial-zero") / "meta.json").read_text(encoding="utf-8")
    )
    assert meta["status"] == "completed"
    assert meta["termination_reason"] == "zero_rounds"
    assert meta["max_rounds"] == 0


def test_execute_trial_indexes_prompt_artifact_for_canonical_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoExecContainer:
        is_running = True

        def reset_directory(self, _path: str) -> None:
            return None

        def exec_async(self, *_args, **_kwargs):
            raise AssertionError("max_rounds=0 must not execute the agent")

        def exec(self, *_args, **_kwargs):
            raise AssertionError("max_rounds=0 must not chown/snapshot post-state")

    class NoExecAgentType:
        name = "demo_agent"

        def env_vars(self, **_kwargs):
            raise AssertionError("max_rounds=0 must not build agent env")

        def build_launch_command(self, *_args, **_kwargs):
            raise AssertionError("max_rounds=0 must not build launch command")

    class ZeroRoundBenchmark(_OneSampleBenchmark):
        name = "demo"

        def prepare_trial(self, _container, sample, workspace_dir):
            sample["prepared_workspace"] = workspace_dir

        def build_prompt(self, sample):
            return f"prompt for {sample['id']}"

        def scorer(self):
            return SimpleNamespace(strategy="post_run")

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project_file = project_dir / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    agent = SimpleNamespace(
        subject_plan_id="demo_agent:demo-model:stateless",
        home="/workspace",
        id="demo_agent",
        effective_state_paths=[],
        shared_paths=[],
        max_rounds=-1,
        model=SimpleNamespace(id="demo-model"),
        agent_type=NoExecAgentType(),
        session_args=[],
        stateful=False,
        max_concurrent=1,
        label=lambda: "demo_agent:demo-model:stateless",
    )
    config = ExperimentRun(
        name="demo",
        project_file=project_file,
        benchmark=ZeroRoundBenchmark(),
        benchmark_dir=project_dir,
        agents=[agent],
        models={},
        hooks=HookRegistry(),
        logging=LoggingConfig(),
        execution=ExecutionConfig(max_trials_global=1, passk=1, max_rounds=0),
        target=TargetConfig(enabled=False),
    )
    storage = RunStorage(
        project_dir / ".cage_runs" / "demo_agent:demo-model:stateless" / "run-1",
        agent_label="demo_agent:demo-model:stateless",
    )
    monkeypatch.setattr(
        "cage.experiment.engine.trial_runner.snapshot_state",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot_dir=tmp_path / "snapshot",
            timestamp_ms=123,
            has_failures=False,
        ),
    )
    trial = Trial(
        id="trial-zero",
        index=0,
        type=TrialType.TASK,
        sample={"id": "sample-zero", "content": "task", "max_rounds": 150},
    )
    _save_canonical_experiment_snapshot(config, agent, storage, [trial], "run-1")

    execute_trial(
        trial=trial,
        agent=agent,
        run=config,
        container=NoExecContainer(),
        storage=storage,
        hook_ctx=HookRegistry(),
        scheduler=RunScheduler.inactive(),
    )

    record = json.loads(
        (storage.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    trial_ref = record["trials"]["records"][0]["record_ref"]
    trial_record = json.loads((storage.run_dir / trial_ref).read_text(encoding="utf-8"))
    prompt_artifact = next(
        item for item in trial_record["artifacts"] if item["kind"] == "prompt"
    )
    assert prompt_artifact["path"] == "trials/trial-zero/prompt.txt"
    assert prompt_artifact["schema_version"] == "prompt.txt.v1"


def test_append_session_args_quotes_and_appends_args():
    command = _append_session_args(
        "claude -p 'hello'",
        ["--permission-mode", "bypassPermissions", "--append-system-prompt", "CTF mode"],
    )

    assert command == (
        "claude -p 'hello' --permission-mode bypassPermissions "
        "--append-system-prompt 'CTF mode'"
    )


def test_create_container_mounts_only_current_trial_proxy_dir_for_realtime_proxy_sync(tmp_path):
    project_file = tmp_path / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")
    run_dir = tmp_path / ".cage_runs" / "agent:model" / "run-fixed"
    trial_id = "trial-one"
    agent = SimpleNamespace(
        plugins=[],
        effective_image="cage/agent:latest",
        extra_env={},
    )
    config = SimpleNamespace(
        project_file=project_file,
        execution=SimpleNamespace(agent_network_mode=None),
        proxy=SimpleNamespace(enabled=True),
    )

    container = _create_container(
        config,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        "cage-demo",
        run_dir=run_dir,
        trial_id=trial_id,
    )

    proxy_dir = run_dir / "trials" / trial_id / "proxy"
    assert proxy_dir.is_dir()
    assert str(run_dir / "trials") not in container.volumes
    assert container.volumes == {str(proxy_dir): "/var/lib/cage/proxy"}
    assert all(
        not container_path.split(":", 1)[0].startswith("/tmp/")
        for container_path in container.volumes.values()
    )


def test_create_container_includes_agent_container_resources(tmp_path):
    project_file = tmp_path / "project.yml"
    project_file.write_text("project:\n  name: demo\n", encoding="utf-8")

    class _AgentType:
        def container_resources(self, *, home_dir, model):
            assert home_dir == "/home/agent"
            assert model == "model-config"
            return AgentContainerResources(
                volumes={
                    "/host/.credentials.json": (
                        "/home/agent/.claude/.credentials.json"
                    ),
                },
                group_add=["1002"],
            )

    agent = SimpleNamespace(
        plugins=[],
        effective_image="cage/agent:latest",
        extra_env={},
        agent_type=_AgentType(),
        model="model-config",
    )
    config = SimpleNamespace(
        project_file=project_file,
        execution=SimpleNamespace(agent_network_mode=None),
        proxy=SimpleNamespace(enabled=False),
    )

    container = _create_container(
        config,  # type: ignore[arg-type]
        agent,  # type: ignore[arg-type]
        "cage-demo",
    )

    assert container.volumes == {
        "/host/.credentials.json": "/home/agent/.claude/.credentials.json",
    }
    assert container.group_add == ["1002"]


# --------------------------------------------------------------------- #
# Host-side per-run services (e.g. the Claude Code OAuth refresher).
# --------------------------------------------------------------------- #

def _host_service_config(http_proxy, services):
    """Build a stub config whose agents all declare the same `services`."""

    def make_agent(label):
        agent_type = SimpleNamespace(host_run_services=lambda model, *, http_proxy="": services)
        return SimpleNamespace(
            agent_type=agent_type, model=object(), label=lambda: label,
        )

    return SimpleNamespace(
        agents=[make_agent("a:1"), make_agent("a:2")],
        proxy=SimpleNamespace(upstream_http_proxy=http_proxy),
    )


def test_start_host_run_services_dedups_and_stop_terminates(tmp_path):
    import sys

    from cage.agents.base import HostRunService
    from cage.experiment.engine.run_cleanup import RunCleanup
    from cage.experiment.engine.scheduler import RunScheduler

    sleeper = HostRunService(
        name="stub-sleeper",
        argv=[sys.executable, "-c", "import time; time.sleep(60)"],
        env={"FOO": "bar"},
        dedup_key="shared-key",
    )
    config = _host_service_config("http://proxy:7890", [sleeper])

    cleanup = RunCleanup("run-x", RunScheduler.inactive())
    try:
        cleanup.start_host_services(config, tmp_path)
        # Two agents, same dedup_key -> exactly one process.
        assert len(cleanup._host_services) == 1
        proc = cleanup._host_services[0].process
        assert proc.poll() is None
        assert (tmp_path / "stub-sleeper-run-x.log").exists()
    finally:
        cleanup.stop_host_services()

    # Stopped, drained, and SIGTERM'd; second stop is a no-op.
    assert cleanup._host_services == []
    assert proc.poll() is not None
    cleanup.stop_host_services()


def test_start_host_run_services_no_services_is_noop(tmp_path):
    from cage.experiment.engine.run_cleanup import RunCleanup
    from cage.experiment.engine.scheduler import RunScheduler

    config = _host_service_config("", [])
    cleanup = RunCleanup("run-y", RunScheduler.inactive())
    cleanup.start_host_services(config, tmp_path)
    assert cleanup._host_services == []
