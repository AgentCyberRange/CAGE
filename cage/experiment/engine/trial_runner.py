"""Trial execution engine.

This module runs one planned trial end to end: it brings up the agent container,
isolation network, container proxy, and target stack, drives the agent session,
snapshots state, scores the result, and records every resource it allocated for
teardown. ``run_trial_isolated`` is the per-trial entry point the conductor fans
out over; ``execute_trial`` is the inner body it wraps with isolation and
cleanup. Durable record/event/resource writes are delegated to
:class:`cage.artifacts.trial_session.TrialRuntimeSession` and the canonical
record helpers, so this engine owns orchestration, not persistence.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cage.contracts.sample_keys import SAMPLE_MAX_ROUNDS_KEY
from cage.contracts.execution import resolve_max_rounds
from cage.agents.base import AgentInstance
from cage.artifacts.canonical_marks import (
    _mark_canonical_proxy_log_artifact,
    _mark_canonical_trial_final_evidence_artifact,
    _mark_canonical_trial_live_evidence_artifact,
    _mark_canonical_trial_output_artifact,
    _mark_canonical_trial_prompt_artifact,
    _mark_canonical_trial_started,
    _mark_canonical_trial_state_artifact,
    _mark_canonical_trial_trajectory_artifact,
)
from cage.artifacts.record_snapshots import AGENT_HOME
from cage.artifacts.run_storage import RunStorage
from cage.experiment.engine.run_context import ExperimentRun
from cage.experiment.engine.hooks import HookContext
from cage.experiment.model import Trial, TrialResult, TrialStatus
from cage.contracts.logging import bind_run_context, bind_trial_context
from cage.contracts.runtime_state import CHECK_SUPPORTED_KEY
from cage.models import ModelConfig
from cage.proxy.host import (
    CONTAINER_PROXY_LOG_DIR,
    ProxyInstanceConfig,
    start_container_proxy,
)
from cage.proxy.monitor import _ProxyMonitor, _start_live_success_stop_thread
from cage.sandbox.containers import Container
from cage.sandbox.exec import ExecResult, Timing
from cage.experiment.engine.live.monitor import (
    CheckDonePoller,
    ReactiveLiveCheckMonitor,
    _CheckDoneCounter,
)
from cage.artifacts.live_success import live_success_path, load_live_success
from cage.sandbox.naming import _build_agent_container_name, _parse_agent_label
from cage.experiment.engine.resource_recorder import (
    _record_agent_container_resource,
    _record_agent_isolation_network_resource,
    _record_container_proxy_resource,
    _record_target_runtime_resource,
    _stop_container_proxy_resource,
    _target_teardown_resource_status,
    _teardown_agent_isolation_network,
)
from cage.experiment.engine.run_cleanup import RunCleanup
from cage.experiment.engine.scheduler import RunScheduler
from cage.target.services.submit.service import (
    SubmitServiceHandle,
    needs_submit_service,
    start_submit_service,
)
from cage.sandbox.state import (
    diff_snapshots,
    reset_state,
    restore_state,
    snapshot_state,
)
from cage.experiment.engine.termination import (
    cancelled_before_start_termination,
    classify_trial_termination,
    looks_like_model_timeout,
    target_unavailable_termination,
    user_interrupted_termination,
)
from cage.scoring.lifecycle import _score_one_trial
from cage.target.client import ChallengeClient
from cage.target.provisioning import (
    AgentIsolationNetwork,
    attach_agent_to_target,
    build_target_config,
    capture_trial_check_done,
    target_challenge_id,
    inject_ctf_info,
    target_runtime_args,
)

logger = logging.getLogger(__name__)


def _get_challenge_data_with_setup_gate(
    *,
    scheduler: RunScheduler,
    run: ExperimentRun,
    challenge_client: ChallengeClient,
    chal_id: str,
    sample: dict[str, Any],
    trial_id: str,
) -> dict[str, Any]:
    """Launch/wait for a target under the target setup concurrency cap."""

    with scheduler.target_setup_gate(chal_id, trial_id):
        return challenge_client.get_challenge_data(
            chal_id,
            runtime_args=target_runtime_args(run, sample),
        )


def _persist_target_logs(
    storage: RunStorage, trial_id: str, container_logs: Any
) -> None:
    """Best-effort write of captured target container logs to the trial dir.

    Called on both target-launch failure (logs come back in the 500 body) and
    teardown (logs come back in the DELETE body). Never raises — log capture is
    an audit aid, not part of the trial's success path.
    """
    if not container_logs:
        return
    try:
        storage.save_target_logs(trial_id, container_logs)
    except Exception as exc:
        logger.warning("Failed to persist target logs for trial %s: %s", trial_id, exc)


def _target_launch_failure_termination(detail: str, *, scheduler: RunScheduler):
    """Route target-launch failures to the right termination reason.

    When the run scheduler's stop event is set, any exception raised while bringing up
    the target stack (most commonly ``RuntimeError: target setup gate
    cancelled`` from ``RunScheduler.target_setup_gate``) is a symptom of the user
    pressing Ctrl+C — the target itself didn't fail to launch, the caller
    just stopped waiting for it. Classify as ``user_interrupted`` in that
    case so resume re-runs the trial and the inspector chip reflects the
    real cause. Otherwise fall back to ``target_unavailable``.
    """
    if scheduler.is_stopped():
        return user_interrupted_termination()
    return target_unavailable_termination(detail)

def _container_trial_proxy_dir(trial_id: str) -> str:
    """Return the container path that maps to this trial's host proxy dir."""
    del trial_id
    return CONTAINER_PROXY_LOG_DIR

def _effective_trial_max_rounds(
    agent: Any,
    sample: dict[str, Any],
    config: Any,
) -> int:
    """Resolve the round budget for one trial.

    Explicit per-agent and runtime values are user/project overrides. The
    benchmark sample value is only the fallback default for that task profile.
    """

    return resolve_max_rounds(
        getattr(agent, "max_rounds", -1),
        getattr(getattr(config, "execution", None), "max_rounds", -1),
        sample.get(SAMPLE_MAX_ROUNDS_KEY) if isinstance(sample, dict) else None,
    )

def _has_proxy_artifact_mount(container: Container) -> bool:
    """Return whether the current trial proxy dir is bind-mounted in the container."""
    for container_path in container.volumes.values():
        mount_path = str(container_path).split(":", 1)[0]
        if mount_path == CONTAINER_PROXY_LOG_DIR:
            return True
    return False

def _append_session_args(command: str, session_args: list[str]) -> str:
    """Append project.yml session_args to an agent launch command."""
    if not session_args:
        return command
    return f"{command} {' '.join(shlex.quote(str(arg)) for arg in session_args)}"

def _find_agent_pid(container: Container, pattern: str = "claude") -> str:
    """Find the agent process PID inside the container."""
    # Give the process a moment to start
    time.sleep(1.0)
    result = container.exec(f"pgrep -f '{pattern}' | head -1", timeout=5.0)
    pid = result.stdout.strip()
    return pid if pid.isdigit() else ""

def _create_container(
    run: ExperimentRun,
    agent: AgentInstance,
    name: str,
    *,
    run_dir: Path | None = None,
    trial_id: str | None = None,
    run_id: str = "",
) -> Container:
    """Create a Container with plugin volumes and env vars."""
    volumes: dict[str, str] = {}
    group_add: list[str] = []
    privileged = False
    if agent.plugins:
        project_dir = run.project_file.resolve().parent
        volumes = _resolve_plugin_volumes(agent.plugins, project_dir)
    if "openviking-memory" in agent.plugins:
        host_ov_conf = Path.home() / ".openviking" / "ov.conf"
        if host_ov_conf.exists():
            volumes[str(host_ov_conf)] = "/home/agent/.openviking/ov.conf:ro"
    proxy_enabled = bool(getattr(getattr(run, "proxy", None), "enabled", False))
    if run_dir is not None and trial_id and proxy_enabled:
        proxy_dir = run_dir / "trials" / trial_id / "proxy"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        volumes[str(proxy_dir)] = CONTAINER_PROXY_LOG_DIR
    agent_type = getattr(agent, "agent_type", None)
    model = getattr(agent, "model", None)
    if agent_type is not None and model is not None:
        agent_resources = agent_type.container_resources(
            home_dir=AGENT_HOME,
            model=model,
        )
        volumes.update(agent_resources.volumes)
        group_add.extend(agent_resources.group_add)
        privileged = privileged or agent_resources.privileged

    # ``cage.component=agent`` distinguishes orchestrator-owned containers from
    # target_server target containers that carry the same ``cage.run_id`` label.
    # Per-component teardown narrows the SIGTERM sweep so target stacks still
    # go through the server's graceful cleanup path (subnet release, instance
    # unregistration) instead of being force-killed.
    labels: dict[str, str] = {"cage.component": "agent"}
    if run_id:
        labels["cage.run_id"] = run_id
    # A benchmark may require a specific runtime image for the trial (e.g. a
    # white-box debug image ABI-matched to a binary it stages in). It falls back
    # to the agent's configured image when the benchmark expresses no preference.
    image = agent.effective_image
    benchmark = getattr(run, "benchmark", None)
    if benchmark is not None:
        override = benchmark.container_image_override()
        if override:
            image = override
    return Container(
        name=name,
        image=image,
        env_vars={
            "HOME": AGENT_HOME,
            # Surface the host user's UID/GID so per-trial cleanup can chown
            # snapshot artifacts back before docker cp. Agent runs as the
            # in-container ``agent`` user (uid 1000); the host operator's
            # uid may differ, and `sudo` paths still produce root-owned
            # files. Without this, ``cage score`` / manual edits fail with
            # EACCES.
            "CAGE_HOST_UID": str(os.getuid()),
            "CAGE_HOST_GID": str(os.getgid()),
            **agent.extra_env,
        },
        volumes=volumes,
        #extra_hosts=resolve_extra_hosts_for_url(str(agent.model.base_url)),
        network_mode=run.execution.agent_network_mode,
        cap_add=["NET_RAW", "NET_ADMIN"],
        group_add=group_add,
        labels=labels,
        privileged=privileged,
    )

def _setup_container(container: Container, agent: AgentInstance) -> None:
    """Start container, install agent CLI, run agent-specific setup, install plugins."""
    container.start()
    container.setup_workspace(agent.home)

    # Install agent CLI (skip if already pre-installed in image)
    version_check = container.exec(agent.agent_type.version_command(), timeout=10.0)
    if version_check.exit_code == 0 and version_check.stdout.strip() not in ("", "unknown"):
        logger.info("Agent CLI already installed: %s", version_check.stdout.strip())
    else:
        install_cmd = agent.agent_type.install_command(agent.version)
        logger.info("Installing agent CLI: %s", install_cmd)
        install_result = container.exec(install_cmd, timeout=120.0)
        if install_result.exit_code != 0:
            raise RuntimeError(f"CLI install failed: {install_result.stderr[:500]}")

    agent.agent_type.setup_container(
        container, home_dir=AGENT_HOME, model=agent.model,
    )

    # Install plugins
    for plugin_name in agent.plugins:
        logger.info("Installing plugin: %s", plugin_name)
        agent.agent_type.install_plugin(
            container, name=plugin_name, home_dir=AGENT_HOME,
            agent_id=agent.label(),
        )
    if "openviking-memory" in agent.plugins and hasattr(
        agent.agent_type, "start_openviking_server"
    ):
        agent.agent_type.start_openviking_server(container, home_dir=AGENT_HOME)

def run_trial_isolated(
    run: ExperimentRun,
    agent: AgentInstance,
    trial: Trial,
    cage_runs: Path,
    run_id: str,
    reporter: Any | None = None,
    *,
    scheduler: RunScheduler,
    cleanup: RunCleanup,
) -> TrialResult:
    """Run a single trial in full isolation: own container + own target stack.

    Lifecycle: start+setup container → target launch+connect → execute trial →
    target teardown → stop.

    Honours the run-wide admission gate: when host memory is hot, this call
    blocks here (before any container is created) until pressure clears or
    the run's stop_event fires.
    """
    agent_label = agent.label()
    bind_run_context(agent_label=agent_label)
    trial.index = trial.index or 0
    bind_trial_context(trial_id=trial.id, trial_index=trial.index, sample_id=trial.sample_id)

    # Admission gate — block before we spend any docker resources.
    if not scheduler.admission.wait_until_ok(stop_event=scheduler.stop_event):
        logger.info("Trial %s cancelled at admission gate (run interrupted)", trial.id)
        return TrialResult(
            trial_id=trial.id,
            trial_index=trial.index,
            trial_type=trial.type.value,
            sample_id=trial.sample_id,
            output="",
            exit_code=-1,
            timing=Timing(started_at_ms=0, ended_at_ms=0, duration_ms=0),
            error="cancelled at admission gate",
            # Never started executing the agent (blocked at the admission gate
            # when the run was interrupted) → cancelled_before_start, NOT the
            # bare "interrupted" which reads as "was running when stopped".
            metadata=cancelled_before_start_termination().to_metadata(),
        )

    # Two-level trial concurrency gate. ``concurrency_gate`` blocks the worker
    # thread until both per-agent and global Semaphores are available. If the
    # run is interrupted while waiting, it raises RuntimeError below and we
    # surface that as a cancelled TrialResult.
    try:
        gate_cm = scheduler.concurrency_gate(agent)
        gate_cm.__enter__()
    except RuntimeError as exc:
        logger.info("Trial %s cancelled at concurrency gate: %s", trial.id, exc)
        return TrialResult(
            trial_id=trial.id, trial_index=trial.index, trial_type=trial.type.value,
            sample_id=trial.sample_id, output="", exit_code=-1,
            timing=Timing(started_at_ms=0, ended_at_ms=0, duration_ms=0),
            error=str(exc),
            # Cancelled while still queued behind the concurrency gate — the
            # agent never ran → cancelled_before_start, not "interrupted".
            metadata=cancelled_before_start_termination().to_metadata(),
        )

    # Only NOW is the trial actually executing: it has cleared the admission
    # gate and holds a concurrency slot. Report "started" here (not when the
    # worker thread was picked up) so the live ``running=N`` count reflects
    # trials in flight, not trials queued behind the concurrency gate. A pool
    # sized to ``max_trials_global`` can hold many more threads than
    # ``max_concurrent`` lets run at once; reporting before the gate made every
    # queued trial look "running".
    if reporter is not None:
        reporter.trial_started(
            agent_label=agent_label,
            trial_id=trial.id,
            sample_id=trial.sample_id,
            trial_index=trial.index,
        )

    # Directory layout: .cage_runs/{label}/run-{timestamp}/
    agent_dir_name, _mode = _parse_agent_label(agent_label)
    run_root = cage_runs / agent_dir_name / run_id
    storage = RunStorage(run_root, agent_label=agent_label)

    # Canonical "running" mark fires HERE — after the admission + concurrency
    # gates, co-located with the live-progress ``trial_started`` above — so the
    # durable record.json flips planned→running only for trials genuinely in
    # flight. Marking it at pool-admission (before the gate) made every trial
    # queued behind the concurrency semaphore look "running" on disk, even
    # though the CLI correctly counted only the N trials past the gate.
    _mark_canonical_trial_started(storage=storage, agent=agent, trial=trial)

    container_name = _build_agent_container_name(
        agent_dir_name,
        run_id,
        f"t{trial.index}-{int(time.time()) % 100000}",
    )
    container = _create_container(
        run,
        agent,
        container_name,
        run_dir=run_root,
        trial_id=trial.id,
        run_id=run_id,
    )
    _record_agent_container_resource(
        storage=storage,
        run_id=run_id,
        agent=agent,
        trial=trial,
        container=container,
        status="created",
    )

    target_env = build_target_config(run, cage_runs, run_id=run_id)
    challenge_client = ChallengeClient(config=target_env) if target_env else None
    cleanup.register_client(challenge_client)

    target_data: dict[str, Any] | None = None
    attached_network: str | None = None
    agent_isolation: AgentIsolationNetwork | None = None
    target_launch_error: str | None = None

    try:
        _setup_container(container, agent)
        _record_agent_container_resource(
            storage=storage,
            run_id=run_id,
            agent=agent,
            trial=trial,
            container=container,
            status="started",
        )

        # Launch target stack + connect to network
        attach = _launch_and_attach_target(
            agent=agent,
            challenge_client=challenge_client,
            container=container,
            run=run,
            run_id=run_id,
            scheduler=scheduler,
            storage=storage,
            trial=trial,
            trial_id=trial.id,
        )
        target_data = attach.target_data
        attached_network = attach.attached_network
        agent_isolation = attach.agent_isolation
        target_launch_error = attach.target_launch_error

        if target_launch_error is not None:
            # Fail fast — no LLM calls, no agent CLI exec. Persist a structured
            # TrialResult so the inspector / dashboard show *why* this trial
            # never started instead of silently skipping it.
            now_ms = int(time.time() * 1000)
            term = _target_launch_failure_termination(target_launch_error, scheduler=scheduler)
            chal_for_log = target_challenge_id(trial.sample, trial.sample_id) or trial.sample_id
            logger.error(
                "Trial %s skipped: target %s unavailable (%s) — %s",
                trial.id, chal_for_log, term.reason.value, target_launch_error,
            )
            storage.save_trial_meta(trial.id, {
                "trial_id": trial.id,
                "trial_index": trial.index,
                "trial_type": trial.type.value,
                "sample_id": trial.sample_id,
                "exit_code": -1,
                "error": term.detail,
                **term.to_metadata(),
                "timing": {
                    "started_at_ms": now_ms,
                    "ended_at_ms": now_ms,
                    "duration_ms": 0,
                },
            })
            return TrialResult(
                trial_id=trial.id,
                trial_index=trial.index,
                trial_type=trial.type.value,
                sample_id=trial.sample_id,
                output="",
                exit_code=-1,
                timing=Timing(started_at_ms=now_ms, ended_at_ms=now_ms, duration_ms=0),
                error=term.detail,
                metadata=term.to_metadata(),
            )

        if target_data:
            inject_ctf_info(trial.sample, target_data)

        hook_ctx = HookContext(
            experiment_config={"name": run.name},
            samples=[trial.sample],
            trials_completed=[],
            trials_pending=[],
            run_artifacts_dir=str(storage.run_dir),
        )

        return execute_trial(
            trial=trial, agent=agent, run=run,
            container=container, storage=storage, hook_ctx=hook_ctx,
            scheduler=scheduler,
            challenge_client=None,  # target already handled above
            run_id=run_id,
            reporter=reporter,
        )

    finally:
        _cleanup_trial_resources(
            agent=agent,
            agent_isolation=agent_isolation,
            attached_network=attached_network,
            challenge_client=challenge_client,
            container=container,
            run_id=run_id,
            storage=storage,
            submit_handle=None,
            target_data=target_data,
            trial=trial,
            trial_id=trial.id,
        )
        if challenge_client is not None:
            # Per-trial close MUST skip the run-wide DELETE — sibling trials
            # share ``cage_run_id`` and would lose their target stacks too.
            # Run-end DELETE is handled by ``_teardown_all_active_clients``.
            try:
                challenge_client.close(delete_run=False)
            except Exception:
                pass
        stop_error = None
        try:
            container.stop()
        except Exception as exc:
            stop_error = str(exc)
            _record_agent_container_resource(
                storage=storage,
                run_id=run_id,
                agent=agent,
                trial=trial,
                container=container,
                status="cleanup_failed",
                cleanup_error=stop_error,
            )
            raise
        else:
            _record_agent_container_resource(
                storage=storage,
                run_id=run_id,
                agent=agent,
                trial=trial,
                container=container,
                status="released",
            )
        # Release per-model + global trial semaphores last so they cover
        # the entire container/target lifecycle (LLM calls happen inside
        # ``execute_trial`` above; bounding the whole window is correct).
        try:
            gate_cm.__exit__(None, None, None)
        except Exception:
            pass

def _resolve_plugin_volumes(
    plugins: list[str], project_dir: Path,
) -> dict[str, str]:
    """Build host→container volume mappings for plugin marketplaces.

    Looks for ``plugins/{name}-marketplace/`` under *project_dir*, then
    falls back to the cage repo root ``plugins/`` directory.
    """
    volumes: dict[str, str] = {}
    for name in plugins:
        marketplace_dir = f"{name}-marketplace"
        # Try project-local first, then cage package root
        candidate = project_dir / "plugins" / marketplace_dir
        if not candidate.is_dir():
            candidate = Path(__file__).resolve().parent.parent / "plugins" / marketplace_dir
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"Plugin marketplace not found for '{name}'. "
                f"Expected directory: plugins/{marketplace_dir}"
            )
        container_path = f"/opt/cage-plugins/{marketplace_dir}"
        volumes[str(candidate)] = f"{container_path}:ro"
    return volumes

def _model_for_trial(model: ModelConfig, trial_index: int) -> ModelConfig:
    """Pin a trial to one API key when the model declares a key pool.

    Multi-account load balancing: ``config/models.yml`` may list several
    ``api_keys`` for one endpoint (e.g. separate accounts, each with its own
    rate/quota window). Each trial is assigned one key round-robin by its global
    ``trial.index``, so load spreads evenly across accounts while a single
    conversation stays on one key (keeping the endpoint's prompt cache warm).
    The returned model differs from the input only in ``api_key`` and is used
    for both the agent's per-trial credential seeding and the proxy upstream,
    so the rotated key is what actually reaches the endpoint. With zero or one
    key this is a no-op.
    """
    pool = getattr(model, "api_key_pool", []) or []
    if len(pool) <= 1:
        return model
    return replace(model, api_key=pool[trial_index % len(pool)])


def _trial_model_for_agent(agent: AgentInstance, trial_index: int) -> ModelConfig:
    """Resolve the concrete endpoint a trial should use for ``agent``.

    Composes two round-robins keyed on the global ``trial.index``:

    * **Multi-source** (``agent.model_sources``) — interchangeable registered
      endpoints behind one logical model key. Each trial picks one source, so
      base_url / api_key / upstream model-name / protocol rotate. The run is
      still grouped under ``agent.model.id`` (the logical key); the chosen
      source's own id rides along on the returned model for per-trial
      traceability and does not change grouping.
    * **Multi-account** (``api_keys`` on the chosen source) — keys rotate within
      the source, advancing once per full pass over the sources so every
      (source, key) pair is used before any repeats.

    Single source + single key is a no-op. The returned model feeds the agent
    launch command, per-trial credential seeding, and the proxy upstream alike.
    """
    sources = getattr(agent, "model_sources", None) or [agent.model]
    n_src = len(sources)
    base = sources[trial_index % n_src]
    pool = getattr(base, "api_key_pool", []) or []
    if len(pool) <= 1:
        return base
    return replace(base, api_key=pool[(trial_index // n_src) % len(pool)])


@dataclass
class _TargetAttach:
    """Result of launching + attaching the per-trial target."""
    target_data: dict[str, Any] | None
    attached_network: str | None
    agent_isolation: AgentIsolationNetwork | None
    target_launch_error: str | None


def _launch_and_attach_target(
    *,
    agent: AgentInstance,
    challenge_client: ChallengeClient | None,
    container: Container,
    run: ExperimentRun,
    run_id: str,
    scheduler: RunScheduler,
    storage: RunStorage,
    trial: Trial,
    trial_id: str,
) -> _TargetAttach:
    """Launch the per-trial target (if any) and attach the agent container to its network."""
    target_data: dict[str, Any] | None = None
    attached_network: str | None = None
    agent_isolation: AgentIsolationNetwork | None = None
    target_launch_error: str | None = None
    if challenge_client is not None:
        chal_id = target_challenge_id(trial.sample, trial.sample_id)
        if chal_id:
            try:
                target_data = _get_challenge_data_with_setup_gate(
                    scheduler=scheduler,
                    run=run,
                    challenge_client=challenge_client,
                    chal_id=chal_id,
                    sample=trial.sample,
                    trial_id=trial_id,
                )
            except Exception as exc:
                logger.error("ChallengeClient launch failed for chal_id=%s: %s", chal_id, exc)
                target_launch_error = f"{type(exc).__name__}: {exc}"
                target_data = None

            if (
                target_launch_error is None
                and isinstance(target_data, dict)
                and target_data.get("target_status") == "stopped"
            ):
                # See the matching block in ``run_trial_isolated`` for the
                # rationale — embed the actual response body when available
                # so the trial page shows the real cause.
                init_err = str(target_data.get("target_init_error") or "").strip()
                if init_err:
                    target_launch_error = (
                        "target_status=stopped — server-side launch failed:\n\n"
                        + init_err
                    )
                else:
                    target_launch_error = (
                        "target_status=stopped — server-side launch failed; "
                        "check 'Init failed:' in the run log and "
                        "target_server-<run_id>.log for the response body"
                    )

            # Persist the target container logs the server captured on failure
            # (carried in the 500 body) so the real crash cause is auditable
            # instead of being reduced to the one-line termination_detail.
            if isinstance(target_data, dict):
                _persist_target_logs(
                    storage, trial.id, target_data.get("target_container_logs")
                )

            if target_launch_error is None and target_data:
                _record_target_runtime_resource(
                    storage=storage,
                    run_id=run_id,
                    agent=agent,
                    trial=trial,
                    chal_id=chal_id,
                    target_data=target_data,
                    status="created",
                )
                server_network = (
                    target_data.get("runtime", {}).get("network_name")
                    if target_data
                    else None
                )
                try:
                    attached_network, agent_isolation = attach_agent_to_target(
                        container,
                        trial_id,
                        target_data or {},
                        server_network,
                        run.target.agent_network_isolation,
                    )
                    if agent_isolation is not None:
                        _record_agent_isolation_network_resource(
                            storage=storage,
                            run_id=run_id,
                            agent=agent,
                            trial=trial,
                            isolation=agent_isolation,
                            status="created",
                        )
                except Exception as exc:
                    logger.error(
                        "Agent attach to target failed for chal_id=%s: %s",
                        chal_id, exc,
                    )
                    target_launch_error = f"agent_attach: {type(exc).__name__}: {exc}"
    return _TargetAttach(
        target_data=target_data,
        attached_network=attached_network,
        agent_isolation=agent_isolation,
        target_launch_error=target_launch_error,
    )


def _target_launch_failure_result(
    *,
    scheduler: RunScheduler,
    started: int,
    storage: RunStorage,
    target_launch_error: str,
    trial: Trial,
    trial_id: str,
) -> TrialResult:
    """Build the fail-fast TrialResult for a target that never came up (no LLM calls)."""
    now_ms = int(time.time() * 1000)
    term = _target_launch_failure_termination(target_launch_error, scheduler=scheduler)
    chal_for_log = target_challenge_id(trial.sample, trial.sample_id) or trial.sample_id
    logger.error(
        "Trial %s skipped: target %s unavailable (%s) — %s",
        trial_id, chal_for_log, term.reason.value, target_launch_error,
    )
    storage.save_trial_meta(trial_id, {
        "trial_id": trial_id,
        "trial_index": trial.index,
        "trial_type": trial.type.value,
        "sample_id": trial.sample_id,
        "exit_code": -1,
        "error": term.detail,
        **term.to_metadata(),
        "timing": {
            "started_at_ms": started,
            "ended_at_ms": now_ms,
            "duration_ms": now_ms - started,
        },
    })
    return TrialResult(
        trial_id=trial_id,
        trial_index=trial.index,
        trial_type=trial.type.value,
        sample_id=trial.sample_id,
        output="",
        exit_code=-1,
        timing=Timing(
            started_at_ms=started, ended_at_ms=now_ms,
            duration_ms=now_ms - started,
        ),
        error=term.detail,
        metadata=term.to_metadata(),
    )


def _trial_setup_interrupted_result(
    *,
    started: int,
    storage: RunStorage,
    trial: Trial,
    trial_id: str,
) -> TrialResult:
    """Build the TrialResult for a trial cut short by a stop during pre-agent setup.

    When the run is stopping, an exception in the pre-agent phase (state snapshot
    → workspace reset → proxy start) is a symptom of the interrupt, not a trial
    error — most commonly the SIGINT handler force-removed the container, so the
    next container op sees ``No such container``. The agent never produced output,
    so classify as ``user_interrupted`` (the INTERRUPTED bucket) — mirroring
    :func:`_target_launch_failure_termination` — so resume re-runs the trial and
    the inspector shows the real cause instead of a spurious ``trial_error``.
    """
    now_ms = int(time.time() * 1000)
    term = user_interrupted_termination()
    storage.save_trial_meta(trial_id, {
        "trial_id": trial_id,
        "trial_index": trial.index,
        "trial_type": trial.type.value,
        "sample_id": trial.sample_id,
        "exit_code": -1,
        "error": term.detail,
        **term.to_metadata(),
        "timing": {
            "started_at_ms": started,
            "ended_at_ms": now_ms,
            "duration_ms": now_ms - started,
        },
    })
    return TrialResult(
        trial_id=trial_id,
        trial_index=trial.index,
        trial_type=trial.type.value,
        sample_id=trial.sample_id,
        output="",
        exit_code=-1,
        timing=Timing(started_at_ms=started, ended_at_ms=now_ms, duration_ms=now_ms - started),
        error=term.detail,
        metadata=term.to_metadata(),
    )


def _zero_rounds_result(
    *,
    agent: AgentInstance,
    effective_max_rounds: int,
    run: ExperimentRun,
    started: int,
    storage: RunStorage,
    trial: Trial,
    trial_id: str,
) -> TrialResult:
    """Build the completed TrialResult when max_rounds=0 (target+prompt prepared, agent skipped)."""
    ended = int(time.time() * 1000)
    output = (
        "Skipped agent execution because max_rounds=0; "
        "no model-call rounds were requested."
    )
    timing = Timing(
        started_at_ms=started,
        ended_at_ms=ended,
        duration_ms=ended - started,
    )
    termination_meta = {
        "status": TrialStatus.COMPLETED.value,
        "termination_reason": "zero_rounds",
        "termination_detail": (
            "Target and prompt were prepared, but agent execution was "
            "skipped because max_rounds=0."
        ),
        "termination_source": "orchestrator",
    }
    storage.save_trial_meta(trial_id, {
        "trial_id": trial_id,
        "trial_index": trial.index,
        "trial_type": trial.type.value,
        "sample_id": trial.sample_id,
        "exit_code": 0,
        "max_rounds": effective_max_rounds,
        **termination_meta,
        "timing": {
            "started_at_ms": started,
            "ended_at_ms": ended,
            "duration_ms": ended - started,
        },
    })
    storage.save_trial_output(trial_id, {
        "output": output,
        "exit_code": 0,
        "sample": trial.sample,
    })
    _mark_canonical_trial_output_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
    )
    trial.status = TrialStatus.COMPLETED
    trial.output = output
    trial.exit_code = 0
    trial.timing = timing

    result = TrialResult(
        trial_id=trial_id,
        trial_index=trial.index,
        trial_type=trial.type.value,
        sample_id=trial.sample_id,
        output=output,
        exit_code=0,
        timing=timing,
        state_pre=storage.trial_state_pre_dir(trial_id),
        metadata={
            "sample": trial.sample,
            **termination_meta,
        },
    )
    if getattr(run.benchmark.scorer(), "strategy", "per_trial") == "per_trial":
        _score_one_trial(result, run.benchmark, storage)
    return result


def _trial_failure_result(
    *,
    exc: Exception,
    run: ExperimentRun,
    started: int,
    storage: RunStorage,
    trial: Trial,
    trial_id: str,
) -> TrialResult:
    """Build the FAILED TrialResult + persist meta for an unexpected trial exception."""
    ended = int(time.time() * 1000)
    error = str(exc)
    logger.exception("Trial %s failed", trial_id)

    trial.status = TrialStatus.FAILED
    trial.error = error

    storage.save_trial_meta(trial_id, {
        "trial_id": trial_id,
        "trial_index": trial.index,
        "trial_type": trial.type.value,
        "sample_id": trial.sample_id,
        "error": error,
        **_trial_termination_metadata(
            exit_code=-1,
            timed_out=False,
            terminated_by_limit=False,
            error=error,
            timeout_seconds=run.execution.timeout,
        ),
        "timing": {
            "started_at_ms": started,
            "ended_at_ms": ended,
            "duration_ms": ended - started,
        },
    })

    return TrialResult(
        trial_id=trial_id,
        trial_index=trial.index,
        trial_type=trial.type.value,
        sample_id=trial.sample_id,
        output="",
        exit_code=-1,
        timing=Timing(started_at_ms=started, ended_at_ms=ended, duration_ms=ended - started),
        error=error,
        metadata=_trial_termination_metadata(
            exit_code=-1,
            timed_out=False,
            terminated_by_limit=False,
            error=error,
            timeout_seconds=run.execution.timeout,
        ),
    )


def _cleanup_trial_resources(
    *,
    agent: AgentInstance,
    agent_isolation: AgentIsolationNetwork | None,
    attached_network: str | None,
    challenge_client: ChallengeClient | None,
    container: Container,
    run_id: str,
    storage: RunStorage,
    submit_handle: SubmitServiceHandle | None,
    target_data: dict[str, Any] | None,
    trial: Trial,
    trial_id: str,
) -> None:
    """Trial teardown: stop submit service, disconnect network, teardown target instance."""
    # Submit service cleanup
    if submit_handle is not None:
        try:
            submit_handle.stop()
            logger.info("Submit service stopped for trial %s", trial_id)
        except Exception as exc:
            logger.warning("Submit service cleanup failed for trial %s: %s", trial_id, exc)
    # target cleanup: disconnect from runtime network + teardown target
    if attached_network and container.is_running:
        try:
            container.sync_runtime_network(None)
        except Exception:
            pass
    if agent_isolation is not None:
        _teardown_agent_isolation_network(
            storage=storage,
            run_id=run_id,
            agent=agent,
            trial=trial,
            isolation=agent_isolation,
        )
    if challenge_client is not None and target_data:
        chal_id = target_challenge_id(trial.sample, trial.sample_id)
        if chal_id:
            try:
                teardown_result = challenge_client.finish_challenge(chal_id)
                _persist_target_logs(
                    storage, trial_id, getattr(teardown_result, "container_logs", None)
                )
                teardown_status, cleanup_error = _target_teardown_resource_status(
                    teardown_result,
                )
                _record_target_runtime_resource(
                    storage=storage,
                    run_id=run_id,
                    agent=agent,
                    trial=trial,
                    chal_id=chal_id,
                    target_data=target_data,
                    status=teardown_status,
                    cleanup_error=cleanup_error,
                )
                if teardown_status == "cleanup_failed":
                    logger.warning(
                        "target teardown failed for chal_id=%s: %s",
                        chal_id,
                        cleanup_error or "unknown error",
                    )
            except Exception as exc:
                _record_target_runtime_resource(
                    storage=storage,
                    run_id=run_id,
                    agent=agent,
                    trial=trial,
                    chal_id=chal_id,
                    target_data=target_data,
                    status="cleanup_failed",
                    cleanup_error=str(exc),
                )
                logger.warning("target teardown failed for chal_id=%s: %s", chal_id, exc)


def execute_trial(
    *,
    trial: Trial,
    agent: AgentInstance,
    run: ExperimentRun,
    container: Container,
    storage: RunStorage,
    hook_ctx: HookContext,
    scheduler: RunScheduler,
    challenge_client: ChallengeClient | None = None,
    run_id: str = "",
    reporter: Any | None = None,
) -> TrialResult:
    """Execute a single trial inside a container.

    When challenge_client is provided, this will:
    1. Launch a target instance (per_agent mode) via ChallengeClient
    2. Connect the agent container to the target's runtime network
    3. Inject target stack info into sample (alias or network mode)
    4. Execute the agent
    5. Disconnect from the runtime network
    6. Teardown the target instance

    When live_check is enabled for supported benchmarks, this will:
    - Start a root-owned submit service inside the agent container
    - Set sample metadata for prompt rendering (check_supported)
    - Stop the submit service after trial execution
    """
    trial_id = trial.id
    workspace_dir = agent.home
    started = int(time.time() * 1000)

    # target stack lifecycle: launch + connect network
    attach = _launch_and_attach_target(
        agent=agent,
        challenge_client=challenge_client,
        container=container,
        run=run,
        run_id=run_id,
        scheduler=scheduler,
        storage=storage,
        trial=trial,
        trial_id=trial_id,
    )
    target_data = attach.target_data
    attached_network = attach.attached_network
    agent_isolation = attach.agent_isolation
    target_launch_error = attach.target_launch_error
    if target_launch_error is not None:
        return _target_launch_failure_result(
            scheduler=scheduler,
            started=started,
            storage=storage,
            target_launch_error=target_launch_error,
            trial=trial,
            trial_id=trial_id,
        )

    # Live submit service lifecycle
    submit_handle: SubmitServiceHandle | None = None
    try:
        # Stop already requested by the time this trial reached the pre-agent
        # phase: don't drive it deeper into setup (snapshot → reset → proxy/agent)
        # only to have its container force-removed under it. The agent never ran →
        # interrupted, not a trial_error. The admission gate applies the same rule
        # one step earlier, to trials still queued behind it.
        if scheduler.is_stopped():
            return _trial_setup_interrupted_result(
                started=started, storage=storage, trial=trial, trial_id=trial_id,
            )

        # 1. Snapshot pre-state
        pre_snapshot = snapshot_state(
            container,
            state_paths=agent.effective_state_paths,
            home_dir=AGENT_HOME,
            output_dir=storage.trial_state_pre_dir(trial_id),
        )
        _mark_canonical_trial_state_artifact(
            storage=storage,
            agent=agent,
            trial=trial,
            state_dir=storage.trial_state_pre_dir(trial_id),
            phase="pre",
        )

        # 2. Reset workspace (always fresh per trial)
        container.reset_directory(workspace_dir)

        # 2b. Restore shared paths that were wiped by workspace reset.
        # Only restore paths that are inside the workspace dir — paths outside
        # it (e.g. /home/agent/.claude) are not affected by reset_directory.
        if agent.shared_paths:
            paths_in_workspace = [
                p for p in agent.shared_paths
                if p.startswith(workspace_dir.rstrip("/") + "/")
            ]
            if paths_in_workspace:
                from cage.sandbox.state import StateSnapshot
                ws_snapshot = StateSnapshot(
                    snapshot_dir=pre_snapshot.snapshot_dir,
                    state_paths=tuple(paths_in_workspace),
                    timestamp_ms=pre_snapshot.timestamp_ms,
                )
                restore_state(
                    container,
                    snapshot=ws_snapshot,
                    home_dir=AGENT_HOME,
                )

        # 3. Benchmark prepares workspace (user-controlled)
        run.benchmark.prepare_trial(container, trial.sample, workspace_dir)

        # Inject target stack info into sample for prompt building
        if target_data:
            inject_ctf_info(trial.sample, target_data)

        # Live check: start in-container submit service for supported benchmarks
        benchmark_name = run.benchmark.name
        live_check = run.execution.live_check
        if needs_submit_service(run.benchmark, live_check.enabled):
            expected_answer = str(trial.sample.get("flag", "")).strip()
            question_id = str(trial.sample.get("id") or trial.sample_id)
            if expected_answer:
                try:
                    submit_handle = start_submit_service(
                        container=container,
                        question_id=question_id,
                        expected_answer=expected_answer,
                        max_checks=live_check.max_calls,
                        benchmark=benchmark_name,
                        trial_artifact_dir=storage.trial_dir(trial_id),
                        trial_id=trial_id,
                        container_artifact_path="",
                    )
                    # Set sample metadata for prompt rendering
                    trial.sample[CHECK_SUPPORTED_KEY] = True
                    logger.info(
                        "Live submit service started for trial %s",
                        trial_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to start submit service for trial %s: %s",
                        trial_id, exc,
                    )
            else:
                logger.warning(
                    "Live check enabled but no expected answer for trial %s; "
                    "skipping submit service",
                    trial_id,
                )

        # 4. Build prompt (user-controlled)
        prompt = run.benchmark.build_prompt(trial.sample)
        storage.save_trial_prompt(trial_id, prompt)
        _mark_canonical_trial_prompt_artifact(
            storage=storage,
            agent=agent,
            trial=trial,
        )

        effective_max_rounds = _effective_trial_max_rounds(
            agent, trial.sample, run,
        )
        if effective_max_rounds == 0:
            return _zero_rounds_result(
                agent=agent,
                effective_max_rounds=effective_max_rounds,
                run=run,
                started=started,
                storage=storage,
                trial=trial,
                trial_id=trial_id,
            )

        # Multi-account load balancing: when the model lists an ``api_keys``
        # pool, pin this trial to one key round-robin by ``trial.index``. The
        # rotated key flows to both the proxy upstream and the agent's
        # per-trial credential seeding below, so it's the key that reaches the
        # endpoint. No-op for single-key models.
        model = _trial_model_for_agent(agent, trial.index)
        _srcs = getattr(agent, "model_sources", None) or []
        if len(_srcs) > 1:
            logger.info(
                "Trial %s pinned to source %d/%d (%s -> %s)",
                trial_id, trial.index % len(_srcs), len(_srcs),
                agent.model.id, model.id,
            )
        _pool = getattr(model, "api_key_pool", []) or []
        if len(_pool) > 1:
            logger.info(
                "Trial %s pinned to api_key pool slot %d/%d",
                trial_id,
                (trial.index // max(1, len(_srcs) or 1)) % len(_pool), len(_pool),
            )

        # 5. Start per-trial proxy (inside container)
        proxy = None
        proxy_url = ""
        if run.proxy.enabled:
            proxy_artifact_dir = storage.trial_proxy_dir(trial_id)
            proxy_logs_mounted = _has_proxy_artifact_mount(container)
            # Subscription-mode fallback: when the model declares an
            # ``auth_source`` (OAuth credentials on the host) and no
            # explicit ``base_url`` is given, forward to anthropic.com so
            # the CLI's OAuth Bearer header reaches the real subscription
            # backend. ``upstream_api_key`` is left empty — the client
            # already supplies Authorization and ``container_proxy.py``
            # passes it through verbatim.
            _sub_mode = bool(model.auth_source) and not model.base_url
            # RL mode: tag every upstream LLM call with the trial's join key so an
            # external trainer can group one trajectory's calls. Rides the
            # existing ``extra_headers`` channel (host writes it into the proxy
            # config, the in-container proxy forwards it on every request), so no
            # proxy edit is needed. Same key the reward report uses → they match
            # byte-for-byte. Off unless the model declares ``rl_reward_sink``.
            _extra_headers = dict(model.extra_headers or {})
            if model.rl_enabled:
                from cage.rl.reward_sink import rl_trial_id

                _extra_headers["X-Trial-Id"] = rl_trial_id(run_id, trial_id)
            proxy_config = ProxyInstanceConfig(
                upstream_base_url=(
                    "https://api.anthropic.com" if _sub_mode
                    else model.base_url
                ),
                upstream_api_key=model.api_key,
                upstream_protocol=model.protocol,
                artifact_dir=proxy_artifact_dir,
                trial_id=trial_id,
                system_template=run.proxy.rewrite_system,
                port=0,
                request_timeout=run.proxy.request_timeout,
                http_proxy=run.proxy.upstream_http_proxy,
                extra_headers=_extra_headers,
                upstream_extra_body=dict(model.upstream_extra_body or {}),
                container_log_dir=(
                    _container_trial_proxy_dir(trial_id) if proxy_logs_mounted else ""
                ),
                logs_mounted=proxy_logs_mounted,
                max_requests=_effective_trial_max_rounds(agent, trial.sample, run),
                max_input_tokens=run.execution.max_input_tokens,
                max_output_tokens=run.execution.max_output_tokens,
                max_cost=run.execution.max_cost,
                input_cost_per_1m=model.input_cost_per_1m,
                output_cost_per_1m=model.output_cost_per_1m,
                upstream_max_retries=max(0, int(model.max_retries or 0)),
            )
            proxy = start_container_proxy(container, proxy_config)
            proxy_url = proxy.base_url
            _record_container_proxy_resource(
                storage=storage,
                run_id=run_id,
                agent=agent,
                trial=trial,
                proxy=proxy,
                status="started",
            )

        try:
            # 6. Build env vars and execute agent
            agent_env = agent.agent_type.env_vars(
                proxy_url=proxy_url,
                model=model,
                context_compaction_threshold=agent.context_compaction_threshold,
                container=container,
                home_dir=AGENT_HOME,
                max_rounds=effective_max_rounds,
                workspace_dir=agent.home,
            )

            launch_cmd = agent.agent_type.build_launch_command(
                prompt, model=model, max_rounds=effective_max_rounds,
                proxy_url=proxy_url,
            )
            launch_cmd = _append_session_args(launch_cmd, agent.session_args)

            env_exports = " ".join(f"{k}='{v}'" for k, v in agent_env.items())
            agent_cmd = f"{env_exports} {launch_cmd}" if env_exports else launch_cmd
            full_cmd = f"cd {shlex.quote(workspace_dir)} && {agent_cmd}"
            # Launch agent via exec_async so runtime monitors can stop it on success.
            # Run as the unprivileged ``agent`` user. claude-code's CLI
            # refuses ``--dangerously-skip-permissions`` / ``--permission-mode
            # bypassPermissions`` when uid==0, so we cannot run as root.
            # Raw-socket tools (nmap/fscan/masscan/hping3) are made usable
            # for non-root via file capabilities baked into the Dockerfile
            # (see ``setcap cap_net_raw,cap_net_admin+eip`` block in
            # ``docker/*_pentestenv.Dockerfile``). The agent also has a known
            # sudo password (``cage``) for rare cases that need real root —
            # the system prompt tells it so.
            proc = container.exec_async(
                full_cmd,
                interactive=True,
                user="agent",
            )

            # Start proxy monitor for runtime observability
            proxy_monitor: _ProxyMonitor | None = None
            if proxy:
                proxy_monitor = _ProxyMonitor(
                    container=container,
                    log_dir=proxy.log_dir,
                    trial_id=trial_id,
                    artifact_dir=storage.trial_proxy_dir(trial_id),
                    reporter=reporter,
                    agent_label=agent.label(),
                    process=proc,
                    max_rounds=effective_max_rounds,
                    max_output_tokens=run.execution.max_output_tokens,
                    max_input_tokens=run.execution.max_input_tokens,
                    max_cost=run.execution.max_cost,
                )
                proxy_monitor.start()

            reactive_monitor: ReactiveLiveCheckMonitor | None = None
            poller: CheckDonePoller | None = None
            live_stop_event: threading.Event | None = None
            live_stop_thread: threading.Thread | None = None
            live_check = run.execution.live_check
            trial_dir = storage.trial_dir(trial_id)
            # Shared so polling + reactive emit monotonic poll_index into
            # the merged check_done_polls.jsonl audit log.
            live_call_counter = _CheckDoneCounter()

            if live_check.enabled and live_check.reactive.enabled:
                reactive_monitor = ReactiveLiveCheckMonitor(
                    benchmark=run.benchmark,
                    container=container,
                    sample=trial.sample,
                    trial_dir=trial_dir,
                    trial_id=trial_id,
                    proxy_jsonl=storage.trial_proxy_dir(trial_id) / "proxy.jsonl",
                    live_checks_jsonl=trial_dir / "live_checks.jsonl",
                    check_on_submit=live_check.reactive.check_on_submit,
                    check_on_9091_call=live_check.reactive.check_on_9091_call,
                    call_counter=live_call_counter,
                )
                reactive_monitor.start()

            if live_check.enabled and live_check.polling.enabled:
                poller = CheckDonePoller(
                    benchmark=run.benchmark,
                    container=container,
                    sample=trial.sample,
                    trial_dir=trial_dir,
                    trial_id=trial_id,
                    poll_interval=live_check.polling.interval_seconds,
                    call_counter=live_call_counter,
                    confirm_polls=live_check.polling.confirm_polls,
                )
                poller.start()

            live_stop_monitors: list[tuple[Any, bool]] = []
            if reactive_monitor is not None:
                live_stop_monitors.append(
                    (reactive_monitor.success_event, live_check.stop_on_success)
                )
            if poller is not None:
                live_stop_monitors.append(
                    (poller.success_event, live_check.polling.stop_on_success)
                )
            live_stop_event, live_stop_thread = _start_live_success_stop_thread(
                proc,
                container,
                live_stop_monitors,
            )

            try:
                timed_out = False
                trial_timeout = run.execution.timeout or None  # 0 = unlimited → None
                stdout, stderr = proc.communicate(timeout=trial_timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                # proc.kill() only kills the host-side `docker exec` client, not
                # the agent inside the container; without an in-container kill the
                # agent keeps running (still burning model quota) and the drain
                # below would block indefinitely on its still-open stdout pipe.
                # Reap the in-container agent tree first, then guard the drain so
                # a wedged pipe can never hang the worker thread.
                container.kill_agent()
                proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = "", ""
                exit_code = -1
                logger.warning("Agent execution timed out after %ss", run.execution.timeout)

            # Stop monitors
            if proxy_monitor:
                proxy_monitor.stop()
            if reactive_monitor:
                reactive_monitor.stop()
            if poller:
                poller.stop()
            if live_stop_event is not None:
                live_stop_event.set()
            if live_stop_thread is not None:
                live_stop_thread.join(timeout=5.0)

            # Collect monitor results
            tool_call_count = 0
            terminated_by_limit = False
            terminated_by_max_rounds = bool(
                proxy_monitor is not None and proxy_monitor.terminated_by_max_rounds
            )

            live_success = (
                load_live_success(storage.trial_dir(trial_id))
                or (reactive_monitor.verdict if reactive_monitor else None)
                or (poller.verdict if poller else None)
            )
            terminated_by_live_success = bool(live_success)
            if live_success:
                _mark_canonical_trial_live_evidence_artifact(
                    storage=storage,
                    agent=agent,
                    trial=trial,
                    evidence_path=live_success_path(storage.trial_dir(trial_id)),
                )

            duration_ms = (int(time.time() * 1000) - started) if started else 0

            exec_result = ExecResult(
                command=full_cmd,
                stdout=stdout or "",
                stderr=stderr or "",
                exit_code=exit_code,
                duration_ms=duration_ms,
            )

            logger.info(
                "agent_exec_completed",
                extra={
                    "exit_code": exit_code,
                    "duration_ms": duration_ms,
                    "stdout_len": len(exec_result.stdout),
                    "stderr_len": len(exec_result.stderr),
                    "tool_call_count": tool_call_count,
                    "terminated_by_limit": terminated_by_limit,
                    "terminated_by_max_rounds": terminated_by_max_rounds,
                    "terminated_by_live_success": terminated_by_live_success,
                },
            )

            output = agent.agent_type.parse_output(exec_result)
            exit_code = exec_result.exit_code

        finally:
            if proxy:
                _stop_container_proxy_resource(
                    storage=storage,
                    run_id=run_id,
                    agent=agent,
                    trial=trial,
                    proxy=proxy,
                    artifact_dir=storage.trial_proxy_dir(trial_id),
                )

        final_evidence_path = capture_trial_check_done(
            benchmark=run.benchmark,
            container=container,
            sample=trial.sample,
            trial_dir=storage.trial_dir(trial_id),
            trial_id=trial_id,
        )
        if final_evidence_path is not None:
            _mark_canonical_trial_final_evidence_artifact(
                storage=storage,
                agent=agent,
                trial=trial,
                evidence_path=final_evidence_path,
            )

        # 6.4 Benchmark on_trial_complete hook
        try:
            run.benchmark.on_trial_complete(
                container=container,
                sample=trial.sample,
                trial_dir=str(storage.trial_dir(trial_id)),
            )
        except Exception as exc:
            logger.warning("on_trial_complete error: %s", exc)

        # 6.45 Pull any agent-declared artifact files out of the container
        # (best-effort; e.g. a custom agent's per-node cage_trace.jsonl). The
        # runner stays agent-agnostic — it just collects what the agent declares.
        try:
            declared_artifacts = list(agent.agent_type.artifact_files())
        except Exception:
            declared_artifacts = []
        for container_path, artifact_name in declared_artifacts:
            try:
                dest = storage.trial_dir(trial_id) / artifact_name
                result = container.copy_from(container_path, str(dest))
                if getattr(result, "exit_code", 1) != 0 and dest.exists():
                    dest.unlink()  # docker cp left a partial/empty target
            except Exception as exc:
                logger.debug("artifact %s not collected: %s", artifact_name, exc)

        # 6.5 Generate .traj file from proxy.jsonl
        from cage.proxy.trajectory import generate_traj
        proxy_jsonl = storage.trial_proxy_dir(trial_id) / "proxy.jsonl"
        # trial_id may contain "/" (e.g. nested challenge/variant layout); flatten for filename
        traj_filename = trial_id.replace("/", "_") + ".traj"
        traj_path = storage.trial_dir(trial_id) / traj_filename
        generate_traj(proxy_jsonl, traj_path)
        _mark_canonical_trial_trajectory_artifact(
            storage=storage,
            agent=agent,
            trial=trial,
            traj_path=traj_path,
        )
        if run.execution.store_proxy:
            _mark_canonical_proxy_log_artifact(
                storage=storage,
                agent=agent,
                trial=trial,
            )

        # Clean up proxy.jsonl if store_proxy is False
        if not run.execution.store_proxy and proxy_jsonl.exists():
            proxy_jsonl.unlink()

        # 7. Snapshot post-state + diff
        # Before snapshotting, chown the workspace back to the host user so
        # files docker-cp'd out are readable by the user running `cage`.
        # Even though the agent runs as the unprivileged ``agent`` user
        # (uid 1000 in the container), the host user's uid may not match —
        # and `sudo`-invoked commands inside the container still produce
        # root-owned files. This step normalises everything so
        # ``.cage_runs/.../trials/<id>/state_post/`` is owned by the cage
        # operator's uid:gid and ``cage score`` / manual edits work.
        try:
            container.exec(
                f"chown -R \"$CAGE_HOST_UID:$CAGE_HOST_GID\" "
                f"{shlex.quote(workspace_dir)} 2>/dev/null || true",
                timeout=30.0,
            )
        except Exception as exc:  # best-effort: never block snapshotting
            logger.debug("workspace chown skipped: %s", exc)
        post_snapshot = snapshot_state(
            container,
            state_paths=agent.effective_state_paths,
            home_dir=AGENT_HOME,
            output_dir=storage.trial_state_post_dir(trial_id),
        )
        _mark_canonical_trial_state_artifact(
            storage=storage,
            agent=agent,
            trial=trial,
            state_dir=storage.trial_state_post_dir(trial_id),
            phase="post",
        )
        state_diff = diff_snapshots(pre_snapshot, post_snapshot)

        # 8. Reset agent state if not stateful
        if not agent.stateful:
            reset_state(
                container,
                state_paths=agent.effective_state_paths,
                home_dir=AGENT_HOME,
            )

        # Structural interrupt detection: SIGINT handler force-removed the
        # container while the agent was still running. Two independent signals
        # (either is sufficient) — process-level stop flag, or post-state
        # snapshot failed because the container was already gone (``docker cp``
        # returns ``RWLayer of container ... is unexpectedly nil`` / ``No such
        # container``). Either signal overrides ``exit_code`` in the classifier,
        # because ``docker exec`` returncode is unreliable when the underlying
        # container is force-removed mid-stream (we've observed it return 0
        # under some Docker versions, which would otherwise be mis-classified
        # as a clean ``completed``).
        snapshot_failed = bool(post_snapshot.has_failures)
        run_stopped = scheduler.is_stopped()
        interrupted = run_stopped or snapshot_failed

        ended = int(time.time() * 1000)
        timing = Timing(started_at_ms=started, ended_at_ms=ended, duration_ms=ended - started)
        termination_meta = _trial_termination_metadata(
            exit_code=exit_code,
            timed_out=timed_out,
            terminated_by_limit=terminated_by_limit,
            terminated_by_max_rounds=terminated_by_max_rounds,
            error="",
            timeout_seconds=run.execution.timeout,
            output=output,
            proxy_jsonl_path=storage.trial_proxy_dir(trial_id) / "proxy.jsonl",
            max_rounds=effective_max_rounds,
            max_input_tokens=run.execution.max_input_tokens,
            max_output_tokens=run.execution.max_output_tokens,
            max_cost=run.execution.max_cost,
            interrupted=interrupted,
        )
        if terminated_by_live_success:
            termination_meta.update(
                {
                    "status": TrialStatus.COMPLETED.value,
                    "termination_reason": "live_success",
                    "termination_detail": "Stopped after a successful live-check verdict.",
                }
            )

        storage.save_trial_meta(trial_id, {
            "trial_id": trial_id,
            "trial_index": trial.index,
            "trial_type": trial.type.value,
            "sample_id": trial.sample_id,
            "exit_code": exit_code,
            "max_rounds": effective_max_rounds,
            "snapshot_failed": snapshot_failed,
            **termination_meta,
            "live_success": bool(live_success),
            "live_success_verdict": live_success or {},
            "terminated_by_live_success": terminated_by_live_success,
            "terminated_by_max_rounds": terminated_by_max_rounds,
            "timing": {
                "started_at_ms": started,
                "ended_at_ms": ended,
                "duration_ms": ended - started,
            },
            "state_diff": {
                "summary": state_diff.summary(),
                "has_changes": state_diff.has_changes,
            },
        })

        storage.save_trial_output(trial_id, {
            "output": output,
            "exit_code": exit_code,
            "sample": trial.sample,
        })
        _mark_canonical_trial_output_artifact(
            storage=storage,
            agent=agent,
            trial=trial,
        )

        trial.status = TrialStatus(termination_meta.get("status", TrialStatus.COMPLETED.value))
        trial.output = output
        trial.exit_code = exit_code
        trial.timing = timing

        result = TrialResult(
            trial_id=trial_id,
            trial_index=trial.index,
            trial_type=trial.type.value,
            sample_id=trial.sample_id,
            output=output,
            exit_code=exit_code,
            timing=timing,
            proxy_log=storage.trial_proxy_dir(trial_id) / "proxy.jsonl",
            state_pre=storage.trial_state_pre_dir(trial_id),
            state_post=storage.trial_state_post_dir(trial_id),
            metadata={
                "state_diff": state_diff.summary(),
                "sample": trial.sample,
                "live_success": live_success or {},
                "terminated_by_live_success": terminated_by_live_success,
                **termination_meta,
            },
            tool_call_count=tool_call_count,
            terminated_by_limit=terminated_by_limit,
        )

        # Inline per-trial scoring. Runs before we return so dashboards /
        # the web inspector can see scores as soon as each trial completes.
        # Skipped for ``strategy="post_run"`` scorers — those run in
        # ``_score_trials`` at end-of-agent with the full cohort visible.
        # Failures are swallowed (logged in _score_one_trial); the end-of-
        # agent pass is the recovery path.
        if getattr(run.benchmark.scorer(), "strategy", "per_trial") == "per_trial":
            _score_one_trial(result, run.benchmark, storage)

        return result

    except Exception as exc:
        if scheduler.is_stopped():
            # Residual race: the stop's teardown landed mid-setup (e.g. the
            # container was force-removed → "No such container" on the next
            # container op). The agent never ran → interrupted, not a genuine
            # trial_error. Mirrors _target_launch_failure_termination.
            return _trial_setup_interrupted_result(
                started=started, storage=storage, trial=trial, trial_id=trial_id,
            )
        return _trial_failure_result(
            exc=exc,
            run=run,
            started=started,
            storage=storage,
            trial=trial,
            trial_id=trial_id,
        )

    finally:
        _cleanup_trial_resources(
            agent=agent,
            agent_isolation=agent_isolation,
            attached_network=attached_network,
            challenge_client=challenge_client,
            container=container,
            run_id=run_id,
            storage=storage,
            submit_handle=submit_handle,
            target_data=target_data,
            trial=trial,
            trial_id=trial_id,
        )

def _trial_termination_metadata(
    *,
    exit_code: int,
    timed_out: bool,
    terminated_by_limit: bool,
    terminated_by_max_rounds: bool = False,
    error: str,
    timeout_seconds: int | float,
    output: str = "",
    proxy_jsonl_path: Path | None = None,
    max_rounds: int = 0,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_cost: float | None = None,
    interrupted: bool = False,
) -> dict[str, str]:
    """Return stable trial termination metadata for dashboards and web UI.

    ``proxy_jsonl_path`` lets the classifier inspect the trial's upstream
    error history — distinguishing quota/rate-limit/5xx/auth from a generic
    ``agent_exit_nonzero``. Callers without a proxy log (early failures,
    target launch errors) can omit it; classification still works on
    output text alone.

    ``max_rounds`` is the effective per-trial round budget. When the
    proxy enforced it (recorded request count ≥ max_rounds), the
    classifier returns ``max_rounds_reached`` instead of mis-attributing
    the failure to any unrelated transient upstream error earlier in
    the run.

    ``interrupted`` is the structural Ctrl+C signal — caller sets it when
    the run scheduler's stop event is set and/or ``StateSnapshot.has_failures`` (the
    SIGINT handler force-removed the container before snapshot). The
    classifier then returns ``user_interrupted`` regardless of what
    ``docker exec`` reported as exit_code, because docker exec returncode
    is unreliable when the container is force-removed mid-stream.
    """
    return classify_trial_termination(
        exit_code=exit_code,
        timed_out=timed_out,
        terminated_by_limit=terminated_by_limit,
        terminated_by_max_rounds=terminated_by_max_rounds,
        error=error,
        timeout_seconds=timeout_seconds,
        output=output,
        proxy_jsonl_path=proxy_jsonl_path,
        max_rounds=max_rounds,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_cost=max_cost,
        interrupted=interrupted,
    ).to_metadata()

def _looks_like_model_timeout(text: str) -> bool:
    return looks_like_model_timeout(text)
