"""Experiment conductor — the trial-lifecycle entry point.

``run_experiment`` is the single entry for executing an experiment. It owns only
the conductor responsibilities: resolving the run id, installing the run-scoped
:class:`~cage.experiment.engine.scheduler.RunScheduler` and
:class:`~cage.experiment.engine.run_cleanup.RunCleanup`, fanning trials out across agents
(serial or parallel), aggregating results, and driving scoring + dashboards.
Everything else lives in collaborators: a single trial runs through
:func:`cage.experiment.engine.trial_runner.run_trial_isolated`; resource/record/dashboard
writing lives under ``cage.artifacts``; the proxy, target, and
scheduling primitives live under ``cage.sandbox``/``cage.proxy``/``cage.target``.

This module replaced the legacy ``cage/orchestrator.py`` god-module.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from cage.agents.base import AgentInstance
from cage.artifacts.canonical_marks import (
    _mark_canonical_experiment_records_finished,
    _mark_canonical_run_scored,
    finalize_canonical_running_trials,
    _mark_canonical_run_summary_artifact,
    _mark_canonical_trial_finished,
    _mark_canonical_trial_scoring_for_results,
    _reset_canonical_trial_for_rerun,
    _save_canonical_experiment_snapshot,
    _save_run_metadata_snapshot,
)
from cage.artifacts.jsonio import _load_json_file, _write_json_file
from cage.artifacts.record_snapshots import (
    _build_run_manifest,
    _result_terminal_status,
    _save_planned_trials,
    _save_run_manifest,
)
from cage.artifacts.run_storage import RunStorage, create_run_id, validate_run_id
from cage.contracts.logging import (
    add_file_handlers,
    bind_run_context,
    clear_all_context,
    quiet_console_logging,
    remove_file_handlers,
)
from cage.contracts.reporter import ReporterFactory
from cage.experiment.engine.hooks import (
    HookContext,
    default_trial_sequence,
    expand_trials_for_passk,
)
from cage.experiment.engine.reporting import _write_dashboards
from cage.experiment.engine.resume import (
    ResumeCompatibilityError,
    _AgentResumePlan,
    _archive_trial_dir_before_resume,
    _assert_resume_compatible,
    _cage_runs_root,
    _cap_trials_for_execution,
    _format_resume_decision_breakdown,
    _resolve_retry_reasons,
    _resume_decisions,
    _resume_keep_if_summary,
    _resume_should_preserve_canonical_snapshot,
    _split_resume_decisions,
    analyze_resume_plan,
)
from cage.experiment.engine.run_cleanup import RunCleanup
from cage.experiment.engine.run_context import ExperimentRun
from cage.experiment.engine.scheduler import RunScheduler
from cage.experiment.engine.trial_runner import run_trial_isolated
from cage.experiment.model import Trial, TrialResult
from cage.sandbox.admission import HostMemoryGate
from cage.sandbox.naming import _parse_agent_label
from cage.scoring.lifecycle import _build_summary, _score_trials
from cage.target.provisioning import (
    discover_benchmark_root,
    spawn_embedded_target_server,
    target_server_timeout_env,
)

logger = logging.getLogger(__name__)

# Sentinel pool/semaphore size standing in for "unlimited" global concurrency
# (``runtime.max_trials_global`` <= 0). Large enough never to be the real
# bottleneck; the per-agent cap and host resources do the actual throttling.
_UNLIMITED_CONCURRENCY = 100_000


class _TrialCancelled(Exception):
    """A queued trial was skipped because a graceful stop (Ctrl+C) was requested."""


def _archive_existing_run_dirs_for_force(existing_dirs: list[Path], run_id: str) -> None:
    archived_at = time.strftime("%Y%m%dT%H%M%S")
    for run_dir in existing_dirs:
        archive = _next_force_archive_path(run_dir, archived_at)
        run_dir.rename(archive)
        archived_run_id = archive.name
        _tag_force_archived_dashboard(archive, run_id, archived_run_id)
        _write_json_file(
            archive / "force_archive.json",
            {
                "archived_from_run_id": run_id,
                "archived_at": archived_at,
                "archive_reason": "force_run_id_reuse",
            },
        )
        logger.debug("force: archived existing run %s -> %s", run_dir, archive)

def _tag_force_archived_dashboard(
    archive: Path,
    original_run_id: str,
    archived_run_id: str,
) -> None:
    dashboard_path = archive / "dashboard.json"
    if not dashboard_path.exists():
        return
    dashboard = _load_json_file(dashboard_path)
    if not isinstance(dashboard, dict):
        return
    dashboard["run_id"] = archived_run_id
    dashboard["archived_from_run_id"] = original_run_id
    dashboard["archive_reason"] = "force_run_id_reuse"
    _write_json_file(dashboard_path, dashboard)

def _next_force_archive_path(run_dir: Path, archived_at: str) -> Path:
    base = run_dir.with_name(f"{run_dir.name}.previous_{archived_at}")
    if not base.exists():
        return base
    suffix = 2
    while True:
        candidate = run_dir.with_name(f"{base.name}_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1

def _resolve_run_id(run: ExperimentRun, cage_runs: Path) -> str:
    """Pick the run id for this experiment and verify it's not already in use.

    Resolution order: ``run.run_id`` (from ``--run-id`` or ``project.run_id``)
    wins; otherwise we auto-generate one. The id is validated against the
    user-facing format contract.

    When ``run.resume`` is true a pre-existing run directory is *required*
    (something to resume from); otherwise a pre-existing directory is an error
    so we don't silently overwrite results from a previous run.
    """
    raw = (run.run_id or "").strip()
    if raw:
        run_id = validate_run_id(raw)
        user_supplied = True
    else:
        run_id = create_run_id()
        user_supplied = False

    existing_dirs: list[Path] = []
    if cage_runs.is_dir():
        for agent_dir in cage_runs.iterdir():
            if not agent_dir.is_dir():
                continue
            if (agent_dir / run_id).exists():
                existing_dirs.append(agent_dir / run_id)

    if run.resume and getattr(run, "force", False):
        raise ValueError("--force and --resume are mutually exclusive")

    if run.resume:
        if not existing_dirs:
            raise ValueError(
                f"--resume requested but no existing run directory found "
                f"for run_id {run_id!r} under {cage_runs}."
            )
        logger.info(
            "resume: reusing %d existing run dir(s) for run_id=%s",
            len(existing_dirs), run_id,
        )
        return run_id

    if existing_dirs:
        if getattr(run, "force", False):
            _archive_existing_run_dirs_for_force(existing_dirs, run_id)
            return run_id
        if user_supplied:
            raise ValueError(
                f"run_id {run_id!r} already exists under "
                f"{existing_dirs[0].parent}. Pick a different --run-id, or "
                f"pass --resume to continue the existing run, or pass --force "
                f"to archive it and start over."
            )
        # Auto-generated collision is astronomically unlikely; if it
        # ever happens, bail loudly rather than silently overwrite.
        raise RuntimeError(
            f"Auto-generated run_id {run_id!r} collides with "
            f"{existing_dirs[0]}. Retry."
        )
    return run_id

def run_experiment(
    run: ExperimentRun,
    *,
    make_reporter: ReporterFactory | None = None,
) -> dict[str, Any]:
    """Run a complete experiment. Returns summary dict.

    Handles KeyboardInterrupt (Ctrl+C) gracefully: cancels pending work,
    collects results from completed trials, and writes a partial
    dashboard.json before exiting.

    ``make_reporter`` is the optional reporter factory the CLI injects to render
    terminal progress; the conductor itself is headless, so when it is ``None``
    the run produces no progress output. It receives the run's display
    parameters and returns a :class:`cage.contracts.reporter.Reporter`.
    """
    from cage.experiment.engine.preflight import run_preflight

    cage_runs = _cage_runs_root(run)
    run_id = _resolve_run_id(run, cage_runs)
    resume_plans: list[_AgentResumePlan] = []
    if run.resume:
        # Pure read-only guard before target_server/preflight side effects.
        resume_plans = analyze_resume_plan(run)
    all_results: dict[str, list[TrialResult]] = {}
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    interrupted = False
    fatal_resume_error: ResumeCompatibilityError | None = None

    bind_run_context(run_id=run_id)
    logger.debug("Starting experiment: %s (run_id=%s)", run.name, run_id)
    logger.debug("Agents: %s", ", ".join(a.label() for a in run.agents))
    if run.admission.enabled:
        logger.debug(
            "Admission gate enabled: pause>%.0f%%, resume<=%.0f%%, poll=%.1fs",
            run.admission.memory_pause_at * 100,
            run.admission.memory_resume_at * 100,
            run.admission.poll_seconds,
        )

    # Install the per-run scheduler (admission gate + stop event + two-level
    # concurrency gate) for thread workers to read.
    #
    # Two-level concurrency gate: global cap on in-flight trials (sized to
    # ``runtime.max_trials_global``) + per-agent cap on simultaneous in-flight trials
    # (``AgentInstance.max_concurrent``; 0/unset → unbounded, keyed by
    # ``id(agent)`` so two agents sharing a model id keep independent quotas).
    # max_trials_global 0/unset = unlimited -> effectively no global gate.
    _gc = int(run.execution.max_trials_global or 0)
    _trial_cap = _gc if _gc > 0 else _UNLIMITED_CONCURRENCY
    _agent_caps = {
        id(_agent): int(getattr(_agent, "max_concurrent", 0) or 0) or 1024
        for _agent in run.agents
    }
    _target_setup_cap = max(0, int(getattr(run.execution, "max_target_setups", 1) or 0))
    logger.debug(
        "concurrency: global cap=%d trials, target setup cap=%s, per-agent caps=%s",
        _trial_cap,
        _target_setup_cap or "∞",
        {
            a.label(): int(getattr(a, "max_concurrent", 0) or 0) or "∞"
            for a in run.agents
        },
    )
    scheduler = RunScheduler(
        trial_cap=_trial_cap,
        agent_caps=_agent_caps,
        target_setup_cap=_target_setup_cap,
        admission=HostMemoryGate(run.admission),
    )
    cleanup = RunCleanup(run_id, scheduler)
    cleanup.install_signal_handlers()
    # The loaded configuration IS the live run context: bind the per-invocation
    # collaborators onto it so one object threads through trial execution.
    run.run_id = run_id
    run.cage_runs = cage_runs
    run.scheduler = scheduler
    run.cleanup = cleanup
    # On a forced teardown (2nd Ctrl+C / SIGTERM), persist interrupted status
    # for trials killed mid-flight so their record never stays stuck at
    # "running". Bound now (before any trial starts) so a signal at any point is
    # covered; the sweep is idempotent and a no-op on the graceful path.
    cleanup.finalize_running_trials_hook = lambda: finalize_canonical_running_trials(
        cage_runs, run_id=run_id, completed_at=time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    # target_server lifecycle: ALWAYS embedded and bound to this run. The
    # orchestrator spawns a subprocess on a free localhost port and sets
    # ``run.target.server_url`` programmatically; users cannot point cage
    # at an external/shared hub via project.yml. Tear-down on run exit is
    # handled by ``RunCleanup.teardown_all`` + ``atexit``.
    if run.target.enabled:
        bench_root = discover_benchmark_root(run.benchmark)
        embedded_log = cage_runs / f"target_server-{run_id}.log"
        try:
            embedded_hub = spawn_embedded_target_server(
                run_id=run_id, benchmark_root=bench_root,
                log_path=embedded_log,
                extra_env=target_server_timeout_env(
                    run.target,
                    benchmark_id=str(run.metadata.get("benchmark_id") or ""),
                ),
            )
            run.cleanup.embedded_hub = embedded_hub
            run.target.server_url = embedded_hub.server_url
            logger.debug(
                "embedded target_server ready: %s", embedded_hub.server_url,
            )
        except Exception as exc:
            logger.error("embedded target_server failed to start: %s", exc)
            raise

    # Pre-flight checks
    bench_limit = run.sample_limit
    bench_sample_ids = run.sample_ids
    preflight_samples = list(
        run.benchmark.iter_samples_limited(bench_limit, bench_sample_ids, run.sample_slice)
    )
    if bench_sample_ids and not preflight_samples:
        raise RuntimeError(
            f"--sample {bench_sample_ids} matched no samples in benchmark "
            f"{run.benchmark.name!r}; check the IDs."
        )
    try:
        preflight_result = run_preflight(run, samples=preflight_samples)
    except Exception as e:
        logger.error("Preflight failed: %s", e)
        raise
    # Stash for per-agent storage to dump as <run-dir>/preflight.json — keeps
    # the diagnosis trail visible in the inspector ("preflight said PASS but
    # trials still failed" vs "preflight FAIL'd, here is which agent").
    run.metadata["_preflight_result"] = preflight_result.to_dict()

    # Host-side per-run services declared by agents (e.g. Claude Code's OAuth
    # token refresher). Started once preflight has confirmed the run is viable;
    # torn down on exit by ``RunCleanup.stop_host_services`` on the happy path
    # and by ``RunCleanup.teardown_all`` on SIGTERM/SIGINT/atexit.
    run.cleanup.start_host_services(run, cage_runs)

    # Outer pool fans out ALL agents in parallel; trial-level throttling now
    # happens via the global trial Semaphore inside run_trial_isolated, so
    # max_trials_global does not gate the outer pool any more.
    agent_pool_size = max(1, len(run.agents))
    planned_trials_per_agent = (
        len(preflight_samples) * max(1, int(run.execution.passk or 1))
    )
    estimated_trials_per_agent = planned_trials_per_agent
    if run.execution.max_trial is not None and run.execution.max_trial >= 0:
        estimated_trials_per_agent = min(
            estimated_trials_per_agent,
            run.execution.max_trial,
        )
    planned_total_trials = planned_trials_per_agent * len(run.agents)
    estimated_total_trials = estimated_trials_per_agent * len(run.agents)
    contract_runnable_trials = estimated_total_trials
    resume_replayed_trials = 0
    if resume_plans:
        contract_runnable_trials = sum(len(plan.rerun) for plan in resume_plans)
        resume_replayed_trials = sum(
            len(plan.replay) + len(plan.capped) for plan in resume_plans
        )

    primary_run_dir = (
        cage_runs / run.agents[0].label() / run_id
        if run.agents
        else cage_runs / run_id
    )
    inspect_mode = str(getattr(run.logging, "inspect_mode", "on") or "on").lower()
    board_url = ""
    run_url = ""
    dashboard_url = ""
    view_links: list[Any] = []
    if inspect_mode != "off":
        from cage.config import load_repo_config
        from cage.web.inspect_board import ensure_inspector_board, run_url_variants

        inspect_root = Path(run.benchmark_dir or run.project_file.parent).resolve()
        web_config = load_repo_config(Path.cwd()).web_inspector
        if inspect_mode == "on" and str(getattr(web_config, "host", "")).lower() in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            web_config = replace(web_config, host="0.0.0.0")
        try:
            board_info = ensure_inspector_board(
                inspect_root,
                web_config,
                mode=inspect_mode,
                interactive=bool(getattr(sys.stderr, "isatty", lambda: False)()),
            )
        except ValueError:
            if inspect_mode == "on":
                raise
            logger.warning("managed inspector board disabled by unsafe config", exc_info=True)
        else:
            if getattr(board_info, "enabled", False) and getattr(board_info, "url", ""):
                board_url = board_info.url
                view_links = run_url_variants(board_info, inspect_root, primary_run_dir)
                if view_links:
                    run_url = view_links[0].run_url
                    dashboard_url = view_links[0].dashboard_url

    run_reporter = (
        make_reporter(
            config=run,
            run_id=run_id,
            run_dir=primary_run_dir,
            enabled=bool(getattr(run.logging, "terminal_ui", True)),
            total_trials=estimated_total_trials,
            planned_trials=planned_total_trials,
            runnable_trials=contract_runnable_trials,
            resume_replayed_trials=resume_replayed_trials,
            samples=preflight_samples,
            board_url=board_url,
            run_url=run_url,
            dashboard_url=dashboard_url,
            view_links=view_links,
        )
        if make_reporter is not None
        else None
    )
    run.reporter = run_reporter
    # Forced exit (Ctrl+C×2 / SIGTERM) ``os._exit``s from the signal handler and
    # never reaches the end-of-run results table; let it print the reporter's
    # final status line first so the user always sees the final situation.
    if run_reporter is not None and hasattr(run_reporter, "print_interrupt_banner"):
        cleanup.final_summary_hook = run_reporter.print_interrupt_banner
    if run_reporter is not None and hasattr(run_reporter, "print_graceful_stop_notice"):
        cleanup.first_interrupt_hook = run_reporter.print_graceful_stop_notice
    quiet_terminal_output = bool(getattr(run_reporter, "enabled", False))
    run_contract = getattr(run_reporter, "contract", None)
    if run_contract is not None and not quiet_terminal_output:
        print(run_contract.to_plain_text(), file=sys.stderr)

    future_to_label: dict[Any, str] = {}

    def run_agent_pool() -> None:
        nonlocal fatal_resume_error, future_to_label
        with quiet_console_logging(quiet_terminal_output):
            with ThreadPoolExecutor(max_workers=agent_pool_size) as pool:
                future_to_label = {}
                for agent in run.agents:
                    future = pool.submit(_run_single_agent, run, agent)
                    future_to_label[future] = agent.label()

                for future in as_completed(future_to_label):
                    label = future_to_label[future]
                    try:
                        results = future.result()
                    except ResumeCompatibilityError as exc:
                        logger.error("Agent %s resume compatibility failed: %s", label, exc)
                        fatal_resume_error = exc
                        run.scheduler.request_stop()
                        for pending in future_to_label:
                            if pending is not future:
                                pending.cancel()
                        break
                    except Exception as exc:
                        logger.error("Agent %s failed: %s", label, exc)
                        results = []
                    all_results[label] = results

    try:
        reporter_cm = (
            run_reporter.live()
            if hasattr(run_reporter, "live")
            else contextlib.nullcontext(run_reporter)
        )
        with reporter_cm:
            run_agent_pool()

    except KeyboardInterrupt:
        interrupted = True
        with quiet_console_logging(quiet_terminal_output):
            # Signal any threads currently blocked in the admission gate to bail.
            run.scheduler.request_stop()
            logger.warning("Experiment interrupted by user (Ctrl+C). Saving partial results...")
            # Cancel pending futures and collect whatever completed
            for future in future_to_label:
                future.cancel()
            for future, label in future_to_label.items():
                if label in all_results:
                    continue  # already collected
                if future.done() and not future.cancelled():
                    try:
                        all_results[label] = future.result(timeout=0)
                    except Exception:
                        all_results[label] = []
                else:
                    all_results[label] = []

    with quiet_console_logging(quiet_terminal_output):
        try:
            run.benchmark.teardown()
        except Exception as exc:
            logger.warning("Benchmark teardown error: %s", exc)
        # Stop the embedded target_server *after* benchmark.teardown (which may
        # talk to the server through its ChallengeClient). Signal-handler
        # teardown also stops the hub, but doing it here covers the happy
        # path and any non-signal exception.
        embedded_hub = run.cleanup.embedded_hub
        if embedded_hub is not None:
            try:
                embedded_hub.stop()
            except Exception as exc:
                logger.warning("embedded target_server stop error: %s", exc)
            run.cleanup.embedded_hub = None
        # Stop host-side per-run services (e.g. the OAuth refresher) on the happy
        # path; the signal/atexit teardown covers Ctrl+C and SIGTERM.
        run.cleanup.stop_host_services()
        clear_all_context()

        if fatal_resume_error is not None:
            raise fatal_resume_error

        completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Write dashboards — always, even on interrupt
        agent_artifacts = _write_dashboards(
            run, all_results, cage_runs, run_id,
            started_at, completed_at, interrupted=interrupted,
        )
        run_status = "interrupted" if interrupted else "completed"
        _mark_canonical_experiment_records_finished(
            cage_runs,
            agent_labels=all_results.keys(),
            run_id=run_id,
            status=run_status,
            completed_at=completed_at,
        )

        ordered_labels = list(all_results.keys())
        primary_label = ordered_labels[0] if ordered_labels else ""
        primary_artifacts = agent_artifacts.get(primary_label, {})
        run_dir = Path(primary_artifacts.get("run_dir", str(cage_runs / run_id)))
        dashboard_path = Path(
            primary_artifacts.get("dashboard_path", str(run_dir / "dashboard.json"))
        )

        global_summary: dict[str, Any] = {
            "run_id": run_id,
            "experiment": run.name,
            "run_dir": str(run_dir),
            "dashboard_path": str(dashboard_path),
            "status": run_status,
            # Managed inspector board URLs (same ones shown in the launch banner).
            # The board is a detached subprocess that outlives this run, so the
            # CLI re-surfaces these at the end as a "open the run view" reminder.
            "board_url": board_url,
            "run_url": run_url,
            "dashboard_url": dashboard_url,
            "view_links": [
                {
                    "label": str(getattr(link, "label", "") or ""),
                    "base_url": str(getattr(link, "base_url", "") or ""),
                    "run_url": str(getattr(link, "run_url", "") or ""),
                    "dashboard_url": str(getattr(link, "dashboard_url", "") or ""),
                }
                for link in view_links
            ],
            "agents": {
                label: {
                    **_build_summary(all_results.get(label, [])),
                    **agent_artifacts.get(label, {}),
                }
                for label in ordered_labels
            },
        }
        logger.info(
            "Experiment %s: %s",
            "interrupted" if interrupted else "complete",
            run_id,
        )
    return global_summary

def _run_agent_trials_parallel(
    run: ExperimentRun,
    agent: AgentInstance,
    storage: RunStorage,
    trials: list[Trial],
    hook_ctx: HookContext,
    max_workers: int,
    *,
    passk: int = 1,
    reporter: Any | None = None,
) -> list[TrialResult]:
    """Run all trials for a single agent with trial-level parallelism.

    Each trial runs in its own container with its own target stack (if enabled).
    max_workers=1 means serial execution, but still uses one fresh container
    and one fresh target stack per trial.

    Scheduling: every trial is submitted to a single ThreadPoolExecutor so the
    worker pool is always full whenever there is pending work. Trials arrive
    here pre-ordered as ``[pass1_sample1, pass1_sample2, ..., pass2_sample1, ...]``
    (see the pass@k expansion in ``run_experiment``), and ThreadPoolExecutor
    pulls from that queue in FIFO order — so pass 1 trials start before pass 2
    trials, but pass N stragglers never block pass N+1 from filling worker
    slots. The previous design grouped trials into pass-batches with a hard
    barrier between batches, which starved workers whenever any single trial
    in a batch was slow.
    """
    cage_runs = storage.run_dir.parent.parent  # .cage_runs/
    run_id = storage.run_dir.name
    passk = max(1, passk)

    logger.info(
        "Agent %s: queuing %d trials onto %d workers (passes=%d, FIFO by pass-then-sample)",
        agent.label(), len(trials), max_workers, passk,
    )

    all_results: list[TrialResult] = []

    def _run_one_trial_with_reporting(trial: Trial) -> TrialResult:
        agent_label = agent.label()
        # Graceful stop (1st Ctrl+C): trials already in flight keep running, but
        # ones that haven't started yet are cancelled here — before any container
        # or agent is launched — so the run drains instead of starting new work.
        if run.scheduler is not None and run.scheduler.is_stopped():
            raise _TrialCancelled(trial.id)
        # NOTE: the canonical "running" mark is NOT here. It fires inside
        # run_trial_isolated AFTER the admission + concurrency gates (co-located
        # with the live-progress ``trial_started``), so the durable record.json
        # flips planned→running only for trials genuinely in flight. Marking it at
        # pool admission made every trial queued behind the concurrency gate look
        # "running" on disk while the CLI correctly counted only the N in-flight.
        try:
            result = run_trial_isolated(
                run, agent, trial, cage_runs, run_id, reporter=reporter,
                scheduler=run.scheduler, cleanup=run.cleanup,
            )
        except Exception:
            if reporter is not None:
                reporter.trial_finished(
                    agent_label=agent_label,
                    trial_id=trial.id,
                    status="failed",
                    duration_ms=None,
                    exit_code=-1,
                )
            raise
        _mark_canonical_trial_finished(
            storage=storage,
            agent=agent,
            trial=trial,
            result=result,
        )
        if reporter is not None:
            reporter.trial_finished(
                agent_label=agent_label,
                trial_id=result.trial_id,
                status=_result_terminal_status(result),
                duration_ms=(
                    result.timing.duration_ms if result.timing is not None else None
                ),
                exit_code=result.exit_code,
            )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_index = {}
        for trial in trials:
            future = pool.submit(
                _run_one_trial_with_reporting, trial,
            )
            future_to_index[future] = trial.index

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                all_results.append(result)
            except _TrialCancelled:
                # Trial never started (graceful stop requested) — not a failure.
                continue
            except Exception as exc:
                logger.error(
                    "Trial %d for agent %s failed: %s", idx, agent.label(), exc,
                )

    all_results.sort(key=lambda r: r.trial_index)
    return all_results

def _run_single_agent(run: ExperimentRun, agent: AgentInstance) -> list[TrialResult]:
    """Run all trials for a single agent instance. Thread-safe for parallel execution."""
    cage_runs = run.cage_runs
    run_id = run.run_id
    reporter = run.reporter
    agent_label = agent.label()
    bind_run_context(agent_label=agent_label)
    logger.info("--- Agent: %s ---", agent_label)

    # Directory layout: .cage_runs/{label}/run-{timestamp}/
    agent_dir_name, _mode = _parse_agent_label(agent_label)
    run_root = cage_runs / agent_dir_name / run_id

    storage = RunStorage(run_root, agent_label=agent.label())

    # Add file logging for this agent run
    debug_path = storage.debug_log_path() if run.logging.debug_file_enabled else None
    add_file_handlers(
        run_log_path=storage.log_file_path(),
        debug_log_path=debug_path,
        file_level=run.logging.file_level,
    )

    try:
        # Build trial sequence
        bench_limit = run.sample_limit
        bench_sample_ids = run.sample_ids
        samples = list(
            run.benchmark.iter_samples_limited(bench_limit, bench_sample_ids, run.sample_slice)
        )
        logger.info("Benchmark: %s (%d samples)", run.benchmark.name, len(samples))

        hook_ctx = HookContext(
            experiment_config={"name": run.name},
            samples=samples,
            trials_completed=[],
            trials_pending=[],
            run_artifacts_dir=str(storage.run_dir),
        )
        if run.hooks.pre_run:
            trial_results = run.hooks.fire("pre_run", hook_ctx)
            trials = trial_results[0] if trial_results else default_trial_sequence(hook_ctx)
        else:
            trials = default_trial_sequence(hook_ctx)

        # pass@k: replay the full trial sequence k times in
        # [pass1_all_samples, pass2_all_samples, ...] order. This order is
        # also the worker-pool admission order under parallelism (single
        # FIFO ThreadPool — see _run_agent_trials_parallel), so pass 1
        # trials get worker slots before pass 2 trials, but a slow pass-1
        # straggler never blocks pass-2 work from filling idle slots.
        passk = max(1, run.execution.passk)
        if passk > 1:
            base_count = len(trials)
            trials = expand_trials_for_passk(trials, passk)
            logger.info(
                "pass@k enabled: %d passes × %d samples = %d trials",
                passk, base_count, len(trials),
            )
        for index, trial in enumerate(trials):
            trial.index = index

        run_manifest = _build_run_manifest(run, agent, trials)
        if run.resume:
            _assert_resume_compatible(storage.run_dir, run_manifest)

        if not _resume_should_preserve_canonical_snapshot(storage, run.resume):
            _save_canonical_experiment_snapshot(run, agent, storage, trials, run_id)
        _save_run_metadata_snapshot(run, agent, storage)
        _save_planned_trials(storage, trials)
        _save_run_manifest(storage, run_manifest)

        # Execution cap (``--max-trial`` / runtime.max_trial): keep the full
        # plan recorded above, but only act on trials below the index cap this
        # invocation. The rest stay pending for a later capless run.
        trials_exec = _cap_trials_for_execution(trials, run.execution.max_trial)
        if reporter is not None:
            reporter.agent_started(agent_label, len(trials_exec))
        if len(trials_exec) != len(trials):
            logger.info(
                "max_trial=%s: executing %d of %d planned trials this "
                "invocation (global index < %s); the rest stay pending.",
                run.execution.max_trial, len(trials_exec), len(trials),
                run.execution.max_trial,
            )

        # Resume: replay results for trials whose meta.json is already on disk
        # (and not flagged for retry by reason); the executor only sees the
        # remainder. See ``_partition_resumed_trials`` for the policy. For
        # trials that DO get re-run, archive the previous attempt's directory
        # so its proxy.jsonl / state snapshots don't bleed into the new run.
        if run.resume:
            # Classify once, then split — the same helper the dry-run preview
            # uses, so a real resume re-runs exactly what --dry-run promised.
            decisions = _resume_decisions(
                storage, trials_exec,
                _resolve_retry_reasons(run.resume_retry_reasons),
                run.resume_max_attempts,
                run.resume_keep_if,
                run.resume_select_id_pattern,
            )
            replayed_results, pending_trials, capped_trials = (
                _split_resume_decisions(storage, decisions)
            )
            for t, n in capped_trials:
                logger.info(
                    "resume: %s exhausted retry budget (%d attempts, cap=%d); "
                    "replaying last failed result.",
                    t.id, n, run.resume_max_attempts,
                )
            archived = 0
            for t in pending_trials:
                if _archive_trial_dir_before_resume(storage, t) is not None:
                    archived += 1
                    # The archive carried this trial's co-located canonical
                    # record away; recreate a live planned record so
                    # experiment_record.json stays resolvable and the re-run's
                    # marks have a record to update.
                    _reset_canonical_trial_for_rerun(storage, agent, t)
            logger.info(
                "resume: replaying %d trial(s) from disk; %d to re-run "
                "(archived %d previous trial dir(s); retry reasons: default + %s; "
                "max_attempts=%s; select=%s; keep_if=%s)",
                len(replayed_results), len(pending_trials), archived,
                list(run.resume_retry_reasons) or "[]",
                (
                    run.resume_max_attempts
                    if run.resume_max_attempts > 0
                    else "unlimited"
                ),
                run.resume_select_id_pattern or "(all)",
                _resume_keep_if_summary(run.resume_keep_if),
            )
            # Per-decision breakdown so operators see *why* each trial was
            # kept vs re-run (the salvage labels are the interesting bit).
            for line in _format_resume_decision_breakdown(decisions):
                logger.info("resume:   %s", line)
            if reporter is not None:
                for replayed in replayed_results:
                    reporter.trial_replayed(
                        agent_label=agent_label,
                        trial_id=replayed.trial_id,
                        sample_id=replayed.sample_id,
                        trial_index=replayed.trial_index,
                        status=_result_terminal_status(replayed),
                    )
        else:
            replayed_results = []
            pending_trials = trials_exec

        # Size this agent's trial pool. max_trials_global>0 caps it; when
        # unlimited (0), fall back to the agent's own max_concurrent, else one
        # worker per pending trial (the global + per-agent semaphores still
        # throttle actual in-flight work).
        _gc = int(run.execution.max_trials_global or 0)
        if _gc > 0:
            trial_workers = _gc
        else:
            _agent_cap = int(getattr(agent, "max_concurrent", 0) or 0)
            trial_workers = _agent_cap if _agent_cap > 0 else max(1, len(pending_trials))
        if not pending_trials:
            results = []
        elif trial_workers > 1:
            # Trial-level parallel: each trial gets its own container + target
            results = _run_agent_trials_parallel(
                run, agent, storage, pending_trials, hook_ctx, trial_workers,
                passk=passk, reporter=reporter,
            )
        else:
            results = _run_agent_trials_serial(
                run, agent, storage, pending_trials, hook_ctx, run_id,
                passk=passk, reporter=reporter,
            )
        # Merge replayed results back into the result list, preserving the
        # planned trial_index ordering so dashboards line up.
        results = sorted(replayed_results + list(results), key=lambda r: r.trial_index)
    finally:
        remove_file_handlers()

    # Score via Benchmark
    _score_trials(results, run.benchmark, storage)
    _mark_canonical_trial_scoring_for_results(storage, agent, run.benchmark, results)

    summary = _build_summary(results)
    storage.save_summary(summary)
    _mark_canonical_run_summary_artifact(storage)
    _mark_canonical_run_scored(storage, results)
    logger.info(
        "Agent %s: %d/%d completed, scores: %s",
        agent_label, summary["completed"], summary["total"],
        summary.get("mean_scores", {}),
    )

    return results

def _run_agent_trials_serial(
    run: ExperimentRun,
    agent: AgentInstance,
    storage: RunStorage,
    trials: list[Trial],
    hook_ctx: HookContext,
    run_id: str = "",
    *,
    passk: int = 1,
    reporter: Any | None = None,
) -> list[TrialResult]:
    """Run trials sequentially while preserving per-trial container isolation."""
    logger.info(
        "Agent %s: serial execution uses isolated per-trial containers",
        agent.label(),
    )
    kwargs: dict[str, Any] = {"passk": passk}
    if reporter is not None:
        kwargs["reporter"] = reporter
    return _run_agent_trials_parallel(
        run,
        agent,
        storage,
        trials,
        hook_ctx,
        max_workers=1,
        **kwargs,
    )
