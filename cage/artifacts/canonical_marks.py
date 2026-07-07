"""Canonical artifact marking: the durable-record WRITE path into the ArtifactIndex.

The ``_mark_canonical_*`` family records trial/run artifacts (started/finished,
outputs, prompts, proxy logs, scoring/trajectory/state evidence) and derives
the canonical per-agent spec/plan snapshot from ``ExperimentRun.spec`` via the
pure :mod:`cage.experiment.model` builders.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from cage.artifacts.record_snapshots import (
    RUN_HISTORY_FILE,
    _result_terminal_status,
)
from cage.artifacts.run_storage import RunStorage
from cage.artifacts.trial_session import TrialRuntimeSession
from cage.artifacts.writer import ExperimentArtifactWriter
from cage.benchmarks import Benchmark
from cage.experiment.model import (
    Trial,
    TrialResult,
    TrialTermination,
    experiment_spec_from_project_mapping,
    narrow_spec_to_agent_run,
    parse_trial_id,
    plan_from_trial_sequence,
)
from cage.experiment.model.plan import SubjectPlan
from cage.experiment.model.spec import BenchmarkReference
from cage.sandbox.naming import _parse_agent_label

if TYPE_CHECKING:
    from cage.agents.base import AgentInstance
    from cage.experiment.engine.run_context import ExperimentRun

logger = logging.getLogger(__name__)


def plan_trial_id(agent: "AgentInstance", trial: Any) -> str:
    """Trial id for one agent/trial (or trial-stub) pair.

    This is the **runtime** trial id (``<task>`` or ``<task>/pass_<n>``) — the
    same value as ``TrialPlan.trial_id`` / ``runtime_id`` and the on-disk
    ``trials/<id>/`` tree, so the canonical record, its ``record_ref``, and the
    physical directory all agree. The subject is kept in the separate
    ``subject_id`` field and is **never** prefixed into the trial id: a
    subject-qualified id (``agent:model:mode/<task>/pass_n``) split the
    inspector's row dedup and leaked the agent label into every displayed id.
    ``agent`` is retained for call-site compatibility. Accepts the lightweight
    result stubs used by score-time marking, hence the duck-typed ``trial``.
    """

    del agent  # subject no longer participates in the trial id
    return str(getattr(trial, "id", "") or "")




def _reset_canonical_trial_for_rerun(
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
) -> None:
    """Restore the live canonical record for a trial about to be re-run.

    Resume archives the prior attempt's whole trial directory, which (now that
    the record co-locates with runtime artifacts) carries its ``record.json``
    away too. This best-effort hook recreates the live planned record at the
    run record's ref so ``experiment_record.json`` stays resolvable and the
    re-run's lifecycle marks have a record to update. Best-effort: a failure
    here must not abort the resume.
    """

    try:
        ExperimentArtifactWriter(storage.run_dir).reset_trial_planned_record(
            plan_trial_id(agent, trial)
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord reset for resume re-run failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_run_output_artifacts(
    *,
    run_dir: Path,
    dashboard_path: Path,
    results_path: Path,
    dashboard_view_path: Path | None = None,
) -> None:
    """Index run-level compatibility and presentation artifacts.

    ``dashboard.json`` and ``results.csv`` remain compatibility projections for
    current CLI/web consumers. ``dashboard_view.json`` is the benchmark-owned
    presentation projection. Indexing all three makes export and inspect code
    resolve them through the same canonical artifact boundary as plans,
    records, and trial artifacts.
    """

    try:
        writer = ExperimentArtifactWriter(run_dir)
        history_path = run_dir / RUN_HISTORY_FILE
        if history_path.is_file():
            writer.mark_run_artifact(
                artifact_id="run.history",
                path=history_path,
                kind="run_history",
                schema_version="run_history.v1",
                producer="_append_run_history",
                replayability="replayable",
                content_type="application/json",
            )
        writer.mark_run_artifact(
            artifact_id="run.compat.dashboard",
            path=dashboard_path,
            kind="compat_dashboard",
            schema_version="dashboard.compat.v1",
            producer="_write_dashboards",
            replayability="compatibility",
            content_type="application/json",
        )
        writer.mark_run_artifact(
            artifact_id="run.compat.results_csv",
            path=results_path,
            kind="compat_results_csv",
            schema_version="results_csv.compat.v1",
            producer="_write_results_csv",
            replayability="compatibility",
            content_type="text/csv",
        )
        if dashboard_view_path is not None and dashboard_view_path.is_file():
            writer.mark_run_artifact(
                artifact_id="run.dashboard_view",
                path=dashboard_view_path,
                kind="dashboard_view",
                schema_version="dashboard_view.v1",
                producer="Benchmark.build_dashboard",
                replayability="replayable",
                content_type="application/json",
            )
    except Exception as exc:
        logger.warning(
            "canonical run output artifact registration failed for %s: %s",
            run_dir,
            exc,
        )


def _build_canonical_run_spec(
    run: ExperimentRun,
    agent: AgentInstance,
    trials: list[Trial],
    run_id: str,
) -> tuple[Any, Any]:
    """Resolve the canonical ``(spec, plan)`` for one single-agent run.

    Shared by the full initial-snapshot write and the resume-time spec refresh
    so both project the exact same resolved config.
    """
    spec_source = run.spec or experiment_spec_from_project_mapping(
        {},
        project_file=run.project_file,
        base_dir=run.project_file.parent,
    )
    benchmark_name = str(getattr(run.benchmark, "name", "") or run.name or "benchmark")
    benchmark_ref = BenchmarkReference(
        id=benchmark_name.strip().lower().replace("-", "_"),
        project_name=str(run.name or spec_source.benchmark.project_name),
        module=type(run.benchmark).__module__ if run.benchmark is not None else "",
        class_name=type(run.benchmark).__name__ if run.benchmark is not None else "",
        benchmark_root=str(run.benchmark_dir or ""),
        package_ref=str(run.benchmark_dir or ""),
        default_config_ref=str(run.project_file),
    )
    agent_id = agent.id or agent.agent_type.name
    spec = narrow_spec_to_agent_run(
        spec_source,
        run_id=run_id,
        agent_id=agent_id,
        agent_kind=agent.agent_type.name,
        model_id=agent.model.id,
        max_concurrent=agent.max_concurrent,
        task_ids=tuple(parse_trial_id(str(trial.id))[0] for trial in trials),
        passk=max(1, int(run.execution.passk or 1)),
        profile="stateful" if agent.stateful else "stateless",
        experiment_name=str(run.name or "") or None,
        benchmark=benchmark_ref,
    )
    subject = SubjectPlan(
        subject_id=agent.subject_plan_id,
        agent=agent_id,
        kind=agent.agent_type.name,
        profile="stateful" if agent.stateful else "stateless",
        model=agent.model.id,
        max_concurrent=agent.max_concurrent,
    )
    plan = plan_from_trial_sequence(spec, subject=subject, trials=trials)
    return spec, plan


def _refresh_canonical_spec_snapshot(
    run: ExperimentRun,
    agent: AgentInstance,
    storage: RunStorage,
    trials: list[Trial],
    run_id: str,
) -> None:
    """Refresh only ``experiment_spec.json`` on resume.

    Resume preserves the canonical record + per-trial evidence (see
    ``_resume_should_preserve_canonical_snapshot``), which skips the full
    snapshot write — so the spec projection keeps the original invocation's
    config and drifts from what actually executes (resolved fresh from the
    current YAML each run). Rewrite just the spec file to the current resolved
    config; it is a pure projection nothing executes from, so this touches no
    preserved evidence. Best-effort: a failure must never abort a resume.
    """
    try:
        spec, _plan = _build_canonical_run_spec(run, agent, trials, run_id)
        ExperimentArtifactWriter(storage.run_dir).refresh_spec(spec)
    except Exception as exc:  # noqa: BLE001 - cosmetic projection refresh
        logger.warning("resume: could not refresh experiment_spec.json: %s", exc)


def _save_canonical_experiment_snapshot(
    run: ExperimentRun,
    agent: AgentInstance,
    storage: RunStorage,
    trials: list[Trial],
    run_id: str,
) -> None:
    """Write canonical Experiment contract artifacts for one legacy agent run.

    The current orchestrator still schedules each agent under its own
    ``.cage_runs/<agent_label>/<run_id>`` directory. Until ``ExperimentRun`` owns
    a unified run directory, this helper snapshots the resolved single-agent
    plan beside the existing legacy artifacts. It has no runtime side effects
    beyond writing JSON files, and it runs after resume compatibility checks so
    a changed trial plan cannot overwrite old run metadata.
    """

    spec, plan = _build_canonical_run_spec(run, agent, trials, run_id)
    writer = ExperimentArtifactWriter(storage.run_dir)
    writer.write_initial_snapshot(
        spec=spec,
        plan=plan,
        run_id=run_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        inspector_enabled=(
            str(getattr(run.logging, "inspect_mode", "on") or "on").lower() != "off"
        ),
    )
    writer.mark_run_started(started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))


def _mark_canonical_experiment_records_finished(
    cage_runs: Path,
    *,
    agent_labels: Iterable[str],
    run_id: str,
    status: str,
    completed_at: str,
) -> None:
    """Best-effort terminal update for per-agent canonical run records."""

    status_reason = "user_interrupted" if status == "interrupted" else ""
    for label in agent_labels:
        agent_dir_name, _mode = _parse_agent_label(label)
        run_dir = cage_runs / agent_dir_name / run_id
        if not (run_dir / "experiment_record.json").is_file():
            continue
        try:
            ExperimentArtifactWriter(run_dir).mark_run_finished(
                status=status,
                completed_at=completed_at,
                status_reason=status_reason,
            )
        except Exception as exc:
            logger.warning(
                "canonical ExperimentRecord terminal update failed for %s: %s",
                run_dir,
                exc,
            )


def finalize_canonical_running_trials(
    cage_runs: Path,
    *,
    run_id: str,
    completed_at: str,
) -> list[str]:
    """Finalize trials left at ``running`` across every run dir for ``run_id``.

    A force-exit (second Ctrl+C / SIGTERM driving ``teardown_all`` then
    ``os._exit``) kills in-flight trials before ``mark_trial_finished`` runs, so
    their canonical record stays stuck at ``status="running"`` and the inspector
    shows a phantom "Running" forever. Sweep each per-agent run dir for this
    ``run_id`` and write a terminal ``interrupted`` status. Best-effort and
    idempotent. Returns the finalized trial ids.
    """

    finalized: list[str] = []
    for record_path in cage_runs.glob(f"*/{run_id}/experiment_record.json"):
        run_dir = record_path.parent
        try:
            finalized.extend(
                ExperimentArtifactWriter(run_dir).finalize_running_trials_as_interrupted(
                    completed_at=completed_at,
                )
            )
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            logger.warning("finalize running trials failed for %s: %s", run_dir, exc)
    return finalized


def _save_run_metadata_snapshot(
    run: ExperimentRun,
    agent: AgentInstance,
    storage: RunStorage,
) -> None:
    agent_label = agent.label()
    storage.save_config({
        "experiment": run.name,
        "agent": agent_label,
        "model": agent.model.id,
        "project_file": str(run.project_file.resolve()),
        "benchmark_dir": (
            str(Path(run.benchmark_dir).resolve())
            if run.benchmark_dir is not None
            else ""
        ),
        "stateful": agent.stateful,
        "shared_paths": agent.shared_paths,
    })

    # Persist the source project.yml so inspector/score/offline tooling can
    # reconstruct the Benchmark without the caller hunting down the original
    # file. Best-effort — never block a run on this.
    try:
        src = run.project_file.resolve()
        if src.is_file():
            dst = storage.run_dir / "project.yml"
            dst.write_bytes(src.read_bytes())
    except OSError as exc:
        logger.warning("Could not snapshot project.yml into run dir: %s", exc)

    # Persist preflight outcome alongside the run so the inspector can
    # surface "preflight said FAIL/PASS" next to trial results.
    pf_dict = run.metadata.get("_preflight_result")
    if isinstance(pf_dict, dict):
        (storage.run_dir / "preflight.json").write_text(
            json.dumps(pf_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _mark_canonical_run_metadata_artifacts(storage)


def _mark_canonical_run_metadata_artifacts(storage: RunStorage) -> None:
    """Index run-level provenance snapshots written for legacy tools.

    These files are compatibility/provenance projections. The canonical
    ``ExperimentSpec`` and ``ExperimentPlan`` remain the source of truth, but
    users and older tools still inspect these snapshots directly.
    """

    artifacts = (
        {
            "artifact_id": "run.compat.config",
            "path": storage.run_dir / "config.yaml",
            "kind": "compat_run_config",
            "schema_version": "config_yaml.compat.v1",
            "producer": "RunStorage.save_config",
            "content_type": "application/x-yaml",
        },
        {
            "artifact_id": "run.compat.project_yml",
            "path": storage.run_dir / "project.yml",
            "kind": "compat_project_yml",
            "schema_version": "project_yml.compat.v1",
            "producer": "_save_run_metadata_snapshot",
            "content_type": "application/x-yaml",
        },
        {
            "artifact_id": "run.preflight_result",
            "path": storage.run_dir / "preflight.json",
            "kind": "preflight_result",
            "schema_version": "preflight_result.v1",
            "producer": "cage.experiment.engine.preflight",
            "content_type": "application/json",
        },
    )
    try:
        writer = ExperimentArtifactWriter(storage.run_dir)
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_file():
                continue
            writer.mark_run_artifact(
                artifact_id=str(artifact["artifact_id"]),
                path=path,
                kind=str(artifact["kind"]),
                schema_version=str(artifact["schema_version"]),
                producer=str(artifact["producer"]),
                replayability="compatibility",
                content_type=str(artifact["content_type"]),
            )
    except Exception as exc:
        logger.warning(
            "canonical run metadata artifact registration failed for %s: %s",
            storage.run_dir,
            exc,
        )


def _mark_canonical_trial_started(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
) -> None:
    """Best-effort update of the canonical record for a started legacy trial."""

    try:
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        TrialRuntimeSession(
            run_dir=storage.run_dir,
            trial_id=plan_trial_id(agent, trial),
        ).mark_started(
            started_at=started_at,
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord start update failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_trial_finished(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
    result: TrialResult,
) -> None:
    """Best-effort terminal update of the canonical record for a legacy trial."""

    reason = str(result.metadata.get("termination_reason") or "").strip()
    status_reason = reason or (result.error or "")
    try:
        completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        status = _result_terminal_status(result)
        TrialRuntimeSession(
            run_dir=storage.run_dir,
            trial_id=plan_trial_id(agent, trial),
        ).mark_finished(
            status=status,
            completed_at=completed_at,
            status_reason=status_reason,
            termination=TrialTermination(
                reason=reason or None,
                exit_code=result.exit_code,
            ),
            payload={
                "termination_reason": reason,
                "exit_code": result.exit_code,
            },
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord terminal update failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_trial_output_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
) -> None:
    """Best-effort registration of the legacy task output artifact."""

    output_path = storage.trial_dir(trial.id) / "task_output.json"
    try:
        canonical_trial_id = plan_trial_id(agent, trial)
        ExperimentArtifactWriter(storage.run_dir).mark_trial_artifact(
            canonical_trial_id,
            artifact_id=f"trial.{canonical_trial_id}.task_output",
            path=output_path,
            kind="task_output",
            schema_version="task_output.v1",
            producer="RunStorage.save_trial_output",
            replayability="replayable",
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord task output update failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_trial_prompt_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
) -> None:
    """Best-effort registration of the rendered prompt artifact.

    The prompt is user-facing evidence: it explains exactly what the agent saw
    after benchmark-specific envelope rendering. Indexing it on the
    ``TrialRecord`` lets inspect and export code display canonical runs without
    assuming ``trials/<id>/prompt.txt`` exists as a legacy directory file.
    """

    prompt_path = storage.trial_dir(trial.id) / "prompt.txt"
    if not prompt_path.is_file():
        return
    try:
        canonical_trial_id = plan_trial_id(agent, trial)
        ExperimentArtifactWriter(storage.run_dir).mark_trial_artifact(
            canonical_trial_id,
            artifact_id=f"trial.{canonical_trial_id}.prompt",
            path=prompt_path,
            kind="prompt",
            schema_version="prompt.txt.v1",
            producer="Benchmark.build_prompt",
            replayability="replayable",
            content_type="text/plain",
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord prompt update failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_proxy_log_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
) -> None:
    """Best-effort registration of a retained raw ``proxy.jsonl`` artifact.

    Raw proxy logs are the audit source for trajectory reconstruction and
    future ``ProxyArtifactStore`` work. Registering them through the canonical
    trial record lets web, score, and export code resolve the file without
    guessing ``trials/<id>/proxy/proxy.jsonl``. Callers should only invoke this
    when the run intends to retain the raw proxy log on disk.
    """

    proxy_jsonl = storage.trial_proxy_dir(trial.id) / "proxy.jsonl"
    if not proxy_jsonl.is_file():
        return
    try:
        canonical_trial_id = plan_trial_id(agent, trial)
        ExperimentArtifactWriter(storage.run_dir).mark_trial_artifact(
            canonical_trial_id,
            artifact_id=f"trial.{canonical_trial_id}.proxy_log",
            path=proxy_jsonl,
            kind="proxy_log",
            schema_version="proxy_log.jsonl.v1",
            producer="ContainerProxy",
            replayability="audit",
            content_type="application/x-ndjson",
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord proxy log update failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_trial_scoring_evidence_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
    evidence_path: Path,
    kind: str,
    schema_version: str,
    producer: str,
    content_type: str,
) -> None:
    """Best-effort registration of verifier evidence for a trial.

    Evidence refs are scoring inputs, not scores. Preserve the existing scoring
    status so collecting runtime verifier output does not imply that final
    scoring or aggregation has already happened.
    """

    if not evidence_path.is_file():
        return
    try:
        canonical_trial_id = plan_trial_id(agent, trial)
        evidence_ref = Path(
            os.path.relpath(evidence_path, start=storage.run_dir)
        ).as_posix()
        writer = ExperimentArtifactWriter(storage.run_dir)
        updated = writer.mark_trial_artifact(
            canonical_trial_id,
            artifact_id=f"trial.{canonical_trial_id}.{kind}",
            path=evidence_path,
            kind=kind,
            schema_version=schema_version,
            producer=producer,
            privacy="scorer_private",
            replayability="audit",
            content_type=content_type,
        )
        evidence_kwargs: dict[str, str] = {}
        if kind == "live_evidence":
            evidence_kwargs["live_evidence_ref"] = evidence_ref
        elif kind == "final_evidence":
            evidence_kwargs["final_evidence_ref"] = evidence_ref
        writer.mark_trial_scored(
            canonical_trial_id,
            status=updated.scoring.status,
            **evidence_kwargs,
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord %s update failed for %s: %s",
            kind,
            trial.id,
            exc,
        )


def _mark_canonical_trial_final_evidence_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
    evidence_path: Path,
) -> None:
    """Index ``Benchmark.check_done`` output as final verifier evidence."""

    _mark_canonical_trial_scoring_evidence_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        evidence_path=evidence_path,
        kind="final_evidence",
        schema_version="check_done_output.txt.v1",
        producer="Benchmark.check_done",
        content_type="text/plain",
    )


def _mark_canonical_trial_live_evidence_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
    evidence_path: Path,
) -> None:
    """Index live verifier verdicts captured while the trial was running."""

    _mark_canonical_trial_scoring_evidence_artifact(
        storage=storage,
        agent=agent,
        trial=trial,
        evidence_path=evidence_path,
        kind="live_evidence",
        schema_version="live_success.json.v1",
        producer="LiveSuccessMonitor",
        content_type="application/json",
    )


def _mark_canonical_trial_trajectory_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
    traj_path: Path,
) -> None:
    """Register the ``.traj`` text projection for one trial.

    ``.traj`` is a human-readable trajectory projection generated from
    ``proxy.jsonl``. Index it explicitly so file browsers and export tools can
    discover it without assuming the legacy filename convention.
    """

    if not traj_path.is_file():
        return
    try:
        canonical_trial_id = plan_trial_id(agent, trial)
        ExperimentArtifactWriter(storage.run_dir).mark_trial_artifact(
            canonical_trial_id,
            artifact_id=f"trial.{canonical_trial_id}.compat_trajectory",
            path=traj_path,
            kind="compat_trajectory",
            schema_version="traj.compat.v1",
            producer="cage.proxy.trajectory.generate_traj",
            replayability="compatibility",
            content_type="text/plain",
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord trajectory update failed for %s: %s",
            trial.id,
            exc,
        )


def _mark_canonical_trial_state_artifact(
    *,
    storage: RunStorage,
    agent: AgentInstance,
    trial: Trial,
    state_dir: Path,
    phase: str,
) -> None:
    """Register a pre- or post-trial state snapshot directory.

    State snapshots are runtime evidence directories, not benchmark outputs.
    They can be large, but indexing the directory root gives inspect, export,
    and future replay tooling a canonical way to discover the snapshot without
    hard-coding ``state_pre`` / ``state_post`` paths.
    """

    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"pre", "post"}:
        raise ValueError(f"unsupported state snapshot phase: {phase!r}")
    if not state_dir.is_dir():
        return
    try:
        canonical_trial_id = plan_trial_id(agent, trial)
        ExperimentArtifactWriter(storage.run_dir).mark_trial_artifact(
            canonical_trial_id,
            artifact_id=f"trial.{canonical_trial_id}.state_{normalized_phase}",
            path=state_dir,
            kind=f"state_snapshot_{normalized_phase}",
            schema_version="state_snapshot.directory.v1",
            producer="snapshot_state",
            replayability="audit",
            content_type="inode/directory",
        )
    except Exception as exc:
        logger.warning(
            "canonical TrialRecord state_%s update failed for %s: %s",
            normalized_phase,
            trial.id,
            exc,
        )


def _mark_canonical_trial_scoring_for_results(
    storage: RunStorage,
    agent: AgentInstance,
    benchmark: Benchmark,
    results: list[TrialResult],
) -> None:
    """Record score artifact refs on canonical trial records after scoring.

    Existing scoring code writes legacy ``trials/<id>/scores/<benchmark>.json``
    files and mutates ``TrialResult.scores``. This bridge preserves that path
    while making the canonical ``TrialRecord.scoring`` field useful for
    inspect/resume/score migrations. It is best-effort because score recording
    must not change the semantics of a completed run.
    """

    scoring_id = str(getattr(benchmark, "name", "") or "").strip()
    if not scoring_id:
        return
    writer = ExperimentArtifactWriter(storage.run_dir)
    for result in results:
        if not result.scores:
            continue
        trial_stub = SimpleNamespace(
            id=result.trial_id,
            sample=(
                result.metadata.get("sample")
                if isinstance(result.metadata.get("sample"), dict)
                else {"id": result.sample_id}
            ),
        )
        canonical_trial_id = plan_trial_id(agent, trial_stub)
        score_path = storage.trial_dir(result.trial_id) / "scores" / f"{scoring_id}.json"
        score_ref = Path(os.path.relpath(score_path, start=storage.run_dir)).as_posix()
        try:
            writer.mark_trial_artifact(
                canonical_trial_id,
                artifact_id=f"trial.{canonical_trial_id}.score.{scoring_id}",
                path=score_ref,
                kind="trial_score",
                schema_version="trial_score.v1",
                producer="cage orchestrator",
                replayability="replayable",
            )
            writer.mark_trial_scored(
                canonical_trial_id,
                score_ref=score_ref,
                scoring_id=scoring_id,
            )
        except Exception as exc:
            logger.warning(
                "canonical TrialRecord scoring update failed for %s: %s",
                result.trial_id,
                exc,
            )


def _mark_canonical_run_scored(storage: RunStorage, results: list[TrialResult]) -> None:
    """Record the run-level score summary ref when any trial has scores.

    ``summary.json`` is still the legacy aggregate artifact. This bridge makes
    it discoverable from ``ExperimentRecord.score_summary`` without changing the
    summary schema or requiring score consumers to switch APIs in the same step.
    """

    if not any(result.scores for result in results):
        return
    try:
        writer = ExperimentArtifactWriter(storage.run_dir)
        summary_ref = "summary.json"
        if (storage.run_dir / summary_ref).is_file():
            writer.mark_run_artifact(
                artifact_id="run.score_summary",
                path=summary_ref,
                kind="score_summary",
                schema_version="score_summary.v1",
                producer="cage orchestrator",
                replayability="replayable",
            )
        writer.mark_run_scored(summary_ref=summary_ref)
    except Exception as exc:
        logger.warning(
            "canonical ExperimentRecord score summary update failed for %s: %s",
            storage.run_dir,
            exc,
        )


def _mark_canonical_run_summary_artifact(storage: RunStorage) -> None:
    """Index the legacy run ``summary.json`` regardless of scoring state.

    ``summary.json`` is written for every agent run as the current compact
    aggregate of trial counts and mean scores. A run with no scores still has a
    useful summary, so it should be discoverable through ``ArtifactIndex``. If
    scores exist, ``_mark_canonical_run_scored`` later re-indexes the same path
    as ``score_summary`` and updates ``ExperimentRecord.score_summary``.
    """

    try:
        summary_ref = "summary.json"
        if not (storage.run_dir / summary_ref).is_file():
            return
        ExperimentArtifactWriter(storage.run_dir).mark_run_artifact(
            artifact_id="run.compat.summary",
            path=summary_ref,
            kind="compat_run_summary",
            schema_version="summary.compat.v1",
            producer="RunStorage.save_summary",
            replayability="compatibility",
            content_type="application/json",
        )
    except Exception as exc:
        logger.warning(
            "canonical run summary artifact registration failed for %s: %s",
            storage.run_dir,
            exc,
        )

