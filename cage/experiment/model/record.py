"""Durable run/trial records — the persisted truth of an experiment run."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cage.experiment.model._serde import _json_ready, _stable_id
from cage.experiment.model.plan import ExperimentPlan, TrialPlan


@dataclass(frozen=True)
class ArtifactRef:
    """Stable pointer to a run or trial artifact.

    Records should reference large or mutable artifacts instead of embedding
    their contents. This shape gives inspect, score, export, and migration code
    a common vocabulary for following artifact paths and understanding whether
    an artifact can be replayed.
    """

    artifact_id: str
    path: str
    kind: str
    schema_version: str = ""
    producer: str = ""
    privacy: str = "public"
    replayability: str = "unknown"
    content_type: str = "application/json"
    created_at: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ArtifactIndex:
    """Serializable manifest of artifacts written for a run or trial."""

    schema_version: str = "artifact_index.v1"
    artifacts: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class TrialTermination:
    """Terminal condition observed for a trial, if it has finished."""

    reason: str | None = None
    signal: str | None = None
    exit_code: int | None = None


@dataclass(frozen=True)
class TrialScoringRecord:
    """Durable scoring refs for one trial.

    Runtime verifier evidence, offline judge evidence, and aggregate scores are
    separate artifacts. The status here summarizes scoring availability without
    overwriting the trial's runtime status.
    """

    status: str = "not_scored"
    live_evidence_ref: str | None = None
    final_evidence_ref: str | None = None
    judgment_ref: str | None = None
    score_ref: str | None = None


@dataclass(frozen=True)
class TrialRecord:
    """Durable truth for one trial at a point in its lifecycle."""

    schema_version: str
    trial_id: str
    run_id: str
    plan_ref: str
    status: str
    status_reason: str
    subject_id: str
    task_id: str
    pass_index: int
    target_id: str | None = None
    scoring_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    termination: TrialTermination = field(default_factory=TrialTermination)
    artifacts: tuple[ArtifactRef, ...] = ()
    scoring: TrialScoringRecord = field(default_factory=TrialScoringRecord)


@dataclass(frozen=True)
class TrialRecordRef:
    """Run-level reference to an individual trial record artifact."""

    trial_id: str
    record_ref: str


@dataclass(frozen=True)
class TrialRunSummary:
    """Run-level trial counts plus references to per-trial records."""

    total: int
    completed: int = 0
    failed: int = 0
    interrupted: int = 0
    records: tuple[TrialRecordRef, ...] = ()


@dataclass(frozen=True)
class SubjectRunRecord:
    """Run-level status for one planned subject."""

    subject_id: str
    status: str = "planned"


@dataclass(frozen=True)
class ScoreSummaryRecord:
    """Run-level scoring summary pointer."""

    status: str = "not_scored"
    summary_ref: str | None = None


@dataclass(frozen=True)
class InspectorObservation:
    """Inspector availability advertised by an experiment record."""

    enabled: bool = False
    urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationRecord:
    """Human/tool observation metadata for a run."""

    inspector: InspectorObservation = field(default_factory=InspectorObservation)


@dataclass(frozen=True)
class ExperimentRecord:
    """Durable run-level truth created when an experiment run starts."""

    schema_version: str
    run_id: str
    record_id: str
    status: str
    status_reason: str
    created_at: str
    spec_ref: str
    plan_ref: str
    artifact_index_ref: str
    event_log_ref: str
    resource_ledger_ref: str
    trials: TrialRunSummary
    subjects: tuple[SubjectRunRecord, ...]
    score_summary: ScoreSummaryRecord = field(default_factory=ScoreSummaryRecord)
    observation: ObservationRecord = field(default_factory=ObservationRecord)
    started_at: str | None = None
    completed_at: str | None = None
    interrupted_at: str | None = None


def artifact_index_to_mapping(index: ArtifactIndex) -> dict[str, Any]:
    """Return a JSON-ready mapping for an ``ArtifactIndex``."""

    return _json_ready(index)


def artifact_index_to_json(index: ArtifactIndex, *, indent: int | None = 2) -> str:
    """Serialize an ``ArtifactIndex`` to deterministic JSON text."""

    return (
        json.dumps(
            artifact_index_to_mapping(index),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def create_experiment_record(
    plan: ExperimentPlan,
    *,
    run_id: str,
    created_at: str,
    status: str = "planned",
    spec_ref: str = "experiment_spec.yml",
    plan_ref: str = "experiment_plan.json",
    artifact_index_ref: str = "artifact_index.json",
    event_log_ref: str = "events.jsonl",
    resource_ledger_ref: str = "resources.jsonl",
    inspector_enabled: bool = False,
    inspector_urls: tuple[str, ...] = (),
    trial_record_refs: Mapping[str, str] | None = None,
) -> ExperimentRecord:
    """Create the initial durable run record for an ``ExperimentPlan``.

    The returned record is a pure snapshot: it does not create directories,
    write files, start an inspector, or inspect any runtime state. Runtime code
    can write this object as the first run artifact, then update status and
    counts as trial records move through the lifecycle.

    ``trial_record_refs`` maps each ``trial_id`` to its run-relative record path.
    The canonical artifact writer supplies it from ``RunStorage`` so the durable
    record co-locates with the trial's runtime artifacts (one tree on disk). It
    is optional only to keep this builder usable as a pure contract in isolation;
    when omitted, a path is derived from the plan locally.
    """

    refs = trial_record_refs or {}
    trial_refs = tuple(
        TrialRecordRef(
            trial_id=trial.trial_id,
            record_ref=refs.get(trial.trial_id) or _trial_record_ref(trial),
        )
        for trial in plan.trials
    )
    subjects = tuple(
        SubjectRunRecord(subject_id=subject.subject_id)
        for subject in plan.subjects
    )
    record_id = _stable_id(
        "record",
        {
            "schema_version": "experiment_record.v1",
            "run_id": run_id,
            "plan_id": plan.plan_id,
            "trial_ids": tuple(trial.trial_id for trial in plan.trials),
        },
    )
    return ExperimentRecord(
        schema_version="experiment_record.v1",
        run_id=run_id,
        record_id=record_id,
        status=status,
        status_reason="",
        created_at=created_at,
        spec_ref=spec_ref,
        plan_ref=plan_ref,
        artifact_index_ref=artifact_index_ref,
        event_log_ref=event_log_ref,
        resource_ledger_ref=resource_ledger_ref,
        trials=TrialRunSummary(
            total=len(plan.trials),
            records=trial_refs,
        ),
        subjects=subjects,
        observation=ObservationRecord(
            inspector=InspectorObservation(
                enabled=inspector_enabled,
                urls=inspector_urls,
            )
        ),
    )


def create_trial_records(
    plan: ExperimentPlan,
    *,
    run_id: str,
    plan_ref: str = "../../experiment_plan.json",
) -> tuple[TrialRecord, ...]:
    """Create planned trial records for every trial in an ``ExperimentPlan``.

    These records are the pre-runtime baseline for resume, inspect, score, and
    partial-run handling. They intentionally contain no container ids, allocated
    ports, timestamps, or scorer outputs; those fields are filled as runtime
    phases produce durable evidence.
    """

    return tuple(
        TrialRecord(
            schema_version="trial_record.v1",
            trial_id=trial.trial_id,
            run_id=run_id,
            plan_ref=plan_ref,
            status="planned",
            status_reason="",
            subject_id=trial.subject_id,
            task_id=trial.task_id,
            pass_index=trial.pass_index,
        )
        for trial in plan.trials
    )


def experiment_record_to_mapping(record: ExperimentRecord) -> dict[str, Any]:
    """Return a JSON-ready mapping for an ``ExperimentRecord``."""

    return _json_ready(record)


def experiment_record_to_json(
    record: ExperimentRecord,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize an ``ExperimentRecord`` to deterministic JSON text."""

    return (
        json.dumps(
            experiment_record_to_mapping(record),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def trial_record_to_mapping(record: TrialRecord) -> dict[str, Any]:
    """Return a JSON-ready mapping for a ``TrialRecord``."""

    return _json_ready(record)


def trial_record_to_json(record: TrialRecord, *, indent: int | None = 2) -> str:
    """Serialize a ``TrialRecord`` to deterministic JSON text."""

    return (
        json.dumps(
            trial_record_to_mapping(record),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def _trial_record_ref(trial: TrialPlan) -> str:
    """Return the run-relative record path for one planned trial.

    The record co-locates with the runtime artifacts: it lives in the same
    ``trials/<runtime_id>/`` directory the trial runner writes to, so one trial
    is one directory on disk. ``runtime_id`` is the runtime subpath
    (``<task>`` or ``<task>/pass_<n>``); when absent (older plans), fall back to
    deriving a subject-prefixed path from the canonical ``trial_id``.
    """

    sub = str(getattr(trial, "runtime_id", "") or "").strip("/")
    if not sub:
        sub = str(trial.trial_id).replace(":", "_").strip("/")
    return f"trials/{sub}/record.json"
