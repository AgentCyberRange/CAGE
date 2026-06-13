"""Value types, (de)serializers and low-level IO helpers for run artifacts.

Leaf module shared by :mod:`cage.artifacts.writer` and
:mod:`cage.artifacts.reader`. It depends only on the experiment data
model and contracts so the writer/reader can both build on it without a cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from cage.contracts.execution import normalize_max_rounds_config
from cage.contracts.trial_status import (
    COMPLETED_TRIAL_STATUSES,
    FAILED_TRIAL_STATUSES,
    INTERRUPTED_TRIAL_STATUSES,
)
from cage.experiment.model import (
    AgentModelSelection,
    AgentSelection,
    ArtifactIndex,
    ArtifactRef,
    BenchmarkReference,
    BenchmarkTaskPlan,
    ExperimentIdentity,
    ExperimentPlan,
    ExperimentRecord,
    ExperimentSpec,
    InspectorObservation,
    ObservationRecord,
    ObservationSpec,
    PlanSource,
    ProtocolControls,
    ProxySpec,
    ResourceRecord,
    RuntimeSpec,
    SchedulerSpec,
    ScoreSummaryRecord,
    ScoringSelection,
    SubjectPlan,
    SubjectRunRecord,
    TaskSelection,
    TimeoutSpec,
    TrialEvent,
    TrialPlan,
    TrialRecord,
    TrialRecordRef,
    TrialRunSummary,
    TrialScoringRecord,
    TrialTermination,
    WorkloadSpec,
    resource_record_to_mapping,
)

_TERMINAL_RUN_STATUSES = frozenset({"completed", "interrupted", "failed", "cancelled"})
_TERMINAL_TRIAL_STATUSES = (
    COMPLETED_TRIAL_STATUSES | FAILED_TRIAL_STATUSES | INTERRUPTED_TRIAL_STATUSES
)


@dataclass(frozen=True)
class ExperimentArtifactSnapshot:
    """Paths written for an initial canonical experiment snapshot."""

    run_dir: Path
    spec_path: Path
    plan_path: Path
    record_path: Path
    artifact_index_path: Path
    trial_record_paths: Mapping[str, Path]
    trial_event_log_paths: Mapping[str, Path]
    trial_resource_paths: Mapping[str, Path]
    record: ExperimentRecord
    trial_records: tuple[TrialRecord, ...]

@dataclass(frozen=True)
class ExperimentArtifactReadSnapshot:
    """Canonical artifacts loaded from one durable run directory.

    This is the reader-side counterpart to ``ExperimentArtifactSnapshot``. It is
    intentionally data-only: no benchmark imports, Docker inspection, dashboard
    parsing, or runtime probing. CLI commands should prefer this object when
    they need a consistent run view for inspect, resume, score, export, or gc
    decisions.
    """

    run_dir: Path
    spec: ExperimentSpec
    plan: ExperimentPlan
    record: ExperimentRecord
    artifact_index: ArtifactIndex
    trial_records: tuple[TrialRecord, ...]
    events: tuple[TrialEvent, ...]
    trial_events: Mapping[str, tuple[TrialEvent, ...]]
    resources: tuple[ResourceRecord, ...]

@dataclass(frozen=True)
class ResolvedTrialArtifact:
    """An indexed trial artifact resolved to an absolute local path.

    Produced by :meth:`ExperimentArtifactReader.resolve_trial_artifacts`: it
    represents an artifact present on BOTH the trial record and the
    ``ArtifactIndex``, never an unindexed path guess.
    """

    ref_path: str
    kind: str
    path: Path

def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a same-directory temporary file.

    The temporary file lives beside the destination so ``Path.replace`` is an
    atomic rename on normal local filesystems. Callers should still handle
    exceptions as artifact-write failures at the runtime layer.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def _terminal_run_status(status: str) -> str:
    """Normalize and validate a terminal run status."""

    normalized = status.strip().lower()
    if normalized not in _TERMINAL_RUN_STATUSES:
        allowed = ", ".join(sorted(_TERMINAL_RUN_STATUSES))
        raise ValueError(f"run status must be one of: {allowed}")
    return normalized

def _terminal_trial_status(status: str) -> str:
    """Normalize and validate a terminal trial status."""

    normalized = status.strip().lower()
    if normalized not in _TERMINAL_TRIAL_STATUSES:
        allowed = ", ".join(sorted(_TERMINAL_TRIAL_STATUSES))
        raise ValueError(f"trial status must be one of: {allowed}")
    return normalized

def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object from ``path`` with a contract-oriented error message."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data

def _spec_from_mapping(data: Mapping[str, object]) -> ExperimentSpec:
    """Reconstruct an ``ExperimentSpec`` from its canonical JSON snapshot."""

    identity_data = _mapping(data.get("identity"), "identity")
    benchmark_data = _mapping(data.get("benchmark"), "benchmark")
    workload_data = _mapping(data.get("workload"), "workload")
    task_selection_data = _mapping(
        workload_data.get("task_selection"),
        "workload.task_selection",
    )
    protocol_data = _mapping(data.get("protocol"), "protocol")
    runtime_data = _mapping(data.get("runtime"), "runtime")
    scheduler_data = _mapping(runtime_data.get("scheduler"), "runtime.scheduler")
    timeouts_data = _mapping(runtime_data.get("timeouts"), "runtime.timeouts")
    proxy_data = _mapping(runtime_data.get("proxy"), "runtime.proxy")
    scoring_data = _mapping(data.get("scoring"), "scoring")
    observation_data = _mapping(data.get("observation"), "observation")
    variants_data = _mapping(workload_data.get("variants"), "workload.variants")
    return ExperimentSpec(
        schema_version=str(data["schema_version"]),
        project_file=Path(str(data["project_file"])),
        base_dir=Path(str(data["base_dir"])),
        identity=ExperimentIdentity(
            experiment_id=str(identity_data["experiment_id"]),
            display_name=str(identity_data["display_name"]),
            description=str(identity_data.get("description") or ""),
            tags=tuple(str(item) for item in _list(identity_data.get("tags"))),
            run_id=str(identity_data.get("run_id") or ""),
        ),
        benchmark=BenchmarkReference(
            id=str(benchmark_data["id"]),
            project_name=str(benchmark_data["project_name"]),
            module=str(benchmark_data["module"]),
            class_name=str(benchmark_data.get("class_name") or ""),
            benchmark_root=str(benchmark_data.get("benchmark_root") or ""),
            package_ref=str(benchmark_data.get("package_ref") or ""),
            default_config_ref=str(benchmark_data.get("default_config_ref") or ""),
        ),
        workload=WorkloadSpec(
            subjects=tuple(
                _agent_selection_from_mapping(_mapping(item, "workload.subjects[]"))
                for item in _list(workload_data.get("subjects"))
            ),
            task_selection=TaskSelection(
                samples=tuple(
                    str(item)
                    for item in _list(task_selection_data.get("samples"))
                ),
                max_sample_num=_optional_int(task_selection_data.get("max_sample_num")),
                max_trial_num=_optional_int(task_selection_data.get("max_trial_num")),
            ),
            variants={
                str(key): tuple(str(item) for item in _list(value))
                for key, value in variants_data.items()
            },
            passk=int(workload_data.get("passk") or 1),
        ),
        protocol=ProtocolControls(
            max_rounds=normalize_max_rounds_config(protocol_data["max_rounds"]),
            max_input_tokens=_optional_int(protocol_data.get("max_input_tokens")),
            max_output_tokens=_optional_int(protocol_data.get("max_output_tokens")),
            max_cost=_optional_float(protocol_data.get("max_cost")),
        ),
        runtime=RuntimeSpec(
            scheduler=SchedulerSpec(
                max_trials_global=int(
                    scheduler_data.get("max_trials_global", scheduler_data.get("max_workers", 1))
                ),
                max_concurrent=_optional_int(scheduler_data.get("max_concurrent")),
                max_target_setups=int(scheduler_data.get("max_target_setups") or 1),
            ),
            timeouts=TimeoutSpec(
                trial_timeout_s=float(timeouts_data["trial_timeout_s"]),
                request_timeout_s=float(timeouts_data["request_timeout_s"]),
                target_startup_timeout_s=_optional_float(
                    timeouts_data.get("target_startup_timeout_s")
                ),
                target_compose_timeout_s=_optional_float(
                    timeouts_data.get("target_compose_timeout_s")
                ),
            ),
            proxy=ProxySpec(
                enabled=bool(proxy_data["enabled"]),
                request_timeout_s=float(proxy_data["request_timeout_s"]),
                upstream_http_proxy=str(proxy_data.get("upstream_http_proxy") or ""),
            ),
            target_enabled=bool(runtime_data.get("target_enabled", True)),
            allow_launch_build=bool(runtime_data.get("allow_launch_build", False)),
            inspector_start=bool(runtime_data.get("inspector_start", True)),
        ),
        scoring=ScoringSelection(
            scorer=str(scoring_data.get("scorer") or "benchmark_default"),
            judge_model=_optional_str(scoring_data.get("judge_model")),
        ),
        observation=ObservationSpec(
            terminal_ui=str(observation_data.get("terminal_ui") or "plain"),
            debug_log=bool(observation_data.get("debug_log", False)),
        ),
    )

def _agent_selection_from_mapping(data: Mapping[str, object]) -> AgentSelection:
    """Reconstruct one workload subject selection from serialized spec data."""

    return AgentSelection(
        agent=str(data["agent"]),
        kind=str(data["kind"]),
        profile=str(data.get("profile") or "stateless"),
        models=tuple(
            AgentModelSelection(
                model=str(_mapping(item, "workload.subjects[].models[]")["model"]),
                max_concurrent=_optional_int(
                    _mapping(item, "workload.subjects[].models[]").get(
                        "max_concurrent"
                    )
                ),
                overrides=_optional_mapping(
                    _mapping(item, "workload.subjects[].models[]").get("overrides")
                ),
            )
            for item in _list(data.get("models"))
        ),
        max_concurrent=_optional_int(data.get("max_concurrent")),
    )

def _plan_from_mapping(data: Mapping[str, object]) -> ExperimentPlan:
    """Reconstruct an ``ExperimentPlan`` from a JSON-ready mapping."""

    source_data = _mapping(data.get("source"), "source")
    controls_data = _mapping(data.get("controls"), "controls")
    return ExperimentPlan(
        schema_version=str(data["schema_version"]),
        plan_id=str(data["plan_id"]),
        source=PlanSource(
            project_file=Path(str(source_data["project_file"])),
            benchmark_id=str(source_data["benchmark_id"]),
            cli_overrides=tuple(
                dict(_mapping(item, "source.cli_overrides[]"))
                for item in _list(source_data.get("cli_overrides"))
            ),
        ),
        subjects=tuple(
            _subject_plan_from_mapping(_mapping(item, "subjects[]"))
            for item in _list(data.get("subjects"))
        ),
        tasks=tuple(
            _task_plan_from_mapping(_mapping(item, "tasks[]"))
            for item in _list(data.get("tasks"))
        ),
        trials=tuple(
            _trial_plan_from_mapping(_mapping(item, "trials[]"))
            for item in _list(data.get("trials"))
        ),
        controls=ProtocolControls(
            max_rounds=normalize_max_rounds_config(controls_data["max_rounds"]),
            max_input_tokens=_optional_int(controls_data.get("max_input_tokens")),
            max_output_tokens=_optional_int(controls_data.get("max_output_tokens")),
            max_cost=_optional_float(controls_data.get("max_cost")),
        ),
    )

def _subject_plan_from_mapping(data: Mapping[str, object]) -> SubjectPlan:
    """Reconstruct one planned subject from serialized plan data."""

    return SubjectPlan(
        subject_id=str(data["subject_id"]),
        agent=str(data["agent"]),
        kind=str(data["kind"]),
        profile=str(data["profile"]),
        model=str(data["model"]),
        max_concurrent=_optional_int(data.get("max_concurrent")),
    )

def _task_plan_from_mapping(data: Mapping[str, object]) -> BenchmarkTaskPlan:
    """Reconstruct one benchmark task row from serialized plan data."""

    return BenchmarkTaskPlan(
        task_id=str(data["task_id"]),
        source_sample_id=str(data["source_sample_id"]),
        variant_id=str(data["variant_id"]),
        axis_values={
            str(key): str(value)
            for key, value in _mapping(data.get("axis_values"), "axis_values").items()
        },
    )

def _trial_plan_from_mapping(data: Mapping[str, object]) -> TrialPlan:
    """Reconstruct one trial expansion row from serialized plan data."""

    return TrialPlan(
        trial_id=str(data["trial_id"]),
        subject_id=str(data["subject_id"]),
        task_id=str(data["task_id"]),
        pass_index=int(data["pass_index"]),
        runtime_id=str(data.get("runtime_id", "") or ""),
    )

def _experiment_record_from_mapping(data: Mapping[str, object]) -> ExperimentRecord:
    """Reconstruct an ``ExperimentRecord`` from a JSON-ready mapping."""

    trials_data = _mapping(data.get("trials"), "trials")
    score_data = _mapping(data.get("score_summary"), "score_summary")
    observation_data = _mapping(data.get("observation"), "observation")
    inspector_data = _mapping(observation_data.get("inspector"), "observation.inspector")
    return ExperimentRecord(
        schema_version=str(data["schema_version"]),
        run_id=str(data["run_id"]),
        record_id=str(data["record_id"]),
        status=str(data["status"]),
        status_reason=str(data.get("status_reason") or ""),
        created_at=str(data["created_at"]),
        spec_ref=str(data["spec_ref"]),
        plan_ref=str(data["plan_ref"]),
        artifact_index_ref=str(data["artifact_index_ref"]),
        event_log_ref=str(data["event_log_ref"]),
        resource_ledger_ref=str(data["resource_ledger_ref"]),
        trials=TrialRunSummary(
            total=int(trials_data["total"]),
            completed=int(trials_data.get("completed") or 0),
            failed=int(trials_data.get("failed") or 0),
            interrupted=int(trials_data.get("interrupted") or 0),
            records=tuple(
                TrialRecordRef(
                    trial_id=str(_mapping(item, "trials.records[]")["trial_id"]),
                    record_ref=str(_mapping(item, "trials.records[]")["record_ref"]),
                )
                for item in _list(trials_data.get("records"))
            ),
        ),
        subjects=tuple(
            SubjectRunRecord(
                subject_id=str(_mapping(item, "subjects[]")["subject_id"]),
                status=str(_mapping(item, "subjects[]").get("status") or "planned"),
            )
            for item in _list(data.get("subjects"))
        ),
        score_summary=ScoreSummaryRecord(
            status=str(score_data.get("status") or "not_scored"),
            summary_ref=_optional_str(score_data.get("summary_ref")),
        ),
        observation=ObservationRecord(
            inspector=InspectorObservation(
                enabled=bool(inspector_data.get("enabled", False)),
                urls=tuple(str(item) for item in _list(inspector_data.get("urls"))),
            )
        ),
        started_at=_optional_str(data.get("started_at")),
        completed_at=_optional_str(data.get("completed_at")),
        interrupted_at=_optional_str(data.get("interrupted_at")),
    )

def _trial_record_from_mapping(data: Mapping[str, object]) -> TrialRecord:
    """Reconstruct a ``TrialRecord`` from a JSON-ready mapping."""

    termination_data = _mapping(data.get("termination"), "termination")
    scoring_data = _mapping(data.get("scoring"), "scoring")
    return TrialRecord(
        schema_version=str(data["schema_version"]),
        trial_id=str(data["trial_id"]),
        run_id=str(data["run_id"]),
        plan_ref=str(data["plan_ref"]),
        status=str(data["status"]),
        status_reason=str(data.get("status_reason") or ""),
        subject_id=str(data["subject_id"]),
        task_id=str(data["task_id"]),
        pass_index=int(data["pass_index"]),
        target_id=_optional_str(data.get("target_id")),
        scoring_id=_optional_str(data.get("scoring_id")),
        started_at=_optional_str(data.get("started_at")),
        completed_at=_optional_str(data.get("completed_at")),
        termination=TrialTermination(
            reason=_optional_str(termination_data.get("reason")),
            signal=_optional_str(termination_data.get("signal")),
            exit_code=_optional_int(termination_data.get("exit_code")),
        ),
        artifacts=tuple(
            _artifact_ref_from_mapping(_mapping(item, "artifacts[]"))
            for item in _list(data.get("artifacts"))
        ),
        scoring=TrialScoringRecord(
            status=str(scoring_data.get("status") or "not_scored"),
            live_evidence_ref=_optional_str(scoring_data.get("live_evidence_ref")),
            final_evidence_ref=_optional_str(scoring_data.get("final_evidence_ref")),
            judgment_ref=_optional_str(scoring_data.get("judgment_ref")),
            score_ref=_optional_str(scoring_data.get("score_ref")),
        ),
    )

def _trial_event_from_mapping(data: Mapping[str, object]) -> TrialEvent:
    """Reconstruct one ``TrialEvent`` from a JSONL object."""

    return TrialEvent(
        schema_version=str(data["schema_version"]),
        event_id=str(data["event_id"]),
        run_id=str(data["run_id"]),
        trial_id=str(data["trial_id"]),
        phase=str(data["phase"]),
        type=str(data["type"]),
        timestamp=str(data["timestamp"]),
        subject_id=str(data.get("subject_id") or ""),
        task_id=str(data.get("task_id") or ""),
        payload=dict(_mapping(data.get("payload"), "payload")),
        artifact_refs=tuple(str(item) for item in _list(data.get("artifact_refs"))),
        resource_refs=tuple(str(item) for item in _list(data.get("resource_refs"))),
        error=_optional_str(data.get("error")),
    )

def _artifact_ref_from_mapping(data: Mapping[str, object]) -> ArtifactRef:
    """Reconstruct one ``ArtifactRef`` from serialized record data."""

    return ArtifactRef(
        artifact_id=str(data["artifact_id"]),
        path=str(data["path"]),
        kind=str(data["kind"]),
        schema_version=str(data.get("schema_version") or ""),
        producer=str(data.get("producer") or ""),
        privacy=str(data.get("privacy") or "public"),
        replayability=str(data.get("replayability") or "unknown"),
        content_type=str(data.get("content_type") or "application/json"),
        created_at=_optional_str(data.get("created_at")),
        sha256=_optional_str(data.get("sha256")),
    )

def _artifact_index_from_mapping(data: Mapping[str, object]) -> ArtifactIndex:
    """Reconstruct an ``ArtifactIndex`` from its canonical JSON object."""

    return ArtifactIndex(
        schema_version=str(data.get("schema_version") or "artifact_index.v1"),
        artifacts=tuple(
            _artifact_ref_from_mapping(_mapping(item, "artifacts[]"))
            for item in _list(data.get("artifacts"))
        ),
    )

def _trial_resources_projection_to_json(
    *,
    run_id: str,
    trial_id: str,
    resources: tuple[ResourceRecord, ...],
) -> str:
    """Serialize the trial-local resource projection as deterministic JSON."""

    return (
        json.dumps(
            {
                "resources": [
                    resource_record_to_mapping(record)
                    for record in resources
                ],
                "run_id": run_id,
                "schema_version": "trial_resources.v1",
                "trial_id": trial_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

def _artifact_ref(
    *,
    artifact_id: str,
    path: Path,
    root: Path,
    kind: str,
    schema_version: str,
    producer: str,
    replayability: str,
    privacy: str = "public",
    content_type: str = "application/json",
    created_at: str | None = None,
) -> ArtifactRef:
    """Build an ``ArtifactRef`` for a file or directory below ``root``."""

    return ArtifactRef(
        artifact_id=artifact_id,
        path=_relative_path(path, start=root),
        kind=kind,
        schema_version=schema_version,
        producer=producer,
        privacy=privacy,
        replayability=replayability,
        content_type=content_type,
        created_at=created_at,
        sha256=_sha256_artifact_path(path),
    )

def _upsert_trial_artifact(
    artifacts: tuple[ArtifactRef, ...],
    artifact_ref: ArtifactRef,
) -> tuple[ArtifactRef, ...]:
    """Return trial artifacts with ``artifact_ref`` inserted or refreshed."""

    kept = tuple(
        item
        for item in artifacts
        if item.artifact_id != artifact_ref.artifact_id and item.path != artifact_ref.path
    )
    return (*kept, artifact_ref)

def _artifact_ref_to_mapping(ref: ArtifactRef) -> dict[str, object]:
    """Return a JSON-ready mapping for one artifact ref."""

    return {
        "artifact_id": ref.artifact_id,
        "path": ref.path,
        "kind": ref.kind,
        "schema_version": ref.schema_version,
        "producer": ref.producer,
        "privacy": ref.privacy,
        "replayability": ref.replayability,
        "content_type": ref.content_type,
        "created_at": ref.created_at,
        "sha256": ref.sha256,
    }

def _relative_path(path: Path, *, start: Path) -> str:
    """Return a POSIX-style relative path between two local paths."""

    return Path(os.path.relpath(path, start=start)).as_posix()

def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a written artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _sha256_artifact_path(path: Path) -> str:
    """Return a stable digest for a file, symlink, or directory artifact."""

    if path.is_symlink():
        digest = hashlib.sha256()
        digest.update(b"symlink\0")
        digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        return digest.hexdigest()
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        return _sha256_directory(path)
    raise FileNotFoundError(path)

def _sha256_directory(root: Path) -> str:
    """Return a deterministic tree digest for a directory artifact.

    The digest includes entry type, relative POSIX path, symlink target, and
    file bytes. It deliberately avoids platform-specific absolute paths so the
    same artifact tree hashes identically after moving the run directory.
    """

    digest = hashlib.sha256()
    digest.update(b"directory\0")
    for child in sorted(
        root.rglob("*"),
        key=lambda item: _relative_path(item, start=root),
    ):
        rel = child.relative_to(root).as_posix().encode(
            "utf-8",
            errors="surrogateescape",
        )
        if child.is_symlink():
            digest.update(b"symlink\0")
            digest.update(rel)
            digest.update(b"\0")
            digest.update(os.readlink(child).encode("utf-8", errors="surrogateescape"))
        elif child.is_dir():
            digest.update(b"dir\0")
            digest.update(rel)
        elif child.is_file():
            digest.update(b"file\0")
            digest.update(rel)
            digest.update(b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"other\0")
            digest.update(rel)
            digest.update(str(child.lstat().st_mode).encode("ascii"))
    return digest.hexdigest()

def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Return ``value`` as a mapping or raise with the serialized field name."""

    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{field_name} must be a JSON object")

def _optional_mapping(value: object) -> Mapping[str, object] | None:
    """Return optional serialized mapping values without inventing defaults."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("expected a JSON object or null")

def _list(value: object) -> list[object]:
    """Return ``value`` as a list, accepting missing values as empty lists."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise ValueError("expected a JSON array")

def _optional_str(value: object) -> str | None:
    """Convert optional serialized scalar values to strings."""

    if value is None:
        return None
    return str(value)

def _optional_int(value: object) -> int | None:
    """Convert optional serialized scalar values to integers."""

    if value is None:
        return None
    return int(value)

def _optional_float(value: object) -> float | None:
    """Convert optional serialized scalar values to floats."""

    if value is None:
        return None
    return float(value)
