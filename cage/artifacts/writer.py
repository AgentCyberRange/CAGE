"""Run-artifact writer: builds the immutable run snapshot on disk."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from cage.artifacts.reader import ExperimentArtifactReader
from cage.artifacts.records import (
    ExperimentArtifactSnapshot,
    _artifact_ref,
    _artifact_ref_to_mapping,
    _atomic_write_text,
    _read_json,
    _relative_path,
    _sha256_artifact_path,
    _terminal_run_status,
    _terminal_trial_status,
    _trial_resources_projection_to_json,
    _upsert_trial_artifact,
)
from cage.artifacts.resources import ResourceLedgerReader
from cage.artifacts.run_storage import trial_record_ref as storage_trial_record_ref
from cage.contracts.trial_status import (
    COMPLETED_TRIAL_STATUSES,
    FAILED_TRIAL_STATUSES,
    INTERRUPTED_TRIAL_STATUSES,
)
from cage.experiment.model import (
    ArtifactIndex,
    ArtifactRef,
    ExperimentPlan,
    ExperimentRecord,
    ExperimentSpec,
    ResourceRecord,
    TrialEvent,
    TrialRecord,
    TrialTermination,
    artifact_index_to_json,
    create_experiment_record,
    create_trial_records,
    experiment_plan_to_json,
    experiment_record_to_json,
    experiment_spec_to_json,
    trial_event_to_json,
    trial_record_to_json,
)


class ExperimentEventWriter:
    """Append canonical trial lifecycle events to a run event log."""

    def __init__(self, run_dir: str | Path) -> None:
        """Create an event writer rooted at one experiment run directory."""

        self.run_dir = Path(run_dir).expanduser().resolve()

    def append_trial_event(
        self,
        *,
        run_id: str,
        trial_id: str,
        phase: str,
        event_type: str,
        timestamp: str,
        subject_id: str = "",
        task_id: str = "",
        payload: Mapping[str, object] | None = None,
        artifact_refs: tuple[str, ...] = (),
        resource_refs: tuple[str, ...] = (),
        error: str | None = None,
        log_ref: str | Path = "events.jsonl",
    ) -> TrialEvent:
        """Append one trial event and return the written contract object.

        ``log_ref`` is run-directory relative by default. Runtime sessions use
        it to write the same lifecycle schema to both the run-level timeline
        and the trial-local ``trials/<id>/events.jsonl`` timeline.
        """

        log_path = Path(log_ref)
        if not log_path.is_absolute():
            log_path = self.run_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event = TrialEvent(
            schema_version="trial_event.v1",
            event_id=self._next_event_id(log_path),
            run_id=run_id,
            trial_id=trial_id,
            phase=phase,
            type=event_type,
            timestamp=timestamp,
            subject_id=subject_id,
            task_id=task_id,
            payload=dict(payload or {}),
            artifact_refs=artifact_refs,
            resource_refs=resource_refs,
            error=error,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(trial_event_to_json(event))
        return event

    def _next_event_id(self, log_path: Path) -> str:
        """Return the next monotonic event id for ``log_path``."""

        if not log_path.is_file():
            return "evt_000001"
        count = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
        return f"evt_{count + 1:06d}"

class ExperimentArtifactWriter:
    """Write canonical experiment artifacts under one run directory.

    The writer owns only durable local artifact paths and atomic text writes.
    It deliberately accepts already-built contract objects so callers can keep
    planning, runtime execution, and artifact persistence separate.
    """

    def __init__(self, run_dir: str | Path) -> None:
        """Create a writer rooted at ``run_dir`` without touching runtime state."""

        self.run_dir = Path(run_dir).expanduser().resolve()

    def write_initial_snapshot(
        self,
        *,
        spec: ExperimentSpec,
        plan: ExperimentPlan,
        run_id: str,
        created_at: str,
        inspector_enabled: bool = False,
        inspector_urls: tuple[str, ...] = (),
    ) -> ExperimentArtifactSnapshot:
        """Write the initial canonical run snapshot and return written paths.

        This method creates only files and directories below ``run_dir``. It is
        safe to call before the runtime starts because it does not inspect live
        resources or mutate the supplied contracts. Existing canonical JSON
        files are replaced atomically so a repeated call cannot leave half-written
        JSON behind on normal completion.
        """

        self.run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.run_dir / "experiment_spec.json"
        plan_path = self.run_dir / "experiment_plan.json"
        record_path = self.run_dir / "experiment_record.json"
        artifact_index_path = self.run_dir / "artifact_index.json"
        event_log_path = self.run_dir / "events.jsonl"
        resource_ledger_path = self.run_dir / "resources.jsonl"

        record = create_experiment_record(
            plan,
            run_id=run_id,
            created_at=created_at,
            spec_ref=spec_path.name,
            plan_ref=plan_path.name,
            artifact_index_ref=artifact_index_path.name,
            inspector_enabled=inspector_enabled,
            inspector_urls=inspector_urls,
            trial_record_refs={
                trial.trial_id: storage_trial_record_ref(trial.runtime_id)
                for trial in plan.trials
                if trial.runtime_id
            },
        )
        trial_records = self._trial_records_with_relative_plan_refs(plan, run_id, record)
        trial_record_paths = {
            trial_ref.trial_id: self.run_dir / trial_ref.record_ref
            for trial_ref in record.trials.records
        }
        trial_event_log_paths = {
            trial_id: trial_path.parent / "events.jsonl"
            for trial_id, trial_path in trial_record_paths.items()
        }
        trial_resource_paths = {
            trial_id: trial_path.parent / "resources.json"
            for trial_id, trial_path in trial_record_paths.items()
        }

        written_refs: list[ArtifactRef] = []
        _atomic_write_text(record_path, experiment_record_to_json(record))
        written_refs.append(
            _artifact_ref(
                artifact_id="run.experiment_record",
                path=record_path,
                root=self.run_dir,
                kind="experiment_record",
                schema_version=record.schema_version,
                producer="ExperimentArtifactWriter",
                replayability="replayable",
            )
        )

        _atomic_write_text(spec_path, experiment_spec_to_json(spec))
        written_refs.append(
            _artifact_ref(
                artifact_id="run.experiment_spec",
                path=spec_path,
                root=self.run_dir,
                kind="experiment_spec",
                schema_version=spec.schema_version,
                producer="ExperimentArtifactWriter",
                replayability="replayable",
            )
        )

        _atomic_write_text(plan_path, experiment_plan_to_json(plan))
        written_refs.append(
            _artifact_ref(
                artifact_id="run.experiment_plan",
                path=plan_path,
                root=self.run_dir,
                kind="experiment_plan",
                schema_version=plan.schema_version,
                producer="ExperimentArtifactWriter",
                replayability="replayable",
            )
        )

        written_trial_records: list[TrialRecord] = []
        for trial_record in trial_records:
            trial_path = trial_record_paths[trial_record.trial_id]
            trial_event_log_path = trial_event_log_paths[trial_record.trial_id]
            trial_resource_path = trial_resource_paths[trial_record.trial_id]
            _atomic_write_text(trial_event_log_path, "")
            _atomic_write_text(
                trial_resource_path,
                _trial_resources_projection_to_json(
                    run_id=run_id,
                    trial_id=trial_record.trial_id,
                    resources=(),
                ),
            )
            trial_event_ref = _artifact_ref(
                artifact_id=f"trial.{trial_record.trial_id}.events",
                path=trial_event_log_path,
                root=trial_path.parent,
                kind="trial_event_log",
                schema_version="trial_event.v1",
                producer="ExperimentArtifactWriter",
                replayability="replayable",
                content_type="application/x-jsonlines",
            )
            trial_resource_ref = _artifact_ref(
                artifact_id=f"trial.{trial_record.trial_id}.resources",
                path=trial_resource_path,
                root=trial_path.parent,
                kind="trial_resource_projection",
                schema_version="trial_resources.v1",
                producer="ExperimentArtifactWriter",
                replayability="replayable",
            )
            trial_record = replace(
                trial_record,
                artifacts=_upsert_trial_artifact(
                    _upsert_trial_artifact(trial_record.artifacts, trial_event_ref),
                    trial_resource_ref,
                ),
            )
            _atomic_write_text(trial_path, trial_record_to_json(trial_record))
            written_trial_records.append(trial_record)
            written_refs.append(
                _artifact_ref(
                    artifact_id=f"trial.{trial_record.trial_id}.record",
                    path=trial_path,
                    root=self.run_dir,
                    kind="trial_record",
                    schema_version=trial_record.schema_version,
                    producer="ExperimentArtifactWriter",
                    replayability="replayable",
                )
            )
            written_refs.append(
                _artifact_ref(
                    artifact_id=f"trial.{trial_record.trial_id}.events",
                    path=trial_event_log_path,
                    root=self.run_dir,
                    kind="trial_event_log",
                    schema_version="trial_event.v1",
                    producer="ExperimentArtifactWriter",
                    replayability="replayable",
                    content_type="application/x-jsonlines",
                )
            )
            written_refs.append(
                _artifact_ref(
                    artifact_id=f"trial.{trial_record.trial_id}.resources",
                    path=trial_resource_path,
                    root=self.run_dir,
                    kind="trial_resource_projection",
                    schema_version="trial_resources.v1",
                    producer="ExperimentArtifactWriter",
                    replayability="replayable",
                )
            )

        _atomic_write_text(event_log_path, "")
        written_refs.append(
            _artifact_ref(
                artifact_id="run.events",
                path=event_log_path,
                root=self.run_dir,
                kind="trial_event_log",
                schema_version="trial_event.v1",
                producer="ExperimentArtifactWriter",
                replayability="replayable",
                content_type="application/x-jsonlines",
            )
        )

        _atomic_write_text(resource_ledger_path, "")
        written_refs.append(
            _artifact_ref(
                artifact_id="run.resources",
                path=resource_ledger_path,
                root=self.run_dir,
                kind="resource_ledger",
                schema_version="resource_ledger.v1",
                producer="ExperimentArtifactWriter",
                replayability="replayable",
                content_type="application/x-jsonlines",
            )
        )

        index = ArtifactIndex(
            artifacts=(
                *written_refs,
                ArtifactRef(
                    artifact_id="run.artifact_index",
                    path=artifact_index_path.name,
                    kind="artifact_index",
                    schema_version="artifact_index.v1",
                    producer="ExperimentArtifactWriter",
                    replayability="replayable",
                ),
            )
        )
        _atomic_write_text(artifact_index_path, artifact_index_to_json(index))

        return ExperimentArtifactSnapshot(
            run_dir=self.run_dir,
            spec_path=spec_path,
            plan_path=plan_path,
            record_path=record_path,
            artifact_index_path=artifact_index_path,
            trial_record_paths=trial_record_paths,
            trial_event_log_paths=trial_event_log_paths,
            trial_resource_paths=trial_resource_paths,
            record=record,
            trial_records=tuple(written_trial_records),
        )

    def _trial_records_with_relative_plan_refs(
        self,
        plan: ExperimentPlan,
        run_id: str,
        record: ExperimentRecord,
    ) -> tuple[TrialRecord, ...]:
        """Create planned trial records with path-correct plan references."""

        refs_by_trial = {
            trial_ref.trial_id: self.run_dir / trial_ref.record_ref
            for trial_ref in record.trials.records
        }
        plan_path = self.run_dir / record.plan_ref
        records: list[TrialRecord] = []
        for trial_record in create_trial_records(plan, run_id=run_id):
            trial_path = refs_by_trial[trial_record.trial_id]
            records.append(
                replace(
                    trial_record,
                    plan_ref=_relative_path(plan_path, start=trial_path.parent),
                )
            )
        return tuple(records)

    def mark_run_started(
        self,
        *,
        started_at: str,
        status_reason: str = "",
    ) -> ExperimentRecord:
        """Mark the run-level record as running and persist it atomically."""

        record = self._load_record()
        updated = replace(
            record,
            status="running",
            status_reason=status_reason,
            started_at=started_at,
        )
        self._write_experiment_record(updated)
        return updated

    def mark_run_finished(
        self,
        *,
        status: str,
        completed_at: str,
        status_reason: str = "",
    ) -> ExperimentRecord:
        """Mark the run-level record terminal and persist it atomically."""

        normalized = _terminal_run_status(status)
        record = self._load_record()
        updated = replace(
            record,
            status=normalized,
            status_reason=status_reason,
            completed_at=completed_at,
            interrupted_at=(
                completed_at if normalized == "interrupted" else record.interrupted_at
            ),
        )
        self._write_experiment_record(updated)
        return updated

    def mark_run_scored(
        self,
        *,
        summary_ref: str | None = None,
        status: str = "scored",
    ) -> ExperimentRecord:
        """Update the run-level scoring summary pointer.

        This records that aggregation or post-run scoring produced a durable
        summary artifact. It deliberately does not change run completion status:
        runtime lifecycle and scoring lifecycle are separate, and score reruns
        may happen long after a run has completed.
        """

        record = self._load_record()
        updated = replace(
            record,
            score_summary=replace(
                record.score_summary,
                status=status,
                summary_ref=(
                    summary_ref
                    if summary_ref is not None
                    else record.score_summary.summary_ref
                ),
            ),
        )
        self._write_experiment_record(updated)
        return updated

    def mark_trial_started(
        self,
        trial_id: str,
        *,
        started_at: str,
        target_id: str | None = None,
        status_reason: str = "",
    ) -> TrialRecord:
        """Mark one trial record as running and persist it atomically."""

        trial_ref = self._trial_record_ref(trial_id)
        record = self._load_trial_record(trial_ref)
        updated = replace(
            record,
            status="running",
            status_reason=status_reason,
            started_at=started_at,
            target_id=target_id if target_id is not None else record.target_id,
        )
        self._write_trial_record(trial_ref, updated)
        return updated

    def mark_trial_finished(
        self,
        trial_id: str,
        *,
        status: str,
        completed_at: str,
        status_reason: str = "",
        termination: TrialTermination | None = None,
    ) -> TrialRecord:
        """Mark one trial record terminal and refresh run-level trial counts."""

        normalized = _terminal_trial_status(status)
        trial_ref = self._trial_record_ref(trial_id)
        record = self._load_trial_record(trial_ref)
        updated = replace(
            record,
            status=normalized,
            status_reason=status_reason,
            completed_at=completed_at,
            termination=termination or record.termination,
        )
        self._write_trial_record(trial_ref, updated)
        self._refresh_run_trial_counts()
        return updated

    def finalize_running_trials_as_interrupted(
        self,
        *,
        completed_at: str,
        status_reason: str = "user_interrupted",
    ) -> list[str]:
        """Finalize every trial still recorded as ``running`` to ``interrupted``.

        When a run is force-terminated (a second Ctrl+C / SIGTERM drives
        ``teardown_all`` then ``os._exit``), in-flight trials are killed before
        ``mark_trial_finished`` runs, so their record stays stuck at
        ``status="running"`` — the inspector then shows a phantom "Running" long
        after the run is dead. This sweeps the canonical record and writes a
        terminal ``interrupted`` status for each, so the on-disk truth matches
        what happened. Idempotent: a later call finds no running trials and is a
        no-op. Returns the trial ids it finalized.
        """

        finalized: list[str] = []
        for trial_ref in self._load_record().trials.records:
            trial_record = self._load_trial_record(trial_ref.record_ref)
            if str(trial_record.status or "").strip().lower() != "running":
                continue
            self.mark_trial_finished(
                trial_ref.trial_id,
                status="interrupted",
                completed_at=completed_at,
                status_reason=status_reason,
                termination=TrialTermination(reason=status_reason, exit_code=-1),
            )
            finalized.append(trial_ref.trial_id)
        return finalized

    def mark_trial_scored(
        self,
        trial_id: str,
        *,
        score_ref: str | None = None,
        scoring_id: str | None = None,
        status: str = "scored",
        live_evidence_ref: str | None = None,
        final_evidence_ref: str | None = None,
        judgment_ref: str | None = None,
    ) -> TrialRecord:
        """Update one trial's scoring refs without changing runtime status.

        Scoring is a separate lifecycle from trial execution: a trial can fail
        at runtime and still have scoring evidence, or complete successfully and
        remain ``not_scored``. This method only updates ``TrialRecord.scoring``
        plus the selected ``scoring_id`` so runtime status/counters remain owned
        by ``mark_trial_finished``.
        """

        trial_ref = self._trial_record_ref(trial_id)
        record = self._load_trial_record(trial_ref)
        scoring = replace(
            record.scoring,
            status=status,
            live_evidence_ref=(
                live_evidence_ref
                if live_evidence_ref is not None
                else record.scoring.live_evidence_ref
            ),
            final_evidence_ref=(
                final_evidence_ref
                if final_evidence_ref is not None
                else record.scoring.final_evidence_ref
            ),
            judgment_ref=(
                judgment_ref
                if judgment_ref is not None
                else record.scoring.judgment_ref
            ),
            score_ref=score_ref if score_ref is not None else record.scoring.score_ref,
        )
        updated = replace(
            record,
            scoring=scoring,
            scoring_id=scoring_id if scoring_id is not None else record.scoring_id,
        )
        self._write_trial_record(trial_ref, updated)
        return updated

    def mark_run_artifact(
        self,
        *,
        artifact_id: str,
        path: str | Path,
        kind: str,
        schema_version: str = "",
        producer: str = "",
        privacy: str = "public",
        replayability: str = "unknown",
        content_type: str = "application/json",
        created_at: str | None = None,
    ) -> ArtifactRef:
        """Register a durable run-level artifact in ``artifact_index.json``.

        Run-level artifacts such as offline score summaries are not attached to
        a single ``TrialRecord`` but still need a canonical ref so inspect,
        export, and future score/replay tooling can discover them without path
        guessing. The caller owns writing the artifact first; this method only
        validates that the file or directory exists and indexes it.
        """

        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = self.run_dir / artifact_path
        if not artifact_path.exists() and not artifact_path.is_symlink():
            raise FileNotFoundError(artifact_path)
        artifact_ref = _artifact_ref(
            artifact_id=artifact_id,
            path=artifact_path,
            root=self.run_dir,
            kind=kind,
            schema_version=schema_version,
            producer=producer,
            replayability=replayability,
            privacy=privacy,
            content_type=content_type,
            created_at=created_at,
        )
        self._upsert_artifact_index_entry(artifact_ref)
        return artifact_ref

    def mark_trial_artifact(
        self,
        trial_id: str,
        *,
        artifact_id: str,
        path: str | Path,
        kind: str,
        schema_version: str = "",
        producer: str = "",
        privacy: str = "public",
        replayability: str = "unknown",
        content_type: str = "application/json",
        created_at: str | None = None,
    ) -> TrialRecord:
        """Attach a durable artifact ref to one trial record.

        Runtime code writes the artifact first, then calls this method to make
        the artifact discoverable by inspect, score, replay, and export code.
        The method does not create or mutate the artifact itself. Artifact
        paths may name either files or directories; directory artifacts receive
        a deterministic tree checksum.
        """

        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = self.run_dir / artifact_path
        if not artifact_path.exists() and not artifact_path.is_symlink():
            raise FileNotFoundError(artifact_path)

        artifact_ref = _artifact_ref(
            artifact_id=artifact_id,
            path=artifact_path,
            root=self.run_dir,
            kind=kind,
            schema_version=schema_version,
            producer=producer,
            replayability=replayability,
            privacy=privacy,
            content_type=content_type,
            created_at=created_at,
        )
        trial_ref = self._trial_record_ref(trial_id)
        record = self._load_trial_record(trial_ref)
        updated = replace(
            record,
            artifacts=_upsert_trial_artifact(record.artifacts, artifact_ref),
        )
        self._write_trial_record(trial_ref, updated)
        self._upsert_artifact_index_entry(artifact_ref)
        return updated

    def refresh_trial_resource_projection(self, trial_id: str) -> TrialRecord:
        """Rewrite ``trials/.../resources.json`` from the run resource ledger.

        ``resources.jsonl`` remains the append-only cleanup source of truth.
        The trial-local projection is an inspect/web/debug convenience artifact
        that lets readers explain one trial without scanning the whole run.
        """

        resources = tuple(
            record
            for record in ResourceLedgerReader(self.run_dir).load_records()
            if record.trial_id == trial_id
        )
        return self.write_trial_resource_projection(trial_id, resources=resources)

    def write_trial_resource_projection(
        self,
        trial_id: str,
        *,
        resources: tuple[ResourceRecord, ...],
    ) -> TrialRecord:
        """Persist the trial-local resource projection and refresh refs."""

        trial_ref = self._trial_record_ref(trial_id)
        trial_path = self.run_dir / trial_ref
        projection_path = trial_path.parent / "resources.json"
        record = self._load_trial_record(trial_ref)
        _atomic_write_text(
            projection_path,
            _trial_resources_projection_to_json(
                run_id=record.run_id,
                trial_id=record.trial_id,
                resources=resources,
            ),
        )
        local_ref = _artifact_ref(
            artifact_id=f"trial.{trial_id}.resources",
            path=projection_path,
            root=trial_path.parent,
            kind="trial_resource_projection",
            schema_version="trial_resources.v1",
            producer="ExperimentArtifactWriter",
            replayability="replayable",
        )
        updated = replace(
            record,
            artifacts=_upsert_trial_artifact(record.artifacts, local_ref),
        )
        self._write_trial_record(trial_ref, updated)
        self._upsert_artifact_index_entry(
            _artifact_ref(
                artifact_id=f"trial.{trial_id}.resources",
                path=projection_path,
                root=self.run_dir,
                kind="trial_resource_projection",
                schema_version="trial_resources.v1",
                producer="ExperimentArtifactWriter",
                replayability="replayable",
            )
        )
        return updated

    def reset_trial_planned_record(self, trial_id: str) -> TrialRecord | None:
        """Restore one trial's live canonical record to ``planned``.

        On ``--resume`` the prior attempt's whole trial directory is archived,
        and since the canonical record now co-locates with the runtime artifacts
        (one trial = one directory), that archive carries the record away too.
        This recreates the live planned ``record.json`` plus empty trial-local
        event/resource projections at the run record's ref, so
        ``experiment_record.json`` stays resolvable and the re-run's lifecycle
        marks have a record to update — while the archived copy keeps the prior
        attempt's evidence. Returns ``None`` if the trial id is unknown.
        """

        record_ref = self._trial_record_ref(trial_id)
        trial_path = self.run_dir / record_ref
        run_record = self._load_record()
        plan = ExperimentArtifactReader(self.run_dir).load_plan()
        planned = {
            record.trial_id: record
            for record in create_trial_records(plan, run_id=run_record.run_id)
        }
        trial_record = planned.get(trial_id)
        if trial_record is None:
            return None

        plan_path = self.run_dir / run_record.plan_ref
        event_log_path = trial_path.parent / "events.jsonl"
        resource_path = trial_path.parent / "resources.json"
        _atomic_write_text(event_log_path, "")
        _atomic_write_text(
            resource_path,
            _trial_resources_projection_to_json(
                run_id=run_record.run_id,
                trial_id=trial_id,
                resources=(),
            ),
        )
        event_ref = _artifact_ref(
            artifact_id=f"trial.{trial_id}.events",
            path=event_log_path,
            root=trial_path.parent,
            kind="trial_event_log",
            schema_version="trial_event.v1",
            producer="ExperimentArtifactWriter",
            replayability="replayable",
            content_type="application/x-jsonlines",
        )
        resource_ref = _artifact_ref(
            artifact_id=f"trial.{trial_id}.resources",
            path=resource_path,
            root=trial_path.parent,
            kind="trial_resource_projection",
            schema_version="trial_resources.v1",
            producer="ExperimentArtifactWriter",
            replayability="replayable",
        )
        trial_record = replace(
            trial_record,
            plan_ref=_relative_path(plan_path, start=trial_path.parent),
            artifacts=_upsert_trial_artifact(
                _upsert_trial_artifact(trial_record.artifacts, event_ref),
                resource_ref,
            ),
        )
        self._write_trial_record(record_ref, trial_record)
        return trial_record

    def _load_record(self) -> ExperimentRecord:
        """Load the current run record from this writer's run directory."""

        return ExperimentArtifactReader(self.run_dir).load_record()

    def _load_trial_record(self, record_ref: str) -> TrialRecord:
        """Load one trial record from this writer's run directory."""

        return ExperimentArtifactReader(self.run_dir).load_trial_record(record_ref)

    def _write_experiment_record(self, record: ExperimentRecord) -> None:
        """Persist the run record and refresh its artifact-index checksum."""

        path = self.run_dir / "experiment_record.json"
        _atomic_write_text(path, experiment_record_to_json(record))
        self._refresh_artifact_index_entry(path)

    def _write_trial_record(self, record_ref: str, record: TrialRecord) -> None:
        """Persist one trial record and refresh its artifact-index checksum."""

        path = self.run_dir / record_ref
        _atomic_write_text(path, trial_record_to_json(record))
        self._refresh_artifact_index_entry(path)

    def _trial_record_ref(self, trial_id: str) -> str:
        """Return the run-relative record ref for ``trial_id``."""

        for trial_ref in self._load_record().trials.records:
            if trial_ref.trial_id == trial_id:
                return trial_ref.record_ref
        raise KeyError(f"unknown trial_id: {trial_id}")

    def _refresh_run_trial_counts(self) -> ExperimentRecord:
        """Recount terminal trial records and persist the run-level summary."""

        record = self._load_record()
        completed = failed = interrupted = 0
        for trial_ref in record.trials.records:
            trial = self._load_trial_record(trial_ref.record_ref)
            status = trial.status.strip().lower()
            if status in COMPLETED_TRIAL_STATUSES:
                completed += 1
            elif status in FAILED_TRIAL_STATUSES:
                failed += 1
            elif status in INTERRUPTED_TRIAL_STATUSES:
                interrupted += 1
        updated = replace(
            record,
            trials=replace(
                record.trials,
                completed=completed,
                failed=failed,
                interrupted=interrupted,
            ),
        )
        self._write_experiment_record(updated)
        return updated

    def _refresh_artifact_index_entry(self, path: Path) -> None:
        """Refresh the artifact-index checksum for a mutated artifact."""

        index_path = self.run_dir / "artifact_index.json"
        if not index_path.is_file():
            return
        index_data = _read_json(index_path)
        artifacts = index_data.get("artifacts")
        if not isinstance(artifacts, list):
            return
        rel_path = _relative_path(path, start=self.run_dir)
        next_artifacts: list[object] = []
        changed = False
        for artifact in artifacts:
            if isinstance(artifact, Mapping) and str(artifact.get("path")) == rel_path:
                updated = dict(artifact)
                updated["sha256"] = _sha256_artifact_path(path)
                next_artifacts.append(updated)
                changed = True
            else:
                next_artifacts.append(artifact)
        if not changed:
            return
        index_data = dict(index_data)
        index_data["artifacts"] = next_artifacts
        _atomic_write_text(
            index_path,
            json.dumps(index_data, indent=2, sort_keys=True) + "\n",
        )

    def _upsert_artifact_index_entry(self, artifact_ref: ArtifactRef) -> None:
        """Insert or replace a non-record artifact in ``artifact_index.json``."""

        index_path = self.run_dir / "artifact_index.json"
        if not index_path.is_file():
            return
        index_data = _read_json(index_path)
        artifacts = index_data.get("artifacts")
        if not isinstance(artifacts, list):
            return
        next_artifacts = [
            artifact
            for artifact in artifacts
            if not (
                isinstance(artifact, Mapping)
                and str(artifact.get("path")) == artifact_ref.path
            )
        ]
        next_artifacts.append(_artifact_ref_to_mapping(artifact_ref))
        index_data = dict(index_data)
        index_data["artifacts"] = next_artifacts
        _atomic_write_text(
            index_path,
            json.dumps(index_data, indent=2, sort_keys=True) + "\n",
        )
