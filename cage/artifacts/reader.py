"""Run-artifact reader: rehydrates the run snapshot from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator, Mapping, TypeVar

from cage.artifacts.resources import ResourceLedgerReader
from cage.experiment.model import (
    ArtifactIndex,
    ArtifactRef,
    ExperimentPlan,
    ExperimentRecord,
    ExperimentSpec,
    ResourceRecord,
    TrialEvent,
    TrialRecord,
)
from cage.artifacts.records import (
    ExperimentArtifactReadSnapshot,
    ResolvedTrialArtifact,
    _artifact_index_from_mapping,
    _experiment_record_from_mapping,
    _plan_from_mapping,
    _read_json,
    _spec_from_mapping,
    _trial_event_from_mapping,
    _trial_record_from_mapping,
)

_T = TypeVar("_T")


class ExperimentArtifactReader:
    """Read canonical experiment artifacts from one run directory.

    The reader is the future shared entry point for inspect, resume, score, and
    web projection code. It reconstructs contract dataclasses from JSON files
    and intentionally does not infer run truth from dashboard/progress legacy
    files.
    """

    def __init__(self, run_dir: str | Path) -> None:
        """Create a reader rooted at ``run_dir`` without validating runtime state."""

        self.run_dir = Path(run_dir).expanduser().resolve()
        # Per-instance parse cache for the small run-level index files. A reader
        # is a point-in-time view: every caller constructs a fresh one and uses
        # it transiently (one inspect request, one resume trial, one writer
        # read-modify-write), so caching on the instance reuses each parse
        # within that view without ever serving stale state across writes.
        self._index_cache: dict[str, object] = {}

    # -- run-level index files (small; parsed once per reader instance) -------
    #
    # ``experiment_{spec,plan,record}.json`` and ``artifact_index.json`` are the
    # run's *directory* — small files re-read on nearly every lookup
    # (``find_artifact`` alone parses the 115 KB artifact index per call). The
    # per-instance cache keeps one reader from reparsing them; a new reader
    # always reflects current disk state.

    def _load_index_json(
        self, name: str, parse: Callable[[Mapping[str, object]], _T]
    ) -> _T:
        cached = self._index_cache.get(name)
        if cached is not None:
            return cached  # type: ignore[return-value]
        parsed = parse(_read_json(self.run_dir / name))
        self._index_cache[name] = parsed
        return parsed

    def load_spec(self) -> ExperimentSpec:
        """Load the canonical ``ExperimentSpec`` snapshot for this run."""

        return self._load_index_json("experiment_spec.json", _spec_from_mapping)

    def load_plan(self) -> ExperimentPlan:
        """Load the canonical ``ExperimentPlan`` snapshot for this run."""

        return self._load_index_json("experiment_plan.json", _plan_from_mapping)

    def load_record(self) -> ExperimentRecord:
        """Load the canonical run-level ``ExperimentRecord`` snapshot."""

        return self._load_index_json(
            "experiment_record.json", _experiment_record_from_mapping
        )

    def load_artifact_index(self) -> ArtifactIndex:
        """Load the canonical artifact index for this run.

        The index is the contract-level directory of durable run and trial
        artifacts. Consumers should use it before opening optional artifacts so
        score, inspect, export, and web code agree on what Cage actually wrote.
        """

        return self._load_index_json(
            "artifact_index.json", _artifact_index_from_mapping
        )

    # -- trial-keyed access (per-trial query, not a whole-run snapshot) -------

    def trial_record_by_id(self, trial_id: str) -> TrialRecord | None:
        """Load exactly one ``TrialRecord`` by its canonical ``trial_id``.

        The run record indexes every trial by id with a direct ``record_ref``,
        so this resolves to a single-file read — the right boundary for callers
        (resume, targeted inspect) that want one trial, not the whole run.
        ``None`` if the id is unknown or its record file is unreadable.
        """

        try:
            record = self.load_record()
        except (FileNotFoundError, OSError, ValueError):
            return None
        target = str(trial_id)
        for trial_ref in record.trials.records:
            if str(trial_ref.trial_id) != target:
                continue
            try:
                return self.load_trial_record(trial_ref.record_ref)
            except (FileNotFoundError, OSError, ValueError):
                return None
        return None

    def iter_trial_records(self) -> "Iterator[TrialRecord]":
        """Lazily yield trial records in declared order (one file read each).

        Unlike :meth:`load_trial_records` (which tuples the whole run) this is a
        generator, so a caller scanning for one match — e.g. a legacy trial
        whose id does not equal its record id — stops reading as soon as it
        finds it instead of materializing every record.
        """

        try:
            record = self.load_record()
        except (FileNotFoundError, OSError, ValueError):
            return
        for trial_ref in record.trials.records:
            try:
                yield self.load_trial_record(trial_ref.record_ref)
            except (FileNotFoundError, OSError, ValueError):
                continue

    def find_artifact(
        self,
        *,
        artifact_id: str | None = None,
        path: str | Path | None = None,
        kind: str | None = None,
    ) -> ArtifactRef | None:
        """Return the first indexed artifact matching the supplied filters.

        At least one filter is required. ``path`` is matched after normalizing
        to a run-relative POSIX path so callers can pass either an ``ArtifactRef``
        path string or a local ``Path`` below the run directory.
        """

        if artifact_id is None and path is None and kind is None:
            raise ValueError("at least one artifact filter is required")
        expected_path = (
            self._normalize_artifact_ref_path(path)
            if path is not None
            else None
        )
        for artifact in self.load_artifact_index().artifacts:
            if artifact_id is not None and artifact.artifact_id != artifact_id:
                continue
            if expected_path is not None and artifact.path != expected_path:
                continue
            if kind is not None and artifact.kind != kind:
                continue
            return artifact
        return None

    def resolve_artifact_path(self, artifact: ArtifactRef | str | Path) -> Path:
        """Resolve an indexed artifact reference to an absolute local path.

        Raw string/path inputs must already be present in ``artifact_index``.
        That extra check keeps higher-level consumers from silently inventing
        paths when a canonical run did not record the artifact they wanted.
        """

        if isinstance(artifact, ArtifactRef):
            ref = artifact
        else:
            normalized = self._normalize_artifact_ref_path(artifact)
            ref = self.find_artifact(path=normalized)
            if ref is None:
                raise KeyError(f"artifact is not indexed: {normalized}")
        normalized_path = self._normalize_artifact_ref_path(ref.path)
        if normalized_path != ref.path:
            raise ValueError(f"artifact path escapes run directory: {ref.path}")
        resolved = (self.run_dir / ref.path).resolve()
        try:
            resolved.relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError(
                f"artifact path resolves outside run directory: {ref.path}"
            ) from exc
        return resolved

    def load_trial_records(self) -> tuple[TrialRecord, ...]:
        """Load trial records in the order declared by ``ExperimentRecord``."""

        record = self.load_record()
        return tuple(
            self.load_trial_record(trial_ref.record_ref)
            for trial_ref in record.trials.records
        )

    def load_trial_record(self, record_ref: str | Path) -> TrialRecord:
        """Load one ``TrialRecord`` by its run-relative record reference."""

        return _trial_record_from_mapping(_read_json(self.run_dir / record_ref))

    def resolve_trial_artifacts(
        self, trial_record: TrialRecord
    ) -> list[ResolvedTrialArtifact]:
        """Return the indexed artifacts attached to ``trial_record``.

        This is the canonical "what durable files does this trial own" policy:
        every artifact that appears on BOTH the trial record and the
        ``ArtifactIndex`` (matched by id, kind and run-relative path), resolved
        to an absolute local path that exists. Unindexed path guesses are
        intentionally excluded so consumers cannot bless files Cage never
        recorded. Order follows ``trial_record.artifacts``.
        """

        resolved_artifacts: list[ResolvedTrialArtifact] = []
        for artifact in trial_record.artifacts:
            try:
                indexed = self.find_artifact(
                    artifact_id=artifact.artifact_id,
                    kind=artifact.kind,
                )
                if indexed is None or indexed.path != artifact.path:
                    continue
                resolved = self.resolve_artifact_path(indexed)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
            if resolved.is_file() or resolved.is_dir():
                resolved_artifacts.append(
                    ResolvedTrialArtifact(
                        ref_path=indexed.path,
                        kind=indexed.kind,
                        path=resolved,
                    )
                )
        return resolved_artifacts

    def load_events(self, log_ref: str | Path = "events.jsonl") -> tuple[TrialEvent, ...]:
        """Load a run- or trial-level lifecycle event log.

        ``log_ref`` is run-directory relative unless absolute. The default reads
        the run-level timeline. Callers that already know a trial-local log path
        can pass ``trials/<id>/events.jsonl`` directly.
        """

        path = Path(log_ref)
        if not path.is_absolute():
            path = self.run_dir / path
        if not path.is_file():
            return ()
        events: list[TrialEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            raw = json.loads(text)
            if isinstance(raw, Mapping):
                events.append(_trial_event_from_mapping(raw))
        return tuple(events)

    def load_trial_events(self, trial_id: str) -> tuple[TrialEvent, ...]:
        """Load the trial-local lifecycle event log for ``trial_id``."""

        record = self.load_record()
        for trial_ref in record.trials.records:
            if trial_ref.trial_id == trial_id:
                event_ref = Path(trial_ref.record_ref).parent / "events.jsonl"
                return self.load_events(event_ref)
        raise KeyError(f"unknown trial_id: {trial_id}")

    def load_resources(self) -> tuple[ResourceRecord, ...]:
        """Load the canonical resource ledger in append order."""

        return ResourceLedgerReader(self.run_dir).load_records()

    def load_snapshot(self) -> ExperimentArtifactReadSnapshot:
        """Load the complete canonical run view from durable artifacts.

        This is the shared read boundary that future CLI commands should use
        before deciding whether a run can resume, be inspected, be scored, or
        be cleaned up. It deliberately omits legacy dashboard/progress files so
        callers do not mix canonical state with compatibility projections.
        """

        record = self.load_record()
        return ExperimentArtifactReadSnapshot(
            run_dir=self.run_dir,
            spec=self.load_spec(),
            plan=self.load_plan(),
            record=record,
            artifact_index=self.load_artifact_index(),
            trial_records=tuple(
                self.load_trial_record(trial_ref.record_ref)
                for trial_ref in record.trials.records
            ),
            events=self.load_events(record.event_log_ref),
            trial_events={
                trial_ref.trial_id: self.load_trial_events(trial_ref.trial_id)
                for trial_ref in record.trials.records
            },
            resources=self.load_resources(),
        )

    def try_load_snapshot(self) -> ExperimentArtifactReadSnapshot | None:
        """Best-effort variant of :meth:`load_snapshot` for compatibility paths.

        Consumers such as the web inspector, ``cage score``, and ``cage gc``
        often need to prefer canonical artifacts while still accepting
        historical or partially written run directories. Centralizing the
        "return ``None`` on unreadable snapshot" policy here keeps those callers
        from each defining slightly different exception handling around the
        canonical read boundary.
        """

        try:
            return self.load_snapshot()
        except Exception:
            return None

    def _normalize_artifact_ref_path(self, path: str | Path) -> str:
        """Return a safe run-relative POSIX path for an artifact reference."""

        ref_path = Path(path)
        if ref_path.is_absolute():
            try:
                ref_path = ref_path.resolve().relative_to(self.run_dir)
            except ValueError as exc:
                raise ValueError(
                    f"artifact path is outside run directory: {path}"
                ) from exc
        if any(part == ".." for part in ref_path.parts):
            raise ValueError(f"artifact path escapes run directory: {path}")
        return ref_path.as_posix()
