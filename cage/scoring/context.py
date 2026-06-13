"""Context reconstruction for live, final, and offline scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cage.agents.base.output import normalize_recorded_output
from cage.artifacts.live_success import load_live_success

if TYPE_CHECKING:
    from cage.experiment.model import TrialRecord


@dataclass
class ScoringContext:
    """All data available for scoring a single trial."""

    trial_id: str = ""
    trial_index: int = 0
    sample: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    exit_code: int = 0
    trial_dir: Path | None = None
    run_dir: Path | None = None
    canonical_trial_id: str = ""
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    record_metadata: dict[str, Any] = field(default_factory=dict)
    live_payload: str = ""

    @classmethod
    def from_trial_dir(cls, trial_dir: Path) -> ScoringContext | None:
        """Build a context from a saved trial directory."""

        output_path = trial_dir / "task_output.json"
        if not output_path.is_file():
            return None
        return cls._from_task_output_path(output_path, fallback_trial_id=trial_dir.name)

    @classmethod
    def from_trial_record(
        cls,
        run_dir: str | Path,
        trial_record: TrialRecord,
    ) -> ScoringContext | None:
        """Build a context from canonical ``TrialRecord`` artifact refs."""

        root = Path(run_dir).expanduser().resolve()
        artifact_paths: dict[str, Path] = {}
        for artifact in trial_record.artifacts:
            if artifact.kind not in {
                "task_output",
                "prompt",
                "proxy_log",
                "proxy_jsonl",
                "final_evidence",
                "live_evidence",
                "judgment",
            }:
                continue
            artifact_path = cls._resolve_indexed_trial_artifact(root, artifact)
            if not artifact_path.is_file():
                continue
            artifact_paths.setdefault(artifact.kind, artifact_path)
        output_path = artifact_paths.get("task_output")
        if output_path is not None:
            return cls._from_task_output_path(
                output_path,
                fallback_trial_id=trial_record.trial_id,
                run_dir=root,
                canonical_trial_id=trial_record.trial_id,
                artifact_paths=artifact_paths,
                record_metadata=cls._metadata_from_trial_record(trial_record),
            )
        return None

    @staticmethod
    def _resolve_indexed_trial_artifact(root: Path, artifact: Any) -> Path:
        """Resolve one TrialRecord artifact through the canonical index."""

        from cage.artifacts.reader import ExperimentArtifactReader

        reader = ExperimentArtifactReader(root)
        try:
            indexed = reader.find_artifact(
                artifact_id=artifact.artifact_id,
                path=artifact.path,
                kind=artifact.kind,
            )
            if indexed is None:
                return Path()
            return reader.resolve_artifact_path(indexed)
        except (FileNotFoundError, KeyError, OSError, ValueError):
            return Path()

    @staticmethod
    def _metadata_from_trial_record(trial_record: TrialRecord) -> dict[str, Any]:
        """Project canonical trial record fields into scorer metadata."""

        metadata: dict[str, Any] = {
            "trial_id": trial_record.trial_id,
            "run_id": trial_record.run_id,
            "status": trial_record.status,
            "status_reason": trial_record.status_reason,
            "subject_id": trial_record.subject_id,
            "task_id": trial_record.task_id,
            "pass_index": trial_record.pass_index,
        }
        if trial_record.started_at:
            metadata["started_at"] = trial_record.started_at
        if trial_record.completed_at:
            metadata["completed_at"] = trial_record.completed_at
        if trial_record.termination.reason:
            metadata["termination_reason"] = trial_record.termination.reason
        if trial_record.termination.signal:
            metadata["termination_signal"] = trial_record.termination.signal
        if trial_record.termination.exit_code is not None:
            metadata["exit_code"] = trial_record.termination.exit_code
        return metadata

    @classmethod
    def _from_task_output_path(
        cls,
        output_path: Path,
        *,
        fallback_trial_id: str,
        run_dir: Path | None = None,
        canonical_trial_id: str = "",
        artifact_paths: dict[str, Path] | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> ScoringContext:
        """Build a scorer context from a saved ``task_output.json`` file."""

        trial_dir = output_path.parent
        data = json.loads(output_path.read_text(encoding="utf-8"))
        output = normalize_recorded_output(data.get("output", "") or "")

        trial_id = canonical_trial_id
        if not trial_id:
            meta_path = trial_dir / "meta.json"
            if meta_path.is_file():
                try:
                    trial_id = str(
                        json.loads(meta_path.read_text(encoding="utf-8")).get("trial_id")
                        or ""
                    )
                except (OSError, json.JSONDecodeError):
                    pass
        if not trial_id:
            trial_id = str(data.get("trial_id") or "") or fallback_trial_id

        return cls(
            trial_id=trial_id,
            trial_index=int(data.get("trial_index", 0) or 0),
            sample=data.get("sample", {}) or {},
            output=output,
            exit_code=int(data.get("exit_code", 0) or 0),
            trial_dir=trial_dir,
            run_dir=run_dir,
            canonical_trial_id=canonical_trial_id,
            artifact_paths=dict(artifact_paths or {}),
            record_metadata=dict(record_metadata or {}),
        )

    @cached_property
    def prompt(self) -> str:
        """Prompt text associated with this trial, if available."""

        if self.trial_dir is None:
            return ""
        p = self.artifact_paths.get("prompt") or self.trial_dir / "prompt.txt"
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    @cached_property
    def proxy_log(self) -> list[dict[str, Any]]:
        """Raw proxy JSONL entries associated with this trial."""

        if self.trial_dir is None:
            return []
        p = (
            self.artifact_paths.get("proxy_log")
            or self.artifact_paths.get("proxy_jsonl")
            or self.trial_dir / "proxy" / "proxy.jsonl"
        )
        if not p.is_file():
            return []
        entries: list[dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    @cached_property
    def live_success(self) -> dict[str, Any] | None:
        """The live-success verdict written by mid-trial monitors, if any."""

        canonical = self.artifact_paths.get("live_evidence")
        if canonical is not None:
            try:
                payload = json.loads(canonical.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(payload, dict) and payload.get("success") is True:
                return payload
            return None
        if self.trial_dir is None:
            return None
        return load_live_success(self.trial_dir)

    @cached_property
    def check_done_output(self) -> str:
        """Raw text response from the benchmark's check-done endpoint."""

        if self.live_payload:
            return self.live_payload
        if self.trial_dir is None:
            return ""
        p = (
            self.artifact_paths.get("final_evidence")
            or self.trial_dir / "runtime" / "check_done_output.txt"
        )
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    @cached_property
    def metadata(self) -> dict[str, Any]:
        """Legacy and canonical metadata associated with this trial."""

        if self.record_metadata:
            return dict(self.record_metadata)
        if self.trial_dir is None:
            return {}
        p = self.trial_dir / "meta.json"
        if not p.is_file():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
