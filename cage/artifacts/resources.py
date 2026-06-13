"""Append-only resource ledger artifacts for Cage runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from cage.experiment.model import ResourceRecord, resource_record_to_json


class ResourceLedgerWriter:
    """Append runtime resource lifecycle records to ``resources.jsonl``."""

    def __init__(self, run_dir: str | Path) -> None:
        """Create a resource-ledger writer rooted at one experiment run."""

        self.run_dir = Path(run_dir).expanduser().resolve()

    def append_resource(
        self,
        *,
        run_id: str,
        resource_id: str,
        kind: str,
        provider: str,
        external_id: str,
        status: str,
        cleanup_action: str,
        timestamp: str,
        trial_id: str = "",
        metadata: Mapping[str, object] | None = None,
        cleanup_error: str | None = None,
    ) -> ResourceRecord:
        """Append one runtime resource record and return it.

        The same ``resource_id`` can appear multiple times as status changes.
        The append-only layout keeps cleanup decisions auditable.
        """

        ledger_path = self.run_dir / "resources.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        record = ResourceRecord(
            schema_version="resource_record.v1",
            record_id=self._next_record_id(ledger_path),
            run_id=run_id,
            resource_id=resource_id,
            kind=kind,
            provider=provider,
            external_id=external_id,
            status=status,
            cleanup_action=cleanup_action,
            timestamp=timestamp,
            trial_id=trial_id,
            metadata=dict(metadata or {}),
            cleanup_error=cleanup_error,
        )
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(resource_record_to_json(record))
        return record

    def _next_record_id(self, ledger_path: Path) -> str:
        """Return the next monotonic resource-ledger record id."""

        if not ledger_path.is_file():
            return "res_000001"
        count = sum(
            1
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return f"res_{count + 1:06d}"


class ResourceLedgerReader:
    """Read canonical runtime resource ledger records for one run."""

    def __init__(self, run_dir: str | Path) -> None:
        """Create a resource-ledger reader rooted at one experiment run."""

        self.run_dir = Path(run_dir).expanduser().resolve()

    def load_records(self) -> tuple[ResourceRecord, ...]:
        """Load every valid resource record in append order."""

        ledger_path = self.run_dir / "resources.jsonl"
        if not ledger_path.is_file():
            return ()
        records: list[ResourceRecord] = []
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, Mapping):
                records.append(resource_record_from_mapping(raw))
        return tuple(records)

    def latest_by_resource_id(self) -> dict[str, ResourceRecord]:
        """Return the latest record for each resource id."""

        latest: dict[str, ResourceRecord] = {}
        for record in self.load_records():
            latest[record.resource_id] = record
        return latest


def resource_record_from_mapping(data: Mapping[str, object]) -> ResourceRecord:
    """Reconstruct one ``ResourceRecord`` from a serialized ledger line."""

    metadata = data.get("metadata")
    return ResourceRecord(
        schema_version=str(data.get("schema_version") or "resource_record.v1"),
        record_id=str(data.get("record_id") or ""),
        run_id=str(data.get("run_id") or ""),
        trial_id=str(data.get("trial_id") or ""),
        resource_id=str(data.get("resource_id") or ""),
        kind=str(data.get("kind") or ""),
        provider=str(data.get("provider") or ""),
        external_id=str(data.get("external_id") or ""),
        status=str(data.get("status") or ""),
        cleanup_action=str(data.get("cleanup_action") or ""),
        cleanup_error=(
            str(data["cleanup_error"])
            if data.get("cleanup_error") is not None
            else None
        ),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        timestamp=str(data.get("timestamp") or ""),
    )


__all__ = [
    "ResourceLedgerReader",
    "ResourceLedgerWriter",
    "resource_record_from_mapping",
]
