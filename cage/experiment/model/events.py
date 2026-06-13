"""Append-only lifecycle events and resource-ledger records."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from cage.experiment.model._serde import _json_ready


@dataclass(frozen=True)
class TrialEvent:
    """Append-only lifecycle event for one trial.

    Trial records summarize the latest state. Trial events explain how the
    trial reached that state without forcing inspect, resume, or gc to infer
    lifecycle history from unrelated artifact files.
    """

    schema_version: str
    event_id: str
    run_id: str
    trial_id: str
    phase: str
    type: str
    timestamp: str
    subject_id: str = ""
    task_id: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    resource_refs: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class ResourceRecord:
    """Append-only cleanup ledger entry for one runtime resource.

    Resource records are intentionally facts, not cleanup actions being executed
    immediately. A resource may appear several times in ``resources.jsonl`` as
    its status changes from created to started to released. ``gc`` and inspect
    should read the latest record for each ``resource_id``.
    """

    schema_version: str
    record_id: str
    run_id: str
    resource_id: str
    kind: str
    provider: str
    external_id: str
    status: str
    cleanup_action: str
    timestamp: str
    trial_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cleanup_error: str | None = None


def trial_event_to_mapping(event: TrialEvent) -> dict[str, Any]:
    """Return a JSON-ready mapping for a ``TrialEvent``."""

    return _json_ready(event)


def trial_event_to_json(event: TrialEvent, *, indent: int | None = None) -> str:
    """Serialize a ``TrialEvent`` as one JSONL-compatible object."""

    return (
        json.dumps(
            trial_event_to_mapping(event),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )


def resource_record_to_mapping(record: ResourceRecord) -> dict[str, Any]:
    """Return a JSON-ready mapping for a ``ResourceRecord``."""

    return _json_ready(record)


def resource_record_to_json(
    record: ResourceRecord,
    *,
    indent: int | None = None,
) -> str:
    """Serialize a ``ResourceRecord`` as one JSONL-compatible object."""

    return (
        json.dumps(
            resource_record_to_mapping(record),
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    )
