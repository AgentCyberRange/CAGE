"""Canonical ResourceLedger summary projections.

The ResourceLedger is append-only: a container, network, target runtime, or
proxy process can appear multiple times as it moves from ``started`` to
``released`` or ``cleanup_failed``. Human-facing surfaces should not each
reinterpret that log independently. This module provides the shared read model
for inspect, GC reports, and future dashboard/resource views.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from cage.artifacts.resources import ResourceLedgerReader
from cage.experiment.model import ResourceRecord

_INACTIVE_RESOURCE_STATUSES = frozenset({"released", "cleanup_failed"})


@dataclass(frozen=True)
class ResourceSummaryItem:
    """One latest ResourceLedger record in presentation-safe form."""

    resource_id: str
    kind: str
    provider: str
    external_id: str
    status: str
    trial_id: str
    cleanup_error: str

    @classmethod
    def from_record(cls, record: ResourceRecord) -> "ResourceSummaryItem":
        """Project one ``ResourceRecord`` into a stable JSON-friendly item."""

        return cls(
            resource_id=record.resource_id,
            kind=record.kind,
            provider=record.provider,
            external_id=record.external_id,
            status=record.status,
            trial_id=record.trial_id,
            cleanup_error=record.cleanup_error or "",
        )

    def to_mapping(self) -> dict[str, str]:
        """Return the item shape exposed to web/debug JSON consumers."""

        return {
            "resource_id": self.resource_id,
            "kind": self.kind,
            "provider": self.provider,
            "external_id": self.external_id,
            "status": self.status,
            "trial_id": self.trial_id,
            "cleanup_error": self.cleanup_error,
        }


@dataclass(frozen=True)
class ResourceLedgerSummary:
    """Latest-state ResourceLedger projection for one run."""

    items: tuple[ResourceSummaryItem, ...]

    @property
    def total(self) -> int:
        """Number of latest resources represented in the summary."""

        return len(self.items)

    @property
    def active(self) -> int:
        """Number of resources whose latest status still needs attention."""

        return sum(
            1
            for item in self.items
            if item.status not in _INACTIVE_RESOURCE_STATUSES
        )

    @property
    def released(self) -> int:
        """Number of resources whose latest status is ``released``."""

        return self._status_counts().get("released", 0)

    @property
    def cleanup_failed(self) -> int:
        """Number of resources whose latest status is ``cleanup_failed``."""

        return self._status_counts().get("cleanup_failed", 0)

    def to_mapping(self) -> dict[str, object]:
        """Return a deterministic JSON-ready summary mapping."""

        return {
            "total": self.total,
            "active": self.active,
            "released": self.released,
            "cleanup_failed": self.cleanup_failed,
            "by_kind": dict(sorted(self._kind_counts().items())),
            "by_status": dict(sorted(self._status_counts().items())),
            "items": [item.to_mapping() for item in self.items],
        }

    def _kind_counts(self) -> Counter[str]:
        """Count latest resources by ResourceLedger ``kind``."""

        return Counter(item.kind for item in self.items)

    def _status_counts(self) -> Counter[str]:
        """Count latest resources by ResourceLedger ``status``."""

        return Counter(item.status for item in self.items)


def summarize_resource_ledger(run_dir: str | Path) -> ResourceLedgerSummary:
    """Summarize the latest ResourceLedger state for ``run_dir``.

    The function reads only ``resources.jsonl`` through
    ``ResourceLedgerReader.latest_by_resource_id()``. It never mutates artifacts
    and deliberately ignores historical transitions after selecting the latest
    record for each resource id. Items are sorted by ``resource_id`` so web
    payloads, CLI output, and tests stay deterministic.
    """

    latest = ResourceLedgerReader(run_dir).latest_by_resource_id()
    records = sorted(latest.values(), key=lambda record: record.resource_id)
    return ResourceLedgerSummary(
        tuple(ResourceSummaryItem.from_record(record) for record in records)
    )
