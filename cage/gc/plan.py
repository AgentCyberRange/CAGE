"""ResourceLedger cleanup planning.

The ResourceLedger records every runtime resource transition. Cleanup code
needs a narrower view: latest resources that still matter, grouped into the
Docker buckets Cage can sweep explicitly. Keeping that selection in core avoids
each consumer inventing subtly different definitions of "needs cleanup".
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from cage.experiment.model import ResourceRecord

COUNT_KEYS = ("containers", "networks", "volumes")
RELEASED_RESOURCE_STATUSES = frozenset({"released"})
RESOURCE_KIND_TO_COUNT_KEY = {
    "container": "containers",
    "docker_container": "containers",
    "network": "networks",
    "docker_network": "networks",
    "volume": "volumes",
    "docker_volume": "volumes",
}


@dataclass(frozen=True)
class ResourceCleanupPlan:
    """Latest ResourceLedger resources grouped for cleanup/reporting."""

    containers: tuple[ResourceRecord, ...] = ()
    networks: tuple[ResourceRecord, ...] = ()
    volumes: tuple[ResourceRecord, ...] = ()

    def has_resources(self) -> bool:
        """Return true when any countable resource remains in the plan."""

        return any(self.records_by_key().values())

    def counts(self) -> dict[str, int]:
        """Return GC-style counts for countable unreleased resources."""

        return {
            key: len(records)
            for key, records in self.records_by_key().items()
        }

    def docker_ids(self) -> dict[str, tuple[str, ...]]:
        """Return structured Docker identifiers safe for explicit cleanup."""

        return {
            key: tuple(record.external_id for record in records)
            for key, records in self.cleanup_records_by_key().items()
        }

    def records_by_key(self) -> dict[str, tuple[ResourceRecord, ...]]:
        """Return all countable records by GC resource bucket."""

        return {
            "containers": self.containers,
            "networks": self.networks,
            "volumes": self.volumes,
        }

    def cleanup_records_by_key(self) -> dict[str, tuple[ResourceRecord, ...]]:
        """Return deduped Docker records with concrete external ids.

        Some ledger records are useful for dry-run counts but cannot be swept
        explicitly: non-Docker providers, diagnostic target handles, or records
        missing ``external_id``. Those stay out of this cleanup view so callers
        never execute free-form ``cleanup_action`` strings.
        """

        return {
            key: _dedupe_resource_records_by_external_id(
                [
                    record
                    for record in records
                    if is_docker_cleanup_candidate(record)
                    and record.external_id.strip()
                ]
            )
            for key, records in self.records_by_key().items()
        }


def build_resource_cleanup_plan(
    records: Iterable[ResourceRecord],
    *,
    namespace: str | None = None,
) -> ResourceCleanupPlan:
    """Build a cleanup plan from latest ResourceLedger records.

    Callers should pass the latest record for each ``resource_id``. The plan
    filters released records, applies optional namespace scoping, maps countable
    Docker-style kinds into ``containers`` / ``networks`` / ``volumes``, and
    preserves record order inside each bucket.
    """

    buckets: dict[str, list[ResourceRecord]] = {key: [] for key in COUNT_KEYS}
    for record in records:
        if resource_is_released(record):
            continue
        if not resource_matches_namespace(record, namespace):
            continue
        count_key = resource_count_key(record)
        if count_key is None:
            continue
        buckets[count_key].append(record)
    return ResourceCleanupPlan(
        containers=tuple(buckets["containers"]),
        networks=tuple(buckets["networks"]),
        volumes=tuple(buckets["volumes"]),
    )


def resource_is_released(record: ResourceRecord) -> bool:
    """Return true when a latest resource status is terminal for cleanup."""

    return record.status.strip().lower() in RELEASED_RESOURCE_STATUSES


def resource_count_key(record: ResourceRecord) -> str | None:
    """Map a ResourceLedger record into a GC resource-count bucket."""

    kind = record.kind.strip().lower()
    if kind in RESOURCE_KIND_TO_COUNT_KEY:
        return RESOURCE_KIND_TO_COUNT_KEY[kind]
    prefix = record.resource_id.split(":", 1)[0].strip().lower()
    return RESOURCE_KIND_TO_COUNT_KEY.get(prefix)


def resource_matches_namespace(record: ResourceRecord, namespace: str | None) -> bool:
    """Return whether a ledger record is in scope for ``--namespace``."""

    if not namespace:
        return True
    metadata = record.metadata
    labels = metadata.get("labels") if isinstance(metadata, Mapping) else None
    if not isinstance(labels, Mapping):
        return True
    labelled_namespace = str(labels.get("cage.target.namespace") or "").strip()
    if not labelled_namespace:
        return True
    return labelled_namespace == namespace


def is_docker_cleanup_candidate(record: ResourceRecord) -> bool:
    """Return true when a record can be removed by structured Docker cleanup."""

    provider = record.provider.strip().lower()
    kind = record.kind.strip().lower()
    resource_prefix = record.resource_id.split(":", 1)[0].strip().lower()
    return (
        provider == "docker"
        or kind.startswith("docker_")
        or resource_prefix.startswith("docker_")
    )


def _dedupe_resource_records_by_external_id(
    records: Iterable[ResourceRecord],
) -> tuple[ResourceRecord, ...]:
    """Return records with unique non-empty external ids in first-seen order."""

    seen: set[str] = set()
    out: list[ResourceRecord] = []
    for record in records:
        external_id = record.external_id.strip()
        if not external_id or external_id in seen:
            continue
        seen.add(external_id)
        out.append(record)
    return tuple(out)
