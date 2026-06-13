"""AgentState — snapshot and diff of agent state between trials.

State is defined by the AgentType's state_paths. Each path is relative
to the agent's home directory in the container.
"""

from __future__ import annotations

import filecmp
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cage.sandbox.containers import Container

logger = logging.getLogger(__name__)
STATE_TRANSFER_TIMEOUT_SECONDS = 5000.0


@dataclass(frozen=True)
class StateSnapshot:
    """A snapshot of agent state at a point in time.

    ``failed_paths`` lists the declared ``state_paths`` that we tried to
    copy out of the container but couldn't. The classic case: the container
    was force-removed mid-trial (``docker rm -f`` by the SIGINT handler)
    and ``docker cp`` returns ``RWLayer of container ... is unexpectedly
    nil`` / ``No such container``. The orchestrator uses this as a
    structural "container disappeared" signal when classifying
    termination.
    """

    snapshot_dir: Path
    state_paths: tuple[str, ...]
    timestamp_ms: int
    failed_paths: tuple[str, ...] = ()

    @property
    def has_failures(self) -> bool:
        return bool(self.failed_paths)


@dataclass(frozen=True)
class StateDiff:
    """Diff between two state snapshots."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"+{len(self.added)} added")
        if self.modified:
            parts.append(f"~{len(self.modified)} modified")
        if self.deleted:
            parts.append(f"-{len(self.deleted)} deleted")
        return ", ".join(parts) if parts else "no changes"


def snapshot_state(
    container: Container,
    *,
    state_paths: list[str],
    home_dir: str,
    output_dir: Path,
) -> StateSnapshot:
    """Take a snapshot of agent state from the container.

    Copies each state_path from the container to output_dir.
    """
    import time

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    failed: list[str] = []

    for rel_path in state_paths:
        # Absolute paths used as-is; relative paths prepended with home_dir
        if rel_path.startswith("/"):
            container_path = rel_path
        else:
            container_path = f"{home_dir}/{rel_path}"
        # For host storage, always use a relative-looking key to avoid deep nesting
        host_key = rel_path.lstrip("/")
        host_dest = output_dir / host_key

        # Check if path exists in container
        check = container.exec(f"test -e {container_path} && echo exists || echo missing")
        if "missing" in check.stdout:
            continue

        host_dest.parent.mkdir(parents=True, exist_ok=True)
        result = container.copy_from(
            container_path,
            str(host_dest),
            timeout=STATE_TRANSFER_TIMEOUT_SECONDS,
        )
        if result.exit_code != 0:
            logger.warning(
                "Failed to snapshot %s: %s", container_path, result.stderr[:200]
            )
            failed.append(rel_path)

    return StateSnapshot(
        snapshot_dir=output_dir,
        state_paths=tuple(state_paths),
        timestamp_ms=ts,
        failed_paths=tuple(failed),
    )
    logger.debug(
        "state_snapshot_completed",
        extra={"state_paths": state_paths, "snapshot_dir": str(output_dir)},
    )


def restore_state(
    container: Container,
    *,
    snapshot: StateSnapshot,
    home_dir: str,
) -> None:
    """Restore agent state from a snapshot into the container."""
    for rel_path in snapshot.state_paths:
        host_key = rel_path.lstrip("/")
        host_src = snapshot.snapshot_dir / host_key
        if not host_src.exists():
            continue

        if rel_path.startswith("/"):
            container_path = rel_path
        else:
            container_path = f"{home_dir}/{rel_path}"
        # Ensure parent exists
        parent = "/".join(container_path.split("/")[:-1])
        container.exec(f"mkdir -p {parent}")

        result = container.copy_to(
            str(host_src),
            container_path,
            timeout=STATE_TRANSFER_TIMEOUT_SECONDS,
        )
        if result.exit_code != 0:
            logger.warning(
                "Failed to restore %s: %s", container_path, result.stderr[:200]
            )


def reset_state(
    container: Container,
    *,
    state_paths: list[str],
    home_dir: str,
) -> None:
    """Clear agent state paths in the container."""
    for rel_path in state_paths:
        if rel_path.startswith("/"):
            container_path = rel_path
        else:
            container_path = f"{home_dir}/{rel_path}"
        container.exec(f"rm -rf {container_path}")


def diff_snapshots(pre: StateSnapshot, post: StateSnapshot) -> StateDiff:
    """Compare two state snapshots and return the diff."""
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []

    pre_files = _list_files(pre.snapshot_dir) if pre.snapshot_dir.exists() else set()
    post_files = _list_files(post.snapshot_dir) if post.snapshot_dir.exists() else set()

    for f in post_files - pre_files:
        added.append(f)

    for f in pre_files - post_files:
        deleted.append(f)

    for f in pre_files & post_files:
        pre_path = pre.snapshot_dir / f
        post_path = post.snapshot_dir / f
        if pre_path.is_file() and post_path.is_file():
            if not filecmp.cmp(str(pre_path), str(post_path), shallow=False):
                modified.append(f)

    return StateDiff(
        added=sorted(added),
        modified=sorted(modified),
        deleted=sorted(deleted),
    )
    logger.debug(
        "state_diff",
        extra={"added": len(added), "modified": len(modified), "deleted": len(deleted)},
    )


def _list_files(root: Path) -> set[str]:
    """List all files under a directory as relative paths."""
    if not root.exists():
        return set()
    files: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file():
            files.add(str(p.relative_to(root)))
    return files
