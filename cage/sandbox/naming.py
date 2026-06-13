"""Naming helpers for run/trial identifiers and Docker container names.

Pure leaf utilities shared across the runtime: parsing the ``agent:model:mode``
label, sanitising identifiers into Docker-safe name fragments, abbreviating run
ids, and composing agent container names within Docker's 63-char limit. No
dependency on the trial lifecycle — safe to import from anywhere.
"""

from __future__ import annotations

import hashlib
import re

_DOCKER_NAME_MAX_LEN = 63


def _safe_docker_name_component(value: str) -> str:
    """Convert a run/trial identifier into a conservative Docker name fragment."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "unknown")).strip(".-")
    return safe or "unknown"

def _short_run_id(run_id: str) -> str:
    """Stable 8-char abbreviation of a run id (uuid8 tail or sha1 prefix)."""
    if not run_id:
        return ""
    # Auto-generated ids end in ``-<uuid8>`` already — reuse that.
    tail = run_id.rsplit("-", 1)[-1]
    if len(tail) == 8 and all(c in "0123456789abcdef" for c in tail):
        return tail
    return hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8]

def _build_agent_container_name(
    agent_dir_name: str,
    run_id: str,
    suffix: str,
) -> str:
    """Compose ``cage-<agent_dir>-<run_id>-<suffix>`` honouring docker's 63-char limit.

    Falls back to an 8-char short form of ``run_id`` when the full id would
    overflow. The full run id is still stamped on the container as the
    ``cage.run_id`` label.
    """
    safe_agent = agent_dir_name.replace(":", "-")
    if not run_id:
        return f"cage-{safe_agent}-{suffix}"
    full = f"cage-{safe_agent}-{run_id}-{suffix}"
    if len(full) <= _DOCKER_NAME_MAX_LEN:
        return full
    return f"cage-{safe_agent}-{_short_run_id(run_id)}-{suffix}"

def _parse_agent_label(label: str) -> tuple[str, str]:
    """Split agent label into (dir_name, mode).

    label = "agent_id:model_id:mode"
    dir_name = full label (used as directory name under .cage_runs/)
    mode = "stateful"|"stateless"
    """
    parts = label.split(":")
    mode = parts[-1] if len(parts) >= 3 else "stateless"
    return label, mode
