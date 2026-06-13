"""Utilities for target/challenge runtime configuration.

Helpers that normalise the ``target_scope`` / ``parallel_mode`` flags
``ChallengeClient`` passes to ``target_server``.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Mapping


# -- From utils/runtime_policy.py ---------------------------------------------

VALID_TARGET_SCOPES = {"per_challenge", "per_agent"}


def normalize_target_scope(value: Any) -> str:
    """Validate and normalize a target_scope string."""
    normalized = str(value or "").strip().lower()
    if normalized in VALID_TARGET_SCOPES:
        return normalized
    return ""


def resolve_target_scope(
    chal_data: Mapping[str, Any] | None = None,
    runtime_args: Mapping[str, Any] | None = None,
) -> str:
    """Resolve target scope from runtime args / challenge metadata, then default.

    Resolution order:
      1. ``runtime_args["target_scope"]`` (per-launch override from the
         orchestrator / client)
      2. ``chal_data["target_scope"]`` (declared at challenge.json level —
         lets a benchmark whose challenges need parallel-friendly per-agent
         instances express that without the framework knowing the family
         name).
      3. ``per_challenge`` default — safe for the common case where a
         compose stack is reused across passes of the same agent.
    """
    requested = normalize_target_scope((runtime_args or {}).get("target_scope"))
    if requested:
        return requested
    declared = normalize_target_scope((chal_data or {}).get("target_scope"))
    if declared:
        return declared
    return "per_challenge"


def should_auto_init_target(
    chal_data: Mapping[str, Any] | None = None,
    runtime_args: Mapping[str, Any] | None = None,
) -> bool:
    return resolve_target_scope(chal_data=chal_data, runtime_args=runtime_args) != "per_agent"


# -- From utils/container_paths.py -------------------------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_OPAQUE_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def sanitize_container_path_token(value: str) -> str:
    """Normalize a string for use as a container path segment."""
    normalized = _NON_ALNUM_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return normalized or "root"


def opaque_token(value: str) -> str:
    """Deterministic opaque UUID from any string — hides semantics like CVE IDs."""
    return str(uuid.uuid5(_OPAQUE_NAMESPACE, str(value or "")))[:12]