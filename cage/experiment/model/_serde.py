"""Shared serialization primitives for the experiment model.

Pure helpers with no dependency on any model class — the leaf of the
``experiment.model`` package internal dependency DAG.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    """Return a deterministic id for a serializable planning payload."""

    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-sha256-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _json_ready(value: Any) -> Any:
    """Convert dataclasses, paths, mappings, and tuples into JSON-ready data."""

    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
