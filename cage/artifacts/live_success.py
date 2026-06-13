"""Live-success verdict artifact (``trials/<id>/runtime/live_success.json``).

Read/write helpers for the per-trial live-success verdict file. This is pure
persistence with no cage dependencies, so it lives in ``artifacts`` (layer 1)
where both the runtime live monitor (which writes it) and the scorer (which
reads it) can import *down* — neither layer reaches up into the other.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LIVE_SUCCESS_REL_PATH = Path("runtime") / "live_success.json"


def live_success_path(trial_dir: Path) -> Path:
    """Return the live-success verdict path for a trial directory."""
    return trial_dir / LIVE_SUCCESS_REL_PATH


def load_live_success(trial_dir: Path | None) -> dict[str, Any] | None:
    """Load a successful live verdict, if one exists."""
    if trial_dir is None:
        return None
    path = live_success_path(Path(trial_dir))
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("success") is True:
        return payload
    return None


def record_live_success(
    *,
    trial_dir: Path,
    trial_id: str,
    benchmark: str,
    mode: str,
    source: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a live-success verdict without storing plaintext answers."""
    verdict = {
        "success": True,
        "mode": mode,
        "source": source,
        "benchmark": benchmark,
        "trial_id": trial_id,
        "ts_ms": int(time.time() * 1000),
        "evidence": evidence or {},
    }
    path = live_success_path(trial_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return verdict


def parse_live_checks_success(line: str) -> dict[str, Any] | None:
    """Return a submit/check JSONL entry only when it is explicitly correct."""
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(entry, dict) or entry.get("correct") is not True:
        return None
    return entry
