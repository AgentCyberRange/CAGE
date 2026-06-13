"""Small JSON file read/write helpers for run artifacts.

Best-effort readers/writers used across the run-recording and dashboard code:
``_load_json_file`` returns ``{}`` on any error (missing/corrupt file), and
``_write_json_file`` creates parent dirs and writes pretty UTF-8 JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
