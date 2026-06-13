from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_sys_path(script_path: str | Path) -> Path:
    path = Path(script_path).resolve()
    repo_root = path.parent.parent.parent
    repo_root_str = str(repo_root)
    if sys.path and sys.path[0] == repo_root_str:
        return repo_root
    try:
        sys.path.remove(repo_root_str)
    except ValueError:
        pass
    sys.path.insert(0, repo_root_str)
    return repo_root
