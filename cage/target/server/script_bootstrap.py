from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path(script_file: str | Path) -> Path:
    repo_root = Path(script_file).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
