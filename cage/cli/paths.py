"""Path display helpers shared by terminal commands."""

from __future__ import annotations

from pathlib import Path


def display_path(path: Path) -> str:
    """Return a cwd-relative path when possible without hiding absolute context."""

    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved.is_relative_to(cwd):
        return str(resolved.relative_to(cwd))
    return str(resolved)
