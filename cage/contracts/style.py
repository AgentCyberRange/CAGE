"""Tiny ANSI styling primitives for Cage terminal output.

Dependency-free presentation primitives shared by anything that writes to a
terminal — the CLI progress reporter and the runtime preflight gate. They sit on
the layer-0 floor so a lower layer (preflight, L3) can colourise its own output
without importing up into the CLI (L7).
"""

from __future__ import annotations

import os
from typing import Any

_RESET = "\033[0m"
_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def should_color(stream: Any | None = None) -> bool:
    value = os.getenv("CAGE_COLOR", "").strip().lower()
    if value in {"1", "always", "true", "yes", "on"}:
        return True
    if value in {"0", "never", "false", "no", "off"} or os.getenv("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def style(text: str, *names: str, enabled: bool = True) -> str:
    if not enabled or not text:
        return text
    codes = [_CODES[name] for name in names if name in _CODES]
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}{_RESET}"
