"""Environment-reference expansion for loaded configuration values.

``expand_env_refs`` substitutes ``${NAME}`` references against the process
environment. It lives in ``contracts`` so both ``config`` (which loads YAML) and
``models`` (which loads the model registry) can use it without importing each
other — keeping the configuration packages acyclic.
"""

from __future__ import annotations

import os
import re
from typing import Any

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env_refs(value: Any) -> Any:
    """Recursively expand ``${NAME}`` environment references in YAML values."""
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_refs(item) for key, item in value.items()}
    return value
