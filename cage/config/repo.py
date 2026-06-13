"""Repo-local Cage configuration.

Only ``./config/cage.yml`` is loaded here. User-level configuration is
intentionally out of scope so a repository can be made public-safe without
surprising cross-project defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cage.contracts.env import expand_env_refs


@dataclass(frozen=True)
class WebInspectorAuthConfig:
    """Authentication settings for ``cage inspect``."""

    enabled: bool = False
    token: str = ""


@dataclass(frozen=True)
class WebInspectorUIConfig:
    """Display defaults for the web inspector."""

    run_filters_open: bool = True
    trial_filters_open: bool = True
    default_min_run_duration_ms: int = 0
    default_min_trial_duration_ms: int = 0


@dataclass(frozen=True)
class WebInspectorConfig:
    """Default settings for the web inspector."""

    host: str = "0.0.0.0"
    # Single shared inspector port. A concrete default (not 0/ephemeral) so a
    # managed board started by `cage run` from a directory without a local
    # config/cage.yml still lands on the one well-known port instead of a random
    # free one.
    port: int = 7777
    open_browser: bool = True
    ui: WebInspectorUIConfig = field(default_factory=WebInspectorUIConfig)
    auth: WebInspectorAuthConfig = field(default_factory=WebInspectorAuthConfig)
    board: "WebInspectorBoardConfig" = field(default_factory=lambda: WebInspectorBoardConfig())


@dataclass(frozen=True)
class WebInspectorBoardConfig:
    """Managed inspector board behavior for ``cage run``."""

    enabled: bool = True
    auto_start_on_run: bool = True


@dataclass(frozen=True)
class CageConfig:
    """Repo-local Cage configuration loaded from ``config/cage.yml``."""

    models_file: str = "config/models.yml"
    web_inspector: WebInspectorConfig = field(default_factory=WebInspectorConfig)


def load_repo_config(repo_root: str | Path | None = None) -> CageConfig:
    """Load ``config/cage.yml`` from *repo_root* if present.

    Missing config is not an error; defaults keep existing CLI behavior.
    Strings may reference environment variables with ``${NAME}``.
    """
    root = Path(repo_root or Path.cwd())
    path = root / "config" / "cage.yml"
    if not path.exists():
        return CageConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    expanded = expand_env_refs(raw)
    return _parse_repo_config(expanded, source=path)


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """Find the nearest ancestor containing ``config/cage.yml``."""
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "config" / "cage.yml").is_file():
            return candidate
    return None


def resolve_models_file(path: str | Path | None = None, *, repo_root: str | Path | None = None) -> Path:
    """Resolve a model registry path using repo config defaults."""
    root = Path(repo_root).resolve() if repo_root is not None else (find_repo_root() or Path.cwd()).resolve()
    raw = path if path is not None and str(path).strip() else load_repo_config(root).models_file
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _parse_repo_config(raw: dict[str, Any], *, source: Path) -> CageConfig:
    models_file = str(raw.get("models_file", "config/models.yml") or "config/models.yml")
    web_raw = raw.get("web_inspector", {})
    if web_raw is None:
        web_raw = {}
    if not isinstance(web_raw, dict):
        raise ValueError(f"{source}: web_inspector must be a mapping")

    auth_raw = web_raw.get("auth", {})
    if auth_raw is None:
        auth_raw = {}
    if not isinstance(auth_raw, dict):
        raise ValueError(f"{source}: web_inspector.auth must be a mapping")
    ui_raw = web_raw.get("ui", {})
    if ui_raw is None:
        ui_raw = {}
    if not isinstance(ui_raw, dict):
        raise ValueError(f"{source}: web_inspector.ui must be a mapping")
    board_raw = web_raw.get("board", {})
    if board_raw is None:
        board_raw = {}
    if not isinstance(board_raw, dict):
        raise ValueError(f"{source}: web_inspector.board must be a mapping")

    auth = WebInspectorAuthConfig(
        enabled=bool(auth_raw.get("enabled", False)),
        token=str(auth_raw.get("token", "") or ""),
    )
    ui = WebInspectorUIConfig(
        run_filters_open=bool(ui_raw.get("run_filters_open", True)),
        trial_filters_open=bool(ui_raw.get("trial_filters_open", True)),
        default_min_run_duration_ms=max(
            0, int(ui_raw.get("default_min_run_duration_ms", 0) or 0),
        ),
        default_min_trial_duration_ms=max(
            0, int(ui_raw.get("default_min_trial_duration_ms", 0) or 0),
        ),
    )
    board = WebInspectorBoardConfig(
        enabled=bool(board_raw.get("enabled", True)),
        auto_start_on_run=bool(board_raw.get("auto_start_on_run", True)),
    )
    web = WebInspectorConfig(
        host=str(web_raw.get("host", "0.0.0.0") or "0.0.0.0"),
        port=int(web_raw.get("port", 7777) or 7777),
        open_browser=bool(web_raw.get("open_browser", True)),
        ui=ui,
        auth=auth,
        board=board,
    )
    return CageConfig(models_file=models_file, web_inspector=web)
