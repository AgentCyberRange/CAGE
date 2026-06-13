from __future__ import annotations

from pathlib import Path
from typing import Any


PRIMARY_BENCHMARK_DIR = "benchmarks"
LEGACY_BENCHMARK_DIR = "benchmark"


def preferred_repo_benchmark_root(project_root: Path) -> Path:
    return (Path(project_root).resolve() / PRIMARY_BENCHMARK_DIR).resolve()


def resolve_repo_benchmark_root(project_root: Path) -> Path:
    project_root = Path(project_root).resolve()
    primary = project_root / PRIMARY_BENCHMARK_DIR
    legacy = project_root / LEGACY_BENCHMARK_DIR

    if primary.exists():
        return primary.resolve()
    if legacy.exists():
        return legacy.resolve()
    return primary.resolve()


def resolve_configured_benchmark_root(raw_root: str | Path) -> Path:
    root = Path(raw_root).expanduser()
    resolved = root.resolve()
    if resolved.name == LEGACY_BENCHMARK_DIR:
        repo_local = resolved.parent / "target_server" / PRIMARY_BENCHMARK_DIR
        if repo_local.exists():
            return repo_local.resolve()
        primary = resolved.with_name(PRIMARY_BENCHMARK_DIR)
        if primary.exists():
            return primary.resolve()

    if resolved.exists():
        return resolved

    return resolved


def normalize_benchmark_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in sources:
        item = dict(source)
        if item.get("adapter_kind") == "challenge_json" and item.get("root"):
            item["root"] = str(resolve_configured_benchmark_root(item["root"]))
        normalized.append(item)
    return normalized
