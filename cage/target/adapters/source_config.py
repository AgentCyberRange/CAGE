"""Default adapter registry for target_server.

The framework only registers the generic ``ChallengeJsonAdapter``. Benchmark-
specific adapters (e.g. a compose-stack launcher) live under
``examples/<name>/target_server_adapter.py`` and are loaded at startup via the
``TARGET_SERVER_ADAPTER_MODULES`` env var (comma-separated
``path/to/module.py:ClassName`` specs).

Spec format
-----------
- ``path:ClassName`` — instantiate ``ClassName()`` from the given file.
- ``path``           — module must expose a top-level ``ADAPTER`` instance.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from cage.target.adapters.base import BenchmarkAdapter
from cage.target.adapters.challenge_json import ChallengeJsonAdapter
from cage.target.adapters.registry import BenchmarkAdapterRegistry
from cage.target.adapters.roots import normalize_benchmark_sources, resolve_configured_benchmark_root


def load_adapter_from_path(spec: str) -> BenchmarkAdapter:
    """Load a ``BenchmarkAdapter`` from a ``path/to/module.py[:ClassName]`` spec."""
    spec = spec.strip()
    if ":" in spec:
        path_str, class_name = spec.rsplit(":", 1)
    else:
        path_str, class_name = spec, None

    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"target_server adapter module not found: {path}")

    module_name = f"_cage_target_server_adapter_{abs(hash(str(path)))}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot load target_server adapter module: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

    if class_name:
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(
                f"Adapter class '{class_name}' not found in {path}"
            ) from exc
        return cls()

    try:
        return module.ADAPTER  # type: ignore[no-any-return]
    except AttributeError as exc:
        raise ImportError(
            f"Adapter module {path} must export a top-level ADAPTER instance "
            f"(or be specified as '{path}:ClassName')"
        ) from exc


def _load_env_adapters() -> list[BenchmarkAdapter]:
    raw = os.environ.get("TARGET_SERVER_ADAPTER_MODULES", "").strip()
    if not raw:
        return []
    return [load_adapter_from_path(spec) for spec in raw.split(",") if spec.strip()]


def build_default_registry() -> BenchmarkAdapterRegistry:
    registry = BenchmarkAdapterRegistry()
    registry.register(ChallengeJsonAdapter())
    for adapter in _load_env_adapters():
        registry.register(adapter)
    return registry


def resolve_benchmark_sources(config: Any) -> list[dict[str, Any]]:
    configured_sources = getattr(config, "benchmark_sources", None)
    if configured_sources:
        return normalize_benchmark_sources(list(configured_sources))

    return normalize_benchmark_sources(
        [
            {
                "adapter_kind": "challenge_json",
                "root": str(resolve_configured_benchmark_root(getattr(config, "benchmark_root", "./target_server/benchmarks"))),
            }
        ]
    )
