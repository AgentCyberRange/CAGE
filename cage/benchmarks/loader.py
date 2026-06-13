"""Load a concrete ``Benchmark`` subclass from a Python module file.

Lives in the benchmarks layer (not config) so benchmark discovery never has to
reach up into the experiment-config loader — keeps ``benchmarks`` below
``config`` in the dependency graph.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from cage.benchmarks.base import Benchmark


def load_benchmark_from_module(
    module_path: Path,
    class_name: str | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Benchmark:
    """Import a Python module and instantiate the Benchmark subclass."""
    spec = importlib.util.spec_from_file_location("_cage_benchmark", str(module_path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec.loader.exec_module(mod)

    init_kwargs = kwargs or {}

    if class_name:
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ValueError(f"Class '{class_name}' not found in {module_path}")
        return cls(**init_kwargs)

    # Auto-discover: find Benchmark subclasses
    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, Benchmark) and obj is not Benchmark:
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0](**init_kwargs)
    elif len(candidates) == 0:
        raise ValueError(f"No Benchmark subclass found in {module_path}")
    else:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Multiple Benchmark subclasses in {module_path}: {names}. "
            f"Specify 'class' in config."
        )
