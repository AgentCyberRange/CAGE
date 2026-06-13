from __future__ import annotations

from pathlib import Path
from typing import Any

from cage.target.adapters.base import BenchmarkAdapter, BenchmarkSource, NormalizedChallenge


class BenchmarkAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BenchmarkAdapter] = {}

    def register(self, adapter: BenchmarkAdapter) -> None:
        self._adapters[adapter.adapter_kind] = adapter

    def get(self, adapter_kind: str) -> BenchmarkAdapter:
        try:
            return self._adapters[adapter_kind]
        except KeyError as exc:
            raise KeyError(f"Unknown benchmark adapter: {adapter_kind}") from exc

    def discover_all(self, sources: list[dict[str, Any] | BenchmarkSource]) -> dict[str, NormalizedChallenge]:
        challenges: dict[str, NormalizedChallenge] = {}
        for raw_source in sources:
            source = self._coerce_source(raw_source)
            adapter = self.get(source.adapter_kind)
            discovered = adapter.discover(source)
            overlap = set(challenges).intersection(discovered)
            if overlap:
                duplicate_ids = ", ".join(sorted(overlap))
                raise ValueError(f"Duplicate benchmark challenge ids discovered: {duplicate_ids}")
            challenges.update(discovered)
        return challenges

    def _coerce_source(self, raw_source: dict[str, Any] | BenchmarkSource) -> BenchmarkSource:
        if isinstance(raw_source, BenchmarkSource):
            return raw_source

        root = Path(raw_source["root"]).resolve()
        options = {
            key: value
            for key, value in raw_source.items()
            if key not in {"adapter_kind", "root"}
        }
        return BenchmarkSource(
            adapter_kind=raw_source["adapter_kind"],
            root=root,
            options=options,
        )
