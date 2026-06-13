"""Benchmark source adapters: normalize a dataset into launchable challenges.

The adapter ABC, its registry, and discovery helpers that let the target
server turn a benchmark's dataset into ``NormalizedChallenge`` launch specs.
"""

from cage.target.adapters.base import (
    BenchmarkAdapter,
    BenchmarkSource,
    LaunchSpec,
    NormalizedChallenge,
)
from cage.target.adapters.challenge_json import ChallengeJsonAdapter
from cage.target.adapters.registry import BenchmarkAdapterRegistry
from cage.target.adapters.source_config import build_default_registry

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkAdapterRegistry",
    "BenchmarkSource",
    "ChallengeJsonAdapter",
    "LaunchSpec",
    "NormalizedChallenge",
    "build_default_registry",
]
