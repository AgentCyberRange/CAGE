"""Benchmark adapter contracts and registry-facing helper types."""

from cage.benchmarks.base import (
    Benchmark,
    BenchmarkBuildResult,
    BenchmarkBuildSummary,
    BenchmarkOption,
    normalize_sample_id,
    parse_sample_slice,
    sample_id_candidates,
    sample_id_matches,
)
from cage.benchmarks.prompt_contract import render_prompt, render_strict

__all__ = [
    "Benchmark",
    "BenchmarkBuildResult",
    "BenchmarkBuildSummary",
    "BenchmarkOption",
    "normalize_sample_id",
    "parse_sample_slice",
    "render_prompt",
    "render_strict",
    "sample_id_candidates",
    "sample_id_matches",
]
