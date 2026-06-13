"""Tests for the prompt rendering contract."""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from cage.benchmarks.prompt_contract import (
    ContractReport,
    PromptContractError,
    check_benchmark,
    check_sample,
    find_required_vars,
    lint_rendered,
    render_strict,
)

# ---------------------------------------------------------------- #
# Static template-variable discovery
# ---------------------------------------------------------------- #

def test_find_required_vars_top_level_names() -> None:
    src = "{{ a }} {{ b }} {% for x in items %}{{ x }}{% endfor %}"
    # ``x`` is loop-local; not a free variable.
    assert find_required_vars(src) == {"a", "b", "items"}


def test_find_required_vars_attribute_access_top_only() -> None:
    # ``foo.bar.baz`` reports ``foo`` only — StrictUndefined will catch the rest at render.
    src = "{{ foo.bar.baz }}"
    assert find_required_vars(src) == {"foo"}


# ---------------------------------------------------------------- #
# lint_rendered
# ---------------------------------------------------------------- #

def test_lint_rendered_clean_prompt_has_no_issues() -> None:
    assert lint_rendered("A perfectly normal prompt with enough content.") == []


def test_lint_rendered_flags_empty() -> None:
    issues = lint_rendered("   \n  ")
    assert issues and "empty" in issues[0]


def test_lint_rendered_flags_too_short() -> None:
    issues = lint_rendered("hi", min_chars=20)
    assert any("suspiciously short" in i for i in issues)


def test_lint_rendered_flags_template_leak() -> None:
    issues = lint_rendered("Hello {{ user }}, nice to meet you. Bonus content here.")
    assert any("template-syntax leak" in i for i in issues)


def test_lint_rendered_flags_missing_expected_substring() -> None:
    issues = lint_rendered(
        "This is long enough but lacks the marker.",
        expected_substrings=("FLAG_FORMAT",),
    )
    assert any("missing expected substring" in i for i in issues)


# ---------------------------------------------------------------- #
# render_strict — happy path + each failure mode
# ---------------------------------------------------------------- #

def test_render_strict_happy_path() -> None:
    src = "Target: {{ host }}:{{ port }}. Description: {{ description }}."
    sample = {"id": "x", "host": "10.0.0.1", "port": 8080,
              "description": "find the flag please"}
    out = render_strict(src, sample)
    assert "10.0.0.1:8080" in out
    assert "{{" not in out


def test_render_strict_raises_on_missing_variable() -> None:
    src = "Hello {{ user }}, your token is {{ token }}."
    sample = {"id": "x", "user": "alice"}  # token missing
    with pytest.raises(PromptContractError) as exc:
        render_strict(src, sample)
    assert "x" in str(exc.value)
    assert "token" in str(exc.value)


def test_render_strict_raises_on_empty_render() -> None:
    src = "{% if false %}never{% endif %}"
    sample = {"id": "x"}
    with pytest.raises(PromptContractError):
        render_strict(src, sample)


def test_render_strict_passes_through_expected_substrings() -> None:
    src = "Some text without the marker. " * 5
    sample = {"id": "x"}
    with pytest.raises(PromptContractError):
        render_strict(src, sample, expected_substrings=("MARKER_THAT_IS_NOT_THERE",))


# ---------------------------------------------------------------- #
# check_sample — diagnostic-only, never raises
# ---------------------------------------------------------------- #

def test_check_sample_diagnostic_records_all_issues() -> None:
    # When `rendered` is NOT provided, check_sample renders itself — so
    # missing template vars become a hard failure here.
    src = "{{ host }} {{ port }}"
    sample = {"id": "broken"}  # missing both vars
    report = check_sample(src, sample, min_chars=5)
    assert report.ok is False
    assert set(report.missing_vars) == {"host", "port"}
    assert report.issues  # at least one violation recorded


def test_check_sample_missing_vars_are_advisory_when_rendered_provided() -> None:
    # When build_prompt has already produced a clean rendered prompt, raw
    # sample missing a var is just informational — not a failure.
    src = "{{ user }} requests {{ resource }}."
    sample = {"id": "x", "user": "alice"}  # 'resource' missing from sample
    pre_rendered = "alice requests the shared bucket."  # filled in by build_prompt
    report = check_sample(src, sample, rendered=pre_rendered, min_chars=5)
    assert report.ok is True
    assert "resource" in report.missing_vars  # surfaced as a hint
    assert report.issues == []


def test_check_sample_clean_passes() -> None:
    src = "## Task\n{{ description }}\nGood luck."
    sample = {"id": "x", "description": "do the thing carefully and report"}
    report = check_sample(src, sample)
    assert report.ok
    assert report.missing_vars == []
    assert report.issues == []
    assert report.rendered_chars > 0


# ---------------------------------------------------------------- #
# check_benchmark — exercises the introspection contract
# ---------------------------------------------------------------- #

class _FakeBenchmark:
    """Minimal benchmark stub: yields N samples; build_prompt uses render_strict."""

    name = "fake"

    def __init__(self, samples: list[dict[str, Any]], template: str) -> None:
        self._samples = samples
        self.prompt_template_source = template

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        yield from self._samples

    def build_prompt(self, sample: dict[str, Any]) -> str:
        return render_strict(self.prompt_template_source, sample)


def test_check_benchmark_all_pass() -> None:
    bench = _FakeBenchmark(
        samples=[
            {"id": "a", "what": "find the bug in foo and write a fix"},
            {"id": "b", "what": "review the patch and check coverage thoroughly"},
        ],
        template="## Task\n{{ what }}\nReport when done.",
    )
    report: ContractReport = check_benchmark(
        bench, template_src=bench.prompt_template_source,
    )
    assert report.ok
    assert len(report.samples) == 2
    assert report.failures == []


def test_check_benchmark_flags_missing_var_in_one_sample() -> None:
    bench = _FakeBenchmark(
        samples=[
            {"id": "ok", "what": "valid sample with description"},
            {"id": "bad"},  # missing 'what'
        ],
        template="## Task\n{{ what }}\nReport when done.",
    )
    report = check_benchmark(
        bench, template_src=bench.prompt_template_source,
    )
    assert not report.ok
    by_id = {s.sample_id: s for s in report.samples}
    assert by_id["ok"].ok is True
    assert by_id["bad"].ok is False


def test_check_benchmark_respects_sample_limit() -> None:
    bench = _FakeBenchmark(
        samples=[
            {"id": str(i), "what": f"description {i}"} for i in range(10)
        ],
        template="{{ what }} (long enough to pass lint).",
    )
    report = check_benchmark(
        bench, template_src=bench.prompt_template_source, sample_limit=3,
    )
    assert len(report.samples) == 3


def test_check_benchmark_respects_sample_id() -> None:
    bench = _FakeBenchmark(
        samples=[
            {"id": "a", "what": "alpha task with enough text"},
            {"id": "b", "what": "beta task with enough text"},
            {"id": "c", "what": "gamma task with enough text"},
        ],
        template="{{ what }}",
    )
    report = check_benchmark(
        bench, template_src=bench.prompt_template_source, sample_id="b",
    )
    assert len(report.samples) == 1
    assert report.samples[0].sample_id == "b"
