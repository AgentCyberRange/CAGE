"""The benchmark prompt contract — render a sample into a validated prompt.

A benchmark turns a sample into the prompt the agent sees. This module owns
that, end to end:

  - **rendering** — ``render_prompt`` (load ``<type>.j2`` / legacy
    ``<type>_template.txt`` from a directory) and ``render_strict`` (render a
    template *string*), both behind one cached Jinja environment factory;
  - **the contract** — every rendered prompt is linted, so an empty prompt or
    one that still contains Jinja syntax (``{{ }}`` / ``{% %}``) raises
    :class:`PromptContractError` *before* it reaches the agent, instead of the
    model being blamed for a half-rendered prompt;
  - **the pre-flight** — ``check_benchmark`` runs ``build_prompt`` over a
    benchmark's samples and collects a non-raising :class:`ContractReport`; this
    is what ``cage benchmark check`` reports.

``render_strict`` uses ``StrictUndefined`` (a missing variable raises).
``render_prompt`` keeps the permissive default undefined because example
templates lean on the ``{% if optional_field %}`` idiom (which ``StrictUndefined``
turns into an error) — it still lints the output.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Undefined, meta
from jinja2.exceptions import TemplateError, UndefinedError

# Substrings that signal the rendered prompt still contains template syntax.
_LEAK_MARKERS = ("{{", "}}", "{%", "%}")

# Default lower bound for "looks like a real prompt".
_DEFAULT_MIN_CHARS = 20


class PromptContractError(ValueError):
    """Raised when a prompt fails the rendering contract."""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _make_env(loader: FileSystemLoader | None = None, *, strict: bool = True) -> Environment:
    """The one Jinja environment factory.

    ``strict`` selects ``StrictUndefined`` (raise on any missing variable) versus
    the permissive default (missing variables are falsy/empty). No autoescape —
    these are plain-text prompts, not HTML.
    """
    return Environment(
        loader=loader,
        undefined=StrictUndefined if strict else Undefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


@functools.lru_cache(maxsize=None)
def _file_env(template_dir: str) -> Environment:
    """Cached file-loading environment (permissive: supports ``{% if optional %}``).

    Deliberately permissive: the example instance templates lean on the
    ``instance_data.optional_key or default`` idiom (optional dict-key access via
    dot syntax). ``StrictUndefined`` turns every such missing-key access into a
    render error, so a strict flip would require rewriting that idiom across all
    instance templates with explicit guards — empirically verified to break
    rendering for several existing example template sets. Benchmarks that want
    fail-fast semantics can call :func:`render_strict` instead.
    """
    return _make_env(FileSystemLoader(template_dir), strict=False)


def _template_name(template_dir: Path, template_type: str) -> str:
    preferred = f"{template_type}.j2"
    if (template_dir / preferred).is_file():
        return preferred
    return f"{template_type}_template.txt"


def find_required_vars(template_src: str) -> set[str]:
    """Return the set of top-level variable names referenced by a Jinja template.

    Note: this catches ``{{ foo }}`` and ``{% for x in foo %}`` but does not
    introspect attribute access (``foo.bar.baz`` reports ``foo``). That matches
    how StrictUndefined catches missing keys at render time.
    """
    ast = _make_env().parse(template_src)
    return set(meta.find_undeclared_variables(ast))


def lint_rendered(
    rendered: str,
    *,
    min_chars: int = _DEFAULT_MIN_CHARS,
    expected_substrings: Iterable[str] = (),
) -> list[str]:
    """Return a list of issues found in a rendered prompt. Empty list = OK."""
    issues: list[str] = []
    stripped = rendered.strip()
    if not stripped:
        issues.append("rendered prompt is empty")
        return issues
    if len(stripped) < min_chars:
        issues.append(
            f"rendered prompt is suspiciously short ({len(stripped)} chars < {min_chars})"
        )
    for marker in _LEAK_MARKERS:
        if marker in rendered:
            issues.append(f"rendered prompt contains template-syntax leak: {marker!r}")
            break
    for needle in expected_substrings:
        if needle and needle not in rendered:
            issues.append(f"rendered prompt is missing expected substring: {needle!r}")
    return issues


def _finalize(
    rendered: str,
    *,
    subject: str,
    min_chars: int,
    expected_substrings: Iterable[str],
) -> str:
    """Lint a freshly rendered prompt; raise PromptContractError on any issue."""
    issues = lint_rendered(rendered, min_chars=min_chars, expected_substrings=expected_substrings)
    if issues:
        raise PromptContractError(f"{subject} failed prompt lint: {'; '.join(issues)}")
    return rendered


def render_strict(
    template_src: str,
    sample: dict[str, Any],
    *,
    min_chars: int = _DEFAULT_MIN_CHARS,
    expected_substrings: Iterable[str] = (),
) -> str:
    """Render a template string strictly and lint the output. Raise on violation.

    This is the function benchmarks should call from ``build_prompt`` so that
    sample-data drift is caught at trial time rather than blamed on the model.
    """
    template = _make_env().from_string(template_src)
    sample_id = sample.get("id")
    try:
        rendered = template.render(**sample)
    except UndefinedError as exc:
        raise PromptContractError(
            f"sample {sample_id!r} is missing a template variable: {exc.message}"
        ) from exc
    except TemplateError as exc:
        raise PromptContractError(
            f"template error while rendering sample {sample_id!r}: {exc}"
        ) from exc
    return _finalize(
        rendered,
        subject=f"sample {sample_id!r}",
        min_chars=min_chars,
        expected_substrings=expected_substrings,
    )


def render_prompt(
    template_dir: Path | str,
    template_type: str = "instance",
    *,
    min_chars: int = _DEFAULT_MIN_CHARS,
    expected_substrings: Iterable[str] = (),
    **context: Any,
) -> str:
    """Render a prompt template from a directory, then lint the output.

    Args:
        template_dir: Directory containing ``{template_type}.j2`` or the
            legacy ``{template_type}_template.txt``.
        template_type: Template basename.
        min_chars: Minimum rendered length before the lint flags it.
        expected_substrings: Substrings that must appear in the output.
        **context: Variables passed to the Jinja2 template.

    Rendering is permissive (a missing variable is empty, so templates can use
    ``{% if optional_field %}`` and ``instance_data.optional_key or default``),
    but the output is linted. Use :func:`render_strict` for fail-fast semantics.

    Raises:
        PromptContractError: the template errors, or the rendered prompt is
            empty / leaks Jinja syntax / fails the lint pass.
    """
    template_dir = Path(template_dir)
    env = _file_env(str(template_dir))
    template_name = _template_name(template_dir, template_type)
    try:
        rendered = env.get_template(template_name).render(**context)
    except UndefinedError as exc:
        raise PromptContractError(
            f"template {template_name!r} is missing a variable: {exc.message}"
        ) from exc
    except TemplateError as exc:
        raise PromptContractError(
            f"error rendering template {template_name!r}: {exc}"
        ) from exc
    return _finalize(
        rendered,
        subject=f"template {template_name!r}",
        min_chars=min_chars,
        expected_substrings=expected_substrings,
    )


# ---------------------------------------------------------------------------
# Pre-flight check (``cage benchmark check``)
# ---------------------------------------------------------------------------


@dataclass
class SampleReport:
    """Per-sample result of a contract check.

    Failure semantics: ``ok`` is False only when ``benchmark.build_prompt`` raises
    or produces a prompt that fails the lint pass. ``missing_vars`` is *advisory*:
    template vars not present in the raw sample dict. Benchmarks often augment the
    sample inside ``build_prompt`` (injecting ``proxy_url``, run-time target info,
    etc.), so a non-empty ``missing_vars`` does not necessarily mean a real bug.
    """

    sample_id: str
    ok: bool
    missing_vars: list[str] = field(default_factory=list)  # advisory only
    issues: list[str] = field(default_factory=list)
    rendered_chars: int = 0
    rendered_preview: str = ""


@dataclass
class ContractReport:
    """Aggregate result over a set of samples."""

    samples: list[SampleReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.samples)

    @property
    def failures(self) -> list[SampleReport]:
        return [s for s in self.samples if not s.ok]


def check_sample(
    template_src: str | None,
    sample: dict[str, Any],
    *,
    rendered: str | None = None,
    min_chars: int = _DEFAULT_MIN_CHARS,
    expected_substrings: Iterable[str] = (),
) -> SampleReport:
    """Diagnostic-only variant of ``render_strict``. Never raises; collects all issues.

    When ``rendered`` is provided (e.g. produced by ``benchmark.build_prompt``),
    that is the ground truth: ``missing_vars`` becomes advisory and we only
    fail on lint issues. When ``rendered`` is None and we have a ``template_src``,
    we render the template ourselves with the raw sample dict — only useful for
    benchmarks that do **not** augment the sample inside ``build_prompt``.
    """
    sample_id = str(sample.get("id", "<unknown>"))
    report = SampleReport(sample_id=sample_id, ok=True)

    # Static introspection — informational regardless of whether we render here.
    if template_src is not None:
        try:
            required = find_required_vars(template_src)
            report.missing_vars = sorted(required - set(sample.keys()))
        except TemplateError as exc:
            report.ok = False
            report.issues.append(f"template parse error: {exc}")
            return report

    # If the caller didn't render via build_prompt, we render the template ourselves
    # using the raw sample dict (no benchmark-side augmentation). In that case
    # missing_vars **is** a hard failure: there's nobody to fill the gap.
    if rendered is None and template_src is not None:
        if report.missing_vars:
            report.ok = False
            report.issues.append(
                f"sample {sample_id!r} is missing template variables "
                f"{report.missing_vars} (no build_prompt path was taken)"
            )
        try:
            rendered = render_strict(
                template_src,
                sample,
                min_chars=min_chars,
                expected_substrings=expected_substrings,
            )
        except PromptContractError as exc:
            report.ok = False
            report.issues.append(str(exc))
            rendered = ""

    if rendered is None:
        rendered = ""
    extra = lint_rendered(
        rendered, min_chars=min_chars, expected_substrings=expected_substrings,
    )
    if extra:
        report.ok = False
        report.issues.extend(extra)

    report.rendered_chars = len(rendered)
    report.rendered_preview = rendered[:300]
    return report


def _matches_sample_id(sample: dict[str, Any], sample_id: str) -> bool:
    actual_id = str(sample.get("id", ""))
    challenge_id = str(sample.get("challenge_id") or actual_id)
    return sample_id in {actual_id, challenge_id}


def check_benchmark(
    benchmark: Any,
    *,
    template_src: str | None = None,
    sample_limit: int | None = None,
    sample_id: str | None = None,
    min_chars: int = _DEFAULT_MIN_CHARS,
    expected_substrings: Iterable[str] = (),
) -> ContractReport:
    """Run the prompt contract over (a subset of) a benchmark's samples.

    Ground truth is always ``benchmark.build_prompt(sample)`` — that's what the
    agent will see at trial time. ``template_src`` (when available via
    :func:`discover_template_source`) is used **only** for advisory static
    introspection: it tells us which vars the template references so we can
    report a hint when the raw sample is missing one. We never short-circuit
    around build_prompt — many benchmarks legitimately augment the sample dict
    (proxy_url, target info, etc.) inside build_prompt.
    """
    report = ContractReport()

    for i, sample in enumerate(benchmark.iter_samples()):
        if sample_limit is not None and i >= sample_limit:
            break
        if sample_id is not None and not _matches_sample_id(sample, sample_id):
            continue

        rendered: str | None = None
        try:
            rendered = benchmark.build_prompt(sample)
        except PromptContractError as exc:
            sr = SampleReport(
                sample_id=str(sample.get("id", "<unknown>")),
                ok=False,
                issues=[str(exc)],
            )
            report.samples.append(sr)
            continue
        except Exception as exc:  # noqa: BLE001
            sr = SampleReport(
                sample_id=str(sample.get("id", "<unknown>")),
                ok=False,
                issues=[f"build_prompt raised: {type(exc).__name__}: {exc}"],
            )
            report.samples.append(sr)
            continue

        report.samples.append(
            check_sample(
                template_src,
                sample,
                rendered=rendered,
                min_chars=min_chars,
                expected_substrings=expected_substrings,
            )
        )

        if sample_id is not None and str(sample.get("id")) == sample_id:
            break

    return report


def discover_template_source(benchmark: Any) -> str | None:
    """Best-effort: return the Jinja source if the benchmark exposes a template.

    Looks for common attribute names used by example benchmarks. Returns None
    if no introspectable template is available (benchmark builds prompts
    programmatically) — in that case ``check_benchmark`` falls back to linting
    only.
    """
    # 1. explicit hook
    src = getattr(benchmark, "prompt_template_source", None)
    if isinstance(src, str) and src:
        return src
    # 2. compiled jinja template
    tpl = getattr(benchmark, "template", None)
    if tpl is not None:
        src = getattr(tpl, "source", None) or getattr(tpl, "_source", None)
        if isinstance(src, str):
            return src
    # 3. file path
    for attr in ("_prompt_file", "prompt_file", "template_path"):
        path = getattr(benchmark, attr, None)
        if path:
            try:
                return Path(path).read_text(encoding="utf-8")
            except OSError:
                pass
    return None
