"""Benchmark base contract for Layer 2 benchmark adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional

from cage.benchmarks.prompt_contract import render_strict
from cage.contracts.scoring import extract_numeric_score_value
from cage.scoring import Scorer

if TYPE_CHECKING:
    from cage.artifacts.dashboard import Dashboard
    from cage.experiment.model import TrialResult
    from cage.sandbox.containers import Container


@dataclass(frozen=True)
class BenchmarkOption:
    """Benchmark-owned CLI option mapped back to a project.yml field."""

    flag: str
    config_path: str
    help: str = ""
    choices: tuple[str, ...] = ()
    multiple: bool = False
    value_type: str = "str"
    metavar: str = ""


@dataclass(frozen=True)
class BenchmarkBuildResult:
    """One benchmark-owned target build result."""

    target_id: str
    status: str
    command: list[str] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    detail: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True)
class BenchmarkBuildSummary:
    """Summary returned by :meth:`Benchmark.build_targets`."""

    total: int
    built: int
    skipped: int
    failed: int
    results: list[BenchmarkBuildResult]
    planned: int = 0

    @classmethod
    def from_results(
        cls,
        results: Iterable[BenchmarkBuildResult],
    ) -> "BenchmarkBuildSummary":
        """Create a summary from individual build result records."""

        items = list(results)
        return cls(
            total=len(items),
            built=sum(1 for item in items if item.status == "built"),
            skipped=sum(1 for item in items if item.status == "skipped"),
            failed=sum(1 for item in items if item.status == "failed"),
            results=items,
            planned=sum(1 for item in items if item.status == "planned"),
        )


def normalize_sample_id(value: object) -> str:
    """Normalize a CLI-facing sample id for forgiving user input matching."""

    return str(value or "").strip().casefold()


def sample_id_candidates(sample: dict[str, Any]) -> set[str]:
    """Return ids that should select a sample from user-facing CLI filters."""

    candidates = {
        str(sample.get("id") or ""),
        str(sample.get("sample_id") or ""),
        str(sample.get("challenge_id") or ""),
    }
    aliases = sample.get("aliases")
    if isinstance(aliases, (list, tuple, set)):
        candidates.update(str(alias or "") for alias in aliases)
    return {candidate for candidate in candidates if candidate}


def sample_id_matches(sample: dict[str, Any], wanted: Iterable[str]) -> bool:
    """Return true when a sample matches any requested id."""

    wanted_raw = {str(item or "").strip() for item in wanted if str(item or "").strip()}
    if not wanted_raw:
        return True
    wanted_normalized = {normalize_sample_id(item) for item in wanted_raw}
    for candidate in sample_id_candidates(sample):
        if candidate in wanted_raw or normalize_sample_id(candidate) in wanted_normalized:
            return True
    return False


def parse_sample_slice(spec: object) -> slice | None:
    """Parse a Python-style slice spec for selecting a window of samples.

    Accepts the same forms as Python's ``list[...]`` subscript:
      ``":100"`` first 100, ``"-100:"`` last 100, ``"-100:-1"`` last 100
      minus one, ``"100:200"`` a middle window, ``"::2"`` every other,
      ``"5"`` the single 6th sample, ``"-1"`` the last one.

    Returns a ``slice`` object (applied to the ordered, already-id-filtered
    sample list) or ``None`` when ``spec`` is empty. Raises ``ValueError`` on
    malformed input so the CLI/yaml fail loudly instead of silently running
    the whole set.
    """
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None

    def _part(token: str) -> int | None:
        token = token.strip()
        return int(token) if token else None

    if ":" not in text:
        # Bare index -> a one-element window. ``-1`` needs ``[-1:]`` because
        # ``[-1:0]`` would be empty.
        idx = int(text)
        if idx == -1:
            return slice(-1, None)
        return slice(idx, idx + 1)

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid slice spec {spec!r}: too many ':' separators")
    start = _part(parts[0])
    stop = _part(parts[1])
    step = _part(parts[2]) if len(parts) == 3 else None
    if step == 0:
        raise ValueError(f"invalid slice spec {spec!r}: step cannot be 0")
    return slice(start, stop, step)


class Benchmark(ABC):
    """Base class for benchmark adapters implemented outside Layer 1."""

    name: str = ""

    # -- Live-check service requirements --------------------------------------
    # Declarative capability flags. The framework branches on these, never on a
    # benchmark name. A benchmark overrides the ones it needs; the default is
    # "no live-check services".

    needs_check_service: bool = False
    """Live verification needs an in-container check daemon for this benchmark."""

    uses_builtin_check: bool = False
    """The benchmark's check is built into the agent image (no daemon needed)."""

    needs_submit_service: bool = False
    """Live verification needs an in-container flag-submit daemon."""

    def setup(self) -> None:
        """Download data, initialize resources, or perform other setup."""

    def teardown(self) -> None:
        """Clean up resources at the end of a run."""

    def on_trial_complete(
        self,
        container: Container,
        sample: dict[str, Any],
        trial_dir: str,
    ) -> None:
        """Run benchmark logic while the trial container is still alive."""

    def build_targets(
        self,
        samples: list[dict[str, Any]],
        reporter: Callable[[str, dict[str, Any]], None] | None = None,
        max_workers: int = 1,
        dry_run: bool = False,
        rebuild: bool = False,
    ) -> BenchmarkBuildSummary:
        """Build benchmark target images for selected samples.

        ``rebuild`` re-builds even images that already exist locally; the default
        skips target images that are already built.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not implement a benchmark build hook"
        )

    @abstractmethod
    def iter_samples(self) -> Iterator[dict[str, Any]]:
        """Yield samples. Each must have ``id`` and ``content`` fields."""

    @abstractmethod
    def prepare_trial(
        self,
        container: Container,
        sample: dict[str, Any],
        workspace_dir: str,
    ) -> None:
        """Prepare the isolated agent workspace before the agent starts."""

    @abstractmethod
    def build_prompt(self, sample: dict[str, Any]) -> str:
        """Build the prompt passed to the agent CLI for a sample."""

    @abstractmethod
    def scorer(self) -> Scorer:
        """Return the default scorer applied to every trial."""

    def container_image_override(self) -> str | None:
        """Optional Docker image this benchmark requires for the agent container.

        Returns ``None`` (default) to use the agent's configured image. A
        benchmark that needs a specific runtime — e.g. a white-box debug image
        whose ABI matches a binary it stages into the workspace — returns the
        image ref here, and the orchestrator uses it instead of the agent's
        default. Layer 1 stays benchmark-agnostic: the concrete image name comes
        from the benchmark/config, never from ``cage/``. This lets a single
        benchmark knob (e.g. a debug flag) swap both the workspace contents and
        the runtime image, so users flip one parameter instead of two.
        """
        return None

    def reward(self, result: "TrialResult") -> float:
        """Scalar RL reward in ``[0, 1]`` for one finished trial.

        This is the Layer-2 seam an external RL trainer consumes (see
        ``cage.rl.reward_sink``). The framework owns only the *mechanism* —
        attaching a trial id to LLM calls and POSTing the number — while *what
        the number means* is benchmark domain knowledge and lives here.

        Default: the trial's primary score, i.e. the best numeric value the
        benchmark's own :meth:`scorer` already produced, clamped to ``[0, 1]``.
        A failed or unscored trial has no numeric score and yields ``0.0`` — so
        timeouts/crashes report reward 0 without special-casing. Benchmarks that
        want shaped reward (partial credit, combined metrics, a penalty term)
        override this in their ``benchmark.py``; they need not touch Layer 1.

        Only consulted when the run's model declares ``rl_reward_sink``; with RL
        off this method is never called, so it costs ordinary evals nothing.
        """

        values = [
            v
            for v in (
                extract_numeric_score_value(s) for s in result.scores.values()
            )
            if v is not None
        ]
        if not values:
            return 0.0
        return max(0.0, min(1.0, max(values)))

    def check_done(
        self,
        container: Container,
        sample: dict[str, Any],
    ) -> str:
        """Query the benchmark's target-side check endpoint, if any."""

        return ""

    def build_dashboard(self, run_dir: Path) -> Optional["Dashboard"]:
        """Build a benchmark-specific visualization for a finished run."""

        return None

    def live_check_triggers(self, sample: dict[str, Any]) -> list[str]:
        """Return command substrings that should trigger live verification."""

        return []

    def live_check_confirm_polls(
        self,
        sample: dict[str, Any],
        check_done_output: str,
    ) -> int | None:
        """Override the consecutive-poll threshold for one live verdict."""

        return None

    def live_check_polling_interval(self, sample: dict[str, Any]) -> float | None:
        """Override the global live-check polling interval for one sample."""

        return None

    def validate_live_verdict(
        self,
        container: Container,
        sample: dict[str, Any],
        verdict_class: str,
        check_done_output: str,
    ) -> bool:
        """Second-opinion check before locking in a positive live verdict."""

        return True

    def cli_options(self) -> list[BenchmarkOption]:
        """Return benchmark-specific ``cage run`` options."""

        return []

    def variant_display_axes(self) -> dict[str, tuple[str, ...]]:
        """Active per-run variant axes, for the Layer-1 run summary only.

        Keys are conventional axis names (e.g. ``"prompt"``, ``"hint"``); values
        are the active level tokens. The framework only *displays* these — it
        never branches on them — so a benchmark that has prompt/hint tiers can
        surface them here instead of forcing the UI to read private attributes.
        Default: no axes.
        """

        return {}

    def iter_samples_limited(
        self,
        limit: int | None = None,
        sample_ids: list[str] | None = None,
        slice_spec: slice | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate samples with an optional id filter, then slice, then limit.

        Selection order, each applied to the result of the previous step:
          1. ``sample_ids`` — keep only matching ids (preserving manifest order).
          2. ``slice_spec`` — a Python ``slice`` (e.g. ``:100``, ``-100:-1``)
             selecting a window of the filtered list. Negative indices require
             the full list, so this materializes the filtered stream.
          3. ``limit`` — cap the count of what remains.
        """

        wanted = list(sample_ids or [])

        def _filtered() -> Iterator[dict[str, Any]]:
            for sample in self.iter_samples():
                if wanted and not sample_id_matches(sample, wanted):
                    continue
                yield sample

        if slice_spec is not None:
            stream: Iterator[dict[str, Any]] = iter(list(_filtered())[slice_spec])
        else:
            stream = _filtered()

        emitted = 0
        for sample in stream:
            if limit is not None and emitted >= limit:
                break
            emitted += 1
            yield sample

    def render_strict(
        self,
        template_src: str,
        sample: dict[str, Any],
        *,
        min_chars: int = 20,
        expected_substrings: Iterable[str] = (),
    ) -> str:
        """Render a benchmark prompt with strict undefined-variable checks."""

        return render_strict(
            template_src,
            sample,
            min_chars=min_chars,
            expected_substrings=expected_substrings,
        )
