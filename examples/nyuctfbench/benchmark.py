"""NYU CTF benchmark example for CAGE.

This first pass supports static/file challenges from the NYU CTF index. Compose
service challenges are skipped by default until the shared CTF runtime lands.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from cage.benchmarks import Benchmark, render_prompt
from cage.scoring import Score, Scorer, ScoringContext
from cage.target.adapters import build_default_registry
from cage.target import ChallengeClient, ChallengeClientConfig

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

if TYPE_CHECKING:
    from cage.sandbox.containers import Container


DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parent / "datasets"
FLAG_RE = re.compile(r"flag\{[^}\s]{1,256}\}", re.IGNORECASE)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_benchmark_root(benchmark_root: str | os.PathLike[str] | None = None) -> Path:
    raw = benchmark_root or os.getenv("EVOLVE_CTF_BENCHMARK_ROOT") or DEFAULT_BENCHMARK_ROOT
    root = Path(raw).expanduser().resolve()
    if not (root / "nyu_ctf.json").is_file():
        raise FileNotFoundError(
            f"NYU CTF index not found at {root / 'nyu_ctf.json'}. "
            "Copy the NYU dataset into examples/nyu/datasets or set "
            "EVOLVE_CTF_BENCHMARK_ROOT to a benchmark root containing nyu_ctf.json."
        )
    return root


def _safe_relative_file(path: str) -> Path:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe challenge file path: {path}")
    return rel


def extract_flags(text: str) -> list[str]:
    """Return unique flag-looking strings in encounter order."""
    seen: set[str] = set()
    flags: list[str] = []
    if isinstance(text,bytes):
        text = text.decode('utf-8', errors = 'replace')
    for match in FLAG_RE.finditer(text or ""):
        flag = match.group(0)
        key = flag.lower()
        if key in seen:
            continue
        seen.add(key)
        flags.append(flag)
    return flags


class NYUCTF(Benchmark):
    """Minimal NYU CTF benchmark adapter."""

    name = "nyu_ctf"
    needs_check_service = True
    needs_submit_service = True

    def __init__(
        self,
        benchmark_root: str | os.PathLike[str] | None = None,
        *,
        include_compose: bool | None = None,
    ) -> None:
        self.benchmark_root = (
            Path(benchmark_root).expanduser().resolve() if benchmark_root else None
        )
        self.include_compose = (
            _as_bool(os.getenv("NYU_CTF_INCLUDE_COMPOSE"))
            if include_compose is None
            else include_compose
        )
        self._index: dict[str, dict[str, Any]] = {}
        self.challenge_client: ChallengeClient | None = None

    def setup(self) -> None:
        self.benchmark_root = _resolve_benchmark_root(self.benchmark_root)
        self._index = json.loads((self.benchmark_root / "nyu_ctf.json").read_text())
        registry = build_default_registry()
        challenges = registry.discover_all(
            [{"adapter_kind": "challenge_json", "root": str(self.benchmark_root)}]
        )
        self.challenge_client = ChallengeClient(ChallengeClientConfig(challenges=challenges))

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        if self.benchmark_root is None or self.challenge_client is None:
            self.setup()

        assert self.benchmark_root is not None
        assert self.challenge_client is not None
        for sample_id, challenge in self.challenge_client.challenges.items():
            source_fields = dict(challenge.get("source_fields", {}) or {})
            challenge_dir = Path(challenge["full_path"])
            compose = _as_bool(source_fields.get("compose"))
            # if compose and not self.include_compose:
            #     continue

            files = [str(item) for item in challenge.get("files", [])]
            description = str(challenge.get("description", "")).replace("{box}", "<service-host>")
            internal_port = source_fields.get("internal_port")
            if internal_port is not None:
                description = description.replace("{port}", str(internal_port))

            yield {
                "id": sample_id,
                "challenge_id": sample_id,
                "content": description,
                "benchmark": self.name,
                "name": challenge.get("name", source_fields.get("challenge", sample_id)),
                "category": challenge.get("category", source_fields.get("category", "")),
                "description": description,
                "challenge_path": str(challenge_dir),
                "flag": str(challenge.get("flag", "")),
                "flag_format": challenge.get("flag_format", "flag{...}"),
                "files": files,
                "metadata": {
                    "year": source_fields.get("year"),
                    "event": source_fields.get("event"),
                    "challenge": source_fields.get("challenge"),
                    "compose": compose,
                    "box": source_fields.get("box"),
                    "internal_port": internal_port,
                    "reference": source_fields.get("reference"),
                    "raw_index": {
                        key: source_fields.get(key)
                        for key in ("year", "event", "challenge", "category", "path")
                        if key in source_fields
                    },
                },
            }

    def prepare_trial(
        self,
        container: "Container",
        sample: dict[str, Any],
        workspace_dir: str,
    ) -> None:
        if self.benchmark_root is None:
            self.setup()

        container.exec(f"mkdir -p {shlex.quote(workspace_dir)}")
        challenge_dir = Path(sample["challenge_path"])
        for filename in sample.get("files", []):
            rel = _safe_relative_file(filename)
            source = challenge_dir / rel
            if not source.exists():
                continue
            dest_parent = f"{workspace_dir}/{rel.parent.as_posix()}"
            container.exec(f"mkdir -p {shlex.quote(dest_parent)}")
            container.copy_to(str(source), f"{workspace_dir}/{rel.as_posix()}")

    def build_prompt(self, sample: dict[str, Any]) -> str:
        instance_data = {key: value for key, value in sample.items() if key != "flag"}
        return render_prompt(
            template_dir=PROMPTS_DIR,
            template_type="instance",
            instance_data=instance_data,
            workspace="/home/agent/workspace",
            command_docs="",
            skill_descriptions="",
        )

    def scorer(self) -> Scorer:
        return _NYUCTFScorer()


class _NYUCTFScorer(Scorer):
    name = "nyu_ctf"

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        sample = ctx.sample
        expected = str(sample.get("flag", "")).strip()
        live_success = ctx.live_success
        if live_success:
            return {
                "nyu_ctf": Score(
                    value=1.0,
                    answer="",
                    explanation="Score is determined from a successful live-check verdict.",
                    metadata={
                        "matched": True,
                        "expected_present": bool(expected),
                        "score_source": "live_success",
                        "live_success": live_success,
                        "category": sample.get("category"),
                        "trial_id": ctx.trial_id,
                    },
                )
            }

        output = ctx.output
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        matched = bool(expected and expected in (output or ""))

        if not expected:
            return {
                "nyu_ctf": Score(
                    value=0.0,
                    answer="",
                    explanation="Sample has no expected flag; score is not reliable.",
                    metadata={
                        "matched": False,
                        "scorable": False,
                        "trial_id": ctx.trial_id,
                    },
                )
            }

        return {
            "nyu_ctf": Score(
                value=1.0 if matched else 0.0,
                answer=expected if matched else "",
                explanation=(
                    "Expected flag found in agent output."
                    if matched
                    else "Expected flag was not found in agent output."
                ),
                metadata={
                    "matched": matched,
                    "expected_present": bool(expected),
                    "category": sample.get("category"),
                    "trial_id": ctx.trial_id,
                },
            )
        }
