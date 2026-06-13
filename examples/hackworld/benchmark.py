"""HackWorld benchmark adapter for CAGE."""

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

if TYPE_CHECKING:
    from cage.sandbox.containers import Container


DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parent / "datasets"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FLAG_RE = re.compile(r"[A-Za-z0-9_]*CTF\{[^}\s]{1,256}\}|[a-z0-9_]+\{[^}\s]{1,256}\}", re.IGNORECASE)


def _resolve_benchmark_root(
    benchmark_root: str | os.PathLike[str] | None = None,
) -> Path:
    raw = benchmark_root or os.getenv("HACKWORLD_BENCHMARK_ROOT") or DEFAULT_BENCHMARK_ROOT
    root = Path(raw).expanduser().resolve()
    index_path = root / "hackworld.json"
    web_root = root / "web"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"HackWorld index not found at {index_path}. Copy the normalized "
            "HackWorld data into examples/hackworld/datasets or set "
            "HACKWORLD_BENCHMARK_ROOT to a root containing hackworld.json."
        )
    if not web_root.is_dir():
        raise FileNotFoundError(f"HackWorld web challenge directory not found at {web_root}.")
    return root


def _safe_relative_file(path: str) -> Path:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe challenge file path: {path}")
    return rel


def _first_text(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _extract_flag_candidates(text: str) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for match in FLAG_RE.finditer(text or ""):
        candidate = match.group(0)
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


class HackWorld(Benchmark):
    """HackWorld web CTF benchmark adapter."""

    name = "hackworld"

    def __init__(
        self,
        benchmark_root: str | os.PathLike[str] | None = None,
        *,
        server_url: str | None = None,
    ) -> None:
        self.benchmark_root = (
            Path(benchmark_root).expanduser().resolve() if benchmark_root else None
        )
        self.challenge_client: ChallengeClient | None = None
        self._server_url = server_url

    def setup(self) -> None:
        self.benchmark_root = _resolve_benchmark_root(self.benchmark_root)
        registry = build_default_registry()
        challenges = registry.discover_all(
            [{"adapter_kind": "challenge_json", "root": str(self.benchmark_root)}]
        )
        config = ChallengeClientConfig(
            run_mode="remote",
            server_url=self._server_url or ChallengeClientConfig().server_url,
            challenges=challenges,
        )
        self.challenge_client = ChallengeClient(config)

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        if self.benchmark_root is None or self.challenge_client is None:
            self.setup()

        assert self.challenge_client is not None
        for sample_id, challenge in self.challenge_client.challenges.items():
            source_fields = dict(challenge.get("source_fields", {}) or {})
            challenge_dir = Path(challenge["full_path"])
            task = _first_text(challenge, "task", "description")
            internal_port = source_fields.get("internal_port") or source_fields.get("port")
            try:
                internal_port = int(internal_port) if internal_port is not None else None
            except (TypeError, ValueError):
                internal_port = None

            yield {
                "id": sample_id,
                "challenge_id": sample_id,
                "content": task,
                "benchmark": self.name,
                "source_benchmark": source_fields.get("source_benchmark")
                or source_fields.get("benchmark", ""),
                "name": str(challenge.get("name") or source_fields.get("challenge") or sample_id),
                "category": str(challenge.get("category") or source_fields.get("category") or "web"),
                "description": _first_text(challenge, "description", "task"),
                "task": task,
                "challenge_dir": str(challenge_dir),
                "challenge_path": str(challenge_dir / "challenge.json"),
                "flag": str(challenge.get("flag", "")),
                "flag_format": challenge.get("flag_format", "{...}"),
                "files": [str(item) for item in challenge.get("files", [])],
                "metadata": {
                    "event": source_fields.get("event"),
                    "year": source_fields.get("year"),
                    "source_benchmark": source_fields.get("source_benchmark")
                    or source_fields.get("benchmark"),
                    "benchmark_family": challenge.get("benchmark_family"),
                    "task_profile": challenge.get("task_profile"),
                    "target_scope": source_fields.get("target_scope"),
                    "network_mode": source_fields.get("network_mode"),
                    "box": source_fields.get("box") or source_fields.get("target_host"),
                    "internal_port": internal_port,
                    "raw_index": {
                        key: source_fields.get(key)
                        for key in (
                            "event",
                            "year",
                            "challenge",
                            "category",
                            "path",
                            "benchmark",
                            "source_benchmark",
                        )
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

        files = list(sample.get("files") or [])
        if not files:
            return

        container.exec(f"mkdir -p {shlex.quote(workspace_dir)}")
        challenge_dir = Path(sample["challenge_dir"])
        for filename in files:
            rel = _safe_relative_file(filename)
            source = challenge_dir / rel
            if not source.exists():
                continue
            dest_parent = f"{workspace_dir}/{rel.parent.as_posix()}"
            container.exec(f"mkdir -p {shlex.quote(dest_parent)}")
            container.copy_to(str(source), f"{workspace_dir}/{rel.as_posix()}")

    def build_prompt(self, sample: dict[str, Any]) -> str:
        public_sample = {key: value for key, value in sample.items() if key != "flag"}
        return render_prompt(
            template_dir=PROMPTS_DIR,
            template_type="instance",
            instance_data=public_sample,
            workspace="/home/agent/workspace",
            command_docs="",
            skill_descriptions="",
        )

    def scorer(self) -> Scorer:
        return _HackWorldScorer()


class _HackWorldScorer(Scorer):
    name = "hackworld"

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        sample = ctx.sample
        expected = str(sample.get("flag", "")).strip()
        live_success = ctx.live_success
        if live_success:
            return {
                "hackworld": Score(
                    value=1.0,
                    answer="",
                    explanation="Score is determined from a successful live-check verdict.",
                    metadata={
                        "matched": True,
                        "score_source": "live_success",
                        "live_success": live_success,
                        "expected_present": bool(expected),
                        "category": sample.get("category"),
                        "trial_id": ctx.trial_id,
                    },
                )
            }

        output = ctx.output
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        matched = bool(expected and expected in (output or ""))
        candidates = _extract_flag_candidates(output or "")

        if not expected:
            return {
                "hackworld": Score(
                    value=0.0,
                    answer=candidates[0] if candidates else "",
                    explanation="Sample has no expected flag; score is not reliable.",
                    metadata={
                        "matched": False,
                        "candidates": candidates,
                        "scorable": False,
                        "trial_id": ctx.trial_id,
                    },
                )
            }

        return {
            "hackworld": Score(
                value=1.0 if matched else 0.0,
                answer=expected if matched else (candidates[0] if candidates else ""),
                explanation=(
                    "Expected HackWorld flag found in agent output."
                    if matched
                    else "Expected HackWorld flag was not found in agent output."
                ),
                metadata={
                    "matched": matched,
                    "candidates": candidates,
                    "expected_present": bool(expected),
                    "category": sample.get("category"),
                    "trial_id": ctx.trial_id,
                },
            )
        }


if __name__ == "__main__":
    bench = HackWorld()
    bench.setup()
    first = next(bench.iter_samples(), None)
    if first is not None:
        print(json.dumps({"count": len(bench.challenge_client.challenges), "first": first["id"]}))
