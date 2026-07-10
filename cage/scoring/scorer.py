"""Scorer abstraction and helper utilities."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cage.contracts.scoring import Score
from cage.scoring.context import ScoringContext

logger = logging.getLogger(__name__)


@dataclass
class GatherRuntime:
    """Inputs for :meth:`Scorer.gather` — how to reach the live target and the
    agent's produced output, WITHOUT assuming an agent container exists.

    - ``sample`` carries target reachability: ``runtime_state.project_name`` (to
      docker-exec into the target's containers) and ``target_info`` (host-published
      scoring endpoints). This is how a scorer reaches the target — never the agent.
    - ``agent_output_dir`` is a host directory holding the agent's produced output.
      In serve-only mode it is the unpacked submission, so a scorer reads the
      agent's findings from here with no container. ``None`` when unavailable.
    - ``container`` is the live agent container: present in ``cage run`` (where the
      agent's output is not yet copied to the host at gather time) and ``None`` in
      serve-only. A scorer uses it ONLY as the agent-output source when
      ``agent_output_dir`` is absent — never to reach the target.
    """

    sample: dict[str, Any]
    agent_output_dir: Path | None = None
    container: Any = None


class Scorer(ABC):
    """Compute ``dict[str, Score]`` for one trial."""

    name: str = ""
    strategy: str = "per_trial"

    @abstractmethod
    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        """Return named score records for the provided trial context."""

    def gather(self, runtime: GatherRuntime) -> str:
        """LIVE evidence-gathering phase — the live half of scoring.

        Observe the **running target** (via ``runtime``) and return a
        serializable evidence string that :meth:`score` later consumes (through
        ``ctx.check_done_output`` / ``ctx.live_payload``). This is the ONLY
        scoring step that requires the target environment to be up: an
        implementation may ``docker exec`` into the target's containers (marker
        checks) or hit a host-published scoring endpoint (an evaluator ``/done``).
        It MUST run before the target is torn down; afterwards there is nothing
        to observe.

        ``runtime`` decouples this from "the agent container": the target is
        reached via ``runtime.sample``, and the agent's produced output comes
        from ``runtime.agent_output_dir`` (a host dir — serve-only) or, when that
        is absent, ``runtime.container`` (cage run, before the workspace is copied
        to the host). Verifiers that judge purely target-side state ignore both.

        Formerly ``Benchmark.check_done`` — it was always the scorer's live half
        (the live monitor already fed its output straight into ``score``), so it
        lives on the scorer, not the benchmark. Default: no live evidence (``""``).
        """
        return ""


class CompositeScorer(Scorer):
    """Run several scorers and merge their result dictionaries."""

    def __init__(self, scorers: list[Scorer], name: str = "") -> None:
        if not scorers:
            raise ValueError("CompositeScorer requires at least one scorer")
        self._scorers = list(scorers)
        self.name = name or scorers[0].name

    def score(self, ctx: ScoringContext) -> dict[str, Score]:
        """Score a trial with each child scorer, keeping later key collisions."""

        merged: dict[str, Score] = {}
        for scorer in self._scorers:
            try:
                merged.update(scorer.score(ctx))
            except Exception:
                logger.exception("Composite scorer %s failed", type(scorer).__name__)
        return merged


def extract_thinking_blocks(proxy_log: list[dict[str, Any]]) -> list[str]:
    """Extract thinking/reasoning text from proxy log entries."""

    blocks: list[str] = []

    for entry in proxy_log:
        for resp_key in ("anthropic_response", "upstream_response"):
            resp = entry.get(resp_key) or {}

            content = resp.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            blocks.append(block.get("thinking", ""))
                        elif block.get("type") == "reasoning":
                            blocks.append(block.get("text", ""))

            for choice in resp.get("choices", []):
                msg = choice.get("message", {})
                rc = msg.get("reasoning_content", "")
                if rc:
                    blocks.append(rc)
                text = msg.get("content", "")
                if text and "<think>" in text:
                    for match in re.finditer(
                        r"<think>(.*?)</think>",
                        text,
                        re.DOTALL,
                    ):
                        blocks.append(match.group(1))

    return [block for block in blocks if block.strip()]


def extract_token_usage(proxy_log: list[dict[str, Any]]) -> dict[str, int]:
    """Sum token usage across successful proxy entries."""

    from cage.proxy.usage import extract_entry_usage

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "num_requests": 0,
    }
    for entry in proxy_log:
        if entry.get("status") != "success":
            continue
        usage = extract_entry_usage(entry)
        totals["input_tokens"] += usage["input_tokens"]
        totals["output_tokens"] += usage["output_tokens"]
        totals["reasoning_tokens"] += usage["reasoning_tokens"]
        totals["num_requests"] += 1
    return totals


def parse_check_done_status(output: str) -> tuple[bool, dict[str, Any]]:
    """Parse common JSON check-done status shapes."""

    text = str(output or "").strip()
    if not text:
        return False, {}
    try:
        payload = json.loads(text[text.index("{"):])
    except (ValueError, json.JSONDecodeError):
        return False, {}
    if not isinstance(payload, dict):
        return False, {}
    for key in ("status", "success", "done"):
        value = payload.get(key)
        if value is True:
            return True, payload
        if isinstance(value, (int, float)) and bool(value):
            return True, payload
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "ok", "success"}:
            return True, payload
    return False, payload


def load_scorer_from_module(module_path: Path) -> Scorer:
    """Load a single ``Scorer`` subclass from a Python file."""

    spec = importlib.util.spec_from_file_location("_user_scorer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["_user_scorer"] = mod
    spec.loader.exec_module(mod)

    scorer_classes = [
        obj for obj in vars(mod).values()
        if isinstance(obj, type)
        and issubclass(obj, Scorer)
        and obj is not Scorer
        and obj is not CompositeScorer
    ]
    if not scorer_classes:
        raise ValueError(f"No Scorer subclass found in {module_path}")
    if len(scorer_classes) > 1:
        names = [c.__name__ for c in scorer_classes]
        raise ValueError(
            f"Multiple Scorer subclasses in {module_path}: {names}. "
            "Define exactly one."
        )
    return scorer_classes[0]()
