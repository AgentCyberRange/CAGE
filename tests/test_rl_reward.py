"""RL integration: model knob, join key, default reward, and reward reporting.

These cover the framework *mechanism* without a full ``cage run``: that the RL
switch is a single model field, that the join key is deterministic, that the
Layer-2 default reward reads the scorer's score (and failures → 0), and that
``report_trial_rewards`` POSTs exactly once per trial with the matching id — and
is a no-op when RL is off (the regression guard).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from cage.contracts.scoring import Score
from cage.models.endpoint import ModelConfig
from cage.models.registry import load_models
from cage.rl.reward_sink import report_trial_rewards, rl_trial_id


# --- the model knob ---------------------------------------------------------

def test_rl_enabled_follows_reward_sink() -> None:
    assert ModelConfig(id="m", provider="openai", model="m").rl_enabled is False
    on = ModelConfig(id="m", provider="openai", model="m", rl_reward_sink="http://x/r")
    assert on.rl_enabled is True


def test_registry_parses_reward_sink(tmp_path) -> None:
    path = tmp_path / "models.yml"
    path.write_text(
        "models:\n"
        "  rl-policy:\n"
        "    provider: openai\n"
        "    model: rl-policy\n"
        "    base_url: http://host:8900/v1\n"
        "    rl_reward_sink: http://host:8900/rl/reward\n"
        "  plain:\n"
        "    provider: openai\n"
        "    model: plain\n",
        encoding="utf-8",
    )
    models = load_models(path)
    assert models["rl-policy"].rl_reward_sink == "http://host:8900/rl/reward"
    assert models["rl-policy"].rl_enabled is True
    # rl_reward_sink is a typed field, so it must NOT leak into ``extra``.
    assert "rl_reward_sink" not in models["rl-policy"].extra
    assert models["plain"].rl_enabled is False


# --- the join key -----------------------------------------------------------

def test_rl_trial_id_is_deterministic_composite() -> None:
    assert rl_trial_id("run-1", "pb-comfyui/pass_1") == "run-1/pb-comfyui/pass_1"
    # Same inputs → same string, so header and reward sites always agree.
    assert rl_trial_id("r", "t") == rl_trial_id("r", "t")


# --- the Layer-2 default reward --------------------------------------------

class _RewardBenchmark:
    """Minimal stand-in exposing the real default ``Benchmark.reward``."""

    name = "stub"

    # Bind the unbound default so we exercise the actual base implementation.
    from cage.benchmarks.base import Benchmark as _Base

    reward = _Base.reward


def _result(trial_id: str, scores: dict[str, Score], *, error: str | None = None,
            exit_code: int = 0, sample_id: str = "pb-comfyui") -> SimpleNamespace:
    return SimpleNamespace(
        trial_id=trial_id, sample_id=sample_id, exit_code=exit_code,
        error=error, scores=scores,
    )


@pytest.mark.parametrize(
    "scores, expected",
    [
        ({"web_exploit_bench": Score(value=1.0)}, 1.0),
        ({"web_exploit_bench": Score(value=0.0)}, 0.0),
        ({}, 0.0),                                   # failed / unscored → 0
        ({"a": Score(value=0.3), "b": Score(value=0.9)}, 0.9),  # best wins
        ({"clamp": Score(value=2.0)}, 1.0),          # clamped to [0,1]
    ],
)
def test_default_reward(scores: dict[str, Score], expected: float) -> None:
    assert _RewardBenchmark().reward(_result("t", scores)) == expected


# --- the reward report ------------------------------------------------------

class _Sink:
    """A throwaway HTTP server that records every /rl/reward payload."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        captured = self.payloads

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                captured.append(json.loads(self.rfile.read(n) or b"{}"))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok": true}')

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/rl/reward"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def sink() -> Iterator[_Sink]:
    s = _Sink()
    try:
        yield s
    finally:
        s.close()


def _run_agent(sink_url: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    run = SimpleNamespace(run_id="run-42", benchmark=_RewardBenchmark())
    agent = SimpleNamespace(model=SimpleNamespace(rl_reward_sink=sink_url))
    return run, agent


def test_reports_once_per_trial_including_failures(sink: _Sink) -> None:
    run, agent = _run_agent(sink.url)
    results = [
        _result("pb-comfyui/pass_1", {"web_exploit_bench": Score(value=1.0)}),
        _result("pb-other/pass_1", {}, error="boom", exit_code=-1, sample_id="pb-other"),
    ]
    report_trial_rewards(run, agent, results)

    by_id = {p["trial_id"]: p for p in sink.payloads}
    assert set(by_id) == {"run-42/pb-comfyui/pass_1", "run-42/pb-other/pass_1"}

    won = by_id["run-42/pb-comfyui/pass_1"]
    assert won["reward"] == 1.0 and won["success"] is True
    assert won["sample_id"] == "pb-comfyui"

    failed = by_id["run-42/pb-other/pass_1"]   # failure still reported, reward 0
    assert failed["reward"] == 0.0 and failed["success"] is False
    assert failed["metrics"]["error"] == "boom"


def test_no_sink_is_a_noop(sink: _Sink) -> None:
    # RL off (no rl_reward_sink) ⇒ nothing is posted: the regression guarantee.
    run = SimpleNamespace(run_id="run-1", benchmark=_RewardBenchmark())
    agent = SimpleNamespace(model=SimpleNamespace(rl_reward_sink=""))
    report_trial_rewards(run, agent, [_result("t", {"s": Score(value=1.0)})])
    assert sink.payloads == []
