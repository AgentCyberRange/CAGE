#!/usr/bin/env python3
"""RL integration smoke — proves a/b/c through the REAL Cage code, no Docker.

It drives the actual in-container proxy (``cage.proxy.sidecar.run_proxy``) and
the actual reward reporter (``cage.rl.reward_sink.report_trial_rewards``) against
an in-memory mock gateway, and checks:

  (a) every LLM call the proxy forwards upstream carries ``X-Trial-Id``, equal to
      the run/trial join key the trial runner injects;
  (b) the reward sink is hit exactly once per trial, with the SAME ``trial_id``
      string and a valid reward in [0,1];
  (c) with RL off (no ``rl_reward_sink``) the header is absent and nothing is
      reported — the eval path is unchanged.

This is the fast, deterministic proof of the wiring. The literal end-to-end
``cage run`` smoke (heavy: real agent + comfyui target) uses the same code paths;
see the command printed at the end.

    python scripts/rl_smoke.py
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from cage.contracts.scoring import Score
from cage.models.endpoint import ModelConfig
from cage.proxy.sidecar import run_proxy
from cage.rl.reward_sink import report_trial_rewards, rl_trial_id

RUN_ID = "smoke-run-1"
TRIAL_ID = "comfyui/pass_1"
EXPECTED_JOIN_KEY = f"{RUN_ID}/{TRIAL_ID}"  # what BOTH sites must produce


# --- in-memory mock gateway -------------------------------------------------

class _Mock:
    def __init__(self) -> None:
        self.chat_headers: list[str] = []   # X-Trial-Id seen on each LLM call
        self.rewards: list[dict[str, Any]] = []
        chat_headers, rewards = self.chat_headers, self.rewards

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_a: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                path = self.path.rstrip("/")
                if path.endswith("/chat/completions"):
                    chat_headers.append(self.headers.get("X-Trial-Id", "<MISSING>"))
                    out = {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "model": "rl-policy",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "ok"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                else:  # /rl/reward
                    rewards.append(json.loads(body or b"{}"))
                    out = {"ok": True}
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 5.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"proxy did not come up on {port}")


def _drive_one_llm_call(mock: _Mock, *, rl: bool, log_dir: str) -> None:
    """Start the REAL sidecar proxy exactly as the trial runner would and send
    one Anthropic request through it."""
    # Mirror trial_runner.py: extra_headers from the model, plus X-Trial-Id when
    # the model is RL-enabled — using the real ModelConfig + rl_trial_id.
    model = ModelConfig(
        id="rl-policy", provider="openai", model="rl-policy",
        base_url=f"http://127.0.0.1:{mock.port}/v1",
        rl_reward_sink=(f"http://127.0.0.1:{mock.port}/rl/reward" if rl else ""),
    )
    extra_headers = dict(model.extra_headers or {})
    if model.rl_enabled:
        extra_headers["X-Trial-Id"] = rl_trial_id(RUN_ID, TRIAL_ID)

    proxy_port = _free_port()
    config = {
        "upstream_base_url": model.base_url,
        "upstream_protocol": model.protocol,   # "openai" → translate + _forward_openai
        "extra_headers": extra_headers,
        "trial_id": TRIAL_ID,
        "port": proxy_port,
        "log_dir": log_dir,
        "max_requests": -1,
    }
    threading.Thread(target=run_proxy, args=(config,), daemon=True).start()
    _wait_port(proxy_port)

    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy_port}/v1/messages",
        data=json.dumps({
            "model": "rl-policy", "max_tokens": 16,
            "messages": [{"role": "user", "content": "ping"}],
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def main() -> int:
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")

    with TemporaryDirectory() as tmp:
        # ---- RL ON: (a) header on the LLM call + (b) reward once, matching id --
        print("RL ON (rl_reward_sink set):")
        mock = _Mock()
        try:
            _drive_one_llm_call(mock, rl=True, log_dir=str(Path(tmp) / "on"))
            check("(a) LLM call carried exactly one X-Trial-Id",
                  len(mock.chat_headers) == 1, f"seen={mock.chat_headers}")
            check("(a) X-Trial-Id == join key",
                  mock.chat_headers == [EXPECTED_JOIN_KEY], EXPECTED_JOIN_KEY)

            run = SimpleNamespace(run_id=RUN_ID, benchmark=_StubBenchmark())
            agent = SimpleNamespace(model=SimpleNamespace(
                rl_reward_sink=f"http://127.0.0.1:{mock.port}/rl/reward"))
            result = SimpleNamespace(trial_id=TRIAL_ID, sample_id="comfyui",
                                     exit_code=0, error=None,
                                     scores={"agent_pentest_bench": Score(value=1.0)})
            report_trial_rewards(run, agent, [result])

            check("(b) reward sink hit exactly once", len(mock.rewards) == 1,
                  f"count={len(mock.rewards)}")
            if mock.rewards:
                r = mock.rewards[0]
                check("(b) reward trial_id == the SAME join key",
                      r.get("trial_id") == EXPECTED_JOIN_KEY, r.get("trial_id"))
                check("(b) reward in [0,1]", isinstance(r.get("reward"), (int, float))
                      and 0.0 <= r["reward"] <= 1.0, str(r.get("reward")))
            # The crux: the id on the LLM call and on the reward are byte-identical.
            check("join key identical at both sites",
                  mock.chat_headers == [r.get("trial_id") for r in mock.rewards])
        finally:
            mock.close()

        # ---- RL OFF: (c) no header, nothing reported -------------------------
        print("RL OFF (no rl_reward_sink) — regression guard:")
        mock = _Mock()
        try:
            _drive_one_llm_call(mock, rl=False, log_dir=str(Path(tmp) / "off"))
            check("(c) LLM call carried NO X-Trial-Id",
                  mock.chat_headers == ["<MISSING>"], f"seen={mock.chat_headers}")
            run = SimpleNamespace(run_id=RUN_ID, benchmark=_StubBenchmark())
            agent = SimpleNamespace(model=SimpleNamespace(rl_reward_sink=""))
            report_trial_rewards(run, agent, [SimpleNamespace(
                trial_id=TRIAL_ID, sample_id="comfyui", exit_code=0, error=None,
                scores={"agent_pentest_bench": Score(value=1.0)})])
            check("(c) nothing reported to the sink", mock.rewards == [],
                  f"count={len(mock.rewards)}")
        finally:
            mock.close()

    print("\n" + ("SMOKE PASS — a/b/c verified" if ok else "SMOKE FAIL"))
    return 0 if ok else 1


# Minimal stand-in exercising the real default reward path.
class _StubBenchmark:
    from cage.benchmarks.base import Benchmark as _Base
    reward = _Base.reward


if __name__ == "__main__":
    raise SystemExit(main())
