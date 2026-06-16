#!/usr/bin/env python3
"""Mock verl gateway for local RL-integration smoke tests.

Stands in for the training side with zero GPU. Two endpoints, stdlib only:

  POST /v1/chat/completions   OpenAI-compatible. Returns one valid chat
                              completion and PRINTS the ``X-Trial-Id`` header
                              Cage attached — proves every LLM call is tagged.
  POST /rl/reward             The reward sink. PRINTS the JSON Cage POSTs and
                              replies ``200 {"ok": true}`` — proves the reward
                              is reported once per trial with the matching id.

Every request is also appended as one JSON line to ``--log`` (if given) so a
test can assert on it after the run.

Usage:
    python scripts/rl_mock_gateway.py --port 8900 [--log /tmp/rl_mock.jsonl]

Then point a model at it (see config/models.example.yml :: rl-policy):
    base_url:        http://HOST:8900/v1
    rl_reward_sink:  http://HOST:8900/rl/reward
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _now() -> int:
    return int(time.time())


class Handler(BaseHTTPRequestHandler):
    log_path: Path | None = None

    # Silence the default noisy per-request stderr line; we print our own.
    def log_message(self, *_args: Any) -> None:  # noqa: D401
        return

    def _record(self, kind: str, data: dict[str, Any]) -> None:
        entry = {"kind": kind, "ts": _now(), **data}
        line = json.dumps(entry, ensure_ascii=False)
        print(line, flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {"_raw": raw.decode("utf-8", "replace")}

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        path = self.path.split("?", 1)[0].rstrip("/")
        body = self._read_body()

        if path.endswith("/v1/chat/completions") or path.endswith("/chat/completions"):
            trial_id = self.headers.get("X-Trial-Id", "<MISSING>")
            self._record(
                "chat_completion",
                {
                    "x_trial_id": trial_id,
                    "model": body.get("model"),
                    "n_messages": len(body.get("messages") or []),
                },
            )
            model = body.get("model") or "rl-policy"
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-mock-{_now()}",
                    "object": "chat.completion",
                    "created": _now(),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "mock-gateway: acknowledged.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
            return

        if path.endswith("/rl/reward"):
            self._record("reward", {"payload": body})
            self._send_json(200, {"ok": True})
            return

        self._record("unknown", {"path": self.path, "payload": body})
        self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_GET(self) -> None:  # noqa: N802 — health probe convenience
        self._send_json(200, {"ok": True, "service": "rl-mock-gateway"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock verl gateway for RL smoke tests")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--log", default="", help="Append every request as JSON to this file")
    args = parser.parse_args()

    if args.log:
        Handler.log_path = Path(args.log)
        Handler.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so each run starts clean.
        Handler.log_path.write_text("", encoding="utf-8")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"rl-mock-gateway listening on http://{args.host}:{args.port}  "
        f"(chat: /v1/chat/completions, reward: /rl/reward)",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
