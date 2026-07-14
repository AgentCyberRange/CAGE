#!/usr/bin/env python3
"""Artifact-validation serve — the always-on boot-check for a designated location.

Started BEFORE the agent runs. The agent writes its target files (Dockerfile,
compose, wrapper, …) into a **designated shared location**; then it triggers a
boot-check via the ``cage-check`` thin client, which hits this server. The server
runs the benchmark's own Scorer against the designated location and returns the
structured verdict + feedback. No submission/upload — the agent and the validator
share a filesystem location.

Reuses CAGE's ``one scorer, N call sites`` invariant: this is just another call
site for the benchmark's Scorer, with ``ScoringContext.trial_dir`` pointed at the
serve root (so ``trial_dir/workspace`` == the designated location).

Endpoints:
  GET  /task    → the task briefing the agent builds from (benchmark.build_prompt)
  POST /check   → boot-check the designated location now; return verdict + feedback
  GET  /check   → same as POST (convenience)
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Make ``cage`` importable when run from source (bind-mounted repo in the container).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cage.benchmarks.loader import load_benchmark_from_module  # noqa: E402
from cage.scoring import ScoringContext  # noqa: E402


class _Ctx:
    """Everything the request handlers need, resolved once at startup."""

    def __init__(
        self,
        benchmark_module: Path,
        sample_id: str,
        serve_root: Path,
        internal_port: int | None = None,
    ) -> None:
        self.serve_root = serve_root
        self.benchmark = load_benchmark_from_module(benchmark_module)
        self.scorer = self.benchmark.scorer()
        # Prefer the benchmark's real sample; fall back to a minimal one when the
        # dataset isn't present (e.g. excluded from the container image) — the
        # boot-check only needs ``internal_port``.
        try:
            samples = {str(s.get("id")): dict(s) for s in self.benchmark.iter_samples()}
        except Exception:  # noqa: BLE001
            samples = {}
        if sample_id in samples:
            self.sample = samples[sample_id]
        else:
            # The boot-check is port-agnostic and reads no sample fields, so a
            # minimal sample is enough when the dataset isn't present in the image.
            self.sample = {"id": sample_id}
        if internal_port:
            self.sample["internal_port"] = internal_port
        # Reuse the benchmark's prompt so the agent's briefing matches cage run.
        try:
            self.task_prompt = str(self.benchmark.build_prompt(self.sample) or "")
        except Exception as exc:  # noqa: BLE001
            self.task_prompt = f"(build_prompt failed: {exc})"

    def check(self) -> dict:
        # trial_dir=serve_root → the scorer reads serve_root/workspace, which is
        # the designated location the agent writes into.
        ctx = ScoringContext(
            trial_id="serve",
            trial_index=0,
            sample=self.sample,
            trial_dir=self.serve_root,
            run_dir=self.serve_root,
        )
        scores = self.scorer.score(ctx) or {}
        # Take the first (build_target has a single score).
        score = next(iter(scores.values()), None)
        if score is None:
            return {"passed": False, "feedback": "scorer returned no score."}
        meta = getattr(score, "metadata", {}) or {}
        return {
            "passed": bool(meta.get("passed")),
            "value": getattr(score, "value", 0.0),
            "failure_class": meta.get("failure_class"),
            "explanation": getattr(score, "explanation", ""),
            "feedback": meta.get("feedback") or getattr(score, "explanation", ""),
            # --- captured for human review ---
            "container_state": meta.get("container_state"),
            "service_logs_tail": meta.get("service_logs_tail"),
            "agent_self_check_present": meta.get("agent_self_check_present"),
            "agent_service_output": meta.get("agent_service_output"),
            "review_files": meta.get("review_files"),
            "evidence": meta.get("evidence", {}),
        }


def _make_handler(app: _Ctx):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _run_check(self) -> None:
            try:
                self._send(200, app.check())
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"passed": False, "feedback": f"validator error: {exc}"})

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/task":
                self._send(200, {"task_prompt": app.task_prompt})
            elif self.path.rstrip("/") == "/check":
                self._run_check()
            else:
                self._send(404, {"error": "not found", "paths": ["/task", "/check"]})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/check":
                self._run_check()
            else:
                self._send(404, {"error": "not found", "paths": ["/check"]})

        def log_message(self, *args) -> None:  # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="CAGE artifact-validation serve")
    ap.add_argument("--benchmark", required=True, help="path to benchmark.py")
    ap.add_argument("--sample", required=True, help="sample id to validate against")
    ap.add_argument("--internal-port", type=int, default=None,
                    help="target port to probe; overrides/replaces the sample's (needed when the dataset isn't present)")
    ap.add_argument("--root", required=True, help="serve root; the agent writes into <root>/workspace")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8911)
    args = ap.parse_args()

    serve_root = Path(args.root).resolve()
    (serve_root / "workspace").mkdir(parents=True, exist_ok=True)
    app = _Ctx(Path(args.benchmark).resolve(), args.sample, serve_root, args.internal_port)

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(app))
    print(
        f"cage artifact-validator serving sample={args.sample!r}\n"
        f"  designated location: {serve_root / 'workspace'}\n"
        f"  endpoints: http://{args.host}:{args.port}/task  ·  POST /check",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
