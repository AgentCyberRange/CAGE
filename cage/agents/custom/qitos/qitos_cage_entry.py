#!/usr/bin/env python3
"""Cage <-> QitOS orchestrator — runs INSIDE one Cage trial container.

One Cage trial = one CyberGym task. This thin entrypoint:

  1. reads the grading-service URL out of the ``submit.sh`` that Cage's cybergym
     benchmark already staged into the workspace (upstream multipart submit.sh),
  2. launches the QitOS harness' HostEnv entrypoint
     ``python -m cybergym_agent.run_local`` against that workspace — it runs all
     its tools in THIS container (no nested Docker) and POSTs candidate PoCs back
     to that grading service, reusing Cage's masked task_id / agent_id / checksum
     (which it parses out of the same submit.sh),
  3. forwards the harness' stdout (Cage's answer channel) verbatim.

It implements ZERO QitOS logic — the vendored harness (submodules
``third_party/qitos`` + ``third_party/cybergym_agent``, unchanged) does the agent
loop, tool use, submission and stop. The pure helper below is unit-tested; the
``main()`` side effects are exercised by the E2E run.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

WORKSPACE = "/home/agent/workspace"
SUBMIT_SH = f"{WORKSPACE}/submit.sh"
RUN_LOCAL_MODULE = "cybergym_agent.run_local"

# submit.sh's curl line is upstream-verbatim:  curl -X POST <server>/submit-vul ...
_SERVER_RE = re.compile(r"-X\s+POST\s+\"?([^\s\"]+)/submit-vul")


def extract_server_url(submit_sh_text: str) -> str | None:
    """Return the grading-service base URL from a staged submit.sh, or None.

    Pure (no I/O) so it stays unit-testable. Matches the upstream multipart
    ``curl -X POST <server>/submit-vul`` line Cage stages.
    """
    m = _SERVER_RE.search(submit_sh_text)
    return m.group(1) if m else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cage->QitOS cybergym launcher")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base (the Cage proxy)")
    ap.add_argument("--api-key", default="", help="empty ok — the proxy injects the real upstream key")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="0/negative => harness default")
    ap.add_argument("--task-dir", default=WORKSPACE)
    args = ap.parse_args(argv)

    submit_path = f"{args.task_dir.rstrip('/')}/submit.sh"
    try:
        server = extract_server_url(open(submit_path, encoding="utf-8").read())
    except OSError:
        server = None
    if not server:
        print(f"[qitos-cage] ERROR: could not read grading server URL from {submit_path}",
              file=sys.stderr)
        return 2

    # The proxy injects the real upstream key; a placeholder keeps run_local's
    # OpenAI client happy when the model entry carries no key of its own.
    api_key = args.api_key or "sk-cage-proxy"
    run_argv = [
        "--task-dir", args.task_dir,
        "--server", server,
        "--base-url", args.base_url,
        "--api-key", api_key,
        "--model", args.model,
    ]
    if args.max_steps and args.max_steps > 0:
        run_argv += ["--max-steps", str(args.max_steps)]

    # Hand off to the vendored harness. run_local uses absolute
    # `from cybergym_agent...` imports and lives on PYTHONPATH (set in the image).
    print(f"[qitos-cage] launching {RUN_LOCAL_MODULE} (server={server}, model={args.model})",
          flush=True)
    from cybergym_agent.run_local import main as run_local_main  # noqa: E402
    old = sys.argv
    sys.argv = [RUN_LOCAL_MODULE, *run_argv]
    try:
        rv = run_local_main()
    finally:
        sys.argv = old
    return int(rv) if isinstance(rv, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
