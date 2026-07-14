#!/usr/bin/env python3
"""Keep a Codex CLI ChatGPT-OAuth ``~/.codex/auth.json`` fresh on disk.

Why this exists
---------------
This is the Codex twin of ``refresh_claude_oauth.py``. Codex "Sign in with
ChatGPT" stores OAuth tokens in ``~/.codex/auth.json``; the access token is a
JWT that expires in ~hours. Cage trial containers receive an on-disk snapshot
of that file at trial start (bind mount / copy) and have no outbound route to
the OAuth endpoint, so they cannot refresh themselves — the ChatGPT backend
then answers ``HTTP 401``. This daemon keeps the on-disk access token's
remaining lifetime above a floor, so every container copies a token that
outlives the whole trial.

A refresh is a pure OAuth exchange (``POST`` to
``https://auth.openai.com/oauth/token`` with the refresh token) and consumes
**no inference / usage quota**. Codex **rotates the refresh token** on every
refresh (the old one dies — sharing one auth.json across concurrent refreshers
triggers ``refresh_token_reused``). So refreshes MUST be serialized: run
exactly ONE instance of this daemon and do not let containers self-refresh.

Auth.json shape (fields this script touches)::

    {
      "auth_mode": "chatgpt",
      "OPENAI_API_KEY": null,
      "tokens": {
        "id_token": "<JWT>", "access_token": "<JWT>",
        "refresh_token": "<...>", "account_id": "<...>"
      },
      "last_refresh": "2026-07-12T10:20:30.123456Z"
    }

There is NO explicit expiry field: remaining lifetime is read from the
``access_token`` JWT ``exp`` claim (falling back to ``last_refresh`` + 8 days,
Codex's own ``TOKEN_REFRESH_INTERVAL``, if the JWT is unparseable).

Usage:
    # daemon: keep >=2.5h of life on disk at all times
    HTTPS_PROXY=http://127.0.0.1:7890 \\
        python refresh_codex_oauth.py --daemon --creds ~/.codex/auth.json

Exit codes (one-shot):
    0   refreshed OR no refresh needed
    1   refresh failed (RT rejected, network error, …)
    2   bad input (missing/unreadable file, no refresh token)
    3   another daemon instance already holds the lock (daemon mode)
"""

from __future__ import annotations

import argparse
import base64
import datetime
import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# OpenAI OAuth token endpoint + public Codex CLI client_id. Both match the
# constants in the Codex source (``codex-rs/login/src/auth/manager.rs``) and
# honour the same env overrides Codex itself reads, so a fully-proxied setup
# can redirect the refresh too.
REFRESH_URL = os.environ.get(
    "CODEX_REFRESH_TOKEN_URL_OVERRIDE", "https://auth.openai.com/oauth/token",
)
CLIENT_ID = os.environ.get(
    "CODEX_APP_SERVER_LOGIN_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann",
)
DEFAULT_CREDS = "~/.codex/auth.json"

# Floor of on-disk AT lifetime. >= Cage trial timeout (7200s) + margin so a
# container that copies the token always has enough life to finish a trial.
DEFAULT_MIN_TTL_SECONDS = 9000          # 2.5 h
DEFAULT_REQUEST_TIMEOUT = 30            # http timeout for the refresh POST
DEFAULT_RETRY_BACKOFF_SECONDS = 60      # wait after a failed refresh before retry
DEFAULT_MAX_SLEEP_SECONDS = 1800        # cap a single sleep so external file
                                        # changes / clock drift are noticed
_FALLBACK_TTL_DAYS = 8                  # Codex TOKEN_REFRESH_INTERVAL fallback


def _log(msg: str) -> None:
    ts = time.strftime("%FT%T%z")
    sys.stderr.write(f"[{ts}] refresh_codex_oauth: {msg}\n")
    sys.stderr.flush()


def _load(path: Path) -> dict:
    with path.open() as fp:
        return json.load(fp)


def _now_rfc3339() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _jwt_exp(jwt: str) -> float | None:
    """Return the ``exp`` (unix seconds) claim of a JWT, or None if unreadable."""
    try:
        parts = jwt.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)  # restore base64 padding
        claims = json.loads(base64.urlsafe_b64decode(payload.encode()))
        exp = claims.get("exp")
        return float(exp) if exp is not None else None
    except Exception:  # noqa: BLE001
        return None


def _ttl_seconds(auth: dict) -> float:
    """Remaining access-token lifetime in seconds (may be negative).

    Primary source is the access-token JWT ``exp``; if that can't be parsed we
    fall back to ``last_refresh`` + 8 days (Codex's own interval). If neither is
    available the token is treated as expired so the caller refreshes.
    """
    tokens = auth.get("tokens") or {}
    exp = _jwt_exp(str(tokens.get("access_token") or ""))
    if exp is not None:
        return exp - time.time()
    last_refresh = auth.get("last_refresh")
    if last_refresh:
        try:
            base = datetime.datetime.fromisoformat(
                str(last_refresh).replace("Z", "+00:00")
            ).timestamp()
            return (base + _FALLBACK_TTL_DAYS * 86400) - time.time()
        except Exception:  # noqa: BLE001
            pass
    return -1.0


def _write_inplace(path: Path, data: dict) -> None:
    """Overwrite ``path`` IN PLACE, keeping the same inode.

    Deliberately NOT an atomic tmp+rename: a single-file Docker bind mount
    binds the inode at container-start, so a rename (new inode) would leave
    running containers reading the orphaned old inode forever. Writing the same
    inode lets bind-mounted containers see the refreshed token. Codex writes
    auth.json ``0o600``; we preserve that.
    """
    payload = (json.dumps(data, indent=2) + "\n").encode()
    with open(path, "r+b") as fp:
        fp.seek(0)
        fp.write(payload)
        fp.truncate(len(payload))
        fp.flush()
        os.fsync(fp.fileno())
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _refresh(refresh_token: str, http_proxy: str | None, timeout: float) -> dict:
    """POST to the OAuth refresh endpoint; return the new token payload.

    Body is a plain JSON object (NOT form-encoded) with no ``scope`` field —
    exactly the ``RefreshRequest`` shape Codex sends.
    """
    body = json.dumps({
        "client_id": CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode()
    req = urllib.request.Request(
        REFRESH_URL,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "cage-refresh-codex-oauth/1",
        },
    )
    if http_proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": http_proxy, "https": http_proxy})
        )
    else:
        opener = urllib.request.build_opener()
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _refresh_and_persist(
    path: Path, http_proxy: str | None, timeout: float,
) -> float:
    """Refresh using the on-disk RT and write the new tokens back.

    Re-reads the file immediately before refreshing so a concurrent external
    write (e.g. the host Codex CLI) is picked up. Returns the new remaining TTL
    in seconds. Raises on any failure.
    """
    auth = _load(path)
    tokens = auth.get("tokens") or {}
    rt = tokens.get("refresh_token")
    if not rt:
        raise RuntimeError("no refresh_token in tokens")

    payload = _refresh(rt, http_proxy, timeout)
    new_at = payload.get("access_token")
    if not new_at:
        raise RuntimeError(f"malformed refresh response keys={list(payload)}")
    tokens["access_token"] = new_at
    tokens["refresh_token"] = payload.get("refresh_token") or rt  # RT rotates
    if payload.get("id_token"):
        tokens["id_token"] = payload["id_token"]
    auth["tokens"] = tokens
    auth["last_refresh"] = _now_rfc3339()
    _write_inplace(path, auth)
    return _ttl_seconds(auth)


def run_once(args: argparse.Namespace) -> int:
    path = Path(args.creds).expanduser()
    if not path.is_file():
        _log(f"FATAL: {path} not found")
        return 2
    try:
        auth = _load(path)
    except Exception as exc:  # noqa: BLE001
        _log(f"FATAL: cannot parse {path}: {exc}")
        return 2

    ttl = _ttl_seconds(auth)
    if not args.force and ttl >= args.min_ttl_seconds:
        _log(f"AT valid for {ttl:.0f}s (>= {args.min_ttl_seconds}s); skip")
        return 0

    _log(f"AT ttl={ttl:.0f}s; refreshing"
         f"{' (forced)' if args.force else ''}"
         f"{' via proxy=' + args.http_proxy if args.http_proxy else ''}")
    try:
        new_ttl = _refresh_and_persist(
            path, args.http_proxy or None, args.request_timeout,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        _log(f"FAIL: HTTP {exc.code} {exc.reason}: {detail}")
        return 1
    except Exception as exc:  # noqa: BLE001
        _log(f"FAIL: {exc}")
        return 1
    _log(f"OK: AT rotated, now valid for {new_ttl:.0f}s")
    return 0


def run_daemon(args: argparse.Namespace) -> int:
    path = Path(args.creds).expanduser()

    # Single-instance guard: two daemons would race each other on the one-shot
    # (rotating) refresh token.
    lock_path = Path(args.lockfile).expanduser() if args.lockfile else \
        path.with_suffix(path.suffix + ".refresh.lock")
    lock_fp = open(lock_path, "w")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _log(f"FATAL: another daemon already holds {lock_path}; exiting")
        return 3

    _log(f"daemon start: creds={path} min_ttl={args.min_ttl_seconds}s "
         f"max_sleep={args.max_sleep_seconds}s retry_backoff="
         f"{args.retry_backoff_seconds}s timeout={args.request_timeout}s "
         f"proxy={args.http_proxy or '<none>'}")

    while True:
        try:
            auth = _load(path)
        except Exception as exc:  # noqa: BLE001
            _log(f"load failed: {exc}; retry in {args.retry_backoff_seconds}s")
            time.sleep(args.retry_backoff_seconds)
            continue

        ttl = _ttl_seconds(auth)
        if ttl > args.min_ttl_seconds:
            sleep_for = min(ttl - args.min_ttl_seconds, args.max_sleep_seconds)
            sleep_for = max(sleep_for, 1.0)
            _log(f"AT ttl={ttl:.0f}s > min_ttl={args.min_ttl_seconds}s; "
                 f"sleeping {sleep_for:.0f}s")
            time.sleep(sleep_for)
            continue

        _log(f"AT ttl={ttl:.0f}s <= min_ttl={args.min_ttl_seconds}s; refreshing")
        try:
            new_ttl = _refresh_and_persist(
                path, args.http_proxy or None, args.request_timeout,
            )
            _log(f"OK: AT rotated, now valid for {new_ttl:.0f}s")
            if new_ttl <= args.min_ttl_seconds:
                # Fresh AT lifetime is shorter than the floor we want —
                # refreshing again immediately would hammer the endpoint.
                _log(f"WARN: fresh AT life {new_ttl:.0f}s <= min_ttl "
                     f"{args.min_ttl_seconds}s (min_ttl too high?); "
                     f"sleeping {args.max_sleep_seconds}s to avoid hammering")
                time.sleep(args.max_sleep_seconds)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            _log(f"refresh FAIL: HTTP {exc.code} {exc.reason}: {detail}; "
                 f"retry in {args.retry_backoff_seconds}s")
            time.sleep(args.retry_backoff_seconds)
        except Exception as exc:  # noqa: BLE001
            _log(f"refresh FAIL: {exc}; retry in {args.retry_backoff_seconds}s")
            time.sleep(args.retry_backoff_seconds)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--creds", default=DEFAULT_CREDS,
                   help=f"Path to auth.json (default {DEFAULT_CREDS})")
    p.add_argument("--min-ttl-seconds", type=int,
                   default=DEFAULT_MIN_TTL_SECONDS,
                   help="Keep on-disk AT lifetime above this floor. For Cage "
                        "set >= trial timeout + margin (default "
                        f"{DEFAULT_MIN_TTL_SECONDS}).")
    p.add_argument("--http-proxy", default=os.environ.get("HTTPS_PROXY")
                   or os.environ.get("https_proxy") or "",
                   help="HTTPS proxy URL for reaching auth.openai.com "
                        "(defaults to $HTTPS_PROXY)")
    p.add_argument("--request-timeout", type=float,
                   default=DEFAULT_REQUEST_TIMEOUT,
                   help=f"HTTP timeout for the refresh POST (default "
                        f"{DEFAULT_REQUEST_TIMEOUT}s)")
    p.add_argument("--daemon", action="store_true",
                   help="Run as a resident loop instead of one-shot")
    p.add_argument("--retry-backoff-seconds", type=float,
                   default=DEFAULT_RETRY_BACKOFF_SECONDS,
                   help="(daemon) wait after a failed refresh before retry "
                        f"(default {DEFAULT_RETRY_BACKOFF_SECONDS}s)")
    p.add_argument("--max-sleep-seconds", type=float,
                   default=DEFAULT_MAX_SLEEP_SECONDS,
                   help="(daemon) cap on a single sleep so external file "
                        f"changes are noticed (default {DEFAULT_MAX_SLEEP_SECONDS}s)")
    p.add_argument("--lockfile", default="",
                   help="(daemon) single-instance lock path "
                        "(default: <creds>.refresh.lock)")
    p.add_argument("--force", action="store_true",
                   help="(one-shot) refresh regardless of TTL")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.daemon:
        return run_daemon(args)
    return run_once(args)


if __name__ == "__main__":
    sys.exit(main())
