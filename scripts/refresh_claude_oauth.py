#!/usr/bin/env python3
"""Keep a Claude Code OAuth ``.credentials.json`` fresh on disk.

Why this exists
---------------
The host's interactive ``claude`` sessions hold token state in memory and
flush to disk only on idle/exit; between writes the on-disk
``.credentials.json`` can stay stale for many hours. Cage trial
containers receive that on-disk snapshot at trial start (copy semantics)
and, because they have no outbound route to the OAuth endpoint, cannot
refresh themselves — Anthropic then responds with ``HTTP 401 Invalid
authentication credentials``.

This script keeps the file's access token fresh so every container
copies a token whose remaining lifetime exceeds the trial duration.

A refresh is a pure OAuth token exchange (``POST`` to
``platform.claude.com/v1/oauth/token`` with the refresh token) — it
consumes **no inference / usage quota**. Each refresh rotates the refresh
token (the old one dies), so refreshes must be serialized: run exactly
ONE instance of this daemon, and keep the trial concurrency model relying
on it (do not also let containers self-refresh).

Two modes
---------
* one-shot (default): refresh iff the AT drops below ``--min-ttl-seconds``,
  then exit. Good for cron.
* daemon (``--daemon``): resident loop. Reads ``expiresAt`` and sleeps
  precisely until the token would fall below ``--min-ttl-seconds``, then
  refreshes. No fixed poll interval; wake time is driven by the token's
  own expiry.

The ``--min-ttl-seconds`` knob is the floor of remaining lifetime kept on
disk. For Cage it MUST be >= the trial timeout (+ margin), NOT 5 minutes:
a container can start at any point in the refresh sawtooth, and it cannot
refresh mid-trial, so the token it copies must outlive the whole trial.

Usage:
    # one-shot (cron): refresh if <2.5h remains
    HTTPS_PROXY=http://127.0.0.1:7890 \\
        python refresh_claude_oauth.py --min-ttl-seconds 9000

    # daemon: keep >=2.5h of life on disk at all times
    HTTPS_PROXY=http://127.0.0.1:7890 \\
        python refresh_claude_oauth.py --daemon --min-ttl-seconds 9000

Exit codes (one-shot):
    0   refreshed OR no refresh needed
    1   refresh failed (RT rejected, network error, …)
    2   bad input (missing/unreadable file, no refresh token)
    3   another daemon instance already holds the lock (daemon mode)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REFRESH_URL = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"  # public Claude Code client_id
DEFAULT_CREDS = "~/.claude_p/.credentials.json"

# Floor of on-disk AT lifetime. >= Cage trial timeout (7200s) + margin so a
# container that copies the token always has enough life to finish a trial.
DEFAULT_MIN_TTL_SECONDS = 9000          # 2.5 h
DEFAULT_REQUEST_TIMEOUT = 30            # http timeout for the refresh POST
DEFAULT_RETRY_BACKOFF_SECONDS = 60      # wait after a failed refresh before retry
DEFAULT_MAX_SLEEP_SECONDS = 1800        # cap a single sleep so external file
                                        # changes / clock drift are noticed


def _log(msg: str) -> None:
    ts = time.strftime("%FT%T%z")
    sys.stderr.write(f"[{ts}] refresh_claude_oauth: {msg}\n")
    sys.stderr.flush()


def _load(path: Path) -> dict:
    with path.open() as fp:
        return json.load(fp)


def _ttl_seconds(creds: dict) -> float:
    """Remaining access-token lifetime in seconds (may be negative)."""
    exp = int((creds.get("claudeAiOauth") or {}).get("expiresAt") or 0)
    return (exp - int(time.time() * 1000)) / 1000.0


def _write_inplace(path: Path, data: dict) -> None:
    """Overwrite ``path`` IN PLACE, keeping the same inode.

    Deliberately NOT an atomic tmp+rename: a single-file Docker bind mount
    binds the inode at container-start, so a rename (new inode) would leave
    running containers reading the orphaned old inode forever. Writing the
    same inode lets bind-mounted containers see the refreshed token.

    The write is one ``write()`` of the full payload followed by a truncate
    to the new length. The payload is ~500 bytes, so the write lands in a
    single syscall; the only torn-read window is the rare case where the new
    content is shorter than the old (between write and truncate). For a
    credentials file that window is sub-millisecond and the file is read at
    most a few hundred times per run — negligible in practice.
    """
    payload = (json.dumps(data, indent=2) + "\n").encode()
    # ``r+b`` opens the existing inode without truncating (unlike ``w``).
    with open(path, "r+b") as fp:
        fp.seek(0)
        fp.write(payload)
        fp.truncate(len(payload))
        fp.flush()
        os.fsync(fp.fileno())


def _refresh(refresh_token: str, http_proxy: str | None, timeout: float) -> dict:
    """POST to the OAuth refresh endpoint; return the new token payload."""
    body = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    }).encode()
    req = urllib.request.Request(
        REFRESH_URL,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "cage-refresh-claude-oauth/2",
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
    """Refresh using the on-disk RT and write the new pair back.

    Re-reads the file immediately before refreshing so a concurrent
    external write (e.g. host CLI) is picked up. Returns the new remaining
    TTL in seconds. Raises on any failure.
    """
    creds = _load(path)
    oauth = creds.get("claudeAiOauth") or {}
    rt = oauth.get("refreshToken")
    if not rt:
        raise RuntimeError("no refreshToken in credentials")

    payload = _refresh(rt, http_proxy, timeout)
    new_at = payload.get("access_token")
    new_rt = payload.get("refresh_token") or rt
    expires_in = int(payload.get("expires_in") or 0)
    if not new_at or expires_in <= 0:
        raise RuntimeError(f"malformed refresh response keys={list(payload)}")

    oauth["accessToken"] = new_at
    oauth["refreshToken"] = new_rt
    oauth["expiresAt"] = int(time.time() * 1000) + expires_in * 1000
    if "scopes" in payload:
        oauth["scopes"] = payload["scopes"]
    creds["claudeAiOauth"] = oauth
    _write_inplace(path, creds)
    return _ttl_seconds(creds)


def run_once(args: argparse.Namespace) -> int:
    path = Path(args.creds).expanduser()
    if not path.is_file():
        _log(f"FATAL: {path} not found")
        return 2
    try:
        creds = _load(path)
    except Exception as exc:  # noqa: BLE001
        _log(f"FATAL: cannot parse {path}: {exc}")
        return 2

    ttl = _ttl_seconds(creds)
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
        body = exc.read().decode("utf-8", "replace")[:500]
        _log(f"FAIL: HTTP {exc.code} {exc.reason}: {body}")
        return 1
    except Exception as exc:  # noqa: BLE001
        _log(f"FAIL: {exc}")
        return 1
    _log(f"OK: AT rotated, now valid for {new_ttl:.0f}s")
    return 0


def run_daemon(args: argparse.Namespace) -> int:
    path = Path(args.creds).expanduser()

    # Single-instance guard: two daemons would race each other on the
    # one-shot refresh token.
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
            creds = _load(path)
        except Exception as exc:  # noqa: BLE001
            _log(f"load failed: {exc}; retry in {args.retry_backoff_seconds}s")
            time.sleep(args.retry_backoff_seconds)
            continue

        ttl = _ttl_seconds(creds)
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
                # AT lifetime is shorter than the floor we want — refreshing
                # again immediately would hammer the endpoint. Back off.
                _log(f"WARN: fresh AT life {new_ttl:.0f}s <= min_ttl "
                     f"{args.min_ttl_seconds}s (min_ttl too high?); "
                     f"sleeping {args.max_sleep_seconds}s to avoid hammering")
                time.sleep(args.max_sleep_seconds)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            _log(f"refresh FAIL: HTTP {exc.code} {exc.reason}: {body}; "
                 f"retry in {args.retry_backoff_seconds}s")
            time.sleep(args.retry_backoff_seconds)
        except Exception as exc:  # noqa: BLE001
            _log(f"refresh FAIL: {exc}; retry in {args.retry_backoff_seconds}s")
            time.sleep(args.retry_backoff_seconds)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--creds", default=DEFAULT_CREDS,
                   help=f"Path to .credentials.json (default {DEFAULT_CREDS})")
    p.add_argument("--min-ttl-seconds", type=int,
                   default=DEFAULT_MIN_TTL_SECONDS,
                   help="Keep on-disk AT lifetime above this floor. For Cage "
                        "set >= trial timeout + margin (default "
                        f"{DEFAULT_MIN_TTL_SECONDS}).")
    p.add_argument("--http-proxy", default=os.environ.get("HTTPS_PROXY")
                   or os.environ.get("https_proxy") or "",
                   help="HTTPS proxy URL for reaching platform.claude.com "
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
