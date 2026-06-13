"""Background cache warmer for the web inspector.

The inspector serves run summaries from process-global caches in
``cage.web.data`` / ``cage.web.cache``. The *first* scan of a NAS-backed
``.cage_runs`` tree is expensive — hundreds of ``stat``/``scandir`` syscalls per
dashboard-less run, ~10 ms each — so doing it on the request path makes the
first page load (and every poll until a run settles) slow. This warmer runs the
scan in a daemon thread off the request path, so by the time a browser polls
``/api/runs`` the shared cache is already warm.

Pure stdlib plus one call into ``cage.web.data.scan_runs`` (which takes ``root``
explicitly and needs no Flask app context). Best-effort: it never raises into
the server and never blocks request handling — the data-layer caches it
populates use their own short-lived locks.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

# Re-scan cadence once warm. The first pass over a cold NAS tree can take
# minutes; subsequent passes are cheap (settled runs hit the shallow-signature
# fast path in ``cage.web.data``) so a short interval keeps live runs fresh
# without adding meaningful NAS load.
_WARM_INTERVAL_S = 30.0


def start_cache_warmer(root: Path, *, interval_s: float = _WARM_INTERVAL_S) -> threading.Thread:
    """Spawn a daemon thread that keeps ``root``'s run cache warm.

    Returns the thread (already started). Daemon, so it dies with the process
    and never blocks shutdown.
    """

    def _loop() -> None:
        from cage.web.data import scan_runs

        while True:
            try:
                scan_runs(root)
            except Exception:
                # Warming is strictly an optimization — a transient FS/NAS error
                # must never take down the inspector. Swallow and retry next tick.
                pass
            time.sleep(interval_s)

    thread = threading.Thread(target=_loop, name="cage-inspect-warmer", daemon=True)
    thread.start()
    return thread
