"""Container-only runtime that puts LangGraph node identity ON THE WIRE.

Lives in the agent **image**, never in the Cage host package (it imports
``langchain_core`` / patches ``httpx``, both present only in the agent image),
and is loaded at interpreter startup by the sibling ``sitecustomize.py``.

When ``CAGE_TRACE`` is set it does two things, with **zero agent code**:

1. Registers a global LangChain callback (the same ``register_configure_hook``
   mechanism LangSmith uses) that, on every chat/LLM call, stashes the current
   LangGraph node (``metadata['langgraph_node']``) plus the run/parent ids in a
   context var.
2. Wraps ``httpx``'s request send so each outgoing model request carries those
   as ``X-Cage-Node`` / ``X-Cage-Run-Id`` / ``X-Cage-Parent-Id`` headers.

The Cage sidecar proxy then records those headers per request in ``proxy.jsonl``
(and strips them before forwarding upstream), so the ONE trajectory artifact is
structure-aware — no separate trace file, no separate view. Everything here is
best-effort: missing langchain/httpx or an unset env var is a silent no-op.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

# Headers to inject on the next outgoing model request (set by the callback,
# read by the httpx wrapper). Per-context, overwritten each call — no leakage.
cage_span_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "cage_span_headers", default=None
)


def _patch_httpx() -> None:
    try:
        import httpx
    except Exception:
        return

    def _inject(request: Any) -> None:
        try:
            headers = cage_span_headers.get()
            if headers:
                for key, value in headers.items():
                    request.headers[key] = value
        except Exception:
            pass

    send = httpx.Client.send
    if not getattr(send, "_cage_wrapped", False):
        def wrapped(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            _inject(request)
            return send(self, request, *args, **kwargs)
        wrapped._cage_wrapped = True  # type: ignore[attr-defined]
        try:
            httpx.Client.send = wrapped  # type: ignore[assignment]
        except Exception:
            pass

    asend = httpx.AsyncClient.send
    if not getattr(asend, "_cage_wrapped", False):
        async def awrapped(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            _inject(request)
            return await asend(self, request, *args, **kwargs)
        awrapped._cage_wrapped = True  # type: ignore[attr-defined]
        try:
            httpx.AsyncClient.send = awrapped  # type: ignore[assignment]
        except Exception:
            pass


def _register() -> None:
    if not os.environ.get("CAGE_TRACE"):
        return
    try:
        from langchain_core.tracers.context import register_configure_hook

        from cage_trace.handler import CageSpanHandler
    except Exception:
        return  # not a LangChain agent — no-op
    _patch_httpx()
    try:
        register_configure_hook(
            ContextVar("cage_span", default=None), True, CageSpanHandler, "CAGE_TRACE"
        )
    except Exception:
        return


_register()
