"""The LangChain callback that stashes the current node for the httpx wrapper.

Container-only (imports ``langchain_core``). On every chat/LLM start it records
the LangGraph node (``metadata['langgraph_node']``) and the run tree ids into the
``cage_span_headers`` context var, which the ``httpx`` wrapper in ``__init__``
reads to stamp the outgoing model request. Best-effort — never raises into the
agent.
"""

from __future__ import annotations

from typing import Any

try:  # present only in the agent image
    from langchain_core.callbacks.base import BaseCallbackHandler
except Exception:  # pragma: no cover - host has no langchain at runtime here
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

from cage_trace import cage_span_headers


class CageSpanHandler(BaseCallbackHandler):  # type: ignore[misc]
    """Stamp the current LangGraph node + run ids for the next model request."""

    def _set(self, run_id: Any, parent_run_id: Any, metadata: Any) -> None:
        try:
            headers: dict[str, str] = {}
            node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
            if node:
                headers["X-Cage-Node"] = str(node)
            if run_id is not None:
                headers["X-Cage-Run-Id"] = str(run_id)
            if parent_run_id is not None:
                headers["X-Cage-Parent-Id"] = str(parent_run_id)
            # Always set (even if only run ids): a node-less call still gets a
            # span, and an empty dict clears any stale value from this context.
            cage_span_headers.set(headers)
        except Exception:
            pass

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *,
        run_id: Any = None, parent_run_id: Any = None,
        tags: Any = None, metadata: Any = None, **_: Any,
    ) -> None:
        self._set(run_id, parent_run_id, metadata)

    def on_llm_start(
        self, serialized: Any, prompts: Any, *,
        run_id: Any = None, parent_run_id: Any = None,
        tags: Any = None, metadata: Any = None, **_: Any,
    ) -> None:
        self._set(run_id, parent_run_id, metadata)
