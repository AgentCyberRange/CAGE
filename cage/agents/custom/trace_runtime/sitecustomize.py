"""Auto-load Cage's LangChain trace hook at interpreter startup.

Python imports ``sitecustomize`` automatically if it is anywhere on
``sys.path``. The custom-agent image puts this directory on ``PYTHONPATH``, so
every ``python`` the agent runs imports ``cage_trace`` — which registers the
global LangChain callback when ``CAGE_TRACE`` is set (and is a silent no-op
otherwise). This is the ONLY thing that makes tracing zero-code for the agent.

Lives in the agent image only (copied by docker/custom_langgraph.Dockerfile);
never imported by the Cage host package.
"""

try:
    import cage_trace  # noqa: F401  (import side effect registers the hook)
except Exception:
    pass
