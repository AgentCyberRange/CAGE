"""Built-in agent package registration helpers."""

from __future__ import annotations


def register_builtin_agents() -> None:
    """Import built-in agent packages so their ``AgentType`` classes register."""

    import cage.agents.claude_code  # noqa: F401
    import cage.agents.codex  # noqa: F401
    import cage.agents.hermes  # noqa: F401
    import cage.agents.kimi_code  # noqa: F401
    import cage.agents.qwen_code  # noqa: F401


__all__ = ["register_builtin_agents"]
