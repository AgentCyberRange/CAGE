"""Registry of agent implementations available to Cage."""

from __future__ import annotations

from cage.agents.base.definition import AgentType

_AGENT_TYPE_REGISTRY: dict[str, type[AgentType]] = {}


def register_agent_type(cls: type[AgentType]) -> type[AgentType]:
    """Register an ``AgentType`` class under its public name."""

    _AGENT_TYPE_REGISTRY[cls.name] = cls
    return cls


def get_agent_type(name: str) -> AgentType:
    """Instantiate a registered agent type by name."""

    cls = _AGENT_TYPE_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_AGENT_TYPE_REGISTRY.keys()))
        raise ValueError(f"Unknown agent type: {name}. Available: {available}")
    return cls()
