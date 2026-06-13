"""Base contracts shared by concrete agent implementations."""

from cage.agents.base.definition import AgentType
from cage.agents.base.instance import AgentInstance
from cage.agents.base.registry import (
    _AGENT_TYPE_REGISTRY,
    get_agent_type,
    register_agent_type,
)
from cage.agents.base.resources import AgentContainerResources, HostRunService

__all__ = [
    "AgentContainerResources",
    "AgentInstance",
    "AgentType",
    "HostRunService",
    "_AGENT_TYPE_REGISTRY",
    "get_agent_type",
    "register_agent_type",
]
