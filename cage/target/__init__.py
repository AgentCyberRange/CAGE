"""The target domain: provisioning, the target server, and its client.

One package owns everything about evaluation targets — the docker-compose
benchmark instances an agent attacks:

  * ``provisioning`` — host-side launch/attach/teardown that wires an agent
    container to its target (the embedded target server, isolation networks).
  * ``build`` / ``check`` / ``debug`` — build, readiness-check, and manual
    debug workflows.
  * ``server`` — the out-of-process FastAPI service that owns docker-compose
    target lifecycle (used by ``cage serve`` and the embedded launcher).
  * ``client`` / ``scope`` / ``adapters`` / ``services`` — the in-process
    client, scope helpers, source adapters, and live-check/submit services.
  * ``compose_files`` — compose-stack loading/expansion shared by the server
    runtime and benchmark authors.

Public surface (importable directly from ``cage.target``):
"""

from cage.target.client import (
    BackendStrategy,
    ChallengeClient,
    ChallengeClientConfig,
    LocalBackend,
    RemoteBackend,
    SSHConfig,
    TargetTeardownResult,
)
from cage.target.compose_files import (
    expand_compose_env_values,
    load_compose_stack,
)
from cage.target.scope import (
    normalize_target_scope,
    resolve_target_scope,
)

__all__ = [
    "BackendStrategy",
    "ChallengeClient",
    "ChallengeClientConfig",
    "LocalBackend",
    "RemoteBackend",
    "SSHConfig",
    "TargetTeardownResult",
    "expand_compose_env_values",
    "load_compose_stack",
    "normalize_target_scope",
    "resolve_target_scope",
]
