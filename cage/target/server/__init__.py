"""Challenge server package — public API re-exports.

Tests and external callers reach into ``target_server.server`` for the
top-level helpers and FastAPI route handlers (e.g.
``target_server.server.resolve_parallel_mode``,
``target_server.server.running_instances``,
``target_server.server.stop_challenge``).

Keep this list explicit so refactors do not silently break test patches.
"""
from __future__ import annotations

from .challenge_server import (  # noqa: F401
    app,
    launch_challenge,
    monitor_instances,
    stop_challenge,
)
from .health_probes import (  # noqa: F401
    is_instance_healthy,
    is_port_open,
    probe_inner_service,
    wait_for_containers_running,
    wait_for_http_ready,
    wait_for_inner_services_ready,
    wait_for_ports_open,
    wait_for_services_healthy,
)
from .cleanup import _cleanup_instance_impl, cleanup_instance  # noqa: F401
from .launch import (  # noqa: F401
    _launch_challenge_impl,
    _reused_launch_response,
    resolve_parallel_mode,
    resolve_server_target_scope,
)
from .launch_workflow import pydantic_to_dict  # noqa: F401
from .network_debug import build_network_debug  # noqa: F401
from .network_admin import (  # noqa: F401
    cleanup_orphan_networks,
    ensure_docker_network,
    list_project_containers,
    remove_network_with_retry,
    resolve_service_inner_ips,
    summarize_project_containers,
)
from .schemas import (  # noqa: F401
    LaunchResponse,
    ServiceInfo,
    StopResponse,
)
from .server_state import (  # noqa: F401
    BENCHMARK_ROOT,
    DOCKER_NETWORK,
    challenge_locks,
    find_free_port,
    get_running_instance,
    invalidate_challenge_cache,
    load_all_challenges,
    load_benchmark_sources,
    pop_running_instance,
    recovery_coordinator,
    running_instances,
    running_instances_lock,
    set_running_instance,
    snapshot_running_instance_ids,
    update_running_instance,
)
