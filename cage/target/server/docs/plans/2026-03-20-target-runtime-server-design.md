# Target Runtime Server Design

## Goal

Refactor `challenge_server.py` into a generic target runtime management service that can launch, stop, monitor, and clean up arbitrary benchmark targets across multiple namespaces.

## Why

The current server is still modeled as a CTF-only service:

- file, class, function, and variable names are challenge-specific
- the active namespace is read from a global environment variable at process startup
- in-memory state and Docker project naming are keyed only by `chal_id`

That creates a correctness problem: the same target cannot run safely in two namespaces at the same time because instance bookkeeping, runtime compose filenames, Docker config directories, and cleanup logic overlap.

## Recommended Approach

Keep the existing HTTP route shape and move namespace selection to a request parameter.

- keep `GET /launch/{target_id}` for startup/reuse/recreate
- keep `DELETE /launch/{target_id}` for teardown
- add `namespace` as a query parameter
- default `namespace` to `default` for backward compatibility

Internally, treat every runtime as an instance identified by `(target_id, namespace)`.

### Why this approach

- It minimizes caller churn while fixing the isolation bug.
- It keeps routing and deployment simple.
- It makes namespace an explicit part of the runtime identity everywhere that matters.
- It supports parallel runs of the same target in different namespaces without introducing a new API surface.

## Rejected Alternatives

### Keep namespace as a process-level environment variable

This preserves the current collision problem and requires one server process per namespace.

### Create separate launch/stop endpoints with request bodies

This would work, but it creates a wider API change than needed. The current GET and DELETE contract can absorb `namespace` cleanly as a query parameter.

### Key runtime state only by `target_id` and store namespace inside the value

This still makes lookups, locking, cleanup, and monitoring ambiguous. The compound key should be the primary identity.

## Design

### File naming

Rename `challenge_server.py` to `target_runtime_server.py`.

The new name matches the broader responsibility:

- target startup
- target shutdown
- namespace-aware runtime materialization
- health monitoring
- Docker cleanup

### Naming cleanup

Rename server-facing and internal identifiers away from CTF-only language:

- `chal_id` -> `target_id`
- `running_instances` -> `runtime_instances`
- `ChallengeLockRegistry` -> `TargetLockRegistry`
- `ChallengeRecoveryCoordinator` -> `TargetRecoveryCoordinator`
- `launch_challenge` -> `launch_target`
- `stop_challenge` -> `stop_target`
- `load_all_challenges` -> `load_all_targets`

### Runtime identity

Introduce a normalized runtime key derived from:

- `target_id`
- `namespace`

This runtime key should be used for:

- in-memory runtime state
- per-instance locks
- recovery serialization
- monitor iteration
- cleanup and health checks

### Namespace handling

Add a small namespace normalization helper that:

- trims whitespace
- falls back to `default`
- converts to a Docker-safe token for project names and runtime filenames

Use the request namespace to derive:

- Docker network name
- Docker project name
- runtime compose filename
- `DOCKER_CONFIG` temp directory

### Docker/network behavior

Preserve the current shared external network model, but make the network namespace-specific per request:

- `ctfnet_<namespace>`

The startup hook should only start the monitor. Network creation should happen at launch time once the request namespace is known.

### Monitoring and cleanup

The monitor should scan runtime entries keyed by `(target_id, namespace)` and pass both values into health checks and recovery.

Shutdown cleanup should clear every tracked runtime instance regardless of namespace.

### Client compatibility

The server should accept requests without `namespace` and treat them as `default`.

This keeps existing callers working while enabling namespace-aware callers to opt into isolation immediately.

## Testing

Add or update focused unit tests for:

1. namespace normalization and runtime key generation
2. launch materialization uses namespace-specific project names and compose filenames
3. runtime state is isolated by `(target_id, namespace)`
4. cleanup only removes the requested namespace instance
5. lock/recovery coordination serializes by runtime key, not just target id
6. client wrapper passes `namespace` through request params and cache/tunnel bookkeeping
