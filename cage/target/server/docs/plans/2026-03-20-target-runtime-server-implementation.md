# Target Runtime Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename `challenge_server.py` to `target_runtime_server.py`, generalize naming, and make namespace a request-scoped runtime identity so the same target can run in parallel across namespaces.

**Architecture:** Keep the existing launch and stop routes, but add a `namespace` query parameter and refactor internal state to be keyed by `(target_id, namespace)`. Thread namespace through runtime materialization, Docker cleanup, locking, recovery, and monitor logic. Update local callers and tests to use the new generic module and namespace-aware request contract.

**Tech Stack:** Python, FastAPI, Docker SDK, unittest

---

### Task 1: Add failing tests for namespace-scoped runtime identity

**Files:**
- Modify: `../tests/test_challenge_server_registry.py`
- Modify: `../tests/test_challenge_server_runtime_locking.py`
- Modify: `../tests/test_challenge_client_registry.py`

**Step 1: Write the failing tests**

Add tests covering:

- namespace normalization and runtime key generation
- launch materialization builds namespace-specific project names and compose filenames
- runtime state stores the same target under different namespaces without collision
- recovery coordination serializes by runtime key
- manager requests forward `namespace` and isolate tunnel/cache bookkeeping by namespace

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_challenge_server_registry tests.test_challenge_server_runtime_locking tests.test_challenge_client_registry -v
```

Expected: failures because namespace-scoped helpers and request plumbing do not exist yet.

### Task 2: Refactor runtime guards to generic names

**Files:**
- Modify: `runtime_guards.py`

**Step 1: Rename lock and recovery types**

Rename:

- `ChallengeLockRegistry` -> `TargetLockRegistry`
- `ChallengeRecoveryCoordinator` -> `TargetRecoveryCoordinator`

Keep compatibility aliases if needed for existing imports during the transition.

**Step 2: Clarify API semantics**

Update parameter names from `chal_id` to `runtime_key` where the coordinator and lock registry are really serializing by runtime identity.

### Task 3: Rename and refactor the server module

**Files:**
- Create: `target_runtime_server.py`
- Modify: `challenge_server.py`

**Step 1: Copy the current server into the new module**

Keep `challenge_server.py` as a thin compatibility shim that re-exports from `target_runtime_server.py`.

**Step 2: Introduce namespace-aware runtime helpers**

Add helpers for:

- namespace normalization
- Docker-safe namespace token
- runtime key creation
- project name generation
- runtime compose filename generation
- namespace-specific Docker network naming

**Step 3: Refactor state and API naming**

Rename generic concepts:

- `chal_id` -> `target_id`
- `running_instances` -> `runtime_instances`
- `load_all_challenges` -> `load_all_targets`
- `launch_challenge` -> `launch_target`
- `stop_challenge` -> `stop_target`

**Step 4: Thread namespace through the full launch lifecycle**

Update:

- runtime lookup and storage
- lock and recovery lookup
- Docker cleanup
- health checks
- shutdown cleanup
- monitor-driven recovery

**Step 5: Keep route compatibility while adding namespace query params**

Expose:

- `GET /launch/{target_id}?namespace=...`
- `DELETE /launch/{target_id}?namespace=...`

Default to `default` when the query parameter is absent.

### Task 4: Update local callers and script wording

**Files:**
- Modify: `test_challenge_server.py`
- Modify: `../common/agent_runtime/challenge_client.py`

**Step 1: Pass namespace through request params**

Keep existing behavior when no namespace is configured.

**Step 2: Clean up generic wording where this code is clearly about target runtimes rather than CTF-only targets**

### Task 5: Verify the refactor

**Files:**
- Modify: `runtime_guards.py`
- Create: `target_runtime_server.py`
- Modify: `challenge_server.py`
- Modify: `test_challenge_server.py`
- Modify: `../tests/test_challenge_server_registry.py`
- Modify: `../tests/test_challenge_server_runtime_locking.py`
- Modify: `../tests/test_challenge_client_registry.py`
- Modify: `../common/agent_runtime/challenge_client.py`

**Step 1: Run focused tests**

Run:

```bash
python -m unittest tests.test_challenge_server_registry tests.test_challenge_server_runtime_locking tests.test_challenge_client_registry -v
```

Expected: pass

**Step 2: Run syntax verification**

Run:

```bash
python -m py_compile target_server/server/target_runtime_server.py target_server/server/challenge_server.py target_server/server/runtime_guards.py target_server/server/test_challenge_server.py common/agent_runtime/challenge_client.py
```

Expected: pass
