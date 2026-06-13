"""The per-trial *target runtime-state* contract.

When a trial launches a target stack, two collaborators write target runtime
details into the benchmark sample dict under a single shared key:

- Layer-1 provisioning (:func:`cage.target.provisioning.inject_ctf_info`) adds
  the run-scoped bits it knows: ``project_name``, ``run_id``, ``scoring``.
- A CTF-style benchmark's ``prepare_trial`` (Layer-2) writes the full target
  descriptor it resolved from the challenge server.

Benchmark scoring/prompt code and Jinja prompt templates then read it back.

Historically this travelled as a fake-private, untyped magic-string key
(``"_runtime_state"``) that no layer declared — an implicit cross-layer config
channel hidden inside ``Trial.sample``. This module gives that contract one
named key and a declared schema. It is a *data* contract: the runtime
representation stays a plain ``dict`` so prompt templates can index it.

Layer-1 floor module: no benchmark names, no engine/target imports.
"""

from __future__ import annotations

from typing import Any, TypedDict

#: The sample-dict key under which target runtime state travels. Use this
#: constant instead of the literal string at every read/write site.
RUNTIME_STATE_KEY = "runtime_state"

#: Sample-dict key the engine sets to ``True`` once the live submit/check
#: service is attached to the trial container. Prompt templates branch on it
#: (``{% if instance_data.check_supported %}``) to tell the agent the live
#: checker exists — the key name is therefore a template contract and must
#: not change. Written at exactly one site (trial_runner, before
#: ``build_prompt``); declared here so the channel is named, not smuggled.
CHECK_SUPPORTED_KEY = "check_supported"


class RuntimeState(TypedDict, total=False):
    """Declared schema for the value stored at :data:`RUNTIME_STATE_KEY`.

    ``total=False``: provisioning writes a partial view (project/run/scoring)
    while a benchmark may write the full descriptor; both are valid.
    """

    benchmark: str
    sample_id: str
    challenge_id: str
    network_name: str | None
    network_subnet: str | None
    scoring: dict[str, Any]
    target_info: dict[str, Any]
    project_name: str
    run_id: str
