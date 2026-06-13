"""Offline garbage collection for Cage runs.

This package owns the *offline* GC domain behind the ``cage gc`` command:
deciding whether a finished run's docker resources are orphaned (``runner``),
planning ledger-driven cleanup (``plan``), and projecting a resource-ledger
summary (``summary``). It is deliberately distinct from the two in-process
teardown concerns so that "cleanup" no longer names three different things:

- ``cage.experiment.engine.run_cleanup`` — in-run teardown of one experiment's resources.
- ``cage.target.local_cleanup`` — the target server's local docker sweep.
"""
