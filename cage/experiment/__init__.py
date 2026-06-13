"""The experiment domain.

This package owns the central concept the framework is built around. It is
split by altitude: :mod:`cage.experiment.model` says *what an experiment is*
(the declarative spec → plan → record data model plus the live trial and
lifecycle events), and the engine (added in a later step) says *how it runs*.
"""
