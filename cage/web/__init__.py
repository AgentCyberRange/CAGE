"""The Cage Inspector: a read-only web UI over run artifacts.

``cage inspect`` serves this FastAPI app to browse runs, trials, dashboards,
and proxy traces. It reads the artifact store; it never drives a run.
"""
