"""The experiment engine: how a planned experiment actually runs.

The conductor (:func:`cage.experiment.engine.conductor.run_experiment`) drives a
run; the trial runner executes one trial; scheduler, run-cleanup, resource
recorder, preflight, termination classification, and live monitoring are its
collaborators. The engine depends on the model, the sandbox substrate, proxy,
target, artifacts, and scoring — i.e. on everything below it. Nothing below the
engine imports it.
"""
