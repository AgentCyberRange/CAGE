"""Cage CLI — entry point for running experiments."""

from __future__ import annotations

import click

from cage.agents import register_builtin_agents
from cage.cli.commands import agent as agent_commands
from cage.cli.commands import benchmark as benchmark_commands
from cage.cli.commands import gc as gc_commands
from cage.cli.commands import inspect as inspect_commands
from cage.cli.commands import model as model_commands
from cage.cli.commands import run as run_commands
from cage.cli.commands import score as score_commands
from cage.cli.commands import serve as serve_commands
from cage.contracts.logging import LoggingConfig, setup_logging

register_builtin_agents()

@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Console log level",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, log_level: str) -> None:
    """CAGE — an evaluation framework for installed AI coding agents.

    CAGE runs each agent in its own container against a pluggable benchmark,
    intercepts every model call, snapshots state, and scores the trial.

    Start with `cage run`: it lists the benchmarks and walks you from a smoke
    trial to a full campaign. Every other command is one slice of a run —
    `benchmark` prepares it (and `benchmark serve` stands its targets up on
    their own as a browsable range, no run needed), `model` and `agent`
    configure it, `inspect` watches it, `score` re-grades it, and `gc` cleans
    up after it.

    Docs: https://github.com/AgentCyberRange/CAGE/tree/main/docs
    """
    ctx.ensure_object(dict)
    if verbose:
        log_level = "DEBUG"
    ctx.obj["log_level"] = log_level
    setup_logging(LoggingConfig(console_level=log_level))


main.add_command(agent_commands.agent_group)
main.add_command(benchmark_commands.benchmark_group)
main.add_command(gc_commands.gc)
main.add_command(inspect_commands.inspect)
main.add_command(model_commands.model_group)
main.add_command(run_commands.run)
main.add_command(score_commands.score)
# `serve` lives under the benchmark group — it stands up a benchmark's targets
# as a browsable range: `cage benchmark serve <benchmark>`.
benchmark_commands.benchmark_group.add_command(serve_commands.serve)


if __name__ == "__main__":
    main()
