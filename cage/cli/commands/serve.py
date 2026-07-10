"""``cage serve`` — run the target range as a standalone, browsable service.

Cage always spawns an embedded target server *per run*; this command exposes
the same server on its own so an operator (or an external tester) can browse
the target library, launch/stop instances, and watch what is running — through
a web console at ``/`` or the JSON API directly. It is the standalone sibling
of the internal ``python -m cage.target.serve`` runner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click


def _repo_examples_root() -> Path:
    """The repo's ``examples/`` dir, resolved from the installed cage package."""
    import cage

    return Path(cage.__file__).resolve().parents[1] / "examples"


def _looks_like_challenge_index(path: Path) -> bool:
    """True if ``path`` is a challenge INDEX (a flat ``{id: {path, ...}}`` map).

    Distinguishes an index (``agent_pentest_bench.json``, ``post_exp_range.json``)
    from a per-challenge ``challenge.json`` (which has ``adapter_kind`` at top and
    no ``path``-bearing entries), so a benchmark's index roots can be discovered
    without hardcoding its dataset layout.
    """
    if path.name == "challenge.json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or not data:
        return False
    entries = [v for v in data.values() if isinstance(v, dict)]
    return bool(entries) and all("path" in v for v in entries[:5])


def _resolve_benchmark_sources(benchmark: str) -> list[dict[str, str]]:
    """Resolve a benchmark NAME to its ``challenge_json`` source roots.

    ``examples/<benchmark>/`` may hold challenge indices at more than one depth
    (e.g. agent_pentest_bench: ``datasets/agent_pentest_bench.json`` for web +
    ``datasets/post_exploit_bench/post_exp_range.json`` for post-exploitation).
    Each directory that contributes NEW challenges becomes one source root, so a
    single ``cage benchmark serve <benchmark>`` serves the whole benchmark.

    Two hazards this avoids:

    - **Never recurse.** Indices sit at the top of a dataset collection — the
      dataset root or one level down. A recursive ``rglob`` would descend into
      every challenge's app-source tree and read tens of thousands of unrelated
      ``*.json`` (node_modules, vendor, …) — minutes of NAS I/O, so the server
      appears to hang before it even binds. Scan only depths 0 and 1.
    - **Skip duplicate indices.** A dataset can ship both a top-level index and
      per-collection ones that re-list the same challenges (e.g.
      ``agent_pentest_bench.json`` and ``web_exploit_bench/pentest_bench.json``
      both list the 15 web challenges). Feeding both roots to the discovery
      registry raises a duplicate-id error, so prefer the shallower index and
      skip any whose challenge ids are already covered.

    Raises ``click.BadParameter`` if the benchmark or any index is missing.
    """
    bench_dir = _repo_examples_root() / benchmark
    if not bench_dir.is_dir():
        raise click.BadParameter(
            f"benchmark not found: examples/{benchmark}", param_hint="BENCHMARK"
        )
    search_root = bench_dir / "datasets"
    if not search_root.is_dir():
        search_root = bench_dir

    # Shallow (depth 0 + 1) only — NEVER rglob. Shallower indices first so a
    # top-level index wins over a per-collection duplicate.
    candidates = sorted(
        (
            p
            for p in [*search_root.glob("*.json"), *search_root.glob("*/*.json")]
            if _looks_like_challenge_index(p)
        ),
        key=lambda p: (len(p.relative_to(search_root).parts), str(p)),
    )

    roots: list[str] = []
    seen_ids: set[str] = set()
    for index_path in candidates:
        try:
            ids = set(json.loads(index_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if ids - seen_ids:  # contributes challenges not already covered
            roots.append(str(index_path.parent))
            seen_ids |= ids

    if not roots:
        raise click.BadParameter(
            f"no challenge index found under {search_root} — is the dataset "
            f"submodule checked out? (git submodule update --init)",
            param_hint="BENCHMARK",
        )
    return [{"adapter_kind": "challenge_json", "root": r} for r in roots]


def _find_nested_key(obj: Any, key: str) -> Any:
    """Depth-first search for a mapping-valued ``key`` anywhere in nested config."""
    if isinstance(obj, dict):
        found = obj.get(key)
        if isinstance(found, dict) and found:
            return found
        for value in obj.values():
            result = _find_nested_key(value, key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_nested_key(item, key)
            if result is not None:
                return result
    return None


def _resolve_benchmark_judge(benchmark: str) -> dict[str, Any] | None:
    """The benchmark's declared default judge config, or ``None``.

    A benchmark whose challenges need the ``LLM_judge`` signal declares its judge
    model in its example ``default_*.yml`` (e.g. agent_pentest_bench's
    ``default_web_exploit.yml`` → ``judge: {models: [deepseek-v4-pro]}``). Serving
    the benchmark should score web challenges out of the box with that model, so
    read it from there. ``--judge-model`` overrides it. The model id still
    resolves against the repo's ``config/models.yml`` at score time.
    """
    import yaml

    bench_dir = _repo_examples_root() / benchmark
    if not bench_dir.is_dir():
        return None
    for cfg in sorted(bench_dir.glob("*.yml")) + sorted(bench_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        judge = _find_nested_key(data, "judge")
        if isinstance(judge, dict) and judge:
            return judge
    return None


@click.command(name="serve")
@click.argument("benchmark", required=False)
@click.option(
    "--benchmark-root",
    "benchmark_root",
    default="",
    help=(
        "Explicit directory of challenge.json targets (overrides the BENCHMARK "
        "argument). Use when serving a dataset dir that is not laid out as "
        "examples/<benchmark>/."
    ),
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind. Use 0.0.0.0 to expose externally.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
@click.option("--namespace", default="default", show_default=True, help="Docker resource namespace (isolates this server's targets).")
@click.option(
    "--external-token",
    default="",
    help=(
        "Enable external-audience mode: non-loopback callers must send "
        "'Authorization: Bearer <token>'. Without it every caller is internal "
        "(legacy single-audience behaviour)."
    ),
)
@click.option(
    "--judge-model",
    default="",
    metavar="MODEL_ID",
    help=(
        "Override the judge model for the LLM_judge scoring signal (id from "
        "config/models.yml). By default the benchmark's own declared judge is "
        "used (e.g. agent_pentest_bench web → deepseek-v4-pro from "
        "default_web_exploit.yml), so you usually need not set this. Marker-only "
        "post-exploitation ranges need no model."
    ),
)
@click.option(
    "--prompt-level",
    "prompt_level",
    default="l0",
    show_default=True,
    type=click.Choice(["l0", "l1", "l2"]),
    help=(
        "DEFAULT hint tier for the agent-facing task briefing at GET /prompt "
        "(l0 = no hints; l1/l2 progressively reveal vuln location / topology). "
        "The effective tier is bound per instance at launch (GET /launch"
        "?prompt_level=), so it can vary without restarting serve; this is the "
        "fallback when a launch doesn't specify one."
    ),
)
@click.option(
    "--adapter",
    "adapters",
    multiple=True,
    help="Load an extra benchmark adapter ('path/to/module.py:ClassName'). Repeatable.",
)
@click.option("--open", "open_browser", is_flag=True, help="Open the console in a browser after start.")
def serve(
    benchmark: str | None,
    benchmark_root: str,
    host: str,
    port: int,
    namespace: str,
    external_token: str,
    judge_model: str,
    prompt_level: str,
    adapters: tuple[str, ...],
    open_browser: bool,
) -> None:
    """Serve a benchmark as a browsable target range (web console + JSON API).

    \b
    BENCHMARK is a name under examples/ (e.g. ``agent_pentest_bench``); its
    challenge indices — at every depth — are discovered and served together.

    \b
    Examples:
      cage benchmark serve agent_pentest_bench
      cage benchmark serve agent_pentest_bench --host 0.0.0.0 --external-token "$(openssl rand -hex 16)"
      cage benchmark serve --benchmark-root path/to/datasets   # explicit dir instead

    \b
    Console:  http://<host>:<port>/
    API:      GET /challenges · GET /launch/<id> · GET /prompt/<run_id> · POST /submit/<run_id> · DELETE /launch/<id>?run_id=<id>
    """
    if benchmark and benchmark_root:
        raise click.BadParameter(
            "pass BENCHMARK or --benchmark-root, not both", param_hint="BENCHMARK"
        )
    if benchmark:
        # Resolve the benchmark's (possibly multiple) index roots and hand them
        # to the server via the env the runner reads, so a single name serves
        # the whole benchmark. ``benchmark_root`` stays empty so the runner does
        # not overwrite this.
        os.environ["TARGET_SERVER_BENCHMARK_SOURCES_JSON"] = json.dumps(
            _resolve_benchmark_sources(benchmark)
        )
    elif benchmark_root:
        root = Path(benchmark_root).expanduser()
        if not root.is_dir():
            raise click.BadParameter(f"benchmark root not found: {root}", param_hint="--benchmark-root")

    # Judge model for the LLM_judge signal (read server-side, never from the
    # submission; models_file defaults to the repo's config/models.yml).
    # Precedence: explicit --judge-model > the benchmark's declared default.
    judge_label = ""
    if judge_model:
        os.environ["TARGET_SERVER_JUDGE_JSON"] = json.dumps({"model_id": judge_model})
        judge_label = judge_model
    elif benchmark:
        default_judge = _resolve_benchmark_judge(benchmark)
        if default_judge:
            os.environ["TARGET_SERVER_JUDGE_JSON"] = json.dumps(default_judge)
            judge_label = (
                str(default_judge.get("model_id") or default_judge.get("id")
                    or default_judge.get("model")
                    or ", ".join(map(str, default_judge.get("models") or [])))
                + " (benchmark default)"
            )

    # DEFAULT task-briefing hint tier for GET /prompt. The effective tier is
    # bound per instance at launch (?prompt_level=); this is only the fallback.
    os.environ["TARGET_SERVER_PROMPT_LEVEL"] = prompt_level

    shown_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    console_url = f"http://{shown_host}:{port}/"
    click.echo(click.style("  cage target range", fg="green", bold=True))
    if benchmark:
        click.echo(f"  benchmark: {benchmark}")
    click.echo(f"  console : {console_url}")
    click.echo(f"  api     : http://{shown_host}:{port}/challenges")
    if judge_label:
        click.echo(f"  judge   : {judge_label}")
    click.echo(f"  prompt  : {prompt_level} default (per-launch ?prompt_level= overrides; GET /prompt/<run_id>)")
    if external_token:
        click.echo("  audience: external ENABLED (bearer token required for non-loopback)")
    click.echo("")

    if open_browser:
        import webbrowser

        webbrowser.open(console_url)

    from cage.target.serve import run_target_server

    run_target_server(
        host=host,
        port=port,
        benchmark_root=benchmark_root,
        namespace=namespace,
        adapters=adapters,
        external_token=external_token,
        parent_pid=0,
    )
