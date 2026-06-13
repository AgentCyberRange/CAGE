"""Built-in benchmark registry for user-facing Cage CLI commands."""

from __future__ import annotations

import csv
import functools
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cage.benchmarks import BenchmarkOption
from cage.benchmarks.loader import load_benchmark_from_module
from cage.contracts.execution import classify_max_rounds, resolve_max_rounds
from cage.contracts.sample_keys import SAMPLE_MAX_ROUNDS_KEY

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True)
class BenchmarkSpec:
    """User-facing benchmark entry."""

    id: str
    display_name: str
    description: str
    project_file: Path
    aliases: tuple[str, ...] = ()

    @property
    def resolved_project_file(self) -> Path:
        if self.project_file.is_absolute():
            return self.project_file
        return REPO_ROOT / self.project_file


class UnknownBenchmarkError(ValueError):
    """Raised when a CLI benchmark id or alias is not registered."""


@functools.lru_cache(maxsize=1)
def _registered_specs() -> tuple[BenchmarkSpec, ...]:
    """Discover benchmarks from example ``registration:`` blocks.

    Every example that declares a top-level ``registration`` mapping with an
    ``id`` registers itself. Adding a benchmark therefore needs zero edits to
    the framework — only a new ``examples/<name>/`` whose YAML declares a
    registration block. Entries are ordered by their declared ``order``.
    """
    discovered: list[tuple[int, str, BenchmarkSpec]] = []
    if not EXAMPLES_DIR.is_dir():
        return ()
    for yml in sorted(EXAMPLES_DIR.glob("*/*.yml")):
        # ``local*.yml`` are the user's private, git-ignored run configs
        # (see ``.gitignore``: ``examples/*/local*.yml``). They are never the
        # canonical registration surface — only committed ``default_*.yml`` are
        # — so a local override must not hijack the ``cage benchmark list``
        # entry (or get printed as the project path users are told to run).
        if yml.name.startswith("local"):
            continue
        try:
            raw = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict):
            continue
        reg = raw.get("registration")
        if not isinstance(reg, dict) or not reg.get("id"):
            continue
        spec = BenchmarkSpec(
            id=str(reg["id"]),
            display_name=str(reg.get("display_name") or reg["id"]),
            description=str(reg.get("description") or ""),
            project_file=yml.relative_to(REPO_ROOT),
            aliases=tuple(str(alias) for alias in (reg.get("aliases") or ())),
        )
        order = reg.get("order")
        order_key = order if isinstance(order, int) else 1_000_000
        discovered.append((order_key, spec.id, spec))
    discovered.sort(key=lambda item: (item[0], item[1]))
    return tuple(spec for _, _, spec in discovered)


def normalize_benchmark_key(value: str) -> str:
    """Normalize ids and aliases for forgiving CLI lookup."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def list_benchmarks() -> list[BenchmarkSpec]:
    """Return registered built-in benchmarks in display order."""
    return list(_registered_specs())


def resolve_benchmark(value: str) -> BenchmarkSpec:
    """Resolve a benchmark id or alias."""
    wanted = normalize_benchmark_key(value)
    specs = _registered_specs()
    for spec in specs:
        keys = [spec.id, *spec.aliases]
        if any(normalize_benchmark_key(key) == wanted for key in keys):
            return spec
    available = ", ".join(spec.id for spec in specs)
    raise UnknownBenchmarkError(
        f"Unknown benchmark {value!r}. Available benchmarks: {available}"
    )


def is_registered_benchmark(value: str) -> bool:
    """Return whether ``value`` resolves to a registered benchmark."""
    try:
        resolve_benchmark(value)
    except UnknownBenchmarkError:
        return False
    return True


def load_project_yaml(project_file: Path) -> dict[str, Any]:
    """Read a project.yml file without constructing an experiment."""
    raw = yaml.safe_load(project_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{project_file} must contain a YAML mapping")
    return raw


def _build_benchmark_instance(
    project_file: Path, raw: dict[str, Any] | None = None
) -> Any | None:
    """Construct the configured Benchmark from a project file (without ``setup()``).

    Reads the ``eval.benchmark`` block, imports the declared module/class, and
    instantiates it with the declared kwargs. Returns ``None`` when the project
    declares no benchmark module. The framework hardcodes no benchmark name — it
    only follows the example's own ``module``/``class`` declaration.
    """
    project_file = project_file.resolve()
    if raw is None:
        raw = load_project_yaml(project_file)
    eval_raw = raw.get("eval", {})
    bench_cfg = eval_raw.get("benchmark", eval_raw) if isinstance(eval_raw, dict) else {}
    if not isinstance(bench_cfg, dict) or "module" not in bench_cfg:
        return None
    module_path = project_file.parent / bench_cfg["module"]
    kwargs = {k: v for k, v in bench_cfg.items() if k not in ("module", "class")}
    return load_benchmark_from_module(
        module_path,
        class_name=bench_cfg.get("class"),
        kwargs=kwargs,
    )


def load_benchmark_options(project_file: Path) -> list[BenchmarkOption]:
    """Instantiate the configured Benchmark class just enough to read CLI options."""
    benchmark = _build_benchmark_instance(project_file)
    if benchmark is None:
        return []
    options_fn = getattr(benchmark, "cli_options", None)
    if not callable(options_fn):
        return []
    return list(options_fn())


#: Cap on samples consulted when resolving a deferred round budget for help text.
#: Per-sample budgets are uniform (or a tiny per-profile set) within one project,
#: so a small prefix settles the distinct set without iterating a large dataset.
_ROUND_BUDGET_SAMPLE_CAP = 128


def project_round_budget_default_label(
    project_file: Path, raw: dict[str, Any]
) -> str | None:
    """What ``runtime.max_rounds: -1`` (defer) actually resolves to for a benchmark.

    ``-1`` is a sentinel meaning "use the benchmark/sample default"; surfacing the
    bare sentinel in ``--help`` is opaque. This builds the benchmark, reads each
    sample's own declared round budget, resolves it against the project's runtime
    value, and returns a display label — ``"150"`` (one value) or
    ``"mixed: 150, 500"`` (several). The number comes from the samples themselves,
    so the framework hardcodes no per-domain default.

    Returns ``None`` (caller keeps the raw "-1 defers" text) when the runtime value
    is not a defer, when the project declares no benchmark module, or when the
    samples cannot be read cheaply (missing dataset / construction error).
    """
    runtime = raw.get("runtime", {})
    execution_value = runtime.get("max_rounds") if isinstance(runtime, dict) else None
    kind, _ = classify_max_rounds(execution_value)
    if kind not in ("defer", "inherit"):
        # A concrete count / unlimited already renders meaningfully; nothing defers.
        return None
    try:
        benchmark = _build_benchmark_instance(project_file, raw)
        if benchmark is None:
            return None
        setup = getattr(benchmark, "setup", None)
        if callable(setup):
            setup()
        values: set[int] = set()
        for sample in itertools.islice(
            benchmark.iter_samples(), _ROUND_BUDGET_SAMPLE_CAP
        ):
            if not isinstance(sample, dict):
                continue
            resolved = resolve_max_rounds(
                None, execution_value, sample.get(SAMPLE_MAX_ROUNDS_KEY)
            )
            if resolved >= 0:
                values.add(resolved)
    except Exception:  # noqa: BLE001 - help text is best-effort; fall back to raw
        return None
    if not values:
        return None
    labels = [str(value) for value in sorted(values)]
    return labels[0] if len(labels) == 1 else "mixed: " + ", ".join(labels)


def project_agent_ids(raw: dict[str, Any]) -> list[str]:
    """Return configured agent ids from a raw project mapping."""
    ids: list[str] = []
    for agent in raw.get("agents", []) or []:
        if isinstance(agent, dict) and agent.get("id"):
            ids.append(str(agent["id"]))
    return ids


def project_agent_model_matrix(raw: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Return user-facing agent class to model ids from a raw project mapping."""
    subjects = _subject_ids(raw.get("subjects", []) or [])
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for agent in raw.get("agents", []) or []:
        if not isinstance(agent, dict):
            continue
        name = str(agent.get("id") or agent.get("kind") or agent.get("agent_type") or "")
        if not name:
            continue
        models: list[str]
        if agent.get("model"):
            models = [str(agent["model"])]
        elif agent.get("models"):
            models = _model_ids(agent["models"])
        else:
            models = subjects
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        for model_id in models:
            if model_id and model_id not in grouped[name]:
                grouped[name].append(model_id)
    return [(name, grouped[name]) for name in order]


def project_benchmark_root(project_file: Path, raw: dict[str, Any]) -> Path | None:
    """Resolve ``eval.benchmark.benchmark_root`` relative to the project file."""
    bench_cfg = _benchmark_config(raw)
    root = bench_cfg.get("benchmark_root")
    if not root:
        return None
    root_path = Path(str(root)).expanduser()
    if root_path.is_absolute():
        return root_path.resolve()
    return (project_file.parent / root_path).resolve()


def project_sample_summary(project_file: Path, raw: dict[str, Any]) -> str:
    """Best-effort sample count without constructing benchmark services.

    Driven by the example's declared ``registration.samples`` block (a CSV path
    or a JSON index, optionally multiplied by a declared level key). The
    framework reads the declaration; it knows no benchmark-specific filenames.
    """
    bench_cfg = _benchmark_config(raw)
    samples = _registration_samples(raw)
    eval_raw = raw.get("eval", {}) if isinstance(raw.get("eval"), dict) else {}
    limit = eval_raw.get("limit")
    suffix = f"; project limit: {limit}" if limit is not None else ""

    try:
        csv_rel = samples.get("csv")
        if csv_rel:
            path = (project_file.parent / str(csv_rel)).resolve()
            count = _count_csv_rows(path, split=str(bench_cfg.get("split", "test")))
            return f"{count} ({_display_path(path)}{suffix})"

        root = project_benchmark_root(project_file, raw)
        index = samples.get("index")
        if root is not None and index:
            count, detail = _count_indexed_samples(root, str(index), samples, bench_cfg)
            if count is not None:
                return f"{count} ({detail}{suffix})"
            return f"unavailable ({_display_path(root)} has no {index}{suffix})"

        dataset_path = bench_cfg.get("dataset_path")
        if dataset_path:
            path = Path(str(dataset_path)).expanduser()
            if not path.is_absolute():
                path = (project_file.parent / path).resolve()
            count = _count_csv_rows(path, split=str(bench_cfg.get("split", "test")))
            return f"{count} ({_display_path(path)}{suffix})"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__}: {exc})"
    return f"unavailable (no dataset declared{suffix})"


def _registration_samples(raw: dict[str, Any]) -> dict[str, Any]:
    reg = raw.get("registration")
    if not isinstance(reg, dict):
        return {}
    samples = reg.get("samples")
    return samples if isinstance(samples, dict) else {}


def _benchmark_config(raw: dict[str, Any]) -> dict[str, Any]:
    eval_raw = raw.get("eval", {})
    if not isinstance(eval_raw, dict):
        return {}
    bench_cfg = eval_raw.get("benchmark", eval_raw)
    return bench_cfg if isinstance(bench_cfg, dict) else {}


def _model_ids(raw: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = str(item.get("id") or item.get("model") or "")
        else:
            model_id = ""
        model_id = str(model_id).strip()
        if model_id:
            ids.append(model_id)
    return ids


def _subject_ids(raw: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if isinstance(item, str):
            subject_id = item
        elif isinstance(item, dict):
            subject_id = str(item.get("id") or "")
        else:
            subject_id = ""
        subject_id = str(subject_id).strip()
        if subject_id:
            ids.append(subject_id)
    return ids


def _count_indexed_samples(
    root: Path,
    index: str,
    samples: dict[str, Any],
    bench_cfg: dict[str, Any],
) -> tuple[int | None, str]:
    """Count entries in a declared JSON index, optionally per declared level."""
    path = root / index
    if not path.is_file():
        return None, ""
    base = _count_json_entries(path)
    multiplier_key = samples.get("multiplier_levels")
    if multiplier_key:
        levels = bench_cfg.get(str(multiplier_key))
        level_count = len(levels) if isinstance(levels, list) and levels else 1
        unit = str(multiplier_key).removesuffix("_levels")
        return base * level_count, f"{base} tasks x {level_count} {unit} level(s)"
    return base, _display_path(path)


def _count_json_entries(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, (dict, list)):
        return len(raw)
    return 0


def _count_csv_rows(path: Path, *, split: str) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return sum(1 for row in rows if not row.get("split") or row.get("split") == split)


def _display_path(path: Path) -> str:
    absolute = path if path.is_absolute() else (REPO_ROOT / path)
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(absolute)
