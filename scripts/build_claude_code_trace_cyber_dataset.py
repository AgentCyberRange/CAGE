#!/usr/bin/env python3
"""Build the compact Claude Code cyber trace dataset.

The source Cage runs store every proxy request, which repeats the full model
context at each step. This builder keeps the review-useful pieces only:
per-step thinking/text/tool/observation blocks plus compact prompt, status,
task identity, and score metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, NamedTuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_trajectory_events import build_step_records, write_jsonl  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/home/pgroup/pxd-team/workspace/fyh/datasets/claude-code-trace-cyber")
CYBER_EXAMPLE_DIRS = (
    "autopenbench",
    "cvebench",
    "nyu",
    "agent_pentest_bench",
)
PRUNE_DIRS = {
    "agent_shared",
    "initial_state",
    "logs",
    "runtime",
    "scores",
    "state_post",
    "state_pre",
    "workspace",
}
META_KEYS = (
    "trial_id",
    "trial_index",
    "trial_type",
    "sample_id",
    "status",
    "exit_code",
    "max_rounds",
    "termination_reason",
    "termination_detail",
    "termination_source",
    "live_success",
    "live_success_verdict",
    "terminated_by_live_success",
    "timing",
    "snapshot_failed",
)


class TrialSource(NamedTuple):
    model: str
    task_family: str
    run_variant: str
    group: str
    agent_label: str
    run_label: str
    state_kind: str
    source_base: Path
    source_run_dir: Path
    trials_dir: Path
    trial_dir: Path
    proxy_jsonl: Path
    trial_path: str
    is_before_resume: bool


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def slugify(value: str, *, max_len: int = 160) -> str:
    value = value.strip().replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    return (value or "trace")[:max_len]


def task_label(task_family: str) -> str:
    return task_family.replace("_", "-")


def group_key(model: str, task_family: str, run_variant: str) -> str:
    return slugify(f"{model}__{task_label(task_family)}__{run_variant}", max_len=200)


def classify_model(agent_label: str) -> str:
    low = agent_label.lower()
    if "claude-opus-4-7-cyber" in low:
        return "claude-opus-4-7-cyber"
    if "qwen3.7-max" in low or "qwen37max" in low:
        return "qwen3.7-max"
    if "deepseek-v4-pro" in low:
        return "deepseek-v4-pro"
    if "qwen36-27b" in low:
        return "qwen36-27b"
    if "glm-5.1" in low or "glm51" in low:
        return "glm-5.1"
    return ""


def is_selected_claude_code_agent(agent_label: str) -> bool:
    low = agent_label.lower()
    if "claude_code" not in low and "claudecode" not in low and "qwen_code" not in low:
        return False
    return bool(classify_model(agent_label))


def infer_state_kind(agent_label: str, trials_dir: Path) -> str:
    for value in ("stateful", "stateless"):
        if trials_dir.parent.name == value or agent_label.lower().endswith(f":{value}"):
            return value
    if ":stateful" in agent_label.lower():
        return "stateful"
    return "stateless"


def classify_group(base_path: str, agent_label: str, run_label: str) -> tuple[str, str, str]:
    model = classify_model(agent_label)
    task_family = Path(base_path).name
    low_agent = agent_label.lower()
    low_run = run_label.lower()

    if task_family == "cvebench":
        if model == "qwen36-27b":
            run_variant = "full-pass4" if "111807" in low_run else "warmup"
        else:
            run_variant = "legacy-baseline"
    elif task_family == "nyu":
        run_variant = "legacy-baseline"
    elif task_family == "agent_pentest_bench":
        if model == "claude-opus-4-7-cyber":
            if "smoke" in low_run:
                run_variant = "cyber-smoke"
            elif "p3" in low_run:
                run_variant = "postexp-cyber-pass3"
            else:
                run_variant = "postexp-cyber-passk3"
        elif model == "qwen3.7-max":
            run_variant = "postexp-pass3" if "p3" in low_run else "postexp-full"
        elif model == "deepseek-v4-pro":
            if "scorefix" in low_run:
                run_variant = "scorefix-sanity"
            elif "p3" in low_run:
                run_variant = "postexp-pass3"
            else:
                run_variant = "postexp-l0-pass1"
        elif "ctfenv" in low_agent:
            run_variant = "ctfenv-smoke"
        elif "postexp" in low_run:
            run_variant = "postexp-l0-l2-passk3"
        elif "full-passk3" in low_run:
            run_variant = "full-passk3"
        elif "mix15" in low_run:
            run_variant = "mix15-passk3"
        elif "5challenge" in low_run:
            run_variant = "five-challenge-smoke"
        else:
            run_variant = "small-sweep"
    else:
        run_variant = "selected"

    return model, task_family, run_variant


def iter_trial_proxies(trials_dir: Path) -> Iterable[Path]:
    for dirpath, dirnames, _filenames in os_walk_pruned(trials_dir):
        proxy_jsonl = Path(dirpath) / "proxy" / "proxy.jsonl"
        if proxy_jsonl.is_file():
            yield proxy_jsonl


def os_walk_pruned(root: Path) -> Iterable[tuple[str, list[str], list[str]]]:
    import os

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in PRUNE_DIRS and name != "proxy"]
        yield dirpath, dirnames, filenames


def run_label_for_trials(run_dir: Path, trials_dir: Path) -> str:
    if trials_dir.parent.name in {"stateful", "stateless"}:
        return run_dir.name
    return trials_dir.parent.name


def discover_trials(
    repo_root: Path,
    *,
    include_before_resume: bool = False,
    include_worktrees: bool = False,
) -> list[TrialSource]:
    bases = [repo_root / "examples" / name / ".cage_runs" for name in CYBER_EXAMPLE_DIRS]
    if include_worktrees:
        bases.append(
            repo_root
            / ".worktrees"
            / "skill-inject"
            / "examples"
            / "skill_inject"
            / ".cage_runs"
        )

    trials: list[TrialSource] = []
    for base in bases:
        if not base.is_dir():
            continue
        for agent_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            agent_label = agent_dir.name
            if not is_selected_claude_code_agent(agent_label):
                continue
            for run_dir in sorted(p for p in agent_dir.iterdir() if p.is_dir()):
                candidates = (
                    run_dir / "trials",
                    run_dir / "stateless" / "trials",
                    run_dir / "stateful" / "trials",
                )
                for trials_dir in candidates:
                    if not trials_dir.is_dir():
                        continue
                    run_label = run_label_for_trials(run_dir, trials_dir)
                    model, family, variant = classify_group(
                        str(base.parent),
                        agent_label,
                        run_label,
                    )
                    state_kind = infer_state_kind(agent_label, trials_dir)
                    for proxy_jsonl in iter_trial_proxies(trials_dir):
                        is_archive = any(".before_resume_" in part for part in proxy_jsonl.parts)
                        if is_archive and not include_before_resume:
                            continue
                        trial_dir = proxy_jsonl.parent.parent
                        trials.append(
                            TrialSource(
                                model=model,
                                task_family=family,
                                run_variant=variant,
                                group=group_key(model, family, variant),
                                agent_label=agent_label,
                                run_label=run_label,
                                state_kind=state_kind,
                                source_base=base.parent,
                                source_run_dir=trials_dir.parent,
                                trials_dir=trials_dir,
                                trial_dir=trial_dir,
                                proxy_jsonl=proxy_jsonl,
                                trial_path=trial_dir.relative_to(trials_dir).as_posix(),
                                is_before_resume=is_archive,
                            )
                        )
    return sorted(trials, key=lambda item: (item.group, item.run_label, item.trial_path))


def count_before_resume_archives(repo_root: Path) -> int:
    return sum(
        1
        for item in discover_trials(repo_root, include_before_resume=True)
        if item.is_before_resume
    )


def compact_meta(meta: Any) -> dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    return {key: meta[key] for key in META_KEYS if key in meta}


def load_scores(trial_dir: Path) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    scores_dir = trial_dir / "scores"
    if scores_dir.is_dir():
        for path in sorted(scores_dir.glob("*.json")):
            scores[path.stem] = load_json_file(path)
    return scores


def load_final_output(trial_dir: Path) -> str:
    output = load_json_file(trial_dir / "task_output.json")
    if isinstance(output, dict):
        for key in ("output", "result", "final_output"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    return ""


def derive_task_info(
    source: TrialSource,
    meta: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    trial_id = str(meta.get("trial_id") or source.trial_path)
    sample_id = str(meta.get("sample_id") or trial_id)
    info: dict[str, Any] = {
        "id": trial_id,
        "sample_id": sample_id,
        "benchmark": source.task_family,
        "model": source.model,
        "agent": source.agent_label,
        "run": source.run_label,
        "run_variant": source.run_variant,
        "stateful": source.state_kind == "stateful",
        "trial_path": source.trial_path,
    }

    if source.task_family == "cvebench":
        match = re.search(r"(CVE-\\d{4}-\\d+)", sample_id)
        if match:
            info["cve"] = match.group(1)
        for mode in ("zero_day", "one_day"):
            if mode in source.trial_path:
                info["mode"] = mode
                break
        pass_match = re.search(r"pass_(\\d+)", source.trial_path)
        if pass_match:
            info["pass"] = int(pass_match.group(1))

    pentest_score = scores.get("agent_pentest_bench")
    if isinstance(pentest_score, dict):
        metadata = pentest_score.get("metadata_summary")
        if isinstance(metadata, dict):
            for key in (
                "challenge",
                "markers_passed",
                "markers_total",
                "successful",
                "total",
                "rooted",
            ):
                if key in metadata:
                    info[key] = metadata[key]
    return info


def step_stats(records: list[dict[str, Any]], proxy_jsonl: Path) -> dict[str, Any]:
    tokens = Counter()
    block_counts = Counter()
    for record in records:
        rec_tokens = record.get("tokens")
        if isinstance(rec_tokens, dict):
            for key, value in rec_tokens.items():
                tokens[str(key)] += int(value or 0)
        for block in record.get("blocks", []):
            if isinstance(block, dict):
                block_counts[str(block.get("type") or "unknown")] += 1
    progress = load_json_file(proxy_jsonl.parent / "progress.json")
    return {
        "steps": len(records),
        "tokens": dict(tokens),
        "progress": progress if isinstance(progress, dict) else {},
        "block_counts": dict(block_counts),
    }


def build_bundle(
    source: TrialSource,
    bundle_dir: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    records = build_step_records(source.proxy_jsonl)
    prompt = ""
    prompt_path = source.trial_dir / "prompt.txt"
    if prompt_path.is_file():
        prompt = prompt_path.read_text(encoding="utf-8", errors="replace")

    raw_meta = load_json_file(source.trial_dir / "meta.json")
    meta = compact_meta(raw_meta)
    scores = load_scores(source.trial_dir)
    final_output = load_final_output(source.trial_dir)
    task = derive_task_info(source, meta, scores)
    stats = step_stats(records, source.proxy_jsonl)

    write_jsonl(records, bundle_dir / "steps.jsonl")
    write_text(bundle_dir / "prompt.txt", prompt)
    write_text(bundle_dir / "final_output.txt", final_output)
    write_json(bundle_dir / "meta.json", meta)
    write_json(bundle_dir / "task.json", task)
    write_json(bundle_dir / "scores.json", scores)

    summary = {
        "schema": "cage_trace_steps_v1",
        "group": source.group,
        "model": source.model,
        "task_family": source.task_family,
        "run_variant": source.run_variant,
        "agent_label": source.agent_label,
        "state_kind": source.state_kind,
        "source_run_dir": str(source.source_run_dir.resolve()),
        "source_trial_dir": str(source.trial_dir.resolve()),
        "source_proxy_jsonl": str(source.proxy_jsonl.resolve()),
        "trial_path": source.trial_path,
        "trial_id": task["id"],
        "sample_id": task["sample_id"],
        "status": meta.get("status", ""),
        "termination_reason": meta.get("termination_reason", ""),
        "score_summary": scores,
        "step_stats": stats,
        "relative_source_trial_dir": source.trial_dir.resolve()
        .relative_to(repo_root.resolve())
        .as_posix(),
        "files": {
            "steps": "steps.jsonl",
            "prompt": "prompt.txt",
            "final_output": "final_output.txt",
            "meta": "meta.json",
            "task": "task.json",
            "scores": "scores.json",
        },
    }
    write_json(bundle_dir / "summary.json", summary)
    return summary


def write_readme(
    output_root: Path,
    group_summaries: list[dict[str, Any]],
    *,
    include_before_resume: bool,
) -> None:
    group_lines = "\n".join(
        f"- `{item['group']}/` - {item['trial_count']} traces"
        for item in sorted(group_summaries, key=lambda value: value["group"])
    )
    archive_note = (
        "This build includes `.before_resume_*` archive attempts."
        if include_before_resume
        else (
            "`.before_resume_*` archive attempts are intentionally skipped because "
            "they duplicate resumed context."
        )
    )
    text = f"""# claude-code-trace-cyber

Lightweight cyber trajectory dataset exported from Cage `.cage_runs`.

Most groups are Claude Code runs. The `qwen3.7-max` group is the exception:
it comes from the Qwen Code agent, and is included because it was part of the
same AgentPentestBench cyber trace collection.

## Source Scope

Repository root used for export:

```text
/data/pxd-team/workspace/fyh/cage
```

The exporter scans only these cyber benchmark run directories:

- `examples/cvebench/.cage_runs/`
- `examples/nyu/.cage_runs/`
- `examples/agent_pentest_bench/.cage_runs/`

`examples/autopenbench/.cage_runs/` is in the scanner allowlist, but no
selected model/run from that directory is present in this export. Worktrees,
`skill_inject`, `strongreject`, target-server logs, and non-cyber runs are not
included.

The source trajectory file for every trial is:

```text
<run>/trials/<trial>/proxy/proxy.jsonl
```

The raw `proxy.jsonl` is not copied because it repeats the full model context
on each request. `steps.jsonl` is derived from the same parser used by the Cage
web inspector (`cage.web.data.parse_trajectory`), which reconstructs model
steps as `thinking`, `response`, `tool_call`, and tool observation blocks.
Older `*.traj` text files are not used as the source of truth.

Only live trial directories are exported. {archive_note}

## Included Run Families

- `qwen36-27b__cvebench__full-pass4`
  - Source agent: `claude_code_baseline:qwen36-27b:stateless`
  - Source run: `run-20260512T111807`
- `qwen36-27b__cvebench__warmup`
  - Source agent: `claude_code_baseline:qwen36-27b:stateless`
  - Source runs: `run-20260512T103625`, `run-20260512T103946`,
    `run-20260512T105505`
- `glm-5.1__cvebench__legacy-baseline`
  - Source agent: `claude_code_baseline:glm-5.1-sii`
  - Source run: `run-20260427T124739`
- `glm-5.1__nyu__legacy-baseline`
  - Source agent: `claude_code_baseline:glm-5.1-sii`
  - Source runs: `run-20260425T223927`, `run-20260427T113654`
- `glm-5.1__agent-pentest-bench__*`
  - Source agents: `claude_code_glm51`, `claudecode_glm51`,
    `claude_code_ctfenv`
  - Source runs: GLM-5.1 AgentPentestBench full, postexp, smoke, mix, and
    small-sweep runs
- `deepseek-v4-pro__agent-pentest-bench__*`
  - Source agents: `claude_code_deepseek_v4_pro*`
  - Source runs: `postexp-deepseek-l0-p1-20260523`,
    `postexp-ds-p3-20260525`, `ds-limit1-scorefix-0524`, and DeepSeek
    smoke/mount checks
- `claude-opus-4-7-cyber__agent-pentest-bench__*`
  - Source agent:
    `claude_code_opus47_cyber:claude-opus-4-7-cyber:stateless`
  - Source runs: `postexp-claude-cyber-20260526`,
    `postexp-cyber-p3-20260526`, and cyber smoke runs
- `qwen3.7-max__agent-pentest-bench__postexp-pass3`
  - Source agent: `qwen_code_qwen37max:qwen3.7-max:stateless`
  - Source run: `postexp-qwen37max-p3-20260526`

The Claude Cyber runs are configured by
the AgentPentestBench post-exploit example config with
`CLAUDE_CODE_EFFORT_LEVEL=xhigh`. The Qwen3.7 Max run used the same benchmark
family with `enable_thinking: true` and `preserve_thinking: true`.

## Layout

Each top-level directory is one selected model/task/run family:

{group_lines}

Inside each group, `traces/<NNNN_trial>/` contains the minimal review bundle:

- `steps.jsonl`: primary trajectory, one model step per line. Blocks are
  `thinking`, `response`, and `tool_call` with inline `observation`.
- `prompt.txt`: initial task prompt given to Claude Code.
- `final_output.txt`: final parsed agent output.
- `meta.json`: compact trial status and termination metadata.
- `task.json`: compact task identity and benchmark/run labels.
- `scores.json`: compact scorer result.
- `summary.json`: per-trial index and counts.

Raw `proxy.jsonl`, `.traj`, state snapshots, target logs, and full runtime
artifacts are omitted to keep this dataset small and avoid repeated context
dumps.

## Schema

`steps.jsonl` records look like:

```json
{{"type":"step","step":0,"tokens":{{"in":1,"out":2,"reasoning":0}},"blocks":[{{"type":"thinking","content":"..."}},{{"type":"response","content":"..."}},{{"type":"tool_call","name":"Bash","input":{{"command":"id"}},"observation":"uid=1000(agent)"}}]}}
```

This is derived from the same parser used by the Cage web inspector trajectory
page, not by parsing the older `.traj` text files.
"""
    write_text(output_root / "README.md", text)


def build_dataset(
    repo_root: Path,
    output_root: Path,
    *,
    force: bool = False,
    include_before_resume: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists; pass --force to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sources = discover_trials(repo_root, include_before_resume=include_before_resume)
    archive_count = 0 if include_before_resume else count_before_resume_archives(repo_root)
    group_counts: defaultdict[str, int] = defaultdict(int)
    group_manifests: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    group_stats: defaultdict[str, Counter] = defaultdict(Counter)
    group_block_counts: defaultdict[str, Counter] = defaultdict(Counter)
    manifest: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        group_counts[source.group] += 1
        group_index = group_counts[source.group]
        trace_name = f"{group_index:04d}_{slugify(source.trial_path)}"
        bundle_rel = Path(source.group) / "traces" / trace_name
        bundle_dir = output_root / bundle_rel
        try:
            summary = build_bundle(source, bundle_dir, repo_root=repo_root)
        except Exception as exc:  # Keep long batch exports moving.
            errors.append(
                {
                    "source_trial_dir": str(source.trial_dir.resolve()),
                    "group": source.group,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        entry = {
            "group": source.group,
            "model": source.model,
            "task_family": source.task_family,
            "run_variant": source.run_variant,
            "agent_label": source.agent_label,
            "state_kind": source.state_kind,
            "run_label": source.run_label,
            "trial_path": source.trial_path,
            "bundle_dir": bundle_rel.as_posix(),
            "steps": summary["step_stats"]["steps"],
            "status": summary.get("status", ""),
            "termination_reason": summary.get("termination_reason", ""),
            "sample_id": summary.get("sample_id", ""),
            "score_summary": summary.get("score_summary", {}),
        }
        manifest.append(entry)
        group_manifests[source.group].append(entry)
        group_stats[source.group]["trials"] += 1
        group_stats[source.group]["steps"] += int(summary["step_stats"]["steps"])
        group_stats[source.group]["prompt_chars"] += (bundle_dir / "prompt.txt").stat().st_size
        group_stats[source.group]["final_output_chars"] += (
            bundle_dir / "final_output.txt"
        ).stat().st_size
        group_block_counts[source.group].update(summary["step_stats"]["block_counts"])

        if index % 25 == 0 or index == len(sources):
            print(f"exported {index}/{len(sources)} traces", flush=True)

    group_summaries: list[dict[str, Any]] = []
    for group in sorted(group_manifests):
        first = group_manifests[group][0]
        group_summary = {
            "group": group,
            "model": first["model"],
            "task_family": first["task_family"],
            "run_variant": first["run_variant"],
            "trial_count": len(group_manifests[group]),
            "stats": dict(group_stats[group]),
            "block_counts": dict(group_block_counts[group]),
        }
        group_summaries.append(group_summary)
        write_json(output_root / group / "MANIFEST.json", group_manifests[group])
        write_json(output_root / group / "GROUP_SUMMARY.json", group_summary)

    root_summary = {
        "schema": "cage_trace_dataset_v1",
        "description": "Lightweight Claude Code trajectory exports from Cage proxy captures.",
        "selection": {
            "models": [
                "qwen36-27b",
                "glm-5.1",
                "deepseek-v4-pro",
                "claude-opus-4-7-cyber",
                "qwen3.7-max",
            ],
            "source_scope": (
                "examples/*/.cage_runs only; worktrees and non-cyber "
                "skill-inject runs excluded"
            ),
            "include_before_resume": include_before_resume,
            "before_resume_archives_skipped": archive_count,
        },
        "trial_count": len(manifest),
        "source_trial_count": len(sources),
        "groups": group_summaries,
        "errors": errors,
    }
    write_json(output_root / "MANIFEST.json", manifest)
    write_json(output_root / "SUMMARY.json", root_summary)
    write_readme(output_root, group_summaries, include_before_resume=include_before_resume)
    return root_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build compact Claude Code cyber trace dataset.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    parser.add_argument(
        "--include-before-resume",
        action="store_true",
        help="Include .before_resume_* archive attempts instead of skipping them",
    )
    args = parser.parse_args(argv)
    summary = build_dataset(
        args.repo_root,
        args.output_root,
        force=args.force,
        include_before_resume=args.include_before_resume,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "trial_count": summary["trial_count"],
                "groups": len(summary["groups"]),
                "errors": len(summary["errors"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
