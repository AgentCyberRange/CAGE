# Datasets

CAGE is **infrastructure** — it ships no benchmark data. Every benchmark keeps
its targets, prompts, and verifiers in its own dataset, vendored into
`examples/<benchmark>/datasets/` as a git submodule. Datasets are acquired in up
to two steps:

1. **Git submodule** — pulls the dataset repository (tooling + a runnable
   subset).
2. **Hugging Face** *(only where a benchmark publishes a larger set)* — pulls the
   complete dataset on top of the submodule checkout.

You only need the dataset(s) for the benchmark you intend to run.

## Step 1 — Initialize the submodule

```bash
git submodule update --init examples/<benchmark>/datasets/<name>
```

| Benchmark | Submodule path |
|---|---|
| WebExploitBench | `examples/agent_pentest_bench/datasets/web_exploit_bench` |
| PostExploitBench | `examples/agent_pentest_bench/datasets/post_exploit_bench` |
| CVEBench | `examples/cvebench/datasets` |
| NYU CTF | `examples/nyuctfbench/datasets` |
| AutoPenBench | `examples/autopenbench/datasets` |

After this step the submodule's `scripts/` tooling and its bundled sample
subset are available, which is enough for a smoke run.

## Step 2 — Fetch the full dataset from Hugging Face

Some benchmarks publish only a small, immediately runnable **subset** on GitHub
(targets you can build and run right after `git submodule update --init`) and
host the **complete** dataset on Hugging Face. This is the model used by
**AgentPentestBench**:

| Family | GitHub subset | Hugging Face full dataset |
|---|---|---|
| WebExploitBench | `comfyui`, `dataease`, `prestashop` | [AgentCyberRange/WebExploitBench](https://huggingface.co/datasets/AgentCyberRange/WebExploitBench) — 15 apps / 110 vulns |
| PostExploitBench | `range-4`, `range-6` | [AgentCyberRange/PostExploitBench](https://huggingface.co/datasets/AgentCyberRange/PostExploitBench) — 8 ranges / 156 hosts |

Each dataset repository carries a `scripts/fetch` helper that downloads the
remaining targets in place:

```bash
# from inside the submodule checkout
cd examples/agent_pentest_bench/datasets/web_exploit_bench
scripts/fetch
```

`scripts/fetch` wraps the Hugging Face CLI and only adds **data** — it preserves
the repository's own `README`, `LICENSE`, and `scripts/`. It is resumable and
safe to re-run.

Prerequisites:

```bash
pip install -U huggingface_hub   # provides the `hf` CLI
hf auth login                    # only if the dataset is gated/private
```

Once the full dataset is present, every sample listed in the benchmark index
becomes runnable — e.g. `cage run web_exploit_bench --sample pb-siyucms ...`.
Until then, only the bundled subset (`pb-comfyui`, `pb-dataease`,
`pb-prestashop`, `pb-postexp-range-4`, `pb-postexp-range-6`) is available.

## Notes

- Dataset repositories use **Git LFS** for heavy binaries (image dumps, SQL
  dumps, prebuilt exploit binaries). Install Git LFS before cloning if you use a
  full `git clone` instead of `scripts/fetch`.
- Each dataset repository documents its own per-target build and local-run
  tooling (`scripts/targetctl` for web targets, `scripts/rangectl` for ranges).
  See the benchmark's own README, linked from the **Benchmarks** menu.
