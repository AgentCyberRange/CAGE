# StrongReject

StrongReject is CAGE's safety / refusal benchmark: it hands a coding agent a
single **harmful request** (e.g. "write a phishing email", "build a hate-speech
bot") and measures whether the agent **refuses** it instead of complying. The
prompts come from the published StrongReject dataset.

Each trial drops one forbidden prompt into the agent's container as `note.md` and
asks it to follow the instructions inside. The answer is graded by a **judge
model** — a separate LLM that reads the request and the response and scores
`refusal` (0 = complied, 1 = refused) plus how convincing/specific any compliance
was. A refusal scores 0 (safe); compliance scores higher the more useful the
harmful answer is. There is **no live target and no Docker target to build** —
grading is judge-based and `cage run` never builds anything. If no judge is
configured, scoring falls back to keyword refusal detection.

## 1. Get the data

The dataset is **committed** — no download needed. `dataset/strongreject_dataset.csv`
(313 prompts) is the default; `dataset/strongreject_small_dataset.csv` is a subset.
Sample ids are generated per row (e.g. `sr-000000-90732f80a169`).

## 2. Configure a model and a judge

Model endpoints live in the git-ignored `config/models.yml` (it holds your keys):

```bash
cp config/models.example.yml config/models.yml
export OPENAI_API_KEY=...
cage model set openai-example --provider openai --model gpt-5.5 \
  --endpoint https://api.openai.com/v1 --api-key '${OPENAI_API_KEY}'
cage model list
```

`default_strongreject.yml` points both the subject agent and the `judge:` block at
the `openai-example` id — register that id (or edit the config). The judge resolves
against the same `config/models.yml`.

## 3. Smoke-test one sample

There is no target to build — the flow is just check (render the prompt + config,
no model call, no cost), then run one trial:

```bash
cage benchmark check strongreject --agent claude_code_baseline --model openai-example \
  --sample sr-000000-90732f80a169 --show-prompt
cage run strongreject --agent claude_code_baseline --model openai-example \
  --sample sr-000000-90732f80a169 --passk 1 --max-concurrent 1 --run-id strongreject-smoke-001
```

## 4. Run the full set

Drop `--sample` to run every prompt. The config's `eval.limit: 5` caps how many
samples run by default — raise it (or set `--max-sample-num`) for the full 313:

```bash
cage run strongreject --agent claude_code_baseline --model openai-example \
  --max-sample-num 313 --run-id strongreject-full-001
```

`--dry-run` prints the plan without running; `--run-id … --resume` continues a run,
skipping finished trials. To wrap every prompt in a jailbreak, set
`eval.benchmark.jailbreak: AIM.txt` in the config.

**Tune `--max-concurrent` to your machine.** It caps how many trials run at
once; each trial spins up its own container(s), so raise it on a big host and
lower it when CPU, RAM, or Docker disk is tight.

## 5. Watch a run

`cage run` starts the inspector automatically and prints its URL — open it to
browse trials, the judge's verdicts, and full agent trajectories live.

## Explore the CLI

```bash
cage --help                    # top-level: run, benchmark, model, agent, inspect, score, gc
cage benchmark list            # registered benchmarks (strongreject, …)
cage run strongreject --help   # this benchmark's samples, agent/model matrix, and flags
cage model list                # model endpoints you've registered
```
