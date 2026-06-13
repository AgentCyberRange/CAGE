# Target Build And Launch Debugging

CAGE no longer exposes the old low-level target smoke-test command as part of
the public CLI. Target launch is benchmark-owned Layer 2 behavior, so the
public workflow now goes through registered benchmark commands.

## Public Workflow

First validate config and prompt rendering:

```bash
cage benchmark check web_exploit_bench \
  --sample pb-comfyui \
  --prompt-level l0 \
  --show-prompt
```

Then run the benchmark-owned build hook without agents or model calls:

```bash
cage benchmark build web_exploit_bench --sample pb-comfyui --dry-run
cage benchmark build web_exploit_bench --sample pb-comfyui
```

Finally run one constrained smoke trial and inspect it in the web UI:

```bash
cage run web_exploit_bench \
  --agent codex \
  --model gpt-5.5 \
  --sample pb-comfyui \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id target-smoke-001

cage inspect examples/agent_pentest_bench
```

## Where Target Launch Happens

Target lifecycle still uses the same internal pieces:

- `cage.target.build` selects samples and calls `Benchmark.build_targets()`.
- `cage.target.check` contains the lower-level target readiness mechanics used
  by internal tests and future public target workflows.
- `cage.target.serve` is the internal Python module entrypoint for the
  embedded FastAPI target server. It is not a user-facing Click command.
- `cage.experiment.engine.conductor` starts the embedded target server during real runs and
  writes the server log under `.cage_runs/`.

## Debugging Failed Launches

For a failed smoke trial, start in the inspector run page, then check:

- the run log and debug log shown by the run page;
- the target server log under `.cage_runs/target_server-<run_id>.log`;
- the trial runtime directory for `runtime/check_done_output.txt`;
- Docker compose logs for the specific target project if startup failed before
  CAGE could collect artifacts.

When target images are missing or stale, rerun the benchmark build hook for the
specific sample before retrying the smoke trial:

```bash
cage benchmark build web_exploit_bench --sample pb-comfyui
```
