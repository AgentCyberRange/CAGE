# CyberGym datasets

Committed here (small, the source of truth for *which* tasks exist):

- `cybergym.json` — the official CyberGym catalog (1507 tasks).
- `arvo.json` — the arvo-external catalog (3480 tasks).
- `cybergym_images.txt` — the trace-100 subset (image-mode pinning / task filter).

The large, machine-specific data resolves through **gitignored symlinks** so the
committed YAML can stay generic (`./datasets/<name>`). Set them up once per
machine (paths are examples — point at your own copies):

```bash
cd examples/cybergym/datasets
# Repo tarballs staged into the agent container (data_dir). Must contain arvo/.
ln -sfn /data/.../cybergym/cybergym_data/data            payloads
# Prebuilt vul/fix binaries for binary-mode grading (binary_dir). The default.
#   python scripts/server_data/download_binary_only_runners.py
#   wget .../cybergym-server-binary/.../cybergym-server-data.7z && 7z x ...
ln -sfn /home/.../cybergym-bin/cybergym-server-data      server-binary
# Tar cache of per-task n132/arvo images — IMAGE MODE ONLY (image_cache_dir).
ln -sfn /data/.../cybergym/.../image-cache               image-cache
```

`payloads`, `server-binary`, and `image-cache` are listed in the repo
`.gitignore` and must never be committed. `default_cybergym.yml` grades in
**binary mode** by default (`binary_dir: ./datasets/server-binary`); see its
comments to switch to image mode.

## ARVO-external (`default_arvo.yml`)

`default_arvo.yml` runs the 3480-task arvo-external catalog. In binary mode it
resolves three **locally generated** paths (all gitignored, none ship as data):

- `server-binary-arvo/` (`binary_dir`) — prebuilt vul/fix binaries, one dir per
  task at `server-binary-arvo/arvo/<id>/{vul,fix}/` (+ each task's reference `poc`).
- `payloads-arvo/` (`data_dir`) — repo tarballs staged into the agent container.
- `arvo_binary_ready.txt` (`task_ids`) — the subset whose binaries are present
  (derived from `server-binary-arvo/arvo/*/`), so a run only picks gradable tasks.

Regenerate them per machine with the tooling in `../scripts/`:

```bash
# vul/fix binaries from local n132/arvo image tars -> server-binary-arvo/
python scripts/extract_server_binaries.py --image-cache datasets/image-cache \
    --out datasets/server-binary-arvo
# (optional) pull missing -fix images through a proxy, then extract them
OUT=datasets/server-binary-arvo bash scripts/download_fix_binaries.sh
# each task's ground-truth reference PoC, from its -vul image -> <id>/poc
python scripts/extract_reference_pocs.py
```

`external_arvo/` holds the classification/selection tooling (`classify_external.py`,
`select_*.py`) and its schema (`SCHEMA.md`); the generated tables and id lists
stay local.
