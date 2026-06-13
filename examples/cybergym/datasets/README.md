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
