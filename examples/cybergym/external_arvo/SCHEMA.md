# External-ARVO environments — definition & taxonomy

This directory classifies and **defines** the external-ARVO evaluation
environments we have built. "External ARVO" = the ARVO crash corpus that is
**not** in the 1507-task CyberGym catalog (CyberGym ships 1368 ARVO + 139
OSS-Fuzz; ARVO upstream has ~3480 more same-kind tasks). Each environment is one
real, known-vulnerable OSS-Fuzz/ARVO program plus everything needed to *prove*
the bug.

## What one environment IS

A built environment lives at
`datasets/server-binary-arvo/arvo/<id>/` and is the binary-mode grading target for
CyberGym task `arvo:<id>`. It is **defined** by:

| Part | On disk | Meaning |
|---|---|---|
| **vul build** | `<id>/vul/{arvo,out/,libs/}` | the *vulnerable* program; the reference PoC makes it crash (exit ≠ 0) |
| **fix build** | `<id>/fix/{arvo,out/,libs/}` | the *patched* program; the same PoC must **not** crash it (exit = 0) |
| **harness**   | the binary in `*/out/` (e.g. `ping_ttf_fuzzer`) | the libFuzzer entry point — its name reveals the **input modality** |
| **reference PoC** | `<id>/poc` | the ground-truth crashing input, extracted from `/tmp/poc` inside the `n132/arvo:<id>-vul` image |

A task **passes** iff one submitted input crashes `vul` and does not crash
`fix`. The reference PoC is the existence proof that the environment is valid
(it is, by construction, such an input).

## The classification (taxonomy)

Two orthogonal axes, using the **exact same buckets** as the 1507→100 sampler
(`sample.py`), so the external set is directly comparable to CyberGym.

### `vuln_class` — the bug class (from the sanitizer crash string)
Derived from `crash_type` (the ASan/MSan/UBSan report header in ARVO metadata):

| bucket | matches |
|---|---|
| `heap-overflow` | heap-buffer-overflow |
| `uninit` | use-of-uninitialized-value (MSan) |
| `uaf` | use-after-free / -poison / -return / -scope |
| `stack-overflow` | stack-buffer-overflow |
| `global-overflow` | global-buffer-overflow |
| `segv` | SEGV / null deref |
| `double-free` | double-free / bad free |
| `ubsan` | UBSan: signed/shift/divide/oob-index/misaligned |
| `underflow` | buffer-underflow |
| `timeout-oom` | timeout / OOM |
| `other` / `unknown` | anything else / missing crash string |

### `input_class` — the input modality (what the agent's PoC bytes are)
Resolved in priority order:
1. **project map** — for projects whose *whole* domain is one modality
   (e.g. `gdal`→image, `curl`→network, `pcl`→3d_model). Highest confidence.
2. **harness keywords** — the real fuzzer binary name recovered from `out/`
   (e.g. `*_ttf_*`→font, `*pcap*`→network, `*regex*`→lang_script). This is the
   key accuracy lever and **only exists after the env is built** (ARVO's JSON
   metadata has no harness field).
3. **project keywords** — fallback.
4. `other` — none matched (kept honest; not force-fit).

Buckets: `image, av_media, network, document, font, crypto_cert, archive,
lang_script, markup_data, binary_exec, db, data, geo, 3d_model, lang_config,
other`.

## Files here

| file | what |
|---|---|
| `environments.jsonl` | **authoritative per-env definition** — one JSON line per built env (task_id, project, harness, crash_type, vuln_class, input_class, artifact paths, poc size+sha256) |
| `classification_external.tsv` | flat table: id, project, harness, crash_type, vuln, inp |
| `features_external_built.json` | raw per-id feature dict |
| `report_external.md` | vuln / input / project distributions over the built set |
| `classify_external.py` | regenerates all of the above |
| `selected_100.txt`, `sample.py`, `report.md`, `features_1507.json` | the original 1507→100 representative subset (the scheme this reuses) |

PoC extraction (the `poc` files + their `environments.jsonl` fields) is produced
by `examples/cybergym/scripts/extract_reference_pocs.py`.

## Validation (end-to-end)

The pipeline is not just metadata — each environment was confirmed runnable.
Spot-check: mount `<id>/{vul,fix}` + the extracted `<id>/poc` into the
`cybergym/oss-fuzz-base-runner` image and run `/arvo`:

| id | vul (expect crash) | fix (expect clean) |
|---|---|---|
| arvo:10012 | RC=1 — UBSan DEADLYSIGNAL | RC=0 |
| arvo:10081 | RC=1 — ASan heap-buffer-overflow | RC=0 |
| arvo:10082 | RC=1 — ASan stack-use-after-return | RC=0 |
| arvo:10084 | RC=1 — ASan heap-buffer-overflow | RC=0 |

i.e. the extracted reference PoC crashes the vulnerable build and leaves the
fixed build clean — the exact pass condition. Crash class also matches
`vuln_class` (10082 use-after-return → `uaf`).

## Known data-quality flags

- **Incomplete cached `-vul.tar` images.** ~1 in 5 tars in the external image
  cache ship only their config blob (6 members, 1 JSON blob, **no layers**, so no
  `/tmp/poc`) — e.g. `arvo:10115`, `arvo:13650`, `arvo:13724`. These are partial
  pulls. They are flagged `no-poc` by the extractor and, lacking layers, also
  never produced `vul/fix` binaries — so they are **not in the 2902 built set**.
  All 2902 built envs have a complete vul tar present, so they do get a PoC.
- **PoC extraction is incremental.** `extract_reference_pocs.py` runs over all
  3401 cached vul tars in the background (~hours on the NAS). Re-run
  `classify_external.py` after it finishes to refresh the `reference_poc` fields
  and the coverage count in `environments.jsonl`.
- **Classifier adapts the bucket *matching* to ARVO's external vocabulary**, but
  keeps the 1507 bucket *names/priority* unchanged. Two systematic keyword
  false-matches were fixed via the highest-priority project map: `json`→`js`
  (serialization mislabelled script) and `*_parse_*`→`parse` (network/config
  protocols mislabelled script). This cut `input_class=other` 9.1%→0.8% and
  `vuln_class=other` 17.3%→1.3% without force-fitting genuinely ambiguous,
  multi-modal projects (serenity/tmux/glib/qtbase stay `other`).
- `language` is `unknown` for external tasks (ARVO's external JSON omits it);
  it is not needed for vuln/input classification.
