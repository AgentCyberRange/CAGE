#!/usr/bin/env python3
"""Pick the next 1000 highest-priority external-ARVO environments to run, on top
of the already-run balanced 500 (selected_500.txt).

Priority (env distribution only — difficulty deliberately ignored, per request):
  1. NEW (project, harness) coverage — each distinct fuzz target the 500 did NOT
     already cover is a new code/format path => the single highest-value signal
     for training data. (Most thin modality classes are already exhausted by the
     500, so "more balance" is no longer the lever; "more distinct targets" is.)
  2. NEW project, then net-new-vs-CyberGym-1507 (novel beyond the published set).
  3. Keep the cumulative 1500 modality mix as even as the remaining pool allows
     (water-fill quota per input_class), and spread across projects via a global
     cumulative per-project cap so no repo floods the set.

Outputs selected_1000.txt (+ .json) and SELECTION_1000.md.
"""
import json, collections, hashlib, os

HERE = os.path.dirname(os.path.abspath(__file__))
N_NEW = 1000
PROJECT_CAP = 15          # cumulative cap per project across the 500 + this 1000
TOTAL = 1500              # cumulative target the water-fill balances toward

env = {json.loads(l)["id_num"]: json.loads(l) for l in open(f"{HERE}/environments.jsonl")}
done = [l.strip().split(":")[1] for l in open(f"{HERE}/selected_500.txt") if l.strip()]
done_set = set(done)

# projects in CyberGym-1507 (for the net-new bonus). Optional: if the features
# file isn't present, skip the bonus instead of failing.
_f1507 = os.environ.get(
    "FEATURES_1507",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                 "artifacts", "cybergym_subset100", "features_1507.json"),
)
try:
    f1507 = json.load(open(_f1507))
    proj1507 = set((v.get("project") or "").lower() for v in f1507.values())
except FileNotFoundError:
    print(f"[warn] {_f1507} not found; skipping net-new-vs-CyberGym-1507 bonus", flush=True)
    proj1507 = set()

def h(s): return int(hashlib.md5(s.encode()).hexdigest(), 16)
def proj(e): return (e["project"] or "?").lower()
def harn(e): return (proj(e), (e["harness"] or "").lower())

# ---- state seeded by the already-run 500 -----------------------------------
covered_harn = set(); covered_proj = set()
proj_count = collections.Counter()           # cumulative per-project (incl. the 500)
class_count = collections.Counter()          # cumulative per input_class (incl. 500)
for i in done:
    if i in env:
        e = env[i]; covered_harn.add(harn(e)); covered_proj.add(proj(e))
        proj_count[proj(e)] += 1; class_count[e["input_class"]] += 1
seed_harn = set(covered_harn); seed_proj = set(covered_proj)   # snapshot of the 500

cands = [env[i] for i in env if i not in done_set]
byclass = collections.defaultdict(list)
for e in cands:
    byclass[e["input_class"]].append(e)

# ---- cumulative water-fill target per input_class (balance what we can) -----
cls_supply = {c: class_count[c] + len(byclass[c]) for c in set(class_count) | set(byclass)}
def total_at(L): return sum(min(L, cls_supply[c]) for c in cls_supply)
L = 0
while total_at(L) < TOTAL:
    L += 1
cum_target = {c: min(L, cls_supply[c]) for c in cls_supply}
# how many NEW picks each class gets = cumulative target minus what the 500 gave,
# clamped to availability; the leftover (from exhausted classes) is redistributed
# to classes that still have supply.
new_quota = {c: max(0, min(cum_target[c] - class_count[c], len(byclass[c]))) for c in cls_supply}
short = N_NEW - sum(new_quota.values())
# redistribute shortfall to classes with remaining headroom, richest first
while short > 0:
    headroom = [(len(byclass[c]) - new_quota[c], c) for c in cls_supply if len(byclass[c]) - new_quota[c] > 0]
    if not headroom: break
    headroom.sort(reverse=True)
    for _, c in headroom:
        if short <= 0: break
        new_quota[c] += 1; short -= 1

# ---- rank within each class by priority, fill the quota ---------------------
def rank_key(e):
    return (
        0 if harn(e) not in covered_harn else 1,   # new fuzz target first
        0 if proj(e) not in covered_proj else 1,    # new project next
        0 if proj(e) not in proj1507 else 1,        # novel vs CyberGym-1507
        proj_count[proj(e)],                        # spread: fewer-picked project first
        h(e["task_id"]),                            # deterministic tiebreak
    )

chosen = []
for c in byclass:
    pool = sorted(byclass[c], key=rank_key)
    picked = 0
    # re-rank lazily as we pick (covered sets + counts change) — re-sort each take
    while picked < new_quota[c] and pool:
        pool.sort(key=rank_key)
        e = None
        for cand in pool:
            if proj_count[proj(cand)] < PROJECT_CAP:
                e = cand; break
        if e is None: break                          # all remaining hit the cap
        pool.remove(e); chosen.append(e); picked += 1
        covered_harn.add(harn(e)); covered_proj.add(proj(e))
        proj_count[proj(e)] += 1

# top-up if some class was cap/supply-bound, to hit exactly N_NEW
if len(chosen) < N_NEW:
    rest = sorted((e for c in byclass for e in byclass[c] if e not in chosen),
                  key=rank_key)
    for e in rest:
        if len(chosen) >= N_NEW: break
        if proj_count[proj(e)] >= PROJECT_CAP: continue
        chosen.append(e); covered_harn.add(harn(e)); covered_proj.add(proj(e)); proj_count[proj(e)] += 1

chosen.sort(key=lambda e: int(e["id_num"]))
chosen_ids = {e["id_num"] for e in chosen}

# ---- write -----------------------------------------------------------------
with open(f"{HERE}/selected_1000.txt", "w") as fh:
    fh.write("\n".join(e["task_id"] for e in chosen) + "\n")
json.dump(chosen, open(f"{HERE}/selected_1000.json", "w"), indent=0)

# ---- report ----------------------------------------------------------------
new_h = len({harn(e) for e in chosen} - seed_harn)
new_p = len({proj(e) for e in chosen} - seed_proj)
nn = sum(1 for e in chosen if proj(e) not in proj1507)
ci = collections.Counter(e["input_class"] for e in chosen)
cum = collections.Counter()                                  # cumulative 1500 modality
for i in done:
    if i in env: cum[env[i]["input_class"]] += 1
for e in chosen: cum[e["input_class"]] += 1
cp = collections.Counter(proj(e) for e in chosen)

lines = []
def out(s=""):
    print(s); lines.append(s)
out(f"# Priority-1000 selection (next batch after the run 500)\n")
out(f"picked {len(chosen)} new envs | project_cap(cumulative)={PROJECT_CAP}")
out(f"distinct NEW fuzz targets (project,harness) not in the 500: {new_h} / {len(chosen)}")
out(f"distinct projects in the 1000: {len(cp)} | brand-new projects (not in 500): {new_p} | net-new vs CyberGym-1507: {nn}")
out(f"max envs from one project (this 1000): {cp.most_common(1)[0]}\n")
out("## modality — this 1000 vs cumulative 1500\n")
out("| input_class | this_1000 | cum_1500 | cum% |")
out("|---|---|---|---|")
for k, _ in cum.most_common():
    out(f"| {k} | {ci.get(k,0)} | {cum[k]} | {100*cum[k]/sum(cum.values()):.1f}% |")
open(f"{HERE}/SELECTION_1000.md", "w").write("\n".join(lines) + "\n")
