#!/usr/bin/env python3
"""Pick the 500 most distribution-balanced external-ARVO environments for
training-data collection.

Balance objective (difficulty intentionally ignored — env distribution only):
  primary  : flatten the `input_class` (modality) marginal as much as the data allows
  secondary: spread `vuln_class` inside each modality
  diversity: cap envs per PROJECT, and inside a project prefer DISTINCT harnesses
             (same project+harness = near-duplicate code path -> low training value)

Method: water-fill class quotas (raise a uniform level until quotas sum to 500,
each class clamped to what its projects can supply under the per-project cap),
then fill each class by round-robin over projects (distinct-harness-first).
Classes that hit their ceiling before the level = the SCARCE data.
"""
import json, collections, hashlib, os

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 500
PROJECT_CAP = 6          # max envs from one project (breaks imagemagick/harfbuzz)

env = [json.loads(l) for l in open(f"{HERE}/environments.jsonl")]

def h(s): return int(hashlib.md5(s.encode()).hexdigest(), 16)

# index: input_class -> project -> [envs], harness-diverse + deterministic order
byclass = collections.defaultdict(lambda: collections.defaultdict(list))
for e in env:
    byclass[e["input_class"]][(e["project"] or "?").lower()].append(e)
for c in byclass:
    for p in byclass[c]:
        # one env per (harness) first (distinct code paths), then the rest;
        # deterministic by id hash within each tier
        seen, prim, extra = set(), [], []
        for e in sorted(byclass[c][p], key=lambda x: h(x["task_id"])):
            hn = (e["harness"] or "")
            (prim if hn not in seen else extra).append(e)
            seen.add(hn)
        byclass[c][p] = prim + extra

# capacity of each class under the project cap
cap = {c: sum(min(len(v), PROJECT_CAP) for v in projs.values())
       for c, projs in byclass.items()}

# water-fill: smallest level L with sum_c min(L, cap[c]) >= TARGET
def total_at(L):
    return sum(min(L, cap[c]) for c in cap)
L = 0
while total_at(L) < TARGET:
    L += 1
quota = {c: min(L, cap[c]) for c in cap}
# trim overshoot from the classes still at the (non-clamped) level L
over = sum(quota.values()) - TARGET
for c in sorted([c for c in quota if quota[c] == L], key=lambda c: -h(c)):
    if over <= 0: break
    take = min(over, L)  # never below the level's floor sharing; remove 1 at a time
    quota[c] -= 1; over -= 1

# select each class's quota by round-robin over projects (distinct-harness-first).
# PROJECT_CAP is GLOBAL (a project can't dominate across modalities either);
# fill scarce classes first so they claim their few projects before rich classes.
chosen = []
taken_global = collections.Counter()
for c in sorted(quota, key=lambda c: cap[c]):       # least-capacity class first
    q = quota[c]
    projs = {p: list(v) for p, v in byclass[c].items()}
    order = sorted(projs, key=lambda p: (-min(len(projs[p]), PROJECT_CAP), h(p)))
    picked = 0
    while picked < q:
        progressed = False
        for p in order:
            if picked >= q: break
            if taken_global[p] >= PROJECT_CAP or not projs[p]:
                continue
            chosen.append(projs[p].pop(0)); taken_global[p] += 1; picked += 1
            progressed = True
        if not progressed:
            break  # class exhausted (scarcity)

# top-up: global cap may starve rich classes below quota -> refill toward TARGET,
# class-round-robin over leftover envs (still honouring the global project cap),
# richest classes first so the head absorbs the shortfall the tail can't supply.
chosen_ids = {e["task_id"] for e in chosen}
leftover = collections.defaultdict(list)
for e in env:
    if e["task_id"] not in chosen_ids and taken_global[(e["project"] or "?").lower()] < PROJECT_CAP:
        leftover[e["input_class"]].append(e)
for c in leftover:
    leftover[c].sort(key=lambda e: h(e["task_id"]))
rich = sorted(leftover, key=lambda c: -cap[c])
while len(chosen) < TARGET:
    progressed = False
    for c in rich:
        if len(chosen) >= TARGET: break
        while leftover[c]:
            e = leftover[c].pop(0)
            p = (e["project"] or "?").lower()
            if taken_global[p] >= PROJECT_CAP: continue
            chosen.append(e); chosen_ids.add(e["task_id"]); taken_global[p] += 1
            progressed = True; break
    if not progressed:
        break

chosen.sort(key=lambda e: int(e["id_num"]))

# ---- write selection + report ----------------------------------------------
with open(f"{HERE}/selected_500.txt", "w") as fh:
    fh.write("\n".join(e["task_id"] for e in chosen) + "\n")
json.dump(chosen, open(f"{HERE}/selected_500.json", "w"), indent=0)

ci = collections.Counter(e["input_class"] for e in chosen)
cv = collections.Counter(e["vuln_class"] for e in chosen)
cp = collections.Counter((e["project"] or "?").lower() for e in chosen)
allinp = collections.Counter(e["input_class"] for e in env)
# truly scarce = capacity-bound below the uniform level (NOT just rounding trim)
scarce = {c: (cap[c], allinp[c], len(byclass[c])) for c in cap if cap[c] < L}

lines = []
def out(s=""):
    print(s); lines.append(s)
out(f"# Balanced-500 selection (env distribution only; difficulty ignored)\n")
out(f"selected {len(chosen)} / target {TARGET}   water-level L={L}   global project_cap={PROJECT_CAP}")
out(f"unique projects: {len(cp)}/{len(chosen)}   max per project: {cp.most_common(1)[0]}\n")
out("## input_class — full set % -> picked %  (flattened from 30.8% to ~8%)\n")
out("| input_class | picked | picked% | fullset% |")
out("|---|---|---|---|")
for k, _ in allinp.most_common():
    out(f"| {k} | {ci[k]} | {100*ci[k]/len(chosen):.1f}% | {100*allinp[k]/len(env):.1f}% |")
out("\n## vuln_class — secondary axis (still heap-heavy; only input_class was balanced)\n")
out("| vuln_class | picked | picked% |")
out("|---|---|---|")
for k, v in cv.most_common():
    out(f"| {k} | {v} | {100*v/len(chosen):.1f}% |")
out(f"\n## SCARCEST data — classes that CANNOT reach the uniform level {L}\n")
out("Scarcity is driven by **distinct projects/harnesses**, not raw env count: no")
out("amount of sampling evens these out — you must COLLECT MORE distinct programs.\n")
out("| input_class | ceiling@cap | #envs | distinct_projects | verdict |")
out("|---|---|---|---|---|")
for c, (cp2, n, ndp) in sorted(scarce.items(), key=lambda x: x[1][2]):
    verdict = "critical" if ndp <= 6 else "thin"
    out(f"| {c} | {cp2} | {n} | {ndp} | {verdict} |")
open(f"{HERE}/SELECTION_500.md", "w").write("\n".join(lines) + "\n")
