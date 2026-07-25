#!/usr/bin/env python3
"""Track D Phase 1 — build an IDENTITY-DISJOINT Celeb-DF dev/test split.
celebdf_dev  (~50%) : all cross-dataset iteration/decisions use THIS only.
celebdf_test (~50%) : SEALED — evaluated exactly once at the very end.

Celeb-DF-v2 identities: Celeb-real `id{X}_*`, Celeb-synthesis `id{X}_id{Y}_*` (two ids),
YouTube-real `NNNNN` (no id -> its own singleton identity). Fakes pair two identities, so a naive
per-identity coin flip drops ~42% of fakes (spanning). The 59 identities form 4 disconnected
components (38/10/10/1); we assign whole small components and split the dominant component with a
balanced spectral (Fiedler) sweep cut, minimizing dropped fakes while keeping ~50/50 by video count.
Spanning fakes (two ids on opposite sides) are DROPPED (cannot be placed without leakage).
Deterministic (seed 42). Output: splits/celebdf_dev_test.json + loud identity-disjoint assertion.
"""
import os, sys, json, re, hashlib, subprocess, datetime, collections
import numpy as np, pandas as pd
sys.path.insert(0, "src")
SEED = 42
FEAT = "features/celebdf_features.csv"
OUT = "splits/celebdf_dev_test.json"

def idints(p):
    return tuple(sorted(int(x) for x in re.findall(r"id(\d+)", os.path.basename(str(p)))))
def base(p): return os.path.basename(str(p))
def commit():
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except: return "nogit"

d = pd.read_csv(FEAT); d["ids"] = d.video_path.map(idints)
allids = sorted({i for t in d.ids for i in t})
adj = collections.defaultdict(collections.Counter); vidmass = collections.Counter()
for t in d.ids:
    for i in t: vidmass[i] += 1
    if len(t) == 2: adj[t[0]][t[1]] += 1; adj[t[1]][t[0]] += 1
# connected components among identities
seen = set(); comps = []
for i in allids:
    if i in seen: continue
    st = [i]; c = set()
    while st:
        x = st.pop()
        if x in seen: continue
        seen.add(x); c.add(x); st += [y for y in adj[x] if y not in seen]
    comps.append(sorted(c))
comps.sort(key=len, reverse=True)
c0 = comps[0]
# balanced spectral sweep cut of the dominant component c0
idx = {i: k for k, i in enumerate(c0)}; n = len(c0); W = np.zeros((n, n))
for a in c0:
    for b, cnt in adj[a].items():
        if b in idx: W[idx[a], idx[b]] = cnt
L = np.diag(W.sum(1)) - W
_, vecs = np.linalg.eigh(L); fiedler = vecs[:, 1]
order = [c0[k] for k in np.argsort(fiedler)]
masses = np.array([vidmass[i] for i in order]); cut = int(np.argmin(np.abs(np.cumsum(masses) - masses.sum()/2))) + 1
devc0, testc0 = set(order[:cut]), set(order[cut:])
# whole small components: c1->dev, c2->test, c3->test (deterministic)
part = {}
for i in comps[1]: part[i] = "dev"
for i in comps[2]: part[i] = "test"
for i in comps[3]: part[i] = "test"
for i in devc0: part[i] = "dev"
for i in testc0: part[i] = "test"
# YouTube-real (no id): deterministic alternate on sorted basename
yt = sorted(base(p) for p, t in zip(d.video_path, d.ids) if not t)
ytpart = {b: ("dev" if k % 2 == 0 else "test") for k, b in enumerate(yt)}

def vpart(row):
    t = row["ids"]
    if not t: return ytpart[base(row["video_path"])]
    ps = {part[i] for i in t}
    return ps.pop() if len(ps) == 1 else "DROP"
d["partition"] = d.apply(vpart, axis=1)

# ---- ASSERT identity-disjoint: no integer id in both dev and test ----
dev_ids = {i for t in d[d.partition=="dev"].ids for i in t}
test_ids = {i for t in d[d.partition=="test"].ids for i in t}
overlap = dev_ids & test_ids
assert not overlap, f"IDENTITY OVERLAP between dev and test: {sorted(overlap)}"
# and no placed video spans (guaranteed by DROP, but re-check)
for _, r in d[d.partition.isin(["dev","test"])].iterrows():
    ps = {part[i] for i in r["ids"]} if r["ids"] else {r["partition"]}
    assert len(ps) == 1, f"spanning video placed: {base(r['video_path'])}"
print("IDENTITY-DISJOINT assertion PASSED (0 shared identities across dev/test)", flush=True)

cnt = d.groupby(["partition","label"]).size().unstack(fill_value=0)
def g(p,l): return int(cnt.loc[p,l]) if (p in cnt.index and l in cnt.columns) else 0
out = dict(
    seed=SEED, method="component assignment + balanced spectral sweep cut of dominant component; spanning fakes dropped",
    source=FEAT, git_commit=commit(), date=datetime.date.today().isoformat(),
    identity_components={f"comp{k}": len(c) for k, c in enumerate(comps)},
    id2split={str(i): part[i] for i in allids},
    youtube2split=ytpart,
    dropped_videos=sorted(base(p) for p, vp in zip(d.video_path, d.partition) if vp == "DROP"),
    counts=dict(
        dev=dict(total=int((d.partition=="dev").sum()), real=g("dev",0), fake=g("dev",1)),
        test_SEALED=dict(total=int((d.partition=="test").sum()), real=g("test",0), fake=g("test",1)),
        dropped=int((d.partition=="DROP").sum()),
        dev_identities=len(dev_ids), test_identities=len(test_ids)))
os.makedirs("splits", exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out["counts"], indent=1))
print(f"saved {OUT} (commit {commit()})")
