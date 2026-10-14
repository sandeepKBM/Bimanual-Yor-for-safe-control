"""rebaseline_v2/analysis/completeness.py -- completeness / duplicate / Slurm EXIT check of the 4 rebaseline arrays. SAVED DATA ONLY."""
import glob, json, re, sys
from collections import Counter
from pathlib import Path
from world_model.v2.tray_rep.bridge import filters as F
from world_model.v2.tray_rep.bridge import run as R

RB = Path("/common/users/ss5772/bimanual_data/tray_rep/rebaseline_v2")
TASKS = Path("/common/users/ss5772/bimanual/world_model/v2/tray_rep/rebaseline_v2/slurm")
out = dict(gaps=[], dups=[], checked=0)

def argd(line):
    t = line.split(); d = {}
    for i, x in enumerate(t):
        if x.startswith("--") and i + 1 < len(t) and not t[i + 1].startswith("--"):
            d[x[2:]] = t[i + 1]
    return d

def chk(path, want_n, label):
    rows = R.read_jsonl(path) if Path(path).exists() else []
    eps = [r["ep"] for r in rows]
    c = Counter(eps)
    dup = [e for e, k in c.items() if k > 1]
    missing = [e for e in range(want_n) if e not in c]
    extra = [e for e in c if e >= want_n]
    out["checked"] += 1
    if dup: out["dups"].append((label, len(dup)))
    if missing or extra: out["gaps"].append((label, f"missing {len(missing)} extra {len(extra)} (n_lines={len(rows)})"))
    return len(rows)

# expA and expB (bridge.run layout)
val_counts = Counter()
for name, root in (("expA_v2", RB / "sets_expA"), ("expB_v2", RB / "vlmprior_expB")):
    for ln in (TASKS / f"{name}_tasks.txt").read_text().splitlines():
        if not ln.strip(): continue
        d = argd(ln)
        spec = d["filter"]; setup = d["setup"]; seed = int(d["seed"]); split = d["split"]
        _, pj, js = R.paths(root, d["tag"], split, setup, spec, seed)
        n_want = 100 if spec.startswith("F4") else 600
        if split == "val":
            n_want = len(R.read_jsonl(pj))      # val: contiguity + dup only
            val_counts[(setup, n_want)] += 1
        n = chk(pj, n_want, f"{name}|{setup}|{split}|{spec}|s{seed}")
        if not js.exists(): out["gaps"].append((f"{name}|{setup}|{split}|{spec}|s{seed}", "no summary .json"))
# val: check eps contiguous from 0 (done by chk with n_want=len) ; report val sizes
out["val_sizes"] = {f"{k[0]}:{k[1]}": v for k, v in val_counts.items()}

# BOED (shards)
for name, root, pre in (("cf5_v2", RB / "boed_f5", "FIX"), ("cf5cl_v2", RB / "boed_f5_cl", "CL")):
    for ln in (TASKS / f"{name}_tasks.txt").read_text().splitlines():
        if not ln.strip(): continue
        d = argd(ln)
        spec, I, seed = d["filt"], float(d["I"]), int(d["seed"]); i, n = d["shard"].split("/")
        p = root / F.spec_tag(spec) / d["split"] / d["setup"] / f"{pre}__I-{I:g}__s{seed}__p{i}of{n}.jsonl"
        rows = R.read_jsonl(p); eps = [r["ep"] for r in rows]
        c = Counter(eps); out["checked"] += 1
        want = set(range(int(i), 600, int(n)))
        lab = f"{name}|{spec}|I{I:g}|s{seed}|p{i}of{n}"
        if set(c) != want: out["gaps"].append((lab, f"eps differ: have {len(c)} want {len(want)} (lines {len(rows)})"))
        if any(k > 1 for k in c.values()): out["dups"].append((lab, sum(k > 1 for k in c.values())))
    # union per (spec, I, seed)
    for ln in (TASKS / f"{name}_tasks.txt").read_text().splitlines():
        pass

# union check per config
for name, root, pre in (("cf5_v2", RB / "boed_f5", "FIX"), ("cf5cl_v2", RB / "boed_f5_cl", "CL")):
    cfgs = set()
    for ln in (TASKS / f"{name}_tasks.txt").read_text().splitlines():
        if ln.strip():
            d = argd(ln); cfgs.add((d["filt"], float(d["I"]), int(d["seed"])))
    for spec, I, seed in sorted(cfgs):
        eps = []
        for p in glob.glob(str(root / F.spec_tag(spec) / "test" / "piper_trk" / f"{pre}__I-{I:g}__s{seed}__p*of4.jsonl")):
            eps += [r["ep"] for r in R.read_jsonl(p)]
        if sorted(eps) != list(range(600)): out["gaps"].append((f"{name} union {spec} I{I:g} s{seed}", f"{len(eps)} recs, {len(set(eps))} unique"))

# slurm logs
ex = Counter(); bad = []
for p in glob.glob(str(RB / "logs" / "slurm_*.out")):
    t = Path(p).read_text()
    m = re.findall(r"^EXIT (\d+)", t, re.M)
    ex[tuple(m)] += 1
    if m != ["0"] or "Traceback" in t: bad.append(Path(p).name)
out["slurm_exit"] = {str(k): v for k, v in ex.items()}; out["slurm_bad"] = bad
out["slurm_counts"] = dict(Counter(re.sub(r"_\d+\.out$", "", Path(p).name) for p in glob.glob(str(RB / "logs" / "slurm_*.out"))))
print(json.dumps(out, indent=1, default=str))
(RB / "analysis" / "completeness.json").write_text(json.dumps(out, indent=1, default=str))
