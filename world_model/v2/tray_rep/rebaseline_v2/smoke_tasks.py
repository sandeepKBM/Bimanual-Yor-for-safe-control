"""Builds the smoke command list: one task line per distinct (filter-or-arm, setup) of each v2 task file, turned into 10 VAL carries."""
import re, sys
from pathlib import Path
H = Path(__file__).resolve().parent / "slurm"; RB = "/common/users/ss5772/bimanual_data/tray_rep/rebaseline_v2"
out = []
for name, mod, key in (("expA", "bridge.run", "--filter"), ("expB", "bridge.run", "--filter"), ("cf5", "boed.run_fxi", "--filt")):
    seen = set()
    for l in (H / f"{name}_v2_tasks.txt").read_text().splitlines():
        tok = l.split(); spec = tok[tok.index(key) + 1]; setup = tok[tok.index("--setup") + 1]
        k = (re.sub(r"rho=[0-9.]+,?|,pseed=0", "", spec) if spec.startswith("F5P") else spec, setup, "val" if "--split val" in l else "test")
        if k in seen: continue
        seen.add(k)
        l = l.replace("--split test", "--split val").replace(f"{RB}/", f"{RB}/smoke/") + " --n-eps 10"
        if name == "cf5": l += " --shard 0/1"
        out.append(f"{mod} {l} --workers 2")
print("\n".join(out))
