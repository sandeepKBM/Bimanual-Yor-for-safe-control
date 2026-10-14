"""wrongpoint/cond.py -- drop rate conditional on the carry's prior being confidently wrong with err>0 (mu over-estimated) / err<0, and mu bin."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); glob = sys.argv[2] if len(sys.argv) > 2 else "*"; rho = sys.argv[3] if len(sys.argv) > 3 else "0.4"
def rd(p):
    return {json.loads(l)["ep"]: json.loads(l) for l in p.read_text().splitlines()} if p.exists() else {}
print(f"rho={rho}: conditional drop rates (point / conf / oracle), n = carries in stratum")
for d in sorted(ROOT.glob(glob)):
    if not (d / "cfg.json").exists(): continue
    P, C, O = rd(d / f"point@{rho}.jsonl"), rd(d / f"conf@{rho}.jsonl"), rd(d / "oracle.jsonl"); F0 = rd(d / "F0.jsonl")
    eps = sorted(set(P) & set(C) & set(O))
    if not eps: continue
    out = [f"{d.name:26s}"]
    for lab, f in (("over-est conf-wrong", lambda r: r["prior_conf"] and r["prior_err"] > 0),
                   ("under-est conf-wrong", lambda r: r["prior_conf"] and r["prior_err"] < 0),
                   ("honest", lambda r: not r["prior_conf"])):
        s = [e for e in eps if f(P[e])]
        out.append(f"{lab}: n={len(s)} {sum(P[e]['dropped'] for e in s)}/{sum(C[e]['dropped'] for e in s)}/{sum(O[e]['dropped'] for e in s)}")
    s = [e for e in eps if P[e]["prior_conf"] and P[e]["prior_err"] > 0 and P[e]["mu"] < 1e-3]
    out.append(f"over-est & mu<1e-3: n={len(s)} {sum(P[e]['dropped'] for e in s)}/{sum(C[e]['dropped'] for e in s)}/{sum(O[e]['dropped'] for e in s)} F0 {sum(F0[e]['dropped'] for e in s if e in F0)}")
    print(" | ".join(out))
