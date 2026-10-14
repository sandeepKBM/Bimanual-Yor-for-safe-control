"""wrongpoint/mech.py -- list the drops of the point/conf arms with the carry's true mu and the prior error (ln units), pooled over cells."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); glob = sys.argv[2] if len(sys.argv) > 2 else "*"
rows = []
for d in sorted(ROOT.glob(glob)):
    for a in sorted(d.glob("*@*.jsonl")):
        for l in a.read_text().splitlines():
            r = json.loads(l)
            if r["dropped"]:
                rows.append((d.name, a.stem, r["ep"], r["mu"], r["prior_conf"], r["prior_err"], r["drop_tick"], r["progress"]))
print(f"{len(rows)} drops")
for x in rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 40]:
    print(f"{x[0]:28s} {x[1]:10s} ep{x[2]:4d} mu={x[3]*1e3:.2f}e-3 conf={int(x[4])} err={x[5]:+.2f} drop_tick={x[6]} prog={x[7]:.2f}")
import collections
print("conf-wrong share:", np.mean([x[4] for x in rows]) if rows else None, " err>0 (over-estimated) share:", np.mean([x[5] > 0 for x in rows]) if rows else None,
      " mu median:", np.median([x[3] for x in rows]) if rows else None)
