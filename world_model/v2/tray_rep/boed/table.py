"""boed/table.py -- build the predicted-drop-vs-intensity table from the VAL cell runs, and print it. SIMULATION / SAVED DATA ONLY.

Cells are runs of the policy spec "FIX:I=<I_rel>,j=<node index>" (episode.py): the F3-filtered carry at intensity I_rel with the hidden
mu_roll forced to NODES[j], on VAL carries. They are ordinary boed runs (same resumable jsonl layout), so the launcher and the Slurm array
build them like any other job. `assemble` pools the value seeds it finds (pool = all seeds, or one seed with --seed) and writes
<out>/<tag>/table_<setup>.json plus table_<setup>.md.

  python -m world_model.v2.tray_rep.boed.table assemble --setup piper_trk --tag main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as RN


def assemble(out, tag, setup, seeds=None, igrid=PO.IGRID, nodes=PO.NODES, split="val"):
    k = np.zeros((len(igrid), len(nodes))); n = np.zeros_like(k)
    prog = np.zeros_like(k); goal = np.zeros_like(k); ticks = np.zeros_like(k)
    seen = set()
    for i, I in enumerate(igrid):
        for j in range(len(nodes)):
            spec = f"FIX:I={I:g},j={j}"
            for vs in (seeds if seeds is not None else range(10)):
                recs = RN.read_all(out, tag, split, setup, spec, vs)
                if not recs:
                    continue
                seen.add(vs)
                k[i, j] += sum(r["dropped"] for r in recs); n[i, j] += len(recs)
                prog[i, j] += sum(r["progress"] for r in recs); goal[i, j] += sum(r["goal_reached"] for r in recs)
                ticks[i, j] += sum(r["n_ticks"] for r in recs)
    nn = np.maximum(n, 1)
    t = PO.Table(igrid, nodes, k, n, extra=dict(progress=(prog / nn).tolist(), goal=(goal / nn).tolist(), ticks=(ticks / nn).tolist()),
                 meta=dict(setup=setup, split=split, seeds=sorted(seen), tag=tag))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["assemble"])
    ap.add_argument("--setup", required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(RN.OUT))
    ap.add_argument("--seed", type=int, default=None, help="one value seed (default: pool all)")
    ap.add_argument("--igrid", default=",".join(f"{x:g}" for x in PO.IGRID))
    ap.add_argument("--n-nodes", type=int, default=PO.N_NODES)
    a = ap.parse_args()
    ig = tuple(float(x) for x in a.igrid.split(","))
    from world_model.v2.tray_rep import env as E
    nodes = np.exp(np.linspace(np.log(E.MU_LO), np.log(E.MU_HI), a.n_nodes))
    t = assemble(Path(a.out), a.tag, a.setup, None if a.seed is None else [a.seed], ig, nodes)
    p = Path(a.out) / a.tag / f"table_{a.setup}{'' if a.seed is None else f'_s{a.seed}'}"
    p.parent.mkdir(parents=True, exist_ok=True)
    t.save(str(p) + ".json")
    Path(str(p) + ".md").write_text(t.text() + "\n")
    print(t.text(), flush=True)
    print("wrote", str(p) + ".json", flush=True)


if __name__ == "__main__":
    main()
