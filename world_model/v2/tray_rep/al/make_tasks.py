"""al/make_tasks.py -- task lines for the value-of-information map Slurm array (one line = one (I, auth, pop, set) cell of N VAL carries). No hardware."""
import argparse
from pathlib import Path

OUT = "/common/users/ss5772/bimanual_data/tray_rep/al/vimap"
SPEEDS = (0.4, 0.55, 0.75, 0.99)
AUTHS = (1.0, 0.5, 0.25)
POPS = ("full", "low")
SETS = ("oracle", "narrow", "medium", "mis", "full")      # + 'popr' (the regime's own population range) for pop=low only


def lines(n_eps=80, out=OUT, shards=1):
    L = []
    for pop in POPS:
        for auth in AUTHS:
            for I in SPEEDS:
                for s in SETS + (("popr",) if pop == "low" else ()):
                    for k in range(shards):
                        L.append(f"--auth {auth:g} --pop {pop} --set {s} --I {I:g} --split val --seed 0 --n-eps {n_eps} --shard {k}/{shards} --out {out}")
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=80)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "slurm" / "vimap_tasks.txt"))
    a = ap.parse_args()
    L = lines(a.n_eps, a.out, a.shards)
    Path(a.tasks).write_text("\n".join(L) + "\n")
    print(len(L), "tasks ->", a.tasks)
