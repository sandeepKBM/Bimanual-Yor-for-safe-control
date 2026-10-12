"""al/make_tasks_al.py -- write the TEST Slurm task list of AL-1 (one run_al call per line; 4 shards x 600 TEST carries per (policy, regime, seed)). Does NOT submit anything.
  python -m world_model.v2.tray_rep.al.make_tasks_al [--value placeholder | cells:<cells.csv>] [--out slurm/al_test_tasks.txt]
Policies: AL_static AL_online AL_online0 AL_probe AL_transductive AL_oracle AL_unif AL_point  x regimes normal, stressed  x seeds 0-2 (tracker-noise seed = prior seed)  x shards 0..3.
Task order: seed-major, so a partial run gives complete seeds first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

D = "/common/users/ss5772/bimanual_data/tray_rep"
POLICIES = ["AL_static", "AL_online", "AL_online0", "AL_probe", "AL_transductive", "AL_oracle", "AL_unif", "AL_point"]
REGIMES = ["normal", "stressed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="placeholder")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).parent / "slurm" / "al_test_tasks.txt"))
    a = ap.parse_args()
    lines = []
    for s in [int(x) for x in a.seeds.split(",")]:
        for reg in REGIMES:
            for pol in POLICIES:
                for i in range(a.shards):
                    extra = f" --voi {D}/al/voi/voi_{reg}.json --value {a.value}" if pol == "AL_transductive" else ""
                    lines.append(f"--policy {pol} --regime {reg} --split test --seed {s} --shard {i}/{a.shards} --out {D}/al/runs --data {D}/data_piper_trk_full_v2{extra}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks -> {a.out}  (sbatch --array=0-{len(lines) - 1})")


if __name__ == "__main__":
    main()
