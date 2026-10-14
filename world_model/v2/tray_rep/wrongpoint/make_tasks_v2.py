"""Writes slurm/wrongpoint_v2_tasks.txt (+ wrongpoint_v2_smoke.txt): TEST re-run of the key wrongpoint cells under the closed-loop planner, fixed-physics
data/calibration. Pure text generation. Task = (cell, arm-config, shard i/4)."""
from pathlib import Path
D = "/common/users/ss5772/bimanual_data/tray_rep"; OUT = f"{D}/wrongpoint_v2"; DATA = f"{D}/data_piper_trk_full_v2"
HERE = Path(__file__).resolve().parent / "slurm"; HERE.mkdir(exist_ok=True)
V2 = "calib_file=v2,eps_a=0,lam=0.1,sa=0.04"
LOW = "--mu-range 5e-4,1.5e-3"
CELLS = {                                  # eps_scale is moot with eps_a=0 (box = 0 * eps_scale) and is dropped
    "lowI": f"--auth 0.25 {LOW}",
    "lowJ": f"--wrong 3,4 {LOW}",
    "lowE": f"--wrong 2,3 {LOW}",
    "lowK": f"--wrong 2,3 --auth 0.5 {LOW}",
    "lowA": f"{LOW}",
    "fullA": "",
}
def arms():
    out = []
    for a in ("point", "conf"):
        for rho in (0.2, 0.4):
            for ps in (0, 1, 2):
                out.append((f"{a}@{rho:g}__ps{ps}", f"F5P:arm={a},rho={rho:g},pseed={ps},{V2}", f"--rho {rho:g} --pseed {ps}"))
    out.append(("oracle", f"F5P:arm=oracle,{V2}", ""))
    out.append(("F0", "F0", ""))
    return out
def line(cell, knobs, key, filt, extra, split, shard, nep=""):
    return (f"--cell {cell} --arm-key {key} --filt {filt} {extra} --I 0.99 --setup piper_trk --seed 0 --split {split} --shard {shard}{nep} "
            f"{knobs} --data {DATA} --out {OUT}/{'runs' if split == 'test' else 'smoke'}").replace("  ", " ")
main, smoke = [], []
for cell, knobs in CELLS.items():
    for key, filt, extra in arms():
        for s in range(4):
            main.append(line(cell, knobs, key, filt, extra, "test", f"{s}/4"))
    for key in ("point@0.4__ps0", "conf@0.4__ps0", "oracle", "F0"):
        k, f, e = [x for x in arms() if x[0] == key][0]
        smoke.append(line(cell, knobs, k, f, e, "val", "0/1", " --n-eps 4"))
(HERE / "wrongpoint_v2_tasks.txt").write_text("\n".join(main) + "\n")
(HERE / "wrongpoint_v2_smoke.txt").write_text("\n".join(smoke) + "\n")
print(len(main), len(smoke))
