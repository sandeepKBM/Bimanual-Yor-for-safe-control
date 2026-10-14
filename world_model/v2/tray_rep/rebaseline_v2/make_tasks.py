"""Writes slurm/{expA,expB,cf5}_v2_tasks.txt for the fixed-physics re-runs (v2 data + v2 calibration). Pure text generation."""
import json, sys
from pathlib import Path
D = "/common/users/ss5772/bimanual_data/tray_rep"; RB = f"{D}/rebaseline_v2"; HERE = Path(__file__).resolve().parent / "slurm"
DATA = {s: f"{D}/data_{s}_full_v2" for s in ("piper", "piper_trk")}
cal = json.loads(Path(f"{RB}/vlmprior_cal/online_calib.json").read_text())["chosen"]
EPS = 0.0                                      # rule-chosen on v2 VAL (calib_v2/eps_a_v2.json)
V2 = f"calib_file=v2,eps_a={EPS:g}"
tr = lambda s: ("piper_trk", "piper")[s == "piper"]
lines = {"A": [], "B": [], "C": []}
# ---------------- Experiment A (bridge.run). F3 dropped.
def A(filt, setup, seed, split="test", extra=""):
    lines["A"].append(f"--filter {filt} --setup {setup} --seed {seed} --split {split} --tag main --out {RB}/sets_expA --data {DATA[setup]}{extra}")
for setup, seeds in (("piper_trk", (0, 1, 2)), ("piper", (0,))):
    f1 = "F1:l_align=4,l_prox=0" if setup == "piper_trk" else "F1:l_align=4,l_prox=10"
    for sd in seeds:
        A(f"F5:{V2}", setup, sd); A(f"F5:calib=0,eps_a={EPS:g}", setup, sd); A(f1, setup, sd)
    A("F0", setup, 0); A("F2:theta_max=0.05,a_cap=0.4", setup, 0); A("F4:check_every=5", setup, 0, extra=" --n-eps 100")
    for sd in seeds:                                       # sensitivity: the old physics' eps_a = 0.08 kept
        A("F5:calib_file=v2,eps_a=0.08", setup, sd)
    for sd in seeds:                                       # VAL (as in the old expA)
        A(f"F5:{V2}", setup, sd, "val"); A(f"F5:calib=0,eps_a={EPS:g}", setup, sd, "val")
# ---------------- Experiment B (F5P arms of PREREG_B_FROZEN; same arm/rho/seed grid as sets/slurm/expB_tasks.txt)
old = [l for l in Path("/common/users/ss5772/bimanual/world_model/v2/tray_rep/sets/slurm/expB_tasks.txt").read_text().splitlines() if l.strip()]
for l in old:
    tok = l.split()
    spec = tok[tok.index("--filter") + 1]; setup = tok[tok.index("--setup") + 1]
    c = cal[setup]
    spec2 = spec + ("," if ":" in spec else ":") + f"{V2},lam={c['lam']:g},sa={c['sigma_a']:g}"
    tok[tok.index("--filter") + 1] = spec2
    tok[tok.index("--out") + 1] = f"{RB}/vlmprior_expB"
    lines["B"].append(" ".join(tok) + f" --data {DATA[setup]}")
# ---------------- C-F5 (boed/run_fxi): fxi_tasks_test.txt minus F3
fx = [l for l in Path("/common/users/ss5772/bimanual/world_model/v2/tray_rep/boed/slurm/fxi_tasks_test.txt").read_text().splitlines() if l.strip()]
for l in fx:
    tok = l.split()
    spec = tok[tok.index("--filt") + 1]
    if spec == "F3":
        continue
    setup = tok[tok.index("--setup") + 1]; c = cal[setup]
    if spec == "F5":
        spec2 = f"F5:{V2}"
    else:
        spec2 = spec + f",{V2},lam={c['lam']:g},sa={c['sigma_a']:g}"
    tok[tok.index("--filt") + 1] = spec2
    tok[tok.index("--out") + 1] = f"{RB}/boed_f5"
    lines["C"].append(" ".join(tok) + f" --data {DATA[setup]}")
names = {"A": "expA_v2_tasks.txt", "B": "expB_v2_tasks.txt", "C": "cf5_v2_tasks.txt"}
for k, v in lines.items():
    (HERE / names[k]).write_text("\n".join(v) + "\n")
    print(names[k], len(v))
