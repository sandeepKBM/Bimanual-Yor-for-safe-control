"""eps_a re-tuning of F5's reachable sets on the v2 (fixed-physics) VAL data. Wraps sets/validate_reach.py without editing it:
data_{setup}_full is redirected to data_{setup}_full_v2 by a ROOT shim. Same protocol as the original: branches every 8th tick from tick 10,
VAL episodes 0-79 = tuning, 80-159 = held-out check.
  collect:  python -m ...val_reach_v2 collect --setup piper_trk --ep-start 0      (writes calib_v2/coverage_branches/*.npz)
  eval:     python -m ...val_reach_v2 eval --out calib_v2/eps_a_v2.json            (all 4 branch files; old-vs-v2 mu map; eps grid)
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
assert not os.environ.get("BIMANUAL_LEGACY_PIN")
from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.sets import validate_reach as V
from world_model.v2.tray_rep.sets import reach as RC

REAL = K.ROOT
class _Root:
    def __truediv__(self, name):
        return REAL / (name + "_v2" if str(name).startswith("data_") else name)
V.K = type("KShim", (), {"ROOT": _Root()})            # validate_reach only uses K.ROOT for the data dir (OUT is evaluated at import)
CAL = REAL / "calib_v2"
GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5]

def bpath(setup, start):
    return CAL / "coverage_branches" / f"branches_{setup}_val_kd3_kp2_H60_n80_s{start}.npz"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "eval"])
    ap.add_argument("--setup", default="piper_trk"); ap.add_argument("--ep-start", type=int, default=0)
    ap.add_argument("--n-eps", type=int, default=80); ap.add_argument("--out", default=str(CAL / "eps_a_v2.json"))
    a = ap.parse_args()
    if a.cmd == "collect":
        rp = RC.ReachParams()
        p = bpath(a.setup, a.ep_start) if a.n_eps == 80 else CAL / "coverage_branches" / f"smoke_{a.setup}_n{a.n_eps}_s{a.ep_start}.npz"
        recs = V.collect(a.setup, "val", a.n_eps, a.ep_start, 8, 10, 60, rp, p)
        print(f"collected {len(recs)} branches -> {p}", flush=True); return
    res = {}
    for setup in ["piper", "piper_trk"]:
        meta = json.loads((REAL / f"data_{setup}_full_v2" / "meta.json").read_text())
        a_max, w_max = 5.0 * meta["a_scale"], 2.0 * meta["w_scale"]
        tune = V.load_records(bpath(setup, 0)); held = V.load_records(bpath(setup, 80))
        for cname, cf in [("old_map", ""), ("v2_map", "v2")]:
            rp = RC.ReachParams(calib_file=cf)
            rows = []
            for e in GRID:
                q = RC.ReachParams(**{**rp.__dict__, "eps_a": e})
                rt = V.evaluate(tune, q, a_max, w_max, meta["alpha"]); rh = V.evaluate(held, q, a_max, w_max, meta["alpha"])
                rows.append(dict(eps_a=e, tune_traj=rt["traj_cov"], tune_step=rt["step_cov"], held_traj=rh["traj_cov"], held_step=rh["step_cov"],
                                 hw_mm_k10_30_60=[rt["mean_hw_mm_at"][1], rt["mean_hw_mm_at"][3], rt["mean_hw_mm_at"][6]]))
            chosen = next((r["eps_a"] for r in rows if r["tune_traj"] >= 0.95), None)
            res[f"{setup}/{cname}"] = dict(n_tune=len(tune), n_held=len(held), chosen_eps_a=chosen, rows=rows)
            print(setup, cname, "chosen eps_a", chosen, flush=True)
            for r in rows: print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
