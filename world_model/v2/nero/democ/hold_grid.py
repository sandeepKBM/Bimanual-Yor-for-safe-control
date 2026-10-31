"""democ/hold_grid.py -- merge per-case hold_profile JSONs into the small grid result.  SIM ONLY.
Per case (CPU, ~3-4 min each):  python -m world_model.v2.nero.democ.hold_profile --mu M --mass W [--fix] --out DIR/mu{M}_m{W}_{base|fix}.json
Merge:  python -m world_model.v2.nero.democ.hold_grid DIR OUT.json
"""
import json
import sys
from pathlib import Path


def merge(d, out):
    rows = []
    for f in sorted(Path(d).glob("mu*_m*_*.json")):
        r = json.loads(f.read_text())
        rows.append(dict(
            mu=r["mu"], mass=r["mass"], variant=r["variant"],
            slip_mm=max(r["slip_max_mm"].values()), slip_deg=max(r["slip_max_deg"].values()),
            creep_mm_s=max(r["creep_mm_s"].values()), dropped=r["box_dropped"], tilt_reached_deg=r["max_tilt_deg"],
            tau_peak_Nm=r["tau_peak_Nm"]["max"], tau_limit_Nm=r["tau_limit_Nm"],
            grip_force_max_N=max(v["max"] for v in r["gripper_actuator_force_N"].values()), grip_force_limit_N=r["gripper_force_limit_N"],
            n_infeasible=r["n_infeasible"], meets_target=r["meets_target"], box_end_tilt_deg=r["box_end_tilt_deg"],
            box_z_end_minus_start_mm=1e3 * (r["box_z_m"]["end"] - r["box_z_m"]["start"])))
    res = dict(profile="grasp->lift 5cm->rot +y 40deg->back->set down->release", target=dict(slip_mm=2.0, slip_deg=1.0, creep_mm_s=0.2),
               note="slip = max over the profile of box pose change in each gripper frame (includes elastic pad compliance); "
                    "fix tuned on mu0.5/0.6kg only; box mass/mu are ASSUMED sim parameters", fix_params=json.loads(next(Path(d).glob("*_fix.json")).read_text())["fix"],
               rows=rows)
    Path(out).write_text(json.dumps(res, indent=1))
    for x in rows:
        print(f"mu{x['mu']} m{x['mass']} {x['variant']:8s} slip {x['slip_mm']:6.2f}mm {x['slip_deg']:5.2f}deg creep {x['creep_mm_s']:5.2f} drop {x['dropped']} tilt {x['tilt_reached_deg']:.1f} tau {x['tau_peak_Nm']:.1f} F {x['grip_force_max_N']:.1f} inf {x['n_infeasible']} ok {x['meets_target']}")


if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2])
