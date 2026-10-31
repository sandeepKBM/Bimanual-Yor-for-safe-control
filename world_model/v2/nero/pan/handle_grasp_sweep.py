"""Sweep pan/handle_grasp.run.  SIMULATION ONLY, CPU.
  --set v1 (default): placements x grip modes (+ squeeze-force sweep on the empty pan), pinch at 0.6.
  --set v2: one-claw grasp-position sweep + two-claw lifts (force grip 10 N).
    python -m world_model.v2.nero.pan.handle_grasp_sweep OUT_DIR [--procs 8] [--set v2]"""
import json, os, sys, traceback
from multiprocessing import Pool
from pathlib import Path

PLACES = ("none", "near_handle", "far_centre", "far_left", "far_right", "overload")
# left claw: grasp_frac = FINGERTIP point; the fingers cover the handle from there towards its end (render check 10-28)
CASES_V2 = [dict(placement=p, grip="force", grip_force=10.0, grasp_frac=f) for p in PLACES for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8)] + \
    [dict(placement=p, grip="force", grip_force=10.0, grasp_frac=lf, second_frac=rf, second_grip=g)
     for p in PLACES for (g, rf, lf) in (("tilt", 0.2, 0.45), ("tilt", 0.2, 0.6), ("tilt", 0.3, 0.6), ("side", 0.5, 0.7))]
CASES_V3 = [dict(placement=p, grip="force", grip_force=10.0, grasp_frac=lf, second_frac=rf, second_grip=g)
            for p in PLACES for (g, rf, lf) in (("tilt", 0.2, 0.6), ("tilt", 0.3, 0.6), ("side", 0.5, 0.7))]   # right claw wraps the handle
CASES = [(p, g, 10.0) for p in ("none", "near_handle", "far_centre", "far_left", "far_right", "overload") for g in ("servo", "force")] + \
        [("none", "force", f) for f in (2.5, 5.0, 7.5)]


def one_v2(kw):
    from world_model.v2.nero.pan import handle_grasp as H
    tag = f"{kw['placement']}_L{round(kw['grasp_frac'] * 100)}" + (f"_R{kw['second_grip']}{round(kw['second_frac'] * 100)}" if "second_frac" in kw else "")
    out = Path(sys.argv[1]) / f"{'v3' if 'v3' in sys.argv else 'v2'}_{tag}.json"
    try:
        r = H.run(**kw)
    except Exception:
        r = dict(kw, error=traceback.format_exc()[-800:])
    out.write_text(json.dumps(r, indent=1, default=float))
    return out.name


def one(case):
    from world_model.v2.nero.pan import handle_grasp as H
    p, g, f = case
    out = Path(sys.argv[1]) / f"{p}_{g}{'' if g == 'servo' else int(f * 10)}.json"
    try:
        r = H.run(p, grip=g, grip_force=f)
    except Exception:
        r = dict(placement=p, grip=g, grip_force_N=f, error=traceback.format_exc()[-800:])
    out.write_text(json.dumps(r, indent=1, default=float))
    return out.name


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 8
    st = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "v1"
    v2 = st in ("v2", "v3")
    with Pool(n, maxtasksperchild=3) as pool:
        for name in pool.imap_unordered(one_v2 if v2 else one, {"v2": CASES_V2, "v3": CASES_V3}.get(st, CASES)):
            print(name, flush=True)
    (Path(sys.argv[1]) / f"_DONE_{st}").write_text("ok")
