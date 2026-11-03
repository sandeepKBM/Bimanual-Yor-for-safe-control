"""pan/demo_cycle.py -- step 8 of the 10-29 segment rule: try, detect the slip from joint torques, back off, confirm task-ready,
re-plan with the narrowed Theta, try again.  SIMULATION ONLY, CPU only.

Attempt 1 uses the plan's payload model (pan + lid, contents unknown).  The lift runs under segments.default_monitor (encoder gaps
+ right-claw load share from contact torques); at its end the claws must have risen >= LIFT_MIN * LIFT (encoders).  During the lift the robot also reads the load (segments.load_estimate).  If the
monitor aborts, the lift backup runs (lower, release right, pull the pan back, release left, retreat) and in_R is checked.
Re-plan: contents mass interval narrowed to [m_hat - MASS_TOL, m_hat + MASS_TOL] (ASSUMED tolerance, from n = 2 calibration holds),
payload model = pan + lid + m_hat; if the upper end exceeds REFUSE_KG (ASSUMED lift-table entry) the robot refuses and stays in R.
Attempt 2 re-approaches the handle (undo the retreat), then pinch, push, clamp, lift with the new payload model.

Ground truth (success, U entries, R) is read from the simulator; the robot only uses gaps, joint torques and the load estimate.

    python -m world_model.v2.nero.pan.demo_cycle --out DIR [--tags ...] [--set overload|full|iid] --procs P [--video]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

MASS_TOL = 0.1    # ASSUMED: half-width of the narrowed contents-mass interval [kg] (calibration: 0.5 N error at n = 2)
REFUSE_KG = 0.65  # ASSUMED lift table: contents above this -> refuse.  Set from the 10-29 lever test (2 poses, payload model = true
                  # mass): 0.5 / 0.6 kg lifted, 0.7 / 0.8 kg twisted in the left claw.  Not a bound.
LIFT_MIN = 0.8    # ASSUMED: lift boundary check = both claws rose >= LIFT_MIN * LIFT (arm encoders / FK)


def _seg(ep, G, seg, monitor=None, loads=None):
    def mon(e):
        if loads is not None and seg == "lift" and (e.k - st["k0"]) * G.DT > G.MON["t_share"]:
            loads.append(G.load_estimate(e)["contents_kg"])
        return monitor(e) if monitor else None
    st = dict(k0=ep.k)
    if seg == "push":
        ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
    status, info = ep.run_phases(ep.segment_phases(seg), monitor=mon)
    return status, info.get("why"), info.get("t")


def _boundary(ep, G, seg):
    o = ep.observe_boundary(seg)
    return G.boundary_ok(ep, seg, dict(nominal=dict(status="done", observed=o))), o


def attempt(ep, G, label, out, loads):
    """pinch -> push -> clamp -> lift; returns 'lifted' | 'aborted:<seg>' after running that segment's backup."""
    rec = dict(label=label, payload_model_kg=round(float(ep.ctl.mass), 3), segs=[])
    out["attempts"].append(rec)
    for seg in G.SEGMENTS:
        mon = G.default_monitor(seg)
        status, why, t = _seg(ep, G, seg, monitor=mon, loads=loads)
        if seg == "lift":
            rise = min(float(ep.tcp(sd)[2] - ep.mem["lift_start"][sd][2]) for sd in G.SIDES)
            ok_b, obs = rise >= LIFT_MIN * G.LIFT, dict(rise_mm=round(1e3 * rise, 1), load=G.load_estimate(ep))
            if status == "done" and not ok_b:
                why = f"lift short: claws rose {1e3 * rise:.1f} mm of {1e3 * G.LIFT:.0f} (payload model too light?)"
        else:
            ok_b, obs = _boundary(ep, G, seg)
        s = dict(seg=seg, status=status, why=why, t_abort=t, boundary_ok=ok_b, observed=obs,
                 success_truth=ep.success(seg) if status == "done" else False, U=ep.track["viol"])
        rec["segs"].append(s)
        if status != "done" or not ok_b:
            ep.run_phases(ep.backup_phases(seg))
            ok, mg, st = ep.in_R(reach=True)
            rec.update(outcome=f"aborted:{seg}", retreats_m=ep.sense.get("retreats_m"), backup_inR=ok, R_margin=mg, R_state=st, U_after_backup=ep.track["viol"])
            return rec["outcome"]
    rec.update(outcome="lifted", task_success=ep.success("lift"), U=ep.track["viol"])
    return "lifted"


def run(world, tag, plan=None, video=None):
    from . import segments as G
    plan = plan or dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
    ep = G.Episode(world, observe="zed", obs_seed=G.world_seed(tag), **plan)
    out = dict(tag=tag, world=world, plan=plan, obs=ep.obs, refused=ep.refused, attempts=[], bounds=G.BOUNDS,
               assumptions=dict(MASS_TOL=(MASS_TOL, "ASSUMED"), REFUSE_KG=(REFUSE_KG, "ASSUMED"), LIFT_MIN=(LIFT_MIN, "ASSUMED"),
                                MON=(G.MON, "ASSUMED")))
    if ep.refused:
        out["result"] = "refused at B0"
        return out
    ep.reset_track()
    ep.run_phases(ep.segment_phases("start"))
    q_pre_left = ep.tcp("left").copy()
    loads = []
    r1 = attempt(ep, G, "attempt 1 (nominal payload model)", out, loads)
    if r1 == "lifted":
        out["result"] = "success on attempt 1" if out["attempts"][-1]["task_success"] else "lifted, not successful (attempt 1)"
        return out
    a1 = out["attempts"][-1]
    if not a1.get("backup_inR"):
        out["result"] = "COUNTEREXAMPLE: backup did not end in R"
        return out
    if r1 != "aborted:lift" or not loads:
        out["result"] = f"stopped task-ready after {r1} (no mass reading to re-plan with)"
        return out
    m_hat = float(np.max(loads))      # a slipping rim under-reads the load: take the largest reading
    theta = dict(contents_kg=[round(max(0.0, m_hat - MASS_TOL), 3), round(m_hat + MASS_TOL, 3)],
                 truth=None if ep.hid is None else round(float(ep.m.body_mass[ep.hid]), 3), n_readings=len(loads))
    out["narrowed_theta"] = theta
    if theta["contents_kg"][1] > REFUSE_KG:
        out["result"] = "refused after re-plan (narrowed mass above the lift table); task-ready"
        return out
    ep.ctl.mass = float(ep.m.body_mass[ep.pan] + ep.m.body_mass[ep.lid]) + max(m_hat, 0.0)
    back = q_pre_left - ep.tcp("left")
    ep.run_phases([("move_to", {"left": lambda: q_pre_left}, max(0.8, 1.6 * np.linalg.norm(back) / 0.06), max(np.linalg.norm(back), 1e-3)),
                   ("hold", 0.3)])
    r2 = attempt(ep, G, "attempt 2 (re-planned payload model)", out, [])
    a2 = out["attempts"][-1]
    if r2 == "lifted":
        out["result"] = "success on attempt 2" if a2["task_success"] else "lifted, not successful (attempt 2)"
    else:
        out["result"] = f"attempt 2 {r2}; backup in R = {a2.get('backup_inR')}"
    return out


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, w = job
    t0 = time.time()
    try:
        r = run(w, tag)
    except Exception:
        r = dict(tag=tag, world=w, error=traceback.format_exc()[-1500:])
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main(argv=None):
    from .segcert import worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="overload", choices=["full", "overload", "iid"])
    ap.add_argument("--tags", default=None, help="comma list; default = the whole set")
    ap.add_argument("--procs", type=int, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    ws = worlds(a.set)
    if a.tags:
        keep = set(a.tags.split(","))
        ws = [(t, w) for t, w in ws if t in keep]
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, ws):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            print(r["tag"], r.get("result"), r.get("narrowed_theta"), r.get("error", "")[-300:], flush=True)


if __name__ == "__main__":
    main()
