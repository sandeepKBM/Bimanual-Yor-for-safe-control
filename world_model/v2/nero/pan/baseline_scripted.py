"""pan/baseline_scripted.py -- hand-written success baseline for the covered pan (user, 10-29).  SIMULATION ONLY, CPU only.

The pipeline we must beat on task success: no certificate, no backups, one fixed script with sensing-based choices.
  1. observe_pose (depth fit, zed noise), re-aim both claws at the estimate (same Episode as pan/segments.py, clamp v2 geometry)
  2. pinch the handle; read yaw from the arm (left-claw FK).  NOTE: with the top/bottom pinch the claw keeps the aimed yaw, so this
     reading equals the camera estimate (MEASURED in the 10-29 smoke runs); it is recorded, not used to re-aim.
  3. tilt probe for mass: raise the handle 15 mm while the far rim stays on the stove, read the moment about the far-rim pivot line
     from the left arm's joint torques, lower again.  Contents mass from statics with the contents assumed at the pan centre (ASSUMED).
  4. push with torque sensing: stop and give up if the claw feels no push force (claw sliding on the handle) or a stall force.
  5. clamp, then lift with the controller's payload model from a small table (ASSUMED): contents > 0.45 kg -> do not lift (refuse);
     otherwise payload = pan + lid + estimated contents.

    python -m world_model.v2.nero.pan.baseline_scripted --out DIR --set full|overload --procs P
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

PROBE_DZ = 0.015          # design: handle raise for the tilt probe [m]
PUSH_MIN_N, PUSH_STALL_N = 0.2, 15.0   # ASSUMED: push-force window along -u felt at the left claw
REFUSE_KG = 0.45          # ASSUMED table entry: estimated contents above this -> do not lift


def _claw_wrench6(ep, s):
    import mujoco
    m, d = ep.m, ep.d
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jp, jr, ep.gb[s])
    J = np.vstack([jp, jr])[:, ep.dofs[s]]
    return np.linalg.lstsq(J.T, ep.contact_torque()[ep.dofs[s]], rcond=None)[0]   # contact torques only (see segments.claw_wrench)


def probe_moment(ep, centre_aim, u_aim):
    """moment [Nm] the pan exerts on the claw about the far-rim pivot line (direction w = z x u), from joint torques."""
    w6 = _claw_wrench6(ep, "left")
    F, M = w6[:3], w6[3:]
    piv = centre_aim - ep.r_out * u_aim
    wax = np.cross([0, 0, 1.0], u_aim)
    r = ep.tcp("left") - piv
    r[2] = ep.tcp("left")[2] - ep.p.pan_base_z()
    return float((np.cross(r, F) + M) @ wax)


def run(world, tag, plan=None):
    from . import segments as G
    from . import observe_pose as O
    plan = plan or dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
    ep = G.Episode(world, observe="zed", obs_seed=G.world_seed(tag), **plan)
    out = dict(tag=tag, world=world, plan=plan, obs=ep.obs, refused=ep.refused, stage=None)
    if ep.refused:
        out["stage"] = "refused at start"
        return out
    ep.reset_track()
    ep.run_phases(ep.segment_phases("start"))
    ep.run_phases(ep.segment_phases("pinch"))
    out["yaw_arm"] = ep.observe_boundary("pinch")
    est = ep.obs["est"]
    aim, Rz = O.aim_transform(est, ep.p.pan_centre_xy)
    centre_aim = aim(np.array([ep.p.pan_centre_xy[0], ep.p.pan_centre_xy[1], 0.0]))
    # 3. tilt probe
    A0 = []
    ep.run_phases([("hold", 0.3)], branch=lambda e, k, i, n: (A0.append(probe_moment(e, centre_aim, ep.u_aim)) or None)
                  if i >= n - 20 else None)
    A = []
    ep.run_phases([("move", {"left": np.array([0, 0, PROBE_DZ])}, 0.6), ("hold", 0.6)],
                  branch=lambda e, k, i, n: (A.append(probe_moment(e, centre_aim, ep.u_aim)) or None) if (k == "hold" and i >= n - 60) else None)
    A_probe = float(np.mean(A)) - float(np.mean(A0))
    W0 = 9.81 * float(ep.m.body_mass[ep.pan] + ep.m.body_mass[ep.lid])
    r = ep.r_out
    m_hat = (abs(A_probe) - W0 * r) / (9.81 * r)              # contents at the centre (ASSUMED)
    out["probe"] = dict(A_Nm=round(A_probe, 4), contents_kg_est=round(m_hat, 3),
                        contents_kg_true=None if ep.hid is None else round(float(ep.m.body_mass[ep.hid]), 3),
                        tilt_deg=round(ep.pan_state()["tilt"], 2))
    ep.run_phases([("move", {"left": np.array([0, 0, -PROBE_DZ])}, 0.6), ("hold", 0.3)])
    # 4. push with torque sensing
    def push_mon(e, st={}):
        st.setdefault("k0", e.k)
        if (e.k - st["k0"]) * G.DT < 0.5:
            return None
        f = float(-_claw_wrench6(e, "left")[:3] @ (-ep.u_aim))      # force from the pan resisting the push
        st.setdefault("f", []).append(f)
        if abs(f) > PUSH_STALL_N:
            return f"push stall {f:.1f} N"
        if len(st["f"]) >= 4 and max(abs(x) for x in st["f"][-4:]) < PUSH_MIN_N:
            return "no push force (claw sliding on the handle)"
        return None
    ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
    status, info = ep.run_phases(ep.segment_phases("push"), monitor=push_mon)
    out["push"] = dict(status=status, why=info.get("why"), success=ep.success("push") if status == "done" else False)
    if status != "done":
        out["stage"] = "push aborted"
        return _finish(ep, out)
    ep.run_phases(ep.segment_phases("clamp"))
    out["clamp"] = dict(success=ep.success("clamp"), gap_R_mm=round(ep.gap("right") * 1e3, 1))
    if max(m_hat, 0.0) > REFUSE_KG:
        out["stage"] = "refused lift (estimated overload)"
        return _finish(ep, out)
    ep.ctl.mass = W0 / 9.81 + max(m_hat, 0.0)
    out["payload_model_kg"] = round(ep.ctl.mass, 3)
    ep.run_phases(ep.segment_phases("lift"))
    out["lift"] = dict(success=ep.success("lift"))
    out["stage"] = "lifted"
    return _finish(ep, out)


def _finish(ep, out):
    out["U_entered"] = ep.track["viol"]
    out["task_success"] = bool(out.get("lift", {}).get("success"))
    return out


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, w = job
    t0 = time.time()
    try:
        r = run(w, tag)
    except Exception:
        r = dict(tag=tag, world=w, error=traceback.format_exc()[-1200:])
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main(argv=None):
    from .segcert import worlds
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", required=True, choices=["full", "overload", "iid"])
    ap.add_argument("--procs", type=int, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [(t, w) for t, w in worlds(a.set) if t not in done]
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            print(r["tag"], r.get("stage"), r.get("task_success"), (r.get("probe") or {}).get("contents_kg_est"),
                  (r.get("probe") or {}).get("contents_kg_true"), r.get("U_entered") and r["U_entered"]["kind"], r.get("error", "")[-200:],
                  flush=True)
    (out / f"_DONE_{a.set}").write_text("ok")


if __name__ == "__main__":
    main()
