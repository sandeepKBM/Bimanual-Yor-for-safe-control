"""intrusion_test.py -- end-to-end check that the NERO proxy/pair-barrier adapter actually drives FL-MPC's arm-arm clearance rows.

No tray.  Both arms at the tray-hold pose; the right (intruder) arm's joint reference ramps over `t_ramp` s to an IK pose that
puts its ee_grasp point `overlap` m past the LEFT ee_grasp point (a commanded collision).  FL-MPC (NV=7, unmodified) must keep the
arms apart.  Reports min proxy h, min TRUE hull distance (mj_geomDistance over left x right collision geoms), contacts, infeasible QPs.
    python -m world_model.v2.nero.intrusion_test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

from . import config as C
from . import adapters as A
from . import gripper as G
from . import ik as IK
from . import hold_test as H


true_min_dist = A.true_min_dist


def run(overlap=0.03, t_ramp=1.5, t_total=3.0, tau_preset=None, **fl_kw):
    tau = C.tau_limit(tau_preset)       # F3 11-03: FL-MPC limit; MJCF forcerange keeps tau_limits()
    tau_mj = C.tau_limits(tau_preset)
    m = A.compose_scene(with_tray=False, tau=tau_mj)
    d = mujoco.MjData(m)
    # hold pose without tray: reuse IK pair (tray present only for the weld; here a tray-less model, so use a tray model for IK then copy q)
    mt = A.compose_scene(with_tray=True, tau=tau_mj)
    dt_ = mujoco.MjData(mt)
    info = IK.set_hold_state(mt, dt_)
    hq = info["hold_q"]
    qa_l, _ = A.arm_addrs(m, "left"); qa_r, _ = A.arm_addrs(m, "right")
    d.qpos[qa_l] = hq["left"]; d.qpos[qa_r] = hq["right"]
    for side in C.SIDES:
        G.set_opening(m, d, side, info["finger"])
    mujoco.mj_forward(m, d)
    A.disable_arm_servos(m)
    ikL, ikR = IK.ArmIK(m, "left"), IK.ArmIK(m, "right")
    pL, _ = ikL.pose(np.array(hq["left"]))
    target = pL + np.array([0.0, -overlap, 0.0])            # right hand arrives from -y; overlap m past the left hand point
    c, hl, hr, al, ar, jw = IK.tray_hold_targets()
    qgoal, e = ikR.solve(target, ar, jw, np.array(hq["right"]), iters=600)
    ctl = H.Ctl(m, hq, tau, **fl_kw)
    NP = ctl.fl.NP
    q0r = np.array(hq["right"])

    def qref(t):
        a = np.clip(t / t_ramp, 0.0, 1.0); a = a * a * (3 - 2 * a)
        return q0r + a * (qgoal - q0r)

    dt = m.opt.timestep
    gl = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("left_arm_") and m.geom_contype[g]]
    gr = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("right_arm_") and m.geom_contype[g]]
    min_h = 1e9; min_true = 1e9; n_con = 0; qdev_l = 0.0
    for k in range(int(t_total / dt)):
        t = k * dt
        ctl.qi_ref = np.array([qref(t + i * dt) for i in range(NP + 1)])
        te, ti = ctl.tau(d)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te; d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        min_h = min(min_h, ctl.pairs.min_h(d))
        n_con += int(sum(1 for j in range(d.ncon)
                         if {m.body(m.geom_bodyid[d.contact[j].geom1]).name[:4], m.body(m.geom_bodyid[d.contact[j].geom2]).name[:4]} == {"left", "righ"}))
        if k % 10 == 0:
            min_true = min(min_true, true_min_dist(m, d, gl, gr))
        qdev_l = max(qdev_l, float(np.abs(d.qpos[ctl.lq] - np.array(hq["left"])).max()))
    # what would the goal pose give without avoidance? (open-loop kinematic check)
    d2 = mujoco.MjData(m); d2.qpos[:] = d.qpos
    d2.qpos[qa_l] = hq["left"]; d2.qpos[qa_r] = qgoal; mujoco.mj_forward(m, d2)
    return dict(overlap_cmd_m=overlap, ik_goal_err=e, goal_pose_min_proxy_h_m=ctl.pairs.min_h(d2),
                goal_pose_true_min_dist_m=true_min_dist(m, d2, gl, gr), goal_pose_contacts=int(d2.ncon),
                closed_loop=dict(min_proxy_h_m=min_h, min_true_hull_dist_m=min_true, left_right_contact_steps=n_con,
                                 n_infeasible=ctl.n_infeasible, n_fallback=ctl.n_fallback, max_left_arm_deviation_rad=qdev_l))


if __name__ == "__main__":
    r = run()
    out = Path(__file__).parent / "results" / "intrusion.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
