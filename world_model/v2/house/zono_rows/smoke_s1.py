"""smoke_s1.py -- S1 canonical: command the LEFT hand through a (yawed) box obstacle, with and without the zonotope rows.  SIMULATION ONLY, CPU only.

    PYTHONPATH=. python -m world_model.v2.house.zono_rows.smoke_s1 [--out DIR]

Scene: the office model without tray (arms free), FL-MPC (unmodified) holds the right arm at its hold pose and tracks a joint-space ramp for
the left arm that carries the hand ~0.3 m sideways THROUGH a solid MuJoCo box geom.  The same box as a zonotope (exact: a yawed box is a
zonotope) is given to ZonoCtl.  Measured with MuJoCo's own geometry: mj_geomDistance between every collidable arm geom and the box, and the
contacts with it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import mujoco

from world_model.v2.house import build as B
from world_model.v2.house import phase1 as P1
from world_model.v2.house.office import compose as CP, scene as S
from world_model.v2.house.zono_rows.ctl import ZonoCtl
from world_model.v2.house.zono_rows import rows as R

DT = 0.005
OUT = Path(os.environ.get("ZONO_ROWS_OUT", "data/molmospaces/zono_rows"))


def build(box_half, box_yaw, box_pos=(0.0, 0.0, -5.0), solid=True):
    spec = S.load_office_spec()
    W = spec.worldbody
    qz = [math.cos(box_yaw / 2), 0, 0, math.sin(box_yaw / 2)]
    W.add_geom(name="zobs_col", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(box_half), pos=list(box_pos), quat=qz, rgba=[0.9, 0.2, 0.2, 0.5],
               group=4, contype=8 if solid else 0, conaffinity=15 if solid else 0, friction=[0.9, 0.9, 0.001])   # (static geoms must be placed BEFORE compile: MuJoCo caches static BVH)
    wp = S.office_meta()["waypoints"]["lab_desk_place"]
    C = CP.new_hold_office(spawn_xy=wp["xy"], yaw=wp["heading"], dt=DT, lift_h=0.0, servos_off=True, head_cam=False, with_tray=False, spec=spec)
    return C


def _gd(m, d, a, b, dmax, ft):
    v = mujoco.mj_geomDistance(m, d, a, b, dmax, ft)
    if v < dmax and abs(np.linalg.norm(ft[3:] - ft[:3]) - abs(v)) > 1e-4 and v >= 0:
        return dmax                          # spurious zero for far mesh/box pairs (see phase1.robot_house_min_distance)
    return v


def run(C, q_hold_l, dq, T_ramp, T_total, rows, box_c, box_G, margin, alpha, d_act, max_rows):
    m, d = C.m, C.d
    ctl = ZonoCtl(C, zonos=[(box_c, box_G)] if rows else None, margin=margin, alpha1=alpha, alpha2=alpha, d_act=d_act, max_rows=max_rows, use_tray=False, tilt=False)
    cls = P1.geom_class(m)
    gbox = m.geom("zobs_col").id
    arm_g = [g for g in range(m.ngeom) if cls[g] in ("L", "R") and (m.geom_contype[g] or m.geom_conaffinity[g])]
    ft = np.zeros(6)
    n = int(T_total / DT)
    dmin, dmin_t, ncon = 9.0, 0.0, 0
    trace = []
    q0 = np.array(q_hold_l)
    for k in range(n):
        t = k * DT
        s = min(t / T_ramp, 1.0)
        ctl.qe_ref[:] = q0 + s * dq
        te, ti = ctl.tau(d)
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        ncon += sum(1 for c in range(d.ncon) if gbox in (d.contact[c].geom1, d.contact[c].geom2))
        if k % 4 == 0:
            dd = min(_gd(m, d, g, gbox, 0.5, ft) for g in arm_g)
            trace.append(float(dd))
            if dd < dmin:
                dmin, dmin_t = float(dd), t
    ee = d.xpos[m.body("left_arm_gripper_base").id].copy()
    return dict(min_true_dist_m=dmin, min_dist_t_s=dmin_t, contact_steps=int(ncon), final_dist_m=trace[-1], final_ee=ee.tolist(),
                final_q_err_to_goal=float(np.abs(d.qpos[ctl.lq] - (q0 + dq)).max()), ctl=ctl.summary(), trace=trace[::5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--d-act", type=float, default=0.15)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    half = (0.03, 0.08, 0.10)
    yaw = math.radians(35.0)
    C = build(half, yaw)
    m, d = C.m, C.d
    ql = np.array(C.info["hold_q"]["left"])
    lq = P1.Ctl.__new__(P1.Ctl)    # only to get addresses
    import sim_common as sc
    _, lqa, lda = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
    gb = m.body("left_arm_gripper_base").id
    # joint-space ramp: candidates = one joint moved by +-(0.4..1.0) rad; keep those that move the hand 0.2-0.45 m, then place the box ON the
    # hand path at the latest s in [0.5, 0.95] for which the INITIAL pose keeps >= 8 cm (keypoint spheres, exact zonotope distance) from it
    from world_model.v2.house.zono_rows import zono_hrep as ZH
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    G = Rz @ np.diag(half)
    ks = R.arm_keypoints(m, "left") + R.arm_keypoints(m, "right")
    p0 = np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ o[:, 0] for b, o in zip(ks.bodies, ks.offs)])
    jr = m.jnt_range[[m.joint(n).id for n in sc.LEFT_ARM_JOINTS]]
    ds = mujoco.MjData(m)
    c = dq = None
    for j in (0, 1, 2, 3):
        for sgn in (1, -1):
            for mag in (0.4, 0.6, 0.8, 1.0):
                dq_ = np.zeros(6); dq_[j] = sgn * mag
                if not (jr[j, 0] + 0.05 < ql[j] + dq_[j] < jr[j, 1] - 0.05):
                    continue
                ds.qpos[:] = d.qpos; ds.qpos[lqa] = ql + dq_; mujoco.mj_kinematics(m, ds)
                disp = ds.xpos[gb] - d.xpos[gb]
                if not 0.2 < np.linalg.norm(disp) < 0.45:
                    continue
                for s_ in np.arange(0.95, 0.49, -0.05):
                    ds.qpos[lqa] = ql + s_ * dq_; mujoco.mj_kinematics(m, ds)
                    cc = ds.xpos[gb].copy()
                    H = ZH.zono_hrep(cc, G)
                    clr = min(ZH.sd_exact(H, q) - r for q, r in zip(p0, ks.radii))
                    if clr > 0.08:
                        c, dq = cc, dq_
                        print("ramp joint", j, "dq", sgn * mag, "hand displacement", np.round(disp, 3), "box at s =", round(float(s_), 2),
                              "initial keypoint clearance", round(float(clr), 3), flush=True)
                        break
                if c is not None:
                    break
            if c is not None:
                break
        if c is not None:
            break
    assert c is not None, "no collision-free-start ramp found"
    C = build(half, yaw, box_pos=c)                   # rebuild with the box at its final place
    m, d = C.m, C.d
    mujoco.mj_forward(m, d)
    ft0 = np.zeros(6)
    gid = m.geom("zobs_col").id
    cls0 = P1.geom_class(m)
    d0 = min(mujoco.mj_geomDistance(m, d, g, gid, 1.0, ft0) for g in range(m.ngeom) if cls0[g] in ("L", "R") and (m.geom_contype[g] or m.geom_conaffinity[g]))
    print("initial arm-box distance", round(d0, 3), flush=True)
    assert d0 > 0.04, "box overlaps the initial pose; shrink/move it"
    snap = (d.qpos.copy(), d.qvel.copy(), d.mocap_pos.copy(), d.mocap_quat.copy())
    res = dict(setup=dict(box_centre=c.tolist(), box_half=list(half), box_yaw_deg=35.0, ramp_dq=dq.tolist(), margin=a.margin, alpha=a.alpha,
                          d_act=a.d_act, max_rows=a.max_rows, physics_dt=DT, ctl_DT="FLMPC DT == physics dt (one tau() per mj_step)"))
    Cp = build(half, yaw, box_pos=c, solid=False)          # phantom twin: same pose, no contact -> the unobstructed intrusion depth
    for name, rows, CC in (("without_rows_phantom_box", False, Cp), ("without_rows_solid_box", False, C), ("with_rows_solid_box", True, C)):
        m, d = CC.m, CC.d
        snap = (d.qpos.copy(), d.qvel.copy(), d.mocap_pos.copy(), d.mocap_quat.copy()) if name.startswith("without_rows_phantom") else snap
        d.qpos[:], d.qvel[:] = snap[0], snap[1]; d.mocap_pos[:], d.mocap_quat[:] = snap[2], snap[3]; d.qfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        t0 = time.time()
        r = run(CC, ql, dq, 1.5, a.T, rows, c, G, a.margin, a.alpha, a.d_act, a.max_rows)
        r["wall_s"] = time.time() - t0
        res[name] = r
        print(name, {k: v for k, v in r.items() if k not in ("trace",)}, flush=True)
    (out / f"s1_results{a.tag}.json").write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


if __name__ == "__main__":
    main()
