"""verify_fd.py -- independent check of the inverse-dynamics torque (robot_authority.Ctx.solve).
Builds a SECOND model: the YOR MJCF + a free tray body (0.4 kg lumped payload, COM/inertia as in robot_authority) welded to
both gripper_base bodies at the IK grasp pose.  We apply the torque tau that Ctx.solve returned for a target tray accel as
qfrc_applied and run FORWARD dynamics (mj_forward).  If the inverse-dynamics + wrench-split math is right, the tray body's
linear acceleration equals the target.  (Weld forces take care of the internal-force split; contacts/limits are off.)
  python -m world_model.v2.tray_rep.authority.verify_fd
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from world_model.v2.tray_rep.authority import robot_authority as RA  # noqa: E402


def build_welded(cx):
    spec = mujoco.MjSpec.from_file(str(RA.OUT / "tmp/yor_copy/robot.xml"))   # byte-identical copy (.xml ext) + meshes symlink
    for b in spec.bodies:
        b.gravcomp = 0.0
    tray = spec.worldbody.add_body(name="tray_pl", pos=(cx.o + RA.C_TRAY).tolist())
    tray.add_freejoint()
    tray.mass = RA.MASS
    tray.ipos = [0, 0, 0]
    ev, evec = np.linalg.eigh(RA.I_COM)                    # tray level: principal frame
    tray.inertia = ev.tolist()
    q = np.zeros(4); mujoco.mju_mat2Quat(q, evec.flatten()); tray.iquat = q.tolist()
    for arm, nm in (("L", "left_arm_gripper_base"), ("R", "right_arm_gripper_base")):
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2 = nm, "tray_pl"
        eq.solref = [0.002, 1.0]
        eq.data = np.zeros(11)                             # anchor + relpose filled below
    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS | mujoco.mjtDisableBit.mjDSBL_LIMIT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    m.eq_solimp[:] = [0.9999, 0.9999, 0.001, 0.5, 2.0]      # near-rigid welds (default impedance 0.9 leaves ~10% slack)
    m.opt.iterations, m.opt.tolerance = 500, 1e-12
    d = mujoco.MjData(m)
    d.qpos[:12] = cx.q
    d.qpos[12:15] = cx.o + RA.C_TRAY
    d.qpos[15:19] = [1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    # weld relpose: pose of body2 (tray) in body1 (gripper) frame; anchor at body2 origin => data[0:3]=0
    for k, arm in enumerate("LR"):
        b1 = m.body(("left" if arm == "L" else "right") + "_arm_gripper_base").id
        b2 = m.body("tray_pl").id
        R1, p1 = d.xmat[b1].reshape(3, 3), d.xpos[b1]
        R2, p2 = d.xmat[b2].reshape(3, 3), d.xpos[b2]
        rel_p = R1.T @ (p2 - p1)
        qr = np.zeros(4); mujoco.mju_mat2Quat(qr, (R1.T @ R2).flatten())
        m.eq_data[k, :3] = 0.0
        m.eq_data[k, 3:6] = rel_p
        m.eq_data[k, 6:10] = qr
        m.eq_data[k, 10] = 1.0
    return m, d


def main():
    res = json.loads((RA.OUT / "robot_authority.json").read_text())
    rig = RA.Rig()
    rows = []
    for key in ("side/mid_mid", "front/mid_mid", "front/far_low", "side/far_mid"):
        v = res["poses"][key]
        cx = RA.Ctx(rig, np.array(v["q"]), v["tray_o"], key.split("/")[0])
        m, d = build_welded(cx)
        for name, a, al in (("static", [0, 0, 0], [0, 0, 0]), ("a_x=3", [3, 0, 0], [0, 0, 0]), ("a_y=-2,a_z=1", [0, -2, 1], [0, 0, 0]),
                            ("alpha_y=8", [0, 0, 0], [0, 8, 0])):
            a, al = np.array(a, float), np.array(al, float)
            for mode in ("ls", "lp"):
                s, tau = cx.solve(RA.CAPS["real8"], a=a, al=al, mode=mode)
                d.qpos[:12] = cx.q; d.qpos[12:15] = cx.o + RA.C_TRAY; d.qpos[15:19] = [1, 0, 0, 0]
                d.qvel[:] = 0
                d.qfrc_applied[:] = 0; d.qfrc_applied[:12] = tau
                mujoco.mj_forward(m, d)
                acc_tray = d.qacc[12:15].copy()                       # COM linear accel (free joint, world frame)
                al_tray = d.qacc[15:18].copy()
                # target COM accel = a_origin + al x r_c
                tgt = a + np.cross(al, RA.C_TRAY)
                rows.append((key, name, mode, float(np.linalg.norm(acc_tray - tgt)), float(np.linalg.norm(al_tray - al))))
                print(f"{key:14s} {name:13s} {mode}: |dacc|={rows[-1][3]:.3f} m/s^2  |dalpha|={rows[-1][4]:.3f} rad/s^2  (tau max {np.abs(tau).max():.2f})", flush=True)
    err = max(r[3] for r in rows); erra = max(r[4] for r in rows)
    print(f"MAX linear-accel error {err:.3f} m/s^2, MAX angular-accel error {erra:.3f} rad/s^2")
    (RA.OUT / "verify_fd.json").write_text(json.dumps(dict(rows=rows, max_lin=err, max_ang=erra)))


if __name__ == "__main__":
    main()
