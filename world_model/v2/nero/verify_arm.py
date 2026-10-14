"""verify_arm.py -- standalone NERO MJCF vs the official URDF: FK, mass/inertia, gravity torques.

    python -m world_model.v2.nero.verify_arm [--n 50] [--out world_model/v2/nero/results/arm_verification.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from . import config as C
from . import urdf_fk as U
from . import urdf_to_mjcf as X

URDF2MJ = {"base_link": "link0", "link1": "link1", "link2": "link2", "link3": "link3", "link4": "link4", "link5": "link5",
           "link6": "link6", "link7": "link7", "flange_link": "flange", "gripper_base": "gripper_base",
           "gripper_link1": "finger1", "gripper_link2": "finger2"}


def load(path=None, **kw):
    p = X.write_standalone(path or C.ARM_OUT, **kw)
    m = mujoco.MjModel.from_xml_path(str(p))
    return m, mujoco.MjData(m)


def _rot_err(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1, 1)))


def fk_check(m, d, n=50, seed=0):
    links, joints = U.parse()
    rng = np.random.default_rng(seed)
    lo, hi = C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1]
    pe, oe = 0.0, 0.0
    per_body = {k: [0.0, 0.0] for k in URDF2MJ}
    for s in range(n):
        q = rng.uniform(lo, hi) if s else np.zeros(7)
        g = rng.uniform(0.0, 0.05)
        qd = {f"joint{i+1}": q[i] for i in range(7)}
        qd.update(gripper_joint1=g, gripper_joint2=-g)
        poses = U.fk(joints, qd)
        d.qpos[:] = 0
        for i in range(7):
            d.qpos[m.jnt_qposadr[m.joint(f"joint{i+1}").id]] = q[i]
        d.qpos[m.jnt_qposadr[m.joint("gripper_joint1").id]] = g
        d.qpos[m.jnt_qposadr[m.joint("gripper_joint2").id]] = -g
        mujoco.mj_kinematics(m, d)
        for un, mn in URDF2MJ.items():
            T = poses[un]
            b = m.body(mn).id
            ep = float(np.linalg.norm(d.xpos[b] - T[:3, 3]))
            eo = _rot_err(d.xmat[b].reshape(3, 3), T[:3, :3])
            per_body[un][0] = max(per_body[un][0], ep)
            per_body[un][1] = max(per_body[un][1], eo)
            pe, oe = max(pe, ep), max(oe, eo)
    return dict(n_configs=n, max_pos_err_m=pe, max_rot_err_rad=oe,
                per_body_max={k: dict(pos_m=v[0], rot_rad=v[1]) for k, v in per_body.items()})


def inertia_check(m):
    """mass, COM (body frame), full inertia tensor about COM (body frame): MuJoCo vs URDF."""
    links, _ = U.parse()
    worst = dict(mass=0.0, com=0.0, inertia_rel=0.0)
    rows = {}
    for un, mn in URDF2MJ.items():
        L = links[un]
        b = m.body(mn).id
        # body-frame tensor from MuJoCo's principal diag + iquat
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_iquat[b])
        R = R.reshape(3, 3)
        I_mj = R @ np.diag(m.body_inertia[b]) @ R.T
        em = abs(float(m.body_mass[b]) - L.mass)
        ec = float(np.linalg.norm(m.body_ipos[b] - L.com))
        ei = float(np.abs(I_mj - L.inertia).max() / max(np.abs(L.inertia).max(), 1e-12))
        rows[un] = dict(mass_urdf=L.mass, mass_mjcf=float(m.body_mass[b]), com_err_m=ec, inertia_rel_err=ei)
        worst["mass"], worst["com"], worst["inertia_rel"] = max(worst["mass"], em), max(worst["com"], ec), max(worst["inertia_rel"], ei)
    tot_u = sum(links[k].mass for k in URDF2MJ)
    return dict(worst=worst, total_mass_urdf=tot_u, total_mass_mjcf=float(m.body_mass.sum()), per_body=rows)


def gravity_torques(m, d, payload_kg=0.0):
    """tau_g = qfrc_inverse at qvel=qacc=0 (gravity only; no gravcomp), for the 7 arm joints."""
    ad = [m.jnt_dofadr[m.joint(f"joint{i+1}").id] for i in range(7)]
    saved = m.body_mass.copy(), m.body_inertia.copy()
    if payload_kg:
        b = m.body("gripper_base").id
        m.body_mass[b] += payload_kg
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    tau = d.qfrc_inverse[ad].copy()
    m.body_mass[:], m.body_inertia[:] = saved
    return tau


def gravity_report(m, d, n_rand=3000, seed=1):
    lo, hi = C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1]
    rng = np.random.default_rng(seed)
    ad = [m.jnt_qposadr[m.joint(f"joint{i+1}").id] for i in range(7)]

    def at(q, payload=0.0):
        d.qpos[:] = 0
        d.qpos[ad] = q
        mujoco.mj_forward(m, d)
        return gravity_torques(m, d, payload)

    poses = {
        "zero (arm vertical)": np.zeros(7),
        "J2=+90deg, rest 0 (arm horizontal, fully extended: worst case)": np.array([0, np.pi / 2, 0, 0, 0, 0, 0]),
        "J2=+90deg, J4=+90deg (forearm up)": np.array([0, np.pi / 2, 0, np.pi / 2, 0, 0, 0]),
        "J2=+45deg, J4=+90deg": np.array([0, np.pi / 4, 0, np.pi / 2, 0, 0, 0]),
    }
    rep = {k: dict(tau=at(q).round(3).tolist(), tau_with_3kg_payload=at(q, 3.0).round(3).tolist()) for k, q in poses.items()}
    peaks, peaks3 = np.zeros(7), np.zeros(7)
    for _ in range(n_rand):
        q = rng.uniform(lo, hi)
        peaks = np.maximum(peaks, np.abs(at(q)))
        peaks3 = np.maximum(peaks3, np.abs(at(q, 3.0)))
    # exact worst case for the structure (arm horizontal) is within the sampled set's closure; also report the sampled max
    presets = {k: v.tolist() for k, v in C.TORQUE_PRESETS.items()}
    ratio = {k: (peaks / np.array(v)).round(3).tolist() for k, v in presets.items()}
    ratio3 = {k: (peaks3 / np.array(v)).round(3).tolist() for k, v in presets.items()}
    return dict(poses=rep, sampled_peak_abs_tau_g=peaks.round(3).tolist(), sampled_peak_with_3kg=peaks3.round(3).tolist(),
                n_random=n_rand, limits=presets, peak_over_limit=ratio, peak3kg_over_limit=ratio3)


def limits_table(links=None, joints=None):
    links, joints = U.parse()
    rows = []
    for i in range(7):
        j = joints[f"joint{i+1}"]
        lo, hi = C.URDF_LIMITS[i]
        rows.append(dict(joint=j.name, urdf_lo_rad=j.lower, urdf_hi_rad=j.upper,
                         urdf_lo_deg=round(np.degrees(j.lower), 2), urdf_hi_deg=round(np.degrees(j.upper), 2),
                         sdk_doc_lo_rad=C.SDK_DOC_LIMITS[i, 0], sdk_doc_hi_rad=C.SDK_DOC_LIMITS[i, 1],
                         urdf_effort=j.effort, urdf_velocity=j.velocity))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "arm_verification.json"))
    a = ap.parse_args(argv)
    m, d = load()
    # model joint ranges == URDF
    jr = np.array([m.jnt_range[m.joint(f"joint{i+1}").id] for i in range(7)])
    out = dict(source=dict(repo=C.URDF_REPO, commit=C.URDF_COMMIT),
               model=dict(nbody=m.nbody, nv=m.nv, ngeom=m.ngeom, jnt_range_max_dev_from_urdf=float(np.abs(jr - C.URDF_LIMITS).max())),
               fk=fk_check(m, d, a.n), inertia=inertia_check(m), gravity=gravity_report(m, d), limits=limits_table())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(fk={k: out["fk"][k] for k in ("n_configs", "max_pos_err_m", "max_rot_err_rad")},
                          inertia=out["inertia"]["worst"], mass=(out["inertia"]["total_mass_urdf"], out["inertia"]["total_mass_mjcf"]),
                          peak=out["gravity"]["sampled_peak_abs_tau_g"], peak3=out["gravity"]["sampled_peak_with_3kg"],
                          peak_over_limit=out["gravity"]["peak_over_limit"]), indent=1))


if __name__ == "__main__":
    main()
