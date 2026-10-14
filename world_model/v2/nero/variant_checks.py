"""variant_checks.py -- YOR-NERO variant checks: compile (+flags), zero self-collision, reach of the tray hold pose, tray-pose
reachability grid, proxy conservativeness, lift flag.   python -m world_model.v2.nero.variant_checks"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import mujoco

from . import config as C
from . import adapters as A
from . import ik as IK
from . import proxy as P


def compile_checks():
    out = {}
    for name, kw in dict(default={}, lift=dict(lift=True), gravcomp=dict(gravcomp=True)).items():
        t = time.time()
        m = A.compose_scene(**kw)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        out[name] = dict(compile_s=round(time.time() - t, 2), nbody=m.nbody, njnt=m.njnt, nq=m.nq, nv=m.nv, nu=m.nu, neq=m.neq,
                         ngeom=m.ngeom, ncon_at_zero_pose=int(d.ncon))
    m0 = mujoco.MjModel.from_xml_path(str(C.SCENE_OUT))      # the on-disk generated scene file compiles stand-alone
    out["scene_file_standalone"] = dict(path=str(C.SCENE_OUT), nbody=m0.nbody, nv=m0.nv)
    return out


def self_collision_scan(m, hold_q, n=800, seed=0):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    qa_l, _ = A.arm_addrs(m, "left"); qa_r, _ = A.arm_addrs(m, "right")
    lo, hi = C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1]
    res = {}
    for label, sampler in (("uniform_in_limits", lambda: (rng.uniform(lo, hi), rng.uniform(lo, hi))),
                           ("hold_pose_plus_N(0,0.4)", lambda: (np.clip(np.array(hold_q["left"]) + rng.normal(0, 0.4, 7), lo, hi),
                                                                np.clip(np.array(hold_q["right"]) + rng.normal(0, 0.4, 7), lo, hi)))):
        same = armarm = 0
        for _ in range(n):
            ql, qr = sampler()
            d.qpos[qa_l], d.qpos[qa_r] = ql, qr
            mujoco.mj_forward(m, d)
            s = a = False
            for j in range(d.ncon):
                c = d.contact[j]
                n1, n2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                if n1[:4] == n2[:4] and n1.startswith(("left", "right")):
                    s = True
                elif n1.startswith(("left", "right")) and n2.startswith(("left", "right")):
                    a = True
            same += s; armarm += a
        res[label] = dict(n=n, frac_with_same_arm_contact=same / n, frac_with_arm_arm_contact=armarm / n)
    return res


def proxy_conservative(m, hold_q, n=300, seed=1):
    """proxy gap (min over sphere pairs of dist - r_e - r_i, no D_MIN) vs true hull distance: proxy must be <= true (conservative)."""
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    pb = A.PairBarriers(m, d_min=0.0, mask_static=False)
    qa_l, _ = A.arm_addrs(m, "left"); qa_r, _ = A.arm_addrs(m, "right")
    lo, hi = C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1]
    gl, gr = A.arm_geoms(m, "left"), A.arm_geoms(m, "right")
    viol = []; false_safe = 0; n_valid = 0; n_close = 0; ratios = []
    for i in range(n):
        s = 0.9 if i % 2 else 0.35
        d.qpos[qa_l] = np.clip(np.array(hold_q["left"]) + rng.normal(0, s, 7), lo, hi)
        d.qpos[qa_r] = np.clip(np.array(hold_q["right"]) + rng.normal(0, s, 7), lo, hi)
        mujoco.mj_forward(m, d)
        pg = pb.min_h(d)
        tg = A.true_min_dist(m, d, gl, gr, dmax=0.6)
        if tg >= 0.6 and pg > 0.5:
            continue
        n_valid += 1
        viol.append(pg - tg)
        if pg > 0 and tg < 0:
            false_safe += 1
        if tg < 0.1:
            n_close += 1
            ratios.append(pg - tg)
    viol = np.array(viol)
    return dict(n_samples=n_valid, n_within_10cm=n_close, max_proxy_minus_true_m=float(viol.max()), mean_proxy_minus_true_m=float(viol.mean()),
                false_safe_count=false_safe, note="proxy_gap - true_gap <= 0 means conservative (proxy reports less clearance)")


def tray_reach_grid(m, hold_q, xs=np.arange(0.30, 0.651, 0.05), zs=np.arange(0.65, 1.051, 0.05), y=-0.21):
    """can both arms hold the tray (same short-rim pinch, same orientation) at tray centre (x, y, z)?  IK warm-started from the hold q."""
    ikL, ikR = IK.ArmIK(m, "left"), IK.ArmIK(m, "right")
    d = mujoco.MjData(m)
    pb = A.PairBarriers(m)
    grid = []
    for z in zs:
        row = []
        for x in xs:
            c, hl, hr, al, ar, jw = IK.tray_hold_targets((x, y, z))
            best = None
            for qpL, qpR in ((np.array(hold_q["left"]), np.array(hold_q["right"])), (None, None)):
                ql, el = ikL.solve(hl, al, jw, np.array(hold_q["left"]), qpL, iters=300)
                qr, er = ikR.solve(hr, ar, jw, np.array(hold_q["right"]), qpR, iters=300)
                if max(el[0], er[0]) < 2e-4 and max(el[1], er[1]) < 2e-3 and max(el[2], er[2]) < 3e-2:
                    best = (ql, qr)
                    break
            ok = best is not None
            if ok:
                d.qpos[ikL.qa], d.qpos[ikR.qa] = best
                mujoco.mj_forward(m, d)
                ok = d.ncon == 0 and pb.min_h(d) > 0.0
            row.append(int(ok))
        grid.append(row)
    return dict(x_m=[round(float(v), 3) for v in xs], z_m=[round(float(v), 3) for v in zs], y_m=y, reachable_rows_by_z=grid,
                note="1 = both arms reach the short-rim pinch with zero contacts and proxy h>0; IK warm-started from the hold pose then posture=hold q / mid-range")


def lift_check(hold_q, seconds=2.0):
    """lift=True: Slider_1/2 (0..0.208 m each, +-56 N).  Servos on, arms at hold q, lift commanded to 0.30 (of 0.416).  Does the lift actuator hold the arm weight?"""
    out = {}
    for gc in (False, True):
        m = A.compose_scene(with_tray=False, lift=True, gravcomp=gc)
        d = mujoco.MjData(m)
        for side in C.SIDES:
            q = hold_q[side]
            qa, _ = A.arm_addrs(m, side)
            d.qpos[qa] = q
            for i in range(7):
                d.ctrl[m.actuator(f"{side}_arm_joint{i+1}_pos").id] = q[i]
        d.ctrl[m.actuator("Lift").id] = 0.30
        s1, s2 = m.joint("Slider_1").id, m.joint("Slider_2").id
        d.qpos[m.jnt_qposadr[s1]] = d.qpos[m.jnt_qposadr[s2]] = 0.15
        mujoco.mj_forward(m, d)
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
        out["arm_gravcomp" if gc else "no_arm_gravcomp"] = dict(
            slider1_m=float(d.qpos[m.jnt_qposadr[s1]]), slider2_m=float(d.qpos[m.jnt_qposadr[s2]]),
            lift_actuator_force_N=float(d.actuator_force[m.actuator("Lift").id]), ctrl=0.30,
            total_mass_kg_above_lift=float(m.body_subtreemass[m.body("Lift_Top").id]))
    return out


def main():
    t0 = time.time()
    res = {"compile": compile_checks()}
    m = A.compose_scene()
    d = mujoco.MjData(m)
    info = IK.set_hold_state(m, d)
    hq = info["hold_q"]
    res["hold_pose"] = dict(info={k: v for k, v in info.items() if k != "hold_q"}, hold_q=hq, ncon_with_tray_welded=int(d.ncon),
                            joint_limit_margin_per_joint_left=[float(v) for v in np.minimum(np.array(hq["left"]) - C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1] - np.array(hq["left"]))],
                            joint_limit_margin_per_joint_right=[float(v) for v in np.minimum(np.array(hq["right"]) - C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1] - np.array(hq["right"]))])
    res["self_collision"] = self_collision_scan(A.compose_scene(with_tray=False), hq)
    res["proxy_conservative"] = proxy_conservative(A.compose_scene(with_tray=False), hq)
    res["tray_reach_grid"] = tray_reach_grid(m, hq)
    res["lift"] = lift_check(hq)
    res["wall_s"] = time.time() - t0
    out = Path(__file__).parent / "results" / "variant_checks.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("hold_pose",)}, indent=1))


if __name__ == "__main__":
    main()
