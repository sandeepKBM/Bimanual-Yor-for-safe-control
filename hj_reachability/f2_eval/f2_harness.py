#!/usr/bin/env python
"""F2 smoke harness: deploy the TRAINED DeepReach arm-arm value net on the
cup-carry CROSSING task, same harness style as F1
(/common/home/ss5772/.claude/jobs/f1580deb/tmp/hj_smoke.py), SIMULATION ONLY.

RIGHT arm = ego (filtered joint-velocity command -> inner velocity servo
tau = M*KV*(v_cmd - qd) + bias); LEFT arm = neighbor holding the lifted cup
(computed-torque PD, unfiltered). Cube welds omitted (as in F1).

Modes
-----
nominal          : no filter (v_cmd = v_nom) -- for validation data / reference.
analytic_hard    : F1 analytic fail-safe (bang-bang switch at h < margin).
analytic_blend   : F1 analytic eps-deadzone blend over h in [0, margin].
learned_hard     : saturated fail-safe u* = -ubar * dV/dq_r / |dV/dq_r| from the
                   TRAINED V, switched on when V > V_ENG (NeHMO/HaMMaR form).
learned_blend    : same u*, eps-deadzone blend weight ramping 0 -> 1 as V goes
                   V_ENG -> V_FULL.
learned_qp       : least-restrictive "value as constraint" filter
                   min ||u - u_nom||^2  s.t.  dV/dt <= alpha * (-V - eps)
                   (i.e. b = -V - eps >= 0 is a CBF, bdot >= -alpha b), with
                   dV/dt = gV_r.u + gV_l.qd_l (the neighbor's MEASURED joint
                   velocity -- centralized cooperative setting) and a per-joint
                   box |u_i| <= U_BOX. Solved EXACTLY (single half-space + box,
                   separable -> 1-D dual bisection).
learned_qp_logit : same QP on the raw pre-tanh logit (less saturated, more
                   distance-like) instead of tanh(logit).
analytic_qp      : same QP form on the analytic clearance, b = h - eps_h
                   (control-form ablation: trained value vs analytic value).
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

REPO = Path("/common/users/ss5772/bimanual")
F2 = REPO / "hj_reachability" / "f2_eval"
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "eval"),
           str(REPO / "experiments" / "ab_smooth_vs_bangbang"),
           str(REPO / "hj_reachability"), str(F2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc                                   # noqa: E402
import make_cup_collect_safety_demo as demo               # noqa: E402
import comprehensive_eval as ce                           # noqa: E402  (EXACT jerk metric)
import scene_setup                                         # noqa: E402
import run_ab4                                             # noqa: E402
from scripts import cup_collect_task_utils as cct         # noqa: E402
from control import realgrasp_flmpc as rg                  # noqa: E402
from analytic_safety_filter import (                       # noqa: E402
    AnalyticHJSafetyFilter, velocity_servo_torque)

DT = sc.DT
KV = 50.0
KP_POS = 12.0
U_BAR = 1.5
MARGIN = 0.05
# learned-value engagement levels (tanh V in [-1,1], V<=0 safe); see README/report
V_ENG = -0.5
V_FULL = 0.0
# QP-filter params
ALPHA = 5.0        # [1/s] class-K gain on b
EPS_V = 0.5        # tanh-V safe set  V <= -EPS_V
EPS_LOGIT = 0.6    # logit safe set   logit <= -EPS_LOGIT  (atanh(0.5)=0.55)
EPS_H = 0.03       # analytic safe set h >= EPS_H [m]
U_BOX = 3.0        # [rad/s] per-joint velocity box for the QP modes

OUT_DIR = Path("/common/home/ss5772/.claude/jobs/f1580deb/tmp/f2_eval")

_LEARNED = None


def learned():
    global _LEARNED
    if _LEARNED is None:
        from learned_value import LearnedArmArmValue
        _LEARNED = LearnedArmArmValue()
    return _LEARNED


def qp_halfspace_box(u_nom, a, c, box):
    """argmin ||u-u_nom||^2 s.t. a.u <= c, |u_i| <= box. Exact (KKT):
    u(lam) = clip(u_nom - lam*a, -box, box), a.u(lam) nonincreasing in lam."""
    u0 = np.clip(u_nom, -box, box)
    if a @ u0 <= c:
        return u0, 0.0, True
    na = float(a @ a)
    if na < 1e-18:
        return u0, 0.0, False
    lo, hi = 0.0, 1.0
    while a @ np.clip(u_nom - hi * a, -box, box) > c and hi < 1e12:
        hi *= 2.0
    feas = bool(a @ np.clip(u_nom - hi * a, -box, box) <= c + 1e-9)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if a @ np.clip(u_nom - mid * a, -box, box) > c:
            lo = mid
        else:
            hi = mid
    return np.clip(u_nom - hi * a, -box, box), hi, feas


def run_episode(seed, mode, max_steps=None, cup_jitter=0.04, record_states=False,
                v_eng=V_ENG, v_full=V_FULL, alpha=ALPHA, eps_v=EPS_V, eps_h=EPS_H):
    demo.N_CUBES = 3
    demo.V_EE_MAX_MS = 0.28
    sc.D_MIN = 0.03
    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                              cup_jitter_m=cup_jitter, n_cubes=3)
    model, data = info["model"], info["data"]
    left_dofadr, right_dofadr = info["left_dofadr"], info["right_dofadr"]
    left_qposadr, right_qposadr = info["left_qposadr"], info["right_qposadr"]
    left_ee_body, right_ee_body = info["left_ee_body"], info["right_ee_body"]
    cup_body = info["cup_body"]
    _cup_dofadr = int(model.jnt_dofadr[model.body_jntadr[cup_body]])
    track_l, track_r = info["track_l"], info["track_r"]
    q_home = info["q_home"]

    # gripper ("left EE") tilt reference axis: EE-local axis that is world-up
    # at the settled q_lift pose -- SAME definition as demo's left_cup_tilt_rad
    _scr = mujoco.MjData(model)
    _scr.qpos[:] = data.qpos
    _scr.qpos[left_qposadr] = info["q_lift"]
    mujoco.mj_forward(model, _scr)
    ee_axis_local = _scr.xmat[left_ee_body].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])

    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)
    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    from teleop import bimanual_teleop as bt
    right_finger_act = bt.finger_actuators(model, "right")

    tau_filter_l = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    tau_filter_r = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    left_geoms, right_geoms = run_ab4.left_right_geom_sets(model)

    ahj = None
    if mode in ("analytic_hard", "analytic_blend"):
        ahj = AnalyticHJSafetyFilter(right_dofadr, u_bar=U_BAR, margin=MARGIN,
                                     mode=mode.split("_")[1])
    use_learned = mode.startswith("learned") or record_states
    L = learned() if use_learned else None

    _ball_rel = scene_setup.canonical_ball_rel(model, cup_body)
    _prev_qpos = None
    n_steps = info["n_steps_total"] if max_steps is None else int(min(info["n_steps_total"], max_steps))
    RIGHT_START_T = demo.RIGHT_START_T

    keys = ["t", "h", "collision", "w", "V", "logit", "right_ee_pos", "left_ee_pos",
            "grip_tilt", "cup_tilt", "spilled", "v_cmd", "qp_feas"]
    if record_states:
        keys += ["q_left", "q_right", "qd_left", "gV", "glogit", "gh"]
    log = {k: [] for k in keys}
    t_filter = 0.0

    for k in range(n_steps):
        t = k * DT
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        bt.set_gripper(data, right_finger_act, 1.0)
        mujoco.mj_forward(model, data)
        scene_setup.reattach_cup_rigidly(model, data, left_ee_body, cup_body,
                                         info["cup_rel_pos0"], info["cup_rel_quat0"])
        if t < RIGHT_START_T:
            scene_setup.carry_balls_with_cup(model, data, cup_body, _ball_rel)
        if _prev_qpos is not None:
            _qvel_fd = scene_setup.cup_velocity_from_motion(model, _prev_qpos, data.qpos, DT)
            data.qvel[_cup_dofadr:_cup_dofadr + 6] = _qvel_fd[_cup_dofadr:_cup_dofadr + 6]
        _prev_qpos = data.qpos.copy()

        q_left_des = track_l.sample(t)
        qd_left_des = track_l.sample_dot(t)
        t_r = t - RIGHT_START_T
        if t_r >= 0:
            q_right_des = track_r.sample(t_r)
            qd_right_des = track_r.sample_dot(t_r)
        else:
            q_right_des, qd_right_des = q_home, np.zeros(6)

        M_full = sc.full_mass_matrix(model, data)
        M_ll = M_full[np.ix_(left_dofadr, left_dofadr)]
        bias_l = data.qfrc_bias[left_dofadr]
        M_box, bias_box = sc.payload_reflected_terms(model, data, left_ee_body, cup_body, left_dofadr)
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
        q_l = data.qpos[left_qposadr].copy()
        qd_l = data.qvel[left_dofadr].copy()
        qacc_des_l = sc.KP_TASK * (q_left_des - q_l) + sc.KD_TASK * (qd_left_des - qd_l)
        tau_l = M_ll @ qacc_des_l + bias_l

        M_rr = M_full[np.ix_(right_dofadr, right_dofadr)]
        bias_r = data.qfrc_bias[right_dofadr]
        q_r = data.qpos[right_qposadr].copy()
        qd_r = data.qvel[right_dofadr].copy()
        v_nom = qd_right_des + KP_POS * (q_right_des - q_r)

        # analytic clearance + gradient (ground truth / analytic modes)
        h, dist, eb, ib, n, eoff, ioff = sc.min_pair_distance(model, data)
        Jl = sc.point_jac_at(model, data, eb, eoff)
        Ji = sc.point_jac_at(model, data, ib, ioff)
        gh = np.concatenate([n @ Jl[:, left_dofadr], -(n @ Ji[:, right_dofadr])])  # dh/d[q_l,q_r]

        t0 = time.perf_counter()
        V = lg = np.nan
        gV = glg = np.zeros(12)
        if L is not None:
            V, lg, gV, glg = L.value_and_grad(q_l, q_r)
        w = 0.0
        feas = 1.0
        v_cmd = v_nom
        if mode == "nominal":
            pass
        elif ahj is not None:
            fi = ahj.filter_velocity(model, data, v_nom)
            v_cmd, w = fi.v_cmd, fi.blend
        elif mode in ("learned_hard", "learned_blend"):
            g_r = gV[6:]
            gn = float(np.linalg.norm(g_r))
            u_star = -U_BAR * g_r / max(gn, 1e-12)
            if mode == "learned_hard":
                w = 1.0 if V > v_eng else 0.0
            else:
                w = float(np.clip((V - v_eng) / (v_full - v_eng), 0.0, 1.0))
            if gn < 1e-9:
                w = 0.0
            v_cmd = (1.0 - w) * v_nom + w * u_star
        elif mode in ("learned_qp", "learned_qp_logit", "analytic_qp"):
            if mode == "learned_qp":
                b, gb = -V - eps_v, -gV            # b>=0 safe
            elif mode == "learned_qp_logit":
                b, gb = -lg - EPS_LOGIT, -glg
            else:
                b, gb = h - eps_h, gh
            # bdot = gb_l.qd_l + gb_r.u >= -alpha b  <=>  (-gb_r).u <= alpha b + gb_l.qd_l
            a = -gb[6:]
            c = alpha * b + float(gb[:6] @ qd_l)
            v_cmd, lam, ok = qp_halfspace_box(v_nom, a, c, U_BOX)
            w = float(lam > 0)
            feas = float(ok)
        else:
            raise ValueError(mode)
        t_filter += time.perf_counter() - t0

        tau_r = velocity_servo_torque(M_rr, bias_r, qd_r, v_cmd, kv=KV)
        tau_l_cmd = np.clip(tau_filter_l.apply(tau_l, DT), -sc.TAU_MAX, sc.TAU_MAX)
        tau_r_cmd = np.clip(tau_filter_r.apply(tau_r, DT), -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[left_dofadr] = tau_l_cmd
        data.qfrc_applied[right_dofadr] = tau_r_cmd

        collision = False
        for i in range(data.ncon):
            con = data.contact[i]
            if (con.geom1 in left_geoms and con.geom2 in right_geoms) or \
               (con.geom2 in left_geoms and con.geom1 in right_geoms):
                collision = True
                break

        log["t"].append(t)
        log["h"].append(h)
        log["collision"].append(float(collision))
        log["w"].append(w)
        log["V"].append(V)
        log["logit"].append(lg)
        log["right_ee_pos"].append(data.xpos[right_ee_body].copy())
        log["left_ee_pos"].append(data.xpos[left_ee_body].copy())
        log["grip_tilt"].append(rg.ee_tilt_rad(data, left_ee_body, ee_axis_local))
        log["cup_tilt"].append(rg.cup_true_tilt_rad(data, cup_body))
        log["spilled"].append(int(cct.count_spilled(model, data)))
        log["v_cmd"].append(np.asarray(v_cmd, float).copy())
        log["qp_feas"].append(feas)
        if record_states:
            log["q_left"].append(q_l)
            log["q_right"].append(q_r)
            log["qd_left"].append(qd_l)
            log["gV"].append(gV)
            log["glogit"].append(glg)
            log["gh"].append(gh)

        mujoco.mj_step(model, data)

    out = {k: np.asarray(v) for k, v in log.items()}
    out["filter_ms"] = 1e3 * t_filter / max(n_steps, 1)
    out["idx_cross"] = int(np.searchsorted(out["t"], RIGHT_START_T))
    return out


def summarize(log):
    right_ee = log["right_ee_pos"]
    jerk = ce.jerk_rms(right_ee, DT)
    rspeed = np.linalg.norm(np.diff(right_ee, axis=0) / DT, axis=1)
    try:
        sparc = float(sc.sparc(rspeed, DT))
    except Exception:
        sparc = float("nan")
    w = log["w"]
    sw = int(np.sum(np.abs(np.diff((w > 0).astype(int)))))
    dv = np.linalg.norm(np.diff(log["v_cmd"], axis=0), axis=1)
    ic = log["idx_cross"]
    return dict(
        n_steps=int(len(log["t"])), jerk_rms=float(jerk), sparc=sparc,
        collision=bool(np.any(log["collision"] > 0.5)),
        coll_steps=int(np.sum(log["collision"] > 0.5)),
        h_min=float(np.min(log["h"])),
        grip_tilt_max_deg=float(np.degrees(np.max(log["grip_tilt"][ic:]))),
        cup_tilt_max_deg=float(np.degrees(np.max(log["cup_tilt"][ic:]))),
        cup_tilt0_deg=float(np.degrees(log["cup_tilt"][ic])),
        engaged_frac=float(np.mean(w > 0)), switches=sw,
        dvcmd_max=float(np.max(dv)), dvcmd_p99=float(np.percentile(dv, 99)),
        spilled_max=int(np.max(log["spilled"])),
        right_ee_speed_max=float(np.max(rspeed)),
        qp_infeas_frac=float(np.mean(log["qp_feas"] < 0.5)),
        filter_ms=float(log["filter_ms"]),
        V_max=float(np.nanmax(log["V"])) if np.any(np.isfinite(log["V"])) else float("nan"),
    )


def run_flmpc(seed, max_steps=None):
    """FL-MPC on the SAME seed via the demo driver (as comprehensive_eval.run_one,
    faked grasp); metrics recomputed with the same definitions."""
    demo.N_CUBES = 3
    demo.V_EE_MAX_MS = 0.28
    demo.FL_W_JERK_OVERRIDE = None
    demo.FL_W_POS_INT_OVERRIDE = None
    sc.D_MIN = 0.03
    sc.ALPHA1 = sc.ALPHA2 = 30.0
    sc.COMBINED_RADIUS = sc.R_EGO + sc.R_INT + sc.D_MIN
    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                              cup_jitter_m=0.04, n_cubes=3)
    log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                              ee_speed_limit=True, table_floor=True, max_steps=max_steps)
    m = demo.summarize(log)
    right_ee = np.asarray(log["right_ee_pos"])
    rspeed = np.linalg.norm(np.diff(right_ee, axis=0) / DT, axis=1)
    ic = int(np.searchsorted(log["t"], demo.RIGHT_START_T))
    return dict(n_steps=len(right_ee), jerk_rms=float(ce.jerk_rms(right_ee, DT)),
                sparc=float(sc.sparc(rspeed, DT)), collision=bool(m["collision"]),
                h_min=float(m["h_min"]),
                grip_tilt_max_deg=float(np.degrees(np.max(log["left_cup_tilt_rad"][ic:]))),
                cup_tilt_max_deg=float(np.degrees(np.max(log["true_cup_tilt_rad"][ic:]))),
                cup_tilt0_deg=float(np.degrees(log["true_cup_tilt_rad"][ic])),
                spilled_max=int(m["spilled_max"]), right_ee_speed_max=float(np.max(rspeed)),
                engaged_frac=float("nan"), switches=-1, dvcmd_max=float("nan"),
                dvcmd_p99=float("nan"), qp_infeas_frac=float(m["qp_infeasible_frac_right"]),
                filter_ms=float(m["solve_time_mean_ms"]), coll_steps=-1, V_max=float("nan"))


def _job(args):
    seed, mode, max_steps, kw = args
    t0 = time.perf_counter()
    try:
        if mode == "flmpc":
            s = run_flmpc(seed, max_steps)
        else:
            log = run_episode(seed, mode, max_steps=max_steps, record_states=kw.get("record", False),
                              **{k: v for k, v in kw.items() if k != "record"})
            s = summarize(log)
            if kw.get("record", False):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(OUT_DIR / f"states_{mode}_s{seed}.npz",
                                    **{k: log[k] for k in ("t", "h", "V", "logit", "q_left", "q_right",
                                                           "qd_left", "gV", "glogit", "gh",
                                                           "collision", "w")},
                                    idx_cross=log["idx_cross"])
        s.update(seed=seed, mode=mode, ok=True, wall_s=time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001
        import traceback
        s = dict(seed=seed, mode=mode, ok=False, err=str(e)[:300], tb=traceback.format_exc()[-1500:])
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,2,5,6,7")
    ap.add_argument("--modes", default="learned_hard,learned_blend,learned_qp")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--eps-v", type=float, default=EPS_V)
    ap.add_argument("--v-eng", type=float, default=V_ENG)
    ap.add_argument("--eps-h", type=float, default=EPS_H)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    kw = dict(record=args.record)
    if args.alpha != ALPHA:
        kw["alpha"] = args.alpha
    if args.eps_v != EPS_V:
        kw["eps_v"] = args.eps_v
    if args.eps_h != EPS_H:
        kw["eps_h"] = args.eps_h
    if args.v_eng != V_ENG:
        kw["v_eng"] = args.v_eng
    jobs = [(s, m, args.max_steps, kw) for m in modes for s in seeds]
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
    else:
        rows = [_job(j) for j in jobs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"results_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
