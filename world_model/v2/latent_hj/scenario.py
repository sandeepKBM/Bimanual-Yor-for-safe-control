"""scenario.py -- Stage 2: aggressive cup CARRY where spill is actually possible.

Scene: the flagship cup-collect scene (make_cup_collect_safety_demo.build_episode,
"crossing"), but
  * LEFT (cup) arm starts at the settled q_lift pose and TRACKS an aggressive,
    randomized cup-pivot trajectory: 2-4 min-jerk back-and-forth strokes
    (8-16 cm, 0.28-0.5 s each, i.e. peak cup accel ~3-8 m/s^2, near the 4 Hz
    slosh resonance), optionally with a pour-like tilt (<= 0.35 rad) synced to the
    strokes, then a 1.0 s hold (late spills count);
  * the geometric cup-tilt cone is NOT on by default (mode-dependent);
  * contents = LIQUID PROXY (slosh_pend.PendulumObserver on the realized cup motion,
    fill h = 4 cm, zeta = 0.02 -> spill angle 54.5 deg); spill = phi >= phi*;
  * RIGHT arm runs the scene's crossing track (time-shifted so its closest approach
    to the cup falls inside the carry) -> arm-arm rows stay relevant;
  * ZERO actuation latency (FL-MPC is latency-fragile: see experiments/actuation --
    1 step of latency alone gave 1/50 collisions at N=50). No delay compensation here.

Modes (all FL-MPC underneath: hard arm-arm clearance + torque box + collision-relaxed fallback):
  a_none      no cup rows
  b_box       current FL-MPC cup-upright tilt BOX (tilt_max 0.10 rad, demo default)
  b_cone      effective-gravity cone (8 facets, tilt_max 0.30 rad) [optional]
  c_switch    LatentSafe least-restrictive SWITCH: every WM tick Q(z, task plan) < eps
              -> fallback policy (greedy argmax_u V(f(z,u))) mapped to a joint-accel
              reference, else task reference (bang-bang)
  d_value     OURS: FL-MPC + linearized latent-value rows V_k >= eps (slackable)
  e_margin    physics oracle: same rows with the TRUE instantaneous margin l over the
              160 ms horizon (no reachability)
  f_cbf_sample  LatentCBF (Nakamura et al. 2025, arXiv 2511.18606), faithful: one-step
              Q(z,a)-eps >= alpha (Q(z,pi_dag)-eps), sampling-based zeroth-order solve around
              {pi_nom, pi_dag}; see cbf_filter.py
  g_cbf_row   same one-step CBF condition, linearised, as ONE slackable row in FL-MPC
              (mechanism-isolating variant vs d_value's horizon rows)

  python -m world_model.v2.latent_hj.scenario --modes a_none,b_box,c_switch,d_value,e_margin \
      --seeds 0-4 --workers 5 --ckpt $S/v0/value.pt --tag smoke
SIMULATION ONLY (MuJoCo). No hardware, no network.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "eval"), str(REPO / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402
from world_model.v2.latent_hj import slosh_pend as sp  # noqa: E402
from world_model.v2.latent_hj import flmpc_value as fv  # noqa: E402

DT = sc.DT
SCR = Path(os.environ.get("LATENT_HJ_SCR", "/common/home/ss5772/.claude/jobs/f1580deb/tmp/latent_hj"))
OUT_DIR = Path("/common/users/ss5772/bimanual/world_model/v2/latent_hj/results")
EPS_V = 0.3           # value threshold (c and d share it)
EPS_L = 0.3           # margin threshold (e)
REFRESH = 1           # value-row refresh [sim steps]; 13 (15.4 Hz ZOH) fails on this task -- see report
FB_QDD_MAX = 40.0     # [rad/s^2] fallback joint-accel clip (== ValueFLMPC trust region)
FL_KW = dict(NC=4, L=8, W_POS_EGO=3000.0, W_VEL_EGO=15.0)
CBF_MODES = ("f_cbf_sample", "g_cbf_row")
# LatentCBF defaults: alpha in [0.85, 0.925] (paper); eps=None -> --eps-v; N=7600 (paper's hardware count);
# steps = control steps per latent step (2 x 5 ms = the 10 ms Bellman step V was trained with)
CBF_DEFAULTS = dict(alpha=0.9, eps=None, n=7600, steps=2, sig_lo=0.5, sig_hi=30.0, rows="all")


# =============================================================================== task generation
def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def make_task(seed, p0, rng=None):
    """Cup-pivot waypoint path + per-step tilt (rad) about the axis perp to the strokes."""
    rng = np.random.default_rng(seed) if rng is None else rng
    n_str = int(rng.integers(2, 5))
    th = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(th), np.sin(th), 0.0])
    amp = rng.uniform(0.06, 0.14)
    durs = rng.uniform(0.32, 0.60, n_str)
    pour = rng.random() < 0.35
    tilt_amp = rng.uniform(0.15, 0.35) if pour else 0.0
    wps = [np.zeros(3)]
    for i in range(n_str - 1):
        wps.append(d * amp * (1 if i % 2 == 0 else -1))
    wps.append(np.zeros(3))
    t_pre, t_hold = 0.2, 1.0
    T = int(round((t_pre + durs.sum() + t_hold) / DT))
    P = np.tile(p0, (T, 1)).astype(float); TILT = np.zeros(T)
    k = int(t_pre / DT)
    for i in range(n_str):
        n = int(round(durs[i] / DT))
        s = min_jerk(np.linspace(0, 1, n + 1)[1:])
        P[k:k + n] = p0 + wps[i] + (wps[i + 1] - wps[i]) * s[:, None]
        tgt_i = tilt_amp * (np.linalg.norm(wps[i]) > 0); tgt_n = tilt_amp * (np.linalg.norm(wps[i + 1]) > 0)
        TILT[k:k + n] = tgt_i + (tgt_n - tgt_i) * s
        k += n
    P[k:] = p0; TILT[k:] = 0.0
    tilt_axis = np.cross([0, 0, 1.0], d)
    return dict(P=P, TILT=TILT, tilt_axis=tilt_axis, T=T, n_str=n_str, amp=amp, durs=durs.tolist(),
                pour=bool(pour), tilt_amp=tilt_amp, t_motion_end=(t_pre + durs.sum()))


def axis_angle_R(axis, ang):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = sc.skew(axis)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def ik_track(model, data0, qadr, dadr, ee, piv_local, task, q0, iters=3, lam=0.02):
    """Per-step DLS IK of the pivot point + EE orientation (R_des = Rtilt @ R0)."""
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    d.qpos[qadr] = q0; mujoco.mj_kinematics(model, d)
    R0 = d.xmat[ee].reshape(3, 3).copy()
    rng_lo = model.jnt_range[[model.dof_jntid[i] for i in dadr], 0]
    rng_hi = model.jnt_range[[model.dof_jntid[i] for i in dadr], 1]
    q = q0.copy(); Q = np.zeros((task["T"], len(qadr))); err = np.zeros(task["T"])
    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
    for k in range(task["T"]):
        Rd = axis_angle_R(task["tilt_axis"], task["TILT"][k]) @ R0
        for _ in range(iters):
            d.qpos[qadr] = q; mujoco.mj_kinematics(model, d); mujoco.mj_comPos(model, d)
            R = d.xmat[ee].reshape(3, 3)
            p = d.xpos[ee] + R @ piv_local
            ep = task["P"][k] - p
            Re = Rd @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            mujoco.mj_jac(model, d, jp, jr, p, ee)
            J = np.vstack([jp[:, dadr], 0.3 * jr[:, dadr]])
            e = np.concatenate([ep, 0.3 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + dq, rng_lo, rng_hi)
        err[k] = np.linalg.norm(ep)
        Q[k] = q
    return Q, err


# =============================================================================== episode
def _load_meta(ckpt):
    from world_model.v2.latent_hj import reach_value as rv
    return rv.load(ckpt)[1]


def _load_value(ckpt):
    from world_model.v2.latent_hj import reach_value as rv
    net, meta = rv.load(ckpt)
    P = sp.PendParams.from_meta(meta["P"])
    return net, P


def run_episode(seed, mode, ckpt=None, max_steps=None, eps_v=EPS_V, eps_l=EPS_L, refresh=REFRESH,
                record=False, rho_v=1.0e4, trust=10.0, wm_path=None, wm_value=None, a_cap=60.0, cbf=None):
    import make_cup_collect_safety_demo as demo
    import scene_setup
    import run_ab4
    from teleop import bimanual_teleop as bt
    import comprehensive_eval as ce
    from control import realgrasp_flmpc as rg

    demo.N_CUBES = 3
    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True, cup_jitter_m=0.0, n_cubes=3)
    model, data = info["model"], info["data"]
    ld, rd_ = info["left_dofadr"], info["right_dofadr"]
    lq, rq = info["left_qposadr"], info["right_qposadr"]
    lee, ree, cup = info["left_ee_body"], info["right_ee_body"], info["cup_body"]
    track_r, q_home = info["track_r"], info["q_home"]
    cup_dof = int(model.jnt_dofadr[model.body_jntadr[cup]])
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)
    lj7, lj8 = model.actuator("left_arm_joint7_pos").id, model.actuator("left_arm_joint8_pos").id
    rfing = bt.finger_actuators(model, "right")
    left_geoms, right_geoms = run_ab4.left_right_geom_sets(model)
    ball_rel = scene_setup.canonical_ball_rel(model, cup)

    # ---- initial state: left arm at q_lift holding the cup, right arm at home
    data.qpos[lq] = info["q_lift"]; data.qpos[rq] = q_home; data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    scene_setup.reattach_cup_rigidly(model, data, lee, cup, info["cup_rel_pos0"], info["cup_rel_quat0"])
    scene_setup.carry_balls_with_cup(model, data, cup, ball_rel)
    mujoco.mj_forward(model, data)

    Pp = sp.PendParams() if (ckpt is None or not Path(ckpt).exists()) else _load_value(ckpt)[1]
    Pp.dt, Pp.sub = 0.01, 4            # fallback Bellman step (Stage-0 value was trained with these)
    Rc = data.xmat[cup].reshape(3, 3)
    piv0 = data.xpos[cup] + Rc[:, 2] * Pp.h
    R_ee = data.xmat[lee].reshape(3, 3)
    piv_local = R_ee.T @ (piv0 - data.xpos[lee])

    rng = np.random.default_rng(10_000 + seed)
    task = make_task(seed, piv0, rng)
    Qd, ik_err = ik_track(model, data, lq, ld, lee, piv_local, task, info["q_lift"])
    QDd = np.gradient(Qd, DT, axis=0); QDd[:int(0.2 / DT)] = 0
    QDDd = np.gradient(QDd, DT, axis=0)

    # right-arm time shift: closest approach of the right EE to the cup inside the carry window
    scr = mujoco.MjData(model); scr.qpos[:] = data.qpos
    t_grid = np.arange(0.0, 25.0, 0.05)       # track_r.sample clamps past its end
    dmin = []
    for tt in t_grid:
        scr.qpos[rq] = track_r.sample(tt); mujoco.mj_kinematics(model, scr)
        dmin.append(np.linalg.norm(scr.xpos[ree] - piv0))
    t_close = float(t_grid[int(np.argmin(dmin))])
    # closest approach at the END of the episode (during the post-carry hold): with the
    # approach inside the strokes the clearance rows were infeasible ~60% of steps in
    # every mode (smoke dbg2), swamping the spill comparison with collision fallbacks.
    t_r0 = max(0.0, t_close - task["T"] * DT)

    T = task["T"] if max_steps is None else min(task["T"], int(max_steps))
    cup_kps = info["cup_barrier_keypoints"]

    # ---- controller
    use_wm = mode.endswith("_wm")
    base = mode[:-3] if use_wm else mode
    cbfc = None                      # LatentCBF modes only (f_cbf_sample / g_cbf_row); None for every other mode
    if base in CBF_MODES:
        from world_model.v2.latent_hj import cbf_filter as cbf_mod
        cbf = dict(CBF_DEFAULTS, **(cbf or {}))
        if cbf["eps"] is None:
            cbf["eps"] = eps_v
        fl = cbf_mod.CBFFLMPC(model, rho_v=rho_v, trust=trust, a_cap=(a_cap if a_cap and a_cap > 0 else None), **FL_KW)
    else:
        fl = fv.ValueFLMPC(model, rho_v=rho_v, trust=trust, a_cap=(a_cap if a_cap and a_cap > 0 else None), **FL_KW)
    vc = None; net = None; wobs = None; dyn_fb = None; contents = None
    fb_hold = None
    import torch
    if use_wm:
        # Stage 1: LEARNED latent (kin_wm) + latent reachability value; the true liquid
        # proxy below is only the ground-truth spill metric, never a controller input.
        from world_model.v2.latent_hj import kin_wm as kwm, reach_value as rv, wm_adapter as wa
        wm = kwm.load_wm(wm_path)
        net, _ = rv.load(wm_value)
        for p_ in net.parameters():
            p_.requires_grad_(False)
        contents = wa.WMContents(wa.KinWMAdapter(wm), net, stride=kwm.STRIDE, score="value")
        wobs = kwm.KinWMObserver(wm, DT)
        wobs.update(data.xpos[cup].copy(), Rc[:, 2].copy())
        dyn_fb = kwm.WMDyn(wm, Pp, "cpu")
    elif base in ("c_switch", "d_value"):
        net, Pv = _load_value(ckpt)
        for p_ in net.parameters():
            p_.requires_grad_(False)
        Pv.dt, Pv.sub = 0.01, 4
        contents = fv.PendContents(Pv, net, "value")
        dyn_fb = _PD(Pv, torch.tensor(sp.control_set(Pv), dtype=torch.float32), sp.torch_fns())
    elif base == "e_margin":
        contents = fv.PendContents(Pp, None, "margin")
    elif base in CBF_MODES:
        net, Pv = _load_value(ckpt)
        for p_ in net.parameters():
            p_.requires_grad_(False)
        Pv.dt, Pv.sub = 0.01, 4
        dyn_fb = _PD(Pv, torch.tensor(sp.control_set(Pv), dtype=torch.float32), sp.torch_fns())
        gamma = float(_load_meta(ckpt)["gamma"])
        cbfc = cbf_mod.LatentCBF(fv.PendContents(Pv, net, "value"), net, dyn_fb, gamma=gamma, alpha=cbf["alpha"],
                                 eps=cbf["eps"], n_samples=cbf["n"], steps=cbf["steps"], dt=DT,
                                 sig_lo=cbf["sig_lo"], sig_hi=cbf["sig_hi"], a_cap=a_cap, rows_mode=cbf["rows"],
                                 seed=20_000 + seed)
        cbf["gamma"] = gamma
    if contents is not None:
        vc = fv.ValueConstraint(fl, contents, eps=(eps_l if base == "e_margin" else eps_v), refresh_every=refresh)
    obs = sp.PendulumObserver(Pp, DT, sub=4)
    obs.update(piv0, Rc[:, 2])

    tf_l = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    tf_r = sc.TorqueCommandFilter(6, sc.FILTER_LOWPASS_ALPHA, np.full(6, sc.FILTER_RATE_LIMIT_NM_S))
    NP = fl.NP
    log = {k: [] for k in ("phi", "h", "coll", "lee", "piv", "piv_des", "a_e0", "feas", "vslack", "vbind",
                            "fb", "ms", "vms", "V", "cup_tilt")}
    fb_mode = False
    prev_qpos = None
    CL = {k: [] for k in ("ovr", "anom", "q_nom", "q_star", "q_pi", "thr", "fb", "fms", "n_ok", "n_hard", "n_cbf")}
    for k in range(T):
        t = k * DT
        data.ctrl[lj7] = 0.0; data.ctrl[lj8] = 0.0
        bt.set_gripper(data, rfing, 1.0)
        mujoco.mj_forward(model, data)
        scene_setup.reattach_cup_rigidly(model, data, lee, cup, info["cup_rel_pos0"], info["cup_rel_quat0"])
        scene_setup.carry_balls_with_cup(model, data, cup, ball_rel)
        if prev_qpos is not None:
            qv = scene_setup.cup_velocity_from_motion(model, prev_qpos, data.qpos, DT)
            data.qvel[cup_dof:cup_dof + 6] = qv[cup_dof:cup_dof + 6]
        prev_qpos = data.qpos.copy()
        mujoco.mj_forward(model, data)

        Rc = data.xmat[cup].reshape(3, 3); cax = Rc[:, 2].copy()
        piv = data.xpos[cup] + cax * Pp.h
        z0 = wobs.z if use_wm else obs.state()
        q_l, qd_l = data.qpos[lq].copy(), data.qvel[ld].copy()

        idx = np.minimum(np.arange(k, k + NP + 1), task["T"] - 1)
        q_ego_ref, qd_ego_ref = Qd[idx], QDd[idx]
        tr = t + t_r0
        q_int_ref = np.stack([track_r.sample(tr + j * DT) for j in range(NP + 1)])
        qd_int_ref = np.stack([track_r.sample_dot(tr + j * DT) for j in range(NP + 1)])

        pairs = sc.all_pair_barriers(model, data, ld, rd_, extra_ego_keypoints=cup_kps)
        M_box, bias_box = sc.payload_reflected_terms(model, data, lee, cup, ld)

        a_nom_cbf = None
        if base == "g_cbf_row":
            # nominal (no CBF row; same trust region / a-cap) accel for the override metric ONLY -- untimed dry run
            a_nom_cbf, _ = cbf_mod.dry_nominal(fl, (model, data, lq, ld, rq, rd_, q_ego_ref, qd_ego_ref, q_int_ref,
                                                    qd_int_ref, pairs),
                                               dict(M_ego_extra=M_box, bias_ego_extra=bias_box, ego_hold=True),
                                               dummy_row=True)
        t0 = time.perf_counter()
        vrows = None; vms = 0.0; Vnow = np.nan
        jac = None
        if base in CBF_MODES:
            jac = fv.pivot_jacobians(model, data, lee, piv, ld)
        if vc is not None:
            jac = fv.pivot_jacobians(model, data, lee, piv, ld)
        if base in ("d_value", "e_margin"):
            vrows = vc.update(z0, qd_l, jac)
            vms = vc.last["t_ms"]
            Vnow = float(np.min(vc.last["V_ck"])) if vc.last.get("V_ck") is not None else np.nan
        elif base == "c_switch":
            t1 = time.perf_counter()
            if k % refresh == 0:
                # a_task = what the UNFILTERED task controller would plan now: a dry-run
                # FL-MPC solve on the task references (shallow copy -> the real
                # controller's warm-start state is untouched; FLMPC rebinds, never
                # mutates, its state arrays).
                import copy
                fl_dry = copy.copy(fl)
                fl_dry.step(model, data, lq, ld, rq, rd_, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
                            pairs, M_ego_extra=M_box, bias_ego_extra=bias_box, ego_hold=True, value_rows=None)
                x_task = fl_dry.last_plan_ego
                Vnow, _ = vc.plan_q(z0, qd_l, jac, x_task)
                fb_mode = Vnow < eps_v
                if fb_mode:
                    zt = (z0[None] if use_wm else torch.tensor(np.concatenate(z0), dtype=torch.float32)[None])
                    with torch.no_grad():
                        from world_model.v2.latent_hj import reach_value as rv
                        _, _, ib = rv.greedy_next(net, dyn_fb, zt)
                    u = dyn_fb.U[int(ib[0])].numpy().astype(float)
                    Jp, Jr, Jpd, Jrd = jac
                    w_now = Jr @ qd_l
                    rhs = np.concatenate([u[:3] - Jpd @ qd_l, (u[3:] - w_now) / 0.1 - Jrd @ qd_l])
                    J6 = np.vstack([Jp, Jr])
                    qdd_fb = J6.T @ np.linalg.solve(J6 @ J6.T + 1e-2 * np.eye(6), rhs)
                    qdd_fb = np.clip(qdd_fb, -FB_QDD_MAX, FB_QDD_MAX)   # same scale as d's trust region
                    fb_hold = (q_l, qd_l, qdd_fb, k)
            if fb_mode:
                q0f, qd0f, qddf, kf = fb_hold
                tt = (np.arange(NP + 1) + (k - kf)) * DT
                q_ego_ref = q0f[None] + np.outer(tt, qd0f) + 0.5 * np.outer(tt ** 2, qddf)
                qd_ego_ref = qd0f[None] + np.outer(tt, qddf)
            vms = 1e3 * (time.perf_counter() - t1)
        elif base == "g_cbf_row":
            vrows = cbfc.row(fl, z0, qd_l, jac)
            vms = cbfc.last["ms"]
        kw = {}
        if mode == "b_box":
            kw = dict(left_ee_body=lee, tilt_max_rad=0.10)
        elif mode == "b_cone":
            kw = dict(left_ee_body=lee, tilt_max_rad=0.30, cup_cone_facets=8)
        tau_l, tau_r, feas = fl.step(model, data, lq, ld, rq, rd_, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
                                     pairs, M_ego_extra=M_box, bias_ego_extra=bias_box, ego_hold=True,
                                     value_rows=vrows, **kw)
        if base == "f_cbf_sample":
            # LatentCBF filter on the ego first-block accel; the FL-MPC step above (no cup rows) IS pi_nom
            a_nom_cbf = fl.a_ego_prev.copy()
            a_star = cbfc.filter(fl, z0, qd_l, jac, a_nom_cbf)
            tau_l = fl.last_M_e @ a_star + fl.last_c_e
            fl.a_ego_prev = a_star.copy()
            vms = cbfc.last["ms"]
        # joint dry-friction feedforward (frictionloss 0.3 Nm/joint is a constraint force FL's c
        # does not cancel; without it realized qacc deviates ~60% from the plan -- validate_rows),
        # same compensation the demo applies (control/realgrasp_flmpc.friction_feedforward)
        tau_l = tau_l + rg.friction_feedforward(model, ld, data.qvel[ld], fl.a_ego_prev, DT)
        tau_r = tau_r + rg.friction_feedforward(model, rd_, data.qvel[rd_], fl.a_int_prev, DT)
        ms = 1e3 * (time.perf_counter() - t0)
        data.qfrc_applied[ld] = np.clip(tf_l.apply(tau_l, DT), -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[rd_] = np.clip(tf_r.apply(tau_r, DT), -sc.TAU_MAX, sc.TAU_MAX)

        coll = False
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 in left_geoms and c.geom2 in right_geoms) or (c.geom2 in left_geoms and c.geom1 in right_geoms):
                coll = True; break
        sl = fl._last_slack
        log["phi"].append(obs.phi); log["h"].append(min(p["h"] for p in pairs)); log["coll"].append(coll)
        log["lee"].append(data.xpos[lee].copy()); log["piv"].append(piv.copy()); log["piv_des"].append(task["P"][min(k, task["T"] - 1)])
        log["a_e0"].append(fl.a_ego_prev.copy()); log["feas"].append(bool(feas))
        log["vslack"].append(float(sl.max()) if len(sl) else 0.0)
        log["vbind"].append(bool(len(fl._last_lam_v) and (np.max(fl._last_lam_v) > 1e-9 or (len(sl) and sl.max() > 1e-7))))
        if base in CBF_MODES:
            a_app = fl.a_ego_prev
            if base == "g_cbf_row":
                Vnow = cbfc.q_of(z0, jac, qd_l, a_app)
                cbfc.last.update(q_star=Vnow, q_nom=cbfc.q_of(z0, jac, qd_l, a_nom_cbf),
                                 ovr=float(np.linalg.norm(a_app - a_nom_cbf)), fb=0, n_ok=np.nan, n_hard=np.nan, n_cbf=np.nan)
            else:
                Vnow = cbfc.last["q_star"]
            for k_, v_ in (("ovr", cbfc.last["ovr"]), ("anom", float(np.linalg.norm(a_nom_cbf))),
                           ("q_nom", cbfc.last["q_nom"]), ("q_star", cbfc.last["q_star"]), ("q_pi", cbfc.last["q_pi"]),
                           ("thr", cbfc.last["thr"]), ("fb", cbfc.last["fb"]), ("fms", vms),
                           ("n_ok", cbfc.last["n_ok"]), ("n_hard", cbfc.last["n_hard"]), ("n_cbf", cbfc.last["n_cbf"])):
                CL[k_].append(v_)
        log["fb"].append(bool(fb_mode)); log["ms"].append(ms); log["vms"].append(vms); log["V"].append(Vnow)
        log["cup_tilt"].append(float(np.arccos(np.clip(cax[2], -1, 1))))

        mujoco.mj_step(model, data)
        # contents: liquid proxy on the realized motion (after the step)
        mujoco.mj_kinematics(model, data)
        scene_setup.reattach_cup_rigidly(model, data, lee, cup, info["cup_rel_pos0"], info["cup_rel_quat0"])
        mujoco.mj_kinematics(model, data)
        Rc2 = data.xmat[cup].reshape(3, 3)
        obs.update(data.xpos[cup] + Rc2[:, 2] * Pp.h, Rc2[:, 2])
        if wobs is not None:
            wobs.update(data.xpos[cup].copy(), Rc2[:, 2].copy())
    L = {k: np.asarray(v) for k, v in log.items()}
    summ = summarize(L, Pp, task, mode, seed, ce)
    if cbfc is not None:
        summ.update(cbf_mod.summarize_cbf(CL, base, cbf))
        summ["cbf_gamma"] = cbf["gamma"]
        L.update({"cbf_" + k: np.asarray(v, float) for k, v in CL.items()})
    return summ, (L if record else None)


class _PD:
    """minimal Dyn for greedy_next on the analytic latent (fallback policy of c_switch)."""

    def __init__(self, P, U, fn):
        self.P, self.U, self.fn = P, U, fn

    def step(self, z, u):
        n, nd, c = z[..., 0:3], z[..., 3:6], z[..., 6:9]
        shp = torch_broadcast(n, u[..., 0:3])
        n2, nd2, c2 = self.fn["step"](n.expand(shp), nd.expand(shp), c.expand(shp),
                                      u[..., 0:3].expand(shp), u[..., 3:6].expand(shp), self.P)
        import torch
        return torch.cat([n2, nd2, c2], -1)

    def margin(self, z):
        return self.fn["margin"](z[..., 0:3], z[..., 6:9], self.P)

    def features(self, z):
        return self.fn["feats"](z[..., 0:3], z[..., 3:6], z[..., 6:9], nd_scale=1.0 / self.P.w1)


def torch_broadcast(a, b):
    import torch
    return torch.broadcast_shapes(a.shape, b.shape)


def summarize(L, P, task, mode, seed, ce):
    lee = L["lee"]
    speed = np.linalg.norm(np.diff(lee, axis=0) / DT, axis=1)
    try:
        sparc = float(sc.sparc(speed, DT))
    except Exception:
        sparc = float("nan")
    da = np.linalg.norm(np.diff(L["a_e0"], axis=0), axis=1)
    err = np.linalg.norm(L["piv"] - L["piv_des"], axis=1)
    kend = int(task["t_motion_end"] / DT)
    fb = L["fb"].astype(int); vb = L["vbind"].astype(int)
    eng = fb if mode.startswith("c_switch") else vb
    return dict(
        seed=seed, mode=mode, n_steps=int(len(L["phi"])),
        spill=bool(np.any(L["phi"] >= P.phi_star)), phi_max_deg=float(np.degrees(L["phi"].max())),
        t_spill=float(np.argmax(L["phi"] >= P.phi_star) * DT) if np.any(L["phi"] >= P.phi_star) else None,
        collision=bool(L["coll"].any()), h_min=float(L["h"].min()),
        track_rms_cm=float(100 * np.sqrt(np.mean(err ** 2))), track_max_cm=float(100 * err.max()),
        final_err_cm=float(100 * err[-1]),
        stroke_reach=float(np.mean(err[:kend] < 0.03)) if kend > 0 else float("nan"),
        jerk_rms=float(ce.jerk_rms(lee, DT)), sparc=sparc,
        switches=int(np.sum(np.abs(np.diff(eng)))), engaged_frac=float(eng.mean()),
        da_p99=float(np.percentile(da, 99)), da_max=float(da.max()),
        qp_infeas_steps=int((~L["feas"]).sum()), value_relaxed_steps=int((L["vslack"] > 1e-6).sum()),
        ms_mean=float(L["ms"].mean()), ms_p99=float(np.percentile(L["ms"], 99)),
        value_ms_mean=float(L["vms"].mean()), value_ms_max=float(L["vms"].max()),
        cup_tilt_max_deg=float(np.degrees(L["cup_tilt"].max())),
        task=dict(n_str=task["n_str"], amp=task["amp"], durs=task["durs"], pour=task["pour"], tilt_amp=task["tilt_amp"]),
    )


# =============================================================================== reference-only spill check
def reference_spill_rate(seeds, P=None):
    """Replay the liquid proxy on the COMMANDED pivot path (ideal tracking, no arm) --
    calibrates that the task demands spill-capable motion. Pure numpy, fast."""
    P = sp.PendParams() if P is None else P
    out = []
    for s in seeds:
        rng = np.random.default_rng(10_000 + s)
        task = make_task(s, np.zeros(3), rng)
        c = np.stack([axis_angle_R(task["tilt_axis"], a)[:, 2] for a in task["TILT"]])
        _, _, _, phi = sp.replay_np(task["P"] + c * P.h, c, P)
        out.append(dict(seed=s, spill=bool(np.any(phi >= P.phi_star)), phi_max=float(np.degrees(phi.max())),
                        pour=task["pour"], n_str=task["n_str"], amp=task["amp"], dmin=min(task["durs"])))
    return out


def _job(a):
    seed, mode, kw = a
    kw = dict(kw); out_dir = Path(kw.pop("_out", OUT_DIR))
    t0 = time.perf_counter()
    try:
        s, L = run_episode(seed, mode, **kw)
        s.update(ok=True, wall_s=time.perf_counter() - t0)
        if L is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg = "" if mode not in CBF_MODES else "_a%g_e%g_n%d" % (s["cbf_alpha"], s["cbf_eps"], s["cbf_N"])
            np.savez_compressed(out_dir / f"rec_{mode}{cfg}_s{seed}.npz", **L)
    except Exception as e:  # noqa: BLE001
        s = dict(seed=seed, mode=mode, ok=False, err=str(e)[:300], tb=traceback.format_exc()[-2000:])
    print(json.dumps(s, default=float), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="a_none,b_box,c_switch,d_value,e_margin")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--ckpt", default=str(SCR / "v1" / "value.pt"))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eps-v", type=float, default=EPS_V)
    ap.add_argument("--eps-l", type=float, default=EPS_L)
    ap.add_argument("--refresh", type=int, default=REFRESH)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--wm", default=str(SCR / "wm" / "wm.pt"))
    ap.add_argument("--wm-value", default=str(SCR / "wm_v" / "value.pt"))
    ap.add_argument("--rho-v", type=float, default=1.0e4)
    ap.add_argument("--trust", type=float, default=10.0)
    ap.add_argument("--a-cap", type=float, default=60.0, help="0 = no absolute ego-accel cap")
    ap.add_argument("--record", action="store_true", help="save per-step logs (npz) next to the json")
    ap.add_argument("--out-dir", default=None, help="output dir for the json / npz (default: results/ next to this file)")
    # ---- LatentCBF modes (f_cbf_sample, g_cbf_row); ignored by every other mode
    ap.add_argument("--alpha", type=float, default=CBF_DEFAULTS["alpha"], help="CBF alpha (paper 0.85-0.925)")
    ap.add_argument("--eps-cbf", type=float, default=None, help="CBF epsilon on the V scale (default: --eps-v)")
    ap.add_argument("--cbf-alphas", default=None, help="comma list -> grid over alpha for the CBF modes")
    ap.add_argument("--cbf-epss", default=None, help="comma list -> grid over eps for the CBF modes")
    ap.add_argument("--n-samples", type=int, default=CBF_DEFAULTS["n"], help="f_cbf_sample: #sampled actions (paper: 7600)")
    ap.add_argument("--cbf-steps", type=int, default=CBF_DEFAULTS["steps"], help="control steps (5 ms) per latent step; 2 = Bellman dt")
    ap.add_argument("--sig-lo", type=float, default=CBF_DEFAULTS["sig_lo"], help="sample noise sigma range [rad/s^2], log-uniform")
    ap.add_argument("--sig-hi", type=float, default=CBF_DEFAULTS["sig_hi"])
    ap.add_argument("--cbf-rows", default=CBF_DEFAULTS["rows"], choices=("block1", "all"),
                    help="hard FL-MPC rows a sample must satisfy: only those on the first blocks, or all")
    ap.add_argument("--ref-only", action="store_true")
    a = ap.parse_args()
    if "-" in a.seeds:
        lo, hi = a.seeds.split("-"); seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
    global OUT_DIR
    if a.out_dir:
        OUT_DIR = Path(a.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.ref_only:
        r = reference_spill_rate(seeds)
        print(json.dumps(dict(ref_spill_rate=float(np.mean([x["spill"] for x in r])), n=len(r))))
        (OUT_DIR / f"ref_{a.tag}.json").write_text(json.dumps(r, indent=1))
        return
    modes = [m for m in a.modes.split(",") if m]
    kw = dict(ckpt=a.ckpt, max_steps=a.max_steps, eps_v=a.eps_v, eps_l=a.eps_l, refresh=a.refresh,
              wm_path=a.wm, wm_value=a.wm_value, rho_v=a.rho_v, trust=a.trust, record=a.record, a_cap=a.a_cap)
    kw["_out"] = str(OUT_DIR)
    jobs = []
    for s in seeds:
        for m in modes:
            if m in CBF_MODES:
                als = [float(x) for x in a.cbf_alphas.split(",")] if a.cbf_alphas else [a.alpha]
                eps_l = [float(x) for x in a.cbf_epss.split(",")] if a.cbf_epss else [a.eps_cbf]
                for al in als:
                    for ep in eps_l:
                        kw_c = dict(kw, cbf=dict(alpha=al, eps=ep, n=a.n_samples, steps=a.cbf_steps,
                                                 sig_lo=a.sig_lo, sig_hi=a.sig_hi, rows=a.cbf_rows))
                        jobs.append((s, m, kw_c))
            else:
                jobs.append((s, m, kw))
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs)), maxtasksperchild=3) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
    else:
        rows = [_job(j) for j in jobs]
    (OUT_DIR / f"scenario_{a.tag}.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT_DIR / f'scenario_{a.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
