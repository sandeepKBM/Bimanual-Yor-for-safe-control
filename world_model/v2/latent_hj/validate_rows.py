"""validate_rows.py -- checks of the linearized value-constraint math (Stage 0).

1. ROBOT PART: FL-MPC exact double integrator + frozen-at-t0 pivot Jacobians
   (flmpc_value.HorizonMaps) vs the TRUE MuJoCo evolution when the planned
   accelerations are realized open-loop with tau = M a + c (payload-reflected, no
   torque filter): error of predicted pivot accel / cup angular velocity / pivot
   position over the 160 ms horizon.
2. CONTENTS + VALUE PART: first-order prediction V_k(x_ref) + g_k.(x - x_ref) vs the
   exact nonlinear rollout V_k(x) for random plan perturbations of size delta.
3. gradient check: autograd g_k vs central finite differences.

  python -m world_model.v2.latent_hj.validate_rows --ckpt $S/v0/value.pt --n 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.latent_hj import scenario as S  # noqa: E402  (sets sys.path for sim modules)
from world_model.v2.latent_hj import flmpc_value as fv, slosh_pend as sp  # noqa: E402
import sim_common as sc  # noqa: E402
from control import realgrasp_flmpc as rg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=str(S.OUT_DIR / "validate_rows.json"))
    a = ap.parse_args()
    import make_cup_collect_safety_demo as demo
    import scene_setup
    rng = np.random.default_rng(0)
    net, P = S._load_value(a.ckpt)
    for p in net.parameters():
        p.requires_grad_(False)
    info = demo.build_episode(seed=0, scenario="crossing", ee_speed_limit=True, cup_jitter_m=0.0, n_cubes=3)
    model, data = info["model"], info["data"]
    ld, lq, rq, rd_ = info["left_dofadr"], info["left_qposadr"], info["right_qposadr"], info["right_dofadr"]
    lee, cup = info["left_ee_body"], info["cup_body"]
    sc.disable_left_arm_position_servos(model); sc.disable_right_arm_position_servos(model)
    fl = fv.ValueFLMPC(model, **S.FL_KW)
    maps = fv.HorizonMaps(fl)
    cont = fv.PendContents(P, net, "value")
    cont._dt = fl.DT
    robot, lin, grad = [], [], []
    for trial in range(a.n):
        d = mujoco.MjData(model); d.qpos[:] = data.qpos
        d.qpos[lq] = info["q_lift"] + rng.normal(0, 0.05, 6)
        d.qpos[rq] = info["q_home"]
        d.qvel[:] = 0; d.qvel[ld] = rng.normal(0, 0.4, 6)
        mujoco.mj_forward(model, d)
        scene_setup.reattach_cup_rigidly(model, d, lee, cup, info["cup_rel_pos0"], info["cup_rel_quat0"])
        mujoco.mj_forward(model, d)
        Rc = d.xmat[cup].reshape(3, 3); c0 = Rc[:, 2].copy(); piv = d.xpos[cup] + c0 * P.h
        R_ee = d.xmat[lee].reshape(3, 3).copy(); piv_loc = R_ee.T @ (piv - d.xpos[lee])
        jac = fv.pivot_jacobians(model, d, lee, piv, ld)
        qd0 = d.qvel[ld].copy()
        A0, A1, W0, W1 = maps.build(qd0, *jac)
        x = rng.normal(0, 15.0, fl.n_blk)                       # planned block accels [rad/s^2]
        a_pred = A0 + np.einsum("tik,k->ti", A1, x); w_pred = W0 + np.einsum("tik,k->ti", W1, x)
        # realize open-loop in MuJoCo: tau = M a_t + c each step (exact FL), cup pinned
        P_true, C_true = [piv.copy()], [c0.copy()]
        qacc_err = []
        for t in range(fl.NP):
            M = sc.full_mass_matrix(model, d)[np.ix_(ld, ld)]
            c = d.qfrc_bias[ld] - d.qfrc_passive[ld]
            at = maps.S[t] @ x
            d.qfrc_applied[:] = 0
            d.qfrc_applied[ld] = M @ at + c + rg.friction_feedforward(model, ld, d.qvel[ld], at, fl.DT)
            mujoco.mj_step(model, d)
            qacc_err.append(np.linalg.norm(d.qacc[ld] - at) / max(np.linalg.norm(at), 1e-9))
            mujoco.mj_kinematics(model, d)
            R = d.xmat[lee].reshape(3, 3)
            P_true.append(d.xpos[lee] + R @ piv_loc)
            C_true.append(R @ (R_ee.T @ c0))
        P_true = np.array(P_true); C_true = np.array(C_true)
        a_true = (P_true[2:] - 2 * P_true[1:-1] + P_true[:-2]) / fl.DT ** 2
        w_true = np.cross(C_true[:-1], C_true[1:]) / fl.DT
        e_a = np.linalg.norm(a_true - a_pred[1:], axis=1)
        e_w = np.linalg.norm((w_true - np.cross(C_true[:-1], np.cross(w_pred, C_true[:-1]))), axis=1)
        # same comparison against the FROZEN-JACOBIAN map fed with the REALIZED joint
        # accels isolates the kinematic (frozen J) error from the FL (dynamics) error
        robot.append(dict(qacc_rel_err_med=float(np.median(qacc_err)), acc_err_med=float(np.median(e_a)), acc_err_max=float(e_a.max()),
                          acc_mag_med=float(np.median(np.linalg.norm(a_true, axis=1))),
                          w_err_med=float(np.median(e_w)), w_mag_med=float(np.median(np.linalg.norm(w_true, axis=1)))))
        # contents: random slosh state from a replayed pendulum (moderate)
        th, thd = rng.uniform(-0.6, 0.6), rng.uniform(-8, 8)
        n0, nd0, _ = sp.planar_to_3d(th, thd, 0.0)
        ang = rng.uniform(0, 2 * np.pi); Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        z0 = (Rz @ n0, Rz @ nd0, c0)
        s0, J = cont.rollout_scores(z0, A0, A1, W0, W1, x, maps.checkpoints)
        for delta in (2.0, 5.0, 10.0, 20.0, 40.0):
            dx = rng.normal(0, 1, fl.n_blk); dx *= delta / np.abs(dx).max()
            s1, _ = cont.rollout_scores(z0, A0, A1, W0, W1, x + dx, maps.checkpoints, need_grad=False)
            lin.append(dict(delta=delta, err=float(np.abs(s1 - (s0 + J @ dx)).max()), change=float(np.abs(s1 - s0).max())))
        eps = 1e-3
        gfd = np.zeros_like(J)
        for j in range(fl.n_blk):
            e = np.zeros(fl.n_blk); e[j] = eps
            gfd[:, j] = (cont.rollout_scores(z0, A0, A1, W0, W1, x + e, maps.checkpoints, need_grad=False)[0]
                         - cont.rollout_scores(z0, A0, A1, W0, W1, x - e, maps.checkpoints, need_grad=False)[0]) / (2 * eps)
        grad.append(float(np.abs(gfd - J).max() / max(np.abs(J).max(), 1e-9)))
    out = dict(robot={k: float(np.median([r[k] for r in robot])) for k in robot[0]},
               lin={str(dl): dict(err_med=float(np.median([r["err"] for r in lin if r["delta"] == dl])),
                                  change_med=float(np.median([r["change"] for r in lin if r["delta"] == dl])))
                    for dl in (2.0, 5.0, 10.0, 20.0, 40.0)},
               grad_rel_err_max=float(np.max(grad)))
    print(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
