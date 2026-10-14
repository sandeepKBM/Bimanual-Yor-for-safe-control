"""Diagnose WHICH joints saturate and WHEN, from run_actuation_eval --dump-dir
npz files. Recomputes the pure gravity torque g(q) along the logged joint path
(qvel=0, arm gravcomp zeroed) to split the command into gravity vs dynamics.
SIMULATION ONLY.

  python experiments/actuation/diagnose_saturation.py <dump_dir>/act_c_s0.npz [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "experiments" / "actuation"), str(REPO / "scripts"), str(REPO),
           str(REPO / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from actuation_model import piper_ff_clip_limits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cubes", type=int, default=None)
    a = ap.parse_args()
    import make_cup_collect_safety_demo as demo
    z = np.load(a.npz)
    t = z["t"]
    info = demo.build_episode(seed=a.seed, scenario="crossing", ee_speed_limit=True, n_cubes=a.n_cubes)
    m, d = info["model"], info["data"]
    arm_bodies = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("left_arm_", "right_arm_"))]
    m.body_gravcomp[arm_bodies] = 0.0
    lim = piper_ff_clip_limits()
    t_lift_end = demo.T_LIFT_MOVE + demo.T_LIFT_HOLD
    phases = [("lift(<%.2fs)" % t_lift_end, t < t_lift_end),
              ("hold/wait", (t >= t_lift_end) & (t < demo.RIGHT_START_T)),
              ("crossing(>=%.2fs)" % demo.RIGHT_START_T, t >= demo.RIGHT_START_T)]
    np.set_printoptions(precision=2, suppress=True, linewidth=160)
    for side, key_cmd, key_q, qa, da in (("left", "cmd_l", "left_q", info["left_qposadr"], info["left_dofadr"]),
                                          ("right", "cmd_r", "right_q", info["right_qposadr"], info["right_dofadr"])):
        cmd = z[key_cmd]
        q = z[key_q][:len(cmd)]
        g = np.zeros_like(cmd)
        qpos0 = d.qpos.copy()
        for k in range(len(cmd)):
            d.qpos[:] = qpos0
            d.qpos[qa] = q[k]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            g[k] = d.qfrc_bias[da]
        sat = np.abs(cmd) > lim + 1e-9
        print(f"== {side} arm  (limits {lim})")
        print(f"  |cmd| max per joint      {np.abs(cmd).max(0)}")
        print(f"  |cmd| p99 per joint      {np.percentile(np.abs(cmd), 99, axis=0)}")
        print(f"  |g(q)| max per joint     {np.abs(g).max(0)}   (gravity-only, arm links; payload excluded)")
        print(f"  |cmd-g| max per joint    {np.abs(cmd - g).max(0)}   (dynamics/tracking part)")
        print(f"  headroom min (lim-|g|)   {(lim - np.abs(g)).min(0)}")
        print(f"  sat frac per joint       {sat.mean(0)}")
        for name, mask in phases:
            if mask[:len(cmd)].any():
                mm = mask[:len(cmd)]
                print(f"  sat frac [{name:18s}] {sat[mm].mean(0)}")
        anyj = sat.any(1)
        if anyj.any():
            ts = t[:len(cmd)][anyj]
            print(f"  saturation time span: {ts[0]:.3f}..{ts[-1]:.3f}s ; n_sat_steps={anyj.sum()}")
    if "h_min" in z:
        k = int(np.argmin(z["h_min"]))
        print(f"h_min {z['h_min'].min():.4f} at t={t[k]:.3f}s ; collision steps={int(z['collision'].sum())}; "
              f"tilt max {np.degrees(z['tilt'].max()):.2f} deg at t={t[int(np.argmax(z['tilt']))]:.3f}s")


if __name__ == "__main__":
    main()
