"""hold_test.py -- 5 s static bimanual tray hold on the YOR-NERO variant with the UNMODIFIED FL-MPC (NV=7).

Mirrors world_model/v2/house/phase1.carry: tray welded between the two gripper_base bodies (captured relpose), ball on the
tray, arm position servos disabled, torque applied via qfrc_applied = FLMPC.step output, mj_forward before the controller.

    python -m world_model.v2.nero.hold_test [--seconds 5] [--tau-preset urdf_effort] [--gravcomp] [--out results/hold_*.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from . import config as C
from . import adapters as A
from . import ik as IK

FL_KW = dict(NC=4, L=8, W_POS_EGO=3000.0, W_VEL_EGO=15.0)       # == house/phase1.FL_KW (== latent_hj.scenario.FL_KW)


class Ctl:
    """Driver around the unmodified FLMPC (NV=7): holds q_hold for both arms."""

    def __init__(self, m, hold_q, tau_vec, **kw):
        import mpc_fl
        self.m = m
        self.lq, self.ld = A.arm_addrs(m, "left")
        self.rq, self.rd = A.arm_addrs(m, "right")
        k = dict(FL_KW)
        k.update(kw)
        self.fl = mpc_fl.FLMPC(m, NV=C.N_ARM, tau_max_vec=tau_vec, **k)
        NP = self.fl.NP
        self.qe_ref = np.tile(np.array(hold_q["left"]), (NP + 1, 1))
        self.qi_ref = np.tile(np.array(hold_q["right"]), (NP + 1, 1))
        self.z = np.zeros((NP + 1, C.N_ARM))
        self.pairs = A.PairBarriers(m)
        self.n_infeasible = 0
        self.n_fallback = 0

    def tau(self, d):
        mujoco.mj_forward(self.m, d)           # qM / qfrc_bias / xpos are stale after mj_step otherwise
        pairs = self.pairs(d)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd,
                                  self.qe_ref, self.z, self.qi_ref, self.z, pairs)
        if not ok:
            self.n_infeasible += 1
        if self.fl.last_fallback:
            self.n_fallback += 1
        return te, ti


def contact_classes(m):
    cls = np.empty(m.ngeom, dtype="<U1")
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        nm = m.body(b).name or ""
        cls[g] = "L" if nm.startswith("left_arm_") else "R" if nm.startswith("right_arm_") else "T" if nm == "tray" else "P" if nm == "ball" else "B"
    return cls


def run(seconds=5.0, tau_preset=None, gravcomp=False, lift=False, hold_cache=None, **fl_kw):
    tau = C.tau_limits(tau_preset)
    m = A.compose_scene(lift=lift, gravcomp=gravcomp, tau=tau)
    d = mujoco.MjData(m)
    info = IK.set_hold_state(m, d)
    A.disable_arm_servos(m)
    ctl = Ctl(m, info["hold_q"], tau, **fl_kw)
    cls = contact_classes(m)
    tray, ball = m.body("tray").id, m.body("ball").id
    n = int(round(seconds / m.opt.timestep))
    q_hold = np.concatenate([info["hold_q"]["left"], info["hold_q"]["right"]])
    tray0 = d.xpos[tray].copy()
    ts_c, ts_p = [], []
    pair_tot = {}
    maxoff = 0.0; min_bz = 1e9; max_tilt = 0.0; max_dq = 0.0; drift = 0.0; lost_at = None
    tau_peak = np.zeros(14); min_h = 1e9
    for k in range(n):
        t0 = time.perf_counter()
        te, ti = ctl.tau(d)
        t1 = time.perf_counter()
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        mujoco.mj_step(m, d)
        t2 = time.perf_counter()
        ts_c.append((t1 - t0) * 1e3); ts_p.append((t2 - t1) * 1e3)
        tau_peak = np.maximum(tau_peak, np.abs(np.concatenate([te, ti])))
        for j in range(d.ncon):
            c = d.contact[j]
            a, b = sorted((cls[c.geom1], cls[c.geom2]))
            pair_tot[a + b] = pair_tot.get(a + b, 0) + 1
        R = d.xmat[tray].reshape(3, 3)
        rel = R.T @ (d.xpos[ball] - d.xpos[tray])
        maxoff = max(maxoff, float(np.hypot(rel[0], rel[1])))
        min_bz = min(min_bz, float(rel[2]))
        max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
        q = np.concatenate([d.qpos[ctl.lq], d.qpos[ctl.rq]])
        max_dq = max(max_dq, float(np.abs(q - q_hold).max()))
        drift = max(drift, float(np.linalg.norm(d.xpos[tray] - tray0)))
        if lost_at is None and (abs(rel[0]) > 0.15 or abs(rel[1]) > 0.10 or rel[2] < 0.0):
            lost_at = k * m.opt.timestep
        if k % 50 == 0:
            min_h = min(min_h, ctl.pairs.min_h(d))
    c = np.array(ts_c[5:]) if len(ts_c) > 10 else np.array(ts_c)       # skip first calls (numba cache / warmup)
    p = np.array(ts_p)
    lim = np.concatenate([tau, tau])
    return dict(
        config=dict(seconds=seconds, tau_preset=tau_preset or C.TORQUE_PRESET, tau_limits=tau.tolist(), gravcomp=gravcomp, lift=lift,
                    dt=float(m.opt.timestep), FL_KW={**FL_KW, **fl_kw}, NV=C.N_ARM),
        hold=dict(hold_q=info["hold_q"], ik_err=info["ik_err"], min_limit_margin=info["min_limit_margin"], min_pair_h_at_hold=info["min_pair_h"]),
        n_steps=n, n_infeasible=ctl.n_infeasible, n_fallback=ctl.n_fallback, n_taumax_relaxed=ctl.fl.n_taumax_relaxed,
        ball_stays=bool(lost_at is None), lost_at_s=lost_at, max_ball_offset_xy_m=maxoff, min_ball_z_above_tray_m=min_bz,
        max_tray_tilt_deg=max_tilt, max_tray_drift_m=drift, max_joint_dev_rad=max_dq,
        contacts_by_class_sum=pair_tot, min_proxy_h_sampled_m=min_h,
        ctrl_ms=dict(mean=float(c.mean()), median=float(np.median(c)), p95=float(np.percentile(c, 95)), max=float(c.max())),
        phys_ms=dict(mean=float(p.mean()), p95=float(np.percentile(p, 95))),
        tau_peak_Nm=tau_peak.round(3).tolist(), tau_peak_over_limit=float((tau_peak / lim).max()),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--tau-preset", default=None)
    ap.add_argument("--gravcomp", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.seconds, a.tau_preset, a.gravcomp)
    out = Path(a.out or Path(__file__).parent / "results" / f"hold_{a.tau_preset or C.TORQUE_PRESET}{'_gc' if a.gravcomp else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k not in ("hold", "config")}, indent=1))


if __name__ == "__main__":
    main()
