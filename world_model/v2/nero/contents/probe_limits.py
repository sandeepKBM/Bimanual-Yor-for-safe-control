"""nero/contents/probe_limits.py -- what the two NERO arms can probe from their joint torques, at the pose the probe runs in.
SIMULATION / MODEL ONLY (no hardware).  USER 11-05: "with the limits of our arm what is possible to be probed first, let us establish
that and note down".

Sensing assumed: joint encoders + a per-joint torque read (commanded torque or motor current) on each 7-DoF arm.  NERO has no wrist F/T
sensor.  Whether the real NERO reports a usable joint torque, and how accurate it is, is UNKNOWN: every number below is conditional on it.

Method (MODEL, linear, exact for the map): each claw's wrench is read as w_s = (J_s^T)^+ (tau_s - bias_s(q)).  A per-joint read error
|d tau_j| <= DTAU maps to a worst-case error sum_j |A_s[k, j]| DTAU on component k, where A_s carries w_s to the total wrench about
the object origin o.  The two DTAU levels are both ASSUMED:
    0.05 Nm  probe.TAU_NOISE (read noise only; friction known or dithered out)
    0.30 Nm  the sim joint frictionloss (config.ASSUMED["frictionloss"]): the static-friction band if friction is not modelled
Quantities (static equilibrium, object level, at the rest pose):
    total mass           M   = F_z / g                         error  e_Fz / g
    CoM horizontal       c_y = tau_x / (M g), c_x = -tau_y / (M g)   error  e_tau / (M g)   (first order, CoM at o)
    CoM height           needs a tilt phi:  error  e_tau / (M g sin phi)  (Jacobian of the level pose; tilt changes it slightly)
    moving contents      a lagging mass m changes the horizontal force by up to m a: smallest detectable m = e_Fxy / a
                         (the arm's own inertia model error M(q) qdd is NOT included: MODEL caveat)
    weighable range      largest total mass with the rest-pose row u <= U_ADMIT (probe.rung_torque_u: torque 16 Nm ASSUMED and the
                         per-claw payload rating 3 kg - gripper, MEASURED-spec, worst CoM over the prior's contents region)
    tray per-mug fill    a static weigh gives 3 numbers (M, M c_x, M c_y); n fills are identifiable only if rank = n.

    python -m world_model.v2.nero.contents.probe_limits --tasks box,tray2,tray4,tray6 --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from . import carry as CR
from . import envelope as E
from . import probe as P

DTAUS = (0.05, 0.30)                     # Nm, ASSUMED (see module doc)
MASSES = (0.5, 1.0, 2.0, 3.0)            # kg total, evaluation points for the M-dependent rows
PHIS = (3.0, 10.0)                       # deg: carry.R_TILT_OBJ (level, R) and carry.U_TILT_OBJ (U) -- both ASSUMED thresholds
ACCS = (0.5, 1.0)                        # m/s^2: E.A_NOM (ASSUMED) and 2x


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rest_maps(task, seed=0):
    """per-arm 6x7 maps tau -> total wrench [F; tau about o] (world frame), at the rest pose after a 0.3 s hold with no payload ff."""
    ep = P._episode(P._world(task, P._seed("calib", task, seed)))
    ep.ctl.mass = 0.0
    ep.run_phases([("hold", 0.3)])
    m, d = ep.m, ep.d
    mujoco.mj_forward(m, d)
    o = d.xpos[ep.obj].copy()
    A = {}
    for s in CR.SIDES:
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, ep.site[s])
        J = np.vstack([jp[:, ep.dofs[s]], jr[:, ep.dofs[s]]])
        W = np.linalg.pinv(J.T)                                       # 6x7: tau -> claw wrench [F; tau] at the site
        T = np.eye(6)
        T[3:, :3] = _skew(d.site_xpos[ep.site[s]] - o)                # moment about o
        A[s] = T @ W
    cups = [(d.xpos[c] - o)[:2].tolist() for c in ep.cups]
    return A, cups, dict(o=o.round(4).tolist(), cond={s: round(float(np.linalg.cond(A[s][:, :])), 1) for s in A})


def worst(A, dtau):
    """worst-case error per total-wrench component [Fx Fy Fz tx ty tz] for |d tau_j| <= dtau on every joint of both arms."""
    return sum(np.abs(A[s]).sum(axis=1) for s in A) * dtau


def weigh_max(task, m_cap=8.0, tol=1e-3):
    """largest total mass with probe.rung_torque_u <= U_ADMIT at the rest pose (MODEL); and which row binds there."""
    lo, hi = P.empty_mass(task), m_cap
    if P.rung_torque_u(task, lo) > E.U_ADMIT:
        return None, None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if P.rung_torque_u(task, mid) <= E.U_ADMIT else (lo, mid)
    c = max(lo - P.empty_mass(task), 0.0)
    Pc = P.interval_spec(task, *P.prior_contents(task)).contents.P
    env = P._rest_envelope(task)
    u, det = env.worst(P.OB.Contents(c, c, Pc), ["p0"], 0.0, "two_lever")
    if det is None or "tau" not in det:
        return round(lo, 3), "unreachable" if det and det.get("unreachable") else None
    r_tau = max(max(abs(x) for x in v) for v in det["tau"].values()) / float(np.min(env.tau_lim))
    r_pay = max(abs(v) for v in det["Fz"].values()) / env.F_cap
    return round(lo, 3), dict(u=round(u, 3), torque_ratio=round(r_tau, 3), payload_ratio=round(r_pay, 3),
                              binds="torque" if r_tau >= r_pay else "payload rating")


def task_rows(task):
    A, cups, info = rest_maps(task)
    g = P.G
    rows = dict(task=task, empty_mass=P.empty_mass(task), prior_total=[P.empty_mass(task) + x for x in P.prior_contents(task)], **info)
    Mmax, bind = weigh_max(task)
    rows["weigh_max_total_kg"], rows["weigh_max_binding"] = Mmax, bind
    per = {}
    for dt in DTAUS:
        e = worst(A, dt)
        r = dict(dM_kg=round(float(e[2] / g), 4),
                 dc_xy_mm={str(M): [round(float(1e3 * e[4] / (M * g)), 1), round(float(1e3 * e[3] / (M * g)), 1)] for M in MASSES},
                 dc_z_mm={f"{phi}deg": {str(M): round(float(1e3 * max(e[3], e[4]) / (M * g * np.sin(np.radians(phi)))), 1) for M in MASSES}
                          for phi in PHIS},
                 dm_moving_kg={str(a): round(float(max(e[0], e[1]) / a), 3) for a in ACCS})
        per[str(dt)] = r
    rows["by_dtau_Nm"] = per
    if cups:
        X = np.array([[1.0, x, y] for x, y in cups]).T            # fills -> (M, M c_x, M c_y)
        rk = int(np.linalg.matrix_rank(X, tol=1e-6))
        rows["per_mug_fill"] = dict(n_cups=len(cups), rank=rk, unidentified_dims=len(cups) - rk, cup_xy_m=np.round(cups, 3).tolist())
    return rows


def write_table(rs, out):
    L = ["# probe limits at the rest pose (MODEL; machine-written by contents/probe_limits.py)", "",
         "Torque-read error per joint dtau is ASSUMED (0.05 Nm = read noise; 0.30 Nm = sim frictionloss band). Real NERO torque read: UNKNOWN.", "",
         "| task | prior total kg | weighable max kg (binds) | dtau Nm | dM kg | dCoM x/y mm @1 kg | @3 kg | dCoM z mm, tilt 3 deg @1 kg | tilt 10 deg @1 kg | moving mass kg @0.5 m/s2 | per-mug fill unknown dims |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        for dt, p in r["by_dtau_Nm"].items():
            b = r["weigh_max_binding"]
            bs = b if isinstance(b, str) or b is None else f"{b['binds']}, u {b['u']}"
            pm = r.get("per_mug_fill")
            L.append(f"| {r['task']} | {r['prior_total'][0]:.2f}-{r['prior_total'][1]:.2f} | {r['weigh_max_total_kg']} ({bs}) | {dt} | "
                     f"{p['dM_kg']} | {p['dc_xy_mm']['1.0']} | {p['dc_xy_mm']['3.0']} | {p['dc_z_mm']['3.0deg']['1.0']} | "
                     f"{p['dc_z_mm']['10.0deg']['1.0']} | {p['dm_moving_kg']['0.5']} | "
                     f"{'-' if not pm else str(pm['unidentified_dims']) + ' of ' + str(pm['n_cups'])} |")
    (out / "TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="box,tray2,tray4,tray6")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rs = [task_rows(t) for t in a.tasks.split(",")]
    meta = dict(date=time.strftime("%Y-%m-%d"), planner=CR.resolve_planner(None), dtau_Nm=dict(zip(map(str, DTAUS), ("ASSUMED read noise", "ASSUMED frictionloss band"))),
                U_ADMIT=E.U_ADMIT, tau_lim=f"{float(E.TAU_LIM[0])} Nm (config.tau_limit)", payload="3 kg MEASURED-spec minus gripper (MJCF)",
                phis_deg=PHIS, accs=ACCS)
    (out / "LIMITS.json").write_text(json.dumps(dict(meta=meta, rows=rs), indent=1, default=float))
    write_table(rs, out)


if __name__ == "__main__":
    main()
