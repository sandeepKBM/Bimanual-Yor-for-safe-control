"""Branch-coverage equivalence for rt_flmpc vs mpc_fl.FLMPC on random robot states of the cup-collect
model (no episode needed). SIMULATION ONLY. Covers the option combinations the flagship G1 runs do
not exercise: tilt box / true cone / eff-gravity cone, leash on/off, table rows on/off, speed cap
on/off (with/without the u_dir row), ego_hold True/False, F6 per-joint tau_max_vec, F5 delay_steps
(queue prediction), and a large H_MARGIN (forces the relaxed / zero fallbacks).
For each case: rows (f, A, b) max |diff| vs the py QP, feasibility + fallback decision, and
(drop-in vs lean) row/solution agreement.
"""
import itertools
import json
import os
import sys

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_REPO, os.path.join(_REPO, "scripts"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import cup_collect_task_utils as cct  # noqa: E402
import mpc_fl  # noqa: E402
import rt_flmpc  # noqa: E402
import rt_kernels as K  # noqa: E402
import sim_common as sc  # noqa: E402
from rt_dynamics import LeanRobotDynamics  # noqa: E402


def main():
    m, d = cct.load_yor_cup_collect_model(keep_zone_marker=False)
    _, lq, ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
    _, rq, rd = sc.joint_addrs(m, sc.RIGHT_ARM_JOINTS)
    lee = m.body("left_arm_gripper_base").id
    ree = m.body("right_arm_gripper_base").id
    cup = m.body("cup").id
    lk = sc.barrier_keypoints(m, sc.EGO_PREFIX)
    rk = sc.barrier_keypoints(m, sc.INT_PREFIX)
    lean = LeanRobotDynamics(m)
    qp_inner = sc.solve_constrained_box_qp
    rng = np.random.default_rng(7)
    results = []
    cases = list(itertools.product(("box", "true", "effgrav"), (True, False), (True, False), (True, False),
                                   (True, False), (None, "vec"), (0, 1), (0.02, 0.15)))
    rng.shuffle(cases)
    for ci_, (tilt, leash, table, speed, ego_hold, tauv, delay, hm) in enumerate(cases[:96]):
        kw = dict(NC=4, L=8, W_POS_EGO=800.0, H_MARGIN=hm, delay_steps=delay,
                  cone_mode=("true" if tilt == "true" else "effgrav"))
        if speed:
            kw.update(ee_speed_cap=0.28, ee_speed_body_int=ree)
        if tauv:
            kw.update(tau_max_vec=np.array([40.0, 40.0, 30.0, 10.0, 10.0, 5.0]))
        py = mpc_fl.FLMPC(m, **kw)
        dr = rt_flmpc.RTFLMPC(m, **kw)
        ln = rt_flmpc.RTFLMPC(m, **kw)
        ln.configure_lean(lean, m, lq, ld, rq, rd, lee, cup_body=cup, right_ee_body=ree, speed_body=ree,
                          left_table_kps=lk, right_table_kps=rk, table_z=0.72, table_margin=0.006,
                          table_kp_radius=True, table_gravfix=True)
        for trial in range(4):
            mujoco.mj_resetData(m, d)
            d.qpos[lq] = rng.normal(0, 0.4, 6)
            d.qpos[rq] = rng.normal(0, 0.4, 6)
            d.qvel[ld] = rng.normal(0, 0.6, 6)
            d.qvel[rd] = rng.normal(0, 0.6, 6)
            mujoco.mj_forward(m, d)
            NP = py.NP
            qer = np.tile(d.qpos[lq] + rng.normal(0, 0.05, 6), (NP + 1, 1))
            qir = np.tile(d.qpos[rq] + rng.normal(0, 0.1, 6), (NP + 1, 1))
            z = np.zeros((NP + 1, 6))
            pairs = sc.all_pair_barriers(m, d, ld, rd)
            Mf = sc.full_mass_matrix(m, d)
            Mb, bb = sc.payload_reflected_terms(m, d, lee, cup, ld)
            Ml = Mf[np.ix_(ld, ld)] + Mb
            Mr = Mf[np.ix_(rd, rd)]
            bl = d.qfrc_bias[ld] - d.qfrc_passive[ld] + bb
            br = d.qfrc_bias[rd] - d.qfrc_passive[rd]
            tr = None
            if table:
                a_te, b_te = sc.table_plane_barrier_rows(m, d, lk, ld, np.linalg.inv(Ml), bl, 0.72, margin=0.006, kp_radius=True)
                a_ti, b_ti = sc.table_plane_barrier_rows(m, d, rk, rd, np.linalg.inv(Mr), br, 0.72, margin=0.006, kp_radius=True)
                tr = (a_te, b_te, a_ti, b_ti)
            q_in = [(rng.normal(0, 1, 6), rng.normal(0, 1, 6)) for _ in range(delay)]
            a_prev = (rng.normal(0, 1, 6), rng.normal(0, 1, 6))
            for c in (py, dr, ln):
                c.a_ego_prev, c.a_int_prev = a_prev[0].copy(), a_prev[1].copy()
                c._a_queue = [(a.copy(), b.copy()) for a, b in q_in]
                c._cup_axis_local = None
            kwa = dict(M_ego_extra=Mb, bias_ego_extra=bb, table_rows=tr, ego_hold=ego_hold,
                       left_ee_body=lee, tilt_max_rad=0.1, cup_cone_facets=(0 if tilt == "box" else 8))
            if leash:
                kwa.update(leash_ee_body=lee, leash_max_m=0.08)
            cap = []

            def rec(Hm, f, lo, hi, A=None, b=None, **k):
                x, lam, fe = qp_inner(Hm, f, lo, hi, A, b, **k)
                cap.append((np.array(f), np.array(A), np.array(b), x, fe))
                return x, lam, fe
            sc.solve_constrained_box_qp = rec
            try:
                out_py = py.step(m, d, lq, ld, rq, rd, qer, z, qir, z, pairs, **kwa)
            finally:
                sc.solve_constrained_box_qp = qp_inner
            out_dr = dr.step(m, d, lq, ld, rq, rd, qer, z, qir, z, pairs, **kwa)
            lkw = {k: v for k, v in kwa.items() if k not in ("M_ego_extra", "bias_ego_extra", "table_rows")}
            out_ln = ln.step_lean(d, qer, z, qir, z, table_floor=table,
                                  cup_pose=(d.xpos[cup].copy(), d.xmat[cup].reshape(3, 3).copy()), **lkw)
            f0, A0, b0, x0, fe0 = cap[0]
            mm = A0.shape[0]
            wd, wl = dr._ws, ln._ws
            res = dict(case=ci_, tilt=tilt, leash=leash, table=table, speed=speed, ego_hold=ego_hold,
                       tauv=bool(tauv), delay=delay, hm=hm,
                       m_py=mm, m_rt=int(dr.last_oi[K.OI_M]),
                       f=float(np.max(np.abs(wd.f - f0))),
                       A=float(np.max(np.abs(wd.A[:mm] - A0))) if int(dr.last_oi[K.OI_M]) == mm else float("inf"),
                       b=float(np.max(np.abs(wd.b[:mm] - b0))) if int(dr.last_oi[K.OI_M]) == mm else float("inf"),
                       feas_py=bool(fe0), feas_rt=bool(out_dr[2]), fb_py=py.last_fallback, fb_rt=dr.last_fallback,
                       lean_A=float(np.max(np.abs(wl.A[:mm] - wd.A[:mm]))) if int(ln.last_oi[K.OI_M]) == mm else float("inf"),
                       lean_b=float(np.max(np.abs(wl.b[:mm] - wd.b[:mm]))) if int(ln.last_oi[K.OI_M]) == mm else float("inf"),
                       lean_tau=float(max(np.max(np.abs(out_ln[0] - out_dr[0])), np.max(np.abs(out_ln[1] - out_dr[1])))),
                       fb_lean=ln.last_fallback, taumax_relaxed=(py.n_taumax_relaxed, dr.n_taumax_relaxed, ln.n_taumax_relaxed),
                       rt_hard=(int(dr.last_oi[K.OI_HARD_STATUS]), int(dr.last_oi[K.OI_HARD_IT])),
                       rt_relax=(int(dr.last_oi[K.OI_RELAX_STATUS]), int(dr.last_oi[K.OI_RELAX_IT])))
            if (bool(fe0) != bool(out_dr[2])) or (py.last_fallback != dr.last_fallback):
                # independent arbiter: HiGHS LP feasibility of the hard rows / of the relaxed problem
                from scipy.optimize import linprog
                n = f0.shape[0]
                bx = [(-py.a_box, py.a_box)] * n
                lp_h = linprog(np.zeros(n), A_ub=A0, b_ub=b0, bounds=bx, method="highs").status
                ns = int(dr.last_oi[K.OI_NS])
                srows = wd.slack_rows[:ns].copy()
                Aa = np.zeros((mm, n + ns)); Aa[:, :n] = A0
                for k_, r_ in enumerate(srows):
                    Aa[r_, n + k_] = -1.0
                lp_r = linprog(np.zeros(n + ns), A_ub=Aa, b_ub=b0, bounds=bx + [(0, py.S_MAX)] * ns, method="highs").status
                viol_py = float(np.max(A0 @ x0 - b0))
                res.update(lp_hard_status=int(lp_h), lp_relaxed_status=int(lp_r), py_hard_x_rowviol=viol_py,
                           py_final_rowviol=float(np.max(A0 @ np.concatenate([py.a_ego_prev, np.zeros(18), py.a_int_prev, np.zeros(18)]) - b0)))
                print("DISAGREEMENT", json.dumps(res, default=str), flush=True)
            results.append(res)
    R = results
    print(f"{len(R)} random states x option combos")
    print("rows count equal:", sum(r["m_py"] == r["m_rt"] for r in R), "/", len(R))
    print("max |df| %.2e  |dA| %.2e  |db| %.2e" % (max(r["f"] for r in R), max(r["A"] for r in R), max(r["b"] for r in R)))
    print("lean vs drop-in: |dA| %.2e |db| %.2e |dtau| %.2e ; lean fallback == drop-in: %d/%d" % (
        max(r["lean_A"] for r in R), max(r["lean_b"] for r in R), max(r["lean_tau"] for r in R),
        sum(r["fb_lean"] == r["fb_rt"] for r in R), len(R)))
    agree = sum(r["feas_py"] == r["feas_rt"] for r in R)
    print(f"hard feasibility py vs rt agree: {agree}/{len(R)}; py infeasible {sum(not r['feas_py'] for r in R)}, "
          f"rt not-optimal {sum(not r['feas_rt'] for r in R)}; py-false-infeasible "
          f"{sum((not r['feas_py']) and r['feas_rt'] for r in R)}, rt-infeasible-but-py-feasible "
          f"{sum(r['feas_py'] and (not r['feas_rt']) for r in R)}")
    print("fallback (0/1/2) py vs rt:", {k: sum((r['fb_py'], r['fb_rt']) == k for r in R) for k in
                                         set((r['fb_py'], r['fb_rt']) for r in R)})
    print("taumax_relaxed counters equal:", sum(len(set(r["taumax_relaxed"])) == 1 for r in R), "/", len(R))
    worst = sorted(R, key=lambda r: -max(r["A"], r["b"], r["f"]))[:3]
    print("worst row cases:", json.dumps(worst, default=str)[:800])


if __name__ == "__main__":
    main()
