"""democ/turnover_sweep.py -- KINEMATIC box-turnover sweep (no dynamics).  SIMULATION ONLY, CPU.

The box is held rigidly between the two grasp frames (closed chain).  Rotating it by theta about the hand-to-hand axis (world y
through the box centre) leaves both hand positions fixed and rolls both gripper frames about their approach axis by theta
(jaw axis = Ry(theta) z).  For theta in [0, 180 deg] (and the mirrored direction) both arms are solved with FULL 6-D pose IK
(no jaw-sign symmetry exploited: the grasp is continuous), warm-started, with a mid-range nullspace posture task.

Per arm per angle: min joint-limit margin [rad] + which joint, sigma_min of the 6x7 Jacobian, IK residual (|[pos m; rotvec rad]|),
arm-arm proxy clearance (sphere surfaces, m) and robot contact count.  TRAP = first angle where an arm's IK residual exceeds
1e-4 (cannot hold the grasp pose inside its joint limits).  At the last feasible angle a NULL-SPACE test walks the 1-D
self-motion manifold (grasp pose held, Newton-projected) and then re-runs the sweep from the best manifold points to see
whether null-space motion alone restores the turnover (it does not if the new trap angle is not larger).

    python -m world_model.v2.nero.democ.turnover_sweep [--step-deg 3] [--n-starts 6] [--out results/democ_turnover_sweep.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from . import scene as S
from . import kin as K

RES_FAIL = 1e-4
FINGER_CLEAR = 0.0335          # fingers just clear of the 6 cm box (no pad contact) for the clearance / contact check


def box_quat(theta):
    return np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])


def null_vec(J):
    return np.linalg.svd(J)[2][-1]


def project(pik, q, p, R, iters=30, lam=1e-10):
    for _ in range(iters):
        e = pik.err(q, p, R)
        if np.linalg.norm(e) < 1e-11:
            break
        J = pik.ik.jac6(q)
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e), pik.lo, pik.hi)
    return q


def walk_manifold(pik, q0, p, R, ds=0.01, max_steps=500, lim=1e-3):
    """both branches of the self-motion manifold at fixed grasp pose -> list of configurations (incl. q0)."""
    out = [q0.copy()]
    for sign in (+1, -1):
        q = q0.copy(); nprev = None
        for _ in range(max_steps):
            n = null_vec(pik.ik.jac6(q))
            n = n * sign if nprev is None and n[2] * sign >= 0 else (n if nprev is None else (n if n @ nprev >= 0 else -n))
            if nprev is None and n[2] * sign < 0:
                n = -n
            nprev = n
            qn = project(pik, q + ds * n, p, R)
            if np.linalg.norm(pik.err(qn, p, R)) > 1e-8 or np.any(qn <= pik.lo + lim) or np.any(qn >= pik.hi - lim):
                break
            q = qn
            out.append(q.copy())
    return out


class Sweeper:
    def __init__(self, m, p: S.DemoCParams, jaw_sign, q_ref=None):
        self.m, self.p, self.jaw_sign = m, p, jaw_sign
        self.q_ref = q_ref            # {'left','right'} reference configs for the arm NOT being checked (contact check only)
        self.pik = {s: K.PoseIK(m, s) for s in C.SIDES}
        self.pb = A.PairBarriers(m)
        self.d = mujoco.MjData(m)

    def targets(self, theta):
        return K.box_grasp_targets(self.p, theta, self.jaw_sign)

    def solve_side(self, side, theta, q_prev, iters=80):
        pt, R = self.targets(theta)[side]
        return self.pik[side].solve(pt, R, q_prev, k_post=0.3, iters=iters, tol=1e-9)

    def record(self, side, q, e):
        pik = self.pik[side]
        mg, jn = pik.margin(q)
        return dict(margin_rad=mg, margin_joint=jn, sigma_min=pik.sigma_min(q), ik_residual=e)

    def arm_hits(self, side, q, theta):
        """contacts of this arm with table/pedestal/floor/box/itself at box angle theta (other arm parked at q_ref; its contacts ignored)."""
        if self.q_ref is None:
            return []
        other = "right" if side == "left" else "left"
        ql, qr = (q, self.q_ref[other]) if side == "left" else (self.q_ref[other], q)
        d, m = self.d, self.m
        K.set_state(m, d, self.p, ql, qr, FINGER_CLEAR)
        ba = m.jnt_qposadr[m.joint("democ_box_free").id]
        d.qpos[ba + 3:ba + 7] = box_quat(theta)
        mujoco.mj_forward(m, d)
        return [c for c in K.bad_contacts(m, d) if any(n.startswith(f"{side}_arm_") for n in c) and not any(n.startswith(f"{other}_arm_") for n in c)]

    def pair_check(self, theta, ql, qr):
        d, m = self.d, self.m
        K.set_state(m, d, self.p, ql, qr, FINGER_CLEAR)
        ba = m.jnt_qposadr[m.joint("democ_box_free").id]
        d.qpos[ba + 3:ba + 7] = box_quat(theta)
        mujoco.mj_forward(m, d)
        bc = K.bad_contacts(m, d)
        return dict(proxy_clearance_m=float(self.pb.min_h(d) + self.pb.d_min), n_contacts_robot=len(bc), names=bc[:4])

    def sweep(self, q_start, theta0=0.0, direction=+1, step=np.radians(3.0), max_theta=np.pi, sides=C.SIDES, check_pair=True):
        """-> dict(side -> list of records per angle) up to the first IK failure of that side (inclusive); `trap` summary."""
        n = int(round((max_theta - theta0) / step))
        qprev = {s: np.array(q_start[s], float) for s in sides}
        rec = {s: [] for s in sides}
        alive = {s: True for s in sides}
        trap = {s: None for s in sides}
        qlast_ok = {s: (theta0, qprev[s].copy()) for s in sides}
        pair = []
        for i in range(n + 1):
            th = theta0 + direction * i * step
            for s in sides:
                if not alive[s]:
                    continue
                q, e = self.solve_side(s, th, qprev[s])
                r = self.record(s, q, e)
                r["theta_deg"] = float(np.degrees(th)) * 1.0
                r["q"] = q.tolist()
                hits = self.arm_hits(s, q, th) if e <= RES_FAIL else []
                r["arm_contacts"] = len(hits)
                rec[s].append(r)
                if e > RES_FAIL or hits:
                    alive[s] = False
                    trap[s] = dict(theta_deg=r["theta_deg"], reason="joint_limit" if e > RES_FAIL else "collision:" + "/".join(hits[0]),
                                   joint_at_limit=r["margin_joint"], margin_rad=r["margin_rad"], residual=e,
                                   last_ok_theta_deg=float(np.degrees(qlast_ok[s][0])))
                else:
                    qprev[s] = q
                    qlast_ok[s] = (th, q.copy())
            if check_pair and all(alive[s] for s in sides) and len(sides) == 2:
                c = self.pair_check(th, qprev["left"], qprev["right"])
                c["theta_deg"] = float(np.degrees(th)); pair.append(c)
            if not any(alive.values()):
                break
        return dict(records=rec, trap=trap, last_ok={s: (float(np.degrees(qlast_ok[s][0])), qlast_ok[s][1].tolist()) for s in sides}, pair=pair)

    def nullspace_test(self, side, theta_ok, q_ok, direction, step, n_cand=8):
        """at the last feasible angle: walk the self-motion manifold; re-run the single-arm sweep from candidate points."""
        pt, R = self.targets(theta_ok)[side]
        pik = self.pik[side]
        q_ok = np.array(q_ok)
        path = walk_manifold(pik, q_ok, pt, R)
        margins = np.array([pik.margin(q)[0] for q in path])
        ib = int(np.argmax(margins))
        idx = sorted(set([ib] + list(np.linspace(0, len(path) - 1, n_cand).astype(int))))
        base = self.sweep({side: q_ok}, theta_ok, direction, step, sides=(side,), check_pair=False)
        base_end = base["last_ok"][side][0]
        best = dict(end_deg=base_end, idx=None)
        cands = []
        for i in idx:
            sw = self.sweep({side: path[i]}, theta_ok, direction, step, sides=(side,), check_pair=False)
            end = sw["last_ok"][side][0]
            cands.append(dict(path_index=int(i), margin_rad=float(margins[i]), reach_deg=end))
            if abs(end - np.degrees(theta_ok)) > abs(best["end_deg"] - np.degrees(theta_ok)) + 1e-9:
                best = dict(end_deg=end, idx=int(i))
        return dict(theta_deg=float(np.degrees(theta_ok)), manifold_points=len(path), margin_here_rad=float(margins[0]),
                    best_margin_on_manifold_rad=float(margins.max()), margin_gained_rad=float(margins.max() - margins[0]),
                    reach_without_nullspace_deg=base_end, best_reach_with_nullspace_deg=best["end_deg"],
                    nullspace_extends_reach=bool(abs(best["end_deg"] - np.degrees(theta_ok)) > abs(base_end - np.degrees(theta_ok)) + 1.0),
                    reaches_target=bool(abs(best["end_deg"]) >= 179.9), candidates=cands)


def summarize(sw, rows_every=5):
    out = {}
    for s, rows in sw["records"].items():
        ok = [r for r in rows if r["ik_residual"] <= RES_FAIL]
        out[s] = dict(
            n_ok=len(ok), reach_deg=ok[-1]["theta_deg"] if ok else None,
            min_margin_rad=min(r["margin_rad"] for r in ok) if ok else None,
            min_sigma_min=min(r["sigma_min"] for r in ok) if ok else None,
            max_ik_residual_ok=max(r["ik_residual"] for r in ok) if ok else None,
            table=[{k: v for k, v in r.items() if k != "q"} for r in rows[::rows_every] + ([rows[-1]] if rows else [])],
        )
    return out


def global_feasibility(m, p, side, step_deg=10.0, seeds=24):
    """Which box angles theta in (-180, 180] admit ANY joint-limit-respecting 6-D IK solution for this arm (jaw sign +1; jaw -1
    at theta == jaw +1 at theta+-180 because the gripper is symmetric)?  Multi-start, no continuity."""
    pik = K.PoseIK(m, side)
    rng = np.random.default_rng(3)
    feas = {}
    for th in np.arange(-180.0, 180.0 + 1e-9, step_deg):
        pt, R = K.box_grasp_targets(p, np.radians(th), (1.0, 1.0))[side]
        ok = False
        for sd in range(seeds):
            q0 = pik.q_post if sd == 0 else rng.uniform(pik.lo, pik.hi)
            q, e = pik.solve(pt, R, q0, rng.uniform(pik.lo, pik.hi) if sd else None, iters=150)
            if e < 1e-6:
                ok = True
                break
        feas[float(th)] = ok
    return feas


def intervals(feas):
    out, cur = [], None
    for th in sorted(feas):
        if feas[th] and cur is None:
            cur = [th, th]
        elif feas[th]:
            cur[1] = th
        elif cur is not None:
            out.append(cur); cur = None
    if cur is not None:
        out.append(cur)
    return out


def run(p: S.DemoCParams, step_deg=3.0, n_starts=8, verbose=True):
    t0 = time.time()
    m = S.compose(p)
    step = np.radians(step_deg)
    pre = K.find_pregrasp(m, p)
    res = dict(params={k: (list(v) if isinstance(v, tuple) else v) for k, v in S.dataclasses.asdict(p).items()},
               step_deg=step_deg, trap_definition=f"first angle with 6-D IK residual > {RES_FAIL} (joint limits clip the solution)",
               max_margin_pregrasp=dict(q_left=pre["ql"].tolist(), q_right=pre["qr"].tolist(), jaw_sign=list(pre["jaw_sign"]),
                                        margin_rad=min(K.PoseIK(m, "left").margin(pre["ql"])[0], K.PoseIK(m, "right").margin(pre["qr"])[0])),
               note="'best_start' = per arm, among n_starts IK solutions at theta=0 (both jaw signs) the one whose continuous sweep reaches furthest in that direction; "
                    "'max_margin_pregrasp' (the grasp_smoke pose, chosen for margin) is reported for comparison.")
    q_ref = {"left": pre["ql"], "right": pre["qr"]}
    res["best_start"] = {}
    for direction, key in ((+1, "plus_y_rotation"), (-1, "minus_y_rotation")):
        best = {}
        for side in C.SIDES:
            cand = []
            for js in (+1.0, -1.0):
                sws = Sweeper(m, p, (js, js), q_ref)
                tg = K.box_grasp_targets(p, 0.0, (js, js))[side]
                for q, e in K.collect_pose(sws.pik[side], tg[0], tg[1], n_starts, 60, seed=7):
                    r = sws.sweep({side: q}, 0.0, direction, step, sides=(side,), check_pair=False)
                    cand.append((abs(r["last_ok"][side][0]), js, q))
            cand.sort(key=lambda c: -c[0])
            best[side] = cand[0]
            res.setdefault("start_distribution", {})[f"{side}_{key}"] = dict(n=len(cand), reach_deg_sorted=[round(c[0], 1) for c in cand],
                                                                           jaw_sign_of_each=[c[1] for c in cand])
        # joint sweep from the two per-arm best starts (jaw sign may differ per arm -> use per-arm Sweeper targets)
        sw = Sweeper(m, p, (best["left"][1], best["right"][1]), q_ref)
        r = sw.sweep({"left": best["left"][2], "right": best["right"][2]}, 0.0, direction, step)
        entry = dict(jaw_sign=[best["left"][1], best["right"][1]], summary=summarize(r), trap=r["trap"],
                     pair_check=dict(min_proxy_clearance_m=min((c["proxy_clearance_m"] for c in r["pair"]), default=None),
                                     max_robot_contacts=max((c["n_contacts_robot"] for c in r["pair"]), default=None),
                                     first_contact_angle_deg=next((c["theta_deg"] for c in r["pair"] if c["n_contacts_robot"]), None),
                                     first_contact_names=next((c["names"] for c in r["pair"] if c["n_contacts_robot"]), None),
                                     n_angles=len(r["pair"])))
        entry["nullspace"] = {}
        for s in C.SIDES:
            if r["trap"][s] is not None:
                th_ok, q_ok = r["last_ok"][s]
                entry["nullspace"][s] = sw.nullspace_test(s, np.radians(th_ok), q_ok, direction, step)
        first = [(abs(t["theta_deg"]), s) for s, t in r["trap"].items() if t]
        entry["first_trap"] = None if not first else dict(arm=min(first)[1], **r["trap"][min(first)[1]])
        entry["full_180_both_arms"] = not first
        res["best_start"][key] = entry
        if verbose:
            print(key, "first_trap:", entry["first_trap"], "ns:", {s: (n["margin_gained_rad"], n["nullspace_extends_reach"]) for s, n in entry["nullspace"].items()}, flush=True)
    # default (max-margin grasp pose) sweep, for reference
    sw = Sweeper(m, p, pre["jaw_sign"], q_ref)
    res["max_margin_pregrasp_sweep"] = {}
    for direction, key in ((+1, "plus_y_rotation"), (-1, "minus_y_rotation")):
        r = sw.sweep({"left": pre["ql"], "right": pre["qr"]}, 0.0, direction, step, check_pair=False)
        res["max_margin_pregrasp_sweep"][key] = dict(trap=r["trap"], reach_deg={s: v[0] for s, v in r["last_ok"].items()})
    # global (multi-start, no continuity) feasibility of each arm over box angle: does the pose family exist at all?
    res["global_feasibility"] = {}
    for side in C.SIDES:
        f = global_feasibility(m, p, side)
        res["global_feasibility"][side] = dict(feasible_intervals_deg=intervals(f), step_deg=10.0, note="jaw sign +1; jaw -1 == this shifted by 180 deg")
    res["elapsed_s"] = time.time() - t0
    return res


def main(argv=None):
    ap = S.add_args(argparse.ArgumentParser())
    ap.add_argument("--step-deg", type=float, default=3.0)
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "democ_turnover_sweep.json"))
    a = ap.parse_args(argv)
    res = run(S.params_from_args(a), a.step_deg, a.n_starts)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", a.out, f"({res['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
