"""robot_authority.py -- how hard can the REAL arms push a bimanually held tray?  SIMULATION ONLY (offline, CPU).

Loads the UNMODIFIED YOR MJCF (YOR/robot/yor-description/robot-welded-base-and-lift.mjcf). Two things are changed
IN MEMORY only (nothing on disk): body gravcomp is set to 0 (the MJCF fakes gravity compensation; the real arm pays the
gravity torque, mostly as software feedforward) and nothing else.

Payload = tray 0.3 kg (inertia 2e-3, 2e-3, 3e-3, from b_tray_ball.tray_env) + ball 0.1 kg r=0.03 sitting on the tray.
Held RIGIDLY by both grippers (6-D wrench per hand, ASSUMED: a pinch grasp may not carry the full moment).

For each carry pose:
  1. IK both grasp points (fingertip midpoint = gripper_base + [0,0,0.1]) to the tray pose; keep reachable solutions.
  2. Inverse dynamics: tray twist/accel -> each hand's twist/accel -> joint qvel/qacc through its 6x6 Jacobian ->
     mj_inverse gives the torque of the bare arm (gravity included); the hand wrench adds J^T w.
  3. The two hand wrenches must sum to the tray's Newton-Euler wrench; the 6-D internal (squeeze) freedom is chosen
     by a small LP that minimises s = max_j |tau_j| / cap_j.  s <= 1 means the motion is feasible.
  4. Bisect the largest linear accel (per direction) and angular accel (per axis) with s <= 1 (s is convex in the
     accel, so bisection is exact).
  caps:  (a) +-100 Nm (MJCF forcerange) (b) real MIT-mode ff cap +-8 Nm, wrist (j4-j6, ASSUMED) +-6.4 Nm.
  Joint friction loss (MJCF 0.3 Nm) is subtracted from every cap.

  python -m world_model.v2.tray_rep.authority.robot_authority
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parents[4]
YOR_MJCF = REPO / "YOR/robot/yor-description/robot-welded-base-and-lift.mjcf"
OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/authority")
G = 9.81
GRIP_OFF = np.array([0.0, 0.0, 0.1])            # fingertip midpoint in gripper_base frame (matches *_ee_grasp site)
# tray payload (b_tray_ball.tray_env: tray 0.3 kg, diaginertia 2e-3 2e-3 3e-3; ball 0.1 kg r=0.03 on the tray top)
M_TRAY, I_TRAY = 0.30, np.diag([2e-3, 2e-3, 3e-3])
M_BALL, R_BALL = 0.10, 0.03
HX, HY = 0.15, 0.10                              # tray half sizes (b_tray_ball.tray_env)
VLIM = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 3.0])  # rad/s, YOR/piperlib/urdf/piper_description.urdf joint1-6
FINT, NMAX = 15.0, 0.5                            # ASSUMED grip limits for mode "lp": squeeze force [N], hand moment [Nm]
FRIC = 0.3                                       # MJCF piper-class joint frictionloss [Nm]
CAPS = {"mjcf100": np.full(6, 100.0),
        "real8": np.array([8.0, 8.0, 8.0, 6.4, 6.4, 6.4])}   # wrist = j4-j6 (ASSUMED which joints)


# ------------------------------------------------------------------------------------------------ payload
def payload_inertia():
    """-> mass, COM in tray frame, inertia about the COM (tray frame). Tray frame origin = tray centre, z = normal."""
    c = np.array([0.0, 0.0, M_BALL * R_BALL / (M_TRAY + M_BALL)])
    I_ball = 0.4 * M_BALL * R_BALL ** 2 * np.eye(3)
    rb = np.array([0.0, 0.0, R_BALL]) - c
    rt = -c
    par = lambda m, r: m * (r @ r * np.eye(3) - np.outer(r, r))
    return M_TRAY + M_BALL, c, I_TRAY + par(M_TRAY, rt) + I_ball + par(M_BALL, rb)


MASS, C_TRAY, I_COM = payload_inertia()


# ------------------------------------------------------------------------------------------------ model
class Rig:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(YOR_MJCF))
        self.m.body_gravcomp[:] = 0.0                      # in memory only: pay the real gravity torque
        # pure rigid-body torque: no contact / friction-loss / limit constraint forces in mj_inverse (friction is a cap margin)
        self.m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
                                    | mujoco.mjtDisableBit.mjDSBL_LIMIT | mujoco.mjtDisableBit.mjDSBL_EQUALITY)
        self.d = mujoco.MjData(self.m)
        self.sl = {"L": slice(0, 6), "R": slice(6, 12)}
        self.body = {"L": self.m.body("left_arm_gripper_base").id, "R": self.m.body("right_arm_gripper_base").id}
        self.lo, self.hi = self.m.jnt_range[:12, 0].copy(), self.m.jnt_range[:12, 1].copy()
        self.nv = self.m.nv

    # ---- kinematics
    def fk(self, q):
        self.d.qpos[:12] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def grasp(self, arm):
        b = self.body[arm]
        R = self.d.xmat[b].reshape(3, 3).copy()
        return self.d.xpos[b] + R @ GRIP_OFF, R

    def jac6(self, arm):
        """6x6 Jacobian [lin; ang] of the grasp point wrt the arm's 6 joints at the current d state."""
        P, _ = self.grasp(arm)
        jp, jr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jp, jr, P, self.body[arm])
        return np.vstack([jp[:, self.sl[arm]], jr[:, self.sl[arm]]])

    def jdot_qd(self, arm, q, qd, h=1e-6):
        """J-dot * qd by one-sided finite difference (6,)."""
        qq = q.copy()
        self.fk(qq); J0 = self.jac6(arm)
        qq[self.sl[arm]] += h * qd[self.sl[arm]]
        self.fk(qq); J1 = self.jac6(arm)
        self.fk(q)
        return ((J1 - J0) / h) @ qd[self.sl[arm]]

    # ---- IK
    def ik(self, arm, p_t, R_t, q0, iters=250, damp=1e-2):
        q = q0.copy()
        s = self.sl[arm]
        for _ in range(iters):
            self.fk(q)
            P, R = self.grasp(arm)
            ep = p_t - P
            er = 0.5 * sum(np.cross(R[:, k], R_t[:, k]) for k in range(3))
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
                return q, True
            J = self.jac6(arm)
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + damp ** 2 * np.eye(6), e)
            dq = np.clip(dq, -0.3, 0.3)
            q[s] = np.clip(q[s] + dq, self.lo[s], self.hi[s])
        return q, False

    def ik_solutions(self, arm, p_t, R_t, n_seed=120, seed=0, max_sol=5):
        rng = np.random.default_rng(seed)
        s = self.sl[arm]
        sols = []
        flip = np.diag([-1.0, -1.0, 1.0])               # jaw is symmetric: R and R*Rz(pi) are the same grasp
        for it in range(n_seed):
            q0 = np.zeros(12)
            q0[s] = rng.uniform(self.lo[s], self.hi[s])
            q, ok = self.ik(arm, p_t, R_t if it % 2 == 0 else R_t @ flip, q0)
            if ok and all(np.linalg.norm(q[s] - x[s]) > 0.2 for x in sols):
                sols.append(q.copy())
                if len(sols) >= max_sol:
                    break
        return sols


# ------------------------------------------------------------------------------------------------ carry poses
def tray_grasp_targets(style, tray_o, y_c_unused=None):
    """world grasp points + frames for left ('L', +y side of the robot) and right ('R') hands, tray level.
    'side'  : hands at the two long edges (y = +-(HY-0.01)), approach inward, jaw axis vertical.
    'front' : hands at the near edge (x = -HX+0.01, y = +-0.06), approach forward (+x), jaw axis vertical.
    Frame columns = gripper_base [x_g, y_g, z_g]: z_g = approach, y_g = jaw axis (verified in sim_common/scene docs)."""
    o = np.asarray(tray_o, float)
    up = np.array([0.0, 0.0, 1.0])

    def frame(zg):
        yg = up
        xg = np.cross(yg, zg)
        return np.stack([xg, yg, zg], 1)
    if style == "side":
        pl, pr = o + [0, HY - 0.01, 0], o + [0, -(HY - 0.01), 0]
        Rl, Rr = frame(np.array([0, -1.0, 0])), frame(np.array([0, 1.0, 0]))
    else:
        pl, pr = o + [-HX + 0.01, 0.06, 0], o + [-HX + 0.01, -0.06, 0]
        Rl = Rr = frame(np.array([1.0, 0, 0]))
    return {"L": (pl, Rl), "R": (pr, Rr)}


class Ctx:
    """a solved carry pose (joint config of both arms, tray level)."""
    def __init__(self, rig, q, tray_o, style):
        self.rig, self.q, self.o, self.style = rig, q.copy(), np.asarray(tray_o, float), style
        rig.fk(q)
        self.P = {a: rig.grasp(a)[0] for a in "LR"}
        self.J = {a: rig.jac6(a) for a in "LR"}
        self.cond = {a: float(np.linalg.cond(self.J[a])) for a in "LR"}
        self.C = self.o + C_TRAY                          # tray level: R = I

    def tau_free(self, v, w, a, al):
        """-> (tau0 (12,), J dict at q).  Tray twist (v, w) and accel (a, al), both at the tray origin, world frame."""
        rig = self.rig
        qd, qdd = np.zeros(12), np.zeros(12)
        for arm in "LR":
            r = self.P[arm] - self.o
            V = np.concatenate([v + np.cross(w, r), w])
            qd[rig.sl[arm]] = np.linalg.solve(self.J[arm], V)
        for arm in "LR":
            r = self.P[arm] - self.o
            A = np.concatenate([a + np.cross(al, r) + np.cross(w, np.cross(w, r)), al])
            jd = rig.jdot_qd(arm, self.q, qd) if np.any(qd) else np.zeros(6)
            qdd[rig.sl[arm]] = np.linalg.solve(self.J[arm], A - jd)
        rig.d.qpos[:12], rig.d.qvel[:12], rig.d.qacc[:12] = self.q, qd, qdd
        mujoco.mj_inverse(rig.m, rig.d)
        return rig.d.qfrc_inverse[:12].copy(), qd

    def wrench(self, w, a, al):
        """tray Newton-Euler: net force and moment about the COM that the two hands must supply."""
        r = self.C - self.o
        a_c = a + np.cross(al, r) + np.cross(w, np.cross(w, r))
        F = MASS * (a_c + np.array([0.0, 0.0, G]))
        Mo = I_COM @ al + np.cross(w, I_COM @ w)           # tray level (R = I): world = tray frame
        return F, Mo

    def solve(self, cap, v=None, w=None, a=None, al=None, mode="lp"):
        """Hand wrenches that carry the tray and minimise s = max_j |tau_j|/cap_j.  Returns (s, tau (12,)).
        mode 'lp'    : free squeeze split, but |f_L - F/2|_inf <= FINT N and |n_i|_inf <= NMAX Nm (ASSUMED grip limits).
        mode 'ls'    : plain load sharing: min ||f||^2 + ||n/0.1||^2 s.t. the tray wrench (no tuned internal force).
        mode 'free'  : unbounded squeeze split (unphysical: hands may load each other without limit)."""
        z = np.zeros(3)
        v, w, a, al = (z if x is None else np.asarray(x, float) for x in (v, w, a, al))
        tau0, _ = self.tau_free(v, w, a, al)
        F, Mc = self.wrench(w, a, al)
        sk = lambda r: np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        A_eq = np.zeros((6, 12))
        for i, arm in enumerate("LR"):
            A_eq[:3, 6 * i:6 * i + 3] = np.eye(3)
            A_eq[3:, 6 * i:6 * i + 3] = sk(self.P[arm] - self.C)
            A_eq[3:, 6 * i + 3:6 * i + 6] = np.eye(3)
        b_eq = np.concatenate([F, Mc])
        if mode == "ls":
            W = np.diag(np.tile([1, 1, 1, 100, 100, 100.0], 2))      # weights on f (1) and n (1/0.1^2)
            Wi = np.linalg.inv(W)
            x = Wi @ A_eq.T @ np.linalg.solve(A_eq @ Wi @ A_eq.T, b_eq)
            tau = tau0.copy()
            for i, arm in enumerate("LR"):
                tau[6 * i:6 * i + 6] += self.J[arm].T @ x[6 * i:6 * i + 6]
            capv = np.concatenate([cap, cap]) - FRIC
            return float(np.max(np.abs(tau) / capv)), tau
        capv = np.concatenate([cap, cap]) - FRIC
        A_ub, b_ub = [], []
        for i, arm in enumerate("LR"):
            Jt = self.J[arm].T                            # (6 joints, 6 wrench) ; wrench = [f; n]
            for j in range(6):
                row = np.zeros(13)
                row[6 * i:6 * i + 6] = Jt[j]
                row[12] = -capv[6 * i + j]
                A_ub.append(row.copy()); b_ub.append(-tau0[6 * i + j])
                row[:12] *= -1
                A_ub.append(row); b_ub.append(tau0[6 * i + j])
        c = np.zeros(13); c[12] = 1.0
        bounds = [(None, None)] * 12 + [(0, None)]
        if mode == "lp":
            bounds = [(F[k] / 2 - FINT, F[k] / 2 + FINT) for k in range(3)] + [(-NMAX, NMAX)] * 3 + \
                     [(F[k] / 2 - FINT, F[k] / 2 + FINT) for k in range(3)] + [(-NMAX, NMAX)] * 3 + [(0, None)]
        r = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.hstack([A_eq, np.zeros((6, 1))]), b_eq=b_eq,
                    bounds=bounds, method="highs")
        if r.status != 0:
            return np.inf, tau0
        x = r.x
        tau = tau0.copy()
        for i, arm in enumerate("LR"):
            tau[6 * i:6 * i + 6] += self.J[arm].T @ x[6 * i:6 * i + 6]
        return float(x[12]), tau

    def max_along(self, cap, kind, direction, hi=80.0, v=None, tol=1e-3, mode="lp"):
        """largest scalar k >= 0 with s <= 1 for accel (kind 'a': m/s^2) or ang accel (kind 'al': rad/s^2) = k*dir."""
        d = np.asarray(direction, float)
        f = (lambda k: self.solve(cap, v=v, a=k * d, mode=mode)[0]) if kind == "a" else (lambda k: self.solve(cap, v=v, al=k * d, mode=mode)[0])
        if f(0.0) > 1.0:
            return 0.0
        if f(hi) <= 1.0:
            return hi
        lo_, hi_ = 0.0, hi
        while hi_ - lo_ > tol * max(1.0, lo_):
            mid = 0.5 * (lo_ + hi_)
            lo_, hi_ = (mid, hi_) if f(mid) <= 1.0 else (lo_, mid)
        return lo_

    def vel_limit(self, kind, direction):
        """largest speed (m/s for 'v', rad/s for 'w') along direction before any joint reaches its MJCF/URDF rate limit."""
        d = np.asarray(direction, float)
        worst = np.inf
        for arm in "LR":
            r = self.P[arm] - self.o
            V = np.concatenate([d, np.zeros(3)]) if kind == "v" else np.concatenate([np.cross(d, r), d])
            qd = np.abs(np.linalg.solve(self.J[arm], V))
            worst = min(worst, float(np.min(VLIM / np.maximum(qd, 1e-12))))
        return worst


# ------------------------------------------------------------------------------------------------ driver
DIRS8 = [np.array([np.cos(t), np.sin(t), 0.0]) for t in np.arange(8) * np.pi / 4]
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
# tray centre poses (world, robot faces +x, shoulders at y = -0.21, z ~ 0.67). Representative carry workspace.
YC = -0.209550
POSES = {
    "near_low": (0.30, YC, 0.80), "near_mid": (0.30, YC, 0.95), "mid_low": (0.40, YC, 0.80), "mid_mid": (0.40, YC, 0.95),
    "mid_high": (0.40, YC, 1.10), "far_mid": (0.50, YC, 0.95), "far_low": (0.50, YC, 0.80), "far_high": (0.50, YC, 1.10),
    "mid_left": (0.40, YC + 0.15, 0.95), "mid_right": (0.40, YC - 0.15, 0.95),
    "reach_55": (0.55, YC, 0.95), "high_35": (0.35, YC, 1.20),
}


MARGIN_MIN = 0.10      # rad: a usable carry pose keeps every arm joint at least this far from its limit


def pair_margin(rig, q):
    return float(min(np.min(q - rig.lo), np.min(rig.hi - q)))


def build(rig, name, o, style, seed=0, n_seed=200, max_sol=6):
    """IK both hands; among solution pairs with joint margin >= MARGIN_MIN pick the lowest static s under the real caps.
    If no pair has the margin, fall back to the best pair at any margin (flagged by the caller via joint_margin_rad)."""
    tg = tray_grasp_targets(style, o)
    sols = {a: rig.ik_solutions(a, *tg[a], n_seed=n_seed, seed=seed, max_sol=max_sol) for a in "LR"}
    if not sols["L"] or not sols["R"]:
        return None, {k: len(v) for k, v in sols.items()}
    best, bs = None, np.inf
    for ql in sols["L"]:
        for qr in sols["R"]:
            q = np.zeros(12); q[:6] = ql[:6]; q[6:] = qr[6:]
            cx = Ctx(rig, q, o, style)
            sc, _ = cx.solve(CAPS["real8"], mode="ls")
            sc += 0.0 if pair_margin(rig, q) >= MARGIN_MIN else 10.0          # prefer pairs away from the joint limits
            if sc < bs:
                best, bs = cx, sc
    return best, {k: len(v) for k, v in sols.items()}


def analyse(cx):
    r = {"q": cx.q.tolist(), "cond": cx.cond, "joint_margin_rad": float(min(np.min(cx.q - cx.rig.lo), np.min(cx.rig.hi - cx.q)))}
    cx.rig.d.qpos[:12] = cx.q; cx.rig.d.qvel[:] = 0; cx.rig.d.qacc[:] = 0
    mujoco.mj_inverse(cx.rig.m, cx.rig.d)
    r["arm_only_gravity_tau"] = cx.rig.d.qfrc_inverse[:12].tolist()          # bare arms, no payload
    for cname, cap in CAPS.items():
        for mode in ("lp", "ls"):
            s0, tau0 = cx.solve(cap, mode=mode)
            out = {"static_s": s0, "static_tau": tau0.tolist()}
            lim_a = {}
            for i, d in enumerate(DIRS8):
                lim_a[f"h{i*45}"] = cx.max_along(cap, "a", d, mode=mode)
            lim_a["+z"] = cx.max_along(cap, "a", AX["z"], mode=mode); lim_a["-z"] = cx.max_along(cap, "a", -AX["z"], mode=mode)
            out["a_lim"] = lim_a
            out["a_horiz_min"] = float(min(lim_a[f"h{i*45}"] for i in range(8)))
            al = {}
            for k, ax in AX.items():
                al[f"+{k}"] = cx.max_along(cap, "al", ax, hi=2000.0, mode=mode); al[f"-{k}"] = cx.max_along(cap, "al", -ax, hi=2000.0, mode=mode)
            out["alpha_lim"] = al
            out["alpha_xy_min"] = float(min(al[f"{s}{k}"] for s in "+-" for k in "xy"))
            r[f"{cname}_{mode}"] = out
    r["v_lim"] = {f"h{i*45}": cx.vel_limit("v", d) for i, d in enumerate(DIRS8)}
    r["v_lim"]["z"] = cx.vel_limit("v", AX["z"])
    r["w_lim"] = {k: cx.vel_limit("w", ax) for k, ax in AX.items()}
    # velocity-dependent torque: accel limit while moving at 0.5 m/s along the accel direction (real8, lp)
    r["a_lim_moving_0p5"] = {f"h{i*45}": cx.max_along(CAPS["real8"], "a", DIRS8[i], v=0.5 * DIRS8[i]) for i in (0, 2, 4, 6)}
    return r


def gravity_refs(rig):
    """arm-only static gravity torque (gravcomp off) at the repo's own named arm poses, for cross-checking the 7.4 Nm claim."""
    refs = {"flagship_LEFT_GRASP_Q": np.array([-0.00944173, 2.69291536, -1.83582068, -0.47028722, 1.03733985, -1.30204097]),
            "flagship_RIGHT_HOME": np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78]),
            "flagship_RIGHT_INTRUDE": np.array([-0.15229176, 2.56294267, -1.48373594, 0.93891875, 1.01939931, 0.80796997])}
    out = {}
    for k, q6 in refs.items():
        for arm, sl in (("L", slice(0, 6)), ("R", slice(6, 12))):
            q = np.zeros(12); q[sl] = q6
            rig.d.qpos[:12] = q; rig.d.qvel[:] = 0; rig.d.qacc[:] = 0
            mujoco.mj_inverse(rig.m, rig.d)
            out[f"{k}/{arm}"] = rig.d.qfrc_inverse[sl].tolist()
    return out


def main():
    t0 = time.perf_counter()
    rig = Rig()
    res = {"model": str(YOR_MJCF), "payload": {"mass": MASS, "com_tray": C_TRAY.tolist(), "I_com": I_COM.tolist()},
           "caps": {k: v.tolist() for k, v in CAPS.items()}, "friction_margin": FRIC, "poses": {}}
    res["gravity_refs"] = gravity_refs(rig)
    for style in ("side", "front"):
        for name, o in POSES.items():
            key = f"{style}/{name}"
            cx, ns = build(rig, name, o, style)
            if cx is None:
                res["poses"][key] = {"reachable": False, "ik_solutions": ns, "tray_o": list(o)}
                print(key, "UNREACHABLE", ns, flush=True)
                continue
            a = analyse(cx)
            a.update(reachable=True, ik_solutions=ns, tray_o=list(o), usable=bool(a["joint_margin_rad"] >= MARGIN_MIN))
            res["poses"][key] = a
            print(f"{key:18s} margin={a['joint_margin_rad']:.2f} static s(real8,lp)={a['real8_lp']['static_s']:.2f} a_h_min real8 lp={a['real8_lp']['a_horiz_min']:.2f} "
                  f"ls={a['real8_ls']['a_horiz_min']:.2f}  mjcf100={a['mjcf100_ls']['a_horiz_min']:.1f}  "
                  f"alpha_xy real8 lp={a['real8_lp']['alpha_xy_min']:.1f}", flush=True)
    res["wall_s"] = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robot_authority.json").write_text(json.dumps(res, indent=1, default=float))
    print("wall", res["wall_s"])


def retry():
    """re-run IK (more seeds) for poses marked unreachable in robot_authority.json; merge into the file."""
    f = OUT / "robot_authority.json"
    res = json.loads(f.read_text())
    rig = Rig()
    for key, v in res["poses"].items():
        if v.get("reachable"):
            continue
        style, name = key.split("/")
        tg = tray_grasp_targets(style, POSES[name])
        sols = {a: rig.ik_solutions(a, *tg[a], n_seed=700, seed=7, max_sol=4) for a in "LR"}
        ns = {k: len(x) for k, x in sols.items()}
        if not sols["L"] or not sols["R"]:
            v["ik_solutions_retry"] = ns
            print(key, "still UNREACHABLE (700 seeds)", ns, flush=True)
            continue
        best, bs = None, np.inf
        for ql in sols["L"]:
            for qr in sols["R"]:
                q = np.zeros(12); q[:6] = ql[:6]; q[6:] = qr[6:]
                cx = Ctx(rig, q, POSES[name], style)
                sc, _ = cx.solve(CAPS["real8"])
                if sc < bs:
                    best, bs = cx, sc
        a = analyse(best)
        a.update(reachable=True, ik_solutions=ns, tray_o=list(POSES[name]), found_on_retry=True)
        res["poses"][key] = a
        print(f"{key:18s} (retry) static s={a['real8_lp']['static_s']:.2f} a_h_min lp={a['real8_lp']['a_horiz_min']:.2f} ls={a['real8_ls']['a_horiz_min']:.2f}", flush=True)
    f.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
