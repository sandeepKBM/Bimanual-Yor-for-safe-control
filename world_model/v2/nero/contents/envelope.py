"""nero/contents/envelope.py -- payload / torque envelope of the NERO arms for objects with hidden contents.  SIMULATION ONLY.

Grasp = rigid weld once closed (ASSUMED, user 11-01).  Arms = the pan scene's two NERO 7-DoF arms (MJCF kinematics + inertia, MODEL).

Per arm i at a waypoint (q_i fixed), quasi-static (joint velocity terms dropped: ASSUMED, low speed):
    tau_i = G_i(q) + M_i(q) J_i^+ [a; 0] + J_i(q)^T w_i            w_i = [F_i; T_i] the wrench arm i applies to the object at p_i
Object (rigid, kept level, pure translation with acceleration a; o = object reference point):
    sum_i F_i                         = M_tot (a - g)
    sum_i T_i + (p_i - o) x F_i       = (m0 (c0 - o) + m (c - o)) x (a - g)
Two arms: the split of the wrench is free (closed chain) -> an LP:  u* = min over splits of max_j |tau_ij| / tau_lim_j and
|F_iz| / F_cap.  One arm: the wrench is fixed.  For fixed q and a, every constraint is affine in pi = (m, m c), so u*(pi) is CONVEX in
pi (LP value of a parametric RHS); likewise convex in a for fixed pi.  Hence "u* <= 1 on the whole contents set x acceleration box" holds
iff it holds at the products of their vertices: an exact corner check, guarantee = proof under MODEL (rigid weld, quasi-static, the
arm model, the listed waypoints).  NOT covered: configurations between waypoints, placement/pose error of the object (q is nominal),
joint friction, sensor error, dynamic slosh.  Those are sampled in sim or stated as conditions.

Limits (labels):
    tau_lim  16 Nm all joints  MEASURED-spec: AgileX pyAgxArm firmware_reference (fw >= 1.11) MIT t_ff COMMAND range; binding when the
                               arm is torque-commanded (our controller).  Not a motor rating (none published).  URDF 100 Nm = placeholder.
    F_cap    (3.0 kg - gripper mass) g per arm, vertical   MEASURED-spec: NERO user manual payload 3 kg ("controlled test environment").
    a_nom    0.5 m/s^2 box per axis   ASSUMED (cert/model_check.A_MAX)
    T_cap    1.30 Nm per claw, |T| (2-norm, isotropic)   ASSUMED (user-approved 11-02): the MEASURED 10-28 sim fit of claw pitch
             capacity vs squeeze (vizcoast/certify.CAPACITY_PTS, pan handle, 2.5..10 N, one run each) extrapolated linearly to 40 N.
             The weld hides this limit, so without it one arm "carries" a pot by one side handle.  Opt-in (T_cap=None = no row).
             DROPPED 11-02 pm (USER: "make grasp something we do not care about ... arm joint angles causing issues"): admission
             uses arm-side rows only (tau_lim, F_cap); T_CAP is kept only to reproduce the 11-02 am runs.
             In the LP the ball is replaced by an inner polytope (hull of 26 unit directions; conservative: u* can only rise).
    u_admit  0.9: admit only when the MODEL utilisation <= 0.9   ASSUMED (user-approved 11-02): margin for the controller's feedback
             torque above the model (11-01 smoke: sim 0.548 vs model 0.507 at the true contents, +8%); to be sized from sim runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import mujoco
import numpy as np
from scipy.optimize import linprog

from .. import adapters as A
from .. import config as C
from ..ik import ArmIK
from ..democ import kin as K
from . import objects as OB

G_VEC = np.array([0.0, 0.0, -OB.G])
TAU_LIM = C.tau_limits("mit_fw_ge_1.11")          # 16 Nm, see module doc
PAYLOAD_KG = C.MANUAL["payload_kg"]                # 3.0 kg, NERO manual
A_NOM = 0.5                                        # m/s^2, ASSUMED
T_CAP = 1.30                                       # Nm per claw, ASSUMED, DROPPED 11-02 pm (old-rule reproduction only; module doc)
U_ADMIT = 0.9                                      # ASSUMED admission margin on the MODEL utilisation (see module doc)
# waypoints: offsets of the object (and both TCPs) from the grasp pose [m]  (ASSUMED plan)
# carry is TOWARD the robot (-x): +x 10 cm is out of the right arm's reach at the pot grasp (IK error 105 mm, 11-01 probe)
WAYPOINTS = {"W0_liftoff": (0, 0, 0.0), "W1_lift5cm": (0, 0, 0.05), "W2_carry10cm": (-0.10, 0, 0.05), "W3_shelf25cm": (-0.10, 0, 0.25)}
IK_TOL = 1e-3                                     # m: a waypoint whose IK misses by more is UNREACHABLE for that arm
LIFTOFF = ("W0_liftoff", "W1_lift5cm")
CARRY = ("W0_liftoff", "W1_lift5cm", "W2_carry10cm")
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "contents_envelope_2026-11-01"


# ---------------------------------------------------------------------------------------------------------------------------
def capture(obj="pot", grasp_frac=0.85):
    """nominal grasp configuration(s) from the pan Episode (the robot's own pre-grasp IK), then waypoint IK (orientation held).
    -> dict(q={wp: {side: q}}, tcp={wp: {side: p}}, o0, R0, empty_mass, empty_com, gripper_mass, ...)"""
    from ..pan import hardworlds as HW, segments as SG
    th = {k: (v[0] + v[1]) / 2 for k, v in HW.THETA.items()}
    th.update(mass=0.02, c_r=0.0)
    ep = SG.Episode(HW.world_from_theta(th), grasp_frac=grasp_frac, obj=obj)
    m, d = ep.m, ep.d
    sides = ("left", "right") if obj == "pot" else ("left",)
    q, tcp = {}, {}
    for s in sides:
        pik = K.PoseIK(m, s)
        q0 = ep.q0[s].copy()
        p0, R0 = pik.ik.pose(q0)
        for wp, dl in WAYPOINTS.items():
            qq = q0.copy()
            for i in range(1, 61):                                  # continuation, as Episode._add_path
                qq, err = pik.solve(p0 + np.asarray(dl, float) * i / 60, R0, qq, q_post=q0, k_post=0.1, iters=60, tol=1e-9)
            q.setdefault(wp, {})[s] = qq.tolist()
            tcp.setdefault(wp, {})[s] = pik.ik.pose(qq)[0].tolist()
            if err > 1e-4:
                q[wp][s + "_ik_err"] = float(err)
    pan, lid = m.body("pan").id, m.body("lid").id
    ms = np.array([m.body_mass[pan], m.body_mass[lid]])
    com = (ms[:, None] * np.array([d.xipos[pan], d.xipos[lid]])).sum(0) / ms.sum()
    grip = float(m.body_subtreemass[m.body("left_arm_gripper_base").id])
    return dict(obj=obj, grasp_frac=grasp_frac, sides=list(sides), q=q, tcp=tcp, qpos_rest=d.qpos.tolist(),
                o0=d.xpos[pan].tolist(), R0=d.xmat[pan].reshape(3, 3).tolist(),
                empty_mass=float(ms.sum()), empty_com_world=com.tolist(), gripper_mass=grip)


class Arms:
    """G, M, J (at the TCP site) of each arm at each waypoint, from the MJCF with gravcomp off (the motors carry the arm)."""

    def __init__(self, cap, model=None):
        from ..pan import covered as CV
        self.m = model or CV.compose()
        m = self.m
        gc = m.body_gravcomp.copy()
        m.body_gravcomp[:] = 0.0
        self.terms = {}
        for wp, qs in cap["q"].items():
            d = mujoco.MjData(m)
            d.qpos[:] = np.asarray(cap["qpos_rest"])[: m.nq] if len(cap["qpos_rest"]) == m.nq else d.qpos
            for s in cap["sides"]:
                qa, _ = A.arm_addrs(m, s)
                d.qpos[qa] = qs[s]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            Mf = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, Mf, d.qM)
            for s in cap["sides"]:
                _, da = A.arm_addrs(m, s)
                site = m.site(f"{s}_arm_ee_grasp").id
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jp, jr, site)
                J = np.vstack([jp[:, da], jr[:, da]])
                self.terms[(wp, s)] = dict(G=d.qfrc_bias[da].copy(), M=Mf[np.ix_(da, da)].copy(), J=J,
                                           Jpinv=np.linalg.pinv(J), p=d.site_xpos[site].copy(),
                                           ik_err=float(qs.get(s + "_ik_err", 0.0)))
        m.body_gravcomp[:] = gc


def _twist_polytope():
    """H-rep (A, b) of an INNER polytope of the unit ball: hull of the 26 directions {-1,0,1}^3 normalised (all vertices on the
    sphere, so the hull lies inside the ball: conservative).  |T| <= cap is replaced by A T <= b * cap."""
    from scipy.spatial import ConvexHull
    P = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    h = ConvexHull(P)
    A, b = h.equations[:, :3], -h.equations[:, 3]
    _, idx = np.unique(np.round(np.c_[A, b], 9), axis=0, return_index=True)
    return A[idx], b[idx]


TWIST_A, TWIST_B = _twist_polytope()


def lever_split(Mtot, acc, S, o, pL, pR):
    """the LEVER split (contents/carry.LeverCtl commands it).  Mtot = total mass, acc = a - g, S = Mtot (c - o) the first moment about
    o (world axes).  Each claw carries the load by the lever rule along the grasp line e = pL - pR:
        F_L = ((S + Mtot (o - pR)) . e / |e|^2) acc,   F_R = Mtot acc - F_L
    and the moment left over (COM off the grasp line; with a != 0 also the inertial moment) is split equally as claw twist:
        T_L = T_R = (S x acc - (pL - o) x F_L - (pR - o) x F_R) / 2        (about each claw's own point)
    All linear in pi = (m, m c) -> the corner check stays exact.  -> (F_L, T_L, F_R, T_R)."""
    e = pL - pR
    F = Mtot * acc
    F_L = (float(np.dot(S + Mtot * (o - pR), e)) / float(np.dot(e, e))) * acc
    F_R = F - F_L
    T_res = np.cross(S, acc) - np.cross(pL - o, F_L) - np.cross(pR - o, F_R)
    return F_L, 0.5 * T_res, F_R, 0.5 * T_res


def _cross_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Envelope:
    """u*(pi, a) per waypoint and mode for one object.  pi = [m, m cx, m cy, m cz] with c in the WORLD-aligned object frame at o."""

    def __init__(self, spec: OB.ObjectSpec, arms: Arms, o0, R0, sides, tau_lim=TAU_LIM, payload_kg=PAYLOAD_KG, grip_mass=0.0,
                 offsets=None, T_cap=None):
        """offsets: {key: object offset from o0 [m]} for the keys of arms.terms (default WAYPOINTS).
        T_cap: per-claw twist capacity [Nm] (None = no twist row, the 11-01 runs)."""
        self.T_cap = T_cap
        self.spec, self.arms, self.sides = spec, arms, list(sides)
        self.offsets = offsets if offsets is not None else WAYPOINTS
        self.o0, self.R0 = np.asarray(o0, float), np.asarray(R0, float)
        self.tau_lim = np.asarray(tau_lim, float)
        self.F_cap = (payload_kg - grip_mass) * OB.G
        self.c0 = self.R0 @ spec.empty_com            # empty-object COM offset from o (world axes)

    def wrench(self, pi, a, o):
        m = pi[0]
        mc = self.R0 @ pi[1:4]                         # contents first moment, world axes
        acc = np.asarray(a, float) - G_VEC
        Mtot = self.spec.empty_mass + m
        F = Mtot * acc
        T = np.cross(self.spec.empty_mass * self.c0 + mc, acc)       # about o
        return F, T

    def u(self, wp, pi, a=(0, 0, 0), mode="two"):
        """-> (u*, detail) at one waypoint.  mode 'one' = left arm alone; 'two' = both arms, best split (LP: capability);
        'two_lever' = both arms, the lever split (lever_split; what contents/carry.LeverCtl commands);
        'two_share' = both arms, each takes half the load applied at the total COM (the FL-MPC payload model's split, democ/payload,
        share 0.5: what OUR controller commands; MODEL -- the feedback's share of a model error is not covered)."""
        o = self.o0 + np.asarray(self.offsets[wp], float)
        F, T = self.wrench(pi, a, o)
        a6 = np.concatenate([np.asarray(a, float), np.zeros(3)])
        sides = ["left"] if mode == "one" else self.sides
        tt = [self.arms.terms[(wp, s)] for s in sides]
        miss = [s for s, t in zip(sides, tt) if t["ik_err"] > IK_TOL]
        if miss:
            return float("inf"), dict(unreachable=miss)
        base = [t["G"] + t["M"] @ (t["Jpinv"] @ a6) for t in tt]
        if mode == "two_lever":
            m = pi[0]
            Mtot = self.spec.empty_mass + m
            S = self.spec.empty_mass * self.c0 + self.R0 @ pi[1:4]
            acc = np.asarray(a, float) - G_VEC
            pl, pr = (self.arms.terms[(wp, s)]["p"] for s in ("left", "right"))
            FL, TL, FR, TR = lever_split(Mtot, acc, S, o, pl, pr)
            taus, fz, tw = {}, {}, {}
            for s, t, b, (Fi, Ti) in zip(sides, tt, base, ((FL, TL), (FR, TR))):
                taus[s] = b + t["J"].T @ np.concatenate([Fi, Ti])
                fz[s], tw[s] = float(Fi[2]), float(np.linalg.norm(Ti))
            r = max(max(float(np.max(np.abs(v) / self.tau_lim)) for v in taus.values()), max(abs(v) for v in fz.values()) / self.F_cap)
            if self.T_cap:
                r = max(r, max(tw.values()) / self.T_cap)
            return float(r), dict(tau={s: v.round(3).tolist() for s, v in taus.items()}, Fz={s: round(v, 3) for s, v in fz.items()},
                                  twist={s: round(v, 3) for s, v in tw.items()})
        if mode == "two_share":
            taus, fz, tw = {}, {}, {}
            for s, t, b in zip(sides, tt, base):
                w = 0.5 * np.concatenate([F, T + np.cross(o - t["p"], F)])
                taus[s] = b + t["J"].T @ w
                fz[s] = w[2]
                tw[s] = float(np.linalg.norm(w[3:]))
            r = max(max(float(np.max(np.abs(v) / self.tau_lim)) for v in taus.values()), max(abs(v) for v in fz.values()) / self.F_cap)
            if self.T_cap:
                r = max(r, max(tw.values()) / self.T_cap)
            return float(r), dict(tau={s: v.round(3).tolist() for s, v in taus.items()}, Fz={s: round(float(v), 3) for s, v in fz.items()},
                                  twist={s: round(v, 3) for s, v in tw.items()})
        if mode == "one" or len(sides) == 1:
            t = tt[0]
            w = np.concatenate([F, T - np.cross(t["p"] - o, F)])
            tau = base[0] + t["J"].T @ w
            r_tau = np.abs(tau) / self.tau_lim
            r_pay = abs(w[2]) / self.F_cap
            tw = float(np.linalg.norm(w[3:]))
            r_tw = tw / self.T_cap if self.T_cap else 0.0
            return float(max(r_tau.max(), r_pay, r_tw)), dict(tau=tau.round(3).tolist(), Fz=round(float(w[2]), 3), twist=round(tw, 3),
                                                               worst_joint=int(np.argmax(r_tau)) + 1,
                                                               binding=["torque", "payload", "twist"][int(np.argmax([r_tau.max(), r_pay, r_tw]))])
        # LP over x = [w_L (6), w_R (6), t]; minimise t
        nx = 13
        Aeq = np.zeros((6, nx))
        for k, t in enumerate(tt):
            Aeq[0:3, 6 * k:6 * k + 3] = np.eye(3)
            Aeq[3:6, 6 * k + 3:6 * k + 6] = np.eye(3)
            Aeq[3:6, 6 * k:6 * k + 3] = _cross_mat(t["p"] - o)
        beq = np.concatenate([F, T])
        Aub, bub = [], []
        for k, t in enumerate(tt):
            JT = t["J"].T
            for j in range(7):
                row = np.zeros(nx)
                row[6 * k:6 * k + 6] = JT[j]
                for sg in (1, -1):                      # sg*(base + JT w) <= t*lim
                    r = sg * row.copy()
                    r[12] = -self.tau_lim[j]
                    Aub.append(r)
                    bub.append(-sg * base[k][j])
            for sg in (1, -1):
                r = np.zeros(nx)
                r[6 * k + 2] = sg
                r[12] = -self.F_cap
                Aub.append(r)
                bub.append(0.0)
            if self.T_cap:                              # inner polytope of the twist ball (conservative: u* can only rise)
                for af, bf in zip(TWIST_A, TWIST_B):
                    r = np.zeros(nx)
                    r[6 * k + 3:6 * k + 6] = af
                    r[12] = -bf * self.T_cap
                    Aub.append(r)
                    bub.append(0.0)
        c = np.zeros(nx)
        c[12] = 1.0
        res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=Aeq, b_eq=beq,
                      bounds=[(None, None)] * 12 + [(0, None)], method="highs")
        if res.status != 0:
            return float("inf"), dict(lp=res.message)
        x = res.x
        tau = [base[k] + tt[k]["J"].T @ x[6 * k:6 * k + 6] for k in range(len(tt))]
        return float(x[12]), dict(tau={s: tau[k].round(3).tolist() for k, s in enumerate(sides)},
                                  Fz={s: round(float(x[6 * k + 2]), 3) for k, s in enumerate(sides)},
                                  twist={s: round(float(np.linalg.norm(x[6 * k + 3:6 * k + 6])), 3) for k, s in enumerate(sides)})

    # ---- exact corner checks -------------------------------------------------------------------------------------------
    @staticmethod
    def accel_corners(a):
        return [np.array([sx, sy, sz]) * a for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)] if a > 0 else [np.zeros(3)]

    def worst(self, contents, wps, a=0.0, mode="two"):
        """max of u* over the contents vertices x acceleration-box corners x waypoints (= the max over the whole set: proof, MODEL).
        contents: an OB.Contents or an (n, 4) array of pi vertices [m, m cx, m cy, m cz] (object frame)."""
        best = (-1.0, None)
        V = contents.pi_vertices() if hasattr(contents, "pi_vertices") else np.asarray(contents, float)
        for wp in wps:
            for pi in V:
                for acc in self.accel_corners(a):
                    val, det = self.u(wp, pi, acc, mode)
                    if val > best[0]:
                        best = (val, dict(wp=wp, m=round(float(pi[0]), 4),
                                          c=(pi[1:4] / pi[0]).round(4).tolist() if pi[0] > 0 else None,
                                          a=acc.round(3).tolist(), **det))
        return best

    def m_max(self, contents: OB.Contents, wps, a=0.0, mode="two", m_cap=6.0, tol=1e-3, limit=1.0):
        """largest m_hi with worst u* <= limit (bisection; the feasible m along each vertex ray is an interval by convexity)."""
        if self.worst(contents.scaled(max(contents.m_lo, 1e-9)), wps, a, mode)[0] > limit:
            return 0.0 if contents.m_lo == 0 else None
        lo, hi = contents.m_lo, m_cap
        if self.worst(contents.scaled(hi), wps, a, mode)[0] <= limit:
            return hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.worst(contents.scaled(mid), wps, a, mode)[0] <= limit:
                lo = mid
            else:
                hi = mid
        return round(lo, 4)

    def a_max(self, contents: OB.Contents, wps, mode="two", a_cap=5.0, tol=1e-3, limit=1.0):
        if self.worst(contents, wps, 0.0, mode)[0] > limit:
            return None
        lo, hi = 0.0, a_cap
        if self.worst(contents, wps, hi, mode)[0] <= limit:
            return hi
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.worst(contents, wps, mid, mode)[0] <= limit:
                lo = mid
            else:
                hi = mid
        return round(lo, 4)

    def action(self, contents: OB.Contents, a_nom=A_NOM, modes=("one", "two"), limit=1.0, carry=CARRY, liftoff=LIFTOFF):
        """the action rule (ASSUMED policy, raised to the user): first that holds for ALL contents in the set (exact corner check):
        one arm at a_nom -> two arms at a_nom -> two arms slowed to a* < a_nom -> set down (lift-off holds, carry does not) -> refuse.
        Open-content limits (spill, lid) cap a* for every mode."""
        a_open = min([ol.a_max for ol in self.spec.open_limits], default=np.inf)
        rows = {}
        for md in modes:
            rows[md] = dict(carry_nom=self.worst(contents, carry, a_nom, md)[0], carry_static=self.worst(contents, carry, 0.0, md)[0],
                            liftoff_static=self.worst(contents, liftoff, 0.0, md)[0])
        for md in modes:
            if rows[md]["carry_nom"] <= limit and a_nom <= a_open:
                return dict(action="one arm" if md == "one" else "two arms", mode=md, a_star=None, rows=rows)
        for md in modes:
            if rows[md]["carry_static"] <= limit:
                a_t = self.a_max(contents, carry, md, limit=limit)
                return dict(action=f"{'one arm' if md == 'one' else 'two arms'}, slow", mode=md,
                            a_star=round(min(a_t, a_open), 4), rows=rows)
        for md in modes:
            if rows[md]["liftoff_static"] <= limit:
                return dict(action="set down (lift-off holds, carry does not)", mode=md, a_star=None, rows=rows)
        return dict(action="refuse", mode=None, a_star=None, rows=rows)


# ---------------------------------------------------------------------------------------------------------------------------
def objects_from(cap_pot, cap_pan):
    """ObjectSpecs in each object's frame (o = pan body origin, axes = pan body axes; tray: o = midpoint of the pot TCPs)."""
    out = {}
    for name, cap in (("pan", cap_pan), ("pot", cap_pot)):
        R0, o0 = np.asarray(cap["R0"]), np.asarray(cap["o0"])
        com_local = R0.T @ (np.asarray(cap["empty_com_world"]) - o0)
        out[name] = (OB.pan_like(name, cap["empty_mass"], com_local), cap, o0, R0)
    tl, tr = np.asarray(cap_pot["tcp"]["W0_liftoff"]["left"]), np.asarray(cap_pot["tcp"]["W0_liftoff"]["right"])
    o_tray = 0.5 * (tl + tr) - np.array([0, 0, 0.022])  # MODEL approx: tray base 2.2 cm below the handle grasp point (bar top flush with rim)
    out["coffee_tray"] = (OB.coffee_tray(), cap_pot, o_tray, np.eye(3))
    return out


def _job(args):
    """one (object, torque preset): all thresholds, the action on the full box and the action map.  Pool worker."""
    name, pr, caps = args
    objs = objects_from(caps["pot"], caps["pan"])
    spec, cap, o0, R0 = objs[name]
    A_ = Arms(caps["pan"] if name == "pan" else caps["pot"])
    modes = ("one",) if name == "pan" else ("one", "two")
    env = Envelope(spec, A_, o0, R0, cap["sides"], tau_lim=C.tau_limits(pr), grip_mass=cap["gripper_mass"])
    empty = OB.Contents(0.0, 0.0, spec.contents.P)
    centre = OB.Contents(spec.contents.m_lo, spec.contents.m_hi, np.array([[0.0, 0.0, float(spec.contents.P[:, 2].mean())]]))
    full = spec.contents.scaled(spec.contents.m_lo)
    rec = dict(u_empty={md: env.worst(empty, CARRY, 0.0, md) for md in modes})
    for md in modes:
        rec[f"m_max_{md}"] = dict(
            centre_static=env.m_max(centre.scaled(spec.contents.m_lo), CARRY, 0.0, md),
            worst_com_static=env.m_max(full, CARRY, 0.0, md),
            worst_com_a_nom=env.m_max(full, CARRY, A_NOM, md),
            liftoff_worst_com_static=env.m_max(full, LIFTOFF, 0.0, md),
            shelf_worst_com_static=env.m_max(full, ("W3_shelf25cm",), 0.0, md))
        rec[f"worst_{md}_full_box_a_nom"] = env.worst(spec.contents, CARRY, A_NOM, md)
    rec["action_full_box"] = env.action(spec.contents, modes=modes)
    grid = np.round(np.linspace(spec.contents.m_lo, max(spec.contents.m_hi, 2.5), 9), 3)
    rec["action_map"] = [dict(m_hi=float(mh), **{k: v for k, v in env.action(spec.contents.scaled(float(mh)), modes=modes).items()
                                                if k in ("action", "a_star")}) for mh in grid]
    return name, pr, rec


def run(out_dir=RESULTS, presets=("mit_fw_ge_1.11", "urdf_effort"), procs=6):
    import multiprocessing as mp
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capf = out_dir / "capture.json"
    if capf.exists():
        caps = json.loads(capf.read_text())
    else:
        caps = dict(pot=capture("pot"), pan=capture("pan"))
        capf.write_text(json.dumps(caps, indent=1))
    res = dict(assumptions=dict(
        grasp="ASSUMED rigid weld once closed (user 11-01; wrist cameras + low-level control handle grasping)",
        quasi_static="ASSUMED: joint-velocity (Coriolis) terms dropped; translation only, object level",
        tau_lim="MEASURED-spec 16 Nm (MIT t_ff command range, fw>=1.11); urdf_effort 100 Nm = placeholder (sensitivity only)",
        payload=f"MEASURED-spec {PAYLOAD_KG} kg (NERO manual) minus gripper {caps['pot']['gripper_mass']:.3f} kg (MJCF), vertical force",
        a_nom=f"ASSUMED {A_NOM} m/s^2 per axis (box)", waypoints=f"ASSUMED plan offsets {WAYPOINTS}; IK miss > {IK_TOL} m = unreachable",
        q="MODEL: nominal placement, the robot's own pre-grasp IK (segments.Episode, grasp_frac 0.85) + continuation IK",
        guarantee="proof under MODEL over the contents set x acceleration box at the listed waypoints; between waypoints and "
                  "placement error NOT covered"), objects={})
    objs = objects_from(caps["pot"], caps["pan"])
    for name, (spec, cap, o0, R0) in objs.items():
        res["objects"][name] = dict(empty_mass=spec.empty_mass, contents=dict(m=(spec.contents.m_lo, spec.contents.m_hi),
                                                                              labels=spec.contents.labels),
                                    open_limits=[dict(name=o.name, a_max=round(o.a_max, 3), formula=o.formula, labels=o.labels)
                                                 for o in spec.open_limits],
                                    sim_box=spec.sim_box, plausible=spec.plausible, presets={})
    jobs = [(n, pr, caps) for n in objs for pr in presets]
    with mp.get_context("spawn").Pool(min(procs, len(jobs)), maxtasksperchild=1) as pool:
        for name, pr, rec in pool.imap_unordered(_job, jobs):
            res["objects"][name]["presets"][pr] = rec
            print(f"done {name} {pr}", flush=True)
    (out_dir / "RESULT.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = run(a.out, procs=a.procs)
    print(json.dumps({k: {p: dict(action=v["presets"][p]["action_full_box"]["action"],
                                  a_star=v["presets"][p]["action_full_box"]["a_star"],
                                  m_max={kk: vv for kk, vv in v["presets"][p].items() if kk.startswith("m_max")})
                          for p in v["presets"]} for k, v in res["objects"].items()}, indent=1), flush=True)
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
