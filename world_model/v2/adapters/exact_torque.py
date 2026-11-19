"""adapters/exact_torque.py -- exact inverse-dynamics torque row of a planned ClosedChainPlan (checker step K10).  SIMULATION ONLY.

OPT-IN: nothing registers or calls this module (decisions D5 torque model / split and D8 object rotation are open with the user).  The legacy
quasi-static row stays stack/checks/payload.torque_row over nero/contents/envelope.Envelope.

Model (MODEL = the MJCF arms, a rigid weld, the plan's own samples):
    tau_s(t, pi10) = ID_s(q, qd, qdd) + J_s(q)^T w_s(t, pi10)                 s in {left, right}
ID_s   = mujoco.mj_inverse on the plan's scene model with BOTH arms set to the plan's (q, qd, qdd), arm block s; constraints (contacts, limits, joint
         friction) and passive forces (damping) DISABLED for the call and gravcomp zeroed (the motors carry the arm), exactly the legacy `Arms` terms.  The
         arm links include the gripper; the carried object is NOT in the ID, it is the load wrench w_s.  Free bodies of the scene do not couple to an arm block.
w_s    = the wrench arm s applies to the object, at its TCP site (J_s = mj_jacSite of `<s>_arm_ee_grasp`, world axes, rows [lin; ang]).
Load (Newton-Euler about o, world axes; o = object reference point, o(t) = o_ref + plan offset p(t); object pose R, omega, alpha, a_o = plan pdd):
    S1 = m_e R c_e + R (m c)                 first moment of the whole object about o            (pi: m, m c)
    S  = R S_c R^T,  I_o = R I_e R^T + tr(S) 1 - S      inertia about o (S_c = int rho x x^T of the contents, object frame; pi: 6 entries)
    F  = (m_e + m)(a_o - g) + alpha x S1 + omega x (omega x S1)
    T  = I_o alpha + omega x (I_o omega) + S1 x (a_o - g)                                       (moment about o)
pi10 = (m, m c [3], S_c [6: xx yy zz xy xz yz]).  F, T are affine in pi10 for a fixed motion.

Split of (F, T) between the two grasps (`split`):
  "legacy"  the LEVER split (envelope.lever_split; what contents/carry.LeverCtl commands):  F_L = s_L (a_o - g) + 0.5 F_x,  F_R = F - F_L, with
            s_L = ((S1 + M_tot (o - p_R)) . e)/|e|^2, e = p_L - p_R, F_x = alpha x S1 + omega x (omega x S1) (the rotation force);  the leftover moment
            T - (p_L - o) x F_L - (p_R - o) x F_R is split equally as claw twist, T_L = T_R = half.  GUESS: the legacy law has no rotation term; the
            rotation force F_x is split 50/50 here because the lever fraction times F_x would be bilinear in pi (breaks the vertex proof).  Equals the legacy
            law exactly when omega = alpha = 0.
  "share"   both arms take half of the load applied at the total COM (envelope two_share):  w_s = 0.5 [F, T + (o - p_s) x F].
AFFINITY (needed for the vertex proof): tau is affine in pi10 for fixed (q, qd, qdd, R, omega, alpha, a_o) ONLY IF the split is linear in the load wrench
(F, T).  Both laws above are linear in (pi10 -> F, T); a best-split LP or a law with a data-dependent division is NOT, and must not be passed here.

Contents set in pi10 (the "outer box", checker design 3.1): (m, m c) = the legacy vertices (m in {lo, hi}) x (c in the vertices of P); the second moments
S_c,ij lie in m [smin_ij, smax_ij], smin/smax = min/max of x_i x_j over hull(P) (exact: diag via the coordinate interval, off-diagonal via all vertex-pair
segments; MODEL: the contents mass lies inside hull(P), P = the COM polytope, which for the shipped objects is the interior volume).  The set
{(m, m c, S_c)} is inside the polytope whose vertices are every (m, m v, m s) with s a corner of the box; the max of an affine function over it is
evaluated per (m, v) vertex with the box maximum in closed form (sum of the larger of coef*lo, coef*hi), so no 64-corner enumeration.
GUARANTEE: proof under MODEL for the listed samples (between samples, tracking feedback torque, joint friction, impacts, slosh NOT covered).  The box
couples (m, m c) and S_c loosely, so the row can be conservative; with omega = alpha = 0 the S_c terms vanish and the row is the exact vertex check.

Limits: utilisation u = max over samples, vertices, joints of |tau_j| / tau_limit_j, tau_limit = config.tau_limit() (7,);  ok iff u <= payload.U_ADMIT.
The per-claw vertical payload (F_cap) is NOT in u (the legacy row folds it in); it is reported in worst["u_payload"] for comparison only.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from world_model.v2.nero import adapters as NA
from world_model.v2.nero import config as C
from world_model.v2.nero.contents import objects as OB
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks.row import Row
from world_model.v2.stack.planners import closed_chain as CC

SIDES = ("left", "right")
G_VEC = np.array([0.0, 0.0, -OB.G])
_SIJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))          # order of the 6 second-moment entries in pi10[4:10]
NAME = "torque_exact"


# ----------------------------------------------------------------------------------------------------------------------- contents set
def _split_box(contents_box, empty_mass, empty_com):
    """-> (me, ce, Contents).  contents_box: an ObjectSpec (empty_mass / empty_com taken from it) or a bare Contents (pass empty_mass, empty_com)."""
    if hasattr(contents_box, "contents"):
        return float(contents_box.empty_mass), np.asarray(contents_box.empty_com, float), contents_box.contents
    if empty_mass is None or empty_com is None:
        raise ValueError("a bare Contents needs empty_mass= and empty_com=")
    return float(empty_mass), np.asarray(empty_com, float), contents_box


def second_moment_bounds(P):
    """(smin, smax) (6,): min / max over hull(P) of x_i x_j for (i, j) in _SIJ (MODEL: second moment per unit contents mass is inside this box)."""
    P = np.asarray(P, float).reshape(-1, 3)
    lo, hi = np.zeros(6), np.zeros(6)
    idx = [(a, b) for a in range(len(P)) for b in range(a, len(P))]
    for k, (i, j) in enumerate(_SIJ):
        if i == j:
            xl, xh = P[:, i].min(), P[:, i].max()
            lo[k] = 0.0 if xl <= 0.0 <= xh else min(xl * xl, xh * xh)
            hi[k] = max(xl * xl, xh * xh)
        else:
            vals = list(P[:, i] * P[:, j])
            for a, b in idx:                                   # extremum of the quadratic x_i(t) x_j(t) on every vertex-pair segment
                d = P[b] - P[a]
                qa = d[i] * d[j]
                if abs(qa) > 1e-15:
                    tt = -(P[a, i] * d[j] + P[a, j] * d[i]) / (2.0 * qa)
                    if 0.0 < tt < 1.0:
                        x = P[a] + tt * d
                        vals.append(x[i] * x[j])
            lo[k], hi[k] = min(vals), max(vals)
    return lo, hi


def pi10_vertices(contents, rotating=True):
    """-> V (nv, 4) of (m, m v) vertices and (lo, hi) (nv, 6) the second-moment box per vertex (m * [smin, smax]; zero width when rotating=False)."""
    V = contents.pi_vertices()
    smin, smax = second_moment_bounds(contents.P)
    m = V[:, :1]
    if not rotating:
        z = np.zeros((len(V), 6))
        return V, z, z
    return V, m * smin[None], m * smax[None]


# ----------------------------------------------------------------------------------------------------------------------- load wrench
def _smat(s6):
    xx, yy, zz, xy, xz, yz = s6
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def load_wrench(pi10, k, me, ce, I_e, split):
    """(F_L, T_L, F_R, T_R) the wrenches the two grasps apply to the object (world axes; torque about each grasp point), for pi10 and the motion/pose dict k:
    R, w, al, ao (a_o), o, pl, pr.  Affine in pi10 (see the module docstring)."""
    R, w, al, ao, o, pl, pr = k["R"], k["w"], k["al"], k["ao"], k["o"], k["pl"], k["pr"]
    m = pi10[0]
    S1 = me * (R @ ce) + R @ pi10[1:4]
    Sw = R @ _smat(pi10[4:10]) @ R.T
    Io = R @ I_e @ R.T + np.trace(Sw) * np.eye(3) - Sw
    Mtot = me + m
    acc = ao - G_VEC
    Fx = np.cross(al, S1) + np.cross(w, np.cross(w, S1))
    F = Mtot * acc + Fx
    T = Io @ al + np.cross(w, Io @ w) + np.cross(S1, acc)
    if split == "share":
        return 0.5 * F, 0.5 * (T + np.cross(o - pl, F)), 0.5 * F, 0.5 * (T + np.cross(o - pr, F))
    if split != "legacy":
        raise ValueError(f"split {split!r}: 'legacy' (lever) or 'share'")
    e = pl - pr
    sL = float(np.dot(S1 + Mtot * (o - pr), e)) / float(np.dot(e, e))
    FL = sL * acc + 0.5 * Fx
    FR = F - FL
    Tres = T - np.cross(pl - o, FL) - np.cross(pr - o, FR)
    return FL, 0.5 * Tres, FR, 0.5 * Tres


# ----------------------------------------------------------------------------------------------------------------------- arm side
def _arm_terms(m, d, q, qd, qdd, sites, addr):
    """ID block, jacobian and TCP position of both arms at one sample (constraints / passive already disabled on m)."""
    for s in SIDES:
        qa, da = addr[s]
        d.qpos[qa], d.qvel[da], d.qacc[da] = q[s], qd[s], qdd[s]
    mujoco.mj_inverse(m, d)
    tau, J, p = {}, {}, {}
    for s in SIDES:
        _, da = addr[s]
        tau[s] = d.qfrc_inverse[da].copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sites[s])
        J[s] = np.vstack([jp[:, da], jr[:, da]])
        p[s] = d.site_xpos[sites[s]].copy()
    return tau, J, p


class _Model:
    """context manager: the plan's scene model with constraints + passive forces disabled and gravcomp zeroed (restored on exit)."""

    def __init__(self, m):
        self.m = m

    def __enter__(self):
        m = self.m
        self._flags, self._gc = m.opt.disableflags, m.body_gravcomp.copy()
        m.opt.disableflags = int(self._flags) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
        m.body_gravcomp[:] = 0.0
        return m

    def __exit__(self, *a):
        self.m.opt.disableflags = self._flags
        self.m.body_gravcomp[:] = self._gc


def _piece_samples(plan, sg, zero_motion, rot):
    pc = plan.pieces[sg]
    n = pc.p.shape[0]
    q = {s: np.asarray(pc.q[s], float) for s in SIDES}
    if zero_motion:
        qd = {s: np.zeros_like(q[s]) for s in SIDES}
        qdd = {s: np.zeros_like(q[s]) for s in SIDES}
        ao = np.zeros((n, 3))
    else:
        qd = {s: np.asarray(pc.qd[s], float) for s in SIDES}
        qdd = {s: np.asarray(pc.qdd[s], float) for s in SIDES}
        ao = np.asarray(pc.pdd, float)
    if rot is not None and sg in rot and not zero_motion:
        R, w, al = (np.asarray(x, float) for x in rot[sg])
    else:
        R, w, al = np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.zeros((n, 3))
    t = np.arange(n) * CC.DT
    return dict(t=t, q=q, qd=qd, qdd=qdd, p=np.asarray(pc.p, float), ao=ao, R=R, w=w, al=al)


def _evaluate(plan, contents_box, pieces, split, zero_motion, empty_mass, empty_com, empty_inertia, rot, o_ref, tau_lim, rotating):
    me, ce, cont = _split_box(contents_box, empty_mass, empty_com)
    I_e = np.zeros((3, 3)) if empty_inertia is None else np.asarray(empty_inertia, float)
    if o_ref is None:
        from world_model.v2.adapters import payload_envelope as PE
        o_ref = PE.o_ref()
    o_ref = np.asarray(o_ref, float)
    names = list(pieces) if pieces else list(plan.pieces)
    Vpi, Slo, Shi = pi10_vertices(cont, rotating)
    F_cap = None
    best = dict(u=-1.0)
    u_pay = 0.0
    m = plan.inp.m
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    sites = {s: m.site(f"{s}_arm_ee_grasp").id for s in SIDES}
    addr = {s: NA.arm_addrs(m, s) for s in SIDES}
    n_samples = 0
    eye10 = np.eye(10)
    with _Model(m):
        for sg in names:
            sm = _piece_samples(plan, sg, zero_motion, rot)
            n = len(sm["t"])
            tau0 = np.zeros((n, 14))
            Btau = np.zeros((n, 14, 10))
            Bf = np.zeros((n, 2, 10))                                           # vertical grasp force per claw (payload, reported only)
            f0 = np.zeros((n, 2))
            for i in range(n):
                tau, J, p = _arm_terms(m, d, {s: sm["q"][s][i] for s in SIDES}, {s: sm["qd"][s][i] for s in SIDES},
                                       {s: sm["qdd"][s][i] for s in SIDES}, sites, addr)
                k = dict(R=sm["R"][i], w=sm["w"][i], al=sm["al"][i], ao=sm["ao"][i], o=o_ref + sm["p"][i], pl=p["left"], pr=p["right"])
                w0 = load_wrench(np.zeros(10), k, me, ce, I_e, split)
                cols = [load_wrench(eye10[j], k, me, ce, I_e, split) for j in range(10)]
                for a, s in enumerate(SIDES):
                    wv0 = np.concatenate([w0[2 * a], w0[2 * a + 1]])
                    tau0[i, 7 * a:7 * a + 7] = tau[s] + J[s].T @ wv0
                    f0[i, a] = wv0[2]
                    for j in range(10):
                        wj = np.concatenate([cols[j][2 * a], cols[j][2 * a + 1]]) - wv0
                        Btau[i, 7 * a:7 * a + 7, j] = J[s].T @ wj
                        Bf[i, a, j] = wj[2]
            n_samples += n
            # max over the vertices: per (m, v) vertex the affine part in (m, m c) plus the closed-form box maximum over the S_c entries
            lim = np.concatenate([tau_lim, tau_lim])
            for vi in range(len(Vpi)):
                base = tau0 + Btau[:, :, :4] @ Vpi[vi]                                          # (n, 14)
                cS = Btau[:, :, 4:10]                                                           # (n, 14, 6)
                up = base + np.maximum(cS * Slo[vi], cS * Shi[vi]).sum(-1)                      # max over the box of +tau
                lo = base + np.minimum(cS * Slo[vi], cS * Shi[vi]).sum(-1)                      # min of tau -> max of -tau is -lo
                util = np.maximum(up, -lo) / lim
                flat = int(np.argmax(util))
                if util.flat[flat] > best["u"]:
                    ii, jj = divmod(flat, 14)
                    best = dict(u=float(util.flat[flat]), piece=sg, i=int(ii), t=float(sm["t"][ii]), side=SIDES[jj // 7], joint=int(jj % 7) + 1,
                                tau=float(up[ii, jj] if up[ii, jj] >= -lo[ii, jj] else lo[ii, jj]), m=float(Vpi[vi, 0]),
                                c=(Vpi[vi, 1:4] / Vpi[vi, 0]).round(4).tolist() if Vpi[vi, 0] > 0 else None)
                fz = f0 + Bf[:, :, :4] @ Vpi[vi]
                fz = np.maximum(np.abs(fz + np.maximum(Bf[:, :, 4:10] * Slo[vi], Bf[:, :, 4:10] * Shi[vi]).sum(-1)),
                                np.abs(fz + np.minimum(Bf[:, :, 4:10] * Slo[vi], Bf[:, :, 4:10] * Shi[vi]).sum(-1)))
                u_pay = max(u_pay, float(fz.max()))
    best["n_samples"] = n_samples
    best["n_vertices"] = int(len(Vpi))
    best["u_payload"] = u_pay / ((C.MANUAL["payload_kg"] - _grip_mass(m)) * OB.G)
    return best, names


def _grip_mass(m):
    try:
        return float(m.body_subtreemass[m.body("left_arm_gripper_base").id])
    except Exception:
        return 0.0


# ----------------------------------------------------------------------------------------------------------------------- public API
def exact_torque_row(plan, contents_box, pieces=None, split="legacy", *, empty_mass=None, empty_com=None, empty_inertia=None, rot=None, o_ref=None,
                     tau_limit=None, name=NAME) -> Row:
    """exact-ID torque row over every sample of every piece of a closed_chain plan (module docstring: model, split, contents outer box).
    contents_box: ObjectSpec (preferred) or Contents (+ empty_mass=, empty_com=).  empty_inertia: 3x3 inertia of the EMPTY object about o, object frame
    (GUESS: default zeros, ObjectSpec has none; only matters when rot is given).  rot: {piece: (R (n,3,3), omega (n,3), alpha (n,3))} object rotation per
    sample (default: the plan holds the tool orientation, so R = I, omega = alpha = 0 -- D8).  -> Row(name, ok iff u <= payload.U_ADMIT, margin = U_ADMIT - u)."""
    t0 = time.time()
    tl = np.asarray(C.tau_limit() if tau_limit is None else tau_limit, float)
    best, names = _evaluate(plan, contents_box, pieces, split, False, empty_mass, empty_com, empty_inertia, rot, o_ref, tl, rotating=True)
    return _row(name, best, names, split, False, rot is not None, empty_inertia is not None, time.time() - t0)


def _row(name, best, names, split, zero, rot_given, inertia_given, wall):
    u = best["u"]
    ok = bool(u <= PAY.U_ADMIT)
    worst = {k: v for k, v in best.items()}
    worst.update(u_max=float(u), split=split, pieces=names, wall_s=round(wall, 2))
    assumptions = dict(
        model=("proof under MODEL (MJCF arms, mj_inverse with constraints/passive off, rigid weld, plan samples only; tau affine in pi10 for a split linear in the "
               "load wrench; contents second moments in the entrywise outer box of hull(P)); not covered: between samples, tracking feedback, friction, impacts, slosh"),
        split=f"{split} (GUESS for rotation terms: rotation force split 50/50; identical to the legacy law at omega = alpha = 0)",
        motion=("qd = qdd = 0, object acceleration 0 (legacy-equivalent)" if zero else "plan qd, qdd, pdd (TOPP-RA, stored in the piece)"),
        object_rotation=("given by caller (rot)" if rot_given else "R = I, omega = alpha = 0 (the plan holds the tool orientation; D8 open)"),
        empty_inertia=("given by caller" if inertia_given else "ASSUMED 0 (ObjectSpec has no empty-body inertia; irrelevant while omega = alpha = 0)"),
        tau_limit=f"config.tau_limit() = {float(np.max(C.tau_limit())):g} Nm (MODEL)", u_admit=f"{PAY.U_ADMIT} (payload.U_ADMIT)")
    return Row(name, ok, margin=float(PAY.U_ADMIT - u), unit="utilisation", guarantee="proof", worst=worst, assumptions=assumptions,
               why="" if ok else f"exact-ID utilisation {u:.3f} > {PAY.U_ADMIT}")
