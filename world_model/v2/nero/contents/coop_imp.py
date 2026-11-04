"""nero/contents/coop_imp.py -- cooperative OBJECT-LEVEL impedance carry controller ("coopimp"), SIMULATION ONLY.

Opt-in variant (CarryEpisode(..., ctl="coopimp")); LeverCtl + FL-MPC stay the default.  After Caccavale, Chiacchio, Marino, Villani,
"Six-DOF Impedance Control of Dual-Arm Cooperative Manipulators", IEEE/ASME T-Mech 13(5), 2008.  NOTE: the PDF could not be fetched
(sandbox has no network), so the law below follows the paper's structure from memory, NOT its text.  Deviations: (1) the internal
loop is a force-level PI on the measured internal wrench (not the paper's internal impedance on the relative pose: under the rigid weld
the relative pose is locked); (2) object error is computed from the sim object pose (= claws' FK under the weld, MODEL), gains diagonal
in the WORLD frame (object is held level); (3) torque-level arm inverse dynamics instead of the paper's task-space motion loop.

Law (per control step, world frame, W = [W_L W_R], W_i = [[I,0],[S(r_i),I]], r_i = p_i - p_o, h = [F_L;T_L;F_R;T_R] wrench ON the object,
T about each claw TCP):
  acc    = a_d - g                                  S = M_hat R c_nom   (belief load, same as LeverCtl)
  h_o*   = [M_hat acc ; S x acc] + [Kp e_p + Dp e_v ; Ko e_R + Do e_w]       e_R = 1/2 sum_k r_k x rd_k (object columns vs desired)
  h_I    = (I - W^+ W) h_meas                       h_meas = joint-torque estimate pinv(J^T)(tau_applied - tau_model) (11-03: the real NERO has no wrist F/T sensor; see _h_hat); BIMANUAL_LEGACY_COOP_SENSOR=1 =
                     the sim-only weld constraint wrench (carry.claw_wrench), 2 steps old max
  h_I*   = h_I,des - kP_I (h_I - h_I,des) - kI_I int (h_I - h_I,des) dt        (h_I,des = f_sq along the grasp line, equal and opposite)
  h*     = W^+ h_o* + (I - W^+ W) h_I*
  qdd_i  = J_i^+ (a_cl,i - Jdot_i qdot_i) + N_i [Kn (q_ref,i - q_i) + Dn (qd_ref,i - qdot_i)]      (N_i = I - J_i^+ J_i; arm-only)
  tau_i  = M_i qdd_i + (qfrc_bias - qfrc_passive)_i + J_i^T h_i*       then the caller clips to the 16 Nm box (+ FRIC_COMP if on)
  (a_d, p_d, v_d, w_d, a_cl,i: from the claw reference joint paths by FK; central differences with stencil H_FD.)
The arm-arm clearance rows of FL-MPC are NOT present (no QP); the carry plan keeps the arms apart.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import carry as CR

SIDES = CR.SIDES
# --- gains: all ASSUMED (tuned only enough to be stable on the smoke worlds; see the report) ---
GAINS = dict(
    Kp=500.0,       # ASSUMED object position stiffness [N/m] (2000 saturated/unstable on a 1 kg box)
    Dp=80.0,       # ASSUMED [N s/m]
    Ko=100.0,       # ASSUMED object orientation stiffness [Nm/rad] (400 unstable)
    Do=10.0,       # ASSUMED [Nm s/rad]
    f_sq=(5.0 if CR.C.LEGACY_COOP_SQUEEZE else 0.0),   # USER 11-03: 0 N like Caccavale 2008 (internal forces "must be avoided"); BIMANUAL_LEGACY_COOP_SQUEEZE=1 = the old 5 N ASSUMED. Desired squeeze along the grasp line [N]
    kP_I=0.3,       # ASSUMED internal wrench P gain (dimensionless)
    kI_I=2.0,       # ASSUMED internal wrench I gain [1/s]
    I_max=40.0,     # ASSUMED clamp on the internal integrator (per component, N or Nm)
    Kn=64.0, Dn=16.0,   # ASSUMED null-space posture PD [1/s^2], [1/s]
    wtd=0.0,        # ASSUMED choice: 1 = torque-weighted generalized inverse W^# (Q_i = J_i J_i^T; no gain over pinv on the 5 kg smoke), 0 = plain pinv
    alloc=0.0,      # ASSUMED choice: 1 = internal setpoint chosen to balance joint torques (tried: no gain on the 5 kg smoke; 0 = the 5 N squeeze only)
    mu_alloc=1e-4,  # ASSUMED regularisation of the allocation LS (toward the squeeze setpoint)
    lam=1e-4,       # MODEL damping of J^+
    H_FD=0.05,      # ASSUMED finite-difference stencil for the reference accelerations [s]
    eps_jd=1e-4,    # numerical Jdot qdot step [s]
)
_SCR = {}


def _scratch(m):
    if id(m) not in _SCR:
        _SCR[id(m)] = mujoco.MjData(m)
    return _SCR[id(m)]


def grasp_matrix(p_cl, p_o):
    """W (6x12) for claws (L, R) with wrench about the claw TCP; h_o = W h."""
    W = np.zeros((6, 12))
    for i, p in enumerate(p_cl):
        r = np.asarray(p, float) - np.asarray(p_o, float)
        sk = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        W[:3, 6 * i:6 * i + 3] = np.eye(3)
        W[3:, 6 * i:6 * i + 3] = sk
        W[3:, 6 * i + 3:6 * i + 6] = np.eye(3)
    return W


def rot_err(R, Rd):
    """orientation error vector (world), 1/2 sum_k r_k x rd_k."""
    return 0.5 * sum(np.cross(R[:, k], Rd[:, k]) for k in range(3))


def rot_vec(R):
    """log map of a rotation matrix (axis * angle)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * v if th < 1e-8 else v * th / (2 * np.sin(th))


def object_wrench(Mhat, S, acc, gains, e_p, e_v, e_R, e_w, a_d_unused=None):
    """h_o* (6,) = static/inertial wrench of the belief load + object-level impedance."""
    F = Mhat * acc + gains["Kp"] * e_p + gains["Dp"] * e_v
    T = np.cross(S, acc) + gains["Ko"] * e_R + gains["Do"] * e_w
    return np.concatenate([F, T])


def weighted_inverse(W, Q=None):
    """W^# = Q^-1 W^T (W Q^-1 W^T)^-1 (Q = I -> pinv); W W^# = I, so W h = h_o for h = W^# h_o; (I - W^# W) h is internal."""
    if Q is None:
        return np.linalg.pinv(W)
    Qi = np.linalg.inv(Q)
    return Qi @ W.T @ np.linalg.inv(W @ Qi @ W.T)


def split_wrench(W, h_o, h_I_cmd, Q=None):
    """h* = W^# h_o + (I - W^# W) h_I."""
    Wp = weighted_inverse(W, Q)
    return Wp @ h_o + (np.eye(12) - Wp @ W) @ h_I_cmd


class CoopImpCtl(CR.LeverCtl):
    SAFE = False                          # True in CoopImpSafeCtl: wrench goes to FL-MPC as payload feedforward (see below)

    def __init__(self, *a, gains=None, **kw):
        super().__init__(*a, **kw)
        self.g = dict(GAINS, **(gains or {}))
        self.ref_in = None                    # set by CarryEpisode._step: dict(qr={s:(3,7) q at t-H,t,t+H}, h_meas=(12,))
        self.rel = None                       # object pose in each claw frame (captured at the first call)
        self.I_int = np.zeros(12)
        self.log = {}
        self.qd_prev = None                   # encoder velocities at the previous control step (for the joint-torque wrench estimate)
        self._rng = np.random.default_rng(0)  # joint-torque noise (config.COOP_TAU_NOISE_NM; 0 = off)
        self.err_hist = []                    # per step (|dF| [N], |dT| [Nm], |F_true| [N], |T_true| [Nm]) max over the two claws, diagnostics only

    # ---- helpers ----
    def _fk(self, d, s, q):
        sd = _scratch(self.m)
        sd.qpos[:] = d.qpos
        sd.qpos[self.lq if s == "left" else self.rq] = q
        mujoco.mj_kinematics(self.m, sd)
        i = self.sites[s]
        return sd.site_xpos[i].copy(), sd.site_xmat[i].reshape(3, 3).copy()

    def _jac(self, d, s, dofs):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, jr, self.sites[s])
        return np.vstack([jp[:, dofs], jr[:, dofs]])

    def _h_hat(self, d, M, Js):
        """11-03 item 8: the wrench each claw applies ON the object, estimated from joint-torque feedback + encoders (what the real NERO has: MIT-mode
        joint torque / position / velocity; NO wrist F/T sensor):
            h_hat_i = pinv(J_i^T) (tau_meas_i - tau_model_i)
            tau_meas_i  = the torque applied to the joints on the previous control step (d.qfrc_applied; in sim the servos are off and the torque enters
                          as qfrc_applied, qfrc_actuator = 0) + Gaussian noise of std config.COOP_TAU_NOISE_NM (ASSUMED, default 0)
            tau_model_i = M_i qdd_hat + qfrc_bias - qfrc_passive of the ARM ONLY (gravity + Coriolis + joint damping; no payload: the estimate is the
                          total wrench on the object; its object-wrench part W^+ h_o is removed by the (I - W^+ W) projection anyway),
                          qdd_hat = backward difference of the encoder velocity over one control step.
        Sign: M qdd + c = tau - J^T h  (the object reacts with -h on the arm tip).  Unmodelled: joint friction (frictionloss), which the fc wrapper
        only approximately cancels; the one-step delay of qdd_hat.  Wrench about the TCP site, world frame (same convention as carry.claw_wrench)."""
        C0 = CR.C
        dt = CR.DT
        out = []
        for s_, dofs in (("left", self.ld), ("right", self.rd)):
            qd = d.qvel[dofs].copy()
            qdd = np.zeros(7) if self.qd_prev is None else (qd - self.qd_prev[s_]) / dt
            tau_meas = np.asarray(self.ref_in["tau_prev"][s_], float).copy()
            if C0.COOP_TAU_NOISE_NM > 0.0:
                tau_meas += self._rng.normal(0.0, C0.COOP_TAU_NOISE_NM, 7)
            tau_model = M[np.ix_(dofs, dofs)] @ qdd + d.qfrc_bias[dofs] - d.qfrc_passive[dofs]
            out.append(np.linalg.pinv(Js[s_].T) @ (tau_meas - tau_model))
        self.qd_prev = {"left": d.qvel[self.ld].copy(), "right": d.qvel[self.rd].copy()}
        return np.concatenate(out)

    def _alloc(self, Js, arm, hp, h_sq, Ni):
        """internal wrench h_I = N z (N: 12x6 basis of range(I - W^+ W)) minimising sum_j w_j (tau_j / 16)^2, tau = b + Jb^T (hp + N z),
        IRLS weights w_j = (0.2 + |tau_j|/16)^2 (steers toward min-max); regularised toward the squeeze setpoint."""
        U, sv, _ = np.linalg.svd(Ni)
        N = U[:, :6]
        Jb = np.zeros((14, 12))
        Jb[:7, :6], Jb[7:, 6:] = Js["left"].T, Js["right"].T
        b = np.concatenate([arm["left"], arm["right"]])
        lim = np.concatenate([CR.E.TAU_LIM, CR.E.TAU_LIM])      # MODEL: 16 Nm per joint (MIT command range)
        A0, r0 = Jb @ N, b + Jb @ hp
        zsq = N.T @ h_sq
        z = zsq.copy()
        for _ in range(3):
            tau = r0 + A0 @ z
            D = (0.2 + np.abs(tau) / lim) / lim
            A, r = A0 * D[:, None], r0 * D
            z = np.linalg.solve(A.T @ A + self.g["mu_alloc"] * np.eye(6), -A.T @ r + self.g["mu_alloc"] * zsq)
        return N @ z

    def tau(self, d):
        if not self.on or self.ref_in is None:
            return super().tau(d)
        g, m = self.g, self.m
        mujoco.mj_forward(m, d)
        H = g["H_FD"]
        dt = CR.DT
        o = d.site_xpos[self.ref].copy()
        Ro = d.xmat[self.pan].reshape(3, 3).copy()
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.ref, v6, 0)
        w_o, v_o = v6[:3], v6[3:]
        p_cl = {s: d.site_xpos[self.sites[s]].copy() for s in SIDES}
        if self.rel is None:                  # MODEL: rigid weld => object pose in the claw frames is constant
            self.rel = {s: (d.site_xmat[self.sites[s]].reshape(3, 3).T @ Ro,
                            d.site_xmat[self.sites[s]].reshape(3, 3).T @ (o - p_cl[s])) for s in SIDES}
        # ---- desired claw / object motion from the reference joint paths (FK + central differences) ----
        pd, Rd, a_cl, vd_w = {}, {}, {}, []
        po_d, o_pos = [], []
        for s in SIDES:
            qs = self.ref_in["qr"][s]                       # q_ref at t-H, t, t+H
            P = [self._fk(d, s, q) for q in qs]
            p_m, p_0, p_p = P[0][0], P[1][0], P[2][0]
            a_cl[s] = (p_p - 2 * p_0 + p_m) / H ** 2
            v_cl = (p_p - p_m) / (2 * H)
            w_cl = rot_vec(P[2][1] @ P[0][1].T) / (2 * H)
            a_cl[s] = (a_cl[s], v_cl, w_cl)
            Rr, rp = self.rel[s]
            po_d.append(p_0 + P[1][1] @ rp)
            vd_w.append(v_cl)
            Rd[s] = P[1][1] @ Rr
        p_d = 0.5 * (po_d[0] + po_d[1])
        a_d = 0.5 * (a_cl["left"][0] + a_cl["right"][0])
        v_d = 0.5 * (vd_w[0] + vd_w[1])
        R_d = Rd["left"]
        w_d = a_cl["left"][2]
        e_p, e_v = p_d - o, v_d - v_o
        e_R, e_w = rot_err(Ro, R_d), w_d - w_o
        acc = (-np.asarray(m.opt.gravity, float)) if self.SAFE else (a_d - np.asarray(m.opt.gravity, float))   # SAFE: gravity hold only, FL-MPC supplies M a
        S = self.mass * (Ro @ self.c_rel_nom)
        h_o = object_wrench(self.mass, S, acc, g, e_p, e_v, e_R, e_w)
        Js = {s_: self._jac(d, s_, dofs) for s_, dofs in (("left", self.ld), ("right", self.rd))}
        # ---- arm-only inverse dynamics (b_i = M_i qdd_i + c_i) ----
        arm = {}
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M, d.qM)
        for i, (s, qa, dofs) in enumerate((("left", self.lq, self.ld), ("right", self.rq, self.rd))):
            J = Js[s]
            qd = d.qvel[dofs].copy()
            # Jdot qdot by a forward difference of the Jacobian along the motion (scratch data)
            sd = _scratch(m)
            sd.qpos[:] = d.qpos
            mujoco.mj_integratePos(m, sd.qpos, d.qvel, g["eps_jd"])
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, sd, jp, jr, self.sites[s])
            J2 = np.vstack([jp[:, dofs], jr[:, dofs]])
            Jdqd = (J2 - J) @ qd / g["eps_jd"]
            Jp = J.T @ np.linalg.inv(J @ J.T + g["lam"] * np.eye(6))
            a6 = np.concatenate([a_cl[s][0], np.zeros(3)])          # orientation held: angular accel reference 0
            qr = self.ref_in["qr"][s]
            qdr = (qr[2] - qr[0]) / (2 * H)
            Nn = np.eye(7) - Jp @ J
            qdd = Jp @ (a6 - Jdqd) + Nn @ (g["Kn"] * (qr[1] - d.qpos[qa]) + g["Dn"] * (qdr - qd))
            c = d.qfrc_bias[dofs] - d.qfrc_passive[dofs]
            arm[s] = M[np.ix_(dofs, dofs)] @ qdd + c
        # ---- internal wrench regulation ----
        W = grasp_matrix([p_cl["left"], p_cl["right"]], o)
        Q = None
        if g["wtd"]:
            Q = np.zeros((12, 12))
            for i, s_ in enumerate(SIDES):
                Q[6 * i:6 * i + 6, 6 * i:6 * i + 6] = Js[s_] @ Js[s_].T + 1e-3 * np.eye(6)
        Wp = weighted_inverse(W, Q)
        Pm = Wp @ W
        Ni = np.eye(12) - Pm
        ehat = (p_cl["left"] - p_cl["right"])
        ehat /= max(np.linalg.norm(ehat), 1e-9)
        h_sq = Ni @ np.concatenate([-g["f_sq"] * ehat, np.zeros(3), g["f_sq"] * ehat, np.zeros(3)])
        h_des = h_sq
        if g["alloc"]:                      # internal setpoint = torque-balanced load sharing (IRLS-weighted least squares)
            h_des = self._alloc(Js, arm, Wp @ h_o, h_sq, Ni)
        h_true = np.asarray(self.ref_in["h_meas"], float)       # sim truth: the weld constraint wrench (no real counterpart)
        if CR.C.LEGACY_COOP_SENSOR:                              # BIMANUAL_LEGACY_COOP_SENSOR=1: the pre-11-03 controller input
            h_meas = h_true
        else:
            h_meas = self._h_hat(d, M, Js)
            self.err_hist.append((max(float(np.linalg.norm((h_meas - h_true)[6 * i:6 * i + 3])) for i in range(2)),
                                  max(float(np.linalg.norm((h_meas - h_true)[6 * i + 3:6 * i + 6])) for i in range(2)),
                                  max(float(np.linalg.norm(h_true[6 * i:6 * i + 3])) for i in range(2)),
                                  max(float(np.linalg.norm(h_true[6 * i + 3:6 * i + 6])) for i in range(2))))
        h_I = Ni @ h_meas
        err = h_I - h_des
        self.I_int = np.clip(self.I_int + err * dt, -g["I_max"] / max(g["kI_I"], 1e-9), g["I_max"] / max(g["kI_I"], 1e-9))
        h_Icmd = h_des - g["kP_I"] * err - g["kI_I"] * self.I_int
        hs = Wp @ h_o + Ni @ h_Icmd
        out = {s_: arm[s_] + Js[s_].T @ hs[6 * i:6 * i + 6] for i, s_ in enumerate(SIDES)}
        F_i = [float(np.linalg.norm(h_I[6 * i:6 * i + 3])) for i in range(2)]
        T_i = [float(np.linalg.norm(h_I[6 * i + 3:6 * i + 6])) for i in range(2)]
        self.log = dict(ci_F_int=max(F_i), ci_T_int=max(T_i), ci_e_p_mm=1e3 * float(np.linalg.norm(e_p)),
                        ci_e_R_deg=float(np.degrees(np.linalg.norm(e_R))),
                        ci_tau_cmd=float(max(np.max(np.abs(out[s_])) for s_ in SIDES) / float(CR.E.TAU_LIM[0])))   # pre-clip command / 16 Nm
        if self.SAFE:
            return self._safe_finish(d, hs, Js)
        return out["left"], out["right"]            # the caller clips to the torque box


class CoopImpSafeCtl(CoopImpCtl):
    """"coopimp_safe" (11-03, opt-in): the coopimp object wrench goes THROUGH FL-MPC, so the QP rows (arm-arm clearance, torque box 16 Nm, and with
    rules='full' the table-top and joint position/speed rows) apply to it.
        h_o*  = [M_hat (-g) + Kp e_p + Dp e_v ; S x (-g) + Ko e_R + Do e_w]     (gravity hold only: FL-MPC's M_extra a + its tracking cost carry the
                planned acceleration; adding a_d - g here as well would count it twice)
        h*    = W^+ h_o* + (I - W^+ W) h_I*            (same split and internal-wrench PI as coopimp; f_sq = config.coop_f_sq())
        bias_extra_i = J_i^T h_i* + f tanh(qdot_cmd,i / v_s)        (Coulomb term of the friction wrapper, inside the torque rows; no kp/ki)
        M_extra_i    = LeverCtl's point-mass share (unchanged)
    tau_i = M_i a_i + c_i + bias_extra_i with a_i from the QP.  The caller skips the friction wrapper and the post-clip (a guard counts clip hits)."""
    SAFE = True

    def _safe_finish(self, d, hs, Js):
        fr = (self.ref_in or {}).get("fric") or {}
        terms = self.terms(d)                                   # LeverCtl point-mass shares (bias part replaced below)
        t = {}
        for i, s_ in enumerate(SIDES):
            t[s_] = (terms[s_][0], Js[s_].T @ hs[6 * i:6 * i + 6] + np.asarray(fr.get(s_, np.zeros(7)), float))
        te, ti = self._fl(d, t)
        self.log["ci_tau_cmd"] = float(max(np.max(np.abs(te)), np.max(np.abs(ti))) / float(CR.E.TAU_LIM[0]))
        return te, ti
