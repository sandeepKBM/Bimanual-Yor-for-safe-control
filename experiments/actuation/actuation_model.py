"""Realistic arm-actuation model for the YOR bimanual sim (SIMULATION ONLY).

Models what the REAL arm driver does to a commanded joint torque before it
reaches the motor, so the sim's FL-MPC (which assumes an ideal +-100 Nm
torque source, sim_common.TAU_MAX) can be stress-tested against it.

Source of the defaults -- READ (never imported/run) from the unmodified YOR
driver, YOR/piperlib/src/piper_interface.cpp::set_joint_pos_vel_torque and
YOR/piperlib/include/piper_interface.h (MIT mode, one CAN frame per motor):

    pos 16b over [-12.5, 12.5] | vel 12b over [-45, 45] | kp 12b over [0, 500]
    | kd 12b over [-5, 5] | torque 8b | crc 4b
    raw feed-forward check:  T_MIN=-8 <= tau_ff <= T_MAX=8, else `continue`
        -> that motor's command is SKIPPED this cycle (NOT clipped); the motor
           keeps executing its previous command.
    packing:  t_ref = float_to_int(tau_ff * kt, -8, 8, 8 bits)
              kt = 1/4 (joints 1-3), 1/0.8 (joints 4-6)
    float_to_int(v, lo, hi, bits) = static_cast<int>((v-lo)*(2^bits-1)/(hi-lo))
        (C truncation toward zero, NOT rounding; NO saturation)
    motor torque (MIT law) = kp*(q_ref - q) + kd*(qd_ref - qd) + tau_ff

Consequences modelled here:
  * joints 1-3: tau_ff in [-8, 8] packs into [-2, 2] of the [-8, 8] code range
    -> resolution 16/255*4 = 0.251 Nm, full raw range usable.
  * joints 4-6: tau_ff*1.25 leaves the [-8, 8] code range for |tau_ff| > 6.4
    although it passes the raw [-8, 8] check. float_to_int then produces an
    int outside [0, 255]; only its low 8 bits are sent (t_ref>>4 & 0xF,
    t_ref<<4 & 0xF0), i.e. the command WRAPS (e.g. +7.0 Nm -> about -5.9 Nm).
    overflow="wrap" reproduces those bits faithfully; overflow="clip" models a
    host-side guard that saturates at +-6.4. resolution 16/255/1.25 = 0.050 Nm.
  * The real YOR "gravity compensation" is SOFTWARE: controller_base.cpp sends
    solver_.inverse_dynamics(q, qd, 0) (gravity + Coriolis) through the SAME
    tau_ff channel (subject to the +-8 check). It is NOT a physical force.
    The sim's MJCF instead sets body_gravcomp=1 on every arm link, which
    MuJoCo applies as a physical passive force (qfrc_gravcomp, summed into
    qfrc_passive since jnt_actgravcomp is off). gravcomp=False zeroes it for
    the arm bodies so gravity must come through the (limited) ff channel.

The user's lab arm is a 7-DoF AgileX whose limits are UNKNOWN, so every limit
is a constructor parameter; the Piper MIT numbers are only the defaults.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import mujoco

# ---- Piper MIT-mode defaults (piper_interface.h / .cpp) -------------------
PIPER_T_RAW = 8.0                                   # T_MIN/T_MAX raw ff bound
PIPER_T_PACK = 8.0                                  # packed code range +-8
PIPER_T_BITS = 8
PIPER_KT = np.array([1 / 4.0] * 3 + [1 / 0.8] * 3)  # per-motor packing gain
PIPER_POS_RANGE, PIPER_POS_BITS = 12.5, 16
PIPER_VEL_RANGE, PIPER_VEL_BITS = 45.0, 12
PIPER_KP_MAX, PIPER_KD_MAX = 500.0, 5.0
# driver defaults (config.h): kp=10, kd=0.5, 500 Hz
PIPER_DEFAULT_KP, PIPER_DEFAULT_KD = 10.0, 0.5


def float_to_int(v, lo, hi, bits):
    """Exact C semantics: static_cast<int>((v-lo)*(2^bits-1)/(hi-lo)) --
    truncation toward zero, no saturation."""
    return np.trunc((np.asarray(v, dtype=np.float64) - lo) * ((1 << bits) - 1) / (hi - lo)).astype(np.int64)


def int_to_float(code, lo, hi, bits):
    """Motor-side decode (standard MIT uint->float)."""
    return np.asarray(code, dtype=np.float64) * (hi - lo) / ((1 << bits) - 1) + lo


@dataclass
class ActuationConfig:
    """All limits are parameters (the lab's 7-DoF AgileX limits are unknown)."""
    name: str = "ideal"
    ff_limit: np.ndarray | None = None        # per-joint raw |tau_ff| bound (None = unlimited)
    semantics: str = "clip"                   # 'clip' (host guard saturates) | 'drop' (driver skips -> previous accepted cmd held)
    quantize: bool = False                    # 8-bit packing of tau_ff*kt over +-t_pack
    kt: np.ndarray = field(default_factory=lambda: PIPER_KT.copy())
    t_pack: float = PIPER_T_PACK
    bits: int = PIPER_T_BITS
    overflow: str = "clip"                    # for |tau*kt|>t_pack inside ff_limit: 'clip' | 'wrap' (faithful low-8-bit wrap)
    latency_steps: int = 0                    # k physics steps of transport delay on the whole command
    gravcomp: bool = True                     # False -> zero body_gravcomp on arm bodies
    # MIT PD+ff deployment (prototype ii). kp/kd None -> pure ff (kp=kd=0).
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    quantize_refs: bool = True                # 16b pos / 12b vel ref packing
    motor_peak: np.ndarray | None = None      # optional TOTAL motor torque bound (unknown for real HW)
    ref_reset_rad: float = 0.25               # re-anchor integrated ref to measured q if it drifts this far
    # how (q_ref, qd_ref) are formed in PD mode:
    #  'integrate' -- reference = integral of the controller's planned accel (PD tracks the plan)
    #  'offset'    -- TORQUE-THROUGH-POSITION-OFFSET: q_ref = q_sent + (tau_cmd - ff_sent)/kp,
    #                 qd_ref = qd_sent. The part of tau_cmd the +-8 ff field cannot carry is
    #                 encoded in the (16-bit, +-12.5 rad) position field; at send time the motor
    #                 law reproduces tau_cmd exactly, between/after sends it adds kp/kd stiffness
    #                 and damping about the sent state. Bounded only by the motor's own torque.
    pd_ref: str = "integrate"


def piper_ff_clip_limits(cfg_kt=PIPER_KT, t_raw=PIPER_T_RAW, t_pack=PIPER_T_PACK):
    """Effective saturating limit per joint for 'clip': min(raw bound, t_pack/kt)
    -> [8, 8, 8, 6.4, 6.4, 6.4] for the Piper."""
    return np.minimum(t_raw, t_pack / np.asarray(cfg_kt))


class ActuationModel:
    """Applies the actuation model to BOTH arms' joint torques, once per
    physics step. `arms` = list of (qposadr, dofadr) index arrays."""

    def __init__(self, model, cfg: ActuationConfig, arms, arm_body_prefixes=("left_arm_", "right_arm_")):
        self.cfg = cfg
        self.arms = [(np.asarray(qa), np.asarray(da)) for qa, da in arms]
        n = len(self.arms[0][1])
        self.n = n
        self.prev_accepted = [np.zeros(n) for _ in self.arms]       # last ff each motor accepted (drop)
        self.delay = deque()
        self.ref = [None for _ in self.arms]                        # integrated (q_ref, qd_ref) for PD mode
        # stats
        self.n_steps = 0
        self.sat_count = [np.zeros(n, dtype=np.int64) for _ in self.arms]   # |cmd| beyond effective ff range
        self.drop_count = [np.zeros(n, dtype=np.int64) for _ in self.arms]
        self.wrap_count = [np.zeros(n, dtype=np.int64) for _ in self.arms]
        self.cmd_abs_max = [np.zeros(n) for _ in self.arms]
        self.excess = [[] for _ in self.arms]      # per-step |cmd|-limit (>=0) for PD-requirement stats
        self.pd_abs = [[] for _ in self.arms]
        self.cmd_hist = [[] for _ in self.arms]
        self.real_hist = [[] for _ in self.arms]
        # gravcomp
        self._gc_saved = None
        self._arm_bodies = np.array([b for b in range(model.nbody)
                                     if any((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(p)
                                            for p in arm_body_prefixes)], dtype=np.int64)

    # ---------------- gravcomp -----------------------------------------
    def install(self, model):
        if not self.cfg.gravcomp:
            self._gc_saved = model.body_gravcomp[self._arm_bodies].copy()
            model.body_gravcomp[self._arm_bodies] = 0.0

    def uninstall(self, model):
        if self._gc_saved is not None:
            model.body_gravcomp[self._arm_bodies] = self._gc_saved
            self._gc_saved = None

    # ---------------- ff channel ---------------------------------------
    def _eff_limit(self):
        c = self.cfg
        if c.ff_limit is None:
            return None
        lim = np.asarray(c.ff_limit, dtype=np.float64)
        if c.quantize or c.overflow == "clip":
            # what the channel can physically represent without wrapping
            return np.minimum(lim, c.t_pack / c.kt)
        return lim

    def ff_channel(self, arm_i, tau_cmd):
        """Map commanded ff torque -> torque the motor actually applies (ff part)."""
        c = self.cfg
        tau = np.asarray(tau_cmd, dtype=np.float64).copy()
        eff = self._eff_limit()
        if eff is not None:
            self.sat_count[arm_i] += (np.abs(tau) > eff + 1e-9)
            self.excess[arm_i].append(np.maximum(np.abs(tau) - eff, 0.0))
        if c.ff_limit is None:
            out = tau
        else:
            lim = np.asarray(c.ff_limit, dtype=np.float64)
            if c.semantics == "clip":
                tau = np.clip(tau, -eff, eff)
                accepted = np.ones(self.n, dtype=bool)
            elif c.semantics == "drop":
                accepted = np.abs(tau) <= lim
                self.drop_count[arm_i] += ~accepted
            else:
                raise ValueError(c.semantics)
            out = self.prev_accepted[arm_i].copy()
            if c.quantize:
                packed = tau * c.kt
                code = float_to_int(packed, -c.t_pack, c.t_pack, c.bits)
                over = (code < 0) | (code > (1 << c.bits) - 1)
                if c.overflow == "wrap":
                    self.wrap_count[arm_i] += over & accepted
                    code = code & ((1 << c.bits) - 1)          # low 8 bits actually transmitted
                else:
                    code = np.clip(code, 0, (1 << c.bits) - 1)
                real = int_to_float(code, -c.t_pack, c.t_pack, c.bits) / c.kt
            else:
                if c.overflow == "wrap":   # no quantization but still out-of-code-range -> treat as clip
                    tau = np.clip(tau, -c.t_pack / c.kt, c.t_pack / c.kt)
                real = tau
            out[accepted] = real[accepted]
            self.prev_accepted[arm_i] = out.copy()
        return out

    # ---------------- MIT PD term (prototype ii) ------------------------
    def pd_torque(self, arm_i, q, qd, a_plan, dt):
        """Integrated reference from the controller's planned acceleration:
        q_ref <- q_ref + qd_ref dt + a dt^2/2, qd_ref <- qd_ref + a dt; then
        tau_pd = kp (q_ref - q) + kd (qd_ref - qd). Re-anchored to the
        measured state if it drifts > ref_reset_rad (anti-windup)."""
        c = self.cfg
        if self.ref[arm_i] is None or np.max(np.abs(self.ref[arm_i][0] - q)) > c.ref_reset_rad:
            self.ref[arm_i] = (q.copy(), qd.copy())
        q_ref, qd_ref = self.ref[arm_i]
        q_ref = q_ref + qd_ref * dt + 0.5 * a_plan * dt * dt
        qd_ref = qd_ref + a_plan * dt
        self.ref[arm_i] = (q_ref, qd_ref)
        qr, qdr = q_ref, qd_ref
        if c.quantize_refs:
            qr = int_to_float(float_to_int(qr, -PIPER_POS_RANGE, PIPER_POS_RANGE, PIPER_POS_BITS),
                              -PIPER_POS_RANGE, PIPER_POS_RANGE, PIPER_POS_BITS)
            qdr = int_to_float(float_to_int(qdr, -PIPER_VEL_RANGE, PIPER_VEL_RANGE, PIPER_VEL_BITS),
                               -PIPER_VEL_RANGE, PIPER_VEL_RANGE, PIPER_VEL_BITS)
        kp = np.zeros(self.n) if c.kp is None else np.asarray(c.kp, dtype=np.float64)
        kd = np.zeros(self.n) if c.kd is None else np.asarray(c.kd, dtype=np.float64)
        return kp * (qr - q) + kd * (qdr - qd)

    # ---------------- per-physics-step hook ------------------------------
    def apply(self, model, data, dt):
        """Called right before mj_step: reads the demo's commanded arm torque
        from data.qfrc_applied, replaces it by what the real actuator applies."""
        c = self.cfg
        self.n_steps += 1
        cmds = [data.qfrc_applied[da].copy() for _, da in self.arms]
        use_pd = c.kp is not None or c.kd is not None
        a_plans = []
        if use_pd:
            # planned accel implied by the command: a = M^-1 (tau - (bias - passive))
            # (data derived quantities are current: the demo mj_forward()s every step)
            M = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M, data.qM)
            for (qa, da), tau in zip(self.arms, cmds):
                cc = data.qfrc_bias[da] - data.qfrc_passive[da]
                a_plans.append(np.linalg.solve(M[np.ix_(da, da)], tau - cc))
        # latency on the command stream (what the host sends)
        sent_state = [(data.qpos[qa].copy(), data.qvel[da].copy()) for qa, da in self.arms]
        self.delay.append((cmds, a_plans, sent_state))
        if len(self.delay) > c.latency_steps:
            cmds_d, a_d, st_d = self.delay.popleft()
        else:                       # start-up: oldest available command
            cmds_d, a_d, st_d = self.delay[0]
        for i, (qa, da) in enumerate(self.arms):
            if use_pd and c.pd_ref == "offset":
                ff = self.ff_channel(i, cmds_d[i])
                kp = np.asarray(c.kp, dtype=np.float64)
                kd = np.zeros(self.n) if c.kd is None else np.asarray(c.kd, dtype=np.float64)
                q_s, qd_s = st_d[i]
                qr = q_s + (cmds_d[i] - ff) / kp
                qdr = qd_s
                if c.quantize_refs:
                    qr = int_to_float(float_to_int(qr, -PIPER_POS_RANGE, PIPER_POS_RANGE, PIPER_POS_BITS),
                                      -PIPER_POS_RANGE, PIPER_POS_RANGE, PIPER_POS_BITS)
                    qdr = int_to_float(float_to_int(qdr, -PIPER_VEL_RANGE, PIPER_VEL_RANGE, PIPER_VEL_BITS),
                                       -PIPER_VEL_RANGE, PIPER_VEL_RANGE, PIPER_VEL_BITS)
                pd = kp * (qr - data.qpos[qa]) + kd * (qdr - data.qvel[da])
                self.pd_abs[i].append(np.abs(pd))
                real = ff + pd
            elif use_pd:
                # ff = the command itself (tau = M a + c), PD tracks the integrated plan
                ff = self.ff_channel(i, cmds_d[i])
                pd = self.pd_torque(i, data.qpos[qa].copy(), data.qvel[da].copy(), a_d[i], dt)
                self.pd_abs[i].append(np.abs(pd))
                real = ff + pd
            else:
                real = self.ff_channel(i, cmds_d[i])
            if c.motor_peak is not None:
                real = np.clip(real, -np.asarray(c.motor_peak), np.asarray(c.motor_peak))
            self.cmd_abs_max[i] = np.maximum(self.cmd_abs_max[i], np.abs(cmds[i]))
            self.cmd_hist[i].append(cmds[i])
            self.real_hist[i].append(real)
            data.qfrc_applied[da] = real

    # ---------------- summary ------------------------------------------
    def stats(self):
        n = max(self.n_steps, 1)
        out = dict(steps=self.n_steps)
        for i, side in enumerate(("left", "right")[:len(self.arms)]):
            out[f"{side}_sat_frac"] = (self.sat_count[i] / n).round(4).tolist()
            out[f"{side}_drops"] = self.drop_count[i].tolist()
            out[f"{side}_wraps"] = self.wrap_count[i].tolist()
            out[f"{side}_cmd_abs_max"] = self.cmd_abs_max[i].round(2).tolist()
            if self.excess[i]:
                ex = np.array(self.excess[i])
                out[f"{side}_excess_max"] = ex.max(0).round(2).tolist()
                out[f"{side}_excess_p99"] = np.percentile(ex, 99, axis=0).round(2).tolist()
            if self.pd_abs[i]:
                pd = np.array(self.pd_abs[i])
                out[f"{side}_pd_abs_max"] = pd.max(0).round(2).tolist()
                out[f"{side}_pd_abs_p99"] = np.percentile(pd, 99, axis=0).round(2).tolist()
        out["drops_total"] = int(sum(d.sum() for d in self.drop_count))
        out["wraps_total"] = int(sum(w.sum() for w in self.wrap_count))
        return out
