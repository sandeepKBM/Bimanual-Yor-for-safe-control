"""grasp/assist.py -- wrist-camera-guided REAL friction grasp of the tray with both hands (TeleopSim(grasp_mode="friction")).  SIMULATION ONLY.

No weld, no teleport: the hands move through the same path as the operator's tray jog (DLS IK -> joint references of the unmodified FL-MPC),
the pads close under force control and hold the tray by friction.  Trigger: Cmd.grasp (the operator has brought the hands within the
usual 3.5 cm / 6 deg; the cameras finish the job).  Cmd.release opens the fingers.

State machine (both arms in lock-step; one verified grasp = both hands):
  IDLE -> PRE      open the fingers, move both hands to the PRE-GRASP pose (pulled back `retreat` along the approach axis, raised `lift`),
                   visual servo on the wrist-camera tray estimate (5 Hz, 3-tick latency); needs a confident estimate on both hands
                   -> converged (<= tol) -> APPROACH.   Low confidence for too long -> ABORT (back to the start pose).
        APPROACH   freeze the filtered tray estimate (world), slide the hands in along the approach axis to the grasp pose (the
                   left camera is blind inside its 0.10 m minimum range: dead reckoning from the frozen estimate; the tray is static).
        CLOSE      force-controlled closing (ctrl = q -/+ F/kp, F ramp to F_GRIP, finger actuator limit +-20 N untouched); closed when
                   the jaws have stopped on a plate-thick gap (finger encoders) -> VERIFY, closed on nothing -> failure.
        VERIFY     pad-contact count (>= 1 per pad, both hands) + TUG TEST: lift both hands `tug` and require the wrist-camera tray estimate
                   (world) to rise with the hands and the tray-in-gripper pose to stay put (slip estimate); then set the tray back down.
        HOLD       done: ObjectFrame re-based on the actual hands (the operator's tray jog continues), tilt cone on, payload feed-forward on.
  failure at any step -> open, back to PRE (one regrasp), second failure -> FAILED (fingers open; lift/jog/base stay gated).
While IDLE-excluded states are active `gate()` zeroes the base twist, lift and jog (lift only after both hands are verified).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from world_model.v2.house.teleop import base as BS
from world_model.v2.house.teleop.arm import _log_R, _rotvec_to_R
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY
from . import estimate as E
from . import util as U
from .hands import HandServo, rot_err
from .wrist import WristCam

SIDES = ("left", "right")
SC_CUP_XY = (-0.045, 0.0)             # cup centre in the tray frame (teleop.scene.CUP_TRAY_XY)
ACTIVE = ("PRE", "APPROACH", "CLOSE", "VERIFY")


@dataclass
class Cfg:
    seg: str = "depth"                    # 'depth' (non-privileged) | 'priv' (sim segmentation mask, upper bound)
    log_both: bool = False                # also run the other segmenter every cycle and log its error (paired comparison)
    retreat: tuple = (0.11, 0.13)         # pre-grasp pull-back along the approach axis, (left, right) [m]  (>= camera min range 0.10 m
                                          # + the 0.026 m camera-to-pad offset for the edge to be seen)
    lift: float = 0.02                    # pre-grasp raise [m]  (open jaw inner faces at +-0.035 -> lower face 15 mm under the plate)
    f_grip: float = 10.0                  # per-finger grip force [N] (actuator limit 20 N)
    f_min: float = 3.0                    # minimum mean pad normal force per hand to accept the grasp [N]
    grip_mode: str = "spring"             # 'spring': position servo target = closed, slewed at close_speed; grip force = kp * (plate half-thickness)
                                          #   ~ 9 N per finger on the 6 mm plate (compliant, no contact chatter);  'force': ctrl = q -/+ F/kp (ideal force source)
    close_speed: float = 0.04             # m/s per finger, spring mode
    close_ramp_s: float = 0.4
    tug: float = 0.020                    # verification lift of both hands [m]
    cam_period: int = 10                  # ticks between captures (5 Hz)
    latency_ticks: int = 3
    conf_min: float = 0.55
    pre_tol: tuple = (0.004, math.radians(5.0))     # servo convergence tolerance (hand vs goal)
    t_pre_max: float = 14.0
    t_appr_max: float = 6.0
    t_close_max: float = 3.0
    t_verify_max: float = 5.0
    appr_speed: float = 0.05              # m/s along the approach axis
    max_attempts: int = 2
    gap_range: tuple = (0.003, 0.014)     # finger-encoder jaw gap that means 'plate between the pads' (plate 6 mm) [m]
    min_conf_n: int = 3                   # confident estimates required per hand before leaving PRE
    open_q: float = 0.035
    anchor_gate: tuple = (0.08, math.radians(15.0))   # an estimate this far from the start prior is rejected (tray is static, operator tolerance 3.5 cm)
    seed: int = 0


class TrayFilter:
    """running weighted mean of tray-pose estimates in the WORLD frame (the tray is static until it is gripped), with gating."""

    def __init__(self):
        self.p = None
        self.R = None
        self.n = 0
        self.w = 0.0
        self.last_t = -1e9
        self.rejected = 0

    def reset(self):
        self.__init__()

    def update(self, p, R, conf, t, gate=(0.06, math.radians(15.0))):
        if self.p is None:
            self.p, self.R, self.n, self.w, self.last_t = p.copy(), R.copy(), 1, conf, t
            return True
        if np.linalg.norm(p - self.p) > gate[0] or np.linalg.norm(_log_R(R @ self.R.T)) > gate[1]:
            self.rejected += 1
            return False
        a = max(0.25, conf / (self.w + conf))
        self.p = self.p + a * (p - self.p)
        self.R = _rotvec_to_R(a * _log_R(R @ self.R.T)) @ self.R
        self.w = min(self.w + conf, 6.0)
        self.n += 1
        self.last_t = t
        return True


class GraspAssist:
    def __init__(self, sim, cfg: Cfg | None = None):
        self.sim = sim
        self.cfg = cfg or Cfg()
        m = sim.m
        self.m = m
        self.state = "IDLE"
        self.t_state = 0.0
        self.attempt = 0
        self.hs = HandServo(sim)
        self.hs.w_rot = 0.5
        self.cams = {s: WristCam(m, s) for s in SIDES}
        self.filt = {s: TrayFilter() for s in SIDES}
        self.queue = {s: deque() for s in SIDES}
        self.f_cmd = 0.0                       # per-finger force command [N]; 0 and pos mode -> open
        self.finger_mode = "pos"
        self.finger_q = self.cfg.open_q
        self.k = 0
        self.fa = {s: (int(m.jnt_qposadr[m.joint(f"{s}_arm_joint7").id]), int(m.jnt_qposadr[m.joint(f"{s}_arm_joint8").id])) for s in SIDES}
        self.fact = {s: (int(m.actuator(f"{s}_arm_joint7_pos").id), int(m.actuator(f"{s}_arm_joint8_pos").id)) for s in SIDES}
        self.kp = float(m.actuator_gainprm[self.fact["left"][0], 0])           # finger position gain of the rig (scene.FINGER_KP_GRASP)
        self.pad = {s: (int(m.geom(f"{s}_arm_link7_pad").id), int(m.geom(f"{s}_arm_link8_pad").id)) for s in SIDES}
        self.tray_geom = int(m.geom("tray_top").id)
        self.tray_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "tray"], int)     # plate + handle strips
        self.prior = {}                        # side -> (p, R) world prior (nominal relpose on the start hand pose)
        self.start_hand = {}
        self.pre_goal_off = {}
        self.appr_t0 = 0.0
        self.metrics_ = self._new_metrics()
        self.ref_rel = {}
        self.slip_log = []
        self.tug_before = {}
        self.tug_phase = 0
        self.contact_samples = {s: [] for s in SIDES}
        self.n_cap = 0
        self.t_close = 0.0
        self.grasp_t0 = None
        self.fail_reason = ""
        self.payload_kg = float(sum(m.body_mass[b] for b in (m.body("tray").id, m.body("cup").id)) +
                                sum(m.body_mass[m.body(f"ball_{i}").id] for i in range(sim.rig.n_beads)))

    # ------------------------------------------------------------------ metrics
    def _new_metrics(self):
        return dict(success=None, attempts=0, t_trigger=None, t_done=None, est_log=[], contacts={}, grip_force={}, gap={},
                    slip_cam_mm={}, tug_rise_mm=None, fail_reason="", t_state={})

    def metrics(self):
        mt = dict(self.metrics_)
        mt["state"] = self.state
        if self.slip_log:
            sl = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in self.slip_log])
            mt["carry_slip_mm_max"] = [float(sl[:, 0].max()), float(sl[:, 1].max())]
            mt["carry_slip_deg_max"] = [float(sl[:, 2].max()), float(sl[:, 3].max())]
            mt["carry_contact_slip_mm_max"] = [float(sl[:, 4].max()), float(sl[:, 5].max())]
        return mt

    # ------------------------------------------------------------------ helpers
    def _goto(self, st):
        self.metrics_["t_state"][st] = float(self.sim.t)
        self.state, self.t_state = st, 0.0

    def _fingers_open(self):
        self.finger_mode, self.finger_q, self.f_cmd = "pos", self.cfg.open_q, 0.0

    def _fingers_close(self):
        self.t_close = 0.0
        if self.cfg.grip_mode == "force":
            self.finger_mode, self.f_cmd = "force", 0.0
        else:
            self.finger_mode, self.f_cmd = "pos", 0.0

    def gap(self, s):
        d = self.sim.d
        a7, a8 = self.fa[s]
        return float(d.qpos[a7] - d.qpos[a8])

    def pad_contacts(self, s):
        """(n contacts upper pad, lower pad, sum of normal force over both) between the pads and the tray."""
        m, d = self.m, self.sim.d
        p7, p8 = self.pad[s]
        n7 = n8 = 0
        fn = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = (c.geom1, c.geom2)
            if c.geom1 in self.tray_geoms:
                g = c.geom2
            elif c.geom2 in self.tray_geoms:
                g = c.geom1
            else:
                continue
            if g == p7 or g == p8:
                mujoco.mj_contactForce(m, d, i, f6)
                fn += abs(f6[0])
                if g == p7:
                    n7 += 1
                else:
                    n8 += 1
        return n7, n8, float(fn)

    def _robot_goal(self, k, s, p_tray, R_tray, retreat, lift):
        """hand goal in the ObjectFrame robot frame for a tray pose (world), pulled back by `retreat` along the approach axis, raised `lift`."""
        sim = self.sim
        rel, Rn = U.nominal_T_g_tray(sim.of, s)
        Rg = R_tray @ Rn.T
        pg = p_tray - Rg @ rel - Rg[:, 2] * retreat + np.array([0.0, 0.0, lift])
        root = self.m.body("yor_root").id
        Rr = sim.d.xmat[root].reshape(3, 3)
        return Rr.T @ (pg - sim.d.xpos[root]) - np.array([0.0, 0.0, sim.lift_actual()]), Rr.T @ Rg

    def _tray_est(self, s):
        """current best tray pose (world) for side s: filter if present, else the start prior."""
        f = self.filt[s]
        return (f.p, f.R) if f.p is not None else self.prior[s]

    def _hand_rel(self, s):
        """TRUE tray pose in the gripper frame (privileged, for slip / error metrics only)."""
        d = self.sim.d
        b = self.m.body(f"{s}_arm_gripper_base").id
        Rg, pg = d.xmat[b].reshape(3, 3), d.xpos[b]
        pt, Rt = U.tray_pose(self.sim)
        return Rg.T @ (pt - pg), Rg.T @ Rt

    # ------------------------------------------------------------------ public hooks called by TeleopSim
    def start(self):
        if self.state == "HOLD":
            return True
        if self.state in ACTIVE:
            return False
        sim = self.sim
        self.metrics_ = self._new_metrics()
        self.metrics_["t_trigger"] = float(sim.t)
        self.attempt = 0
        self.fail_reason = ""
        self.slip_log = []
        self._begin_attempt(first=True)
        return False

    def _begin_attempt(self, first=False):
        sim, cfg = self.sim, self.cfg
        self.attempt += 1
        self.metrics_["attempts"] = self.attempt
        self.hs.start_from_current()
        for s in SIDES:
            b = self.m.body(f"{s}_arm_gripper_base").id
            gp, gR = sim.d.xpos[b].copy(), sim.d.xmat[b].reshape(3, 3).copy()
            if first or s not in self.prior:
                self.prior[s] = U.prior_tray_pose(sim.of, s, gp, gR)
            self.filt[s].reset()
            self.queue[s].clear()
        # on a regrasp keep the previous best estimate as the prior
        self._fingers_open()
        self.k = 0
        self._set_pre_goals()
        self._goto("PRE")
        sim.ctl.payload[:] = 0.0

    def _set_pre_goals(self):
        for k, s in enumerate(SIDES):
            p, R = self._tray_est(s)
            gp, gR = self._robot_goal(k, s, p, R, self.cfg.retreat[k], self.cfg.lift)
            self.hs.set_goal(k, gp, gR)

    def release(self):
        sim = self.sim
        self._fingers_open()
        sim.ctl.payload[:] = 0.0
        sim.grasped = False
        sim.ctl.tilt = False
        self.hs.active = False
        if self.state in ("HOLD", "FAILED", "ABORT") or self.state in ACTIVE:
            self.state = "IDLE"

    def gate(self, cmd):
        """hold the base / lift / jog while the assisted grasp is in progress; lift only after both hands are verified."""
        if self.state in ACTIVE or self.state in ("FAILED", "ABORT"):
            from world_model.v2.house.teleop.sim import Cmd
            return Cmd(twist=np.zeros(3), lift_v=0.0, jog=np.zeros(6), grasp=cmd.grasp, release=cmd.release)
        return cmd

    def set_feedforward(self, v, acc, yaw):
        r = np.array(self.sim.rig.info["tray_local"][:2], float)
        a = BS.cup_accel_body(v, acc, r)
        c, s = math.cos(yaw), math.sin(yaw)
        self.sim.ctl.a_world[:] = (c * a[0] - s * a[1], s * a[0] + c * a[1], 0.0)
        if self.sim.ctl.payload.any():
            self.sim.ctl.payload_pt = self.sim.d.xpos[self.sim.tray].copy()

    # ------------------------------------------------------------------ fingers: per physics step
    def substep(self):
        d = self.sim.d
        kp = self.kp
        for s in SIDES:
            a7, a8 = self.fa[s]
            c7, c8 = self.fact[s]
            if self.finger_mode == "pos":
                d.ctrl[c7], d.ctrl[c8] = self.finger_q, -self.finger_q
            else:
                F = self.f_cmd
                d.ctrl[c7] = d.qpos[a7] - F / kp           # joint7 closes toward 0; the position servo force kp (ctrl - q) = -F
                d.ctrl[c8] = d.qpos[a8] + F / kp           # joint8 closes toward 0

    # ------------------------------------------------------------------ per tick (50 Hz)
    def tick_update(self):
        sim, cfg = self.sim, self.cfg
        dt = 0.02
        if self.state == "HOLD":
            self._log_slip()
            return
        if self.state not in ACTIVE:
            return
        self.t_state += dt
        self.k += 1
        if self.state in ("PRE", "APPROACH") and self.k % cfg.cam_period == 0:
            self._capture()
        self._deliver()
        st = self.state
        if st == "PRE":
            self._do_pre()
        elif st == "APPROACH":
            self._do_approach()
        elif st == "CLOSE":
            self._do_close()
        elif st == "VERIFY":
            self._do_verify()
        if self.hs.active:
            self.hs.step(dt)

    # ---- perception
    def _capture(self):
        sim, cfg = self.sim, self.cfg
        for s in SIDES:
            cam = self.cams[s]
            pr_p, pr_R = self._tray_est(s)
            a_p, a_R = self.prior[s]                          # the ROI follows the ANCHOR prior (never the possibly-wrong running estimate)
            roi = cam.roi_for_box(sim.d, a_p + a_R @ np.array([0, 0, 0.01]), a_R, (HX + 0.12, HY + 0.12, 0.07))
            fr = cam.capture(sim.d, self.n_cap, seed=cfg.seed, t=sim.t, roi=roi)
            self.n_cap += 1
            self.queue[s].append((self.k + cfg.latency_ticks, fr, pr_p.copy(), pr_R.copy()))

    def _deliver(self):
        sim, cfg = self.sim, self.cfg
        for k, s in enumerate(SIDES):
            q = self.queue[s]
            while q and q[0][0] <= self.k:
                _, fr, pr_p, pr_R = q.popleft()
                est = E.estimate_tray(fr, pr_p, pr_R, mode=cfg.seg, tray_geoms=self.tray_geoms, seed=self.n_cap)
                row = dict(t=float(fr.t), side=s, state=self.state, conf=est.conf, ok=est.ok, mode=cfg.seg, n_in=est.n_in, ms=est.ms)
                pt, Rt = U.tray_pose(sim)
                if est.ok:
                    # NOTE: the GT tray pose here is the tray at DELIVERY time; the tray is static during PRE/APPROACH, moving only in VERIFY
                    e = E.pose_error(est.p, est.R, pt, Rt)
                    row.update(err_mm=e[1], err_xyz=[float(x) for x in e[0]], err_deg=e[2])
                    a_p, a_R = self.prior[s]
                    near_anchor = (np.linalg.norm(est.p - a_p) < cfg.anchor_gate[0] and
                                   np.linalg.norm(_log_R(est.R @ a_R.T)) < cfg.anchor_gate[1])
                    row["near_anchor"] = bool(near_anchor)
                    if est.conf >= cfg.conf_min and near_anchor and self.state in ("PRE", "APPROACH"):
                        self.filt[s].update(est.p, est.R, est.conf, fr.t)
                    self.last_est = getattr(self, "last_est", {})
                    self.last_est[s] = (est, fr)
                if cfg.log_both:
                    other = "priv" if cfg.seg == "depth" else "depth"
                    e2 = E.estimate_tray(fr, pr_p, pr_R, mode=other, tray_geoms=self.tray_geoms, seed=self.n_cap)
                    if e2.ok:
                        ee = E.pose_error(e2.p, e2.R, pt, Rt)
                        row.update({f"{other}_err_mm": ee[1], f"{other}_err_xyz": [float(x) for x in ee[0]], f"{other}_err_deg": ee[2],
                                    f"{other}_conf": e2.conf})
                self.metrics_["est_log"].append(row)

    # ---- states
    def _both_converged(self, tol_p, tol_r):
        return all(self.hs.goal_error(k)[0] < tol_p and math.radians(self.hs.goal_error(k)[1]) < tol_r for k in (0, 1))

    def _fail(self, why):
        self.fail_reason = why
        self.metrics_["fail_reason"] = why
        self.metrics_.setdefault("fail_log", []).append((float(self.sim.t), self.state, why))
        self._fingers_open()
        self.sim.ctl.payload[:] = 0.0
        if self.attempt < self.cfg.max_attempts:
            self._begin_attempt(first=False)
        else:
            self.metrics_["success"] = False
            self._goto("FAILED")
            self.hs.active = False

    def _do_pre(self):
        cfg = self.cfg
        # keep the goal on the current filtered estimate (moves the goal as the camera refines the tray pose)
        self._set_pre_goals()
        have = all(self.filt[s].n >= cfg.min_conf_n for s in SIDES)
        if have and self._both_converged(*cfg.pre_tol) and self.t_state > 1.0:
            # freeze the estimate; approach on it
            self.frozen = {s: (self.filt[s].p.copy(), self.filt[s].R.copy()) for s in SIDES}
            self.appr_t0 = 0.0
            self._goto("APPROACH")
            return
        if self.t_state > cfg.t_pre_max:
            if have:
                self._fail("pre-grasp did not converge")
            else:
                self._fail("low wrist-camera confidence (n=%s)" % {s: self.filt[s].n for s in SIDES})

    def _do_approach(self):
        cfg = self.cfg
        self.appr_t0 += 0.02
        for k, s in enumerate(SIDES):
            p, R = self.frozen[s]
            f = max(0.0, 1.0 - self.appr_t0 * cfg.appr_speed / cfg.retreat[k])        # 1 -> 0
            lift = cfg.lift * f
            gp, gR = self._robot_goal(k, s, p, R, cfg.retreat[k] * f, lift)
            self.hs.set_goal(k, gp, gR)
        done = self.appr_t0 * cfg.appr_speed >= max(cfg.retreat) and self._both_converged(0.004, math.radians(3.0))
        if done and self.t_state > 1.0:
            self._fingers_close()
            self._goto("CLOSE")
            return
        if self.t_state > cfg.t_appr_max + max(cfg.retreat) / cfg.appr_speed:
            self._fail("approach did not converge")

    def _do_close(self):
        cfg = self.cfg
        self.t_close += 0.02
        if cfg.grip_mode == "force":
            self.f_cmd = cfg.f_grip * min(1.0, self.t_close / cfg.close_ramp_s)
        else:
            self.finger_q = max(0.0, self.finger_q - cfg.close_speed * 0.02)         # target slews to 'closed'; the plate stops the pads
        d = self.sim.d
        settled = self.t_close > (cfg.close_ramp_s if cfg.grip_mode == 'force' else cfg.open_q / cfg.close_speed) + 0.5
        if settled:
            gaps = {s: self.gap(s) for s in SIDES}
            self.metrics_["gap"] = gaps
            if all(cfg.gap_range[0] <= g <= cfg.gap_range[1] for g in gaps.values()):
                for k, s in enumerate(SIDES):
                    g = self.hs.hand_pose(k)
                    self.hs.set_goal(k, *g, jump=True)       # hold where the hands are
                self.tug_phase = 0
                self.tug_t0 = 0.0
                self.contact_samples = {s: [] for s in SIDES}
                self.sim.ctl.payload[:] = self.payload_kg / 2.0
                self.sim.ctl.payload_pt = d.xpos[self.sim.tray].copy()
                self._goto("VERIFY")
            else:
                self._fail("closed on nothing / wrong gap %s" % {s: round(g * 1e3, 1) for s, g in gaps.items()})

    def _verify_contacts(self):
        """contact check over the averaging window (the pad force flickers at ~8 Hz around its mean): both pads touching the plate in >= 70 %
        of the samples and the mean total normal force >= f_min (friction capacity 2 pads x mu_pad(2) x N >> the 2.3 N weight share)."""
        res = {}
        ok = True
        for s in SIDES:
            smp = np.array(self.contact_samples[s])
            if len(smp) == 0:
                return False, {s_: (0, 0, 0.0) for s_ in SIDES}
            up, lo, fn = (smp[:, 0] >= 1).mean(), (smp[:, 1] >= 1).mean(), smp[:, 2].mean()
            res[s] = (int(np.median(smp[:, 0])), int(np.median(smp[:, 1])), float(fn))
            ok &= (up >= 0.7 and lo >= 0.7 and fn >= self.cfg.f_min)
        return ok, res

    def _cup_view(self):
        """right/left wrist camera RGB-equivalent cup track (u, v in px) -> dict side -> (n_px, u, v, range) for the cameras that see the cup."""
        out = {}
        for s in SIDES:
            r = self.cams[s].seg_cup(self.sim.d)
            if r is not None and r[0] >= 400:                  # the cup must fill a solid part of the image (it does on the right camera)
                out[s] = r
        return out

    def _hand_world(self, s):
        b = self.m.body(f"{s}_arm_gripper_base").id
        return self.sim.d.xpos[b].copy(), self.sim.d.xmat[b].reshape(3, 3).copy()

    def _do_verify(self):
        cfg, sim = self.cfg, self.sim
        self.tug_t0 += 0.02
        if self.tug_phase == 0:                            # contact samples, then the wrist-camera snapshot BEFORE the tug
            for s in SIDES:
                self.contact_samples[s].append(self.pad_contacts(s))
            if self.tug_t0 < 0.4:
                return
            ok, res = self._verify_contacts()
            self.metrics_["contacts"] = {s: dict(upper=r[0], lower=r[1]) for s, r in res.items()}
            self.metrics_["grip_force"] = {s: r[2] for s, r in res.items()}
            if not ok:
                self._fail("pad contacts / force too low %s" % {s: (r[0], r[1], round(r[2], 1)) for s, r in res.items()})
                return
            self.cup_before = self._cup_view()
            self.tray_z0 = float(sim.d.xpos[sim.tray][2])                 # TRUE tray height: logged, never used for the decision
            self.ref_rel = {s: self._hand_rel(s) for s in SIDES}
            self.ref_pad = {s: self._pad_in_tray(s) for s in SIDES}
            for k in (0, 1):
                p, R = self.hs.hand_pose(k)
                self.hs.set_goal(k, p + np.array([0, 0, cfg.tug]), R)
            self.tug_phase, self.tug_t0 = 1, 0.0
            return
        if self.tug_phase == 1:                            # rise, then the wrist-camera snapshot AFTER
            if self.tug_t0 < 1.8:
                return
            true_rise = float(sim.d.xpos[sim.tray][2]) - self.tray_z0
            self.metrics_["tug_rise_true_mm"] = true_rise * 1e3
            self.metrics_["tug_slip_true_mm"] = {s: float(np.linalg.norm(self._hand_rel(s)[0] - self.ref_rel[s][0]) * 1e3) for s in SIDES}
            cam_ok, why = True, ""
            after = self._cup_view()
            checked = {}
            for s in SIDES:
                if s in self.cup_before and s in after:
                    n0, u0, v0, r0 = self.cup_before[s]
                    n1, u1, v1, r1 = after[s]
                    f = self.cams[s].spec.f_px
                    # pixel shift -> mm at the NOMINAL cup range (an RGB image has no range): vertical (v) and lateral (u) components
                    rng_nom = self._nominal_cup_range(s)
                    dv_mm, du_mm = (v1 - v0) * rng_nom / f * 1e3, (u1 - u0) * rng_nom / f * 1e3
                    checked[s] = dict(shift_mm=float(math.hypot(dv_mm, du_mm)), dv_mm=float(dv_mm), du_mm=float(du_mm),
                                      area_ratio=float(n1 / max(n0, 1)))
                    # not gripped: the hand rose `tug` while the cup stayed => shift ~ tug (15 mm) in v.  Gripped: ~0 (+ pad compliance).
                    if checked[s]["shift_mm"] > 0.6 * cfg.tug * 1e3:
                        cam_ok, why = False, "wrist camera (%s): the cup/tray did not move with the gripper (image shift %.1f mm)" % (s, checked[s]["shift_mm"])
            self.metrics_["cam_check"] = checked
            self.metrics_["cam_checked"] = bool(checked)
            if not cam_ok:
                self._fail(why)
                return
            for k in (0, 1):
                p, R = self.hs.hand_pose(k)
                self.hs.set_goal(k, p - np.array([0, 0, cfg.tug]), R)
            self.tug_phase, self.tug_t0 = 2, 0.0
            return
        if self.tug_phase == 2:                            # set the tray back down (grip kept), then hand over
            if self.tug_t0 < 1.8:
                return
            self._finish()

    def _nominal_cup_range(self, s):
        """range from the camera to the cup centre if the tray is where the hand expects it (nominal relpose) [m]."""
        sim = self.sim
        gp, gR = self._hand_world(s)
        pr_p, pr_R = U.prior_tray_pose(sim.of, s, gp, gR)
        cup_c = pr_p + pr_R @ np.array([SC_CUP_XY[0], SC_CUP_XY[1], 0.04])
        cam = self.cams[s]
        o, R = cam.pose(sim.d)
        return float(max(0.03, -(R.T @ (cup_c - o))[2] - 0.03))          # to the near wall (cup radius ~0.03)

    def _pad_in_tray(self, s):
        """pad centres (upper, lower) in the TRAY frame -- the tangential part of their motion is the contact slip."""
        d = self.sim.d
        pt, Rt = U.tray_pose(self.sim)
        return np.array([Rt.T @ (d.geom_xpos[g] - pt) for g in self.pad[s]])

    def _finish(self):
        sim = self.sim
        self.hs.rebase_object_frame()
        sim.grasped = True
        sim.ctl.tilt = sim._tilt_cfg
        self.metrics_["success"] = True
        self.metrics_["t_done"] = float(sim.t)
        for s in SIDES:
            n7, n8, fn = self.pad_contacts(s)
            self.metrics_.setdefault("contacts_done", {})[s] = dict(upper=n7, lower=n8)
            self.metrics_.setdefault("grip_force_done", {})[s] = fn
        self.ref_rel = {s: self._hand_rel(s) for s in SIDES}
        self.ref_pad = {s: self._pad_in_tray(s) for s in SIDES}
        self.slip_log = []
        self._goto("HOLD")

    def _log_slip(self):
        """carry-phase slip (TRUE kinematics, instrumentation only): (t, rigid slip L, R [mm], rot L, R [deg], contact slip L, R [mm]).
        rigid slip = displacement of the tray origin in the gripper frame (includes pivoting inside the compliant pinch);
        contact slip = tangential (tray x,y) displacement of the pad centres in the tray frame (what actually slid)."""
        sim = self.sim
        sl, rt, cs = [], [], []
        for s in SIDES:
            r0, R0 = self.ref_rel[s]
            r1, R1 = self._hand_rel(s)
            sl.append(float(np.linalg.norm(r1 - r0) * 1e3))
            rt.append(float(math.degrees(np.linalg.norm(_log_R(R1 @ R0.T)))))
            dp = self._pad_in_tray(s) - self.ref_pad[s]
            cs.append(float(np.linalg.norm(dp[:, :2], axis=1).max() * 1e3))
        self.slip_log.append((float(sim.t), sl[0], sl[1], rt[0], rt[1], cs[0], cs[1]))
