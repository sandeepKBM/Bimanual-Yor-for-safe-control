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
import os
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from world_model.v2.house.teleop import base as BS
from world_model.v2.house.teleop.arm import _log_R, _rotvec_to_R
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY
from . import estimate as E
from . import util as U
from . import proprio as PR
from . import fuse as FU
from . import place as PL
from .hands import HandServo, rot_err
from .wrist import WristCam

SIDES = ("left", "right")
SC_CUP_XY = (-0.045, 0.0)             # cup centre in the tray frame (teleop.scene.CUP_TRAY_XY)
ACTIVE = ("PRE", "APPROACH", "CLOSE", "VERIFY")
PLACE_STATES = ("PLACE_LOWER", "PLACE_OPEN", "PLACE_RETREAT", "PLACE_SETTLE")


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
    # ---- PROPRIO-ONLY (default from env BIMANUAL_PROPRIO_ONLY; OFF = the original privileged behaviour bit-for-bit), see grasp/proprio.py
    proprio: bool = field(default_factory=PR.proprio_env)
    f_min_eff: float = 4.0                # min summed finger motor force per hand to accept the grasp [N]  (plate: ~2 x kp x 3 mm = 12 N)
    gap_slope: float = 0.005              # max |d gap/dt| over the verify window [m/s] (jaws stalled on something; closing runs at 0.08 m/s)
    disable_est: bool = False             # DEBUG: ignore the wrist-camera estimate, grasp where the hands expect the tray (bad-grasp test)
    payload_win: tuple = (1.0, 1.8)       # payload estimation window inside the tug-hold phase [s]
    # ---- 2026-10-12 additions (every one has a switch back to the previous behaviour; see grasp/README in the module docstring of smoke.py)
    look: bool = field(default_factory=lambda: os.environ.get("BIMANUAL_GRASP_LOOK", "1") == "1")
                                          # LATERAL LOOK: the hover pose is shifted along the edge (look_side) for the camera estimate, then the hand returns to the
                                          # edge centre line and inserts.  Measured offline (smoke --look-scan): the along-edge error halves or better; a FAR-HIGH look
                                          # (the idea of seeing corners from above) is kinematically impossible (pads straddle the plate / the hands would meet)
    look_side: tuple = (-0.05, 0.06)      # lateral shift of the (left, right) hand along the edge during the look [m] (gripper x axis)
    fuse: bool = field(default_factory=lambda: os.environ.get("BIMANUAL_GRASP_FUSE", "1") == "1")
                                          # information-fuse the left / right wrist estimates (+ any ExternalTrayPose) into ONE tray pose both hands use
    fast: bool = field(default_factory=lambda: os.environ.get("BIMANUAL_GRASP_FAST", "1") == "1")
                                          # speed preset: faster hand slew / approach / closing, shorter tug (False = the original 12-16 s timing)
    v_transit: float = 0.30               # LOOK -> hover -> descend leg speed [m/s] (fast)
    strip_gap: float = 0.0085             # closed finger gap below this = pads on the bare 6 mm plate, not on the 12 mm handle strip -> re-insert
    reinsert_step: float = 0.007          # extra insertion per re-insert [m]
    max_reinsert: int = 2
    asym_tol: float = 0.0035              # |q7 + q8| / 2 above this at the stall = the pads closed off-centre vertically -> re-centre
    z_weight: float = field(default_factory=lambda: 400.0 if os.environ.get("BIMANUAL_GRASP_FAST", "1") == "1" else 40.0)
                                          # RANSAC preference for the plane at the prior height (see estimate.ransac_plane)
    bounds_xy: float | None = field(default_factory=lambda: 0.045 if os.environ.get("BIMANUAL_GRASP_FAST", "1") == "1" else None)
                                          # estimator search box in x / y around the anchor prior [m] (capture range 3.5 cm + 1 cm); None = the old 7 cm
    z_lo: float | None = field(default_factory=lambda: -0.007 if os.environ.get("BIMANUAL_GRASP_FAST", "1") == "1" else None)
                                          # depth plane search ignores points more than this far BELOW the height the hands expect the plate (counter slab is 12 mm lower)
    bounds_z: float | None = field(default_factory=lambda: 0.005 if os.environ.get("BIMANUAL_GRASP_FAST", "1") == "1" else None)
                                          # estimator height search about the height the hands expect [m] (None = the old 12 mm)
    x_bias: float = 0.0022                # depth-estimate edge-normal bias [m] along the tray x axis (est - truth), subtracted when cfg.fast; measured offline
                                          # (look_scan.py, 48 trials x 3 captures: left +2.5, right +1.9 mm)
    # ---- PLACE / RELEASE sequence (Cmd.release in HOLD): lower with the lift axis until a torque-detected touchdown, open, retreat, settle
    place_on_release: bool = field(default_factory=lambda: os.environ.get("BIMANUAL_GRASP_PLACE", "1") == "1")
    place_speed: float = 0.025            # lowering speed of the lift axis [m/s]
    place_max_drop: float = 0.20          # give up (keep holding) if no touchdown after this much descent [m]
    place_frac: float = 0.5               # touchdown when the carried load falls below this fraction of the descent-calibrated load
    place_open_s: float = 0.6
    place_retreat_speed: float = 0.15
    t_tug_up: float = 1.8
    t_tug_down: float = 1.8
    t_tug_sample: float = 0.4
    align_period: int = 25                # operator ALIGN cue: wrist-camera estimate every this many ticks (0.5 s) while idle
    align_mem_s: float = 30.0             # ... and kept (world frame, the tray is static) for this long
    max_age_s: float = 3.0                # drop a tray measurement older than this from the fusion


class TrayFilter:
    """running weighted mean of tray-pose estimates in the WORLD frame (the tray is static until it is gripped), with gating."""

    def __init__(self):
        self.p = None
        self.R = None
        self.n = 0
        self.w = 0.0
        self.last_t = -1e9
        self.rejected = 0
        self.streak = 0
        self.conf_last = 0.0

    def reset(self):
        self.__init__()

    def update(self, p, R, conf, t, gate=(0.06, math.radians(15.0))):
        if self.p is None:
            self.p, self.R, self.n, self.w, self.last_t = p.copy(), R.copy(), 1, conf, t
            self.conf_last = conf
            return True
        if np.linalg.norm(p - self.p) > gate[0] or np.linalg.norm(_log_R(R @ self.R.T)) > gate[1]:
            self.rejected += 1
            self.streak += 1
            if self.streak >= 4:             # four rejections in a row: the FILTER is the outlier, not the measurements -> restart on this one
                self.p, self.R, self.n, self.w, self.last_t, self.conf_last, self.streak = p.copy(), R.copy(), 1, conf, t, conf, 0
                return True
            return False
        self.streak = 0
        a = max(0.25, conf / (self.w + conf))
        self.p = self.p + a * (p - self.p)
        self.R = _rotvec_to_R(a * _log_R(R @ self.R.T)) @ self.R
        self.w = min(self.w + conf, 6.0)
        self.n += 1
        self.last_t = t
        self.conf_last = conf
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
        if self.cfg.fast:                      # speed preset (the original numbers are the fast=False branch)
            self.hs.v_max, self.hs.w_max = 0.40, 1.2
            self.hs.gate_integral, self.hs.ki = True, 2.5
            self.cfg.appr_speed = 0.12
            self.cfg.close_speed = 0.10
            self.cfg.min_conf_n = 2
            self.cfg.t_tug_up, self.cfg.t_tug_down, self.cfg.t_tug_sample = 0.8, 0.6, 0.25
            self.cfg.payload_win = (0.4, 0.8)
        self.fusion = FU.TrayFusion()
        self.external = None                   # callable (t, sim) -> ExternalTrayPose | None, see fuse.py
        self.fusion_op = FU.TrayFusion()       # OPERATOR-side fusion (ALIGN cue): external tracker + the wrist-camera memory
        self._al_k = -1
        self._ext_cache = (-1, None)
        self.adv = {s: 0.0 for s in SIDES}     # extra insertion of the hand [m] from the finger-stall contact cue
        self.reinserts = 0
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
                                sum(m.body_mass[m.body(f"ball_{i}").id] for i in range(sim.rig.n_beads)))      # TRUE mass: privileged feed-forward + eval only
        self.gsens = PR.GripperSensor(sim, self.fa, self.fact, seed=self.cfg.seed)
        self.pest = None                       # PayloadEstimator (proprio), created on first use (needs sim.ctl)
        self.grasp_rel = {}                    # side -> (p (3,), R (3,3)): tray pose in the gripper_base frame, from the frozen wrist-camera estimate
        self.payload_est = None
        self.verify_samples = []
        self._gap_prev = {}
        self.fail_log_proprio = []
        self._pas_k = -1
        self._pas_q = {s: deque() for s in SIDES}
        self._pas_est = {}                     # side -> (t, p, R) latest accepted passive estimate

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

    def tray_from_hands(self):
        """tray pose (world) from the hands' forward kinematics (arm encoders) and the grasp offset measured by the wrist camera:
        (p (3,), R (3,3), disagreement between the hands [m]).  None until a grasp offset exists."""
        if len(self.grasp_rel) < 2:
            return None
        ps, Rs = [], []
        for s in SIDES:
            gp, gR = self._hand_world(s)
            rp, rR = self.grasp_rel[s]
            ps.append(gp + gR @ rp)
            Rs.append(gR @ rR)
        w = _log_R(Rs[1] @ Rs[0].T)
        return 0.5 * (ps[0] + ps[1]), _rotvec_to_R(0.5 * w) @ Rs[0], float(np.linalg.norm(ps[0] - ps[1]))

    def _payload_point(self):
        """world point of the payload for the FL-MPC feed-forward: privileged = true tray body; proprio = hand FK + grasp offset."""
        if self.cfg.proprio:
            r = self.tray_from_hands()
            if r is not None:
                return r[0]
            return 0.5 * (self._hand_world("left")[0] + self._hand_world("right")[0])
        return self.sim.d.xpos[self.sim.tray].copy()

    def tray_estimate_for_operator(self):
        """(p, R) world tray estimate for the operator-side grasp error / can_grasp (proprio): after a verified grasp = hand FK + offset,
        before that = the latest wrist-camera estimate near the hands' prior; None if there is none."""
        if self.state == "HOLD":
            r = self.tray_from_hands()
            return None if r is None else (r[0], r[1])
        return self._passive_estimate()

    def _passive_estimate(self, max_age=0.6, period=None):
        """IDLE-state wrist-camera tray estimate for the operator (5 Hz, 3-tick latency): prior = nominal relpose on the CURRENT hand pose, the
        same estimator as PRE.  Mean over the sides with a fresh confident estimate; None if no side has one.  Once per sim tick."""
        sim, cfg = self.sim, self.cfg
        if self.state in ACTIVE:
            return None
        if sim.k != self._pas_k:
            self._pas_k = sim.k
            if sim.k % (period or cfg.cam_period) == 0:
                for s in SIDES:
                    gp, gR = self._hand_world(s)
                    pr_p, pr_R = U.prior_tray_pose(sim.of, s, gp, gR)
                    cam = self.cams[s]
                    roi = cam.roi_for_box(sim.d, pr_p + pr_R @ np.array([0, 0, 0.01]), pr_R, (HX + 0.12, HY + 0.12, 0.07))
                    fr = cam.capture(sim.d, self.n_cap, seed=cfg.seed, t=sim.t, roi=roi)
                    self.n_cap += 1
                    self._pas_q[s].append((sim.k + cfg.latency_ticks, fr, pr_p, pr_R))
            for s in SIDES:
                q = self._pas_q[s]
                while q and q[0][0] <= sim.k:
                    _, fr, pr_p, pr_R = q.popleft()
                    est = E.estimate_tray(fr, pr_p, pr_R, mode=cfg.seg, tray_geoms=self.tray_geoms, seed=self.n_cap, z_weight=cfg.z_weight, bounds_xy=cfg.bounds_xy, z_lo=cfg.z_lo, bounds_z=cfg.bounds_z)
                    if est.ok and est.conf >= cfg.conf_min and (np.linalg.norm(est.p - pr_p) < cfg.anchor_gate[0] and
                                                                  np.linalg.norm(_log_R(est.R @ pr_R.T)) < cfg.anchor_gate[1]):
                        self._pas_est[s] = (float(fr.t), est.p.copy(), est.R.copy())
        fresh = [v for v in self._pas_est.values() if sim.t - v[0] <= max_age]
        if not fresh:
            return None
        p = np.mean([v[1] for v in fresh], 0)
        R0 = fresh[0][2]
        w = np.mean([_log_R(v[2] @ R0.T) for v in fresh], 0)
        return p, _rotvec_to_R(w) @ R0

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

    def _robot_goal(self, k, s, p_tray, R_tray, retreat, lift, side=0.0):
        """hand goal in the ObjectFrame robot frame for a tray pose (world), pulled back by `retreat` along the approach axis, raised `lift`."""
        sim = self.sim
        rel, Rn = U.nominal_T_g_tray(sim.of, s)
        Rg = R_tray @ Rn.T
        pg = p_tray - Rg @ rel - Rg[:, 2] * retreat + np.array([0.0, 0.0, lift]) + Rg[:, 0] * side
        root = self.m.body("yor_root").id
        Rr = sim.d.xmat[root].reshape(3, 3)
        return Rr.T @ (pg - sim.d.xpos[root]) - np.array([0.0, 0.0, sim.lift_actual()]), Rr.T @ Rg

    def _tray_est(self, s):
        """current best tray pose (world) for side s: the FUSED estimate (both hands + any external tracker) if cfg.fuse and one exists, else this
        side's own filter, else the start prior."""
        if self.cfg.fuse:
            fz = self._fused()
            if fz is not None:
                return fz[0], fz[1]
        f = self.filt[s]
        return (f.p, f.R) if f.p is not None else self.prior[s]

    def set_external(self, src):
        """connect an external tray-pose source: callable (t, sim) -> fuse.ExternalTrayPose | None, polled at the camera rate while the assist is
        in PRE / APPROACH (None disconnects)."""
        self.external = src

    def _fused(self):
        """(p, R, cov_tray, sources) of the information-fused tray pose, or None.  Each wrist filter enters with its calibrated per-axis sigma
        (fuse.WRIST_SIGMA), inflated for low confidence and shrunk (not to zero: the bias does not average) with the number of captures."""
        cov = {}
        for s in SIDES:
            f = self.filt[s]
            if f.p is not None and f.n >= 1:
                sc = math.sqrt(0.4 + 0.6 / f.n)
                cov[s] = FU.wrist_cov(s, f.conf_last, sc, look=self.cfg.look)
            else:
                self.fusion.drop(s)
        if len(cov) == 2:
            # CONSISTENCY: an axis on which the two hands disagree by more than 2.5 sigma combined is one of them being blind there (e.g. the left camera
            # loses the edge when the tray sits 2 cm closer than expected: closer than its 0.10 m minimum range, so the edge-normal position is just the
            # prior).  Inflate the less certain hand's variance on that axis so the fused value follows the other one.
            fl, fr = self.filt["left"], self.filt["right"]
            R0 = E.level_R(fl.R)
            d = np.concatenate([R0.T @ (fl.p - fr.p), R0.T @ _log_R(fl.R @ fr.R.T)])
            sl, sr = np.sqrt(np.diag(cov["left"])), np.sqrt(np.diag(cov["right"]))
            for ax in (0, 1, 2, 5):
                sc_ = math.hypot(sl[ax], sr[ax])
                if abs(d[ax]) > 2.5 * sc_:
                    worse = "left" if sl[ax] >= sr[ax] else "right"
                    cov[worse][ax, ax] *= (abs(d[ax]) / (2.5 * sc_)) ** 2
                    self.metrics_["n_inflated"] = self.metrics_.get("n_inflated", 0) + 1
        for s in cov:
            self.fusion.set(s, self.filt[s].p, self.filt[s].R, cov[s], self.filt[s].last_t)
        return self.fusion.estimate(t=self.sim.t, max_age_s=self.cfg.max_age_s)

    def _ext_latest(self):
        """poll the external tracker at most once per sim tick (the PRE fusion and the operator cue share its samples)."""
        if self.external is None:
            return None
        if self._ext_cache[0] != self.sim.k:
            self._ext_cache = (self.sim.k, self.external(self.sim.t, self.sim))
        return self._ext_cache[1]

    def operator_tray_estimate(self):
        """OPERATOR ALIGN CUE without the tray body (sim: BIMANUAL_ALIGN_CUE=cam): (p, R) world or None.
        HOLD: the hands' FK + the measured grasp offset.  Otherwise: the information-fusion of (a) the EXTERNAL tracker, if one is connected
        (set_external: the head-camera FoundationPose tracker, or fuse.MockHeadTracker in the sim), and (b) the wrist-camera tray estimate -- the most
        recent confident one, kept in the world frame while the hands are too close for the camera (min range 0.10 m: blind at the hold geometry; the tray
        is static until gripped).  The wrist camera alone cannot FIND the tray from afar (its estimator needs a prior within ~7 cm), so without an
        external tracker this cue is only available once the hands have been within view range of a prior-consistent tray."""
        sim, cfg = self.sim, self.cfg
        if self.state == "HOLD":
            r = self.tray_from_hands()
            if r is not None:
                return r[0], r[1]
        if sim.k != self._al_k:
            self._al_k = sim.k
            e = self._ext_latest()
            if e is not None:
                self.fusion_op.set("ext", e.p, e.R, e.cov, e.t_meas)
            if self.state not in ACTIVE and sim.k % cfg.align_period == 0:
                est = self._passive_estimate(period=cfg.align_period)
                if est is not None:
                    cov = np.diag(np.mean([FU.WRIST_SIGMA[s] for s in SIDES], 0) ** 2)
                    self.fusion_op.set("wrist", est[0], est[1], cov, sim.t)
        fz = self.fusion_op.estimate(t=sim.t, max_age_s=cfg.align_mem_s)
        return None if fz is None else (fz[0], fz[1])

    def _poll_external(self):
        e = self._ext_latest()
        if e is None:
            return
        a_p = np.mean([self.prior[s][0] for s in SIDES], 0)
        a_R = self.prior["left"][1]
        if np.linalg.norm(e.p - a_p) < 2 * self.cfg.anchor_gate[0] and np.linalg.norm(_log_R(e.R @ a_R.T)) < 2 * self.cfg.anchor_gate[1]:
            self.fusion.set("ext", e.p, e.R, e.cov, e.t_meas)
            self.metrics_.setdefault("ext_n", 0)
            self.metrics_["ext_n"] += 1

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
        self.fusion.reset()
        self.adv = {s: 0.0 for s in SIDES}
        self.reinserts = 0
        # on a regrasp keep the previous best estimate as the prior
        self._fingers_open()
        self.k = 0
        self._set_pre_goals()
        self._goto("PRE")
        sim.ctl.payload[:] = 0.0

    def _pre_pose(self, k):
        """(pull-back [m], raise [m], lateral shift [m]) of the PRE servo goal: the hover pose, shifted along the edge for the lateral look."""
        c = self.cfg
        return (c.retreat[k], c.lift, c.look_side[k] if c.look else 0.0)

    def _set_pre_goals(self):
        for k, s in enumerate(SIDES):
            p, R = self._tray_est(s)
            back, up, side = self._pre_pose(k)
            gp, gR = self._robot_goal(k, s, p, R, back, up, side)
            self.hs.set_goal(k, gp, gR)

    # ------------------------------------------------------------------ PLACE sequence
    def _start_place(self):
        sim = self.sim
        c = sim.ctl
        self.td = PL.TouchdownDetector(self.m, c.ld, c.rd, [self.m.body(f"{s}_arm_gripper_base").id for s in SIDES], seed=self.cfg.seed, frac=self.cfg.place_frac)
        self.place_mode = "lift" if sim.lift_actual() > 0.03 else "jog"        # the lift axis is at its lower stop at carry height: lower with the arms (tray-frame jog) then
        self.hand_z0 = self._hand_z()
        self.place = dict(t0=float(sim.t), lift0=sim.lift_actual(), mode=self.place_mode, tray_z0=float(sim.d.xpos[sim.tray][2]), beads0=int(sim.bead_out.sum()),
                          true_touch_t=None, detect_t=None, desc_at_detect=None, m_ref=None, ok=None, reason="")
        self.metrics_["place"] = self.place
        self.td_desc = 0.0
        self._goto("PLACE_LOWER")

    def _hand_z(self):
        return float(0.5 * (self._hand_world("left")[0][2] + self._hand_world("right")[0][2]))

    def _tray_supported(self):
        """EVAL ONLY: is the tray in contact with a table slab (tp_*) -- the ground truth of the touchdown the torque detector has to find."""
        d, m = self.sim.d, self.m
        tg = self.tray_geoms
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in tg) != (g2 in tg):
                o = g2 if g1 in tg else g1
                if m.body(m.geom_bodyid[o]).name.startswith("tp_"):
                    return True
        return False

    def _end_place(self, ok, reason=""):
        self.place["ok"], self.place["reason"] = bool(ok), reason
        self.place["t_total"] = float(self.sim.t - self.place["t0"])
        self.state = "IDLE"

    def _tick_place(self, dt):
        sim, cfg = self.sim, self.cfg
        pl = self.place
        st = self.state
        self.t_state += dt
        if st == "PLACE_LOWER":
            desc = self.hand_z0 - self._hand_z()
            self.td_desc = desc
            if pl["true_touch_t"] is None and self._tray_supported():
                pl["true_touch_t"] = float(sim.t)
            self.td.end_tick(sim.d, self._payload_point(), desc if self.t_state >= 0.3 else 0.0)
            pl["m_ref"] = self.td.m_ref
            if self.td.fired:
                pl["detect_t"], pl["desc_at_detect"] = float(sim.t), float(desc)
                pl["m_at_detect"] = float(self.td.m_lp)
                sim.lift_h = float(sim.lift_actual())                      # stop pushing: the lift reference to where the lift is
                sim.lift_v = 0.0
                self._fingers_open()
                self._goto("PLACE_OPEN")
            elif desc > cfg.place_max_drop:
                sim.lift_h = float(sim.lift_actual())
                sim.lift_v = 0.0
                self.state = "HOLD"
                self._end_place(False, "no touchdown within %.2f m" % cfg.place_max_drop)
                self.state = "HOLD"
        elif st == "PLACE_OPEN":
            if self.t_state >= cfg.place_open_s:
                sim.grasped = False
                sim.ctl.tilt = False
                sim.ctl.payload[:] = 0.0
                self.hs.start_from_current()
                for k, s in enumerate(SIDES):
                    p, R = self.hs.hand_pose(k)
                    self.hs.set_goal(k, p - R[:, 2] * cfg.retreat[k], R)
                self.hs.v_max_saved = self.hs.v_max
                self.hs.v_max = cfg.place_retreat_speed
                self._goto("PLACE_RETREAT")
        elif st == "PLACE_RETREAT":
            if self._both_converged(0.004, math.radians(4.0)) and self.t_state > 0.3 or self.t_state > 4.0:
                self.hs.v_max = self.hs.v_max_saved
                self._goto("PLACE_SETTLE")
        elif st == "PLACE_SETTLE":
            if self.t_state >= 0.6:
                d = sim.d
                tilt = sim.tray_tilt_deg()
                v = float(np.linalg.norm(d.qvel[sim.tray_adr:sim.tray_adr + 3]))
                nb = int(sim.bead_out.sum()) - pl["beads0"]
                pl.update(tray_tilt_deg=float(tilt), tray_speed_mps=v, beads_spilled=nb, tray_dz_mm=float((d.xpos[sim.tray][2] - pl["tray_z0"]) * 1e3),
                          supported=bool(self._tray_supported()), cup_lost=sim.cup_lost_t is not None)
                ok = tilt < 3.0 and v < 0.02 and nb == 0 and pl["supported"] and pl["detect_t"] is not None
                gm = [self.gap(s) for s in SIDES]
                pl["finger_gap_mm"] = [g * 1e3 for g in gm]
                # the ObjectFrame (operator jog) continues from the retreated hands
                mid = 0.5 * (self._hand_world("left")[0] + self._hand_world("right")[0])
                self.hs.rebase_object_frame(tray_world=mid)
                self._end_place(ok, "" if ok else "tilt %.1f speed %.3f beads %d supported %s" % (tilt, v, nb, pl["supported"]))
        if self.hs.active:
            self.hs.step(dt)

    def release(self, force=False):
        if self.state in PLACE_STATES:                  # a second release during the place sequence: open NOW (operator override)
            force = True
            self.hs.v_max = getattr(self.hs, "v_max_saved", self.hs.v_max)
        elif self.cfg.place_on_release and self.state == "HOLD" and not force:
            self._start_place()
            return
        sim = self.sim
        self._fingers_open()
        sim.ctl.payload[:] = 0.0
        sim.grasped = False
        sim.ctl.tilt = False
        self.hs.active = False
        if self.state in ("HOLD", "FAILED", "ABORT") or self.state in ACTIVE or self.state in PLACE_STATES:
            self.state = "IDLE"

    def gate(self, cmd):
        """hold the base / lift / jog while the assisted grasp is in progress; lift only after both hands are verified."""
        if self.state in ACTIVE or self.state in ("FAILED", "ABORT") or self.state in PLACE_STATES:
            from world_model.v2.house.teleop.sim import Cmd
            lowering = self.state == "PLACE_LOWER" and self.t_state >= 0.3             # 0.3 s hold dwell first: the load calibration of the touchdown detector
            lift_v = -self.cfg.place_speed if (lowering and self.place_mode == "lift") else 0.0
            jog = np.array([0.0, 0.0, -self.cfg.place_speed, 0.0, 0.0, 0.0]) if (lowering and self.place_mode == "jog") else np.zeros(6)
            return Cmd(twist=np.zeros(3), lift_v=lift_v, jog=jog, grasp=cmd.grasp, release=cmd.release)
        return cmd

    def set_feedforward(self, v, acc, yaw):
        r = np.array(self.sim.rig.info["tray_local"][:2], float)
        a = BS.cup_accel_body(v, acc, r)
        c, s = math.cos(yaw), math.sin(yaw)
        self.sim.ctl.a_world[:] = (c * a[0] - s * a[1], s * a[0] + c * a[1], 0.0)
        if self.sim.ctl.payload.any():
            self.sim.ctl.payload_pt = self._payload_point()

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
        if self.cfg.proprio:
            self._pestimator().accumulate_substep(d)        # measured arm torque (JointState.torque stand-in): the applied FL-MPC torque
        if self.state == "PLACE_LOWER":
            self.td.accumulate_substep(d)

    def _pestimator(self):
        if self.pest is None:
            c = self.sim.ctl
            self.pest = PR.PayloadEstimator(self.m, c.ld, c.rd, [self.m.body(f"{s}_arm_gripper_base").id for s in SIDES], seed=self.cfg.seed)
        return self.pest

    # ------------------------------------------------------------------ per tick (50 Hz)
    def tick_update(self):
        sim, cfg = self.sim, self.cfg
        dt = 0.02
        if cfg.proprio:                                      # close the per-tick torque average (and collect inside the payload window)
            self._pestimator().end_tick(sim.d, self._payload_point(), collect=(self.state == "VERIFY" and self.tug_phase == 1 and
                                                                                 cfg.payload_win[0] <= self.tug_t0 < cfg.payload_win[1]))
        if self.state == "HOLD":
            self._log_slip()
            return
        if self.state in PLACE_STATES:
            self._tick_place(dt)
            return
        if self.state not in ACTIVE:
            return
        self.t_state += dt
        self.k += 1
        if self.state in ("PRE", "APPROACH") and self.k % cfg.cam_period == 0 and not cfg.disable_est:
            self._capture()
            if self.state == "PRE":
                self._poll_external()
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
            a_p, a_R = self.prior[s]                          # the ROI follows the ANCHOR prior (never the possibly-wrong running estimate)
            # fast mode: the estimator's weak prior is the ANCHOR too.  With the running (fused) estimate as the prior a wrong plane lock re-enters the next
            # fit as 'evidence' (measured: a right-hand lock 46 mm / 12 mm off persisted for 10 s); the anchor is what the offline error tables assume
            pr_p, pr_R = (a_p, a_R) if cfg.fast else self._tray_est(s)
            roi = cam.roi_for_box(sim.d, a_p + a_R @ np.array([0, 0, 0.01]), a_R, (HX + 0.12, HY + 0.12, 0.07))
            fr = cam.capture(sim.d, self.n_cap, seed=cfg.seed, t=sim.t, roi=roi)
            self.n_cap += 1
            settled = all(self.hs.goal_error(k)[0] < 0.012 and self.hs.goal_error(k)[1] < 10.0 for k in (0, 1))
            self.queue[s].append((self.k + cfg.latency_ticks, fr, pr_p.copy(), pr_R.copy(), settled))

    def _deliver(self):
        sim, cfg = self.sim, self.cfg
        for k, s in enumerate(SIDES):
            q = self.queue[s]
            while q and q[0][0] <= self.k:
                _, fr, pr_p, pr_R, settled = q.popleft()
                est = E.estimate_tray(fr, pr_p, pr_R, mode=cfg.seg, tray_geoms=self.tray_geoms, seed=self.n_cap, z_weight=cfg.z_weight, bounds_xy=cfg.bounds_xy, z_lo=cfg.z_lo, bounds_z=cfg.bounds_z)
                if cfg.fast and est.ok and cfg.seg == "depth":
                    est.p = est.p - cfg.x_bias * est.R[:, 0]
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
                    # a frame taken while the hand was still moving to the look pose sees the edge only partly: the weakly observed axes then just
                    # return the prior with a confident-looking small spread -- count only frames taken with the hands settled on the PRE goal
                    if est.conf >= cfg.conf_min and near_anchor and self.state in ("PRE", "APPROACH") and (settled or not cfg.fast):
                        self.filt[s].update(est.p, est.R, est.conf, fr.t, gate=((0.02, math.radians(6.0)) if cfg.fast else (0.06, math.radians(15.0))))
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
        have = all(self.filt[s].n >= cfg.min_conf_n for s in SIDES) or cfg.disable_est
        pre_tol = (0.012, math.radians(10.0)) if cfg.look else cfg.pre_tol        # the look pose only has to keep the tray in view: FK gives the camera pose
        if have and cfg.fast and not self._hands_agree() and self.t_state < cfg.t_pre_max - 2.0:
            have = False                          # the two hands see different trays: keep looking (a wrong plane lock is persistent but rarely agrees with the other hand)
        if have and self._both_converged(*pre_tol) and self.t_state > (0.3 if cfg.fast else 1.0):
            # freeze the estimate; approach on it (disable_est: the hands' own prior -- a deliberately uninformed grasp)
            self.frozen = {s: self._freeze_pose(s) for s in SIDES}
            self.appr_t0 = 0.0
            self.t_fin = 0.0
            self._goto("APPROACH")
            return
        if self.t_state > cfg.t_pre_max:
            if have:
                self._fail("pre-grasp did not converge")
            else:
                self._fail("low wrist-camera confidence (n=%s)" % {s: self.filt[s].n for s in SIDES})

    def _hands_agree(self):
        """left / right wrist estimates grossly consistent (edge normal 40 mm, along edge 60 mm, height 6 mm, yaw 6 deg)?"""
        fl, fr = self.filt["left"], self.filt["right"]
        if fl.p is None or fr.p is None:
            return True
        R0 = E.level_R(fl.R)
        d = R0.T @ (fl.p - fr.p)
        yaw = abs(_log_R(fl.R @ fr.R.T)[2])
        ok = abs(d[0]) < 0.040 and abs(d[1]) < 0.060 and abs(d[2]) < 0.006 and yaw < math.radians(6.0)         # gross inconsistency only (a wrong plane / lock); per-axis blindness is handled by the fusion
        if not ok:
            self.metrics_["n_disagree"] = self.metrics_.get("n_disagree", 0) + 1
        return ok

    def _freeze_pose(self, s):
        """the tray pose the blind approach is dead-reckoned on: the fused estimate (cfg.fuse), else this side's filter, else the prior."""
        if self.cfg.fuse:
            fz = self._fused()
            if fz is not None:
                self.metrics_["fused_sources"] = fz[3]
                self.metrics_["fused_sigma_mm"] = [float(x) for x in np.sqrt(np.diag(fz[2]))[:3] * 1e3]
                return fz[0].copy(), fz[1].copy()
        f = self.filt[s]
        return (f.p.copy(), f.R.copy()) if f.p is not None else self.prior[s]

    def _path(self, k):
        """waypoints (pull-back, raise, lateral) of the blind approach for hand k, and the leg speeds [m/s]:
        look pose -> back to the edge centre line (at the hover height) -> insert to the grasp pose."""
        c = self.cfg
        end = (-self.adv[SIDES[k]], 0.0, 0.0)
        if c.look:
            return [(c.retreat[k], c.lift, c.look_side[k]), (c.retreat[k], c.lift, 0.0), end], [c.v_transit, c.appr_speed]
        return [(c.retreat[k], c.lift, 0.0), end], [c.appr_speed]

    def _path_time(self, k):
        W, V = self._path(k)
        return float(sum(np.linalg.norm(np.array(W[i + 1]) - np.array(W[i])) / V[i] for i in range(len(V))))

    @staticmethod
    def _along(W, V, t):
        """(pull-back, raise) after t seconds along the polyline W with leg speeds V; the 3rd value is True once the end is reached."""
        for i, v in enumerate(V):
            a, b = np.array(W[i]), np.array(W[i + 1])
            L = float(np.linalg.norm(b - a))
            T = L / v
            if t <= T or i == len(V) - 1:
                f = 1.0 if T <= 1e-9 else min(1.0, t / T)
                return tuple(a + f * (b - a)), (i == len(V) - 1 and t >= T)
            t -= T
        return tuple(W[-1]), True

    def _do_approach(self):
        cfg = self.cfg
        self.appr_t0 += 0.02
        if cfg.look or cfg.fast:
            done_all = True
            for k, s in enumerate(SIDES):
                p, R = self.frozen[s]
                W, V = self._path(k)
                (back, up, side), fin = self._along(W, V, self.appr_t0)
                done_all &= fin
                gp, gR = self._robot_goal(k, s, p, R, back, up, side)
                self.hs.set_goal(k, gp, gR)
            self.t_fin = self.t_fin + 0.02 if done_all else 0.0
            # the goal has stopped: let the integral settle the residual.  A rotation the arm cannot reach (the hold pose sits on the wrist-5 limit, so a
            # yawed tray is only partly followable) must not stall the grasp: position converged + a still hand for 0.8 s is accepted up to 8 deg
            done = done_all and ((self.t_fin >= 0.2 and self._both_converged(0.003, math.radians(2.5))) or
                                 (self.t_fin >= 0.8 and self._both_converged(0.004, math.radians(8.0))))
            tmax = cfg.t_appr_max + max(self._path_time(k) for k in (0, 1))
        else:                                                                         # ---- the original approach (bit-for-bit)
            for k, s in enumerate(SIDES):
                p, R = self.frozen[s]
                f = max(0.0, 1.0 - self.appr_t0 * cfg.appr_speed / cfg.retreat[k])        # 1 -> 0
                lift = cfg.lift * f
                gp, gR = self._robot_goal(k, s, p, R, cfg.retreat[k] * f, lift)
                self.hs.set_goal(k, gp, gR)
            done = self.appr_t0 * cfg.appr_speed >= max(cfg.retreat) and self._both_converged(0.004, math.radians(3.0))
            tmax = cfg.t_appr_max + max(cfg.retreat) / cfg.appr_speed
        if done and self.t_state > (0.3 if cfg.fast else 1.0):
            self._final_pose_error()
            self._fingers_close()
            self._goto("CLOSE")
            return
        if self.t_state > tmax:
            self._fail("approach did not converge")

    def _final_pose_error(self):
        """EVAL ONLY (reads the tray body): the hand-to-tray pose error at the moment the pads start closing, in the nominal-grasp frame
        [approach (edge-normal), along-edge, vertical] mm and the rotation error [deg] -- what the blind approach left of the estimate error."""
        out = {}
        for s in SIDES:
            r, Rr = self._hand_rel(s)
            rel, Rn = U.nominal_T_g_tray(self.sim.of, s)
            e = r - rel
            out[s] = dict(approach_mm=float(e[2] * 1e3), along_mm=float(e[0] * 1e3), vert_mm=float(e[1] * 1e3),
                          rot_deg=float(math.degrees(np.linalg.norm(_log_R(Rr @ Rn.T)))), yaw_deg=float(math.degrees(_log_R(Rr @ Rn.T)[1])))
        self.metrics_.setdefault("final_err", []).append(out)

    def _do_close(self):
        cfg = self.cfg
        self.t_close += 0.02
        if cfg.grip_mode == "force":
            self.f_cmd = cfg.f_grip * min(1.0, self.t_close / cfg.close_ramp_s)
        else:
            self.finger_q = max(0.0, self.finger_q - cfg.close_speed * 0.02)         # target slews to 'closed'; the plate stops the pads
        d = self.sim.d
        settled = self.t_close > (cfg.close_ramp_s if cfg.grip_mode == 'force' else cfg.open_q / cfg.close_speed) + (0.35 if cfg.fast else 0.5)
        if settled:
            gaps = {s: (self.gsens.gap(s) if cfg.proprio else self.gap(s)) for s in SIDES}        # proprio: noisy, quantised finger encoder
            self.metrics_["gap"] = gaps
            asym = {s: 0.5 * float(d.qpos[self.fa[s][0]] + d.qpos[self.fa[s][1]]) for s in SIDES}     # finger stall asymmetry: >0 = upper jaw stopped later
            self.metrics_["finger_asym_mm"] = {s: a * 1e3 for s, a in asym.items()}
            if cfg.fast and self._contact_cue(gaps, asym):
                return
            if all(cfg.gap_range[0] <= g <= cfg.gap_range[1] for g in gaps.values()):
                for k, s in enumerate(SIDES):
                    g = self.hs.hand_pose(k)
                    self.hs.set_goal(k, *g, jump=True)       # hold where the hands are
                self.tug_phase = 0
                self.tug_t0 = 0.0
                self.contact_samples = {s: [] for s in SIDES}
                if cfg.proprio:
                    # grasp offset (tray in the gripper frame) from the frozen wrist-camera estimate + the hands' FK; the payload is UNKNOWN
                    # until it is estimated from the measured joint torque during the tug hold (no true masses / tray body read)
                    for s in SIDES:
                        gp, gR = self._hand_world(s)
                        fp, fR = self.frozen[s]
                        self.grasp_rel[s] = (gR.T @ (fp - gp), gR.T @ fR)
                    self.sim.ctl.payload[:] = 0.0
                    self.sim.ctl.payload_pt = self._payload_point()
                    self._pestimator().reset()
                    self.verify_samples = []
                    self.payload_est = None
                else:
                    self.sim.ctl.payload[:] = self.payload_kg / 2.0
                    self.sim.ctl.payload_pt = d.xpos[self.sim.tray].copy()
                self._goto("VERIFY")
            else:
                self._fail("closed on nothing / wrong gap %s" % {s: round(g * 1e3, 1) for s, g in gaps.items()})

    def _contact_cue(self, gaps, asym):
        """CONTACT CUE from the finger stall (gap + asymmetry, finger encoders only).  A gap that stalled on the bare 6 mm plate (< strip_gap) instead
        of the 12 mm plate+strip means the pads closed INBOARD of the handle strip (the estimate left the hand short): open, insert `reinsert_step`
        further, close again.  Returns True when a re-insert was started.  The stall asymmetry is logged (the upper / lower jaw stopping at different
        positions = a vertical offset of the pads about the plate, which the plate's own sliding re-centres); it is not acted on."""
        cfg = self.cfg
        short = [s for s in SIDES if cfg.gap_range[0] <= gaps[s] < cfg.strip_gap]
        if not short or self.reinserts >= cfg.max_reinsert:
            return False
        self.reinserts += 1
        self.metrics_["reinserts"] = self.reinserts
        self.metrics_.setdefault("reinsert_log", []).append((float(self.sim.t), {s: round(g * 1e3, 1) for s, g in gaps.items()}, short))
        for s in short:
            self.adv[s] += cfg.reinsert_step
        self._fingers_open()
        self.sim.ctl.payload[:] = 0.0
        self.appr_t0 = 1e3                           # the path is already at its end: only the end waypoint moved
        self.t_fin = 0.0
        self._goto("APPROACH")
        return True

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

    def _verify_gripper(self):
        """PROPRIO verify over the averaging window from the finger encoders + motor effort only: per hand the gap stays inside the
        plate-thick range in >= 90 % of the samples, STALLS (|gap slope| <= gap_slope: the jaws are held off 'closed' by something between the
        pads; robust to the encoder noise and the ~1 mm pad flicker, unlike a min-max spread) and the mean summed finger effort >= f_min_eff.  A grasp that closed on nothing shows gap -> 0 and effort -> 0."""
        cfg = self.cfg
        res, ok = {}, True
        for k, s in enumerate(SIDES):
            smp = np.array([[r[0][k], r[1][k]] for r in self.verify_samples])           # (n, 2): gap, effort
            if len(smp) == 0:
                return False, {s_: (0.0, 0.0, 0.0) for s_ in SIDES}
            gaps, eff = smp[:, 0], smp[:, 1]
            in_rng = float(((gaps >= cfg.gap_range[0]) & (gaps <= cfg.gap_range[1])).mean())
            t = np.arange(len(gaps)) * 0.02
            slope = float(np.polyfit(t, gaps, 1)[0]) if len(gaps) > 3 else 0.0
            res[s] = (float(gaps.mean()), float(eff.mean()), slope)
            ok &= (in_rng >= 0.9 and abs(slope) <= cfg.gap_slope and float(eff.mean()) >= cfg.f_min_eff)
        return ok, res

    def _cup_view(self):
        """right/left wrist camera RGB-equivalent cup track (u, v in px) -> dict side -> (n_px, u, v, range) for the cameras that see the cup."""
        out = {}
        for s in SIDES:
            r = (self.cams[s].seg_cup_proprio(self.sim.d, tick=self.k, seed=self.cfg.seed) if self.cfg.proprio
                 else self.cams[s].seg_cup(self.sim.d))
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
            if cfg.proprio:
                self.verify_samples.append(([self.gsens.gap(s) for s in SIDES], [self.gsens.effort(s) for s in SIDES]))
            else:
                for s in SIDES:
                    self.contact_samples[s].append(self.pad_contacts(s))
            if self.tug_t0 < cfg.t_tug_sample:
                return
            if cfg.proprio:
                ok, res = self._verify_gripper()
                self.metrics_["grip_gap_m"] = {s: r[0] for s, r in res.items()}
                self.metrics_["grip_effort_N"] = {s: r[1] for s, r in res.items()}
                self.metrics_["grip_gap_slope_mps"] = {s: r[2] for s, r in res.items()}
                self.metrics_["verify_by"] = "gripper_gap_effort"
                if not ok:
                    self._fail("gripper gap/effort verify failed %s" % {s: (round(r[0] * 1e3, 1), round(r[1], 1), round(r[2] * 1e3, 1)) for s, r in res.items()})      # (gap mm, effort N, slope mm/s)
                    return
            else:
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
            if self.tug_t0 < cfg.t_tug_up:
                return
            true_rise = float(sim.d.xpos[sim.tray][2]) - self.tray_z0
            self.metrics_["tug_rise_true_mm"] = true_rise * 1e3
            if cfg.proprio:
                mh, ns = self._pestimator().estimate()
                self.payload_est = mh
                self.metrics_["payload_est_kg"] = [float(x) for x in mh]
                self.metrics_["payload_est_n"] = [int(x) for x in ns]
                self.metrics_["payload_true_kg"] = float(self.payload_kg)          # EVAL only
                self.metrics_["payload_err_frac"] = float(np.nansum(mh) / self.payload_kg - 1.0)
                if not np.isfinite(mh).all():
                    self._fail("payload estimate unavailable (arms not still / no torque samples) n=%s" % list(ns))
                    return
                sim.ctl.payload[:] = np.maximum(mh, 0.0)
                sim.ctl.payload_pt = self._payload_point()
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
            if self.tug_t0 < cfg.t_tug_down:
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
        if self.cfg.proprio:
            tf = self.tray_from_hands()
            self.hs.rebase_object_frame(tray_world=tf[0])        # tray centre from hand FK + the wrist-camera grasp offset (not the tray body)
            self.metrics_["tray_fk_err_mm"] = float(np.linalg.norm(tf[0] - self.sim.d.xpos[self.sim.tray]) * 1e3)      # EVAL only
        else:
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
