"""teleop/operator.py -- SIMULATED operator command streams (+ an optional live keyboard mode).  SIMULATION ONLY.

"Teleop" here is a command-stream generator.  Nothing in this file talks to a device-to-robot link, a robot driver or hardware.

SloppyOperator   closed-loop human model.  It sees the robot through a delayed observation (reaction delay), steers a pure-pursuit
                 style heading controller at a nominal route (the nav-planned polyline of the scene) and perturbs it the way a hurried
                 teleoperator does:
                   * stick saturation (|stick| <= 1 -> full-scale twist base.V_STICK) and a dead-zone,
                   * abrupt starts / stops: the forward stick is bang-bang (full cruise, snap to 0 when the operator thinks he is close),
                   * overshooting turns: high heading gain acting on a DELAYED heading error, forward speed barely reduced in turns,
                   * corner cutting: waypoints are 'reached' at a random radius 0.5-1.1 m and the aim point has a per-waypoint lateral
                     bias (N(0, 0.15 m)); near doors/people the operator does not slow down,
                   * jerky stick: OU noise + Poisson stick bumps (vx / wz / vy cross-talk) and tray-frame jog cross-talk (roll/pitch
                     thumb bumps) while driving,
                   * reverse driving when the next waypoint is behind (backing the tray out of the counter / along the corridor).
                 Deterministic per seed (all random draws come from one np.random.default_rng(seed) in a fixed order, one draw set per
                 control tick).  The command depends on the observed robot state (closed loop), so "the same operator" = same seed.
rollout_stream   runs a SloppyOperator against a GHOST base (hardware slew only, no tray/shield) and records the commands:
                 an OPEN-LOOP stream (OperatorStream), replayed by ReplayOperator -- bit-identical commands for raw and shielded runs.
KeyboardOperator live mode: key_callback for mujoco.viewer.launch_passive (latching sticks); `inject(keycode)` for headless tests.

Task program (task='coffee'): OUT (drive to the counter) -> ALIGN (tray-frame jog until the grippers are within grasp tolerance) -> GRASP
(weld activation) -> LIFT -> BACK (reverse route to the lab desk) -> LOWER -> RELEASE -> DONE.   task='carry': OUT only (the tray is
already held; stop at the pick pose), used for the short smoke.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import base as BS
from .sim import Cmd, TICK


# ------------------------------------------------------------------------------------------------ params
@dataclass
class OperatorParams:
    v_cruise: float = 0.95          # forward stick at cruise (fraction of full scale)
    v_back: float = 0.55            # reverse cruise
    reaction_s: float = 0.30        # observation delay
    kw: float = 2.2                 # heading gain [stick/rad]
    turn_slow: float = 1.2          # |heading error| (rad) at which vx stick falls to vx_min_turn
    vx_min_turn: float = 0.45
    sw_radius: tuple = (0.5, 1.1)   # waypoint switch radius range (corner cutting)
    lat_bias_sigma: float = 0.15    # per-waypoint lateral aim bias [m]
    a_human: float = 0.9            # deceleration the operator believes in [m/s^2] (-> release distance)
    stop_radius: float = 0.12
    stop_yaw: float = 0.12
    ou_sigma: float = 0.06
    ou_tau: float = 0.4
    bump_rate: float = 0.6          # stick bumps per second
    bump_mag: tuple = (0.4, 0.9)
    bump_len: tuple = (0.15, 0.35)
    jog_bump_rate: float = 0.25     # tray-frame tilt cross-talk bumps per second (while driving)
    jog_bump_rate_rad: float = 0.35  # rad/s
    panic_rate: float = 0.15        # 'oh no' stick releases per second while driving
    panic_len: tuple = (0.3, 0.6)
    deadzone: float = 0.04
    lift_stick: float = 0.8
    lift_up_h: float = 0.06
    lift_jerk: float = 0.5          # fraction of full lift stick bursts (on/off abruptness)
    jog_gain: tuple = (1.5, 1.5)    # [1/s] translation, rotation
    jog_max: tuple = (0.05, 0.25)   # m/s, rad/s


def _sat(x, lim=1.0):
    return float(np.clip(x, -lim, lim))


class SloppyOperator:
    def __init__(self, seed=0, task="coffee", out_route=None, back_route=None, pick_pose=None, place_pose=None,
                 params: OperatorParams | None = None, back_mode="reverse", backout_m=0.7):
        self.p = params or OperatorParams()
        self.rng = np.random.default_rng(seed)
        self.task = task
        self.out_route = [np.array(p, float) for p in out_route]
        self.back_route = [np.array(p, float) for p in (back_route if back_route is not None else list(reversed(out_route)))]
        self.pick_pose, self.place_pose = pick_pose, place_pose
        self.back_mode, self.backout_m, self.back_origin, self._backed = back_mode, backout_m, None, back_mode == "reverse"
        self.phase = "OUT"
        self.buf = []                       # delayed observations
        self.ou = np.zeros(3)
        self.bump = np.zeros(3)             # remaining time, magnitude per axis -> see _bumps
        self.bump_t = np.zeros(3)
        self.bump_v = np.zeros(3)
        self.jbump_t = np.zeros(2)
        self.jbump_v = np.zeros(2)
        self.wp_i = 1
        self.sw_r = self._draw_sw()
        self.bias = self._draw_bias()
        self.t_phase = 0.0
        self.stopped_t = 0.0
        self.panic_t = 0.0
        self.grasp_sent = False
        self.release_sent = False
        self.log = []

    # ---------------------------------------------------------------------------- helpers
    def _draw_sw(self):
        return float(self.rng.uniform(*self.p.sw_radius))

    def _draw_bias(self):
        return float(self.rng.normal(0, self.p.lat_bias_sigma))

    def _delayed(self, obs):
        self.buf.append(obs)
        n = int(round(self.p.reaction_s / TICK))
        return self.buf[max(0, len(self.buf) - 1 - n)]

    def _noise(self):
        """OU stick noise + Poisson bumps for (vx, vy, wz) -- draws are made EVERY tick (fixed order => deterministic)."""
        p = self.p
        a = math.exp(-TICK / p.ou_tau)
        self.ou = a * self.ou + math.sqrt(1 - a * a) * p.ou_sigma * self.rng.standard_normal(3)
        self._panic_draw = (self.rng.random(), self.rng.uniform(*p.panic_len))
        new = self.rng.random(3) < np.array([p.bump_rate, p.bump_rate * 0.5, p.bump_rate]) * TICK
        mag = self.rng.uniform(*p.bump_mag, 3) * self.rng.choice([-1.0, 1.0], 3)
        ln = self.rng.uniform(*p.bump_len, 3)
        for i in range(3):
            if new[i] and self.bump_t[i] <= 0:
                self.bump_t[i], self.bump_v[i] = ln[i], mag[i]
        self.bump_t -= TICK
        b = np.where(self.bump_t > 0, self.bump_v, 0.0)
        # jog cross-talk
        newj = self.rng.random(2) < self.p.jog_bump_rate * TICK
        magj = self.rng.choice([-1.0, 1.0], 2) * self.p.jog_bump_rate_rad
        lnj = self.rng.uniform(0.15, 0.3, 2)
        for i in range(2):
            if newj[i] and self.jbump_t[i] <= 0:
                self.jbump_t[i], self.jbump_v[i] = lnj[i], magj[i]
        self.jbump_t -= TICK
        jb = np.where(self.jbump_t > 0, self.jbump_v, 0.0)
        return self.ou + b, jb

    def _deadzone(self, s):
        return np.where(np.abs(s) < self.p.deadzone, 0.0, s)

    # ---------------------------------------------------------------------------- driving
    def _drive(self, route, reverse, d_obs, final_pose=None):
        """one step of the pure-pursuit-like stick law toward route[self.wp_i]. Returns (stick[3], arrived)."""
        p = self.p
        pos, yaw, v = d_obs["pose"][:2], d_obs["pose"][2], d_obs["v"]
        last = self.wp_i >= len(route) - 1
        tgt = route[min(self.wp_i, len(route) - 1)]
        prev = route[max(self.wp_i - 1, 0)]
        seg = tgt - prev
        seg_n = seg / (np.linalg.norm(seg) + 1e-9)
        nrm = np.array([-seg_n[1], seg_n[0]])
        aim = tgt + nrm * self.bias * (0.0 if last else 1.0)
        dvec = aim - pos
        dist = float(np.linalg.norm(dvec))
        if not last and np.linalg.norm(tgt - pos) < self.sw_r:
            self.wp_i += 1
            self.sw_r, self.bias = self._draw_sw(), self._draw_bias()
        head = math.atan2(dvec[1], dvec[0]) + (math.pi if reverse else 0.0)
        e_psi = BS.wrap(head - yaw)
        stick = np.zeros(3)
        # final approach: stop abruptly when the operator thinks he is within braking distance
        speed = abs(v[0])
        d_rel = speed * speed / (2 * p.a_human) + p.stop_radius
        arrived = False
        if last:
            if dist < d_rel and speed > 0.05:
                fwd = 0.0
            else:
                fwd = 1.0
            if dist < p.stop_radius + 0.05:
                arrived = True
        else:
            fwd = 1.0
        scale = max(p.vx_min_turn, 1.0 - abs(e_psi) / p.turn_slow)
        cruise = (p.v_back if reverse else p.v_cruise)
        stick[0] = (-1.0 if reverse else 1.0) * cruise * scale * fwd
        stick[2] = _sat(p.kw * e_psi)
        if last and dist < 0.35 and abs(e_psi) > 1.0:           # very close to the final pose: don't spin on the spot
            stick[2] = 0.0
            stick[0] = 0.0
        return stick, arrived

    def _creep_stick(self, pp, d_obs, lim=0.25):
        e_w = np.array(pp[:2]) - d_obs["pose"][:2]
        c, s = math.cos(d_obs["pose"][2]), math.sin(d_obs["pose"][2])
        e_b = np.array([c * e_w[0] + s * e_w[1], -s * e_w[0] + c * e_w[1]])
        e_yaw = BS.wrap(pp[2] - d_obs["pose"][2])
        stick = np.array([_sat(2.0 * e_b[0] / BS.V_STICK[0], lim), _sat(2.0 * e_b[1] / BS.V_STICK[1], lim), _sat(2.5 * e_yaw / BS.V_STICK[2], lim)])
        return stick, e_w, e_yaw

    def _heading_to(self, yaw_target, d_obs):
        e = BS.wrap(yaw_target - d_obs["pose"][2])
        return _sat(self.p.kw * e), abs(e) < self.p.stop_yaw

    # ---------------------------------------------------------------------------- the stream
    def step(self, obs) -> Cmd:
        """obs: dict(t, pose[3], v[3], lift_h, grasped, grasp_err (tray_err_p[3] base frame, tray_err_r[3]), can_grasp)."""
        p = self.p
        d_obs = self._delayed(obs)
        noise, jb = self._noise()
        cmd = Cmd()
        stick = np.zeros(3)
        lift = 0.0
        jog = np.zeros(6)
        self.t_phase += TICK
        moving = np.linalg.norm(obs["v"]) > 0.03
        v_obs = obs["v"]
        if self.phase == "OUT":
            stick, arrived = self._drive(self.out_route, False, d_obs)
            if self.pick_pose is not None and self.wp_i >= len(self.out_route) - 1 and \
                    np.linalg.norm(np.array(self.pick_pose[:2]) - d_obs["pose"][:2]) < 0.45:
                stick, e_w0, _ = self._creep_stick(self.pick_pose, d_obs, 0.5)       # last 0.45 m: closed-loop creep (backs up if overshot)
                arrived = bool(np.linalg.norm(e_w0) < 0.12)
            if arrived:
                if self.pick_pose is not None:
                    sw, ok = self._heading_to(self.pick_pose[2], d_obs)
                    stick = np.array([0.0, 0.0, sw])
                else:
                    ok = True
                if ok and not moving:
                    self.stopped_t += TICK
                else:
                    self.stopped_t = 0.0
                if self.stopped_t > 0.6:
                    self._goto("DONE" if self.task == "carry" else "CREEP")
        elif self.phase == "CREEP":
            # final docking: slow closed-loop creep onto the counter stand-off (the wrist sits at a joint limit in the hold pose, so
            # the tray-frame jog can only absorb a few cm / ~1 deg; yaw must be fixed by the base)
            stick, e_w, e_yaw = self._creep_stick(self.pick_pose, d_obs)
            if np.linalg.norm(e_w) < 0.02 and abs(e_yaw) < 0.02 and np.linalg.norm(obs["v"]) < 0.03:
                self.stopped_t += TICK
            else:
                self.stopped_t = 0.0
            if self.stopped_t > 0.5:
                self._goto("ALIGN")
        elif self.phase == "ALIGN":
            ep, er = obs["grasp_err_p"], obs["grasp_err_r"]
            jog[:3] = np.clip(p.jog_gain[0] * np.asarray(ep), -p.jog_max[0], p.jog_max[0])
            jog[3:] = np.clip(p.jog_gain[1] * np.asarray(er), -p.jog_max[1], p.jog_max[1])
            if obs["can_grasp"] and self.t_phase > 0.5:
                cmd.grasp = True
                self._goto("LIFT")
        elif self.phase == "LIFT":
            lift = p.lift_stick * (1.0 if obs["lift_h"] < p.lift_up_h else 0.0)
            if obs["lift_h"] >= p.lift_up_h - 0.004 and self.t_phase > 0.4:
                self._goto("BACK")
                self.wp_i = 1
                self.back_origin = np.array(obs["pose"][:2])
        elif self.phase == "BACK":
            if not self._backed:                # 'backout': reverse straight out of the counter stand-off first (no room to rotate there)
                stick = np.array([-0.5, 0.0, 0.0])
                if np.linalg.norm(obs["pose"][:2] - self.back_origin) >= self.backout_m:
                    self._backed = True
                    self.wp_i = 1
                arrived = False
            else:
                stick, arrived = self._drive(self.back_route, self.back_mode == "reverse", d_obs)
            if arrived:
                sw, ok = self._heading_to(self.place_pose[2], d_obs) if self.place_pose is not None else (0.0, True)
                stick = np.array([0.0, 0.0, sw])
                self.stopped_t = self.stopped_t + TICK if (ok and not moving) else 0.0
                if self.stopped_t > 0.6:
                    self._goto("LOWER")
        elif self.phase == "LOWER":
            lift = -p.lift_stick * (1.0 if obs["lift_h"] > 0.004 else 0.0)
            if obs["lift_h"] <= 0.006 and self.t_phase > 0.4:
                cmd.release = True
                self._goto("DONE")
        # human imperfections on the drive sticks only while a drive phase is active
        driving = self.phase in ("OUT", "BACK")
        if driving:
            if np.abs(stick).sum() > 0.05:                       # stick noise only while the operator is actually pushing
                stick = stick + noise
            if self.panic_t <= 0 and self._panic_draw[0] < p.panic_rate * TICK and np.abs(v_obs[0]) > 0.3:
                self.panic_t = self._panic_draw[1]               # panic: let go of the stick (abrupt brake) then grab it again
            if self.panic_t > 0:
                stick = stick * 0.0
                self.panic_t -= TICK
            jog[3] = jb[0]
            jog[4] = jb[1]
        stick = self._deadzone(np.clip(stick, -1.0, 1.0))
        cmd.twist = stick * BS.V_STICK
        cmd.lift_v = lift * 0.12
        cmd.jog = jog
        self.log.append((obs["t"], self.phase, *stick))
        return cmd

    def _goto(self, ph):
        self.phase = ph
        self.t_phase = 0.0
        self.stopped_t = 0.0

    @property
    def done(self):
        return self.phase == "DONE"


# ------------------------------------------------------------------------------------------------ open-loop stream
@dataclass
class OperatorStream:
    twist: np.ndarray            # (N,3)
    lift_v: np.ndarray           # (N,)
    jog: np.ndarray              # (N,6)
    meta: dict = field(default_factory=dict)

    def save(self, path):
        np.savez_compressed(path, twist=self.twist, lift_v=self.lift_v, jog=self.jog)


class ReplayOperator:
    """open-loop: emits the recorded commands tick by tick (grasp/release events are not part of the base-leg stream)."""

    def __init__(self, stream: OperatorStream):
        self.s, self.i = stream, 0
        self.phase = "REPLAY"

    def step(self, obs) -> Cmd:
        i = min(self.i, len(self.s.twist) - 1)
        self.i += 1
        return Cmd(twist=self.s.twist[i].copy(), lift_v=float(self.s.lift_v[i]), jog=self.s.jog[i].copy())

    @property
    def done(self):
        return self.i >= len(self.s.twist)


def rollout_stream(seed, scene, max_s=40.0, params=None, a_hw=BS.A_HW) -> OperatorStream:
    """Drive a SloppyOperator (task='carry') against a ghost base with hardware slew only; record the command stream."""
    wp = scene.waypoints
    op = SloppyOperator(seed, "carry", scene.out_route, pick_pose=wp["cafe_counter_pick"], params=params)
    s0 = wp["lab_start"]
    b = BS.BaseSim(s0[0], s0[1], s0[2])
    tw, lv, jg = [], [], []
    t = 0.0
    while t < max_s and not op.done:
        obs = dict(t=t, pose=np.array([b.x, b.y, b.yaw]), v=b.v.copy(), lift_h=0.0, grasped=True,
                   grasp_err_p=np.zeros(3), grasp_err_r=np.zeros(3), can_grasp=False)
        c = op.step(obs)
        b.advance(c.twist, TICK, a_hw)
        tw.append(c.twist); lv.append(c.lift_v); jg.append(c.jog)
        t += TICK
    return OperatorStream(np.array(tw), np.array(lv), np.array(jg), dict(seed=seed, duration_s=t, final_pose=[b.x, b.y, b.yaw]))


# ------------------------------------------------------------------------------------------------ keyboard
GLFW = dict(RIGHT=262, LEFT=263, DOWN=264, UP=265, SPACE=32, W=87, A=65, S=83, D=68, Q=81, E=69, R=82, F=70, G=71, H=72,
            I=73, K=75, J=74, L=76, U=85, O=79, T=84, Y=89)


class KeyboardOperator:
    """Live mode (OPTIONAL, needs a display):
         with mujoco.viewer.launch_passive(m, d, key_callback=kb.key_callback) as v:  ...loop: cmd = kb.step(obs)...
       Sticks LATCH (the viewer's key_callback only reports presses): UP/DOWN (or W/S) = forward/back in 0.25 steps, LEFT/RIGHT (A/D) =
       turn, Q/E = sidestep, SPACE = all sticks to 0, R/F = lift up/down, I/K = jog tray x, J/L = jog y, U/O = jog z, T/Y = tray yaw,
       G = grasp, H = release.  No device other than the viewer window is read; it never touches a robot."""

    STEP = 0.25

    def __init__(self):
        self.stick = np.zeros(3)
        self.lift = 0.0
        self.jog = np.zeros(6)
        self._grasp = self._release = False
        self.phase = "KEYBOARD"
        self.done = False

    def key_callback(self, keycode):
        g = GLFW
        s = self.STEP
        if keycode in (g["UP"], g["W"]):
            self.stick[0] = min(1.0, self.stick[0] + s)
        elif keycode in (g["DOWN"], g["S"]):
            self.stick[0] = max(-1.0, self.stick[0] - s)
        elif keycode in (g["LEFT"], g["A"]):
            self.stick[2] = min(1.0, self.stick[2] + s)
        elif keycode in (g["RIGHT"], g["D"]):
            self.stick[2] = max(-1.0, self.stick[2] - s)
        elif keycode == g["Q"]:
            self.stick[1] = min(1.0, self.stick[1] + s)
        elif keycode == g["E"]:
            self.stick[1] = max(-1.0, self.stick[1] - s)
        elif keycode == g["SPACE"]:
            self.stick[:] = 0.0
            self.lift = 0.0
            self.jog[:] = 0.0
        elif keycode == g["R"]:
            self.lift = min(1.0, self.lift + s)
        elif keycode == g["F"]:
            self.lift = max(-1.0, self.lift - s)
        elif keycode in (g["I"], g["K"]):
            self.jog[0] = np.clip(self.jog[0] + (0.04 if keycode == g["I"] else -0.04), -0.15, 0.15)
        elif keycode in (g["J"], g["L"]):
            self.jog[1] = np.clip(self.jog[1] + (0.04 if keycode == g["J"] else -0.04), -0.15, 0.15)
        elif keycode in (g["U"], g["O"]):
            self.jog[2] = np.clip(self.jog[2] + (0.04 if keycode == g["U"] else -0.04), -0.15, 0.15)
        elif keycode in (g["T"], g["Y"]):
            self.jog[5] = np.clip(self.jog[5] + (0.1 if keycode == g["T"] else -0.1), -0.4, 0.4)
        elif keycode == g["G"]:
            self._grasp = True
        elif keycode == g["H"]:
            self._release = True

    inject = key_callback

    def step(self, obs=None) -> Cmd:
        c = Cmd(twist=self.stick * BS.V_STICK, lift_v=self.lift * 0.12, jog=self.jog.copy(), grasp=self._grasp, release=self._release)
        self._grasp = self._release = False
        return c
