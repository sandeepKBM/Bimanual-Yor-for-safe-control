"""flagship/mission.py -- the coffee-run mission state machine (operator side).  SIMULATION ONLY.

CoffeeRunOperator extends teleop.operator.SloppyOperator (same stick noise / reaction delay / corner cutting; deterministic per seed) with the
phases the flagship needs.  The operator is CLOSED-LOOP: raw and shielded sides use the same seed (same noise realisation, delays, bump times) but
each reacts to its own robot state.  Drive legs are the operator's; manipulation sub-steps are scripted operator macros (as in operator.py).

  OUT        sloppy drive lab -> corridor -> cafeteria to the PRE-DOCK pose (0.7 m short of the counter stand-off = where the back-out ends)
  SENSE_C    stand still SENSE_HOLD_S; services.sense_counter(obs) -> ZED capture + zonotopes (shield side only; raw side just pauses)
  RAISE      lift up until the grippers are at the tray handles on the serving ledge (counter height, 0.90 m)
  CREEP      closed-loop creep onto the counter stand-off
  ALIGN      tray-frame jog until within grasp tolerance -> GRASP (weld activation)
  LIFT       lift +0.06 m (tray clears the ledge)
  BACKOUT    reverse 0.7 m straight out
  LOWER_C    lift down to the carry height (0)
  BACK       sloppy drive counter -> cafeteria -> corridor -> lab to the desk stand-off (cup aboard)
  CREEP_D    closed-loop creep onto the desk stand-off
  SENSE_D    stand still; services.sense_desk(obs) -> ZED capture, zonotopes, free spot + path (shield side) | naive centre-line target (raw side)
  PLACE      autonomous path follow of the tray to the spot (holonomic base, yaw fixed), arms hold; no teleop sticks
  ABORT      raw side only: the operator hits something while placing -> halts 1 s, retraces the path back to where the placement started,
             keeps the tray in the hands (no release) -> mission ends FAILED
  RELEASE    tray lands on the placement stand (stand moved under the spot), weld off, settle
  DONE
Phases in AUTONOMOUS are executed without the base shield (the zonotope plan is the safety mechanism there; see README).
"""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.house.teleop import base as BS
from world_model.v2.house.teleop.operator import SloppyOperator, OperatorParams, _sat
from world_model.v2.house.teleop.sim import Cmd, TICK

from . import scene as FS

PHASES = ["OUT", "SENSE_C", "RAISE", "CREEP", "ALIGN", "LIFT", "BACKOUT", "LOWER_C", "BACK", "CREEP_D", "SENSE_D", "PLACE", "ABORT", "RELEASE", "DONE"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
AUTONOMOUS = ("PLACE", "ABORT", "RELEASE")
SENSE_HOLD_S = 1.5
STALL_S = 8.0
PREDOCK_M = 0.7
NAIVE_T = np.array([0.0, 0.405])            # operator's naive target: tray centre-line of the desk, 0.2 m in from the front edge (zed_demo.run t_naive)


class NullServices:
    """raw side: no perception / planning."""
    name = "none"

    def sense_counter(self, obs):
        return None

    def sense_desk(self, obs):
        return None


class PathFollower:
    """tracks a world path (x,y arrays at DT=5 ms, yaw const) with a feed-forward + P law; returns body twist commands at the teleop rate."""

    def __init__(self, traj, kp=4.0, kyaw=3.0):
        self.x, self.y, self.yaw, self.t = np.asarray(traj.x), np.asarray(traj.y), np.asarray(traj.yaw), np.asarray(traj.t)
        self.kp, self.kyaw, self.k = kp, kyaw, 0
        self.dt = float(self.t[1] - self.t[0])

    def step(self, pose):
        n = len(self.t)
        i = min(self.k, n - 1)
        j = min(i + 1, n - 1)
        vff = np.array([self.x[j] - self.x[i], self.y[j] - self.y[i]]) / max(self.dt, 1e-9)
        e = np.array([self.x[i] - pose[0], self.y[i] - pose[1]])
        vw = vff + self.kp * e
        c, s = math.cos(pose[2]), math.sin(pose[2])
        vb = np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]])
        wz = self.kyaw * BS.wrap(self.yaw[i] - pose[2])
        self.k += int(round(TICK / self.dt))
        return np.array([vb[0], vb[1], wz]), (self.k >= n), float(np.hypot(*e))


class CoffeeRunOperator(SloppyOperator):
    def __init__(self, seed, scene, services=None, params: OperatorParams | None = None, lift_pick=FS.H_PICK, raw_halt_on_contact=True, start_phase="OUT"):
        wp = scene.waypoints
        self.dock = np.array(wp["cafe_counter_pick"], float)
        self.desk = np.array(wp["lab_desk_place"], float)
        d = self.dock[:2]
        pre = d - PREDOCK_M * np.array([math.cos(self.dock[2]), math.sin(self.dock[2])])
        route = [np.array(p, float) for p in scene.out_route]
        route = [p for p in route if np.linalg.norm(p - d) > 0.9] + [pre]
        super().__init__(seed, "coffee", [list(p) for p in route], back_route=scene.back_route, pick_pose=tuple(self.dock), place_pose=tuple(self.desk),
                         params=params, back_mode="backout")
        self.pre = pre
        self.services = services or NullServices()
        self.h_pick = lift_pick
        self.h_carry_up = lift_pick + FS.LIFT_CARRY_EXTRA
        self.phase = start_phase
        self.events = [(0.0, start_phase)]
        if start_phase in ("BACK", "CREEP_D", "SENSE_D"):
            self._backed = True
        self.cnt = {}
        self.plan_c = None
        self.plan_d = None
        self.follower = None
        self.stand = None
        self.spot_xy = None
        self.halt_on_contact = raw_halt_on_contact
        self.contact_t = 0.0
        self.trace = []
        self.stall_t = 0.0
        self.recover_t = 0.0
        self._retrace = None
        self.t_now = 0.0
        self.place_info = {}
        self._pending_capture = None

    # ------------------------------------------------------------------------------------------------ bookkeeping
    def _goto(self, ph):
        super()._goto(ph)
        self.events.append((self.t_now, ph))

    @property
    def phase_id(self):
        return PHASE_ID[self.phase]

    def _stand_still(self, obs, hold_s):
        """True once the robot has been (nearly) stationary for hold_s."""
        if np.linalg.norm(obs["v"]) < 0.03:
            self.stopped_t += TICK
        else:
            self.stopped_t = 0.0
        return self.stopped_t >= hold_s

    # ------------------------------------------------------------------------------------------------ step
    def step(self, obs) -> Cmd:
        p = self.p
        self.t_now = obs["t"]
        d_obs = self._delayed(obs)
        noise, jb = self._noise()
        cmd = Cmd()
        stick = np.zeros(3)
        lift = 0.0
        jog = np.zeros(6)
        self.t_phase += TICK
        v_obs = obs["v"]
        moving = np.linalg.norm(v_obs) > 0.03
        ph = self.phase
        if ph == "OUT":
            stick, arrived = self._drive(self.out_route, False, d_obs)
            pre_pose = (self.pre[0], self.pre[1], self.dock[2])
            if self.wp_i >= len(self.out_route) - 1 and np.linalg.norm(self.pre - d_obs["pose"][:2]) < 0.45:
                stick, e_w0, _ = self._creep_stick(pre_pose, d_obs, 0.5)        # last 0.45 m: closed-loop creep (backs up if overshot)
                arrived = bool(np.linalg.norm(e_w0) < 0.12)
            if arrived:
                sw, ok = self._heading_to(self.dock[2], d_obs)
                stick = np.array([0.0, 0.0, sw])
                if ok and not moving:
                    self.stopped_t += TICK
                else:
                    self.stopped_t = 0.0
                if self.stopped_t > 0.6:
                    self._goto("SENSE_C")
        elif ph == "SENSE_C":
            if self.t_phase <= TICK * 1.5:
                self.plan_c = self.services.sense_counter(obs)
                self._pending_capture = "counter"
            if self.t_phase >= SENSE_HOLD_S:
                self._goto("RAISE")
        elif ph == "RAISE":
            ez = float(obs["grasp_err_p"][2])
            lift = float(np.clip(6.0 * ez / 0.12, -0.5, 0.8))
            if abs(ez) < 0.004 and self.t_phase > 0.5 and abs(obs["v"]).sum() < 0.03:
                self._goto("CREEP")
        elif ph == "CREEP":
            stick, e_w, e_yaw = self._creep_stick(self.dock, d_obs)
            if np.linalg.norm(e_w) < 0.02 and abs(e_yaw) < 0.02 and not moving:
                self.stopped_t += TICK
            else:
                self.stopped_t = 0.0
            if self.stopped_t > 0.5:
                self._goto("ALIGN")
        elif ph == "ALIGN":
            ep, er = obs["grasp_err_p"], obs["grasp_err_r"]
            jog[:3] = np.clip(p.jog_gain[0] * np.asarray(ep), -p.jog_max[0], p.jog_max[0])
            jog[3:] = np.clip(p.jog_gain[1] * np.asarray(er), -p.jog_max[1], p.jog_max[1])
            if obs["can_grasp"] and self.t_phase > 0.5:
                cmd.grasp = True
                self._goto("LIFT")
        elif ph == "LIFT":
            lift = p.lift_stick * (1.0 if obs["lift_h"] < self.h_carry_up else 0.0)
            if obs["lift_h"] >= self.h_carry_up - 0.004 and self.t_phase > 0.4:
                self._goto("BACKOUT")
                self.back_origin = np.array(obs["pose"][:2])
        elif ph == "BACKOUT":
            stick = np.array([-0.5, 0.0, 0.0])
            if np.linalg.norm(obs["pose"][:2] - self.back_origin) >= self.backout_m:
                self._backed = True
                self._goto("LOWER_C")
        elif ph == "LOWER_C":
            lift = -p.lift_stick * (1.0 if obs["lift_h"] > 0.004 else 0.0)
            if obs["lift_h"] <= 0.006 and self.t_phase > 0.4 and not moving:
                self._goto("BACK")
                self.wp_i = 1
        elif ph == "BACK":
            stick, arrived = self._drive(self.back_route, False, d_obs)
            if self.wp_i >= len(self.back_route) - 1 and np.linalg.norm(self.desk[:2] - d_obs["pose"][:2]) < 0.45:
                stick, e_w0, _ = self._creep_stick(tuple(self.desk), d_obs, 0.5)    # last 0.45 m: closed-loop creep (backs up if overshot)
                arrived = bool(np.linalg.norm(e_w0) < 0.12)
            if arrived:
                sw, ok = self._heading_to(self.desk[2], d_obs)
                stick = np.array([0.0, 0.0, sw])
                self.stopped_t = self.stopped_t + TICK if (ok and not moving) else 0.0
                if self.stopped_t > 0.6:
                    self._goto("CREEP_D")
        elif ph == "CREEP_D":
            stick, e_w, e_yaw = self._creep_stick(self.desk, d_obs)
            if np.linalg.norm(e_w) < 0.02 and abs(e_yaw) < 0.02 and not moving:
                self.stopped_t += TICK
            else:
                self.stopped_t = 0.0
            if self.stopped_t > 0.5:
                self._goto("SENSE_D")
        elif ph == "SENSE_D":
            if self.t_phase <= TICK * 1.5:
                self.plan_d = self.services.sense_desk(obs)
                self._pending_capture = "desk"
            if self.t_phase >= SENSE_HOLD_S:
                self._start_place(obs)
        elif ph == "PLACE":
            tw, done, err = self.follower.step(obs["pose"])
            cmd.twist = tw
            self.trace.append(np.array(obs["pose"][:2]))
            # raw operator stops pushing when the robot touches something (he notices the crash)
            if obs.get("bump_new", 0) > 0:
                self.contact_t += TICK
            if self.halt_on_contact and self.plan_d is None and self.contact_t > 0.0:
                cmd.twist = np.zeros(3)
                self.place_info["aborted"] = True
                self.place_info["abort_t"] = self.t_now
                self._goto("ABORT")
            elif done and err < 0.01:
                self._goto("RELEASE")
                self._release_wait = 0.0
            elif self.t_phase > self.place_timeout:
                self.place_info["timeout"] = True
                self._goto("RELEASE")
                self._release_wait = 0.0
            self.log.append((obs["t"], self.phase, *cmd.twist))
            return cmd
        elif ph == "ABORT":
            from world_model.v2.house.zed_demo import execute as EX
            if self.t_phase <= TICK * 1.5:
                self.stopped_t = 0.0
                self._retrace = None
            if self._retrace is None:
                if self.t_phase >= 1.0:
                    tr = np.array(self.trace)[::-1]
                    tr = tr[::max(len(tr) // 40, 1)]
                    tr = np.vstack([np.array(obs["pose"][:2]), tr])
                    self._retrace = PathFollower(EX.make_traj(tr, np.zeros(2), float(obs["pose"][2]), vmax=0.15))
                self.log.append((obs["t"], self.phase, 0, 0, 0))
                return cmd
            tw, done, err = self._retrace.step(obs["pose"])
            cmd.twist = tw
            if done:
                self._goto("DONE")
            self.log.append((obs["t"], self.phase, *cmd.twist))
            return cmd
        elif ph == "RELEASE":
            self._release_wait += TICK
            if self._release_wait < 0.6 and self._release_wait + TICK >= 0.6:
                cmd.release = True
            if self._release_wait >= 2.0:
                self._goto("DONE")
            self.log.append((obs["t"], self.phase, 0, 0, 0))
            return cmd
        # stall recovery (a real operator would back off and re-aim): stuck > STALL_S while pushing -> reverse 2 s, re-target the nearest route point ahead
        if ph in ("OUT", "BACK"):
            route = self.out_route if ph == "OUT" else self.back_route
            if self.recover_t > 0:
                self.recover_t -= TICK
                stick = np.array([-0.5, 0.0, 0.0])
                if self.recover_t <= 0:
                    P = np.array(route)
                    self.wp_i = int(min(np.argmin(np.linalg.norm(P - obs["pose"][:2], axis=1)) + 2, len(route) - 1))
                    self.stall_t = 0.0
            else:
                if np.linalg.norm(v_obs) < 0.04 and np.abs(stick[:2]).sum() > 0.15:
                    self.stall_t += TICK
                else:
                    self.stall_t = 0.0
                if self.stall_t > STALL_S:
                    self.recover_t = 2.0
                    self.place_info.setdefault("stall_recoveries", []).append([float(self.t_now), ph])
        # human imperfections only on the drive legs
        if ph in ("OUT", "BACK"):
            if np.abs(stick).sum() > 0.05:
                stick = stick + noise
            if self.panic_t <= 0 and self._panic_draw[0] < p.panic_rate * TICK and np.abs(v_obs[0]) > 0.3:
                self.panic_t = self._panic_draw[1]
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

    # ------------------------------------------------------------------------------------------------ placement
    place_timeout = 40.0

    def _start_place(self, obs):
        from world_model.v2.house.zed_demo import execute as EX
        base0 = np.array(obs["pose"][:2], float)
        yaw0 = float(obs["pose"][2])
        if self.plan_d is not None:
            path_t = np.asarray(self.plan_d["path_t"], float)
            spot_t = np.asarray(self.plan_d["spot_t"], float)
            self.place_info["planned"] = True
        else:
            path_t = np.stack([np.linspace(0, NAIVE_T[0], 60), np.linspace(0, NAIVE_T[1], 60)], 1)
            spot_t = NAIVE_T.copy()
            self.place_info["planned"] = False
        traj = EX.make_traj(path_t, base0, yaw0)
        self.follower = PathFollower(traj)
        tray0 = np.array(obs["tray_pos"], float)
        self.spot_xy = tray0[:2] + spot_t
        self.stand = dict(xy=self.spot_xy.copy(), yaw=yaw0 + math.pi / 2, z_under=float(tray0[2]) - 0.006)
        self.place_info.update(spot_t=spot_t.tolist(), base0=base0.tolist(), duration_s=float(traj.t[-1]), tray_xy=self.spot_xy.tolist())
        self._goto("PLACE")
        self.contact_t = 0.0
