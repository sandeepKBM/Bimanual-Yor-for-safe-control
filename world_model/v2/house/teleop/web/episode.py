"""teleop/web/episode.py -- episode lifecycle for the browser teleop (PROTOCOL_BETA.md section 1).  Pure logic, no MuJoCo.  SIMULATION ONLY.

    ep = Episode(task="carry", goal_xy=(x, y))
    ep.update(t_sim, wall_t, enabled, tel, events)    # once per sim tick; returns the state string
    ep.to_json()                                      # telemetry key `episode`
    ep.reset()                                        # back to IDLE (server rebuild / Backspace)

States IDLE -> ACTIVE -> OUTCOME -> (reset) -> IDLE.  `tel` is the dict of per-tick values the telemetry already uses; keys read (all optional):
cup_lost, beads_out, speed, pose, interv_now, shield_label, wall_clr, person_clr, cup_tilt, latched, assist{place}, pick{status}.
`events` is a list of server-event names; "episode_end" (key K) ends an ACTIVE episode.
"""
from __future__ import annotations

import math

IDLE, ACTIVE, OUTCOME = "IDLE", "ACTIVE", "OUTCOME"
GOAL_R = 0.30        # ASSUMED, user-approved 2026-11-12: carry/coffee goal radius [m] around the scene waypoint
STOP_V = 0.05        # ASSUMED, user-approved 2026-11-12: base speed below which the base counts as stopped [m/s]
INTERV_THRESH = 0.02  # same threshold as the shield-intervention fraction in server telemetry / teleop/run.py
FALLBACK_LABELS = ("brake_fallback", "escape_fallback")


def _fin(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


class Episode:
    def __init__(self, task, goal_xy=None, goal_r=GOAL_R, stop_v=STOP_V, timeout_s=0.0, grasp_mode=None):
        self.task, self.goal_r, self.stop_v = task, float(goal_r), float(stop_v)
        self.goal = None if goal_xy is None else (float(goal_xy[0]), float(goal_xy[1]))
        self.timeout_s = float(timeout_s or 0.0)
        self.grasp_mode = grasp_mode          # coffee success needs "friction" (assist reports place)
        self.reset()

    def reset(self):
        self.state = IDLE
        self.outcome = None
        self.want_disarm = False
        self._prev_enabled = False
        self._prev_latched = False
        self._t0 = self._w0 = self._t_last = self._w_last = None
        self._acc = None

    # ------------------------------------------------------------------ helpers
    @property
    def blocks_motion(self):
        """OUTCOME: the server keeps every motion command zero until a reset."""
        return self.state == OUTCOME

    def consume_disarm(self):
        """True once after an outcome that asks the server to disarm the hub."""
        w, self.want_disarm = self.want_disarm, False
        return w

    def dist_to_goal(self, tel):
        pose = tel.get("pose")
        if self.goal is None or not pose:
            return None
        return _fin(math.hypot(float(pose[0]) - self.goal[0], float(pose[1]) - self.goal[1]))

    def _new_acc(self):
        return dict(n=0, n_interv=0, min_wall=None, min_person=None, max_tilt=0.0, n_estop=0, n_fb=0, beads_out=0)

    def _accumulate(self, tel):
        a = self._acc
        a["n"] += 1
        if (_fin(tel.get("interv_now")) or 0.0) > INTERV_THRESH:
            a["n_interv"] += 1
        for key, k in (("wall_clr", "min_wall"), ("person_clr", "min_person")):
            v = _fin(tel.get(key))
            if v is not None and (a[k] is None or v < a[k]):
                a[k] = v
        a["max_tilt"] = max(a["max_tilt"], _fin(tel.get("cup_tilt")) or 0.0)
        if str(tel.get("shield_label", "")) in FALLBACK_LABELS:
            a["n_fb"] += 1
        a["beads_out"] = max(a["beads_out"], int(tel.get("beads_out") or 0))

    def _metrics(self, t_sim, wall_t, tel):
        a = self._acc
        return dict(duration_s=round(t_sim - self._t0, 3), wall_s=round(wall_t - self._w0, 3),
                    interv_frac=round(a["n_interv"] / max(a["n"], 1), 3), min_wall_clr=a["min_wall"], min_person_clr=a["min_person"],
                    beads_out=a["beads_out"], n_estop=a["n_estop"], n_shield_fallback=a["n_fb"], max_cup_tilt=round(a["max_tilt"], 2),
                    dist_to_goal=None if self.goal is None else self.dist_to_goal(tel))

    def _finish(self, result, reason, t_sim, wall_t, tel, disarm=True):
        self.outcome = dict(result=result, reason=reason, metrics=self._metrics(t_sim, wall_t, tel))
        self.state = OUTCOME
        self.want_disarm = disarm

    # ------------------------------------------------------------------ predicates
    def _failure(self, tel):
        if bool(tel.get("cup_lost")) or str(tel.get("cup_state", "")) == "LOST":
            return "cup lost"
        if int(tel.get("beads_out") or 0) > 0:
            return "spill"
        return None

    def _success(self, tel):
        if self.task == "carry":
            d, v = self.dist_to_goal(tel), _fin(tel.get("speed"))
            ok = str(tel.get("cup_state", "ok")) == "ok" and not tel.get("cup_lost")
            if d is not None and v is not None and d <= self.goal_r and v < self.stop_v and ok:
                return "reached the goal and stopped"
        elif self.task == "coffee":
            if self.grasp_mode == "friction" and (tel.get("assist") or {}).get("place") is True:
                return "placed"
        elif self.task == "pick":
            if str((tel.get("pick") or {}).get("status", "")) in ("holding", "done"):
                return "object held"
        return None

    # ------------------------------------------------------------------ the per-tick update
    def update(self, t_sim, wall_t, enabled, tel, events=()):
        enabled = bool(enabled)
        latched = bool(tel.get("latched"))
        if self.state == IDLE:
            if enabled and not self._prev_enabled:
                self.state, self._t0, self._w0, self._acc = ACTIVE, float(t_sim), float(wall_t), self._new_acc()
        if self.state == ACTIVE:
            self._accumulate(tel)
            if latched and not self._prev_latched:
                self._acc["n_estop"] += 1
            fail = self._failure(tel)
            ok = self._success(tel)
            if fail is not None:
                self._finish("fail", fail, t_sim, wall_t, tel)
            elif ok is not None:
                self._finish("success", ok, t_sim, wall_t, tel)
            elif "episode_end" in events:
                self._finish("ended", "operator", t_sim, wall_t, tel)
            elif self.timeout_s > 0 and t_sim - self._t0 >= self.timeout_s:
                self._finish("fail", "timeout", t_sim, wall_t, tel)
        self._t_last, self._w_last = float(t_sim), float(wall_t)
        self._prev_enabled, self._prev_latched = enabled, latched
        return self.state

    def to_json(self):
        t_active = None
        if self.state == ACTIVE and self._t_last is not None:
            t_active = round(self._t_last - self._t0, 2)
        elif self.state == OUTCOME:
            t_active = self.outcome["metrics"]["duration_s"]
        return dict(state=self.state, t_active=t_active, goal=None if self.goal is None else [round(self.goal[0], 3), round(self.goal[1], 3)],
                    outcome=self.outcome)
