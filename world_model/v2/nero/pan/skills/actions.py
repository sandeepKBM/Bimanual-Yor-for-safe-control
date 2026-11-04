"""pan/skills/actions.py -- RoboTwin-style action DSL + lockstep two-arm executor, rebuilt on MuJoCo + mink + FL-MPC.
SIMULATION ONLY.

Ported from RoboTwin 2.0 (MIT, envs/_base_task.py: Action, move(), move_by_displacement, open/close_gripper), with changes:
  - planning = mink IK along a smoothstep pose path (skills/kin.py), each waypoint checked by the exact clearance judge
    (segments._arm_env_clearance); RoboTwin plans with mplib/curobo and checks nothing after;
  - execution = joint waypoints into Episode.paths, tracked by FL-MPC (unchanged); RoboTwin sets PD joint targets;
  - two-arm moves of a held object: MoveObject (both claws carry ONE rigid pan-frame motion, solved jointly), where RoboTwin's
    lift_pot uses two independent z displacements;
  - guarded moves: stop when the claw contact wrench (joint-torque sensing, Episode.claw_wrench) exceeds a threshold;
  - every action returns a record (planned?, min clearance, executed, trigger), and a failed plan stops the list
    (RoboTwin's plan_success flag), with the reason.

    acts = [Move("left", T), Close("left")]                  # one arm
    ex.run([MoveRel("left", [0,0,0.05])], [MoveRel("right", [0,0,0.05])])   # lockstep
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kin as KN

N_WP = 40          # design: joint waypoints per move (v1 _add_path used 60)
V_NOM = 0.05       # design: nominal Cartesian speed [m/s] for duration = max(T_MIN, dist / V_NOM * 1.5) (smoothstep peak 1.5x)
W_NOM = 0.6        # design: nominal angular speed [rad/s]
T_MIN = 0.8        # design: shortest move [s] (= v1 move_to minimum)
GUARD_N = 3.0      # design (ASSUMED): guarded-move contact force threshold at the gripper base [N]
START_TOL = 0.0005  # design: a move that starts below the clearance bound may lose at most this much more [m]


# ---------------- actions ----------------
@dataclass
class Move:
    arm: str
    target: object                       # mink.SE3 of the site {arm}_arm_ee_grasp in the world
    dur: float | None = None
    guard: float | None = None           # stop when |claw contact force| > guard [N]
    label: str = ""


@dataclass
class MoveRel:
    arm: str
    delta: tuple = (0.0, 0.0, 0.0)       # world-frame translation [m] (frame="world") or claw frame (frame="claw")
    rot: object = None                   # optional 3x3 rotation applied about the site point (world frame)
    frame: str = "world"
    dur: float | None = None
    guard: float | None = None
    label: str = ""


@dataclass
class MoveObject:
    """both claws carry one rigid motion of the object frame: T_obj(a) = T_obj0 * exp(a * xi); each claw keeps its offset."""
    delta: tuple = (0.0, 0.0, 0.0)       # world translation of the object frame [m]
    rot: object = None                   # world rotation about the object origin
    origin: tuple | None = None          # object frame origin (default: midpoint of the two claw sites)
    dur: float | None = None
    label: str = ""
    arm: str = "both"
    anchor: str = "ref"                  # "ref": start from the arms' current REFERENCE (v2); "measured": from the MEASURED joints (v3,
                                         # 11-01): the object frame is the claws' real grasp, so the arms stop pushing on each other
    follow: str | None = None            # v4 (11-01): this arm FOLLOWS the other one (Follower) from this move on, until the run ends


@dataclass
class Relax:
    """ease each arm's reference onto its MEASURED joints (smoothstep, dur): removes the reference-vs-measured offset that a closed
    claw holds the arm at (11-01 diagnosis: 24 mm / 14 deg claw-to-claw at the I03 lift start), with no reference step."""
    arm: str = "both"
    dur: float = 0.4
    label: str = ""


@dataclass
class Open:
    arm: str
    width: float | None = None           # per finger [m] past the encoder reading (None = the episode's full open command)
    dur: float = 0.5
    label: str = ""


@dataclass
class Close:
    arm: str
    dur: float = 0.7
    label: str = ""


@dataclass
class Hold:
    arm: str = "both"
    dur: float = 0.3
    label: str = ""


@dataclass
class Result:
    action: str
    arm: str
    ok: bool
    why: str = ""
    clear_mm: float | None = None
    ik_err: tuple | None = None
    dur: float = 0.0
    trigger: str | None = None
    extra: dict = field(default_factory=dict)


# ---------------- planner ----------------
def _slerp_R(R0, R1, a):
    import mujoco
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, R0.ravel())
    mujoco.mju_mat2Quat(q1, R1.ravel())
    if q0 @ q1 < 0:
        q1 = -q1
    th = np.arccos(np.clip(q0 @ q1, -1, 1))
    q = q0 if th < 1e-9 else (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return R.reshape(3, 3)


def _rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


class Planner:
    """plans joint waypoints for one or two arms from the episode's CURRENT reference (continuity with the running path)."""

    def __init__(self, ep, clear=None):
        from .. import segments as S
        self.ep, self.S = ep, S
        self.clear = S.CLAMP["v2"]["clear"] if clear is None else clear

    def _q_ref(self, anchor="ref"):
        """full qpos with each arm at its current REFERENCE (what FL-MPC tracks now), the rest as measured.
        anchor="measured": every joint as measured (the plan then starts where the arms really are)."""
        ep = self.ep
        q = ep.d.qpos.copy()
        if anchor == "measured":
            return q
        t = ep.k * self.S.DT
        for s in self.S.SIDES:
            q[ep.addr[s]] = ep._cmd(s, t)
        return q

    def _pose(self, kin, s):
        T = kin.site_pose(f"{s}_arm_ee_grasp")
        return T.rotation().as_matrix(), T.translation()

    def plan(self, goals, dur=None, anchor="ref"):
        """goals: {arm: fn(a) -> (R, p) for a in [0, 1]} (a = smoothstep phase).  -> (ok, why, {arm: wp list}, dur, clear, err)."""
        import mujoco
        ep, S = self.ep, self.S
        sides = list(goals)
        d = mujoco.MjData(ep.m)
        d.qpos[:] = self._q_ref(anchor)
        mujoco.mj_forward(ep.m, d)
        kin = KN.Kin(ep.m, d, ep.dofs, sides)
        wps = {s: [d.qpos[ep.addr[s]].copy()] for s in sides}
        clear, worst = np.inf, (0.0, 0.0)
        start = min(S._arm_env_clearance(ep.m, d, s) for s in sides)     # a move may start below the bound (e.g. a closed clamp)
        need = min(self.clear, start - START_TOL)                           # ... then it must not get closer than its start
        for i in range(1, N_WP + 1):
            a = i / N_WP
            a = a * a * (3 - 2 * a)
            tg = {f"{s}_arm_ee_grasp": KN.se3(*goals[s](a)) for s in sides}
            q, err = kin.solve(tg)
            worst = (max(worst[0], err[0]), max(worst[1], err[1]))
            dc = mujoco.MjData(ep.m)
            dc.qpos[:] = q
            mujoco.mj_forward(ep.m, dc)
            for s in sides:
                wps[s].append(q[ep.addr[s]].copy())
                clear = min(clear, S._arm_env_clearance(ep.m, dc, s))
        ok = clear >= need - 1e-6 and worst[0] < 2e-3 and worst[1] < 2e-2
        why = "" if ok else ("IK QP infeasible (mink)" if not np.isfinite(worst[0]) else
                             f"clearance {1e3 * clear:.1f} mm < {1e3 * need:.1f} (start {1e3 * start:.1f})" if clear < need - 1e-6
                             else f"IK error {1e3 * worst[0]:.1f} mm / {np.degrees(worst[1]):.1f} deg")
        return ok, why, wps, clear, worst

    @staticmethod
    def duration(dist, ang, dur=None):
        return dur if dur is not None else max(T_MIN, 1.5 * max(dist / V_NOM, ang / W_NOM))


# ---------------- follower (v4 lift, 11-01) ----------------
class Follower:
    """leader-follower two-arm carry, at the REFERENCE level only (FL-MPC, physics and its rows unchanged).
    Every update (the executor's 10-step monitor tick = 50 ms) the follower arm's reference over the next FL-MPC window is re-solved:
        leader target(tau) = measured leader claw pose, moved by the leader's PLANNED motion from now to tau
        follower target(tau) = leader target(tau) * REL0       (REL0 = follower claw in the leader claw frame, measured at the start)
    so the follower tracks where the leader claw really is instead of its own pre-planned path, and the two claws keep REL0 (the
    CLAW_REL assumption) without the arms pushing on each other through the pan.  Sensing = joint encoders + FK (the robot has them).
    IK = the same mink solve + ClearanceLimit as every v2 plan; a failed solve keeps the previous reference (logged)."""

    N_TAU = 4             # design: IK waypoints per window

    def __init__(self, ep, follower, leader):
        import mujoco
        self.ep, self.f, self.l = ep, follower, leader
        m = ep.m
        self.sid = {s: m.site(f"{s}_arm_ee_grasp").id for s in (follower, leader)}
        Rl, pl = self._meas(leader)
        Rf, pf = self._meas(follower)
        self.Rrel, self.prel = Rl.T @ Rf, Rl.T @ (pf - pl)
        self.dk = mujoco.MjData(m)
        self.H = (ep.NP + 1) * ep.FDT + 0.05          # cover the FL-MPC window
        self.n_upd, self.n_fail, self.max_err = 0, 0, (0.0, 0.0)

    def _meas(self, s):
        d = self.ep.d
        return d.site_xmat[self.sid[s]].reshape(3, 3).copy(), d.site_xpos[self.sid[s]].copy()

    def _leader_ref(self, t):
        import mujoco
        ep, d = self.ep, self.dk
        d.qpos[:] = ep.d.qpos
        d.qpos[ep.addr[self.l]] = ep._cmd(self.l, t)
        mujoco.mj_kinematics(ep.m, d)
        return d.site_xmat[self.sid[self.l]].reshape(3, 3).copy(), d.site_xpos[self.sid[self.l]].copy()

    def update(self, ep):
        import mujoco
        from .. import segments as S
        t = ep.k * S.DT
        Rm, pm = self._meas(self.l)
        R0, p0 = self._leader_ref(t)
        d = mujoco.MjData(ep.m)
        d.qpos[:] = ep.d.qpos
        mujoco.mj_forward(ep.m, d)
        kin = KN.Kin(ep.m, d, ep.dofs, [self.f])
        wps, ok = [np.asarray(ep._cmd(self.f, t), float)], True
        for k in range(1, self.N_TAU + 1):
            tau = t + k * self.H / self.N_TAU
            Rt, pt = self._leader_ref(tau)
            Rl, pl = (Rt @ R0.T) @ Rm, pm + (pt - p0)                  # measured leader, moved by its planned motion
            q, err = kin.solve({f"{self.f}_arm_ee_grasp": KN.se3(Rl @ self.Rrel, pl + Rl @ self.prel)})
            if not (np.isfinite(err[0]) and err[0] < 2e-3 and err[1] < 2e-2):
                ok = False
                break
            self.max_err = (max(self.max_err[0], err[0]), max(self.max_err[1], err[1]))
            wps.append(q[ep.addr[self.f]].copy())
        self.n_upd += 1
        if not ok:
            self.n_fail += 1
            return None
        ep.paths[self.f].append((t, t + self.H, wps))
        return None


# ---------------- executor ----------------
class Executor:
    """runs action lists on an Episode; two lists run in lockstep (RoboTwin move(a1, a2)): the i-th actions of both arms start
    together and the step lasts as long as the longer one."""

    def __init__(self, ep, clear=None, monitor=None):
        from .. import segments as S
        self.ep, self.S = ep, S
        self.pl = Planner(ep, clear)
        self.monitor = monitor        # optional extra runtime monitor (ep -> reason | None), e.g. segments.default_monitor
        self.log = []
        self.ok = True
        self.follower = None          # v4: a Follower, active from its MoveObject until the run ends

    # -- helpers
    def _goal_fn(self, act, kin_pose):
        R0, p0 = kin_pose
        if isinstance(act, Move):
            R1, p1 = act.target.rotation().as_matrix(), act.target.translation()
        else:
            dl = np.asarray(act.delta, float)
            if act.frame == "claw":
                dl = R0 @ dl
            Rr = np.eye(3) if act.rot is None else np.asarray(act.rot)
            R1, p1 = Rr @ R0, p0 + dl
        dist, ang = float(np.linalg.norm(p1 - p0)), _rot_angle(R1 @ R0.T)
        return (lambda a: (_slerp_R(R0, R1, a), p0 + a * (p1 - p0))), dist, ang

    def _arm_goals(self, acts):
        """acts: [Move|MoveRel] for distinct arms -> (goal fns, duration)."""
        import mujoco
        ep = self.ep
        d = mujoco.MjData(ep.m)
        d.qpos[:] = self.pl._q_ref()
        mujoco.mj_forward(ep.m, d)
        goals, dist, ang = {}, 0.0, 0.0
        for act in acts:
            sid = ep.m.site(f"{act.arm}_arm_ee_grasp").id
            pose = (d.site_xmat[sid].reshape(3, 3).copy(), d.site_xpos[sid].copy())
            g, dd, aa = self._goal_fn(act, pose)
            goals[act.arm], dist, ang = g, max(dist, dd), max(ang, aa)
        dur = max([a.dur for a in acts if a.dur is not None], default=None)
        return goals, Planner.duration(dist, ang, dur)

    def _object_goals(self, act):
        import mujoco
        ep = self.ep
        d = mujoco.MjData(ep.m)
        d.qpos[:] = self.pl._q_ref(act.anchor)
        mujoco.mj_forward(ep.m, d)
        poses = {}
        for s in self.S.SIDES:
            sid = ep.m.site(f"{s}_arm_ee_grasp").id
            poses[s] = (d.site_xmat[sid].reshape(3, 3).copy(), d.site_xpos[sid].copy())
        o = np.asarray(act.origin, float) if act.origin is not None else 0.5 * (poses["left"][1] + poses["right"][1])
        Rr = np.eye(3) if act.rot is None else np.asarray(act.rot)
        dl = np.asarray(act.delta, float)

        def mk(s):
            R0, p0 = poses[s]
            r = p0 - o
            return lambda a: (_slerp_R(np.eye(3), Rr, a) @ R0, o + a * dl + _slerp_R(np.eye(3), Rr, a) @ r)

        dist = float(np.linalg.norm(dl)) + _rot_angle(Rr) * max(np.linalg.norm(poses[s][1] - o) for s in poses)
        return {s: mk(s) for s in self.S.SIDES}, Planner.duration(dist, _rot_angle(Rr), act.dur)

    def _guard_monitor(self, guards):
        def mon(ep):
            if self.follower is not None:
                self.follower.update(ep)
            for s, thr in guards.items():
                f = float(np.linalg.norm(ep.claw_wrench(s)))
                if f > thr:
                    return f"guard {s} {f:.1f} N > {thr:.1f}"
            if self.monitor is not None:
                return self.monitor(ep)
            return None
        return mon

    def _freeze(self, sides):
        """after a guard trip: hold each arm at its MEASURED configuration from now on."""
        ep = self.ep
        t = ep.k * self.S.DT
        for s in sides:
            q = ep.d.qpos[ep.addr[s]].copy()
            ep.paths[s].append((t, t + 1e-3, [q, q]))

    def _grip(self, act):
        ep = self.ep
        s = act.arm
        if isinstance(act, Close):
            ep.gmode[s], ep.gcmd[s] = "force", ep.plan["grip_force"]
        else:
            ep.gmode[s] = "pos"
            ep.gcmd[s] = ep.open_cmd[s] if act.width is None else min(ep.open_cmd[s], ep.finger_q(s) + float(act.width))
        ep._apply_grip(s)

    # -- one lockstep step
    def step(self, acts):
        """acts: actions starting together (at most one per arm, or one MoveObject).  -> list of Result."""
        ep = self.ep
        moves = [a for a in acts if isinstance(a, (Move, MoveRel))]
        objs = [a for a in acts if isinstance(a, MoveObject)]
        grips = [a for a in acts if isinstance(a, (Open, Close))]
        holds = [a for a in acts if isinstance(a, Hold)]
        relax = [a for a in acts if isinstance(a, Relax)]
        res, dur, guards = [], max([a.dur for a in grips + holds + relax], default=0.0), {}
        for a in relax:
            t = ep.k * self.S.DT
            sides = self.S.SIDES if a.arm == "both" else (a.arm,)
            off = {}
            for s in sides:
                q0, q1 = np.asarray(ep._cmd(s, t), float), ep.d.qpos[ep.addr[s]].copy()
                wp = [q0 + (b * b * (3 - 2 * b)) * (q1 - q0) for b in np.linspace(0, 1, N_WP + 1)]
                ep.paths[s].append((t, t + a.dur, wp))
                off[s] = round(float(np.degrees(np.abs(q1 - q0).max())), 2)
            res.append(Result("Relax", a.arm, True, dur=a.dur, extra=dict(label=a.label, max_offset_deg=off)))
        if objs or moves:
            goals, mdur = self._object_goals(objs[0]) if objs else self._arm_goals(moves)
            ok, why, wps, clear, err = self.pl.plan(goals, anchor=objs[0].anchor if objs else "ref")
            name = type(objs[0] if objs else moves[0]).__name__
            for a in (objs or moves):
                res.append(Result(name, a.arm, ok, why, round(1e3 * clear, 2), (round(1e3 * err[0], 3), round(float(np.degrees(err[1])), 3)), mdur,
                                  extra=dict(label=a.label)))
                if getattr(a, "guard", None):
                    guards[a.arm] = a.guard
            if not ok:
                self.ok = False
                self.log += res
                return res
            t = ep.k * self.S.DT
            for s, wp in wps.items():
                ep.paths[s].append((t, t + mdur, wp))
            dur = max(dur, mdur)
            if objs and objs[0].follow:
                f = objs[0].follow
                self.follower = Follower(ep, f, "left" if f == "right" else "right")
        for g in grips:
            self._grip(g)
            res.append(Result(type(g).__name__, g.arm, True, dur=g.dur, extra=dict(label=g.label)))
        for h in holds:
            res.append(Result("Hold", h.arm, True, dur=h.dur, extra=dict(label=h.label)))
        mon = self._guard_monitor(guards) if (guards or self.monitor or self.follower) else None
        status, info = ep.run_phases([("hold", dur)], monitor=mon)
        if status == "abort":
            why = info.get("why", "")
            for r in res:
                r.trigger = why
            if why.startswith("guard"):
                self._freeze(list(guards))
            else:
                self.ok = False
        self.log += res
        return res

    def run(self, left=(), right=(), both=()):
        """lockstep lists (RoboTwin move(a1, a2)); `both` = a list of two-arm steps (MoveObject / Hold) run after.
        Stops at the first failed plan or non-guard abort.  -> bool."""
        left, right = list(left), list(right)
        n = max(len(left), len(right))
        for i in range(n):
            step = [x for x in (left[i] if i < len(left) else None, right[i] if i < len(right) else None) if x is not None]
            self.step(step)
            if not self.ok:
                return False
        for a in both:
            self.step([a])
            if not self.ok:
                return False
        return True
