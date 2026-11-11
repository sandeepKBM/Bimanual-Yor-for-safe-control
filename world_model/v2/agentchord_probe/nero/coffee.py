"""agentchord_probe/nero/coffee.py -- AgentChord's pipeline on the coffee-tray task "tray0" in the NERO sim.  SIMULATION ONLY (the only network call is the owner-approved
Codex plan generation through the loopback gateway; execution is offline).  New module: nothing of atoms.py / execute.py / tasks.py is edited.

  install()          register the task "tray0" (prompt text, representative world, image) in agentchord_probe.nero.tasks and the coffee scene (nero.contents.coffee).
  make_plan(...)     plan.make_plan("tray0", ...) with the model spec of the comparison ("codex:gpt-5.6-sol:high": ONE spec for both methods).
  run_world(args)    execute the cached compiled graph on the world of index i with THEIR AgentTaskGraph.run + monitors, our coffee atoms (below), optional drop injection.

Atoms (their vocabulary; grasp = rigid weld, ASSUMED for every grasp, user 11-01):
  grasp(robot_name, obj_name)  'capsule' (left arm) | 'tray' (either arm, at its handle): leave the handle if the claw sits on it, go to a pre-grasp pose 5 cm outward, approach, weld.
  move_by_relative_offset / move_to_absolute_position / move_relative_to_object   TCP translation of each arm named in the edge; if both claws hold the tray the two deltas must agree
        within ARM_TOL (rigid object) and the common delta is executed, else rejected_arm_mismatch (as atoms.py).  A claw on a handle but not welded leaves it first.
  place_on_table(robot_name, obj_name, x, y, pre_place_dis)  capsule | tray: object to (x, y) pre_place_dis above its rest height (capsule: in the tray footprint -> on the tray base, else
        on the counter), down, release (weld off), retreat 5 cm outward.  tray: both arms in the edge, x/y must agree within 1 cm.
  open_gripper -> weld of that arm's held object off.   close_gripper -> no-op.   back_to_initial_pose -> that arm to its t = 0 hold pose (joint path to q0).
  rotate_eef / orient_eef / unknown -> "unmappable_atom".
Gripper reading (their monitor_object_held reads the mean |finger qpos|): ASSUMED model, the real sim fingers do not move under the weld: held 0.03 | open 0.04 | "closed on
nothing" 0.0 after an injected drop (best case for their hold_lost monitor).  An atom with arms of different functions in one edge is allowed (independent arms); nothing is checked
for arm-arm clearance beyond the sim's U (collision) rows.
Drops (AgentChord Sec. IV-B): with probability p an object held by an end-effector (weld on) is dropped at each atomic action (= each edge), at a uniformly random step of the edge.
"""
from __future__ import annotations

import json
import pathlib
import time
import types

import numpy as np

from . import atoms as A
from . import tasks as T
from .env import NeroEnv

TASK = "tray0"
SIM = types.SimpleNamespace(CF=None, CR=None, E=None, P=None, SW=None)


def bind(CF, CR, E, P, SW):
    """hand over the sim modules (nero/contents coffee, carry, envelope, probe, same_worlds): dependency rule, agentchord_probe/nero does not import nero/contents internals;
    the driver (nero/coffee_run.py) imports them and binds them once per process, before install() / run_world()."""
    SIM.CF, SIM.CR, SIM.E, SIM.P, SIM.SW = CF, CR, E, P, SW


SIDES = ("left", "right")
OUT = {"left": np.array([0.0, 1.0, 0.0]), "right": np.array([0.0, -1.0, 0.0])}       # approach retreats outward (away from the tray)
RETREAT = 0.05
GRIP_HELD, GRIP_OPEN, GRIP_EMPTY = 0.03, 0.04, 0.0      # ASSUMED, see module doc
AT_TAB_TOL = 0.02
ARM_TOL = 0.03

TEXT = (
    "Use the robot's two arms to set up a coffee tray on the counter (object name 'counter', top surface at z = {counter_top:.2f} m). The objects are 'capsule' "
    "(a coffee capsule, now at x = {cx:.2f} m, y = {cy:.2f} m on a small pad) and 'tray' (the coffee tray, reference point now at x = {x0:.2f} m, y = {y0:.2f} m). "
    "First, grasp the capsule with the left arm, lift it by 0.08 m, and place it inside the tray at an offset of [0.0, 0.06, 0.10] relative to the tray's position, then release it. "
    "Then grasp the tray with both arms simultaneously, the left arm at the tray's left handle and the right arm at the tray's right handle. "
    "Lift the tray by 0.05 m, move it 0.10 m toward the robot (x decreases by {dx:.2f}, y unchanged) and put it back down on the counter at x = {x1:.2f} m, y = {y1:.2f} m, "
    "keeping the tray level so the capsule stays inside. Finally release the tray and return both arms to their initial poses. "
    "Positions are world coordinates in metres (x forward, y left, z up).\n"
    "Known facts: {facts}")
FACTS = {"known": "the capsule weighs 12 g and the tray weighs 0.30 kg; there is nothing else in the tray.",
         "unknown": "the capsule weighs 12 g and the tray weighs 0.30 kg; there may be an additional load of unknown mass lying inside the tray."}


def install():
    """idempotent: coffee scene + tray0 entries in nero.tasks (rep_world, task_prompt, render_rgb)."""
    CF = SIM.CF
    CF.install()
    if getattr(T, "_coffee_installed", False):
        return
    T._coffee_installed = True
    rw0, tp0 = T.rep_world, T.task_prompt
    T.rep_world = lambda task: CF.CoffeeWorld() if task == TASK else rw0(task)       # representative world: the visible scene (the hidden load is invisible and has no geometry)

    def task_prompt(task, world=None):
        if task != TASK:
            return tp0(task, world)
        w = world or CF.CoffeeWorld()
        facts = FACTS[CF.variant()]
        sx, sy = float(w.obj_xy[0]), float(w.obj_xy[1])
        return TEXT.format(counter_top=w.counter_top, cx=w.capsule_xy[0], cy=w.capsule_xy[1], x0=sx, y0=sy, dx=-CF.FORWARD[0], x1=sx + CF.FORWARD[0], y1=sy + CF.FORWARD[1],
                           facts=facts), facts
    T.task_prompt = task_prompt


def make_plan(out, spec="codex:gpt-5.6-sol:high", regenerate=False):
    """one AgentChord plan (Task -> Recovery -> Compile agents) through the loopback gateway; backend 'codex' with the model/reasoning of `spec`."""
    install()
    from . import plan as PL
    kind, model, *r = spec.split(":")
    assert kind == "codex", "the comparison uses ONE codex spec"
    return PL.make_plan(TASK, out, backend="codex", model=model, reasoning=(r[0] if r else None), regenerate=regenerate)


# ------------------------------------------------------------------------------------------------------------------ env
class CoffeeEnv(NeroEnv):
    def __init__(self, ep, inj=None, mon_period=0.1, edge_settle=0.3):
        super().__init__(ep, mon_period=mon_period, edge_settle=edge_settle)
        self.names = ["tray", "capsule", "counter"]
        self.inj = inj
        self.grip = {s: "open" for s in SIDES}
        self.T0 = {s: self.tcp_cmd(s) for s in SIDES}
        self.tray0 = ep.d.xpos[ep.obj].copy()
        self.tab_off = {s: self.T0[s] - self.tray0 for s in SIDES}        # TCP at the handle relative to the tray reference (tray level at the start)
        self.cap_off = np.zeros(3)
        self.update_obj_info()

    def pose(self, name):
        if name == "capsule":
            ep = self.ep
            return _pose4(ep.d.xmat[ep.cap], ep.d.xpos[ep.cap])
        return super().pose(name)

    def get_current_qpos(self, name):
        side = name.split("_")[0]
        v = {"held": GRIP_HELD, "open": GRIP_OPEN, "empty": GRIP_EMPTY}[self.grip[side]]
        return np.array([v, v], np.float32)

    def tray_held(self, side):
        return bool(self.ep.d.eq_active[self.ep.weld_id[side]])

    def cap_held(self):
        return bool(self.ep.d.eq_active[self.ep.cweld_id])

    def holding(self, side):
        return "tray" if self.tray_held(side) else ("capsule" if (side == "left" and self.cap_held()) else None)

    def tab_target(self, side):
        d, ep = self.ep.d, self.ep
        return d.xpos[ep.obj] + d.xmat[ep.obj].reshape(3, 3) @ self.tab_off[side]

    def at_tab(self, side):
        return float(np.linalg.norm(self.tcp_cmd(side) - self.tab_target(side))) < AT_TAB_TOL

    def sync_mass(self):
        ep = self.ep
        ep.ctl.mass = ep.m_nom if all(self.tray_held(s) for s in SIDES) else 0.0


def _pose4(R, p):
    from .env import _pose4 as f
    return f(R, p)


# ------------------------------------------------------------------------------------------------------------------ phases
def _drop_cb(env):
    def cb(held):
        for s in SIDES:
            if env.grip[s] == "held" and (held == "tray" or s == "left"):
                env.grip[s] = "empty"
        env.sync_mass()
    return cb


def _run(env, phases, monitors):
    """phases: ("move", {side: ("d"|"to", vec)}, T) | ("home", {side: q}, T) | ("hold", T) | ("tweld", side, on) | ("cweld", on) | ("grip", side, state).  -> monitor hit | None."""
    CF, CR = SIM.CF, SIM.CR
    ep = env.ep
    for ph in phases:
        t = ep.k * CR.DT
        k = ph[0]
        if k == "tweld":
            ep.weld(ph[1], bool(ph[2]))
            env.sync_mass()
            continue
        if k == "cweld":
            if ph[1]:
                env.cap_off = ep.d.xpos[ep.cap] - env.tcp(("left"))
            CF.cweld(ep, bool(ph[1]))
            continue
        if k == "grip":
            env.grip[ph[1]] = ph[2]
            continue
        if k == "move":
            dur = ph[2]
            for s, (kind, v) in ph[1].items():
                if kind == "d":
                    errs = ep._add_path(s, t, t + dur, np.asarray(v, float))
                else:
                    errs = ep._add_path(s, t, t + dur, to=np.asarray(v, float))
                env.ik_miss = max(env.ik_miss, errs)
        elif k == "home":
            dur = ph[2]
            for s, q in ph[1].items():
                ep._add_joint_path(s, t, t + dur, [ep._cmd(s, t).copy(), np.asarray(q, float).copy()])
        else:
            dur = ph[1]
        for _ in range(int(round(dur / CR.DT))):
            hit = A._step(env, monitors)
            if hit is not None:
                ep.run_phases([("stop", A.STOP_S)])
                env.update_obj_info()
                return hit
    return None


def _dur(phases):
    return float(sum((p[2] if p[0] in ("move", "home") else (p[1] if p[0] == "hold" else 0.0)) for p in phases))


def _mt(env, dist):
    CR = SIM.CR
    return CR.move_time(float(dist), env.ep.a_plan)


def _settle(env):
    return [("hold", env.edge_settle)] if env.edge_settle > 0 else []


def _delta(cmd, env):
    kw, s = cmd.kw, cmd.side
    tcp = env.tcp_cmd(s)
    if cmd.fn == "move_by_relative_offset":
        d = np.array([kw.get("dx", 0.0), kw.get("dy", 0.0), kw.get("dz", 0.0)], float)
        return env.site_R(s) @ d if str(kw.get("mode", "extrinsic")) == "intrinsic" else d
    if cmd.fn == "move_to_absolute_position":
        tgt = tcp.copy()
        for i, k in enumerate("xyz"):
            if kw.get(k) is not None:
                tgt[i] = float(kw[k])
        return tgt - tcp
    name = kw.get("obj_name")
    if name not in env.names:
        raise A.PlanRejected("rejected_unknown_object", dict(obj_name=name, known=env.names))
    off = np.array([kw.get("x_offset", 0.0), kw.get("y_offset", 0.0), kw.get("z_offset", 0.0)], float)
    return env.obj_nominal(name) + off - tcp


def _leave(env, side):
    """a claw sitting on the tray handle but not welded retreats outward first (no sweeping through the handle)."""
    return [("move", {side: ("d", OUT[side] * RETREAT)}, _mt(env, RETREAT)), ("hold", 0.2)] if (not env.tray_held(side) and env.at_tab(side)) else []


def _goto(env, side, pre, cur):
    """straight line to `pre`; if it is far (> 0.10 m in xy) route over the handles at the transit height (as a motion planner would; ASSUMED rule, same as the ours capsule phase)."""
    CF = SIM.CF
    zt = env.T0[side][2] + CF.TRANSIT_UP
    if float(np.linalg.norm(pre[:2] - cur[:2])) <= 0.10:
        return [("move", {side: ("to", pre)}, _mt(env, np.linalg.norm(pre - cur)))]
    w1 = np.array([cur[0], cur[1], max(cur[2], zt)])
    w2 = np.array([pre[0], pre[1], max(pre[2], zt)])
    ph = []
    if np.linalg.norm(w1 - cur) > 1e-3:
        ph.append(("move", {side: ("to", w1)}, _mt(env, np.linalg.norm(w1 - cur))))
    ph.append(("move", {side: ("to", w2)}, _mt(env, np.linalg.norm(w2 - w1))))
    if np.linalg.norm(pre - w2) > 1e-3:
        ph.append(("move", {side: ("to", pre)}, _mt(env, np.linalg.norm(pre - w2))))
    return ph


def _approach(env, side, target):
    """pre-grasp 5 cm outward of `target`, then the approach (the last 5 cm along the claw's approach axis)."""
    pre = target + OUT[side] * RETREAT
    cur = env.tcp_cmd(side) + (OUT[side] * RETREAT if (not env.tray_held(side) and env.at_tab(side)) else 0.0)
    return _goto(env, side, pre, cur) + [("move", {side: ("to", target)}, _mt(env, RETREAT))]


def _plan_edge(env, L, R):
    CF = SIM.CF
    ep = env.ep
    cmds = [c for c in (L, R) if c is not None]
    for c in cmds:
        if c.fn in A.UNMAPPABLE or c.fn not in A._KNOWN:
            raise A.PlanRejected("unmappable_atom", dict(fn=c.fn))
    if not cmds:
        return [], {}
    fns = {c.fn for c in cmds}
    sides = [c.side for c in cmds]
    if len(set(sides)) != len(sides):
        raise A.PlanRejected("rejected_bad_call", dict(err="two commands for one arm"))
    if len(fns) > 1 and ({"place_on_table", "grasp"} & fns):
        raise A.PlanRejected("rejected_arm_mismatch", dict(fns=sorted(fns)))          # a coupled grasp / place needs the same atom on both arms
    fn = cmds[0].fn
    a = ep.a_plan
    if fn == "grasp":
        ph, info = [], {}
        for c in cmds:
            o, s = c.kw.get("obj_name"), c.side
            if o == "capsule":
                if s != "left":
                    raise A.PlanRejected("rejected_unreachable_arm", dict(arm=s, obj=o))
                tgt = ep.d.xpos[ep.cap] + np.array([0, 0, CF.TCP_ABOVE_CAP])
            elif o == "tray":
                tgt = env.tab_target(s)
            else:
                raise A.PlanRejected("rejected_unknown_object", dict(obj_name=o, known=env.names))
            near = float(np.linalg.norm(env.tcp_cmd(s) - tgt))
            if near > 0.01:
                ph += _leave(env, s) + _approach(env, s, tgt)
            ph += [("hold", 0.1)]
            info[s] = dict(obj=o, dist=round(near, 4))
        for c in cmds:
            o, s = c.kw.get("obj_name"), c.side
            ph += [("cweld", True)] if o == "capsule" else [("tweld", s, True)]
            ph += [("grip", s, "held")]
        return ph + [("hold", 0.2)], info
    if fn in ("move_by_relative_offset", "move_to_absolute_position", "move_relative_to_object"):
        d = {c.side: _delta(c, env) for c in cmds}
        if len(d) == 2 and env.tray_held("left") and env.tray_held("right"):
            if float(np.linalg.norm(d["left"] - d["right"])) > ARM_TOL:
                raise A.PlanRejected("rejected_arm_mismatch", dict(delta_left=d["left"].round(4).tolist(), delta_right=d["right"].round(4).tolist()))
            m = 0.5 * (d["left"] + d["right"])
            d = {"left": m, "right": m}
        ph = []
        for s in d:
            ph += _leave(env, s)
        dist = max(float(np.linalg.norm(v)) for v in d.values())
        if dist > 1e-4:
            ph.append(("move", {s: ("d", v) for s, v in d.items()}, _mt(env, dist)))
        return ph + _settle(env), dict(delta={s: v.round(4).tolist() for s, v in d.items()})
    if fn == "place_on_table":
        o = {c.kw.get("obj_name") for c in cmds}
        if len(o) != 1 or o - {"capsule", "tray"}:
            raise A.PlanRejected("rejected_unknown_object", dict(obj_names=sorted(map(str, o))))
        o = o.pop()
        try:
            xy = [(float(c.kw["x"]), float(c.kw["y"])) for c in cmds]
        except (KeyError, TypeError, ValueError) as e:
            raise A.PlanRejected("rejected_bad_call", dict(err=repr(e)))
        if max(abs(xy[0][0] - q[0]) for q in xy) > 0.01 or max(abs(xy[0][1] - q[1]) for q in xy) > 0.01:
            raise A.PlanRejected("rejected_arm_mismatch", dict(xy=xy))
        x, y = np.mean(xy, axis=0)
        pre = max(float(c.kw.get("pre_place_dis", 0.05) or 0.0) for c in cmds)
        w = ep.world
        if o == "capsule":
            if sides != ["left"]:
                raise A.PlanRejected("rejected_unreachable_arm", dict(arms=sides))
            ref = ep.d.site_xpos[ep.m.site("obj_ref").id]
            inside = abs(x - ref[0]) <= CF.INNER_X and abs(y - ref[1]) <= CF.INNER_Y
            z_rest = (ref[2] + w.base_t if inside else w.counter_top) + CF.CAP["h"] / 2 + 0.003
            cur = env.tcp_cmd("left") + env.cap_off if env.cap_held() else env.obj_pos("capsule")
            d1 = np.array([x - cur[0], y - cur[1], z_rest + pre - cur[2]])
            ph = []
            if np.linalg.norm(d1) > 1e-4:
                ph.append(("move", {"left": ("d", d1)}, _mt(env, np.linalg.norm(d1))))
            if pre > 1e-4:
                ph.append(("move", {"left": ("d", np.array([0, 0, -pre]))}, _mt(env, pre)))
            ph += _settle(env) + [("cweld", False), ("grip", "left", "open"), ("hold", 0.3), ("move", {"left": ("d", OUT["left"] * RETREAT)}, _mt(env, RETREAT)), ("hold", 0.2)]
            return ph, dict(obj=o, target_xy=[float(x), float(y)], z_rest=round(float(z_rest), 4), inside_tray=bool(inside))
        z_rest = w.counter_top + (ep.z0 - w.counter_top)
        cur = env.obj_nominal("tray") if all(env.tray_held(s) for s in SIDES) else ep.d.xpos[ep.obj]
        d1 = np.array([x - cur[0], y - cur[1], z_rest + pre - cur[2]])
        ph = []
        if np.linalg.norm(d1) > 1e-4:
            ph.append(("move", {s: ("d", d1) for s in sides}, _mt(env, np.linalg.norm(d1))))
        if pre > 1e-4:
            ph.append(("move", {s: ("d", np.array([0, 0, -pre])) for s in sides}, _mt(env, pre)))
        ph += _settle(env) + [("hold", 0.3)] + [("tweld", s, False) for s in sides] + [("grip", s, "open") for s in sides] + [("hold", 0.3)]
        ph += [("move", {s: ("d", OUT[s] * RETREAT) for s in sides}, _mt(env, RETREAT)), ("hold", 0.2)]
        return ph, dict(obj=o, target_xy=[float(x), float(y)], z_rest=round(float(z_rest), 4))
    if fn == "open_gripper":
        ph = []
        for c in cmds:
            s = c.side
            if env.tray_held(s):
                ph += [("tweld", s, False)]
            elif s == "left" and env.cap_held():
                ph += [("cweld", False)]
            ph += [("grip", s, "open")]
        return ph + [("hold", 0.2)], {}
    if fn == "close_gripper":
        return [], dict(note="no-op (weld model)")
    if fn == "back_to_initial_pose":
        ph = []
        for c in cmds:
            s = c.side
            ph += _leave(env, s)
        for c in cmds:
            s = c.side
            T0 = env.T0[s]
            if float(np.linalg.norm(env.tcp_cmd(s) - T0)) > 1e-3:
                pre = T0 + OUT[s] * RETREAT
                cur = env.tcp_cmd(s) + (OUT[s] * RETREAT if (not env.tray_held(s) and env.at_tab(s)) else 0.0)
                ph += _goto(env, s, pre, cur) + [("move", {s: ("to", T0)}, _mt(env, RETREAT)), ("home", {s: ep.q0[s]}, 0.6)]
        return ph + _settle(env), {}
    raise A.PlanRejected("unmappable_atom", dict(fn=fn))


def drive(left_arm_action=None, right_arm_action=None, monitor_sequences=None, env=None, return_result=False, **kwargs):
    CR = SIM.CR
    assert isinstance(env, CoffeeEnv)
    ep = env.ep
    rec = dict(i=len(env.log), edge_id=env.cur_edge[0], is_recovery=env.cur_edge[1], left=A._fn_of(left_arm_action), right=A._fn_of(right_arm_action),
               kw_left=getattr(left_arm_action, "keywords", None), kw_right=getattr(right_arm_action, "keywords", None), n_monitor_seqs=len(monitor_sequences or []),
               status=None, t0=round(ep.k * CR.DT, 3))
    env.log.append(rec)
    if sum(1 for r in env.log if r["status"] in ("executed", "aborted_by_monitor")) >= A.MAX_EDGES:
        rec["status"] = "budget"
        raise A.BudgetExceeded(f"more than {A.MAX_EDGES} edges")
    t_wall = time.time()
    try:
        try:
            L, R = A._call(left_arm_action, env, kwargs), A._call(right_arm_action, env, kwargs)
        except TypeError as e:
            raise A.PlanRejected("rejected_bad_call", dict(err=repr(e)))
        rec.update(edge_id=env.cur_edge[0], is_recovery=env.cur_edge[1])
        phases, info = _plan_edge(env, L, R)
        rec["info"] = info
        env.ik_miss = 0.0
        if env.inj is not None:                                 # one draw per atomic action (edge): an object held by an end-effector now
            held = "tray" if any(env.tray_held(s) for s in SIDES) else ("capsule" if env.cap_held() else None)
            env.inj.begin(ep, f"edge{rec['i']}:{rec['left'] or rec['right']}", held, _dur(phases))
        hit = _run(env, phases, monitor_sequences)
        rec["ik_miss_m"] = round(float(env.ik_miss), 4)
    except A.PlanRejected as r:
        rec.update(status=r.reason, detail=r.detail)
        raise
    rec.update(status="aborted_by_monitor" if hit else "executed", monitor=hit, t1=round(ep.k * CR.DT, 3), wall_s=round(time.time() - t_wall, 2),
               track={k: (round(float(v), 4) if k != "viol" else v) for k, v in ep.track.items() if k in ("viol", "tau_ratio", "n_sat", "obj_tilt_max")})
    if not hit:
        env.update_obj_info()
    res = dict(actions=[], monitor_index=hit["monitor_index"] if hit else None, monitor_name=hit["monitor_name"] if hit else None, step_index=hit["step_index"] if hit else None)
    return res if return_result else res["actions"]


ATOMS = A._Module("embodichain.agents.agentchord.atom_actions")
for _n in ("grasp", "place_on_table", "move_relative_to_object", "move_to_absolute_position", "move_by_relative_offset", "back_to_initial_pose", "rotate_eef", "orient_eef",
           "close_gripper", "open_gripper"):
    setattr(ATOMS, _n, A._mk(_n))
ATOMS.drive = drive


def run_world(args):
    """args: dict(plan, i, p, inj_seed, mon_period, edge_settle) -> record dict (never raises).  The world = probe._world("tray0", probe._seed("e2e", "tray0", i)) of the variant (env)."""
    import os
    from . import shim
    shim.install(scratch_dir=pathlib.Path(args["plan"]).parent / "_unused_db")
    import sys
    sys.modules["embodichain.agents.agentchord.atom_actions"] = ATOMS          # before agent_graph is imported (it does `from ...atom_actions import drive`)
    install()
    CF, CR, E, P, SW = SIM.CF, SIM.CR, SIM.E, SIM.P, SIM.SW
    SW._init_worker()                                                          # payload row in U (same as every same-worlds run); wraps CarryEpisode BEFORE coffee's subclass? see below
    CF.install()
    t0 = time.time()
    i = args["i"]
    plan = json.loads(pathlib.Path(args["plan"]).read_text())
    rec = dict(method="theirs", variant=CF.variant(), i=i, p=args.get("p", 0.0), plan_id=plan["plan_id"], proposer=plan.get("proposer"), plan_models=plan.get("models"),
               outcome=None, reason=None)
    try:
        seed = P._seed("e2e", CF.TASK, i)
        rec["seed_e2e"] = seed
        w = P._world(CF.TASK, seed)
        rec.update(load_kg=w.load_kg, load_xy=list(map(float, w.load_xy)))
        lo, hi = CF.prior()
        m_hat = 0.5 * (lo + hi)                                                # ASSUMED: AgentChord's payload feedforward = prior mid-contents (as same_worlds 'theirs')
        ep = CR.CarryEpisode(w, a_plan=E.A_NOM, m_hat=m_hat, planner="scripted")
        for s in SIDES:                                                        # nothing grasped at t = 0 (the weld is part of the grasp atom)
            ep.weld(s, False)
        ep.ctl.mass = 0.0
        ep.run_phases([("hold", 0.4)])
        ep.reset_track()
        inj = CF.Injector(args.get("p", 0.0), args.get("inj_seed", seed))
        ep.step_hooks.append(inj.step)
        env = CoffeeEnv(ep, inj=inj, mon_period=args.get("mon_period", 0.1), edge_settle=args.get("edge_settle", 0.3))
        inj.on_drop = _drop_cb(env)
        from embodichain.agents.agentchord import graph_spec as GS
        from embodichain.agents.agentchord import monitor_functions as MF
        from embodichain.agents.agentchord.agent_graph import AgentTaskGraph
        from . import execute as X
        bundle = GS.load_agent_graph_bundle(plan["graph"])
        graph = GS.compile_agent_graph_spec(bundle["task_graph"], bundle.get("recovery_graph"), env=env, graph_cls=AgentTaskGraph, action_module=ATOMS, monitor_module=MF)
        X._wrap_edges(graph, env)
        try:
            graph.run(env=env)
            rec["outcome"] = "completed"
            ep.run_phases([("hold", 0.5)])
        except A.PlanRejected as r:
            rec.update(outcome="plan_rejected", reason=r.reason)
        except A.BudgetExceeded as b:
            rec.update(outcome="budget", reason=str(b))
        except RuntimeError as e:
            rec.update(outcome="graph_error", reason=repr(e))
        done = rec["outcome"] == "completed"
        okR, whyR = ep.in_R()
        viol = ep.track["viol"]
        from world_model.v2.adapters import outcome_carry as OCC      # the ONE goal + success rule (stack.goal / stack.outcome), the same call as ours
        oc = OCC.measure(ep, CF.TASK, done, bool(viol), None, end_in_R=(okR, whyR))
        gen = OCC.measure(ep, CF.GOAL_SETS["general"], done, bool(viol), None, end_in_R=(okR, whyR))
        ex = [e for e in env.log if e["status"] in ("executed", "aborted_by_monitor")]
        fu = None
        if viol:
            e = next((e for e in env.log if (e.get("track") or {}).get("viol")), None)
            fu = dict(seg=e["edge_id"] if e else None, kind=viol.get("kind"), t=viol.get("t"))
        rec.update(edges=env.log, monitor_errors=env.monitor_errors, end_in_R=okR, R_fail=whyR, sim_s=round(ep.k * CR.DT, 2), measured=oc["measured"], goal=oc["goal"], any_U=bool(viol), first_U=fu, viol=viol,
                   result=oc["result"], goal_general=gen["goal"], result_general=gen["result"],
                   n_edges_executed=len(ex), n_rejected=sum(1 for e in env.log if str(e["status"]).startswith(("rejected", "unmappable"))),
                   rejections=[dict(edge=e["edge_id"], status=e["status"], detail=e.get("detail")) for e in env.log if str(e["status"]).startswith(("rejected", "unmappable"))],
                   monitors_fired=[dict(edge=e["edge_id"], **e["monitor"]) for e in env.log if e.get("monitor")],
                   n_recoveries_run=sum(1 for e in ex if e["is_recovery"]), injections=inj.log, n_dropped=inj.n_dropped, n_actions=len(inj.log),
                   path_unreachable=ep.track.get("path_unreachable"), u_payload="strict_claw=True (|Fz| > (3 kg - gripper) g per claw), T_cap=None")
    except Exception as e:
        import traceback
        rec.update(outcome="error", reason=repr(e), tb=traceback.format_exc()[-1500:])
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec
