"""agentchord_probe/nero/atoms.py -- the action namespace + `drive` that THEIR graph runtime (agent_graph.AgentTaskGraph) calls, mapped onto the
coupled two-claw carry of nero.contents.carry.CarryEpisode.  SIMULATION ONLY.

Their atoms return joint-space trajectories that drive() plays through env.step.  Ours return a small Cmd (what the planner asked for); drive()
resolves it against the live MuJoCo state, applies the SAME rigid-weld coupling to BOTH claws and steps CarryEpisode (FL-MPC torques).
Mapping (grasp = rigid weld, ASSUMED by the user 11-01; the weld is on from t = 0 at the scripted grasp pose):
  move_by_relative_offset / move_to_absolute_position / move_relative_to_object -> TCP delta of each arm; the COMMON delta is the mean of the two,
        accepted only if the two arms' deltas agree within ARM_TOL (rigid object => the claws cannot move differently), else "rejected_arm_mismatch";
        once the weld is off (after open_gripper / place_on_table) each arm moves by its OWN delta ("move2"; 11-03 fix: the box plan's
        withdraw step dy +0.1 / -0.1 after release was wrongly rejected).  No arm-arm clearance row is checked for these moves.
  place_on_table(x, y, pre_place_dis) -> object to (x, y) pre_place_dis above the support top (counter; the SHELF if x,y is on it in the box task),
        down to rest, then release = carry.segment_phases("release") (hold, weld off, retreat 5 cm outward, hold)
  open_gripper -> weld off, hold 0.2 s.      close_gripper -> no-op (weld is on).
  grasp -> no-op while the weld is on; if the weld is off (after open_gripper): re-weld only when both TCPs are within REGRASP_TOL of the object's
        grasp sites, else "rejected_regrasp_far" (a geometric re-approach is out of scope).
  back_to_initial_pose -> both TCPs retrace to their t = 0 poses (the object comes along while welded).
  rotate_eef / orient_eef / any unknown name -> "unmappable_atom" (not executed).
IK: the straight-line TCP path is solved by CarryEpisode._add_path exactly as in carry.py (continuation IK); its final-waypoint miss is RECORDED per edge
  (ik_miss_m) and never rejects: the certified box carry itself misses by 37 mm on the right arm (seen 11-03), and the physics shows the consequence.
REJECTIONS (user decision 11-03) stop the episode: the edge is recorded (status rejected_*), drive raises PlanRejected, nothing of the edge or
of the rest of the plan runs, outcome = "plan_rejected".  One-arm: an edge where only one arm has an action, or the two arms' functions differ.
Monitors fire exactly as in their drive: after each sample the monitor partials are called in order; the first True aborts the edge and the graph
runs the compiled recovery branch.  On an abort the commanded path is stopped with carry's ("stop", 0.3 s) phase (their robot would freeze at the
last joint target; our torque-controlled arms need the reference decelerated).
"""
from __future__ import annotations

import time
import types

import numpy as np

from .env import SIDES, NeroEnv
from .sim_imports import CR, SC

ARM_TOL = 0.03          # m, ASSUMED: max disagreement of the two arms' requested TCP deltas that is still accepted as one coupled move
REGRASP_TOL = 0.05      # m, ASSUMED: max TCP-to-grasp-site distance for a re-weld after open_gripper
STOP_S = 0.3            # s, carry.run_phases "stop" duration used by backups (MODEL: same value as backup_phases)
MAX_SIM_S = 120.0       # s, ASSUMED budget per episode (recovery loops can repeat edges up to max_transitions=1000)
MAX_EDGES = 40         # ASSUMED budget on executed edges per episode
UNMAPPABLE = ("rotate_eef", "orient_eef")


class PlanRejected(Exception):
    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class BudgetExceeded(Exception):
    pass


class Cmd:
    def __init__(self, fn, robot_name, kw):
        self.fn, self.kw = fn, dict(kw)
        self.robot_name = robot_name
        self.side = "right" if "right" in str(robot_name) else "left"      # their convention (monitor_utils.get_gripper_distance)

    def __repr__(self):
        return f"Cmd({self.fn}, {self.robot_name}, {self.kw})"


def _mk(fn):
    def atom(robot_name="left_arm", env=None, **kw):
        return Cmd(fn, robot_name, kw)
    atom.__name__ = fn
    return atom


class _Module(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _mk(name)                      # unknown atom: still compiles; drive() rejects it ("unmappable_atom")


ATOMS_MODULE = _Module("embodichain.agents.agentchord.atom_actions")
for _n in ("grasp", "place_on_table", "move_relative_to_object", "move_to_absolute_position", "move_by_relative_offset",
           "back_to_initial_pose", "rotate_eef", "orient_eef", "close_gripper", "open_gripper"):
    setattr(ATOMS_MODULE, _n, _mk(_n))


# ------------------------------------------------------------------------------------------------------------------ stepping
def _eval_monitors(env, monitor_sequences):
    for idx, seq in enumerate(monitor_sequences or []):
        for fn in seq:
            try:
                hit = fn()
            except Exception as e:                       # a monitor that cannot be evaluated is recorded, never silently dropped
                env.monitor_errors.append(f"{getattr(getattr(fn, 'func', fn), '__name__', 'monitor')}: {e!r}")
                continue
            if hit == True:                              # noqa: E712  (as in their drive)
                return dict(monitor_index=idx, monitor_name=getattr(fn.func, "__name__", type(fn).__name__), step_index=env.t_steps)
    return None


def _step(env, monitors):
    ep = env.ep
    ep._step()
    env.t_steps += 1
    if ep.k * CR.DT > MAX_SIM_S:
        raise BudgetExceeded(f"sim time > {MAX_SIM_S} s")
    if env.t_steps % env.mon_period_steps == 0:
        hit = _eval_monitors(env, monitors)
        if hit is not None:
            env.update_obj_info()
            return hit
        env.update_obj_info()
    return None


def _run_phases(env, phases, monitors):
    """interpret carry-style phases step by step with monitor sampling.  -> monitor hit dict or None.
    phases: ("move", delta, T) | ("move2", {side: delta}, T) | ("home", {side: tcp}, T) | ("hold", T) | ("weld", on) | ("retreat", r, T)."""
    ep = env.ep
    for ph in phases:
        t = ep.k * CR.DT
        if ph[0] == "weld":
            ep.run_phases([ph])
            continue
        if ph[0] == "move":
            dur = ph[2]
            errs = [ep._add_path(s, t, t + dur, np.asarray(ph[1], float)) for s in SIDES]
            env.ik_miss = max(env.ik_miss, max(errs))
        elif ph[0] == "move2":                            # per-arm deltas, only issued while the weld is off
            dur = ph[2]
            errs = [ep._add_path(s, t, t + dur, np.asarray(ph[1][s], float)) for s in SIDES]
            env.ik_miss = max(env.ik_miss, max(errs))
        elif ph[0] == "home":
            dur = ph[2]
            errs = []
            for s in SIDES:
                ep._trim(s, t)
                errs.append(ep._add_path(s, t, t + dur, to=ph[1][s]))
            env.ik_miss = max(env.ik_miss, max(errs))
        elif ph[0] == "retreat":
            dur = ph[2]
            for s in SIDES:
                ep._add_path(s, t, t + dur, np.array([0, ph[1] if s == "left" else -ph[1], 0]))
        else:                                              # hold
            dur = ph[1]
        for _ in range(int(round(dur / CR.DT))):
            hit = _step(env, monitors)
            if hit is not None:
                ep.run_phases([("stop", STOP_S)])          # freeze the reference (no monitors while stopping)
                env.update_obj_info()
                return hit
    return None


# ------------------------------------------------------------------------------------------------------------------ resolution
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
    if cmd.fn == "move_relative_to_object":
        name = kw.get("obj_name")
        if name not in env.names:
            raise PlanRejected("rejected_unknown_object", dict(obj_name=name, known=env.names))
        off = np.array([kw.get("x_offset", 0.0), kw.get("y_offset", 0.0), kw.get("z_offset", 0.0)], float)
        return env.obj_nominal(name) + off - tcp
    raise AssertionError(cmd.fn)


def _support(env, x, y):
    w = env.ep.world
    if env.task == "box" and getattr(w, "shelf", False):
        x0, x1 = w.shelf_x
        if x0 <= x <= x1 and abs(y - w.shelf_yc) <= w.shelf_yhalf:
            return "shelf", w.shelf_top
    return "counter", w.counter_top


def _settle(env):
    """dwell after a motion edge so the torque-controlled arms finish tracking: carry.segment_phases has ("hold", 0.3) after every move
    (their position-controlled robot is at its target when the trajectory ends).  env.edge_settle = 0 turns it off."""
    return [("hold", env.edge_settle)] if env.edge_settle > 0 else []


def _plan_edge(env, L, R):
    """-> (phases, info).  Raises PlanRejected."""
    ep = env.ep
    for c in (L, R):
        if c is not None and (c.fn in UNMAPPABLE or c.fn not in _KNOWN):
            raise PlanRejected("unmappable_atom", dict(fn=c.fn))
    if L is None or R is None or L.fn != R.fn:
        raise PlanRejected("rejected_one_arm", dict(left=getattr(L, "fn", None), right=getattr(R, "fn", None)))
    fn = L.fn
    a = ep.a_plan
    if fn in ("move_by_relative_offset", "move_to_absolute_position", "move_relative_to_object"):
        dl, dr = _delta(L, env), _delta(R, env)
        if not env.weld_on():                              # object let go: the arms are independent
            dist = max(float(np.linalg.norm(dl)), float(np.linalg.norm(dr)))
            return (([("move2", dict(left=tuple(dl), right=tuple(dr)), CR.move_time(dist, a))] + _settle(env)) if dist > 1e-4 else []), \
                dict(delta_left=dl.round(4).tolist(), delta_right=dr.round(4).tolist(), released=True)
        if float(np.linalg.norm(dl - dr)) > ARM_TOL:
            raise PlanRejected("rejected_arm_mismatch", dict(delta_left=dl.round(4).tolist(), delta_right=dr.round(4).tolist()))
        d = 0.5 * (dl + dr)
        dist = float(np.linalg.norm(d))
        return ([("move", tuple(d), CR.move_time(dist, a))] + _settle(env) if dist > 1e-4 else []), dict(delta=d.round(4).tolist())
    if fn == "place_on_table":
        for c in (L, R):
            if c.kw.get("obj_name") != env.carried:
                raise PlanRejected("rejected_unknown_object", dict(obj_name=c.kw.get("obj_name"), carried=env.carried))
        try:
            xl, yl, xr, yr = (float(L.kw["x"]), float(L.kw["y"]), float(R.kw["x"]), float(R.kw["y"]))
        except (KeyError, TypeError, ValueError) as e:
            raise PlanRejected("rejected_bad_call", dict(err=repr(e)))
        if max(abs(xl - xr), abs(yl - yr)) > 0.01:
            raise PlanRejected("rejected_arm_mismatch", dict(left=[xl, yl], right=[xr, yr]))
        x, y = 0.5 * (xl + xr), 0.5 * (yl + yr)
        pre = max(float(L.kw.get("pre_place_dis", 0.05) or 0.0), float(R.kw.get("pre_place_dis", 0.05) or 0.0))
        sup, top = _support(env, x, y)
        cur = env.obj_nominal(env.carried)
        z_rest = top + (ep.z0 - ep.world.counter_top)          # object frame origin at rest on that support
        ph = []
        d1 = np.array([x - cur[0], y - cur[1], z_rest + pre - cur[2]])
        if np.linalg.norm(d1) > 1e-4:
            ph.append(("move", tuple(d1), CR.move_time(float(np.linalg.norm(d1)), a)))
        if pre > 1e-4:
            ph.append(("move", (0.0, 0.0, -pre), CR.move_time(pre, a)))
        return ph + _settle(env) + ep.segment_phases("release"), dict(support=sup, target_xy=[x, y], z_rest=round(float(z_rest), 4), pre=pre)
    if fn == "open_gripper":
        return [("weld", False), ("hold", 0.2)], {}
    if fn == "close_gripper":
        return [], dict(note="no-op (weld on)")
    if fn == "grasp":
        if env.weld_on():
            return [], dict(note="no-op (already welded at the scripted grasp pose)")
        far = {s: float(np.linalg.norm(env.tcp(s) - ep.d.site_xpos[ep.m.site(f"obj_grasp_{s}").id])) for s in SIDES}
        if max(far.values()) > REGRASP_TOL:
            raise PlanRejected("rejected_regrasp_far", far)
        return [("weld", True), ("hold", 0.2)], dict(note="re-weld at the grasp sites", dist=far)
    if fn == "back_to_initial_pose":
        dist = max(float(np.linalg.norm(env.init_tcp[s] - env.tcp_cmd(s))) for s in SIDES)
        return ([("home", env.init_tcp, CR.move_time(dist, a))] + _settle(env) if dist > 1e-3 else []), dict(dist=round(dist, 4))
    raise PlanRejected("unmappable_atom", dict(fn=fn))


_KNOWN = {"grasp", "place_on_table", "move_relative_to_object", "move_to_absolute_position", "move_by_relative_offset",
          "back_to_initial_pose", "close_gripper", "open_gripper"} | set(UNMAPPABLE)


def _call(action, env, kwargs):
    if action is None:
        return None
    return action(env=env, **kwargs) if callable(action) else action


def _fn_of(a):
    return None if a is None else getattr(getattr(a, "func", a), "__name__", str(a))


# ------------------------------------------------------------------------------------------------------------------ drive
def drive(left_arm_action=None, right_arm_action=None, monitor_sequences=None, env=None, return_result=False, **kwargs):
    """same signature / return as their atom_actions.drive (return_result=True)."""
    assert isinstance(env, NeroEnv), "atoms.drive needs a NeroEnv"
    ep = env.ep
    rec = dict(i=len(env.log), edge_id=env.cur_edge[0], is_recovery=env.cur_edge[1], left=_fn_of(left_arm_action), right=_fn_of(right_arm_action),
               kw_left=getattr(left_arm_action, "keywords", None), kw_right=getattr(right_arm_action, "keywords", None),
               n_monitor_seqs=len(monitor_sequences or []), status=None, t0=round(ep.k * CR.DT, 3))
    env.log.append(rec)
    n_exec = sum(1 for r in env.log if r["status"] in ("executed", "aborted_by_monitor"))
    if n_exec >= MAX_EDGES:
        rec["status"] = "budget"
        raise BudgetExceeded(f"more than {MAX_EDGES} edges")
    t_wall = time.time()
    try:
        try:
            L, R = _call(left_arm_action, env, kwargs), _call(right_arm_action, env, kwargs)
        except TypeError as e:
            raise PlanRejected("rejected_bad_call", dict(err=repr(e)))
        rec.update(edge_id=env.cur_edge[0], is_recovery=env.cur_edge[1])          # set by execute._wrap_edges when the action is called
        phases, info = _plan_edge(env, L, R)
        rec["info"] = info
        env.ik_miss = 0.0
        hit = _run_phases(env, phases, monitor_sequences)
        rec["ik_miss_m"] = round(float(env.ik_miss), 4)
    except PlanRejected as r:
        rec.update(status=r.reason, detail=r.detail)
        raise
    rec.update(status="aborted_by_monitor" if hit else "executed", monitor=hit, t1=round(ep.k * CR.DT, 3),
               wall_s=round(time.time() - t_wall, 2),
               track={k: (round(float(v), 4) if k != "viol" else v) for k, v in ep.track.items() if k in
                      ("viol", "tau_ratio", "n_sat", "obj_tilt_max", "cup_tilt_max", "spill_ratio")})
    if not hit:
        env.update_obj_info()
    res = dict(actions=[], monitor_index=hit["monitor_index"] if hit else None, monitor_name=hit["monitor_name"] if hit else None,
               step_index=hit["step_index"] if hit else None)
    return res if return_result else res["actions"]


ATOMS_MODULE.drive = drive            # agent_graph.py does `from ...atom_actions import drive`
