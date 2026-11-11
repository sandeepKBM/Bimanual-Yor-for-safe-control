"""agentchord_probe/nero/behind.py -- AgentChord's pipeline on the table-behind tasks tray6_behind / box_behind (swerve base) in the NERO sim.  SIMULATION ONLY
(the only network call is the owner-run Codex plan generation through the loopback gateway; execution is offline; tests use plan.Canned).  New module in the coffee.py
pattern: atoms.py / execute.py / tasks.py / plan.py are NOT edited (design docs/status/table_behind_design_2026-11-09.md section 5).

  install(level)     register the tasks (task text = our proposer's task facts, rep world, image) in agentchord_probe.nero.tasks; extend plan.SETUP_DEVIATIONS (once).
  atom_actions_text()/write_prompts(out)   THEIR atom_actions.txt + one `move_base` entry (style: "Use the following functions exactly as defined"); written to the plan dir
                     (their loader looks in the config dir first).  Their other prompt files are untouched.
  make_plan(task, out, ...)                plan.make_plan with the behind prompt dir.
  run_world(args)    execute the cached compiled graph on world i with THEIR AgentTaskGraph.run + monitors and OUR atoms (below).

New atom  move_base(robot_name, route, **kwargs)   route = [["line", metres] | ["turn", degrees], ...]  (line along the robot's CURRENT +x, turn CCW), as the contract
  base_move.  Both arms' slots of ONE edge with the same route (else rejected_arm_mismatch; only one arm / different atoms: rejected_one_arm, as atoms.py).  The whole robot
  drives; arms hold their joint references (ASSUMED A4) with the same tracker and feedforward mass as the other edges.  Speed = the CELL CAP `level` (no speed argument: they
  have no speed reasoning).  Their monitors run every MON_PERIOD_S during the edge; an abort decelerates the base at the cap (driver state emptied, `PathFollower._limit(0)`),
  then their recovery branch runs.
Frames: atoms take WORLD targets (= the start base frame).  After a base move: tcp_cmd is FK in the CURRENT base pose, obj_off (object minus claw midpoint) is rotated by the
  base yaw change, the release retreat is outward in the ROBOT frame (ep.body_delta), back_to_initial_pose = joint path to the t = 0 arm configuration.
Monitor frame (MONITOR_FRAME): obj_info / get_local_pose read by THEIR monitors are in the ROBOT BASE frame (their robot observes in its base frame): in the world frame
  monitor_object_moved would fire on the base motion itself (object displacement per sample = v * MON_PERIOD_S > its 0.02 m default) and no base edge could run.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch

from world_model.v2.stack import tasks as TK

from . import atoms as A
from . import plan as PL
from . import tasks as T
from .env import SIDES, NeroEnv
from .sim_imports import CR, E, SC

TASKS_BEHIND = ("tray6_behind", "box_behind")
LEVEL = "L3"                 # ASSUMED: default cell cap = the highest level of the grid (design section 6); the CLI passes the cell's cap
MONITOR_FRAME = "base"       # ASSUMED deviation: "base" (poses in the robot base frame, see module doc) | "world"
STOP_MAX_S = 4.0             # ASSUMED s: budget of the deceleration loop after a monitor abort (L1: v/a = 1 s, L3: 1 s; bounded so a bug cannot loop)
V_STOP = 1e-3                # ASSUMED m/s and rad/s: base twist below which the abort deceleration is finished
TURN_FACT = "Turning in place at the pick pose hits counter 1."      # the scene fact our proposer gets (stack/proposer/prompt.py _task_spec_behind); test_behind checks it is still there

SETUP_DEVIATIONS_BEHIND = [
    "table-behind: atom `move_base(robot_name, route)` ADDED to a copy of their atom_actions.txt (their other prompt files unchanged); both arms must carry the same route "
    "(else rejected_arm_mismatch)",
    "table-behind: base speed = the cell cap (level; v, a, yaw rate, yaw accel together), fixed for the run; the model has no speed argument and no speed reasoning "
    "(the cap is stated in the prompt as a robot fact)",
    "table-behind: THEIR monitors run every MON_PERIOD_S during a base edge; on abort the base decelerates at the cap (their robot would freeze; a moving base cannot stop instantly), "
    "then their recovery branch runs; monitor poses are in the robot base frame (world frame would fire on the base motion itself)",
    "table-behind: world-frame targets with a moving base: atoms resolve world coordinates with the sim-truth base pose (tcp FK in the current base pose); their prompt calls the "
    "frame 'robot base frame', here world = the START base frame (extends the world-frame-metres deviation); back_to_initial_pose = the t = 0 arm joint configuration",
    "table-behind: no route help beyond the scene facts our proposer also gets (route R1, the turn-at-pick fact); a colliding route runs and the sim shows the collision; "
    "no tip / claw / speed checks, no admission, no level choice; drop injection off",
]


def install(level=None):
    """idempotent: task text / rep world for TASKS_BEHIND in agentchord_probe.nero.tasks; SETUP_DEVIATIONS extended once."""
    for s in SETUP_DEVIATIONS_BEHIND:
        if s not in PL.SETUP_DEVIATIONS:
            PL.SETUP_DEVIATIONS.append(s)
    if level is not None:
        global LEVEL
        LEVEL = level
    if getattr(T, "_behind_installed", False):
        return
    T._behind_installed = True
    rw0, tp0 = T.rep_world, T.task_prompt
    T.rep_world = lambda task: rep_world(task) if task in TASKS_BEHIND else rw0(task)
    T.task_prompt = lambda task, world=None: task_prompt(task, world) if task in TASKS_BEHIND else tp0(task, world)


def base_task(task):
    return task[:-len("_behind")]


# ------------------------------------------------------------------------------------------------------------------ prompt
def rep_world(task):
    """representative world for the planner's facts: seed 0 over the full prior box (as tasks.rep_world), with the task's world_kw (box_behind: shelf=False)."""
    from dataclasses import replace
    from world_model.v2.nero.base import smoke as SM
    w = SM.PR._world(base_task(task), 0)
    return replace(w, **dict(TK.task(task).world_kw))


def task_prompt(task, world=None):
    """-> (text, facts).  Same scene facts as our proposer (stack.proposer.prompt._task_spec_behind): object, start / goal xy, counters, base start pose, route R1, the
    turn-at-pick fact; plus the base level cap as a robot fact.  Hidden fill / truth are NOT in the text; `facts` = the contents facts text of the other tasks."""
    from world_model.v2.stack.proposer import prompt as PRM
    from world_model.v2.vlm import contents_eval as CE
    from world_model.v2.nero.base import smoke as SM
    w = world or rep_world(task)
    sp = PRM._task_spec_behind(task, w)
    assert TURN_FACT in sp["instruction"], "proposer prompt no longer states the turn-at-pick fact: update TURN_FACT"
    td = TK.task(task)
    facts = CE.facts_and_truth(base_task(task), w)[0]
    ob, sup = sp["object"], sp["supports"]
    what = ob["name"] + (" (it holds hidden contents)" if td.kind == "box" else "")
    r1 = json.dumps([[k, v] for k, v in td.routes[0][1]])
    L = SM.level_limits(LEVEL)
    c1, c2 = sup["counter"], sup["counter2"]
    text = (f"Use the robot's two arms together to carry the {what} (object name '{td.kind}') from counter 1 (object name 'counter', top surface at "
            f"z = {c1['top']:.2f} m, spanning x = {c1['x'][0]:.2f}..{c1['x'][1]:.2f} m, y = {c1['y'][0]:.2f}..{c1['y'][1]:.2f} m) to counter 2 BEHIND the robot (object name 'counter2', "
            f"top surface at z = {c2['top']:.2f} m, spanning x = {c2['x'][0]:.2f}..{c2['x'][1]:.2f} m, y = {c2['y'][0]:.2f}..{c2['y'][1]:.2f} m). The object's reference point is now at "
            f"x = {sp['start_xy'][0]:.2f} m, y = {sp['start_xy'][1]:.2f} m on counter 1. Set it down on counter 2 with its reference point at x = {sp['goal_xy'][0]:.2f} m, "
            f"y = {sp['goal_xy'][1]:.2f} m, then let go of it.{' Do not spill.' if td.kind == 'tray' else ''}\n"
            f"The robot can drive: the atom move_base(robot_name, route) moves the whole robot with the same route given to both arms; the arms hold still and keep carrying the object "
            f"during it, and the robot's +x axis turns with the base (after a 180 degree turn the robot faces -x). The base starts at x = {sp['base']['start_pose_xy_yaw_deg'][0]:.2f} m, "
            f"y = {sp['base']['start_pose_xy_yaw_deg'][1]:.2f} m, yaw {sp['base']['start_pose_xy_yaw_deg'][2]:.0f} deg (rotation centre; in-place turns turn about it). "
            f"Known-good route R1 (it sweeps clear): {r1}. {TURN_FACT}\n"
            f"Robot facts: the base drives at these limits (fixed, not an argument): speed {L.v_max:g} m/s, acceleration {L.a_max:g} m/s^2, turn rate {L.w_max:g} rad/s, "
            f"turn acceleration {L.alpha_max:g} rad/s^2.\n"
            f"Positions are world coordinates in metres (the start base frame: x forward, y left, z up) and stay world coordinates after the base has moved.\n"
            f"Known facts about the load: {facts}")
    return text, facts


def atom_actions_text():
    """THEIR atom_actions.txt + the move_base entry (written in their style)."""
    from . import shim
    shim.install()
    src = pathlib.Path(shim.THEIR_SRC) / "embodichain" / "agents" / "prompts" / "atom_actions.txt"
    entry = (
        '\n\n"move_base":\n'
        "    def move_base(robot_name: str, route: list, **kwargs) -> list[np.ndarray]\n"
        "    Drives the whole robot (the mobile base) along a route while both arms keep holding what they hold. It is one coupled action: give the SAME call, with the same route, "
        "to the left_arm slot and the right_arm slot of the edge.\n"
        '    route is a list of [kind, value] segments run in order: ["line", d] drives d metres (negative = backwards) along the robot\'s CURRENT +x axis; '
        '["turn", deg] turns in place by deg degrees counter-clockwise about the base centre. The robot\'s +x axis turns with the base.\n'
        "    The base speed is the robot's own limit (see the robot facts); there is no speed argument.\n"
        "    Arm targets given after a base move (move_to_absolute_position, place_on_table) stay world coordinates.\n"
        "    Example:\n"
        '        move_base(robot_name=\'left_arm\', route=[["line", -0.30], ["turn", 90.0], ["line", 0.30]])  # Drives the robot 0.30 m backwards, turns it 90 degrees counter-clockwise in place, then drives 0.30 m along its new +x axis. The same call is given to right_arm.'
    )
    return src.read_text().strip() + entry


def write_prompts(out):
    """the behind prompt dir = the plan dir: their loader checks the config dir first (agent_base._resolve_prompt_path), so atom_actions.txt here replaces theirs for this plan."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "atom_actions.txt").write_text(atom_actions_text())
    return out


def render_image(world):
    """MuJoCo torso view of the swerve two-table scene (the camera sees what is in front; counter 2 is behind it)."""
    from world_model.v2.nero.base import scene as BS
    with BS.use_swerve_scene():
        return T.render_rgb(world)


def make_plan(task, out, spec="codex:gpt-5.6-sol:high", level=None, regenerate=False, backend=None, canned=None, rgb=None):
    """one AgentChord plan (Task -> Recovery -> Compile agents) through the loopback gateway.  backend None: the codex backend with the model / reasoning of `spec` (ONE spec for
    both methods); 'canned' + canned=<json path> for tests (no model).  The level cap is recorded in plan.json (behind.level)."""
    install(level)
    out = write_prompts(out)
    task = str(task)
    assert task in TASKS_BEHIND, task
    if backend == "canned":
        meta = PL.make_plan(task, out, backend="canned", canned=canned, regenerate=regenerate, rgb=rgb if rgb is not None else render_image(rep_world(task)))
    else:
        kind, model, *r = spec.split(":")
        assert kind == "codex", "the comparison uses ONE codex spec"
        meta = PL.make_plan(task, out, backend="codex", model=model, reasoning=(r[0] if r else None), regenerate=regenerate,
                            rgb=rgb if rgb is not None else render_image(rep_world(task)))
    from world_model.v2.nero.base import smoke as SM
    L = SM.level_limits(LEVEL)
    meta["behind"] = dict(level=LEVEL, v_max=L.v_max, a_max=L.a_max, w_max=L.w_max, alpha_max=L.alpha_max, monitor_frame=MONITOR_FRAME)
    (out / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ------------------------------------------------------------------------------------------------------------------ env
class BehindEnv(NeroEnv):
    """NeroEnv on a swerve-base episode.  pose(): the frame THEIR monitors read (MONITOR_FRAME); wpose(): world (atoms)."""

    def __init__(self, ep, level=None, mon_period=0.1, edge_settle=0.3):
        self.level = level or LEVEL
        self._yaw0 = float(ep.sb.pose()[2])
        self._xy0 = np.array(ep.sb.pose()[:2], float)
        super().__init__(ep, mon_period=mon_period, edge_settle=edge_settle)
        self.names = self.names + ["counter2"]
        self.base_log = []
        self.update_obj_info()

    # ---- frames
    def _Tbw(self):
        """world -> base-frame transform (4x4) of the current sim-truth base pose (x, y, yaw)."""
        x, y, yaw = (float(v) for v in self.ep.sb.pose())
        c, s = np.cos(yaw), np.sin(yaw)
        T4 = np.eye(4)
        T4[:2, :2] = [[c, s], [-s, c]]
        T4[:2, 3] = -T4[:2, :2] @ [x, y]
        return T4

    def wpose(self, name):
        return super().pose(name)

    def pose(self, name):
        Tw = self.wpose(name)
        if MONITOR_FRAME == "world":
            return Tw
        return torch.from_numpy((self._Tbw() @ Tw.numpy().astype(float)).astype(np.float32))

    def obj_pos(self, name):
        return self.wpose(name)[:3, 3].numpy().astype(float)

    def tcp_cmd(self, side):
        """FK of the last commanded q in the CURRENT base pose (carry._add_path does the same: the IK's own MjData sits at the XML pose)."""
        ep = self.ep
        q = ep.paths[side][-1][2][-1] if ep.paths[side] else ep.q0[side]
        if side not in self._pik:
            from world_model.v2.nero.democ import kin as K
            self._pik[side] = K.PoseIK(ep.m, side)
        pik = self._pik[side]
        pik.ik.d.qpos[:] = ep.d.qpos
        return np.asarray(pik.ik.pose(q)[0], float).copy()

    def obj_nominal(self, name):
        """as NeroEnv, with obj_off rotated by the base yaw change (the offset object-minus-claw-midpoint turns with the robot)."""
        if name == self.carried and self.weld_on():
            a = float(self.ep.sb.pose()[2]) - self._yaw0
            c, s = np.cos(a), np.sin(a)
            off = np.array([c * self.obj_off[0] - s * self.obj_off[1], s * self.obj_off[0] + c * self.obj_off[1], self.obj_off[2]])
            return np.mean([self.tcp_cmd(s_) for s_ in SIDES], axis=0) + off
        return self.obj_pos(name)


# ------------------------------------------------------------------------------------------------------------------ atoms
ATOMS_MODULE = A._Module("embodichain.agents.agentchord.atom_actions")
for _n in ("grasp", "place_on_table", "move_relative_to_object", "move_to_absolute_position", "move_by_relative_offset", "back_to_initial_pose", "rotate_eef", "orient_eef",
           "close_gripper", "open_gripper", "move_base"):
    setattr(ATOMS_MODULE, _n, A._mk(_n))


def _route(c):
    """the route of a move_base Cmd as [[kind, float], ...]; PlanRejected(rejected_bad_call) when malformed."""
    r = c.kw.get("route")
    try:
        out = [[str(s[0]), float(s[1])] for s in r]
        if not out or any(k not in ("line", "turn") for k, _ in out):
            raise ValueError(f"route needs >= 1 segment of kind line | turn, got {r!r}")
        return out
    except (TypeError, ValueError, IndexError, KeyError) as e:
        raise A.PlanRejected("rejected_bad_call", dict(err=repr(e), fn="move_base"))


ROUTE_TOL = 1e-6        # ASSUMED: two routes are "the same" when every segment value agrees within this (floats from JSON)


def _support_behind(env, x, y):
    """(name, top) of the support under world (x, y): counter2 / counter by their world extents (scene.support_tops_behind, MODEL), else the counter."""
    from world_model.v2.nero.base import scene as BS
    for nm, s in BS.support_tops_behind(env.ep.world).items():
        if nm in ("counter2", "counter") and s["x"][0] <= x <= s["x"][1] and s["y"][0] <= y <= s["y"][1]:
            return nm, float(s["top"])
    return "counter", float(env.ep.world.counter_top)


def _plan_edge(env, L, R):
    """-> (phases, info).  move_base / place_on_table / back_to_initial_pose here; everything else = atoms._plan_edge (world-frame deltas work unchanged)."""
    ep = env.ep
    fns = {getattr(c, "fn", None) for c in (L, R)}
    if "move_base" in fns:
        if L is None or R is None or L.fn != R.fn:
            raise A.PlanRejected("rejected_one_arm", dict(left=getattr(L, "fn", None), right=getattr(R, "fn", None)))
        rl, rr = _route(L), _route(R)
        if len(rl) != len(rr) or any(a[0] != b[0] or abs(a[1] - b[1]) > ROUTE_TOL for a, b in zip(rl, rr)):
            raise A.PlanRejected("rejected_arm_mismatch", dict(route_left=rl, route_right=rr))
        return [("base", dict(route=rl, level=env.level))], dict(route=rl, level=env.level)
    if fns == {"place_on_table"}:
        for c in (L, R):
            if c.kw.get("obj_name") != env.carried:
                raise A.PlanRejected("rejected_unknown_object", dict(obj_name=c.kw.get("obj_name"), carried=env.carried))
        try:
            xl, yl, xr, yr = (float(L.kw["x"]), float(L.kw["y"]), float(R.kw["x"]), float(R.kw["y"]))
        except (KeyError, TypeError, ValueError) as e:
            raise A.PlanRejected("rejected_bad_call", dict(err=repr(e)))
        if max(abs(xl - xr), abs(yl - yr)) > 0.01:
            raise A.PlanRejected("rejected_arm_mismatch", dict(left=[xl, yl], right=[xr, yr]))
        x, y = 0.5 * (xl + xr), 0.5 * (yl + yr)
        pre = max(float(L.kw.get("pre_place_dis", 0.05) or 0.0), float(R.kw.get("pre_place_dis", 0.05) or 0.0))
        sup, top = _support_behind(env, x, y)
        cur = env.obj_nominal(env.carried)
        z_rest = top + (ep.z0 - ep.world.counter_top)
        ph = []
        d1 = np.array([x - cur[0], y - cur[1], z_rest + pre - cur[2]])
        if np.linalg.norm(d1) > 1e-4:
            ph.append(("move", tuple(d1), CR.move_time(float(np.linalg.norm(d1)), ep.a_plan)))
        if pre > 1e-4:
            ph.append(("move", (0.0, 0.0, -pre), CR.move_time(pre, ep.a_plan)))
        return ph + A._settle(env) + ep.segment_phases("release"), dict(support=sup, target_xy=[x, y], z_rest=round(float(z_rest), 4), pre=pre)
    if fns == {"back_to_initial_pose"} and L is not None and R is not None:
        if float(abs(ep.sb.pose()[2] - env._yaw0)) < 1e-2 and float(np.linalg.norm(ep.sb.pose()[:2] - env._xy0)) < 1e-2:
            return A._plan_edge(env, L, R)                     # base at its start pose: atoms' TCP form is valid
        if env.weld_on():
            raise A.PlanRejected("rejected_back_to_initial_pose_holding", dict(note="after a base move with the object still welded"))
        t = ep.k * CR.DT
        dur = CR.move_time(1.0, ep.a_plan)               # MODEL: a 1 m-equivalent joint path time (the arms are free; ASSUMED duration scale)
        return [("jhome", {s: ep.q0[s] for s in SIDES}, dur)] + A._settle(env), dict(note="joint path to the t=0 arm configuration", t=round(t, 3))
    return A._plan_edge(env, L, R)


# ------------------------------------------------------------------------------------------------------------------ stepping
def _run_base(env, ref, monitors):
    """one base route at the cell cap with THEIR monitors every MON_PERIOD_S.  -> (record dict, hit | None).  On a hit: driver state emptied, the base decelerates at the cap."""
    from world_model.v2.nero.base import drive as DR, monitors as MON, smoke as SM
    ep = env.ep
    sb, drv = ep.sb, ep.driver
    sb.lim = SM.level_limits(ref["level"])
    route = DR.route_to_segments(ref["route"])
    p0 = sb.pose()
    nom = DR.nominal_end(p0, ref["route"])
    T_s = drv.start_route(route)
    n = int(round((T_s + DR.END_SETTLE_S) / DR.DT))
    hit, k_hit = None, None
    for i in range(n):
        hit = A._step(env, monitors)
        if hit is not None:
            k_hit = i
            break
    out = dict(level=ref["level"], T_s=float(T_s), start=p0.tolist(), nominal_end=nom.tolist(), aborted=hit is not None)
    if hit is not None:
        pa, va = sb.pose(), np.array(sb._v, float)
        drv.abort()                                         # no segment left: __call__ rate-limits the twist to 0 at the level's a / alpha
        for _ in range(int(STOP_MAX_S / DR.DT)):
            ep._step()
            if float(np.max(np.abs(sb._v))) < V_STOP:
                break
        drv.active = False
        ep.run_phases([("stop", A.STOP_S)])
        pb = sb.pose()
        out.update(abort_t_in_edge_s=round(k_hit * DR.DT, 3), v_at_abort=va.round(3).tolist(), stop_dist_m=float(np.linalg.norm(pb[:2] - pa[:2])), end=pb.tolist())
        env.update_obj_info()
        env.base_log.append(out)
        ep.base_log = getattr(ep, "base_log", []) + [out]
        return out, hit
    drv.active = False
    pb = sb.pose()
    dyaw = float((pb[2] - nom[2] + np.pi) % (2 * np.pi) - np.pi)
    e_m, e_deg = float(np.linalg.norm(pb[:2] - nom[:2])), float(np.degrees(dyaw))
    if e_m > MON.ROUTE_TOL_M or abs(e_deg) > MON.ROUTE_TOL_DEG:
        ep.base_flag("route_unfinished", ep.k * DR.DT, err_mm=1e3 * e_m, yaw_err_deg=e_deg, tol=f"{1e3 * MON.ROUTE_TOL_M:.0f} mm / {MON.ROUTE_TOL_DEG} deg ASSUMED")
    out.update(end=pb.tolist(), err_mm=1e3 * e_m, err_deg=e_deg)
    env.base_log.append(out)
    ep.base_log = getattr(ep, "base_log", []) + [out]
    return out, None


def _run_phases(env, phases, monitors, rec):
    """as atoms._run_phases (carry-style phases with monitor sampling) plus ("base", ref), ("jhome", {side: q}, T); retreat is outward in the ROBOT frame."""
    ep = env.ep
    for ph in phases:
        t = ep.k * CR.DT
        if ph[0] == "weld":
            ep.run_phases([ph])
            continue
        if ph[0] == "base":
            out, hit = _run_base(env, ph[1], monitors)
            rec["base"] = out
            if hit is not None:
                return hit
            continue
        if ph[0] == "move":
            dur = ph[2]
            env.ik_miss = max(env.ik_miss, max(ep._add_path(s, t, t + dur, np.asarray(ph[1], float)) for s in SIDES))
        elif ph[0] == "move2":
            dur = ph[2]
            env.ik_miss = max(env.ik_miss, max(ep._add_path(s, t, t + dur, np.asarray(ph[1][s], float)) for s in SIDES))
        elif ph[0] == "home":
            dur = ph[2]
            errs = []
            for s in SIDES:
                ep._trim(s, t)
                errs.append(ep._add_path(s, t, t + dur, to=ph[1][s]))
            env.ik_miss = max(env.ik_miss, max(errs))
        elif ph[0] == "jhome":
            dur = ph[2]
            for s in SIDES:
                ep._add_joint_path(s, t, t + dur, [ep._cmd(s, t).copy(), np.asarray(ph[1][s], float).copy()])
        elif ph[0] == "retreat":
            dur = ph[2]
            for s in SIDES:
                ep._add_path(s, t, t + dur, ep.body_delta(np.array([0, ph[1] if s == "left" else -ph[1], 0])))
        else:                                              # hold
            dur = ph[1]
        for _ in range(int(round(dur / CR.DT))):
            hit = A._step(env, monitors)
            if hit is not None:
                ep.run_phases([("stop", A.STOP_S)])
                env.update_obj_info()
                return hit
    return None


def drive(left_arm_action=None, right_arm_action=None, monitor_sequences=None, env=None, return_result=False, **kwargs):
    """same signature / return as atoms.drive; THEIR agent_graph calls it (run_world rebinds agent_graph.drive to this)."""
    assert isinstance(env, BehindEnv), "behind.drive needs a BehindEnv"
    ep = env.ep
    rec = dict(i=len(env.log), edge_id=env.cur_edge[0], is_recovery=env.cur_edge[1], left=A._fn_of(left_arm_action), right=A._fn_of(right_arm_action),
               kw_left=getattr(left_arm_action, "keywords", None), kw_right=getattr(right_arm_action, "keywords", None),
               n_monitor_seqs=len(monitor_sequences or []), status=None, t0=round(ep.k * CR.DT, 3))
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
        hit = _run_phases(env, phases, monitor_sequences, rec)
        rec["ik_miss_m"] = round(float(env.ik_miss), 4)
    except A.PlanRejected as r:
        rec.update(status=r.reason, detail=r.detail)
        raise
    rec.update(status="aborted_by_monitor" if hit else "executed", monitor=hit, t1=round(ep.k * CR.DT, 3), wall_s=round(time.time() - t_wall, 2),
               track={k: (round(float(v), 4) if k != "viol" else v) for k, v in ep.track.items() if k in
                      ("viol", "tau_ratio", "n_sat", "obj_tilt_max", "cup_tilt_max", "spill_ratio")})
    if not hit:
        env.update_obj_info()
    res = dict(actions=[], monitor_index=hit["monitor_index"] if hit else None, monitor_name=hit["monitor_name"] if hit else None,
               step_index=hit["step_index"] if hit else None)
    return res if return_result else res["actions"]


ATOMS_MODULE.drive = drive

BASE_FLAGS = ("wheel_liftoff", "base_world_contact", "load_world_contact", "route_unfinished")      # nero/base/monitors.py + drive.py


def _world(task, i):
    from world_model.v2.nero.base import smoke as SM
    return SM.behind_world(task, i)                 # the ONE world list of both methods (nero/base/smoke.py)


def run_world(args):
    """args: dict(plan, task, i, level, mon_period, edge_settle, world=(seed, w) | None, m_hat_kg).  -> record dict (never raises).  World = probe._seed('e2e', base task, i) over the prior box
    (as the other tasks' same-worlds list; the box_behind world has shelf=False); feedforward mass = prior mid-contents (the planner has no estimate); strict_claw=True (claw overload in U)."""
    from . import shim
    shim.install(scratch_dir=pathlib.Path(args["plan"]).parent / "_unused_db")           # registers atoms.ATOMS_MODULE: agent_graph binds atoms.drive at its first import; run_world rebinds it for the run
    t0 = time.time()
    task, i = args["task"], args["i"]
    install(args.get("level"))
    plan = json.loads(pathlib.Path(args["plan"]).read_text())
    rec = dict(plan_id=plan["plan_id"], task=task, i=i, seed=None, level=LEVEL, outcome=None, reason=None, proposer=plan.get("proposer"), plan_backend=plan.get("backend"),
               plan_models=plan.get("models"), method="theirs", monitor_frame=MONITOR_FRAME)
    AG = None
    try:
        seed, w = args["world"] if args.get("world") is not None else _world(task, i)
        rec["seed"] = seed
        lo, hi = TK.prior_box(task)
        m_hat = args.get("m_hat_kg", 0.5 * (lo + hi))
        ep = CR.CarryEpisode(w, a_plan=E.A_NOM, m_hat=m_hat, planner="scripted", base="swerve", goal_support="counter2", strict_claw=True)
        ep.run_phases([("hold", 0.4)])
        rec["pre_hold_viol"] = ep.track["viol"]
        rec["physics"] = ep.physics_flags()
        ep.reset_track()
        env = BehindEnv(ep, level=LEVEL, mon_period=args.get("mon_period", 0.1), edge_settle=args.get("edge_settle", 0.3))
        rec["theta"] = CR._theta(w)
        rec["truth_total_kg"] = round(float(SC.object_truth(ep.m, ep.d)["total_mass"]), 3)
        from embodichain.agents.agentchord import graph_spec as GS
        from embodichain.agents.agentchord import monitor_functions as MF
        from embodichain.agents.agentchord import agent_graph as AG
        from embodichain.agents.agentchord.agent_graph import AgentTaskGraph
        from .execute import _wrap_edges
        bundle = GS.load_agent_graph_bundle(plan["graph"])
        graph = GS.compile_agent_graph_spec(bundle["task_graph"], bundle.get("recovery_graph"), env=env, graph_cls=AgentTaskGraph, action_module=ATOMS_MODULE, monitor_module=MF)
        _wrap_edges(graph, env)
        drive0 = AG.drive
        AG.drive = drive                                        # their runtime does `from ...atom_actions import drive` once per process: rebind for this run
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
        finally:
            AG.drive = drive0
        okR, whyR = ep.in_R()
        rec.update(edges=env.log, monitor_errors=env.monitor_errors, end_in_R=okR, R_fail=whyR,
                   track={k: (v if k == "viol" else round(float(v), 4)) for k, v in ep.track.items()}, sim_s=round(ep.k * CR.DT, 2))
        ex = [e for e in env.log if e["status"] in ("executed", "aborted_by_monitor")]
        rec.update(n_edges_executed=len(ex), n_rejected=sum(1 for e in env.log if str(e["status"]).startswith(("rejected", "unmappable"))),
                   rejections=[dict(edge=e["edge_id"], status=e["status"], detail=e.get("detail")) for e in env.log if str(e["status"]).startswith(("rejected", "unmappable"))],
                   monitors_fired=[dict(edge=e["edge_id"], **e["monitor"]) for e in env.log if e.get("monitor")],
                   n_recoveries_run=sum(1 for e in ex if e["is_recovery"]))
        done = rec["outcome"] == "completed"
        from world_model.v2.adapters import outcome_carry as OCC
        viol, pre = ep.track["viol"], rec["pre_hold_viol"]
        rec.update(OCC.measure(ep, task, done, viol is not None or pre is not None, pre, refused=rec["outcome"] == "plan_rejected", reason=str(rec.get("reason") or ""),
                               end_in_R=(okR, whyR)))
        rec["any_U"] = viol is not None or pre is not None
        fl = set(ep.bmon.flagged)
        rec["base_log"] = env.base_log
        rec["base"] = dict(
            n_base_edges=len(env.base_log), n_aborted=sum(1 for b in env.base_log if b.get("aborted")),
            route_end_err_mm=[b.get("err_mm") for b in env.base_log], route_end_err_deg=[b.get("err_deg") for b in env.base_log],
            U_flags={k: (k in fl) for k in BASE_FLAGS}, claw_overload=bool(viol and str(viol.get("kind", "")).startswith("claw")),
            first_viol=viol, base_end_pose=[float(v) for v in ep.sb.pose()])
    except Exception as e:
        import traceback
        rec.update(outcome="error", reason=repr(e), tb=traceback.format_exc()[-1500:])
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def same_world_trial(task, i, plan_dir, level=None, mon_period=0.1, edge_settle=0.3):
    """AgentChord's cached graph on world i of a table-behind task (run_world, their pipeline unchanged) on the SAME world as ours (smoke.behind_world).  Called by
    execute.same_world_trial for *_behind tasks (nero/contents/same_worlds.py, which applies _init_worker first).  U includes the per-claw payload row (run_world: strict_claw=True).
    Record fields as execute.same_world_trial: method, seed_e2e, first_U {seg, kind} | None (pre-hold phase, else the first edge whose track has a viol, else the run track),
    plus base_level (= the cap `level`; default LEVEL L3, ASSUMED)."""
    pj = pathlib.Path(plan_dir)
    pj = pj if pj.suffix == ".json" else pj / "plan.json"
    seed, w = _world(task, i)
    lo, hi = TK.prior_box(task)
    r = run_world(dict(plan=str(pj), task=task, i=i, level=level or LEVEL, world=(seed, w), m_hat_kg=0.5 * (lo + hi), mon_period=mon_period, edge_settle=edge_settle))
    fu = None
    if r.get("any_U"):
        if r.get("pre_hold_viol"):
            fu = dict(seg="pre_hold", kind=r["pre_hold_viol"].get("kind"))
        else:
            e = next((e for e in r.get("edges", []) if (e.get("track") or {}).get("viol")), None)
            v = (e["track"]["viol"] if e else None) or (r.get("track") or {}).get("viol") or {}
            fu = dict(seg=e["edge_id"] if e else None, kind=v.get("kind"))
    pr = r.get("proposer") or {}
    r.update(method="theirs", seed_e2e=seed, first_U=fu, plan_model=pr.get("model"), plan_reasoning=pr.get("reasoning"), base_level=level or LEVEL)
    return r
