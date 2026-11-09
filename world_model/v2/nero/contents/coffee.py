"""nero/contents/coffee.py -- AgentChord's real-robot task "setup coffee tray" (paper Table II, 60 %) ported into the NERO contents sim.  SIMULATION ONLY.

Steps (paper appendix): the left arm grasps the coffee capsule and places it into the tray; both arms grasp the tray rims; the tray is placed
0.10 m "forward".  Here "forward" = 0.10 m TOWARD THE ROBOT (x decreases), as the existing tray task (carry.PLANS; +x 10 cm is out of the right arm's reach).
Everything is a NEW module; shared modules are not edited: `install()` patches (idempotent, per process) the task tables of carry_cert / probe /
objects / scenes for the task name "tray0" (= tray with no mugs; n_cups("tray0") = 0, so every `startswith("tray")` code path works):

  CoffeeWorld(TrayWorld)   tray 0.22 x 0.33 m, 3 cm rims, 0.30 kg (objects.TRAY) + capsule (r 19 mm, h 28 mm, 12 g, free body on a pad) +
                           (variant "unknown") a HIDDEN load: a rigid body welded into the tray (child body "contents", contype 0 = mass only, like the
                           mug fill), mass uniform 0..3.0 kg, COM xy uniform inside the inner footprint (ASSUMED prior, proposed to the user 11-09, not yet ruled).
  VARIANT                  env BIMANUAL_COFFEE_VARIANT = "known" (V1: no hidden load, prior point (0, 0)) | "unknown" (V2, prior (0, 3.0) kg)
  capsule skill            single-arm (left) pick-and-place under a rigid weld (ASSUMED, as every grasp): retreat from the tab, over the pad, approach, weld,
                           lift 0.08 m, over the tray (offset [0, 0.06] from the tray reference), lower, release, retreat, back to the tab pose, re-weld the
                           tray.  Gripper<->capsule contacts are off for the whole phase (weldcontact rule; jaws are pose-only under the weld).
  goal sets                stack/tasks.py: "tray0" (strict: coffee_tray_check numbers on our tray) and "tray0_general" (AgentChord's tray rule + capsule inside);
                           both evaluated by stack.goal.check_goal, outcome by stack.outcome.classify (adapters/outcome_carry.measure).  Prior / empty mass: TaskDef tray0.

Labels (AGENTS.md): positions below are ASSUMED (list in ASSUMPTIONS); masses / sizes MODEL from agentchord_probe/coffee_tray/geometry.json.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import config as C
from . import carry as CR
from . import carry_cert as CC
from . import objects as OB
from . import scenes as SC
from world_model.v2.stack import tasks as TK

TASK = "tray0"
VARIANT = os.environ.get("BIMANUAL_COFFEE_VARIANT", "known")
VARIANTS = ("known", "unknown")
LOAD_PRIOR = TK.prior_box("tray0", "unknown")   # ASSUMED (proposed to the user 11-09, not yet ruled): hidden load mass [kg], variant "unknown"; the ONE copy is stack/tasks.py (tray0)
CAP = dict(r=0.019, h=0.028, mass=0.012)       # MODEL: agentchord_probe/coffee_tray/geometry.json
CAP_MU = 0.5                                   # ASSUMED friction of the capsule
CAPSULE_XY = (0.46, 0.0)                       # ASSUMED world xy of the capsule on its pad (left arm reach: tested, IK margin > 0.1 rad)
PAD_HALF = (0.03, 0.03)                        # ASSUMED fixed pad (top = counter top) the capsule stands on
CAP_IN_TRAY = (0.0, 0.06)                      # AgentChord prompt: capsule offset relative to the tray position (x, y)
RELEASE_DROP = 0.04                            # ASSUMED m: height of the capsule's underside above the tray base at release (the left palm clears the 3 cm rim; the capsule drops this far)
TCP_ABOVE_CAP = 0.035                          # ASSUMED m: the left TCP holds the capsule this far above its centre (the lower jaw clears the pad)
LIFT_CAP = 0.08                                # AgentChord prompt: lift the capsule by 0.08 m
RETREAT_Y = 0.05                               # m, claw retreat along its approach axis (left claw: +y), = carry.RETREAT scale
TRANSIT_UP = 0.075                             # ASSUMED m: height above the tab pose of the left claw while it passes over the tab on the way back
OVER_Z = 0.10                                  # ASSUMED m, clearance height above the tab pose for the transit over the left tab
FORWARD = (-0.10, 0.0)                         # tray goal offset (paper "forward 0.10 m", mirrored toward the robot, see module doc)
CAP_BIT = 128                                  # collision bit used for the capsule (weldcontact uses 2; arm bits 4..64)
INNER_X = OB.TRAY["lx"] / 2 - 0.008            # coffee_tray/env.py INNER_X (rim_t 0.008)
INNER_Y = OB.TRAY["ly"] / 2 - 0.008
LOAD_HALF = (0.03, 0.03, 0.02)                 # ASSUMED hidden-load block half size (rigid, welded into the tray)
LOAD_MARGIN = 0.03                             # ASSUMED m: the load COM xy stays this far inside the inner footprint
EMPTY_KG = OB.TRAY["mass"] + CAP["mass"]       # tray + capsule: the known mass of everything but the hidden load

ASSUMPTIONS = dict(
    capsule_xy=f"{CAPSULE_XY} ASSUMED (on a fixed pad {PAD_HALF} half size, top = counter top)", capsule_target="tray reference + (0, 0.06) (AgentChord prompt offset)",
    tcp_above_cap=f"{TCP_ABOVE_CAP} m ASSUMED", forward="0.10 m toward the robot (x -0.10): +x is outside the right arm's reach (carry.py note); paper says 'forward'",
    hidden_load=f"variant unknown: mass uniform {LOAD_PRIOR} kg, COM xy uniform in the inner footprint minus {LOAD_MARGIN} m, COM z fixed 0.028 m above the tray bottom, "
                "rigid body welded in the tray, ASSUMED prior (proposed to the user 11-09, not yet ruled)",
    grasp="rigid weld for every grasp incl. the capsule (ASSUMED, user 11-01); gripper<->capsule contacts off during the capsule phase",
    capsule_mu=f"{CAP_MU} ASSUMED", success="coffee_tray/env.py thresholds (ASSUMED there): capsule inside inner footprint, |z rel tray| <= 0.04, tray xy <= 0.03 m of start + goal offset, tilt < 10 deg")


@dataclass
class CoffeeWorld(SC.TrayWorld):
    n_cups: int = 0
    cup_xy: list | None = None
    fills_kg: list | None = None
    capsule_xy: tuple = CAPSULE_XY
    load_kg: float = 0.0
    load_xy: tuple = (0.0, 0.0)         # hidden-load COM xy in the tray frame (obj_ref xy)

    def __post_init__(self):
        if self.cup_xy is None:
            self.cup_xy = []
        if self.fills_kg is None:
            self.fills_kg = []


def variant():
    v = os.environ.get("BIMANUAL_COFFEE_VARIANT", VARIANT)
    if v not in VARIANTS:
        raise ValueError(f"BIMANUAL_COFFEE_VARIANT must be one of {VARIANTS}, got {v!r}")
    return v


def prior():
    return TK.prior_box(TASK, variant())


def sample_world(rng):
    """one world of the variant's prior (V1: the known world; V2: hidden load mass / COM uniform)."""
    if variant() == "known":
        return CoffeeWorld()
    hx, hy = INNER_X - LOAD_MARGIN, INNER_Y - LOAD_MARGIN
    m = float(rng.uniform(*LOAD_PRIOR))
    xy = (float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy)))
    return CoffeeWorld(load_kg=m, load_xy=xy)


def seed_for(i):
    """the world seed of index i: probe._seed("e2e", "tray0", i) (the one both methods use, same_worlds style)."""
    from . import probe as P
    install()
    return P._seed("e2e", TASK, i)


def world_for(i):
    return sample_world(np.random.default_rng(seed_for(i)))


# ------------------------------------------------------------------------------------------------------------------ scene
def _extend_spec(spec, w):
    W = spec.worldbody
    top = w.counter_top
    cx, cy = w.capsule_xy
    W.add_geom(name="capsule_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=[PAD_HALF[0], PAD_HALF[1], (top - w.table_top_z) / 2],
               pos=[cx, cy, (top + w.table_top_z) / 2], contype=1, conaffinity=1, friction=[w.counter_mu, *SC.FRIC], rgba=[0.5, 0.5, 0.5, 1])
    cb = W.add_body(name="capsule", pos=[cx, cy, top + CAP["h"] / 2 + 1e-4])
    cb.add_freejoint(name="capsule_free")
    cb.add_geom(name="capsule_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CAP["r"], CAP["h"] / 2, 0], mass=CAP["mass"],
                friction=[CAP_MU, *SC.FRIC], contype=1, conaffinity=1, rgba=[0.9, 0.2, 0.2, 1])
    if w.load_kg > 0:
        ob = spec.body("obj")
        lb = ob.add_body(name="contents", pos=[w.load_xy[0], w.load_xy[1], w.base_t + LOAD_HALF[2]])
        lb.add_geom(name="contents_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(LOAD_HALF), mass=float(w.load_kg), contype=0, conaffinity=0,
                    rgba=[0.3, 0.3, 0.8, 0.0])        # invisible: the hidden load is not seen by the camera
    spec.add_equality(name="cweld_left", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="left_arm_gripper_base", name2="capsule",
                      active=False)


def coffee_spec(c_lo, c_hi):
    """ObjectSpec of the tray + capsule with a hidden load of mass c_lo..c_hi kg, COM anywhere in the inner footprint (MODEL, superset of the sampled set)."""
    P = OB.prism(np.array([[-INNER_X, -INNER_Y], [INNER_X, -INNER_Y], [INNER_X, INNER_Y], [-INNER_X, INNER_Y]]), 0.008 + LOAD_HALF[2], 0.008 + LOAD_HALF[2])
    c = OB.Contents(c_lo, c_hi, P, labels=dict(m=f"ASSUMED hidden load {c_lo}..{c_hi} kg (proposed to the user 11-09, not yet ruled)", P="MODEL: COM anywhere inside the tray's inner footprint at 0.028 m"))
    return OB.ObjectSpec("coffee_tray_capsule", EMPTY_KG, np.array([0.0, 0.0, 0.004]), c, [],
                         sim_box=dict(note="coffee.CoffeeWorld"), plausible=dict(note=f"tray + capsule + hidden load {c_lo}..{c_hi} kg"))


_INSTALLED = False


def install():
    """idempotent per process: register the task "tray0" in the shared task tables (no file of the shared modules is edited)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from . import probe as P
    lay0 = SC.TrayWorld.layout

    def layout(self):
        return ([], []) if self.n_cups == 0 else lay0(self)
    SC.TrayWorld.layout = layout
    bs0 = SC.build_spec

    def build_spec(world):
        spec = bs0(world)
        if isinstance(world, CoffeeWorld):
            _extend_spec(spec, world)
        return spec
    SC.build_spec = build_spec
    ct0 = OB.coffee_tray
    OB.coffee_tray = lambda n_cups=OB.N_CUPS: coffee_spec(*prior()) if n_cups == 0 else ct0(n_cups)
    CC.TASK_ID[TASK] = 10
    wd0, sw0 = CC.world_default, CC.sample_world
    CC.world_default = lambda task: CoffeeWorld() if task == TASK else wd0(task)
    CC.sample_world = lambda task, cell, rng: sample_world(rng) if task == TASK else sw0(task, cell, rng)
    pc0, em0, is0 = P.prior_contents, P.empty_mass, P.interval_spec
    P.prior_contents = lambda task: prior() if task == TASK else pc0(task)
    P.empty_mass = lambda task: EMPTY_KG if task == TASK else em0(task)
    P.interval_spec = lambda task, c_lo, c_hi: coffee_spec(c_lo, c_hi) if task == TASK else is0(task, c_lo, c_hi)
    base = CR.CarryEpisode

    class _Coffee(base):
        def __init__(self, world, *a, **k):
            super().__init__(world, *a, **k)
            self.coffee = isinstance(world, CoffeeWorld)
            if self.coffee:
                _attach(self)
    CR.CarryEpisode = _Coffee


def _attach(ep):
    """capsule bookkeeping + the capsule collision bit (same scheme as weldcontact, bit CAP_BIT)."""
    m, d = ep.m, ep.d
    ep.cap = m.body("capsule").id
    ep.cweld_id = m.equality("cweld_left").id
    ep.carried = set(ep.carried) | {ep.cap}
    for g in range(m.ngeom):
        if (int(m.geom_contype[g]) | int(m.geom_conaffinity[g])) & CAP_BIT:
            raise RuntimeError("coffee: collision bit 128 already used")
    cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == ep.cap]
    for g in cg:
        m.geom_contype[g] = CAP_BIT
        m.geom_conaffinity[g] = int(m.geom_conaffinity[g]) | CAP_BIT | 2        # 2 = weldcontact's object bit: the capsule still hits the tray
    ep.cap_robot = ep.wc.robot_geoms
    for g in ep.cap_robot:
        m.geom_conaffinity[g] = int(m.geom_conaffinity[g]) | CAP_BIT
    ep.cap_grip = list(ep.wc.grip_geoms["left"])
    ep.cap_hit = False
    cap_contacts(ep, True)                      # ASSUMED: the left jaws never collide with the capsule (pose-only under the weld, weldcontact rule)


def cap_contacts(ep, left_off):
    """left gripper <-> capsule contacts off (True) / on (False)."""
    for g in ep.cap_grip:
        ca = int(ep.m.geom_conaffinity[g])
        ep.m.geom_conaffinity[g] = (ca & ~CAP_BIT) if left_off else (ca | CAP_BIT)


def cweld(ep, on):
    i, m, d = ep.cweld_id, ep.m, ep.d
    if on:
        b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
        R1 = d.xmat[b1].reshape(3, 3)
        qi, q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qi, d.xquat[b1])
        mujoco.mju_mulQuat(q, qi, d.xquat[b2])
        m.eq_data[i, 0:3] = 0.0
        m.eq_data[i, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
        m.eq_data[i, 6:10] = q
        m.eq_data[i, 10] = 1.0
    d.eq_active[i] = 1 if on else 0


# ------------------------------------------------------------------------------------------------------------------ geometry
def tray_ref(ep):
    return ep.d.site_xpos[ep.m.site("obj_ref").id].copy()


def cap_pos(ep):
    return ep.d.xpos[ep.cap].copy()


def capsule_targets(ep):
    """(tcp above the capsule on its pad, tcp when the capsule is over the tray, tab TCP) from the CURRENT commanded state; world metres."""
    w = ep.world
    top = w.counter_top
    p_pad = np.array([w.capsule_xy[0], w.capsule_xy[1], top + CAP["h"] / 2 + TCP_ABOVE_CAP])
    ref = tray_ref(ep)
    cap_z = ref[2] + w.base_t + CAP["h"] / 2 + RELEASE_DROP
    p_tray = np.array([ref[0] + CAP_IN_TRAY[0], ref[1] + CAP_IN_TRAY[1], cap_z + TCP_ABOVE_CAP])
    return p_pad, p_tray


# ------------------------------------------------------------------------------------------------------------------ phase interpreter (left arm)
class Abort(Exception):
    pass


def run_left(ep, phases, on_step=None):
    """phases: ("L", delta[3], T) left claw TCP translation | ("Lto", tcp[3], T) | ("Lq", q7, T) joint-space | ("hold", T) | ("cweld", on) | ("cbits", off)
    | ("tweld", on) tray welds of both claws.  on_step(ep) after every control step; a truthy return stops the phase list (returns it)."""
    for ph in phases:
        t = ep.k * CR.DT
        k = ph[0]
        if k == "cweld":
            cweld(ep, ph[1])
            continue
        if k == "cbits":                       # no-op: left gripper <-> capsule contacts are off for the whole episode (ASSUMED, see _attach)
            continue
        if k == "tweld":
            for s in CR.SIDES:
                ep.weld(s, bool(ph[1]))
            continue
        if k == "L":
            ep._add_path("left", t, t + ph[2], np.asarray(ph[1], float))
            dur = ph[2]
        elif k == "Lto":
            ep._add_path("left", t, t + ph[2], to=np.asarray(ph[1], float))
            dur = ph[2]
        elif k == "Lq":
            q = ep._cmd("left", t)
            ep._add_joint_path("left", t, t + ph[2], [q.copy(), np.asarray(ph[1], float).copy()])
            dur = ph[2]
        else:
            dur = ph[1]
        for _ in range(int(round(dur / CR.DT))):
            ep._step()
            if on_step is not None:
                hit = on_step(ep)
                if hit:
                    return hit
    return None


def cmd_tcp(ep, side="left"):
    from ..democ import kin as K
    cache = ep.__dict__.setdefault("_pik", {})
    if side not in cache:
        cache[side] = K.PoseIK(ep.m, side)
    t = ep.k * CR.DT
    return np.asarray(cache[side].ik.pose(ep._cmd(side, t))[0], float).copy()


def capsule_plan(ep):
    """the capsule phase as a list of NAMED atomic actions, each a phase list: [(name, held, phases)].  Positions resolved from the current state.
    The left claw starts at the tab pose (tray welds on) and ends there with the tray welded again."""
    a = ep.a_plan
    mt = lambda dist: CR.move_time(dist, a)
    p_pad, p_tray = capsule_targets(ep)
    T0 = cmd_tcp(ep)
    q0 = ep.q0["left"].copy()
    pre = lambda p: np.array([p[0], p[1] + RETREAT_Y, p[2]])
    A = []
    A.append(("release_tray_left", None, [("tweld", False), ("cbits", True), ("hold", 0.2), ("L", (0, RETREAT_Y, 0), mt(RETREAT_Y)), ("hold", 0.2)]))
    A.append(("move_to_capsule", None, [("Lto", pre(p_pad), mt(float(np.linalg.norm(pre(p_pad) - cmd_tcp(ep) - np.array([0, RETREAT_Y, 0]))))),
                                         ("Lto", p_pad, mt(RETREAT_Y)), ("hold", 0.3)]))
    A.append(("grasp_capsule", None, [("cweld", True), ("hold", 0.2)]))
    A.append(("lift_capsule", "capsule", [("L", (0, 0, LIFT_CAP), mt(LIFT_CAP)), ("hold", 0.2)]))
    top_p = np.array([p_tray[0], p_tray[1], p_pad[2] + LIFT_CAP])
    A.append(("carry_capsule", "capsule", [("Lto", top_p, mt(np.linalg.norm(top_p - (p_pad + [0, 0, LIFT_CAP])))), ("hold", 0.2)]))
    A.append(("lower_capsule", "capsule", [("Lto", p_tray, mt(top_p[2] - p_tray[2])), ("hold", 0.3)]))
    A.append(("release_capsule", None, [("cweld", False), ("hold", 0.3), ("L", (0, RETREAT_Y, 0), mt(RETREAT_Y)), ("cbits", False), ("hold", 0.2)]))
    zt = T0[2] + TRANSIT_UP
    w1 = np.array([p_tray[0], p_tray[1] + RETREAT_Y, zt])
    w2 = np.array([T0[0], T0[1] + RETREAT_Y, zt])
    A.append(("return_to_tray", None, [("Lto", w1, mt(zt - p_tray[2])), ("Lto", w2, mt(np.linalg.norm(w2 - w1))), ("Lto", pre(T0), mt(TRANSIT_UP)),
                                       ("Lto", T0, mt(RETREAT_Y)), ("Lq", q0, 0.6), ("hold", 0.3), ("tweld", True), ("hold", 0.2)]))
    return A


# ------------------------------------------------------------------------------------------------------------------ success
GOAL_SETS = dict(strict="tray0", general="tray0_general")      # named goal sets = TaskDefs of stack/tasks.py (strict = coffee_tray_check numbers; general = AgentChord's tray rule + capsule inside)


def general_result(ep, finished, unsafe=False):
    """the outcome record of the GENERAL goal set (second named set, evaluated only): the same stack.goal / stack.outcome path as the primary rule (adapters.outcome_carry.measure)."""
    from world_model.v2.adapters.outcome_carry import measure
    return measure(ep, GOAL_SETS["general"], finished, unsafe)


# ------------------------------------------------------------------------------------------------------------------ failure injection (AgentChord Sec. IV-B)
class Injector:
    """with probability p an object held by the end-effector is dropped at each atomic action (AgentChord paper protocol): the weld of the holding claw(s) goes off at a
    uniformly random control step inside the action; the object falls.  `held` = names of the objects welded now.  Every draw is recorded in `log`."""

    def __init__(self, p, seed):
        self.p, self.rng, self.log = float(p), np.random.default_rng(seed), []
        self._pending = None
        self.on_drop = None                                  # optional callback(held) after a drop (THEIRS' env updates the gripper reading)

    def begin(self, ep, action, held, dur):
        """call at the start of an atomic action.  held: "capsule" | "tray" | None.  -> None; schedules the drop."""
        rec = dict(action=action, held=held, t0=round(ep.k * CR.DT, 3), dropped=False)
        self.log.append(rec)
        if held is None or self.p <= 0:
            return
        if self.rng.random() < self.p:
            rec["dropped"] = True
            rec["t_drop"] = round(rec["t0"] + float(self.rng.uniform(0.0, max(dur, 0.05))), 3)
            self._pending = (rec["t_drop"], held)

    def step(self, ep):
        """per control step (a step hook): execute a scheduled drop."""
        if self._pending is None:
            return False
        t_drop, held = self._pending
        if ep.k * CR.DT < t_drop:
            return False
        self._pending = None
        if held == "capsule":
            cweld(ep, False)
        else:
            for s in CR.SIDES:
                ep.weld(s, False)
        self.log[-1]["applied_at"] = round(ep.k * CR.DT, 3)
        if self.on_drop is not None:
            self.on_drop(held)
        return True

    @property
    def n_dropped(self):
        return sum(1 for r in self.log if r["dropped"])


def duration(phases):
    return float(sum((ph[2] if ph[0] in ("L", "Lto", "Lq") else (ph[1] if ph[0] == "hold" else 0.0)) for ph in phases))


def run_capsule_phase(ep, inj=None):
    """ours: the capsule skill (place_capsule) = capsule_plan's actions, optionally with drop injection.  -> list of per-action records."""
    from ..contents import coffee as _self                       # noqa: F401  (same module)
    mass0 = ep.ctl.mass
    ep.ctl.mass = 0.0
    out = []
    if inj is not None:
        ep.step_hooks.append(inj.step)
    for name, held, ph in capsule_plan(ep):
        if inj is not None:
            inj.begin(ep, name, held, duration(ph))
        run_left(ep, ph)
        out.append(dict(action=name, t=round(ep.k * CR.DT, 2), viol=ep.track["viol"], capsule=[round(float(x), 4) for x in cap_pos(ep)]))
    if inj is not None:
        ep.step_hooks.remove(inj.step)
    ep.ctl.mass = mass0
    return out


# ------------------------------------------------------------------------------------------------------------------ ours: episode + measure patches
JOB = dict(p=0.0, inj_seed=0, tray_start=None, cap_log=None, inj=None)     # per-process job state (set by coffee_run before each ours job)


_OURS_INSTALLED = False


def install_ours():
    """patch probe._episode (build the episode, run the capsule phase, install the tray-segment injector).  The success / outcome path is NOT patched: the task "tray0" is a
    TaskDef (stack/tasks.py) and probe._e2e_job writes its record through adapters.outcome_carry.measure like every carry task."""
    global _OURS_INSTALLED
    install()
    if _OURS_INSTALLED:                    # idempotent per process: a second wrap would run the capsule phase twice
        return
    _OURS_INSTALLED = True
    from . import probe as P
    ep0 = P._episode

    def episode(world, wp=None):
        ep = ep0(world, wp)
        if not isinstance(world, CoffeeWorld):
            return ep
        JOB["tray_start"] = tray_ref(ep).copy()
        JOB["ep"] = ep                                           # the final state is read by the driver (general goal set, record)
        inj = Injector(JOB["p"], JOB["inj_seed"])
        JOB["inj"] = inj
        ep.run_phases([("hold", 0.3)])
        JOB["cap_log"] = run_capsule_phase(ep, inj)
        ep.reset_track()                                         # U of the capsule phase is reported separately (cap_log); the tray carry starts a fresh track
        JOB["cap_viol"] = [a["viol"] for a in JOB["cap_log"] if a["viol"]][:1] or None
        orig = ep.run_phases

        def run_phases(phases, branch=None):                      # tray atomic actions = nominal motion segments (branch given); release has no held object
            if branch is not None and any(ph[0] in ("move", "track", "to") for ph in phases):
                held = "tray" if all(ep.d.eq_active[ep.weld_id[s]] for s in CR.SIDES) else None
                dur = 0.0
                for s in CR.SIDES:                                # duration of the planned motion: filled in after the first phase is queued; approximate by the plan
                    pass
                inj.begin(ep, f"tray_seg_{len(inj.log)}", held, _seg_dur(ep, phases))
                ep.step_hooks.append(inj.step)
                try:
                    return orig(phases, branch)
                finally:
                    ep.step_hooks.remove(inj.step)
            return orig(phases, branch)
        ep.run_phases = run_phases
        return ep
    P._episode = episode



def record_extras():
    """what the episode patch above recorded for the current ours job (capsule phase log, drop injections, capsule-phase U): JSON-able."""
    return dict(cap_viol=JOB.get("cap_viol"), inj=JOB["inj"].log if JOB.get("inj") else None, cap_log=JOB.get("cap_log"))


def _seg_dur(ep, phases):
    """planned duration of a nominal segment's phases (move / track pieces use the plan; holds add up).  Approximate (only the drop-time draw uses it)."""
    T = 0.0
    for ph in phases:
        if ph[0] == "hold":
            T += ph[1]
        elif ph[0] == "move":
            T += ph[2]
        elif ph[0] == "track":
            try:
                pc = ep.cc.segment(ph[1])
                T += float(pc.T) if hasattr(pc, "T") else 1.5
            except Exception:
                T += 1.5
    return T
