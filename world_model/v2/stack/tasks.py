"""stack/tasks.py -- the ONE table of per-task facts (11-09 plan).  SIMULATION ONLY.

Every other unified module (goal, state, outcome, checks/*) reads a task only through `task(name)`; there is no `if task == ...` outside this file.
PURE DATA: no import of nero / agentchord / house / vlm / flmpc (11-09 dependency rule).  Dimensions, masses and radii are literals whose src points at
the file that defines them; object builders are import STRINGS resolved by the adapter layer.  world_model/v2/adapters/tests/test_tasks_match_nero.py
loads the nero values and asserts this table equals them, so drift is caught there.
Values are today's numbers, moved unchanged; each is a `stack.types.Bound` carrying its tag (MEASURED / MODEL / ASSUMED) and src.
Convention for a Bound used as a limit: inside iff lo <= v <= hi; an upper limit has lo = -inf, a lower limit has hi = +inf.  A `Lim` with
strict=True is inside iff lo < v < hi: it reproduces the OLD code's comparator exactly (e.g. carry tau violates at ratio >= 1 - 1e-6, pan tilt
violates at margin <= 0), so admission / U / R decide the same at equality.

Where the numbers come from (checked by world_model/v2/adapters/tests/test_tasks_match_nero.py and stack/tests/test_unified_core.py):
  carry U / R      nero/contents/carry.py U_TILT_OBJ, U_TILT_CUP, R_TILT_OBJ, R_TILT_CUP, LID_*, R_SPEED, the drop test in CarryEpisode._check
  pan U / R        nero/pan/segments.py U, R
  goal             closed_chain.GOAL_DXY (xy offset), the carry tasks' goal terms = AgentChord ac_success (USER 11-09); pan success(seg); coffee_tray/env.py
  contents prior   was probe.PRIOR = probe_trace.PRIOR = agentchord tasks.PRIOR (deleted 11-09; prior_box() is the one reader)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from world_model.v2.stack.types import Bound

INF = math.inf


@dataclass(frozen=True)
class Rect:
    lx: float
    ly: float


@dataclass(frozen=True)
class Disc:
    r: float


@dataclass(frozen=True)
class Lim(Bound):
    """a Bound with the old code's comparator: strict=True -> inside iff lo < v < hi (the bound itself is OUTSIDE)."""
    strict: bool = False


def ub(x, tag, src, unit="", strict=False):
    """upper limit: inside iff v <= x (v < x when strict)."""
    return Lim(-INF, float(x), tag, src, unit, strict)


def lb(x, tag, src, unit="", strict=False):
    """lower limit: inside iff v >= x (v > x when strict)."""
    return Lim(float(x), INF, tag, src, unit, strict)


@dataclass(frozen=True)
class TaskDef:
    name: str
    kind: str                              # "tray" | "box" | "pot" | "pan" | "coffee_tray"  (replaces plan_key / closed_chain.kind_of)
    backend: str                           # sim backend: "carry" | "pan" | "embodichain"
    world_ref: str = ""                    # "module:Class" of the carry world (lazy import via build_world)
    world_kw: tuple = ()                   # ((field, value), ...) passed to the world class (box_behind: shelf=False)
    n_cups: int = 0                        # tray: number of mugs (visible, ASSUMED)
    spec_ref: str = ""                     # "module:callable" building the ObjectSpec (contents set) -- see checks/admit.spec_for
    start_support: str = ""
    goal_support: str = ""
    goal_dxy: tuple | None = None          # goal xy relative to the start xy [m]
    goal_frame: str = "world"              # "world": goal = world.obj_xy + goal_dxy;  "start": state reader gives xy relative to the start
    goal_terms: tuple = ()                 # ((quantity, Bound), ...) evaluated by goal.check_goal
    shape: Rect | Disc | None = None       # support footprint of the object (contact shape on the goal support); carry tasks only
    geom_tags: tuple = ()                  # ((fact, tag, src), ...) for shape / empty_mass / empty_com (AGENTS.md bounds rule)
    empty_mass: float | None = None        # kg; None = resolved from the scene (pot): see empty_props
    empty_com: tuple | None = None         # object frame [m]
    prior: Bound | None = None             # contents prior (ONE copy); unit in prior.unit
    variant_priors: tuple = ()             # ((variant, Bound), ...): scene variants of the same task with their own prior (coffee tray0: "known" | "unknown"); prior = the default
    rows: tuple = ()                       # admission row names applicable to this task
    U: tuple = ()                          # ((quantity, Bound), ...) evaluated on StepState.worst, in monitor order
    R: tuple = ()                          # ((quantity, Bound), ...) evaluated on StepState.now
    R_v2: tuple = ()                       # pan only: the R_V2 table (no rim / yaw; adds stable_margin_m, reach); R above is R_V1
    r_includes_U: bool = True              # R also fails when the episode entered U (carry.in_R does, pan Episode.in_R does not)
    reader: tuple = ()                     # ((name, Bound), ...) thresholds a state reader needs (adapters read them from here)
    moves: tuple = ()                      # scripted (legacy carry.PLANS) moves: ((segment, (dx, dy, dz)), ...)
    segments: tuple = ()                   # pan: segment names in order
    base: str = ""                         # "" = fixed base (every task before 11-09); "swerve" = the base drives (nero/base: swerve wheels, two-table scene)
    routes: tuple = ()                     # base routes as data: ((name, ((kind, arg), ...)), ...); kind "line" [m] or "turn" [deg]
    obj: str = ""                          # object name in the scene
    note: str = ""
    phenomena: tuple = ()                  # PHI(task): the checker's own list of hidden-contents phenomena that must be measured or on an approved fallback (ids of skills/catalog.PHENOMENA)

    def U_dict(self):
        return dict(self.U)

    def R_dict(self):
        return dict(self.R)

    def reader_dict(self):
        return dict(self.reader)

    def R_table(self, version="v1"):
        if version == "v1":
            return self.R
        if version == "v2" and self.R_v2:
            return self.R_v2
        raise ValueError(f"{self.name}: no R version {version!r}")

    def goal_end_dxy(self):
        """sum of the scripted moves in xy (must equal goal_dxy for the carry tasks)."""
        return (sum(d[0] for _, d in self.moves), sum(d[1] for _, d in self.moves))


# ----------------------------------------------------------------------------------------------------------------------- shared numbers
_AC_SRC = "AgentChord ac_success (execute.py:25,157), adopted USER 11-09 (success = their rule, both methods)"
_ASM = "ASSUMED"
_CARRY = "nero/contents/carry.py"
_PAN = "nero/pan/segments.py"


def _u_carry(cups: bool, lid: bool):
    u = [("tau_ratio", ub(1.0 - 1e-6, "MODEL", f"{_CARRY} _check: tau_sat when ratio >= 1 - 1e-6 (strict); ratio = |tau| / config.tau_limit() = MJCF forcerange (USER 11-05)", "ratio", strict=True)),
         ("obj_tilt_deg", ub(10.0, _ASM, f"{_CARRY} U_TILT_OBJ", "deg")),
         ("drop_m", ub(0.01, _ASM, f"{_CARRY} _check: obj_drop when z < z0 - 0.01 (state_carry reads this bound)", "m"))]
    if cups:
        u += [("cup_tilt_deg", ub(20.0, _ASM, f"{_CARRY} U_TILT_CUP", "deg")),
              ("spill_ratio", ub(1.0, "MODEL", f"{_CARRY} _check: quasi-static spill ratio (D/2 tan(phi)) / freeboard > 1", "ratio")),
              ("cup_off_tray_flag", ub(0.0, "MODEL", f"{_CARRY} _check: cup centre outside the tray rectangle (tray_lx/2, tray_ly/2) or its z below the tray base -5 mm (ASSUMED, carry.py:590) (flag 0/1)", "flag"))]
    if lid:
        u += [("lid_dz_m", ub(0.01, _ASM, f"{_CARRY} LID_OFF_Z", "m")),
              ("lid_tilt_deg", ub(10.0, _ASM, f"{_CARRY} LID_TILT_DEG (= U_TILT_OBJ)", "deg")),
              ("lid_slide_m", ub(0.005, _ASM, f"{_CARRY} LID_SLIDE_M", "m"))]
    u.append(("collision_flag", ub(0.0, "MODEL", f"{_CARRY} _check: robot geom penetration > 1 mm other than the grasp (flag 0/1)", "flag")))
    return tuple(u)


def _r_carry(cups: bool, lid: bool):
    r = [("obj_speed", ub(0.005, _ASM, f"{_CARRY} R_SPEED (linear speed of the object)", "m/s")),
         ("obj_tilt_deg", ub(3.0, _ASM, f"{_CARRY} R_TILT_OBJ", "deg")),
         ("support_contact", lb(1.0, "MODEL", f"{_CARRY} in_R: object touches the goal support, the counter OR (box only, carry.py:628) the shelf (1/0)", "flag"))]
    if cups:
        r.append(("cup_tilt_deg", ub(5.0, _ASM, f"{_CARRY} R_TILT_CUP", "deg")))
    if lid:
        r += [("lid_slide_m", ub(0.005, _ASM, f"{_CARRY} LID_SLIDE_M", "m")), ("lid_dz_m", ub(0.01, _ASM, f"{_CARRY} LID_OFF_Z", "m")),
              ("lid_tilt_deg", ub(10.0, _ASM, f"{_CARRY} LID_TILT_DEG", "deg"))]
    return tuple(r)


_U_PAN = (
    ("obj_tilt_deg", ub(15.0, _ASM, f"{_PAN} U['tilt_deg'] (violated at margin <= 0, so strict)", "deg", strict=True)),
    ("lid_mm", ub(10.0, _ASM, f"{_PAN} U['lid_mm'] (strict)", "mm", strict=True)),
    ("drop_mm", ub(15.0, _ASM, f"{_PAN} U['drop_mm']: pan centre this far below its resting height (strict)", "mm", strict=True)),
    ("joint_margin_rad", lb(0.005, _ASM, f"{_PAN} U['joint_tol']; joint ranges MODEL (MJCF jnt_range) (strict)", "rad", strict=True)),
    ("tau_ratio", ub(1.0 + 1e-6, "MODEL", f"{_PAN} U['effort'] = |tau| / TAU_MAX; _monitor allows ratio <= U['effort'] + 1e-6", "ratio")),
    ("spill_flag", ub(0.0, "MODEL", f"{_PAN} pan_state: puck centre outside (inner radius - puck radius + 5 mm) or > 6 cm from the bottom (both ASSUMED, in pan_state, not re-read here) (flag 0/1)", "flag")),
    ("com_in_m", lb(0.0, "MODEL", f"{_PAN} _monitor 'edge': COM of pan + lid + contents over the stove top (violated at margin <= 0) while not lifted (reader gives +inf when lifted)", "m", strict=True)),
    ("collision_flag", ub(0.0, "MODEL", f"{_PAN} contacts()['coll']: any robot geom touching anything but the pan / lid (flag 0/1)", "flag")),
)
_R_PAN_COMMON = (
    ("obj_tilt_deg", ub(3.0, _ASM, f"{_PAN} R['tilt_deg'] (in_R needs margin > 0: strict)", "deg", strict=True)),
    ("abs(dz_mm)", ub(3.0, _ASM, f"{_PAN} R['dz_mm'] (strict)", "mm", strict=True)),
    ("lid_mm", ub(5.0, _ASM, f"{_PAN} R['lid_mm'] (strict)", "mm", strict=True)),
    ("support_frac", lb(0.9, _ASM, f"{_PAN} R['support_frac']: the stove carries >= this share of the weight (strict)", "frac", strict=True)),
    ("pan_speed", ub(0.005, _ASM, f"{_PAN} R['pan_speed'] (strict)", "m/s", strict=True)),
    ("arm_speed", ub(0.05, _ASM, f"{_PAN} R['arm_speed'] (strict)", "rad/s", strict=True)),
    ("spill_flag", ub(0.0, "MODEL", f"{_PAN} pan_state spill (flag 0/1)", "flag")),
    ("released", lb(1.0, "MODEL", f"{_PAN} in_R: no gripper force on the pan (1/0)", "flag")),
)
_REACH = ("reach", lb(1.0, "MODEL", f"{_PAN} in_R: handle_grasp.find_pregrasp succeeds at the handle (kinematic IK + contact check; 1/0).  v1 fingers FINGER_OPEN, v2 REGRASP_OPEN", "flag"))
_R_PAN = _R_PAN_COMMON + (
    ("abs(yaw_deg)", ub(8.5, _ASM, f"{_PAN} R['yaw_deg'] (R_V1) (strict)", "deg", strict=True)),
    ("rim_in_m", lb(0.0, "MODEL", f"{_PAN} in_R 'rim' (R_V1): whole rim circle over the stove, pan_state rim_in (margin min(1, rim_in/0.01) > 0: strict)", "m", strict=True)),
    _REACH)
_R_PAN_V2 = _R_PAN_COMMON + (
    ("stable_margin_m", lb(0.005, _ASM, f"{_PAN} in_R 'stable' (R_V2): Episode.stable_margin() - STABLE_MARGIN (0.005 ASSUMED = cert/model_check.SUPPORT_MARGIN), strict", "m", strict=True)),
    _REACH)

_XY = ("xy_err", ub(0.05, _ASM, _AC_SRC + ": xy_err <= 0.05 m", "m"))
_Z = ("z_err", ub(0.02, _ASM, _AC_SRC + ": support-relative z_err <= 0.02 m", "m"))
_TILT_G = ("obj_tilt_deg", ub(45.0, _ASM, _AC_SRC + ": object tilt < 45 deg (strict)", "deg", strict=True))
_CUP_TILT_G = ("cup_tilt_deg", ub(45.0, _ASM, _AC_SRC + ": every mug tilt < 45 deg (strict)", "deg", strict=True))
_PRIOR_SRC = "prior box moved here 11-09 from probe.PRIOR = probe_trace.PRIOR = agentchord tasks.PRIOR (the three agreed)"
_PLANS_SRC = "nero/contents/carry.py PLANS / PLANS_EXTRA (ASSUMED, user-confirmed 11-02)"

_TRAY_MOVES = (("lift", (0, 0, 0.05)), ("carry", (-0.10, 0, 0)), ("place", (0, 0, -0.05)))
_BOX_MOVES = (("lift", (0, 0, 0.18)), ("carry", (-0.20, 0, 0)), ("place", (0, 0, -0.03)))
_POT_RADIUS = 0.09 + 0.0025     # nero/pan/scene.py ASSUMED pan_radius 0.09 + pan_wall_t 0.0025.  MODEL: the pot's contact is its bottom disc, radius R + t (pan/scene.py "pan_bottom")

# pan lift / segment goal terms: pan/segments.py Episode.success
_LIFT_M = 0.05                                                         # = pan/handle_grasp.LIFT (test checks)
_PAN_SUCC = f"{_PAN} Episode.success"
_PAN_COMMON = dict(kind="pan", backend="pan", start_support="stove", goal_support="stove", obj="pan", segments=("pinch", "push", "clamp", "lift"),
                   U=_U_PAN, R=_R_PAN, R_v2=_R_PAN_V2, r_includes_U=False,
                   reader=(("lifted_dz_mm", lb(5.0, _ASM, f"{_PAN} _monitor: `lifted = dz_mm > 5.0` switches the edge (COM) monitor off (strict)", "mm", strict=True)),),
                   prior=Bound(0.02, 1.5, _ASM, "pan/theta.py THETA 'mass' (puck, kg)", "kg"))

# coffee tray (AgentChord sim, agentchord_probe/coffee_tray/env.py + geometry.json): inner footprint = lx/2 - rim_t, ly/2 - rim_t
_CT_LX, _CT_LY, _CT_RIM_T = 0.22, 0.34, 0.008                          # geometry.json (their tray; our NERO tray is OB.TRAY, ly 0.33)
_CT_INNER_X, _CT_INNER_Y = _CT_LX / 2 - _CT_RIM_T, _CT_LY / 2 - _CT_RIM_T
_CT_SRC = "agentchord_probe/coffee_tray/env.py coffee_tray_check"
_CT_CAPSULE = (("abs(cap_rel_x)", ub(_CT_INNER_X, _ASM, f"{_CT_SRC}: capsule in the tray's inner footprint (lx/2 - rim_t)", "m")),
               ("abs(cap_rel_y)", ub(_CT_INNER_Y, _ASM, f"{_CT_SRC}: (ly/2 - rim_t)", "m")),
               ("abs(cap_rel_z)", ub(0.04, _ASM, f"{_CT_SRC}: capsule resting on the tray, z within 0.04 m", "m")))
_CT_TILT = ("obj_tilt_deg", ub(10.0, _ASM, f"{_CT_SRC}: tray_level, tilt < 10 deg (strict)", "deg", strict=True))
_CT_COMMON = dict(kind="coffee_tray", backend="embodichain", goal_frame="start", obj="tray")


# PHI per kind (probe-selection build spec 2026-11-09 s.4, rulings 11-09 Q1 = option c): PH4 liquid slosh, PH5 hidden loose item, PH6 visible loose item are OUT OF SCOPE:
# no built skill measures them and no row covers them, so they are kept OUT of PHI and stated as a limit in every result (catalog.OUT_OF_SCOPE).
_PHI = {"box": ("PH1", "PH2", "PH3"), "tray": ("PH1", "PH2", "PH3", "PH7"), "pot": ("PH1", "PH2", "PH3", "PH8")}


def _carry_task(name, kind, world_ref, spec_ref, goal_support, moves, shape, rows, n_cups=0, empty_mass=None, empty_com=None, prior=None, cups=False, lid=False,
                obj="obj", geom_tags=()):
    gx, gy = {"tray": (-0.10, 0.0), "box": (-0.20, 0.0), "pot": (-0.10, 0.0)}[kind]    # ASSUMED (user-confirmed 11-02): goal xy of the object reference relative to its start
    return TaskDef(name=name, kind=kind, backend="carry", world_ref=world_ref, n_cups=n_cups, spec_ref=spec_ref, start_support="counter",
                   goal_support=goal_support, goal_dxy=(gx, gy),
                   goal_terms=(_XY, _Z, _TILT_G) + ((_CUP_TILT_G,) if cups else ()), shape=shape, geom_tags=geom_tags, empty_mass=empty_mass, empty_com=empty_com,
                   prior=prior, rows=rows, U=_u_carry(cups, lid), R=_r_carry(cups, lid), moves=moves, obj=obj,
                   note=f"moves: {_PLANS_SRC}", phenomena=_PHI[kind])


_TRAY_GEOM = (("shape", "MODEL", "nero/contents/objects.py:100 TRAY lx, ly (from agentchord_probe/coffee_tray/build.py geometry; ly 0.33 after 11-02)"),
              ("empty_mass", "MODEL", "objects.py:100 TRAY mass"), ("empty_com", "MODEL", "objects.py coffee_tray(): ObjectSpec com (0, 0, 0.004)"))
_BOX_GEOM = (("shape", "ASSUMED", "nero/contents/objects.py:128 BOX lx, ly (cardboard crate, scenes.py)"), ("empty_mass", "ASSUMED", "objects.py:128 BOX mass"),
             ("empty_com", "ASSUMED", "objects.py boxed(): COM at lz / 2 (uniform empty crate)"))
_POT_GEOM = (("shape", "MODEL", "nero/pan/scene.py:102 cylinder geom 'pan_bottom', radius pan_radius + pan_wall_t (both ASSUMED inputs)"),
             ("empty_mass", "MODEL", "scenes.pot_empty_props() from the scene geometry (adapters/contents_specs.empty_props)"),
             ("empty_com", "MODEL", "scenes.pot_empty_props()"))
_TRAY_PRIOR = Bound(0.0, 0.25, _ASM, _PRIOR_SRC + "; fill of each mug, kg per mug (OB.CUP fill_max_kg)", "kg per mug")
_BOX_PRIOR = Bound(0.5, 5.0, _ASM, _PRIOR_SRC + "; user-approved 11-02 (OB.BOX_CONTENTS_KG)", "kg")
_POT_PRIOR = Bound(0.02, 1.5, _ASM, _PRIOR_SRC + "; scenes.POT_CONTENTS_KG = pan/theta.py THETA mass (kg of puck)", "kg")


def _tray(n):
    return _carry_task(f"tray{n}", "tray", "world_model.v2.nero.contents.scenes:TrayWorld",
                       "world_model.v2.nero.contents.objects:coffee_tray", "counter", _TRAY_MOVES, Rect(0.22, 0.33),
                       ("torque_payload", "open_content", "support_dest"), n_cups=n, empty_mass=0.30, empty_com=(0.0, 0.0, 0.004),
                       prior=_TRAY_PRIOR, cups=True, geom_tags=_TRAY_GEOM)

# coffee tray in the NERO contents sim (nero/contents/coffee.py, task "tray0" = a tray with no mugs): the capsule must end inside OUR tray's inner footprint
_T0_SRC = "nero/contents/coffee.py (port of agentchord_probe/coffee_tray/env.py coffee_tray_check to our tray, objects.TRAY 0.22 x 0.33, rim_t 0.008)"
_T0_IX, _T0_IY = 0.22 / 2 - 0.008, 0.33 / 2 - 0.008
_T0_CAPSULE = (("abs(cap_rel_x)", ub(_T0_IX, _ASM, f"{_T0_SRC}: capsule inside the tray's inner footprint (lx/2 - rim_t)", "m")),
               ("abs(cap_rel_y)", ub(_T0_IY, _ASM, f"{_T0_SRC}: (ly/2 - rim_t)", "m")),
               ("abs(cap_rel_z)", ub(0.04, _ASM, f"{_T0_SRC}: capsule resting on the tray, z within 0.04 m", "m")))
_T0_PRIOR = Bound(0.0, 3.0, _ASM, "PROPOSED to the user 11-09, NOT YET RULED: hidden load inside the tray, kg (variant unknown); coffee.py LOAD_PRIOR", "kg")
_T0_KNOWN = Bound(0.0, 0.0, "MODEL", "variant known: nothing in the tray besides the tray (0.30 kg) and the capsule (12 g)", "kg")


def _tray0(name, goal_terms, note):
    t = _tray(0)
    return replace(t, name=name, goal_terms=goal_terms, prior=_T0_PRIOR, variant_priors=(("unknown", _T0_PRIOR), ("known", _T0_KNOWN)), empty_mass=0.312,
                   rows=("torque_payload", "support_dest"), U=_u_carry(False, False), R=_r_carry(False, False), note=note,
                   phenomena=())                      # coffee tray: out of this build (spec s.4: runs only where the task exists in the NERO sim); PH6 capsule is out of scope


TASKS: dict[str, TaskDef] = {t.name: t for t in (
    _tray(2), _tray(4), _tray(6),
    _carry_task("box", "box", "world_model.v2.nero.contents.scenes:BoxWorld", "world_model.v2.nero.contents.objects:boxed", "shelf", _BOX_MOVES,
                Rect(0.20, 0.30), ("torque_payload", "support_dest"), empty_mass=0.25, empty_com=(0.0, 0.0, 0.15 / 2),
                prior=_BOX_PRIOR, geom_tags=_BOX_GEOM),
    _carry_task("pot", "pot", "world_model.v2.nero.contents.scenes:PotWorld", "world_model.v2.nero.contents.scenes:pot_spec", "counter", _TRAY_MOVES,
                Disc(_POT_RADIUS), ("torque_payload", "open_content", "support_dest"), prior=_POT_PRIOR, lid=True, geom_tags=_POT_GEOM),
    _tray0("tray0", (("xy_err", ub(0.03, _ASM, f"{_T0_SRC}: tray_at_target, xy <= 0.03 m of start + goal offset", "m")),) + _T0_CAPSULE + (_CT_TILT,),
           "coffee tray, STRICT goal set = coffee_tray_check numbers on our tray; the pipeline task name for both coffee variants"),
    _tray0("tray0_general", (_XY, _Z, _TILT_G) + _T0_CAPSULE, "coffee tray, GENERAL goal set = AgentChord's tray rule (xy <= 5 cm, z <= 2 cm, tilt < 45 deg) + capsule inside; evaluated only"),
    TaskDef(name="pan_pinch", goal_terms=(("touch_left_1_n", lb(1, "MODEL", f"{_PAN_SUCC}: pan touched by left finger 1", "n")),
                                          ("touch_left_2_n", lb(1, "MODEL", f"{_PAN_SUCC}: pan touched by left finger 2", "n"))), **_PAN_COMMON),
    TaskDef(name="pan_push", goal_terms=(("push_frac", lb(0.8, _ASM, f"{_PAN_SUCC}: pan moved >= 0.8 x the planned push", "frac")),
                                         ("touch_left_pan_n", lb(1, "MODEL", f"{_PAN_SUCC}: pan still touched by the left claw", "n"))), **_PAN_COMMON),
    TaskDef(name="pan_clamp", goal_terms=(("touch_right_1_n", lb(1, "MODEL", f"{_PAN_SUCC}: right finger 1 touches something", "n")),
                                          ("touch_right_2_n", lb(1, "MODEL", f"{_PAN_SUCC}: right finger 2 touches something", "n")),
                                          ("touch_right_pan_n", lb(1, "MODEL", f"{_PAN_SUCC}: the pan is among them", "n"))), **_PAN_COMMON),
    TaskDef(name="pan_lift", goal_terms=(("rise_mm", lb(0.8 * _LIFT_M * 1e3, _ASM, f"{_PAN_SUCC}: rise >= 0.8 LIFT (LIFT = 0.05 m, pan/handle_grasp.py)", "mm")),
                                         ("support_force_N", ub(0.1, _ASM, f"{_PAN_SUCC}: support < 0.1 N (strict)", "N", strict=True)),
                                         ("obj_tilt_deg", ub(6.0, _ASM, f"{_PAN_SUCC}: tilt < 6 deg (strict)", "deg", strict=True)),
                                         ("spill_flag", ub(0.0, "MODEL", f"{_PAN_SUCC}: no spill", "flag")),
                                         ("lid_mm", ub(5.0, _ASM, f"{_PAN_SUCC}: lid < 5 mm off its seat (strict)", "mm", strict=True))), **_PAN_COMMON),
    TaskDef(name="coffee_tray", goal_dxy=(0.10, 0.0),
            goal_terms=(("xy_err", ub(0.03, _ASM, f"{_CT_SRC}: tray_at_target, xy <= 0.03 m of reset + forward (geometry.json forward)", "m")),) + _CT_CAPSULE + (_CT_TILT,),
            **_CT_COMMON),
    TaskDef(name="coffee_tray_min", goal_dxy=None,
            goal_terms=(("obj_x", lb(0.05, _ASM, f"{_CT_SRC}: tray_moved_forward, >= 5 cm in +x (the minimal prompt gives no distance)", "m")),
                        ("abs(obj_y)", ub(0.05, _ASM, f"{_CT_SRC}: <= 5 cm sideways", "m"))) + _CT_CAPSULE + (_CT_TILT,),
            **_CT_COMMON),
)}


# ---- table-behind tasks (USER 11-09: the base moves the object to the table BEHIND the robot; box added 11-09 "go, recommendations accepted")
# Scene: nero/base/scene.py (swerve base; counter2 = the 180-deg mirror of the counter about the mid-point of route R1 -> SAME height).
# goal_dxy = scene.place_pose_xy(world) - world.obj_xy (MODEL; adapters/tests/test_tasks_match_nero.py recomputes it from the scene).
_BEHIND_DXY = (-1.76, 0.0209)
_R1 = ("R1", (("line", -0.50), ("turn", 180.0), ("line", 0.50)))      # ASSUMED (nero/base/scene.py RETREAT_M / ADVANCE_M, smoke route)
_BEHIND_NOTE = "table-behind (docs/status/table_behind_design_2026-11-09.md); moves = none: the base route carries it, plans come from the planner"


def _behind(t: TaskDef, name, note):
    return replace(t, name=name, base="swerve", routes=(_R1,), goal_support="counter2", goal_dxy=_BEHIND_DXY, moves=(), rows=t.rows + ("base_tip",),
                   note=f"{_BEHIND_NOTE}; {note}; base_tip = stack/checks/base_tip.py (ZMP in the wheel polygon, measured base card)")


TASKS.update({t.name: t for t in (
    _behind(TASKS["tray6"], "tray6_behind", "tray of 6 mugs"),
    _behind(replace(TASKS["box"], world_kw=(("shelf", False),)), "box_behind",
            "crate; BoxWorld(shelf=False): no wall shelf, the goal is counter2 at counter height (USER 11-09)"),
)})

# goal xy offset by kind: the ONE copy (closed_chain.GOAL_DXY deleted 11-09).  Fixed-base tasks only (the table-behind rows have their own goal_dxy).
GOAL_DXY = {t.kind: t.goal_dxy for t in TASKS.values() if t.backend == "carry" and not t.base}


def task(name: str) -> TaskDef:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {sorted(TASKS)}")
    return TASKS[name]


def of_kind(kind: str, backend: str = "carry") -> TaskDef:
    """a TaskDef of this kind ("tray" | "box" | "pot"): the facts every task of a kind shares (goal support, goal offset, U / R tables)."""
    return next(t for t in TASKS.values() if t.kind == kind and t.backend == backend)


def prior_box(name: str, variant: str | None = None) -> tuple:
    """the contents prior box (lo, hi) of a task: ONE copy (TaskDef.prior); tray: kg per mug.  variant: one of TaskDef.variant_priors (None = the default prior)."""
    t = task(name)
    p = t.prior
    if variant is not None:
        vp = dict(t.variant_priors)
        if variant not in vp:
            raise KeyError(f"{name}: no variant {variant!r}; have {sorted(vp)}")
        p = vp[variant]
    return (p.lo, p.hi)


def names(backend: str | None = None):
    return [n for n, t in TASKS.items() if backend is None or t.backend == backend]


def empty_props(t: TaskDef):
    """(mass kg, COM in the object frame) of the empty object.  The pot's (MODEL, from the scene geometry: scenes.pot_empty_props) is not stored here:
    the adapter layer supplies it (adapters/contents_specs.empty_props)."""
    if t.empty_mass is None:
        raise ValueError(f"{t.name}: empty mass comes from the scene; use adapters.contents_specs.empty_props")
    return float(t.empty_mass), tuple(t.empty_com)
