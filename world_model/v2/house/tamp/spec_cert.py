"""tamp/spec_cert.py -- the PLAN-TIME certifier evaluated from a CONSTRAINT SPEC in our IR (tray_rep/sets/ir.py).  SIMULATION ONLY.

Why (user decisions 2026-10-26): the planner plans safely, the runtime checks only the cup + actuator limits, and in the future the VLM AUTHORS the guardrails.  So the
plan-time certificate must be "evaluate a spec" instead of hard-coded rows.  `certify_spec(traj, spec, world, gk, people, t0, pose0)` is plan_cert.certify_plan with the
person / wall rows read from the spec; `spec_default.default_spec` is the hand-written spec equal to today's rows (tests/test_spec_cert.py proves the equivalence).

STATE.  trajectory_state(...) -> TrajState: the IR state with a TIME AXIS K = i_rest + 1 over the trajectory samples t_k = k h (nominal + backup, up to and including the first
at-rest sample).  Entities (make_entities), all with a leading K axis:
    footprint_people  BoxShape = the gatekeeper footprint FP x [0.15, 1.15] m (the obstacle band of loc/mapper.py), pose = traj.poses() composed with bel.pose_people
    footprint_wall    same box, pose composed with bel.pose_wall (the controller-frame pose the wall check is anchored at)
    person_i          Ball(PERSON_CLEAR_R) at the PREDICTED centre c_i + v_hat_i (t_k + latency) with position uncertainty Unc.ball_k = e_p,i + e_v t_e + a_p t_e^2/2 (+ sweep)
    tray, tray_top    the carried tray (pose = base pose composed with the hold pose, level), cup (Point) in the tray_top frame at the hold position
    obs_j             perceived zonotope obstacles (step 2), world frame
CONTINUOUS TIME.  Sampled-only evaluation is not a certificate: between samples k and k+1 every point of the footprint moves at most (h/2) * (the far-point speed
max over the adjacent samples, sl_k = max_{k-1,k,k+1} |v_far|) from the pose of the NEARER sample, and the predicted person disc surface at most (h/2) * rate_k, rate_k =
|v_hat| + e_v + a_p (t_e + h) (centre speed + radius growth rate over the interval).  Both are ADDED to the uncertainty balls (footprint ball += h/2 sl_k, person ball +=
h/2 rate_k), i.e. exactly the 0.5 h (vfar + rate) term of gatekeeper/checks.py (gk.sweep_slack).  At the rest sample (v = 0 from there on) the person sweep terms are 0
(checks.py check2 "rest_person").  Hardware / cup rows are evaluated by checks.check1 / check2 (below), they are sound between samples by their own derivation (cup_chord is opt-in there).
WALLS.  step 1 ("grid"): the EDT wall rows of checks.check1 / check2 are kept (pose box e_xy, e_th, boundary sampling, dock buffer) -- the IR spec has no wall atoms.
step 2 ("zono"): Clear(footprint_wall, obs_j, d_wall) with the footprint ball = e_xy + max|q| e_th (pose box, conservative: the per-point |q_j| e_th of the grid row is
replaced by the max) + the sweep term, minus (d_wall - d_dock) inside a dock radius (the grid row's dock buffer); no fp_step/2 (the footprint is an exact box).
WHAT THE IR CANNOT EXPRESS AND IS KEPT FROM checks.py (called with runtime_rows="cup", or "all" with the people stripped when the walls are grid rows): cup acceleration
|a_cup| <= a_cup_max (needs the base twist V, not a pose), hardware slew / speed caps, rest_speed, no_rest.  The new IR atom ClearEgress (ir.py) expresses the separating-motion
exit rule.  The cup atoms Upright(tray) / Inside(cup, shrink(tray_top)) are constants here (the hold is rigid and the base is planar: tilt 0) -- they exercise the spec format.
Return of certify_spec: (ok, min_margin, reason, per_atom_margins) like plan_cert.certify_plan.  ok = checks.check1/check2 ok AND the spec's robust margin >= -gk.tol.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import Ball, BoxShape, Entity, Pose, Point, Unc, ZonoShape, prism_xy
from world_model.v2.tray_rep.sets.zono import Zonotope

from ..gatekeeper import checks as CK
from ..gatekeeper.traj import Traj
from ..teleop import geom as GM
from . import plan_cert as PC

Z_LO, Z_HI = 0.15, 1.15                    # m, vertical extent of the footprint prism = the obstacle band of loc/mapper.MapCfg (obs_zlo, obs_zhi)
PZ = 0.5 * (Z_LO + Z_HI)
HZ = 0.5 * (Z_HI - Z_LO)
FAR_AWAY = 1e6                             # a person who is not constrained (trajectory at rest throughout) is moved this far away
TRAY_Z = 0.85                              # tray surface height (teleop/scene.TRAY_HOLD_Z)


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


@dataclass
class SpecCfg:
    """what trajectory_state needs besides the trajectory: the planner world + checker config + payload (see plan_cert.certify_plan); bel overrides the belief."""
    world: object
    gk: object
    holding: bool = True
    item: str | None = "tray"
    bel: object = None


class TrajState:
    """the IR state: plain attributes read by the entity providers of make_entities (every array has the leading time axis K)."""

    def __init__(self):
        self.K = 0


def trajectory_state(traj: Traj, pose0, t0: float, people, cfg: SpecCfg) -> TrajState:
    """IR state over the trajectory samples 0..i_rest (all samples if the profile never rests).  pose0: map pose of sample 0; t0: delay [s] between the people observation
    and the start; people: [(c, v_hat), ...] observed at time 0 (None: world.bel_people).  See the module docstring for the entities and the inter-sample terms."""
    world = cfg.world
    gk = PC._plan_gk(cfg.gk)
    pose0 = world.q0.xyth() if pose0 is None else pose0
    bel = cfg.bel if cfg.bel is not None else PC.plan_belief(world, pose0, cfg.holding, cfg.item if cfg.holding else None, people, t0, gk)
    h, N = traj.h, traj.N
    i_rest = traj.i_rest
    K = N + 1 if i_rest is None else i_rest + 1
    st = TrajState()
    st.K, st.bel, st.gk, st.i_rest = K, bel, gk, i_rest
    st.t = np.arange(K) * h
    Pr = traj.poses()[:K]
    sweep = bool(gk.sweep_slack)
    # ---- people frame: the footprint, the carried tray / cup, the predicted persons
    Pp = CK.world_poses(Pr, bel.pose_people)
    sw_mask = np.ones(K)
    if i_rest is not None:
        sw_mask[-1] = 0.0                                                    # the REST sample: no inter-sample sweep (checks.py check2 rest_person)
    vfar, _ = CK._slack(traj, CK._rect_far(bel.fp))
    st.fp_people_R, st.fp_people_p = _rz(Pp[:, 2]), np.stack([Pp[:, 0], Pp[:, 1], np.zeros(K)], 1)
    st.fp_people_ball = 0.5 * h * vfar[:K] * sw_mask if sweep else np.zeros(K)
    active = i_rest is None or i_rest > 0                                    # a trajectory that is at rest throughout constrains no person (checks.py: next plan's problem)
    st.person_c, st.person_ball, st.person_rate = [], [], []
    for ip, c in enumerate(bel.people):
        c = np.asarray(c, float)
        vh = CK._vhat(bel, ip)
        te = st.t + CK._lat(bel, ip)                                         # radius-growth time (observation age incl. the track's staleness and the delay t0)
        tc = st.t + bel.latency_s                                            # centre-advance time (the belief centre is already advanced by t0)
        ev = CK._ev(bel, ip)
        rate = math.hypot(vh[0], vh[1]) + ev + bel.a_p * (te + h)
        ball = CK._pe(bel, ip) + ev * te + 0.5 * bel.a_p * te * te + (0.5 * h * rate * sw_mask if sweep else 0.0)
        ctr = np.stack([c[0] + vh[0] * tc, c[1] + vh[1] * tc, np.full(K, PZ)], 1)
        if not active:
            ctr = np.full((K, 3), FAR_AWAY)
        st.person_c.append(ctr)
        st.person_ball.append(np.asarray(ball, float) * np.ones(K))
        st.person_rate.append(rate)
    # ---- the carried tray / cup (rigid with the base): hold pose in the base frame
    hold = world.hold_rel["tray"]
    ch, sh = np.cos(Pp[:, 2]), np.sin(Pp[:, 2])
    st.tray_R = _rz(Pp[:, 2] + hold[2])
    st.tray_p = np.stack([Pp[:, 0] + ch * hold[0] - sh * hold[1], Pp[:, 1] + sh * hold[0] + ch * hold[1], np.full(K, TRAY_Z)], 1)
    cr = np.asarray(world.cup_rel, float)
    d = cr - np.asarray(hold[:2], float)
    cd, sd = math.cos(hold[2]), math.sin(hold[2])
    st.cup_in_tray = np.broadcast_to(np.array([cd * d[0] + sd * d[1], -sd * d[0] + cd * d[1], 0.0]), (K, 3)).copy()
    st.eye = np.broadcast_to(np.eye(3), (K, 3, 3)).copy()
    st.zero = np.zeros(3)
    # ---- wall frame (step 2: zonotope walls): pose box + sweep + dock buffer in the footprint ball
    Pw = CK.world_poses(Pr, bel.pose_wall)
    B, nq = CK._fp_pts(bel.fp, gk.fp_step)
    qmax = float(nq.max())
    sl_m, _ = CK._slack(traj, qmax)
    dock = bel.d_wall - CK._d_wall_k(bel, Pw)                                # >= 0 inside a dock radius (the grid row's smaller buffer d_wall_dock)
    st.fp_wall_R, st.fp_wall_p = _rz(Pw[:, 2]), np.stack([Pw[:, 0], Pw[:, 1], np.zeros(K)], 1)
    st.fp_wall_ball = bel.e_xy + qmax * bel.e_th + (0.5 * h * sl_m[:K] if sweep else 0.0) - dock
    return st


def make_entities(n_people: int, zonos=(), fp=None, tray_half=(0.20, 0.15)):
    """the entity catalogue the spec names refer to (state-independent: the providers read a TrajState)."""
    fp = dict(GM.FP) if fp is None else fp
    half = np.array([0.5 * (fp["x"][1] - fp["x"][0]), 0.5 * (fp["y"][1] - fp["y"][0]), HZ])
    cen = np.array([0.5 * (fp["x"][1] + fp["x"][0]), 0.5 * (fp["y"][1] + fp["y"][0]), PZ])
    ents = {}
    ents["footprint_people"] = Entity("footprint_people", BoxShape(half, cen), lambda s: Pose(s.fp_people_R, s.fp_people_p), lambda s: Unc(None, 0.0, s.fp_people_ball))
    ents["footprint_wall"] = Entity("footprint_wall", BoxShape(half, cen), lambda s: Pose(s.fp_wall_R, s.fp_wall_p), lambda s: Unc(None, 0.0, s.fp_wall_ball))
    for i in range(n_people):
        ents[f"person_{i}"] = Entity(f"person_{i}", Ball(GM.PERSON_CLEAR_R), lambda s, i=i: Pose(s.eye, s.person_c[i]), lambda s, i=i: Unc(None, 0.0, s.person_ball[i]))
    ents["tray"] = Entity("tray", prism_xy(*tray_half), lambda s: Pose(s.tray_R, s.tray_p))
    ents["tray_top"] = Entity("tray_top", prism_xy(*tray_half), lambda s: Pose(s.tray_R, s.tray_p))
    ents["cup"] = Entity("cup", Point(), lambda s: Pose(s.eye, s.cup_in_tray), parent="tray_top")
    for j, Z in enumerate(zonos):
        Z = Z if isinstance(Z, Zonotope) else Zonotope(*Z)
        ents[f"obs_{j}"] = Entity(f"obs_{j}", ZonoShape(Z), lambda s: Pose(np.eye(3), s.zero))
    return ents


# ====================================================================================== evaluation
def atoms_of(node):
    """all atom nodes of a spec tree (left to right)."""
    if hasattr(node, "A"):
        yield node
    elif hasattr(node, "c"):
        for c in (node.c if isinstance(node.c, list) else [node.c]):
            yield from atoms_of(c)


def spec_margins(spec_node, state, robust=True):
    """-> (overall margin at time 0, {atom tag: worst margin over time}); ClearEgress atoms also report tag:clear and tag:exit.  Every atom is evaluated ONCE: the tree
    is then reduced from the cached atom series (instance-level override of Node.eval, removed afterwards), so the overall margin costs nothing extra."""
    mg, cache = {}, {}
    atoms = list(atoms_of(spec_node))
    for i, a in enumerate(atoms):
        tag = a.tag or f"{type(a).__name__.lower()}_{i}"
        if isinstance(a, ir.ClearEgress):
            p = a.parts(state, robust)
            cache[id(a)] = p["x"]
            mg[tag] = float(p["x"].min())
            mg[tag + ":clear"] = float(p["clear"])
            if p["exit"] is not None:
                mg[tag + ":exit"] = float(p["exit"])
        else:
            v = ir._t(a.eval(state, robust))
            cache[id(a)] = v
            mg[tag] = float(np.min(v))
    try:
        for a in atoms:
            a.eval = (lambda st, robust=False, _v=cache[id(a)]: _v)
        total = float(ir.margin(spec_node, state, robust))
    finally:
        for a in atoms:
            a.__dict__.pop("eval", None)
    return total, mg


def certify_spec(traj: Traj, spec, world, gk, people=None, t0: float = 0.0, pose0=None, holding: bool = True, item: str | None = "tray", bel=None):
    """-> (ok, min_margin, reason, per_atom_margins).  spec: spec_default.Spec (or anything with .node, .walls, .for_people(n)); other arguments as plan_cert.certify_plan.
    per_atom_margins = the checks.py rows (cup, hw_slew, speed_cap, rest_speed, and wall / wall_at_rest for walls="grid") + one entry per IR atom tag
    (+ tag:clear, tag:exit for ClearEgress) + "spec" (the robust margin of the whole tree at time 0)."""
    gk = PC._plan_gk(gk)
    pose0 = world.q0.xyth() if pose0 is None else pose0
    if bel is None:
        bel = PC.plan_belief(world, pose0, holding, item if holding else None, people, t0, gk)
    if traj.i_rest is None:
        return False, -1.0, "no_rest", {"no_rest": -1.0}
    grid = getattr(spec, "walls", "zono") == "grid"
    bel_rows = bel.with_(people=(), people_vel=(), people_e_p=(), people_e_v=(), people_stale=())
    gk_rows = gk if grid else dataclasses.replace(gk, runtime_rows="cup")
    o1, m1, r1, g1 = CK.check1(traj, bel_rows, gk_rows)
    o2, m2, r2, g2 = CK.check2(traj, bel_rows, gk_rows)
    mg = dict(g1)
    mg.update(g2)
    sp = spec.for_people(len(bel.people))
    st = trajectory_state(traj, pose0, t0, people, SpecCfg(world, gk, holding, item, bel))
    m_all, am = spec_margins(sp.node, st, robust=True)
    mg.update(am)
    mg["spec"] = m_all
    if not o1:
        return False, m1, r1, mg
    if not o2:
        return False, m2, r2, mg
    tol = gk.tol
    if m_all < -tol:
        bad = {k: v for k, v in am.items() if v < -tol and ":" not in k}
        k = min(bad, key=bad.get) if bad else "spec"
        return False, float(bad.get(k, m_all)), k, mg
    rows = {k: v for k, v in mg.items() if k not in PC.TIGHT_BY_CONSTRUCTION and k != "spec" and ":" not in k}
    if not rows:
        return True, 0.0, "none", mg
    k = min(rows, key=rows.get)
    return True, float(rows[k]), k, mg
