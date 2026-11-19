"""adapters/collision_rows.py -- the swept collision row (checks/collision.py) built from a planned ClosedChainPlan (physics: MuJoCo kinematics only, no stepping).

Per plan piece (a rest-to-rest segment at the control period CC.DT): set the carry-scene model to the sample (arm q per side, object by the grasp), mj_forward, and take
the min distance per pair class
  arm_arm    left-arm geoms vs right-arm geoms (clearance.classify: every collidable geom of a body named left_arm* / right_arm*, grippers included)
  arm_world  arm geoms vs the world geoms of clearance.classify (names table / counter / shelf / stove) and vs the floor plane(s) (hull min z vs the plane height)
  obj_world  the carry row of USER 11-16 D9: (world geom, load geom) pairs as carry.CarryEpisode._obj_world_pairs (same classification, re-implemented here because that is an
             episode method), same allowance (a separation within carry.SUPPORT_APPROACH_DEG of +z = resting on / approaching a support top is NOT a collision), same fallback
             (carry._box_sat when mj_geomDistance returns exactly 0 for two boxes), same cap carry._OBJ_WORLD_DMAX
Distances arm_arm / arm_world: clearance._dist (exact GJK on the convex hulls; clearance.py says mj_geomDistance returned 0.0 for separated mesh/box pairs, so it is not used there),
capped at clearance._DMAX (0.10 m: "at least that far").  obj_world: mj_geomDistance as in carry.  Capped values are lower bounds, so the swept bound stays valid.

How the object is placed (GUESS, stated): the plan holds the tool pose rigidly fixed to the object, so every free-joint body of the scene (obj, cups, contents, lid: all that carry
treats as the load) sits at its model start pose (m.qpos0) + p_k (the plan's object offset), orientation unchanged (level).  The gripper joints stay at m.qpos0.
Table-behind plans: pieces after the base move (index >= wp.base[0][0] in wp.names) get the planar rigid transform of the base move applied to the base free joint and to the load,
W = (rotate by dl about c0, then move c0 to ce), from geo.info["base"] start/end pose (the same transform as closed_chain._x_obj(after_base)); world geoms stay.  The base move itself
(no piece of the plan) is covered by base_route.check, not here.

Speed bound (MODEL) per geom g over the interval k -> k+1:  s_g = (|x_g(k+1) - x_g(k)| + theta_g * r_g) / dt,  x_g = geom origin, theta_g = angle of R_g(k)^T R_g(k+1),
r_g = max |hull vertex| in the geom frame (clearance._hull_pts) or geom_rbound if no hull.  Pair speed s_p = s_a + s_b (static world geom: s = 0).  Class speed v_k = max s_p over the pairs that can bind:
with dm_p = min(d_p(k), d_p(k+1)) and dm = min_p dm_p, only pairs with dm_p <= dm + s_max * dt can have a lower swept bound below dm's pair (s_max = max_p s_p), so v_k = max s_p over those.
A pair that is allowed (obj_world support approach) at a sample counts as the cap distance at that sample; the allowance is evaluated at the samples only (not between them).

LOAD TILT (USER 11-17 "Checker uses measured tilt"; default ON, BIMANUAL_LEGACY_UNTILTED_COLLISION=1 = the untilted row exactly as before).  Only obj_world changes.
  Cover set at a sample (MODEL, rigid load: every carried body turns with the object): nominal load pose L, rotated about the pivot c by R = R_swing R_twist with R_swing about
  ANY horizontal axis by an angle <= sw and R_twist about world z by <= tw, then translated by any |t| <= de.  c = the nominal position of the object-fixed point pivot_obj
  (the claw midpoint at k = 0 of the measurement episodes, object frame).  sw / tw / de come from the per-plan MEASUREMENT (tilt_measure.measure_plan -> RotTrack: per piece,
  max over n episodes of swing, twist and set-based pivot drift; guarantee "sampled", n ASSUMED there) via TiltSpec / tilt_from_table; no measurement = the row is MISSING.
  Lower bound on the distance over the cover set, per (world geom W, load geom L) pair (MODEL, exact algebra, sound for every R / t in the set):
    separating direction n (unit, W -> L):  d >= min_j lb_j(n) - h_W(n) - de,  lb_j(n) = n.c + f_j - 2 sin(tw/2) |u_j,h|,  u_j = v_j - c (L's vertices, untilted),
      f_j = min over phi in [0, sw] of (A - C) cos phi - B sin phi + C  (closed form: K cos(phi + g) + C, K = hypot(A - C, B), g = atan2(B, A - C); = C - K if g + sw >= pi else value at sw)
      A = n.u, B = |(u x n)_h|, C = (u_h.n_h - |u_h||n_h|) / 2  (from R_a(phi) u = cos phi u + sin phi a x u + (1 - cos phi) a (a.u), a horizontal unit: a.(u x n) <= B, (a.u)(a.n) >= C)
      twist term: |R_s R_t u - R_s u| = |R_t u - u| <= 2 sin(tw/2) |u_h|.   h_W(n) = support of W (box: n.x + sum_i h_i |n.R_i|; plane: n = its normal only).
      L a polytope (box) -> the min of a linear function is at a vertex, so the per-vertex bound bounds the whole geom.  n ranges over the 15 SAT axes of the two boxes (both signs)
      and the untilted closest-point direction (mj_geomDistance fromto); the max over n is taken (each n alone is a valid bound).
    other geom types (no vertices / no support function): d >= d0 - bloat - de, bloat = max point displacement = max_j 2 sin(sw/2)|u_j| + 2 sin(tw/2)|u_j,h| (box) or
      (|x_L - c| + rbound_L)(2 sin(sw/2) + 2 sin(tw/2)).
  Swept composition: within a piece the load only translates (orientation fixed), and c moves with it, so the cover set at an interpolated state = the cover set at sample k
  translated by p(s) - p(k): the bound at k minus |p(s) - p(k)| holds there, which is what the existing speed term (geom displacement / dt) charges; refine_samples carries the
  per-sample tilt into its midpoints (same piece = same tilt).  The support allowance is evaluated at the untilted pose and only where the untilted row evaluates it
  (distance < cap), exactly as before; every other pair gets the tilt bound.

Constants: obj_world d_min = carry.U_OBJ_CLEAR_M (5 mm, ASSUMED, USER 11-16 D9).  arm_arm / arm_world d_min = clearance.CLEARANCE_MIN (5 mm, ASSUMED, "same as pan CLAMP clear"; reused, no new constant).
Not covered (stated): arm links vs the carried object (non-gripper), self-collision of one arm, the base / lift column vs arms, anything between samples that the speed model misses.
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import mujoco

from world_model.v2.nero import clearance as CL
from world_model.v2.stack.checks import collision as K
from world_model.v2.stack.planners import closed_chain as CC

SIDES = CC.SIDES
CLASSES = K.CLASSES


# ---- load tilt cover (USER 11-17)
def legacy_untilted():
    """BIMANUAL_LEGACY_UNTILTED_COLLISION=1: the obj_world row at the planned (untilted) load pose, exactly as before 11-17."""
    return os.environ.get("BIMANUAL_LEGACY_UNTILTED_COLLISION", "0") == "1"


class TiltMissing(Exception):
    """no usable per-plan tilt measurement: the row is MISSING."""


@dataclasses.dataclass(frozen=True)
class TiltSpec:
    """the MEASURED load deviation of one plan (tilt_measure.measure_plan, RotTrack): per piece (swing [rad], twist [rad], pivot drift [m]), max over n episodes (sampled)."""
    per_piece: tuple            # ((name, swing_rad, twist_rad, drift_m, n_episodes), ...)
    pivot_obj: tuple            # pivot in the object frame [m] (claw midpoint at k = 0)
    x0: tuple                   # the measurement model's object start pose (qpos0) [m]
    R0: tuple                   # its orientation (9,)
    n: int                      # episodes with a rotation record
    source: str = ""            # TABLE.json path
    key: str = ""
    a: float = float("nan")
    extra: tuple = ()           # (name, value) pairs recorded in the row's assumptions

    def piece(self, name):
        for nm, sw, tw, de, ne in self.per_piece:
            if nm == name:
                if ne < 1:
                    raise TiltMissing(f"piece {name!r}: no measured episode reached it")
                return (float(sw), float(tw), float(de))
        raise TiltMissing(f"piece {name!r} has no tilt measurement (measured pieces: {[p[0] for p in self.per_piece]})")

    def envelope(self):
        """max over the pieces (for motions that are not one plan piece: Rule 1's recovery)."""
        if not self.per_piece:
            raise TiltMissing("empty tilt measurement")
        return tuple(float(max(p[i] for p in self.per_piece)) for i in (1, 2, 3))

    def tag(self):
        return (self.source, self.key, round(float(self.a), 6), self.per_piece)

    def to_json(self):
        return dict(per_piece={p[0]: dict(swing_deg=math.degrees(p[1]), twist_deg=math.degrees(p[2]), drift_m=p[3], n_episodes=p[4]) for p in self.per_piece},
                    pivot_obj=list(self.pivot_obj), n=self.n, source=self.source, key=self.key, a=self.a, **dict(self.extra))


def tilt_from_table(tab, key, a, source=None):
    """TiltSpec from a tilt_measure.measure_plan table (cells[key].levels[repr(a)].rot).  Raises TiltMissing when the measurement is absent or incomplete
    (no rot record, an errored episode, an episode without a rot record, pre-hold steps not folded into a piece)."""
    try:
        lv = tab["cells"][key]["levels"][repr(float(a))]
    except (KeyError, TypeError) as ex:
        raise TiltMissing(f"no tilt table entry for {key} at a = {a}: {ex!r}")
    rot = lv.get("rot")
    if not rot:
        raise TiltMissing(f"tilt table {key} a = {a} has no per-piece rotation record (tilt_measure RotTrack)")
    if lv.get("n_errors", 0) or rot.get("n_missing_rot", 0) or rot.get("n_pre_unfolded", 0):
        raise TiltMissing(f"tilt measurement incomplete: n_errors {lv.get('n_errors')}, n_missing_rot {rot.get('n_missing_rot')}, n_pre_unfolded {rot.get('n_pre_unfolded')}")
    pp = tuple((s, math.radians(v["swing_deg"]), math.radians(v["twist_deg"]), float(v["drift_m"]), int(v["n_episodes"])) for s, v in sorted(rot["per_seg_max"].items()))
    extra = (("n_stopped", lv.get("n_stopped")), ("pivot_obj_spread_m", rot.get("pivot_obj_spread_m")), ("x0_spread_m", rot.get("x0_spread_m")),
             ("R0_spread_rad", rot.get("R0_spread_rad")), ("n_unattributed_steps", rot.get("n_unattributed_steps")), ("unattributed_max_deg", rot.get("unattributed_max_deg")),
             ("seeds", lv.get("seeds")))
    return TiltSpec(per_piece=pp, pivot_obj=tuple(rot["pivot_obj"]), x0=tuple(rot["x0"]), R0=tuple(np.ravel(rot["R0"])), n=int(rot["n_rot"]),
                    source=str(source or tab.get("file") or ""), key=str(key), a=float(a), extra=extra)


_SG = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def swing_twist_lb(n, U, sw, tw):
    """per vertex j: a lower bound on n.(R u_j) over R = R_swing R_twist, swing about a horizontal axis by <= sw, twist about z by <= tw (MODEL, see the module docstring).
    n (3,) unit, U (k, 3) vertices relative to the pivot -> (k,)."""
    A = U @ n
    ux = np.cross(U, n)
    B = np.hypot(ux[:, 0], ux[:, 1])
    uh = np.linalg.norm(U[:, :2], axis=1)
    C = 0.5 * (U[:, :2] @ n[:2] - uh * float(np.hypot(n[0], n[1])))
    K_ = np.hypot(A - C, B)
    g = np.arctan2(B, A - C)
    f = np.where(g + sw >= np.pi, C - K_, (A - C) * math.cos(sw) - B * math.sin(sw) + C)
    return np.minimum(f, A) - 2.0 * math.sin(0.5 * tw) * uh


def displacement_bound(U, sw, tw):
    """max over the cover set of |R u - u| per vertex: 2 sin(sw/2)|u| + 2 sin(tw/2)|u_h|."""
    return 2.0 * math.sin(0.5 * sw) * np.linalg.norm(U, axis=1) + 2.0 * math.sin(0.5 * tw) * np.linalg.norm(U[:, :2], axis=1)


def d_min_table():
    from world_model.v2.nero.contents import carry
    return dict(arm_arm=CL.CLEARANCE_MIN, arm_world=CL.CLEARANCE_MIN, obj_world=carry.U_OBJ_CLEAR_M)


def _radius(m, g):
    try:
        return CL._hull_pts(m, g)[1]
    except Exception:
        return float(m.geom_rbound[g])


class _Scene:
    """one plan's model + data and the pair lists of the three classes."""

    def __init__(self, plan, tilt=None):
        """tilt: a TiltSpec (the per-plan measurement) or None.  With a TiltSpec (and not legacy_untilted()), set_state(..., tilt=(sw, tw, de)) makes obj_world the cover-set bound."""
        from world_model.v2.nero.contents import carry
        self.carry = carry
        self.tilt, self.cur = (None if legacy_untilted() else tilt), None
        self.plan, self.inp = plan, plan.inp
        m = self.m = self.inp.m
        self.d = mujoco.MjData(m)
        self.qadr = {s: np.asarray(self.inp.arms[s].qadr, int) for s in SIDES}
        robot, world, _ = CL.classify(m, "obj")
        self.robot = robot
        self.nm = lambda g: (m.geom(g).name or f"g{g}:{m.body(m.geom_bodyid[g]).name}") if g >= 0 else "floor"
        planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.planes = planes
        allrobot = robot["left"] + robot["right"]
        self.pairs = dict(arm_arm=[(g, h) for g in robot["left"] for h in robot["right"]],
                          arm_world=[(g, w) for g in allrobot for w in world] + [(g, -1 - p) for g in allrobot for p in range(len(planes))])
        # obj_world: carry._obj_world_pairs
        carried = {b for b in range(m.nbody) if self._is_free_load(b)}      # bodies with a free joint other than the base's = what rides with the grasp (obj, cups, contents, lid)
        self.carried = carried

        def is_load(b):
            while b > 0:
                if b in carried:
                    return True
                b = int(m.body_parentid[b])
            return False
        arm_g = set(allrobot)
        robot_roots = {int(m.body_rootid[m.geom_bodyid[g]]) for g in arm_g}
        load, wld = [], []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            b = int(m.geom_bodyid[g])
            if g in arm_g or (int(m.body_rootid[b]) in robot_roots and int(m.body_rootid[b]) != 0):
                continue
            (load if is_load(b) else wld).append(g)
        self.pairs["obj_world"] = [(w, l) for w in wld for l in load]
        self.load_g = load
        self.cap = dict(arm_arm=CL._DMAX, arm_world=CL._DMAX, obj_world=carry._OBJ_WORLD_DMAX)
        self.moving = set(allrobot) | set(load)                                  # world geoms never move (speed 0)
        self.rad = {g: _radius(m, g) for g in self.moving | set(world) | set(wld)}
        self.free = [(int(m.jnt_qposadr[j]), j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and (m.joint(j).name or "") != "floating_base"]
        jb = [j for j in range(m.njnt) if (m.joint(j).name or "") == "floating_base"]
        self.base_adr = int(m.jnt_qposadr[jb[0]]) if jb else None
        wp = self.inp.wp
        self.post_idx = wp.base[0][0] if (wp is not None and wp.base) else None
        self.names_wp = list(wp.names) if wp is not None else None
        self.obj_b = m.body("obj").id
        if self.tilt is not None:
            jo = next(j for j in range(m.body_jntadr[self.obj_b], m.body_jntadr[self.obj_b] + m.body_jntnum[self.obj_b]) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
            adr = int(m.jnt_qposadr[jo])
            R0 = np.zeros(9)
            mujoco.mju_quat2Mat(R0, np.array(m.qpos0[adr + 3:adr + 7], float))
            dR = R0.reshape(3, 3).T @ np.reshape(self.tilt.R0, (3, 3))
            self.r0_mis = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
            self.x0_mis = float(np.linalg.norm(np.array(m.qpos0[adr:adr + 3]) - np.array(self.tilt.x0)))      # added to the drift (the measurement's nominal = its own qpos0)
            if self.r0_mis > 1e-6:
                raise TiltMissing(f"the tilt measurement's start orientation differs from the row model's by {self.r0_mis:.2e} rad: the measured swing / twist do not apply")
            self.piv = np.array(self.tilt.pivot_obj, float)

    def _is_free_load(self, b):
        m = self.m
        return any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and (m.joint(j).name or "") != "floating_base" for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))

    def post_base(self, sg):
        return self.post_idx is not None and self.names_wp is not None and sg in self.names_wp and self.names_wp.index(sg) >= self.post_idx

    def _W(self):
        b = self.plan.geo.info["base"]
        from world_model.v2.stack.planners import base_route as BR
        c0, ce = np.array(b["start_pose"][:2]), np.array(b["end_pose"][:2])
        dl = float(b["end_pose"][2] - b["start_pose"][2])
        return c0, ce, dl, BR.rot(dl)

    def tilt_of(self, sg=None):
        """(sw, tw, de) for piece sg (None = the envelope over the pieces), de including the start-pose mismatch; None when the row is untilted."""
        if self.tilt is None:
            return None
        sw, tw, de = self.tilt.piece(sg) if sg is not None else self.tilt.envelope()
        return (sw, tw, de + self.x0_mis)

    def set_state(self, ql, qr, p, post=False, tilt=None):
        """tilt: (sw, tw, de) of this sample (tilt_of) or None (untilted: the obj_world distances as before)."""
        self.cur = tilt if self.tilt is not None else None
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.qadr["left"]] = ql
        d.qpos[self.qadr["right"]] = qr
        W = self._W() if post else None
        qz = None
        if post:
            c0, ce, dl, R = W
            qz = np.array([np.cos(dl / 2), 0.0, 0.0, np.sin(dl / 2)])

        def put(adr, shift):
            pos = m.qpos0[adr:adr + 3] + shift
            quat = m.qpos0[adr + 3:adr + 7].copy()
            if post:
                pos = np.r_[ce + R @ (pos[:2] - c0), pos[2]]
                out = np.zeros(4)
                mujoco.mju_mulQuat(out, qz, quat)
                quat = out
            d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = pos, quat
        for adr, _ in self.free:
            put(adr, np.asarray(p, float))
        if self.base_adr is not None and post:
            put(self.base_adr, np.zeros(3))
        mujoco.mj_forward(m, d)

    # ---- distances at the current state
    def distances(self):
        """-> {class: (D (P,), allowed (P,))} at the current state; D capped, allowed pairs counted at the cap."""
        m, d = self.m, self.d
        out = {}
        for c in ("arm_arm", "arm_world"):
            D = np.full(len(self.pairs[c]), self.cap[c])
            A, B, RA, RB = self._arrs(c)
            xa = d.geom_xpos[A]
            xb = np.where((B >= 0)[:, None], d.geom_xpos[np.maximum(B, 0)], xa + np.array([0.0, 0.0, 1e3]))      # floor pairs always evaluated
            near = np.flatnonzero(np.linalg.norm(xa - xb, axis=1) <= RA + RB + self.cap[c])                       # hulls lie within r of the geom origin: farther apart than r_a + r_b + cap = capped (exact)
            for i in near:
                a, b = self.pairs[c][i]
                if b < 0:
                    Pg, _ = CL._hull_pts(m, a)
                    zmin = float((Pg @ d.geom_xmat[a].reshape(3, 3).T + d.geom_xpos[a])[:, 2].min())
                    D[i] = min(zmin - d.geom_xpos[self.planes[-1 - b]][2], self.cap[c])
                else:
                    D[i] = CL._dist(m, d, a, b)
            out[c] = (D, np.zeros(len(D), bool))
        out["obj_world"] = self._obj_world()
        return out

    def _obj_world(self):
        if self.cur is not None:
            return self._obj_world_tilted(*self.cur)
        carry, m, d = self.carry, self.m, self.d
        pairs = self.pairs["obj_world"]
        cap = self.cap["obj_world"]
        D = np.full(len(pairs), cap)
        allowed = np.zeros(len(pairs), bool)
        if not pairs:
            return D, allowed
        gx, rb = d.geom_xpos, m.geom_rbound
        W = np.array([w for w, _ in pairs]); L = np.array([l for _, l in pairs])
        cosl = np.cos(np.radians(carry.SUPPORT_APPROACH_DEG))
        far = (rb[W] > 0) & (rb[L] > 0) & (np.linalg.norm(gx[W] - gx[L], axis=1) > rb[W] + rb[L] + cap)
        fromto = np.zeros(6)
        BOX = carry._BOX
        for i in np.flatnonzero(~far):
            w, l = pairs[i]
            dist = mujoco.mj_geomDistance(m, d, w, l, cap, fromto)
            if dist >= cap:
                continue
            allow = False
            if dist == 0.0 and m.geom_type[w] == BOX == m.geom_type[l]:
                dist, n = carry._box_sat(d, m, w, l)
                if dist >= cap:
                    continue
                allow = n[2] >= cosl
            elif dist <= 0.0:
                n = None
                for c in d.contact[:d.ncon]:
                    if int(c.geom1) == w and int(c.geom2) == l:
                        n = np.array(c.frame[:3]); break
                    if int(c.geom1) == l and int(c.geom2) == w:
                        n = -np.array(c.frame[:3]); break
                if n is None and m.geom_type[w] == BOX == m.geom_type[l]:
                    _, n = carry._box_sat(d, m, w, l)
                allow = n is not None and n[2] >= cosl
            else:
                v = fromto[3:] - fromto[:3]
                allow = v[2] >= cosl * np.linalg.norm(v)
            D[i], allowed[i] = dist, allow
        return D, allowed

    def _verts(self, g):
        """vertices (k, 3) of geom g at the current state (box corners; mesh hull points), or None."""
        m, d = self.m, self.d
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            P = _SG * m.geom_size[g]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            try:
                P = CL._hull_pts(m, g)[0]
            except Exception:
                return None
        else:
            return None
        return d.geom_xpos[g] + P @ d.geom_xmat[g].reshape(3, 3).T

    def _obj_world_tilted(self, sw, tw, de):
        """obj_world over the tilt cover set (module docstring): -> (D lower bounds (P,), allowed (P,)); allowance as in _obj_world, at the untilted pose."""
        carry, m, d = self.carry, self.m, self.d
        pairs = self.pairs["obj_world"]
        cap = self.cap["obj_world"]
        D = np.full(len(pairs), cap)
        allowed = np.zeros(len(pairs), bool)
        if not pairs:
            return D, allowed
        c = d.xpos[self.obj_b] + d.xmat[self.obj_b].reshape(3, 3) @ self.piv          # pivot, nominal position
        gx, rb = d.geom_xpos, m.geom_rbound
        cosl = np.cos(np.radians(carry.SUPPORT_APPROACH_DEG))
        BOX, PLANE = carry._BOX, mujoco.mjtGeom.mjGEOM_PLANE
        vcache, bl = {}, {}
        for _, l in pairs:
            if l in bl:
                continue
            V = self._verts(l)
            vcache[l] = V
            bl[l] = (float(displacement_bound(V - c, sw, tw).max()) if V is not None else
                     (float(np.linalg.norm(gx[l] - c)) + float(rb[l])) * (2.0 * math.sin(0.5 * sw) + 2.0 * math.sin(0.5 * tw)))
        fromto = np.zeros(6)
        for i, (w, l) in enumerate(pairs):
            dmax = cap + bl[l] + de
            if rb[w] > 0 and rb[l] > 0 and np.linalg.norm(gx[w] - gx[l]) > rb[w] + rb[l] + dmax:
                continue
            dist = mujoco.mj_geomDistance(m, d, w, l, dmax, fromto)
            if dist >= dmax:
                continue
            allow, n0 = False, None
            if dist == 0.0 and m.geom_type[w] == BOX == m.geom_type[l]:
                dist, n = carry._box_sat(d, m, w, l)
                if dist >= dmax:
                    continue
                allow = n[2] >= cosl
            elif dist <= 0.0:
                n = None
                for cn in d.contact[:d.ncon]:
                    if int(cn.geom1) == w and int(cn.geom2) == l:
                        n = np.array(cn.frame[:3]); break
                    if int(cn.geom1) == l and int(cn.geom2) == w:
                        n = -np.array(cn.frame[:3]); break
                if n is None and m.geom_type[w] == BOX == m.geom_type[l]:
                    _, n = carry._box_sat(d, m, w, l)
                allow = n is not None and n[2] >= cosl
            else:
                v = fromto[3:] - fromto[:3]
                allow = v[2] >= cosl * np.linalg.norm(v)
                n0 = v / max(float(np.linalg.norm(v)), 1e-15)
            if allow and dist < cap:                         # the allowance region of the untilted row exactly (dist < cap); a pair beyond the cap gets the tilt bound
                D[i], allowed[i] = dist, True
                continue
            V = vcache[l]
            tw_ = m.geom_type[w]
            if V is not None and tw_ in (BOX, PLANE):
                Rw = d.geom_xmat[w].reshape(3, 3)
                if tw_ == PLANE:
                    N = Rw[:, 2][None, :]
                    hW = N @ gx[w]
                else:
                    Rl = d.geom_xmat[l].reshape(3, 3)
                    ax = [Rw[:, k] for k in range(3)] + ([Rl[:, k] for k in range(3)] if m.geom_type[l] == BOX else [])
                    ax += [np.cross(Rw[:, a], Rl[:, b]) for a in range(3) for b in range(3)] if m.geom_type[l] == BOX else []
                    N = np.array([x / np.linalg.norm(x) for x in ax if np.linalg.norm(x) > 1e-9])
                    N = np.vstack([N, -N] + ([n0[None, :]] if n0 is not None else []))
                    hW = N @ gx[w] + np.abs(N @ Rw) @ m.geom_size[w]
                U = V - c
                lb = np.array([float((c @ nn) + swing_twist_lb(nn, U, sw, tw).min()) for nn in N]) - hW
                D[i] = min(cap, float(lb.max()) - de)
            else:
                D[i] = min(cap, float(dist) - bl[l] - de)
        return D, allowed

    def _arrs(self, c):
        if not hasattr(self, "_arr"):
            self._arr = {}
        if c not in self._arr:
            P = self.pairs[c]
            A = np.array([a for a, _ in P], int); B = np.array([b for _, b in P], int)
            ra = lambda g: self.rad[g] if g >= 0 else 0.0
            self._arr[c] = (A, B, np.array([ra(a) for a in A]), np.array([ra(b) if b >= 0 else 0.0 for b in B]))
        return self._arr[c]

    def geom_state(self):
        return self.d.geom_xpos.copy(), self.d.geom_xmat.copy()


def _geom_speeds(sc, X, R, dt):
    """s_g(k) (n-1,) [m/s] for every geom that appears in a pair; dict g -> array."""
    gs = sc.moving
    out = {}
    for g in gs:
        dx = np.linalg.norm(np.diff(X[:, g], axis=0), axis=1)
        Rg = R[:, g].reshape(-1, 3, 3)
        tr = np.einsum("kij,kij->k", Rg[:-1], Rg[1:])               # trace(R_k^T R_{k+1})
        th = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
        out[g] = (dx + th * sc.rad[g]) / dt
    return out


def _pair_speed(sc, c, sp, nint):
    cols = []
    for a, b in sc.pairs[c]:
        sa = sp.get(a, np.zeros(nint))
        sb = sp.get(b, np.zeros(nint)) if b >= 0 else np.zeros(nint)      # a world geom / the floor does not move
        cols.append(sa + sb)
    return np.array(cols).T if cols else np.zeros((nint, 0))


def piece_distances(plan, sg, sc=None):
    """-> t (n,), dict class -> dict(d (n,), v (n,), names (n,)) for piece sg."""
    from world_model.v2.adapters.motion_rows import piece_arrays
    sc = sc or _Scene(plan)
    t, q, _, p = piece_arrays(plan, sg)
    n = len(t)
    post = sc.post_base(sg)
    tl = sc.tilt_of(sg)                                     # None when untilted; raises TiltMissing for a piece without a measurement
    Dall = {c: np.zeros((n, len(sc.pairs[c]))) for c in CLASSES}
    X = np.zeros((n, sc.m.ngeom, 3)); R = np.zeros((n, sc.m.ngeom, 9))
    for k in range(n):
        sc.set_state(q["left"][k], q["right"][k], p[k], post, tl)
        dd = sc.distances()
        for c in CLASSES:
            D, al = dd[c]
            Dall[c][k] = np.where(al, sc.cap[c], D)
        X[k], R[k] = sc.geom_state()
    dt = float(t[1] - t[0]) if n > 1 else CC.DT
    sp = _geom_speeds(sc, X, R, dt)
    res = {}
    for c in CLASSES:
        D = Dall[c]
        if D.shape[1] == 0:
            res[c] = dict(d=np.full(n, sc.cap[c]), v=np.zeros(n), names=[""] * n)
            continue
        S = _pair_speed(sc, c, sp, n - 1)
        dm_p = np.minimum(D[:-1], D[1:])
        dm = dm_p.min(1)
        smax = S.max(1)
        mask = dm_p <= (dm + smax * dt)[:, None]
        v = np.where(mask, S, 0.0).max(1)
        am = D.argmin(1)
        res[c] = dict(d=D.min(1), v=np.r_[v, v[-1] if len(v) else 0.0],
                      names=[f"{sc.nm(sc.pairs[c][i][0])}|{sc.nm(sc.pairs[c][i][1]) if sc.pairs[c][i][1] >= 0 else 'floor'}" for i in am])
    return t, res


TILT_NOTE = ("load tilt cover (USER 11-17): obj_world over {planned load pose rotated about the pivot by swing <= sw (any horizontal axis) and twist <= tw (about z), "
             "then shifted by <= de}; sw / tw / de = per-piece max over n measured episodes (tilt_measure RotTrack): guarantee SAMPLED, not a bound over the cell")
TILT_NOT_COVERED = ("tilt / drift larger than the max over n; contents moving relative to the box (rigid-load MODEL); claw / arm tracking error in the arm classes; "
                    "the base move; the allowance is judged at the untilted pose")


def tilt_assumptions(sc, scope):
    """the row's tilt assumptions (dict)."""
    t = sc.tilt
    return dict(load_tilt=dict(tag="MEASURED (sampled)", what=TILT_NOTE, scope=scope, measurement=t.to_json(), pivot="object-fixed point at the claw midpoint at k = 0 "
                               f"(pivot_obj {np.round(t.pivot_obj, 5).tolist()} m, object frame); rotation about it, drift measured at it",
                               start_pose_mismatch_m=sc.x0_mis, bound="MODEL: per-vertex swing/twist bound on SAT axes + closest-point direction (box / plane), else "
                               "d0 - max vertex displacement 2 sin(sw/2)|u| + 2 sin(tw/2)|u_h|; minus drift de", not_covered=TILT_NOT_COVERED))


def collision_rows(plan, world=None, pieces=None, classes=None, tilt=None):
    """plan: closed_chain.ClosedChainPlan (world is unused: the plan's own model / world; kept for the registry signature).  -> ONE Row ('collision'); the worst piece binds
    and is named in row.worst['piece']; row.worst['by_class'] gives margin / worst case per class for that piece's worst class and 'pieces' the margin per piece.
    tilt (USER 11-17): the per-plan TiltSpec (tilt_from_table on tilt_measure.measure_plan's table).  Default ON: no TiltSpec / an incomplete one = a MISSING row.
    BIMANUAL_LEGACY_UNTILTED_COLLISION=1: tilt is ignored, the untilted row as before."""
    from world_model.v2.stack.checks.row import missing
    if not legacy_untilted() and tilt is None:
        return missing("collision", "no per-plan load tilt measurement passed (collision_rows(plan, tilt=tilt_from_table(...))): the load-vs-world row cannot cover the tilt "
                                    "(USER 11-17); BIMANUAL_LEGACY_UNTILTED_COLLISION=1 = the untilted row")
    try:
        sc = _Scene(plan, tilt=tilt)
        for sg in (list(pieces) if pieces else list(plan.pieces)):
            sc.tilt_of(sg)
    except TiltMissing as ex:
        return missing("collision", f"load tilt measurement unusable: {ex}")
    dmin = d_min_table()
    names = list(pieces) if pieces else list(plan.pieces)
    rows = []
    for sg in names:
        t, res = piece_distances(plan, sg, sc)
        cls = [c for c in CLASSES if (classes is None or c in classes)]
        r = K.swept_clearance_row({c: res[c]["d"] for c in cls}, {c: res[c]["v"] for c in cls}, t, {c: dmin[c] for c in cls},
                                  pair_names={c: res[c]["names"] for c in cls})
        rows.append((sg, r))
    sg, r = min(rows, key=lambda x: x[1].margin)
    r.worst = dict(r.worst, piece=sg, pieces={s: x.margin for s, x in rows}, by_class_all={s: x.worst["by_class"] for s, x in rows})
    if r.why:
        r.why = f"[{sg}] " + r.why
    r.n = int(sum(x.n for _, x in rows))
    if sc.tilt is not None:
        r.assumptions = dict(r.assumptions, **tilt_assumptions(sc, "per plan piece: that piece's measured max"))
        if r.ok and r.guarantee == "proof":
            r.guarantee = "sampled"                         # the tilt / drift are a max over n episodes
    return r


def dense_min_distance(plan, sg, cls, k, upsample=20, scene=None):
    """min distance [m] of class cls (same pairs, allowance and cap as the row) over `upsample`+1 points linearly interpolated in q (both arms) and in p between samples k and k+1
    of piece sg.  For the soundness test: the swept lower bound of interval k <= this."""
    from world_model.v2.adapters.motion_rows import piece_arrays
    sc = scene or _Scene(plan)
    t, q, _, p = piece_arrays(plan, sg)
    post = sc.post_base(sg)
    best = np.inf
    for i in range(upsample + 1):
        u = i / upsample
        ql = (1 - u) * q["left"][k] + u * q["left"][k + 1]
        qr = (1 - u) * q["right"][k] + u * q["right"][k + 1]
        pp = (1 - u) * p[k] + u * p[k + 1]
        sc.set_state(ql, qr, pp, post)
        D, al = sc.distances()[cls]
        D = np.where(al, sc.cap[cls], D)
        best = min(best, float(D.min()) if len(D) else sc.cap[cls])
    return best


# ---- adaptive refinement of the swept row (USER 11-17 "Both", part 1)
SWEPT_MAX_DEPTH = 4      # ASSUMED (USER to confirm): an interval is halved at most this many times -> minimum step = interval / 2**4 = 0.625 mm of a 1 cm geometry interval (allowance / 16)


def swept_coarse():
    import os
    return os.environ.get("BIMANUAL_LEGACY_SWEPT_COARSE", "0") == "1"


def _class_lb(sc, c, Da, Db, X2, R2, h):
    """swept lower bound [m] on class c over ONE interval of length h (same pair-mask rule as the row): min(d) - v h / 2, with v = max pair speed over the pairs that can bind."""
    if Da.shape[0] == 0:
        return float(sc.cap[c])
    sp = _geom_speeds(sc, X2, R2, h)
    S = _pair_speed(sc, c, sp, 1)[0]
    dm_p = np.minimum(Da, Db)
    v = np.where(dm_p <= dm_p.min() + S.max() * h, S, 0.0).max()
    return float(dm_p.min() - v * h / 2.0)


def refine_samples(sc, args, D0, X0, R0, d_min, max_depth=None):
    """Adaptive refinement of a sample sequence.  args[k] = (ql, qr, p, post[, tilt]) of sample k (tilt = the load tilt cover (sw, tw, de), _Scene.tilt_of); D0 {class: (n, P)}, X0 (n, G, 3), R0 (n, G, 9) the states there.  An interval whose swept
    bound is below d_min while both endpoints are above it is halved (state = linear interpolation in q and p, the model the row already assumes), down to max_depth halvings
    (SWEPT_MAX_DEPTH).  Each sub-interval gets its own speed bound, so the row's bound min over sub-intervals stays a sound bound on the whole interval (same MODEL).
    Intervals across a base-move boundary (post differs) are not refined.
    -> t (m,) fractional sample index, {class: (m, P)}, X (m, G, 3), R (m, G, 9), n_added."""
    max_depth = SWEPT_MAX_DEPTH if max_depth is None else max_depth
    n = len(args)
    nodes = [dict(u=float(k), D={c: D0[c][k] for c in CLASSES}, X=X0[k], R=R0[k], a=args[k]) for k in range(n)]
    added = 0

    def tight(a, b, depth):
        if depth >= max_depth:
            return False
        h = b["u"] - a["u"]
        for c in CLASSES:
            Da, Db = a["D"][c], b["D"][c]
            if Da.shape[0] == 0 or min(Da.min(), Db.min()) < d_min[c]:      # no pairs / the endpoint itself already fails: refining cannot change the verdict
                continue
            lb = _class_lb(sc, c, Da, Db, np.stack([a["X"], b["X"]]), np.stack([a["R"], b["R"]]), h)
            if lb < d_min[c]:
                return True
        return False

    def mid(a, b):
        (ql0, qr0, p0, po0), (ql1, qr1, p1, po1) = a["a"][:4], b["a"][:4]
        ar = (0.5 * (np.asarray(ql0) + ql1), 0.5 * (np.asarray(qr0) + qr1), 0.5 * (np.asarray(p0) + p1), po0)
        t0, t1 = (a["a"][4] if len(a["a"]) > 4 else None), (b["a"][4] if len(b["a"]) > 4 else None)
        if t0 is not None or t1 is not None:            # load tilt cover: the larger of the two endpoints' (same piece = equal)
            ar = ar + (tuple(max(x, y) for x, y in zip(t0 or t1, t1 or t0)),)
        sc.set_state(*ar)
        dd = sc.distances()
        X, R = sc.geom_state()
        return dict(u=0.5 * (a["u"] + b["u"]), D={c: np.where(dd[c][1], sc.cap[c], dd[c][0]) for c in CLASSES}, X=X, R=R, a=ar)

    out = []

    def rec(a, b, depth):
        nonlocal added
        if a["a"][3] == b["a"][3] and tight(a, b, depth):
            m = mid(a, b)
            added += 1
            rec(a, m, depth + 1)
            out.append(m)
            rec(m, b, depth + 1)

    seq = []
    for k in range(n - 1):
        seq.append(nodes[k])
        out = []
        rec(nodes[k], nodes[k + 1], 0)
        seq.extend(out)
    seq.append(nodes[-1])
    return (np.array([x["u"] for x in seq]), {c: np.stack([x["D"][c] for x in seq]) for c in CLASSES},
            np.stack([x["X"] for x in seq]), np.stack([x["R"] for x in seq]), added)
