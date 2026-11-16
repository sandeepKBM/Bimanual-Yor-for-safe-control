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

Constants: obj_world d_min = carry.U_OBJ_CLEAR_M (5 mm, ASSUMED, USER 11-16 D9).  arm_arm / arm_world d_min = clearance.CLEARANCE_MIN (5 mm, ASSUMED, "same as pan CLAMP clear"; reused, no new constant).
Not covered (stated): arm links vs the carried object (non-gripper), self-collision of one arm, the base / lift column vs arms, anything between samples that the speed model misses.
"""
from __future__ import annotations

import numpy as np
import mujoco

from world_model.v2.nero import clearance as CL
from world_model.v2.stack.checks import collision as K
from world_model.v2.stack.planners import closed_chain as CC

SIDES = CC.SIDES
CLASSES = K.CLASSES


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

    def __init__(self, plan):
        from world_model.v2.nero.contents import carry
        self.carry = carry
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

    def set_state(self, ql, qr, p, post=False):
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
    Dall = {c: np.zeros((n, len(sc.pairs[c]))) for c in CLASSES}
    X = np.zeros((n, sc.m.ngeom, 3)); R = np.zeros((n, sc.m.ngeom, 9))
    for k in range(n):
        sc.set_state(q["left"][k], q["right"][k], p[k], post)
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


def collision_rows(plan, world=None, pieces=None, classes=None):
    """plan: closed_chain.ClosedChainPlan (world is unused: the plan's own model / world; kept for the registry signature).  -> ONE Row ('collision'); the worst piece binds
    and is named in row.worst['piece']; row.worst['by_class'] gives margin / worst case per class for that piece's worst class and 'pieces' the margin per piece."""
    sc = _Scene(plan)
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
