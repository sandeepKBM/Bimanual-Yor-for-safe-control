"""clearance.py -- min signed distances [m] between the robot's collision geoms and the world / the carried object / the other arm (11-03 grasp fix).

Uses mujoco.mj_geomDistance on the collision geoms (convex hulls for meshes).  Categories:
  world      robot geom vs a non-robot, non-carried collidable geom (table, counter, shelf, floor, support)
  object     robot geoms vs the carried object's geoms, EXCEPT each welded gripper (palm / pole / jaws of a side listed in `welded`) vs the object
             (that pair is switched off by weldcontact; reported separately as `own_gripper_object`, information only)
  arm_arm    left-arm geoms vs right-arm geoms
Same-arm pairs are not checked.  Threshold used by callers: >= CLEARANCE_MIN (5 mm, ASSUMED, the pan pre-grasp threshold).
"""
from __future__ import annotations

import numpy as np
import mujoco

CLEARANCE_MIN = 0.005    # ASSUMED, same as pan CLAMP clear
_DMAX = 0.10


def _robot_side(m, g):
    n = m.body(m.geom_bodyid[g]).name or ""
    return "left" if n.startswith("left_arm") else "right" if n.startswith("right_arm") else None


def _is_grip(m, g):
    n = m.body(m.geom_bodyid[g]).name or ""
    return any(k in n for k in ("gripper", "finger"))


def classify(m, obj_body: str):
    root = m.body(obj_body).id
    robot, world, obj = {"left": [], "right": []}, [], []
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        s = _robot_side(m, g)
        if s:
            robot[s].append(g)
            continue
        b = int(m.geom_bodyid[g])
        bb = b
        carried = False
        while bb != 0:
            if bb == root:
                carried = True
                break
            bb = int(m.body_parentid[bb])
        nm = m.body(b).name or ""
        if carried or nm.startswith(("cup", "contents")):
            if carried:
                obj.append(g)
            continue
        gn = (m.geom(g).name or "").lower()
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue                                                   # floor: handled in report() as hull min-z
        if any(k in gn for k in ("table", "counter", "shelf", "stove")):
            world.append(g)
    return robot, world, obj


_HULL = {}


def _hull_pts(m, g):
    """collision hull vertices of geom g in its own frame (mesh: convex hull of the mesh; box: 8 corners)."""
    key = (id(m), g)
    if key not in _HULL:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            from scipy.spatial import ConvexHull
            did = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[did]:m.mesh_vertadr[did] + m.mesh_vertnum[did]].astype(float)
            P = V[ConvexHull(V).vertices]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            h = m.geom_size[g]
            P = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * h
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, hz = m.geom_size[g][:2]
            t = np.linspace(0, 2 * np.pi, 25)[:-1]
            P = np.array([[r * np.cos(a_), r * np.sin(a_), z] for a_ in t for z in (-hz, hz)])
        else:
            raise NotImplementedError(f"geom type {m.geom_type[g]}")
        _HULL[key] = (P, float(np.linalg.norm(P, axis=1).max()))
    return _HULL[key]


def _gjk(Wa, Wb, tol=1e-10, maxit=60):
    """Gilbert distance between the convex hulls of two point sets (0.0 when they touch / overlap)."""
    from scipy.optimize import nnls
    sup = lambda dr: Wa[np.argmax(Wa @ dr)] - Wb[np.argmin(Wb @ dr)]
    v = Wa[0] - Wb[0]
    S = [v]
    for _ in range(maxit):
        nv = float(v @ v)
        if nv < 1e-12:
            return 0.0
        w = sup(-v)
        if nv - float(v @ w) < tol + 1e-9 * nv:
            break
        S.append(w)
        P = np.array(S)
        M = 1e3
        lam, _ = nnls(np.vstack([P.T, M * np.ones((1, len(P)))]), np.array([0.0, 0.0, 0.0, M]))
        keep = lam > 1e-12
        S = [S[i] for i in range(len(S)) if keep[i]]
        v = lam[keep] @ P[keep]
    return float(np.linalg.norm(v))


def _dist(m, d, a, b):
    """exact distance [m] between the convex hulls of geoms a, b (negative = penetration depth, from the Minkowski-difference hull); capped at _DMAX.
    mj_geomDistance is not used: it returned 0.0 for separated mesh / box pairs in this repo (10-29) and again in the 11-03 hold-pose check."""
    Pa, ra = _hull_pts(m, a)
    Pb, rb = _hull_pts(m, b)
    Wa = Pa @ d.geom_xmat[a].reshape(3, 3).T + d.geom_xpos[a]
    Wb = Pb @ d.geom_xmat[b].reshape(3, 3).T + d.geom_xpos[b]
    if np.linalg.norm(Wa.mean(0) - Wb.mean(0)) > ra + rb + _DMAX + 0.05:
        return _DMAX
    dist = _gjk(Wa, Wb)
    if dist > 2e-5:
        return min(dist, _DMAX)
    from scipy.spatial import ConvexHull
    try:
        D = (Wa[:, None, :] - Wb[None, :, :]).reshape(-1, 3)
        H = ConvexHull(D)
        return -float(np.min(-H.equations[:, 3]))                   # origin inside: depth = distance to the nearest facet
    except Exception:
        return 0.0


def report(m, d, obj_body="obj", welded=("left", "right"), detail=False):
    """-> dict(world, object, arm_arm, own_gripper_object, min) [m]; with detail=True also the geom-name pairs of each minimum."""
    mujoco.mj_forward(m, d)
    robot, world, obj = classify(m, obj_body)
    out = dict(world=(_DMAX, None), object=(_DMAX, None), arm_arm=(_DMAX, None), own_gripper_object=(_DMAX, None))
    nm = lambda g: m.geom(g).name or f"g{g}:{m.body(m.geom_bodyid[g]).name}"
    for s, gs in robot.items():
        for g in gs:
            Pg, _ = _hull_pts(m, g)
            zmin = float((Pg @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
            if zmin < out["world"][0]:
                out["world"] = (zmin, (nm(g), "floor"))
            for w in world:
                v = _dist(m, d, g, w)
                if v < out["world"][0]:
                    out["world"] = (v, (nm(g), nm(w)))
            for o in obj:
                v = _dist(m, d, g, o)
                key = "own_gripper_object" if (s in welded and _is_grip(m, g)) else "object"
                if v < out[key][0]:
                    out[key] = (v, (nm(g), nm(o)))
    for g in robot["left"]:
        for h in robot["right"]:
            v = _dist(m, d, g, h)
            if v < out["arm_arm"][0]:
                out["arm_arm"] = (v, (nm(g), nm(h)))
    res = {k: v[0] for k, v in out.items()}
    res["min"] = min(res["world"], res["object"], res["arm_arm"])
    if detail:
        res["pairs"] = {k: v[1] for k, v in out.items()}
    return res
