"""ocs/util.py -- small geometry helpers shared by producers / consumers / dynamics.  SIMULATION ONLY."""
from __future__ import annotations

import math

import numpy as np

from world_model.v2.tray_rep.sets.ir import Pose, Unc
from world_model.v2.tray_rep.sets.zono import Zonotope, sphere_outer
from world_model.v2.tray_rep.sets.tray_spec import quat_to_R   # noqa: F401  (re-export)

PERSON_R, PERSON_H = 0.22, 1.7          # torso capsule radius / height (teleop.scene.PERSON_R; height = the capsule's visible extent)
# Cup dimensions are NOT constants here: see cup_dims() -- the cup body origin is the cup BASE centre.
HX, HY = 0.15, 0.10                     # tray half sizes (testbeds.b_tray_ball.tray_env.HX/HY)


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_pose(x: float, y: float, yaw: float) -> Pose:
    return Pose(rz(yaw), np.array([x, y, 0.0]))


def yaw_of(R) -> float:
    R = np.asarray(R)
    return float(math.atan2(R[1, 0], R[0, 0]))


def R_to_quat(R) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.asarray(R, float).reshape(9)))
    return q


def angle_between(Ra, Rb) -> float:
    """rotation angle of Ra^T Rb [rad]."""
    M = np.asarray(Ra).T @ np.asarray(Rb)
    return float(math.acos(max(-1.0, min(1.0, (np.trace(M) - 1.0) / 2.0))))


def tilt_of(R) -> float:
    """angle of the body z axis from the world up vector [rad]."""
    return float(math.acos(max(-1.0, min(1.0, float(np.asarray(R)[2, 2])))))


def box_zono(half) -> Zonotope:
    half = np.asarray(half, float)
    return Zonotope(np.zeros(3), np.diag(half))


_UNIT_DISC: dict = {}


def _unit_disc(n_gen: int) -> np.ndarray:
    """generators (2, n_gen) of the circumscribed 2n-gon of the UNIT disc (zono.sphere_outer is a Nelder-Mead search: computed once per n_gen)."""
    if n_gen not in _UNIT_DISC:
        _UNIT_DISC[n_gen] = sphere_outer(2, 1.0, n_gen).G.copy()
    return _UNIT_DISC[n_gen]


def disc_zono(r: float, n_gen: int = 4) -> Zonotope:
    """zero-centred 3-D zonotope that CONTAINS the horizontal disc of radius r (z extent 0): circumscribed 2n-gon."""
    r = float(r)
    if r <= 0:
        return Zonotope(np.zeros(3), np.zeros((3, 0)))
    g = _unit_disc(n_gen)
    G = np.zeros((3, g.shape[1]))
    G[:2] = r * g
    return Zonotope(np.zeros(3), G)


def ellipse_zono(P2, k: float = 2.0, n_gen: int = 4) -> Zonotope:
    """zero-centred 3-D zonotope containing the horizontal ellipse {x : x' P2^-1 x <= k^2} (P2 = 2x2 covariance)."""
    P2 = np.asarray(P2, float)
    w, V = np.linalg.eigh(0.5 * (P2 + P2.T))
    L = V * np.sqrt(np.maximum(w, 0.0))[None, :]
    g = _unit_disc(n_gen)
    G = np.zeros((3, g.shape[1]))
    G[:2] = k * (L @ g)
    return Zonotope(np.zeros(3), G)


def ball_zono(r: float) -> Zonotope:
    """zero-centred 3-D zonotope that contains the ball of radius r (axis-aligned cube: 3 generators, cheap and monotone in r)."""
    return Zonotope(np.zeros(3), np.eye(3) * float(r))


def _cyl(r, h):
    D = disc_zono(r, 4)
    G = np.concatenate([D.G, np.array([[0.0], [0.0], [h / 2.0]])], 1)
    return Zonotope(np.array([0.0, 0.0, h / 2.0]), G)


_SHAPES: dict = {}


def person_shape(r: float = PERSON_R, h: float = PERSON_H) -> Zonotope:
    """vertical cylinder (torso capsule) as a zonotope in the person's local frame (origin on the floor under the axis)."""
    if ("p", r, h) not in _SHAPES:
        _SHAPES[("p", r, h)] = _cyl(r, h)
    return _SHAPES[("p", r, h)]


_CUP_DIMS: dict = {}


def _sim_cup_dims() -> tuple:
    """(r, h) of the teleop-sim cup measured on the COMPILED MJCF cup body (data.contents cup body = what teleop.scene._cup_xml instantiates, in the h5 AND
    the flagship TeleopSim worlds): r = max horizontal extent of any cup geom from the body axis (the 12 wall boxes are a circumscribing 12-gon), h = max z."""
    import mujoco
    from world_model.v2.data import contents as C
    cup = C._extract_body(C.SCENE.read_text(), "cup").replace('material="cup_mat"', 'rgba="0.85 0.15 0.15 0.95"')
    m = mujoco.MjModel.from_xml_string(f'<mujoco model="cup_only"><worldbody>{cup}</worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b = m.body("cup").id
    rb, Rb = d.xpos[b], d.xmat[b].reshape(3, 3)
    r = h = 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        s, Rg, c = m.geom_size[g], d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            pts = [c + Rg @ (np.array([a, b_, c_]) * s) for a in (-1, 1) for b_ in (-1, 1) for c_ in (-1, 1)]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            pts = [c + Rg @ np.array([s[0] * math.cos(t), s[0] * math.sin(t), z * s[1]]) for t in np.linspace(0.0, 2 * math.pi, 48, endpoint=False) for z in (-1, 1)]
        else:
            raise ValueError("unexpected cup geom type")
        for p in pts:
            l = Rb.T @ (p - rb)
            r, h = max(r, math.hypot(l[0], l[1])), max(h, float(l[2]))
    return float(r), float(h)


def cup_dims(kind: str = "flagship") -> tuple:
    """(r, h) [m] of the cup of `kind`, read from its source (cached):
         "sim"      -- the cup the teleop MuJoCo sim actually contains (h5 carry recordings): measured on the compiled MJCF geometry.
         "flagship" -- the flagship paper-cup perception model (flagship.perception.CUP_R/CUP_H, the cup ring of its stereo payload hull).
    The default stays "flagship" (the historical ocs.util value) so callers that never pass a kind (stereo/, exp_e1) are unchanged."""
    if kind not in _CUP_DIMS:
        if kind == "sim":
            _CUP_DIMS[kind] = _sim_cup_dims()
        elif kind == "flagship":
            from world_model.v2.house.flagship import perception as FP
            _CUP_DIMS[kind] = (float(FP.CUP_R), float(FP.CUP_H))
        else:
            raise ValueError(f"unknown cup kind {kind!r}")
    return _CUP_DIMS[kind]


def cup_kind_for_scene(scene: str) -> str:
    """cup kind of a recording's meta['scene']: h5 -> the sim cup; flagship -> the flagship paper cup (env OCS_FLAGSHIP_CUP=sim switches the flagship
    recording to the sim cup it was physically recorded with -- see exp_e234 --flagship-cup)."""
    import os
    if scene == "h5":
        return "sim"
    if scene == "flagship":
        return os.environ.get("OCS_FLAGSHIP_CUP", "flagship")
    raise ValueError(f"unknown scene {scene!r}")


def cup_shape(kind: str = "flagship", r: float | None = None, h: float | None = None) -> Zonotope:
    """cup BODY (origin at the base centre) as a vertical-cylinder zonotope; r / h default to cup_dims(kind)."""
    r0, h0 = cup_dims(kind)
    r, h = (r0 if r is None else r), (h0 if h is None else h)
    if ("c", r, h) not in _SHAPES:
        _SHAPES[("c", r, h)] = _cyl(r, h)
    return _SHAPES[("c", r, h)]


def __getattr__(name):          # back-compat for external `U.CUP_R / U.CUP_H` readers: the flagship paper cup, resolved lazily
    if name in ("CUP_R", "CUP_H"):
        return cup_dims("flagship")[0 if name == "CUP_R" else 1]
    raise AttributeError(name)


def pos_unc(radius=None, box=None, disc=None, rot: float = 0.0) -> Unc:
    """Unc with a position set: radius -> cube(3), disc -> horizontal disc, box -> (3,) half sizes; none -> exact."""
    if box is not None:
        return Unc(box_zono(box), float(rot))
    if disc is not None:
        return Unc(disc_zono(disc), float(rot))
    if radius is not None:
        return Unc(ball_zono(radius), float(rot))
    return Unc(None, float(rot))


def unc_add(a: Unc, b: Unc) -> Unc:
    """Minkowski sum of two uncertainty sets (position zonotopes add, rotation bounds add)."""
    if a.pos is None:
        pos = b.pos
    elif b.pos is None:
        pos = a.pos
    else:
        pos = a.pos.minkowski(b.pos)
    return Unc(pos, float(a.rot) + float(b.rot))


def unc_support(u: Unc, d) -> float:
    """support function of the position set along d (3,) (0 for exact)."""
    return 0.0 if u.pos is None else float(u.pos.support(np.asarray(d, float)) - np.dot(u.pos.c, d))


def in_zonotope(c, G, x, dims=None, tol=1e-9) -> bool:
    """x in c + G [-1,1]^m ?  (exact: LP, minimise the inf-norm of the coefficients).  dims: restrict to these coordinates (e.g. (0, 1))."""
    from scipy.optimize import linprog
    c, G, x = np.asarray(c, float), np.asarray(G, float), np.asarray(x, float)
    if dims is not None:
        c, G, x = c[list(dims)], G[list(dims)], x[list(dims)]
    m = G.shape[1]
    r = x - c
    if m == 0:
        return bool(np.linalg.norm(r) <= tol)
    # variables [s (m), t]; min t s.t. G s = r, -t <= s_i <= t
    A_eq = np.concatenate([G, np.zeros((G.shape[0], 1))], 1)
    A_ub = np.concatenate([np.concatenate([np.eye(m), -np.ones((m, 1))], 1), np.concatenate([-np.eye(m), -np.ones((m, 1))], 1)], 0)
    res = linprog(np.r_[np.zeros(m), 1.0], A_ub=A_ub, b_ub=np.zeros(2 * m), A_eq=A_eq, b_eq=r, bounds=[(None, None)] * m + [(0, None)], method="highs")
    if res.status != 0:
        return False                           # infeasible: r not in range(G)
    return bool(res.x[-1] <= 1.0 + tol)
