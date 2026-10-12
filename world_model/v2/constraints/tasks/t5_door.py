"""Task 5: a tray (0.30 x 0.20 x 0.01 m) passes a door frame with clearance d. Static obstacles: post_L, post_R (0.08 x 0.10 x 1.4 m at y = +-0.39)
and a lintel (0.10 x 0.86 x 0.10 at z = 1.35); opening 0.70 m wide, 1.3 m high. Tray pose sampled near the doorway (position, yaw, roll, pitch).
spec A (obstacle form):  clear(tray, post_L, d) AND clear(tray, post_R, d) AND clear(tray, lintel, d)       [v0 clear and ClearX]
spec B (aperture form):  inside(tray, aperture) with the aperture the prism |y| <= w/2 - d, z <= z_l - d (convex, valid while the tray is in the door slab)
hand-coded reference A: geom_ref.box_box_sd (SAT overlap / vertex+edge distance) minus d, min over the three obstacles; MuJoCo's
mj_geomDistance is a cross-check only (it returns exactly 0.0 for some separated box pairs, see run_all).
hand-coded reference B: vertex enumeration of the tray: min over its 8 corners of (w/2 - d - |y|, z_l - d - z)."""
from __future__ import annotations

import mujoco
import numpy as np

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import BoxShape, Region
from world_model.v2.tray_rep.sets.zono import Polytope
from world_model.v2.constraints import ir_ext
from world_model.v2.constraints import geom_ref
from world_model.v2.constraints.common import PoseState, Task, ent
from world_model.v2.constraints.tasks.t4_bibox import rot_xyz

TRAY_H = (0.15, 0.10, 0.005)
POST_H = (0.04, 0.05, 0.70)
POST_C = {"post_L": np.array([0.0, -0.39, 0.70]), "post_R": np.array([0.0, 0.39, 0.70])}
LINTEL_H, LINTEL_C = (0.05, 0.43, 0.05), np.array([0.0, 0.0, 1.35])
W_OPEN, Z_L = 0.70, 1.30
D_CLR = 0.02
SPEC_A = {"and": [{"clear": {"a": "tray", "b": o, "d": "d", "tag": o}} for o in ("post_L", "post_R", "lintel")]}
SPEC_AX = {"and": [{"clear_x": {"a": "tray", "b": o, "d": "d", "tag": o}} for o in ("post_L", "post_R", "lintel")]}
SPEC_B = {"inside": {"a": "tray", "b": "aperture", "margin": 0.0, "tag": "aperture"}}
XML = f"""<mujoco><worldbody>
<geom name="post_L" type="box" size="{POST_H[0]} {POST_H[1]} {POST_H[2]}" pos="{' '.join(map(str, POST_C['post_L']))}"/>
<geom name="post_R" type="box" size="{POST_H[0]} {POST_H[1]} {POST_H[2]}" pos="{' '.join(map(str, POST_C['post_R']))}"/>
<geom name="lintel" type="box" size="{' '.join(map(str, LINTEL_H))}" pos="{' '.join(map(str, LINTEL_C))}"/>
<body name="tray" pos="0 0 1"><freejoint/><geom name="gtray" type="box" size="{' '.join(map(str, TRAY_H))}" mass="1"/></body>
</worldbody></mujoco>"""
_M = mujoco.MjModel.from_xml_string(XML)
_D = mujoco.MjData(_M)
_G = {n: mujoco.mj_name2id(_M, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("post_L", "post_R", "lintel", "gtray")}


def R_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def build(spec=SPEC_A):
    def b():
        ents = {"tray": ent("tray", BoxShape(TRAY_H), parent="aperture" if "inside" in spec else None), "post_L": ent("post_L", BoxShape(POST_H)), "post_R": ent("post_R", BoxShape(POST_H)),
                "lintel": ent("lintel", BoxShape(LINTEL_H)),
                "aperture": ent("aperture", Region(Polytope(np.array([[0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0]]),
                                                            np.array([W_OPEN / 2, W_OPEN / 2, Z_L]), orthogonal=True)))}
        return ir_ext.from_dict(spec, ents, dict(d=D_CLR)), ents
    return b


def sample_states(n, rng, in_slab=False):
    S = []
    for _ in range(n):
        R = rot_xyz(rng.uniform(-0.3, 0.3), 0.0)
        R = R @ _ry(rng.uniform(-0.3, 0.3))
        R = _rz(rng.uniform(-0.8, 0.8)) @ R
        x = rng.uniform(-0.04, 0.04) if in_slab else rng.uniform(-0.45, 0.45)
        p = np.array([x, rng.uniform(-0.38, 0.38), rng.uniform(0.95, 1.32)])
        poses = {"tray": (R, p), "post_L": (np.eye(3), POST_C["post_L"]), "post_R": (np.eye(3), POST_C["post_R"]),
                 "lintel": (np.eye(3), LINTEL_C), "aperture": (np.eye(3), np.zeros(3))}
        S.append(PoseState(poses))
    return S


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mj_dist(s):
    R, p = s.poses["tray"]
    _D.qpos[0:3], _D.qpos[3:7] = p, R_to_quat(R)
    mujoco.mj_forward(_M, _D)
    ft = np.zeros(6)
    return {o: mujoco.mj_geomDistance(_M, _D, _G["gtray"], _G[o], 3.0, ft) for o in ("post_L", "post_R", "lintel")}


def geo_dist(s):
    R, p = s.poses["tray"]
    out = {}
    for o, (c, h) in {"post_L": (POST_C["post_L"], POST_H), "post_R": (POST_C["post_R"], POST_H), "lintel": (LINTEL_C, LINTEL_H)}.items():
        out[o] = geom_ref.box_box_sd(R, p, np.array(TRAY_H), np.eye(3), c, np.array(h))
    return out


def ref_A(s):
    return float(min(geo_dist(s).values()) - D_CLR)


def ref_B(s):
    R, p = s.poses["tray"]
    V = np.array([R @ (np.array([a, b, c]) * TRAY_H) + p for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    return float(min((W_OPEN / 2 - np.abs(V[:, 1])).min(), (Z_L - V[:, 2]).min()))


def toy(s, rng):
    """u = [v(3), w(3)] over dt = 0.1: the tray translates by dt v and rotates (axis rate w x c)."""
    dt = 0.1
    Jp = np.concatenate([np.eye(3), 0 * np.eye(3)], 1) * dt
    Jc = np.zeros((2, 6))
    Jc[0, 4], Jc[1, 3] = dt, -dt
    return {"u0": rng.normal(size=6) * 0.3, "tray": {"p": Jp, "axis": Jc}}


def toy_norot(s, rng):
    j = toy(s, rng)
    j["tray"] = {"p": j["tray"]["p"]}
    return j


def sample(n, rng):
    return sample_states(n, rng)


def make(kind="A"):
    spec = {"A": SPEC_A, "AX": SPEC_AX, "B": SPEC_B}[kind]
    return Task({"A": "door_clear_v0", "AX": "door_clear_exact", "B": "door_aperture"}[kind], spec, build(spec),
                (lambda n, rng: sample_states(n, rng, in_slab=(kind == "B"))), ref_B if kind == "B" else ref_A, toy, toy_norot,
                notes=dict(atoms=["clear" if kind == "A" else ("clear_x(new)" if kind == "AX" else "inside"), "and"],
                           nonconvex=["clear: complement of a convex set; rows pick the separating side at u0"] if kind != "B" else []))
