"""urdf_fk.py -- minimal URDF parser + independent numpy forward kinematics (no MuJoCo).  Used as the FK oracle."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
import numpy as np

from . import config as C


def rpy_to_R(r, p, y):
    """URDF convention: R = Rz(y) Ry(p) Rx(r) (fixed-axis roll-pitch-yaw)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_quat(R):
    """rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _vec(s, default="0 0 0"):
    return np.array([float(x) for x in (s or default).split()])


@dataclass
class Link:
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))    # about COM, in link frame
    inertial_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coll_mesh: str | None = None
    coll_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coll_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    mimic: tuple | None = None       # (joint, multiplier, offset)

    @property
    def T(self):
        T = np.eye(4)
        T[:3, :3] = rpy_to_R(*self.rpy)
        T[:3, 3] = self.xyz
        return T


def parse(paths=None):
    """-> (links: dict, joints: dict) merged over the arm urdf + flange + gripper xacro files."""
    paths = paths or [C.URDF_ARM, C.URDF_FLANGE, C.URDF_GRIPPER]
    links, joints = {}, {}
    for p in paths:
        root = ET.parse(str(p)).getroot()
        for el in root:
            tag = el.tag.split("}")[-1]
            if tag == "link":
                L = Link(el.get("name"))
                ine = el.find("inertial")
                if ine is not None:
                    o = ine.find("origin")
                    L.com = _vec(o.get("xyz")) if o is not None else np.zeros(3)
                    L.inertial_rpy = _vec(o.get("rpy")) if o is not None else np.zeros(3)
                    L.mass = float(ine.find("mass").get("value"))
                    i = ine.find("inertia")
                    g = {k: float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
                    L.inertia = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]],
                                          [g["ixz"], g["iyz"], g["izz"]]])
                col = el.find("collision")
                if col is not None:
                    o = col.find("origin")
                    L.coll_xyz = _vec(o.get("xyz")) if o is not None else np.zeros(3)
                    L.coll_rpy = _vec(o.get("rpy")) if o is not None else np.zeros(3)
                    L.coll_mesh = col.find("geometry/mesh").get("filename")
                links[L.name] = L
            elif tag == "joint":
                o = el.find("origin")
                lim = el.find("limit")
                mim = el.find("mimic")
                joints[el.get("name")] = Joint(
                    name=el.get("name"), type=el.get("type"), parent=el.find("parent").get("link"),
                    child=el.find("child").get("link"),
                    xyz=_vec(o.get("xyz")) if o is not None else np.zeros(3),
                    rpy=_vec(o.get("rpy")) if o is not None else np.zeros(3),
                    axis=_vec(el.find("axis").get("xyz")) if el.find("axis") is not None else np.array([0., 0., 1.]),
                    lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else 0.0,
                    upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else 0.0,
                    effort=float(lim.get("effort")) if lim is not None and lim.get("effort") else 0.0,
                    velocity=float(lim.get("velocity")) if lim is not None and lim.get("velocity") else 0.0,
                    mimic=(mim.get("joint"), float(mim.get("multiplier")), float(mim.get("offset"))) if mim is not None else None)
    return links, joints


def _axis_angle_T(axis, q, kind):
    T = np.eye(4)
    if kind == "revolute":
        a = axis / np.linalg.norm(axis)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        T[:3, :3] = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    elif kind == "prismatic":
        T[:3, 3] = axis * q
    return T


def fk(joints, qdict, root="base_link"):
    """world pose (4x4) of every link reachable from `root` given {joint_name: q} (missing -> 0)."""
    poses = {root: np.eye(4)}
    todo = True
    while todo:
        todo = False
        for j in joints.values():
            if j.parent in poses and j.child not in poses:
                T = poses[j.parent] @ j.T
                if j.type in ("revolute", "prismatic"):
                    T = T @ _axis_angle_T(j.axis, qdict.get(j.name, 0.0), j.type)
                poses[j.child] = T
                todo = True
    return poses
