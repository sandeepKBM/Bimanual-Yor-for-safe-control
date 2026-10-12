"""Task 4: bimanual box. A box (0.20 x 0.15 x 0.15 m, 0.5 kg) is held between two parallel gripper pads (kinematic mocap bodies, friction 1);
the pair frame is rotated (roll phi, yaw psi) and the squeeze of each pad is randomised in [-12 mm gap, +3 mm penetration], so the box is
held, tilted, sagging or dropping. MuJoCo 3.9 states at t = 0.2/0.3/0.4 s.
spec:  contact(pad_L, box, d_c) AND contact(pad_R, box, d_c) AND between(box_com, pad_L, pad_R) AND upright(box, th)
hand-coded reference: contact margin = d_c - geom_ref.box_box_sd(pad, box) (SAT overlap / vertex+edge distance; MuJoCo's mj_geomDistance
(recorded as extra dL, dR) is only a cross-check: it returns exactly 0.0 for some separated pairs);
between margin = point-in-zonotope margin of hull(pad_L u pad_R) = pad_L (+) segment (equal orientations) via generator-pair facets;
level margin = th - arccos(box z).
NEW atoms: contact (ContactKept), between (Between). v0 can only write the contact as Not(Clear) (margin fine, rows -> NotConvex)."""
from __future__ import annotations

import os

import mujoco
import numpy as np

from world_model.v2.tray_rep.sets.ir import BoxShape, Point
from world_model.v2.constraints import ir_ext
from world_model.v2.constraints import geom_ref
from world_model.v2.constraints.common import PoseState, Task, ent
from world_model.v2.constraints.tasks.t2_cup import quat_R

BOX_H = (0.10, 0.075, 0.075)
PAD_H = (0.03, 0.005, 0.03)
D_C, TH4 = 0.003, 0.35
DT_ = 0.05
W = 2 * (BOX_H[1] + PAD_H[1])
SPEC = {"and": [{"contact": {"a": "pad_L", "b": "box", "d": "d_c", "tag": "contactL"}},
                {"contact": {"a": "pad_R", "b": "box", "d": "d_c", "tag": "contactR"}},
                {"between": {"a": "box_c", "b": "pad_L", "c": "pad_R", "margin": 0.0}},
                {"upright": {"a": "box", "theta_max": "th", "tag": "level"}}]}
CACHE = "/common/users/ss5772/bimanual_data/constraints_v1/tmp/t4_states.npy"

XML = f"""<mujoco><option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="5"/>
<default><geom friction="1.0 0.01 0.001" condim="3"/></default>
<worldbody>
<body name="box" pos="0 0 0.5"><freejoint/><geom name="gbox" type="box" size="{BOX_H[0]} {BOX_H[1]} {BOX_H[2]}" mass="0.5"/></body>
<body name="padL" mocap="true" pos="0 -0.2 0.5"><geom name="gL" type="box" size="{PAD_H[0]} {PAD_H[1]} {PAD_H[2]}"/></body>
<body name="padR" mocap="true" pos="0 0.2 0.5"><geom name="gR" type="box" size="{PAD_H[0]} {PAD_H[1]} {PAD_H[2]}"/></body>
</worldbody></mujoco>"""


def rot_xyz(phi, psi):
    cx, sx, cz, sz = np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx


def R_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def simulate(n=110, seed=0, T=(100, 150, 200), ramp=50):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    gb, gl, gr = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, x) for x in ("gbox", "gL", "gR"))
    rows = []
    for _ in range(n):
        phi, psi = rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3)
        R = rot_xyz(phi, psi)
        c0 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0.6])
        held = rng.random() < 0.6
        sq = rng.uniform(0.001, 0.005, 2) if held else rng.uniform(-0.012, 0.005, 2)      # +: penetration, -: gap
        dz = rng.uniform(-0.02, 0.02) if held else rng.uniform(-0.03, 0.03)               # relative height of the pads (pair frame)
        dx = rng.uniform(-0.02, 0.02) if held else rng.uniform(-0.04, 0.04)
        gap0 = 0.2
        yl, yr = -(BOX_H[1] + PAD_H[1] - sq[0]), (BOX_H[1] + PAD_H[1] - sq[1])
        tgtL, tgtR = c0 + R @ np.array([dx, yl, dz]), c0 + R @ np.array([dx, yr, 0.0])
        stL, stR = c0 + R @ np.array([dx, -gap0, dz]), c0 + R @ np.array([dx, gap0, 0.0])
        mujoco.mj_resetData(m, d)
        d.qpos[0:3], d.qpos[3:7] = c0, R_to_quat(R)
        q = R_to_quat(R)
        d.mocap_quat[0], d.mocap_quat[1] = q, q
        for k in range(max(T) + 1):
            a = min(k / ramp, 1.0)
            m.opt.gravity[2] = 0.0 if k < ramp else -9.81                    # the box hangs in place until the pads close
            d.mocap_pos[0], d.mocap_pos[1] = stL + a * (tgtL - stL), stR + a * (tgtR - stR)
            mujoco.mj_step(m, d)
            if k in T:
                mujoco.mj_forward(m, d)
                ft = np.zeros(6)
                dL = mujoco.mj_geomDistance(m, d, gl, gb, 1.0, ft)
                dR = mujoco.mj_geomDistance(m, d, gr, gb, 1.0, ft)
                rows.append(np.concatenate([d.qpos[0:7], d.mocap_pos[0], d.mocap_pos[1], q, [dL, dR]]))
    return np.array(rows)


def sample(n, rng):
    if not os.path.exists(CACHE):
        np.save(CACHE, simulate())
    A = np.load(CACHE)
    S = []
    for r in A:
        Rb = quat_R(r[3:7])
        Rp = quat_R(r[13:17])
        poses = {"box": (Rb, r[0:3]), "box_c": (Rb, r[0:3]), "pad_L": (Rp, r[7:10]), "pad_R": (Rp, r[10:13])}
        S.append(PoseState(poses, extra=dict(dL=r[17], dR=r[18])))
    idx = rng.permutation(len(S))[:n]
    return [S[i] for i in idx]


def build():
    ents = {"box": ent("box", BoxShape(BOX_H)), "box_c": ent("box_c", Point()), "pad_L": ent("pad_L", BoxShape(PAD_H)),
            "pad_R": ent("pad_R", BoxShape(PAD_H))}
    return ir_ext.from_dict(SPEC, ents, dict(d_c=D_C, th=TH4)), ents


def between_ref(s):
    """pads share one orientation: hull = pad_L (+) segment(0, pR - pL); point-in-zonotope margin by generator-pair facets."""
    from world_model.v2.tray_rep.sets.zono import Zonotope
    Rp, pL = s.poses["pad_L"]
    pR = s.poses["pad_R"][1]
    G = np.concatenate([Rp @ np.diag(PAD_H), ((pR - pL) / 2)[:, None]], 1)
    Z = Zonotope((pL + pR) / 2, G)
    N, h = ir_ext.zono3_facets(Z)
    x = s.poses["box_c"][1]
    return float((h - N @ (x - Z.c)).min())


def dists(s):
    Rb, pb = s.poses["box"]
    return [geom_ref.box_box_sd(s.poses[n][0], s.poses[n][1], np.array(PAD_H), Rb, pb, np.array(BOX_H)) for n in ("pad_L", "pad_R")]


def parts(s):
    tilt = np.arccos(np.clip(s.poses["box"][0][2, 2], -1, 1))
    dL, dR = dists(s)
    return dict(contactL=D_C - dL, contactR=D_C - dR, between=between_ref(s), level=TH4 - tilt)


def ref(s):
    return float(min(parts(s).values()))


def toy(s, rng):
    """u = [vL(3), vR(3)] over dt: pads move by dt v; the held box moves with their mean; roll about x from the height difference."""
    I = np.eye(3)
    JL = np.concatenate([I, 0 * I], 1) * DT_
    JR = np.concatenate([0 * I, I], 1) * DT_
    Jb = 0.5 * (JL + JR)
    Jc = np.zeros((2, 6))
    Jc[1, 2], Jc[1, 5] = DT_ / W, -DT_ / W
    return {"u0": rng.normal(size=6) * 0.3, "pad_L": {"p": JL}, "pad_R": {"p": JR}, "box": {"p": Jb, "axis": Jc}, "box_c": {"p": Jb}}


def toy_norot(s, rng):
    j = toy(s, rng)
    j["box"] = {"p": j["box"]["p"], "axis": np.zeros((2, 6))}      # box does not rotate (translation-only coupling)
    return j


def make():
    return Task("bimanual_box", SPEC, build, sample, ref, toy, toy_norot,
                notes=dict(atoms=["contact(new)", "between(new)", "upright", "and"], nonconvex=["between: hull normals depend on both pads"]))
