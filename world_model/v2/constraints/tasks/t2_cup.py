"""Task 2: cup carried upright with contents (8 beads, bead_rep.bead_env). No spill = every bead centre inside the cup volume
(the env's out_mask: rad <= R, -2 mm <= z <= H) AND cup tilt <= THETA2.
spec (dict): and( forall b in beads: inside_cyl(b, cup_volume), upright(cup, THETA2) )
NEW grammar used: Cylinder shape + inside_cyl atom, forall quantifier (ir_ext).  v0 alone: the cylinder must be approximated by a
box (inscribed square) or an N-gon prism Region -> the margin is then NOT the cup's (reported in the notes by run_all)."""
from __future__ import annotations

import os

import numpy as np

from world_model.v2.bead_rep import bead_env as B
from world_model.v2.data import contents as C
from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import Point
from world_model.v2.constraints import ir_ext
from world_model.v2.constraints.common import PoseState, Task, ent, min_rot

R_, H_, ZM = C.CUP_RADIUS, C.CUP_HEIGHT, C.H_MARGIN
THETA2 = 0.5
NB = 8
BEADS = [f"bead{i}" for i in range(NB)]
SPEC = {"and": [
    {"forall": {"var": "b", "in": BEADS, "of": {"inside_cyl": {"a": "$b", "b": "cup_volume", "margin": 0.0}}}},
    {"upright": {"a": "cup", "theta_max": "theta_max", "tag": "tilt"}}]}
CACHE = "/common/users/ss5772/bimanual_data/constraints_v1/tmp/t2_states.npz"


def build():
    ents = {"cup": ent("cup", ir_ext.Cylinder(R_, -ZM, H_)), "cup_volume": ent("cup_volume", ir_ext.Cylinder(R_, -ZM, H_))}
    for b in BEADS:
        ents[b] = ent(b, Point(), parent="cup_volume")
    return ir_ext.from_dict(SPEC, ents, dict(theta_max=THETA2)), ents


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def simulate(n_eps=18, ticks=110, every=3, seed=0):
    """MuJoCo bead-cup episodes under random held fallback-set actions (half of them aggressive: sustained tilt + accel)."""
    rng = np.random.default_rng(seed)
    U = B.fallback_U()
    out = []
    for ep in range(n_eps):
        env = B.BeadEnv()
        env.reset(np.array([0, 0, 1.0]), np.array([1.0, 0, 0, 0]))
        aggressive = ep % 2 == 1
        u, hold = U[0], 0
        for t in range(ticks):
            if hold == 0:
                u = U[rng.integers(len(U))]
                if aggressive:
                    big = U[np.argsort(-np.abs(U).sum(1))[:20]]
                    u = big[rng.integers(len(big))]
                hold = rng.integers(8, 25)
            hold -= 1
            env.tick(u)
            if t % every == 0:
                out.append((env.pos.copy(), env.quat.copy(), env.local().copy()))
    return out


def states_from(sims):
    S = []
    for pos, quat, loc in sims:
        poses = {"cup": (quat_R(quat), pos), "cup_volume": (quat_R(quat), pos)}
        for i, b in enumerate(BEADS):
            poses[b] = (np.eye(3), loc[i])
        S.append(PoseState(poses))
    return S


def sample(n, rng):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        sims = list(zip(z["pos"], z["quat"], z["loc"]))
    else:
        sims = simulate()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, pos=np.array([s[0] for s in sims]), quat=np.array([s[1] for s in sims]), loc=np.array([s[2] for s in sims]))
    S = states_from(sims)
    idx = rng.permutation(len(S))[:n]
    return [S[i] for i in idx]


def cyl_sdf(loc):
    """independent: signed distance (inside > 0) of points to the solid cylinder via the 2-D rectangle SDF in (rad, z)."""
    rad = np.hypot(loc[:, 0], loc[:, 1])
    qx, qz = rad - R_, np.abs(loc[:, 2] - 0.5 * (H_ - ZM)) - 0.5 * (H_ + ZM)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0))
    inside = np.minimum(np.maximum(qx, qz), 0)
    return -(outside + inside)


def tilt_of(s):
    return float(np.arccos(np.clip(s.poses["cup"][0][2, 2], -1, 1)))


def locs(s):
    return np.array([s.poses[b][1] for b in BEADS])


def ref(s):
    return float(min(cyl_sdf(locs(s)).min(), THETA2 - tilt_of(s)))


def env_gap(states):
    """semantic comparison with the env's own predicates: sign vs oracle out_mask (spill), numeric vs bead_margin_i (env l_i)."""
    spec, _ = build()
    m_bead = []
    agree, n_out, gap = 0, 0, []
    for s in states:
        loc = locs(s)
        out = C.Oracle.out_mask(loc).any()
        m = ir.margin(ir_ext.from_dict({"forall": {"var": "b", "in": BEADS, "of": {"inside_cyl": {"a": "$b", "b": "cup_volume"}}}},
                                       build()[1], {}), s)
        agree += (m < 0) == bool(out)
        n_out += bool(out)
        gap.append(m - B.bead_margin_i(loc).min())
    return dict(sign_vs_out_mask=agree / len(states), n_spill_states=n_out, ir_minus_env_li_min=float(np.min(gap)), ir_minus_env_li_max=float(np.max(gap)))


def toy(s, rng):
    """u = [a(3), w(3)] of the cup: beads shift opposite to a (0.002 m per unit, per bead jitter), rise with a_z; cup axis rate w x c."""
    Jc = np.zeros((2, 6))
    Jc[0, 4], Jc[1, 3] = 0.1, -0.1
    jac = {"u0": rng.normal(size=6) * 0.3, "cup": {"axis": Jc}, "cup_volume": {"axis": Jc * 0}}
    for b in BEADS:
        J = np.zeros((3, 6))
        J[0, 0], J[1, 1], J[2, 2] = -0.002, -0.002, 0.002
        J += rng.normal(size=(3, 6)) * 0.0005
        jac[b] = {"p": J}
    return jac


def v0_proxies(states):
    """what v0 alone can do: the cup volume as an H-rep Region -- (a) 16-gon prism (circumscribed) + caps, (b) the inscribed square.
    Margin = Polytope slack (non-orthogonal). Sign agreement with the env's out_mask (the true spill predicate) and with the exact cylinder SDF."""
    from world_model.v2.tray_rep.sets.ir import Region
    from world_model.v2.tray_rep.sets.zono import Polytope
    out = {}
    ang = (np.arange(16) + 0.5) * 2 * np.pi / 16
    N16 = np.concatenate([np.stack([np.cos(ang), np.sin(ang), 0 * ang], 1), [[0, 0, 1.0], [0, 0, -1.0]]])
    b16 = np.concatenate([np.full(16, R_), [H_, ZM]])
    sq = R_ / np.sqrt(2)
    Nsq = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1.0], [0, 0, -1.0]])
    bsq = np.array([sq, sq, sq, sq, H_, ZM])
    for name, (N, b) in {"v0_16gon": (N16, b16), "v0_inscribed_square": (Nsq, bsq)}.items():
        P = Polytope(N, b)
        agree, errs = 0, []
        for s in states:
            loc = locs(s)
            m = P.signed_distance_point(loc.T).min() if False else min(P.signed_distance_point(x) for x in loc)
            agree += (m < 0) == bool(C.Oracle.out_mask(loc).any())
            sd = cyl_sdf(loc).min()
            if abs(sd) < 0.01:
                errs.append(abs(m - sd))              # error where it matters: within 1 cm of the cup surface
        out[name] = dict(sign_agree_vs_out_mask=agree / len(states), max_abs_err_vs_sdf_within_1cm=float(max(errs)), n_within_1cm=len(errs))
    return out


def make(tilt=True):
    spec = SPEC if tilt else {"and": [SPEC["and"][0]]}
    rf = ref if tilt else (lambda s: float(cyl_sdf(locs(s)).min()))
    return Task("cup_beads" if tilt else "cup_beads_only", spec, (lambda: (ir_ext.from_dict(spec, build()[1], dict(theta_max=THETA2)), build()[1])), sample, rf, toy,
                notes=dict(atoms=["inside_cyl(new)", "upright", "and", "forall(new)"], nonconvex=[]))
