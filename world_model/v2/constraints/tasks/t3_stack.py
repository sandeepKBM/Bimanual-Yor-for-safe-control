"""Task 3: a 3-block stack on the carried plate stays supported (testbeds/c_block_tower). Everything in the PLATE frame (parent 'plate').
Two specs (both v0 atoms only: support, upright, and; SupportRel only fixes the rows' relative Jacobian):
  centre   : each block centre inside the top face of the body below (+ tilt vs plate normal <= TH_FAIL)        == tower_env.margin parts
  substack : the COM of blocks j..N-1 inside the top face of block j-1 (+ same tilt)                                 == analytic.py tip criterion (static)
hand-coded reference: top face of the body below projected on the plate xy plane (parallelogram when tilted), point-in-polygon signed distance by
edge normals (independent of the zonotope code); tilt = arccos(R_rel[2,2])."""
from __future__ import annotations

import os

import numpy as np

from world_model.v2.testbeds.c_block_tower import tower_env as T
from world_model.v2.tray_rep.sets.ir import BoxShape, Point
from world_model.v2.constraints import ir_ext
from world_model.v2.constraints.common import PoseState, Task, ent
from world_model.v2.constraints.tasks.t2_cup import quat_R

CFG = T.TowerCfg()
N = CFG.n_blocks
TH = T.TH_FAIL
HALF = [(CFG.half[0] - 3e-4 * i, CFG.half[1] - 2.5e-4 * i, CFG.half[2]) for i in range(N)]
CACHE = "/common/users/ss5772/bimanual_data/constraints_v1/tmp/t3_states.npz"
below = lambda i: "plate" if i == 0 else f"blk{i - 1}"
SPEC_CENTRE = {"and": [*[{"support_rel": {"a": f"blk{i}", "b": below(i)}} for i in range(N)],
                       *[{"upright": {"a": f"blk{i}", "theta_max": "th"}} for i in range(N)]]}
SPEC_SUB = {"and": [*[{"support_rel": {"a": f"com{j}", "b": below(j)}} for j in range(N)],
                    *[{"upright": {"a": f"blk{i}", "theta_max": "th"}} for i in range(N)]]}


def build_ents():
    ents = {"plate": ent("plate", BoxShape(CFG.plate_half), parent=None)}
    for i in range(N):
        ents[f"blk{i}"] = ent(f"blk{i}", BoxShape(HALF[i]), parent="plate")
        ents[f"com{i}"] = ent(f"com{i}", Point(), parent="plate")
    return ents


def make_build(spec):
    def build():
        ents = build_ents()
        return ir_ext.from_dict(spec, ents, dict(th=TH)), ents
    return build


def simulate(n_eps=60, ticks=100, every=2, seed=0):
    rng = np.random.default_rng(seed)
    U = T.fallback_U()
    out = []
    for ep in range(n_eps):
        env = T.TowerEnv(CFG, seed=int(rng.integers(1 << 30)))
        env.reset(np.array([0, 0, 1.0]), np.array([1.0, 0, 0, 0]))
        u, hold = U[0], 0
        for t in range(ticks):
            if hold == 0:
                big = U[np.argsort(-np.abs(U).sum(1))[:40]] if ep % 2 else U
                u = big[rng.integers(len(big))]
                hold = rng.integers(6, 20)
            hold -= 1
            env.tick(u, check=True)
            if env.failed and t > env.fail_tick + 5:
                break
            if t % every == 0:
                q = env.d.qpos
                Rp = T.quat_mat(env.quat)
                rel_p, rel_R = [], []
                for i in range(N):
                    pb = q[env.bq[i]:env.bq[i] + 3]
                    Rb = quat_R(q[env.bq[i] + 3:env.bq[i] + 7])
                    rel_p.append(Rp.T @ (pb - env.pos))
                    rel_R.append(Rp.T @ Rb)
                out.append((np.array(rel_p), np.array(rel_R)))
    return out


def states_from(P, R):
    S = []
    for p, Rm in zip(P, R):
        poses = {"plate": (np.eye(3), np.zeros(3))}
        for i in range(N):
            poses[f"blk{i}"] = (Rm[i], p[i])
        for j in range(N):
            poses[f"com{j}"] = (np.eye(3), p[j:].mean(0))
        S.append(PoseState(poses))
    return S


def sample(n, rng):
    if not os.path.exists(CACHE):
        sims = simulate()
        np.savez(CACHE, P=np.array([s[0] for s in sims]), R=np.array([s[1] for s in sims]))
    z = np.load(CACHE)
    S = states_from(z["P"], z["R"])
    idx = rng.permutation(len(S))[:n]
    return [S[i] for i in idx]


def poly_margin(poly, x):
    """signed distance (inside > 0) of 2-D point x to a convex polygon given by counter-clockwise vertices: min over edges of the
    signed edge distance when inside; Euclidean distance to the polygon when outside (nearest of segment distances)."""
    n = len(poly)
    sd_edge = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)                    # outward normal for CCW
        sd_edge.append(-(nrm @ (x - a)))
    if min(sd_edge) >= 0:
        return float(min(sd_edge))
    best = np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        t = np.clip((x - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        best = min(best, np.linalg.norm(x - (a + t * (b - a))))
    return -float(best)


def top_polygon(R, p, half):
    """horizontal (plate xy) projection of the top face of a box with pose (R, p): CCW parallelogram."""
    hx, hy, hz = half
    pts = np.array([R @ np.array([sx * hx, sy * hy, hz]) + p for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    P = pts[:, :2]
    if (P[1] - P[0])[0] * (P[2] - P[0])[1] - (P[1] - P[0])[1] * (P[2] - P[0])[0] < 0:
        P = P[::-1]
    return P


def make_ref(sub):
    def ref(s):
        vals = []
        for j in range(N):
            if j == 0:
                R0, p0, h0 = np.eye(3), np.zeros(3), CFG.plate_half
            else:
                R0, p0 = s.poses[f"blk{j - 1}"]
                h0 = HALF[j - 1]
            x = np.mean([s.poses[f"blk{i}"][1] for i in range(j, N)], 0)[:2] if sub else s.poses[f"blk{j}"][1][:2]
            vals.append(poly_margin(top_polygon(R0, p0, h0), x))
        for i in range(N):
            vals.append(TH - np.arccos(np.clip(s.poses[f"blk{i}"][0][2, 2], -1, 1)))
        return float(min(vals))
    return ref


def env_gap(states):
    """centre spec vs tower_env.margin parts (un-normalised): max |ir_support_part - env l_sup * scale| over the support atoms."""
    spec, _ = make_build(SPEC_CENTRE)()
    ph = CFG.plate_half
    gaps = []
    for s in states:
        P = np.array([s.poses[f"blk{i}"][1] for i in range(N)])
        e = [min(ph[0] - abs(P[0, 0]), ph[1] - abs(P[0, 1]))]
        for i in range(1, N):
            dxy = P[i, :2] - P[i - 1, :2]
            e.append(min(CFG.half[0] - abs(dxy[0]), CFG.half[1] - abs(dxy[1])))
        sup = {"and": [SPEC_CENTRE["and"][i] for i in range(N)]}
        m = __import__("world_model.v2.tray_rep.sets.ir", fromlist=["x"]).margin(ir_ext.from_dict(sup, build_ents(), {}), s)
        gaps.append(abs(m - min(e)))
    return dict(max_abs_gap_vs_env_support=float(max(gaps)))


def toy_factory(sub):
    def toy(s, rng):
        jac = {"u0": rng.normal(size=6) * 0.3}
        for i in range(N):
            J = np.zeros((3, 6))
            k = 0.004 * (i + 1)
            J[0, 0], J[1, 1] = -k, -k
            Jc = np.zeros((2, 6))
            Jc[0, 4], Jc[1, 3] = 0.05 * (i + 1), -0.05 * (i + 1)                        # block tilt follows the plate rate
            jac[f"blk{i}"] = {"p": J, "axis": Jc}
        for j in range(N):
            J = np.zeros((3, 6))
            ks = [0.004 * (i + 1) for i in range(j, N)]
            J[0, 0], J[1, 1] = -np.mean(ks), -np.mean(ks)
            jac[f"com{j}"] = {"p": J}
        return jac
    return toy


def make(sub=False):
    spec = SPEC_SUB if sub else SPEC_CENTRE
    return Task("stack_substack" if sub else "stack_centre", spec, make_build(spec), sample, make_ref(sub), toy_factory(sub),
                notes=dict(atoms=["support", "upright", "and"], nonconvex=[]))
