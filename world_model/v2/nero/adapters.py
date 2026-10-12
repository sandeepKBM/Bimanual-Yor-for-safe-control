"""adapters.py -- glue between the NERO variant (7-DoF/arm) and the UNMODIFIED sim_common / mpc_fl.

What is adapted here (each one replaces a sim_common piece that hardcodes the 6-DoF Piper):
  * joint-name patterns            sim_common.LEFT/RIGHT_ARM_JOINTS (range(1,7))    -> config.LEFT/RIGHT_ARM_JOINTS (1..7)
  * collision bit scheme           sim_common._patch_real_collision_geoms           -> patch_collision (same bits; its
                                    _exclude_adjacent_links adds <exclude left_arm_link7,left_arm_link8> which does not exist
                                    on NERO and makes the model fail to compile)
  * keypoint spheres + pair rows   sim_common.barrier_keypoints/all_pair_barriers   -> proxy.keypoints + pair_barriers
                                    (LINK_NAMES incl. 'link8','iphone_body' are Piper bodies)
  * scene with tray + ball + floor build.compose (house) -> compose_scene (no house assets needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

from . import config as C
from . import proxy as P
from . import build_variant as BV

REPO = C.REPO
for _p in (str(REPO), str(REPO / "experiments" / "ab_smooth_vs_bangbang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sim_common as sc   # noqa: E402  (read-only import)

LEFT_BIT, RIGHT_BIT, LEFT_SELF, RIGHT_SELF = 0b0100, 0b1000, 0b1_0000, 0b10_0000   # == sim_common
ARM_BITS = LEFT_BIT | RIGHT_BIT


def patch_collision(spec):
    """sim_common._patch_real_collision_geoms bit scheme, minus its Piper-only link7/link8 exclude."""
    L_ct, L_ca = LEFT_BIT | LEFT_SELF, RIGHT_BIT | LEFT_SELF
    R_ct, R_ca = RIGHT_BIT | RIGHT_SELF, LEFT_BIT | RIGHT_SELF
    for b in spec.bodies:
        if b.name.startswith("left_arm_"):
            ct, ca = L_ct, L_ca
        elif b.name.startswith("right_arm_"):
            ct, ca = R_ct, R_ca
        else:
            continue
        for g in b.geoms:
            if g.contype == 0 and g.conaffinity == 1:
                g.contype, g.conaffinity = ct, ca
    bodies = list(spec.bodies)
    bodies.append(spec.worldbody)
    for b in bodies:
        for g in b.geoms:
            gn = (g.name or "").lower()
            if not (g.contype or g.conaffinity):
                continue
            if "table" in gn or "floor" in gn or "leg" in gn:
                g.conaffinity |= ARM_BITS


def compose_scene(with_tray: bool = True, lift: bool = False, gravcomp: bool = False, tau=None, dt=None,
                  gravity=(0.0, 0.0, -9.81), floor: bool = True):
    """YOR-NERO + floor + tray + ball (free bodies), tray welded to both gripper_base bodies (weld INACTIVE until
    set_hold_state captures the relpose).  Returns MjModel (collision bits patched)."""
    spec = mujoco.MjSpec.from_string(BV.variant_xml(lift=lift, gravcomp=gravcomp, tau=tau))
    if floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=1)
    if with_tray:
        from world_model.v2.testbeds.b_tray_ball import tray_env as TE
        tray = mujoco.MjSpec.from_string(TE.tray_xml(TE.BallParams(), lip=False, home=(0.0, 0.0, 5.0)))
        f2 = spec.worldbody.add_frame(pos=[0, 0, 5.0])
        f2.attach_body(tray.body("tray"), "", "")
        f3 = spec.worldbody.add_frame(pos=[0, 0, 6.0])
        f3.attach_body(tray.body("ball"), "", "")
        for side in C.SIDES:
            spec.add_equality(name=f"hold_{side}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=f"{side}_arm_gripper_base", name2="tray", active=0)
    patch_collision(spec)
    m = spec.compile()
    if dt is not None:
        m.opt.timestep = dt
    m.opt.gravity[:] = gravity
    return m


def arm_addrs(m, side: str):
    """(qposadr(7,), dofadr(7,)) of the arm joints."""
    names = C.LEFT_ARM_JOINTS if side == "left" else C.RIGHT_ARM_JOINTS
    _, qa, da = sc.joint_addrs(m, names)
    return qa, da


def disable_arm_servos(m):
    """Zero the 14 arm position servos (torque via qfrc_applied), as sim_common.disable_*_position_servos / house.build."""
    for side in C.SIDES:
        for i in range(1, C.N_ARM + 1):
            a = m.actuator(f"{side}_arm_joint{i}_pos").id
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0


# ------------------------------------------------------------------------------------------------ pair barriers
class PairBarriers:
    """sim_common.all_pair_barriers replacement for NERO keypoints.  Returns the K_out smallest-h pairs as the dict
    format FLMPC.step consumes (h, dist, n, Je, Ji, v_ego, v_int, hdot, eid, iid), Jacobians sliced to the 7 arm dofs.
    Vectorised distance matrix; Jacobians only for the keypoints that appear in the kept pairs."""

    def __init__(self, m, ego="left", intr="right", d_min=None, k_out=16, spec=None, mask_static=True):
        spec = spec or P.load()
        self.m = m
        self.kp_e = P.keypoints(m, f"{ego}_arm_", spec)
        self.kp_i = P.keypoints(m, f"{intr}_arm_", spec)
        self.qa_e, self.da_e = arm_addrs(m, ego)
        self.qa_i, self.da_i = arm_addrs(m, intr)
        self.d_min = sc.D_MIN if d_min is None else d_min
        self.k_out = k_out
        self.be = np.array([k[0] for k in self.kp_e]); self.oe = np.array([k[1] for k in self.kp_e]); self.re = np.array([k[2] for k in self.kp_e])
        self.bi = np.array([k[0] for k in self.kp_i]); self.oi = np.array([k[1] for k in self.kp_i]); self.ri = np.array([k[2] for k in self.kp_i])
        self._jp = np.zeros((3, m.nv)); self._jr = np.zeros((3, m.nv))
        # link0 (fixed base) spheres are static: a static-static pair has J_e = J_i = 0, i.e. a barrier row 0 <= const that only
        # burns one of FLMPC's K_PAIRS slots.  mask_static drops those pairs (link0 vs the OTHER arm's moving links is kept).
        st_e = np.array([m.body(b).name.endswith("link0") for b in self.be])
        st_i = np.array([m.body(b).name.endswith("link0") for b in self.bi])
        self._mask = (st_e[:, None] & st_i[None, :]) if mask_static else np.zeros((len(self.be), len(self.bi)), bool)

    def _pos(self, d, b, o):
        R = d.xmat[b].reshape(-1, 3, 3)
        return d.xpos[b] + np.einsum("kij,kj->ki", R, o)

    def _jac(self, d, bid, p):
        mujoco.mj_jac(self.m, d, self._jp, self._jr, p, bid)
        return self._jp.copy()

    def h_matrix(self, d):
        pe, pi = self._pos(d, self.be, self.oe), self._pos(d, self.bi, self.oi)
        D = pe[:, None, :] - pi[None, :, :]
        dist = np.linalg.norm(D, axis=-1)
        H = dist - (self.re[:, None] + self.ri[None, :] + self.d_min)
        H = np.where(self._mask, 1e9, H)
        return H, dist, D, pe, pi

    def min_h(self, d):
        return float(self.h_matrix(d)[0].min())

    def __call__(self, d):
        H, dist, D, pe, pi = self.h_matrix(d)
        flat = np.argsort(H, axis=None)[: self.k_out]
        ie, ii = np.unravel_index(flat, H.shape)
        cache_e, cache_i = {}, {}
        out = []
        qe, qi = d.qvel[self.da_e], d.qvel[self.da_i]
        for a, b in zip(ie, ii):
            if a not in cache_e:
                cache_e[a] = self._jac(d, self.be[a], pe[a])[:, self.da_e]
            if b not in cache_i:
                cache_i[b] = self._jac(d, self.bi[b], pi[b])[:, self.da_i]
            Je, Ji = cache_e[a], cache_i[b]
            n = D[a, b] / max(dist[a, b], 1e-9)
            ve, vi = Je @ qe, Ji @ qi
            out.append(dict(h=float(H[a, b]), dist=float(dist[a, b]), eid=int(self.be[a]), iid=int(self.bi[b]), n=n,
                            Je=Je, Ji=Ji, v_ego=ve, v_int=vi, hdot=float(n @ (ve - vi))))
        return out


def true_min_dist(m, d, gl, gr, dmax=0.5):
    """min mj_geomDistance over geom lists gl x gr.  Mesh-mesh pairs beyond dmax can return a spurious 0 (see house/phase1):
    results whose fromto segment length disagrees with |dist| are discarded."""
    ft = np.zeros(6)
    best = dmax
    for a in gl:
        for b in gr:
            dist = mujoco.mj_geomDistance(m, d, a, b, dmax, ft)
            if dist < dmax and abs(np.linalg.norm(ft[3:] - ft[:3]) - abs(dist)) > 1e-4:
                continue
            if dist == 0.0:                  # exact 0.0 from a mesh-mesh pair with no contact = spurious (verified: ncon == 0)
                continue
            best = min(best, float(dist))
    return float(best)


def arm_geoms(m, side):
    """collidable geoms of an arm (hull collision meshes)."""
    return [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"{side}_arm_") and m.geom_contype[g]]
