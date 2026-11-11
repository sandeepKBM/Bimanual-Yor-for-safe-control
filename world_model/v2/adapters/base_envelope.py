"""adapters/base_envelope.py -- MuJoCo backend of stack/checks/base_motion (rows base_torque, base_open_content).  SIMULATION ONLY.

BaseEnvelope(task, world)  (protocol base_motion.BaseLoads)
  hold pose = the compiled swerve scene at the episode start (as adapters/base_tip_model); arms hold their joints.
  arm ID    mujoco.mj_rne (M qacc + C(q, qd), gravity inside) with the base free joint given the twist / acceleration of a rigid motion about the rotation
            centre c (a_c = R0 [a_bx, a_by, 0], alpha z, w) and the arm qd = qdd = 0; gravcomp zeroed (the motors carry the arm, as nero/contents/envelope.Arms).
  object    point masses: the empty object (ObjectSpec empty_mass / empty_com) + the contents vertex (m, c), a_i = a_c + alpha z x r_i - w^2 r_i;
            F = sum m_i (a_i - g), T_o = sum m_i (p_i - o) x (a_i - g); the lever split is nero/contents/envelope.lever_split (uniform acc = F / M)
            plus the inertial moment T_o - S x acc split equally (the lever rule's own twist split).
  spread    bound on the bodies' spread about their lumped points: |dT| <= (alpha + w^2) (m_c rho_c^2 + m_e rho_e^2), rho_c = diameter of the contents COM
            polytope, rho_e = diameter of the object's geoms (MODEL, conservative); per claw dT / 2, mapped to a joint by |J_ang column|.
  limits    tau = config.tau_limit() (MJCF 100 Nm, USER 11-05), F_cap = (3 kg - gripper) g (payload_envelope.caps), the payload row's definitions.
Helpers: torque_row / open_row / mug_geom for one task, level and acceleration source ("measured" admits; "command" is reported only).
"""
from __future__ import annotations

import functools

import mujoco
import numpy as np

from world_model.v2.adapters import base_tip_model as BT
from world_model.v2.adapters import contents_specs as CS
from world_model.v2.adapters import payload_envelope as PE
from world_model.v2.nero import adapters as A
from world_model.v2.nero import config as C
from world_model.v2.nero.contents import envelope as E
from world_model.v2.nero.contents import objects as OB
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import base_motion as BM
from world_model.v2.stack.checks import base_tip as BTIP

G = OB.G
Z = np.array([0.0, 0.0, 1.0])
MU_CUP = 0.5          # ASSUMED = scenes.TrayWorld.cup_mu (ceramic on glaze).  MuJoCo's pair rule would use max(cup 0.5, tray 0.6) = 0.6; the lower one is used.
SIDES = ("left", "right")


class BaseEnvelope:
    def __init__(self, task: str, world=None):
        self.task = task
        w = CS.build_world(task) if world is None else world
        ep = BT._episode(w)
        self.m, self.d = ep.m, ep.d
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        self.R0, o, _ = BT._frame(m, d)
        self.c = np.asarray(o, float)                      # rotation centre on the floor, world
        self.base = m.body("base").id
        self.jadr = int(m.jnt_dofadr[m.body_jntadr[self.base]])
        self.arms = SIDES
        self.da = {s: A.arm_addrs(m, s)[1] for s in SIDES}
        self.site = {s: m.site(f"{s}_arm_ee_grasp").id for s in SIDES}
        self.p_claw = {s: d.site_xpos[self.site[s]].copy() for s in SIDES}
        self.J = {}
        for s in SIDES:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self.site[s])
            self.J[s] = np.vstack([jp[:, self.da[s]], jr[:, self.da[s]]])
        self.o = d.site_xpos[m.site("obj_ref").id].copy()
        ob = m.body("obj").id
        self.Ro = d.xmat[ob].reshape(3, 3).copy()
        spec = CS.spec_from_cell(task, *TK.prior_box(task))
        self.m_e = float(spec.empty_mass)
        self.p_e = self.Ro @ np.asarray(spec.empty_com, float) + self.o
        self.P = np.asarray(spec.contents.P, float)
        self.rho_c = max((float(np.linalg.norm(a - b)) for a in self.P for b in self.P), default=0.0)
        self.rho_e = self._obj_diameter(ob)
        self.tau_lim = np.broadcast_to(np.asarray(C.tau_limit(), float), (7,)).copy()
        self.grip = float(PE.caps()["pot"]["gripper_mass"])
        self.F_cap = (C.MANUAL["payload_kg"] - self.grip) * G
        self._id_cache = {}

    def _obj_diameter(self, ob):
        pts = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != ob:
                continue
            c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
            Rg, pg = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(pg + Rg @ (c + np.array([sx, sy, sz]) * h))
        return max(float(np.linalg.norm(a - b)) for a in pts for b in pts)

    # ---- arm inverse dynamics under the base motion ---------------------------------------------------------------------
    def arm_id(self, a_b, alpha, w2) -> dict:
        """{arm: (7,)} = rne rows of the arm joints: floating model, base rigid motion about c, arm qd = qdd = 0, gravcomp off."""
        key = (round(float(a_b[0]), 9), round(float(a_b[1]), 9), round(float(alpha), 9), round(float(w2), 9))
        if key in self._id_cache:
            return self._id_cache[key]
        m = self.m
        gc = m.body_gravcomp.copy()
        m.body_gravcomp[:] = 0.0
        try:
            dd = mujoco.MjData(m)
            dd.qpos[:] = self.d.qpos
            mujoco.mj_fwdPosition(m, dd)
            Rb = self.d.xmat[self.base].reshape(3, 3)
            r = self.d.xpos[self.base] - self.c
            rh = np.array([r[0], r[1], 0.0])
            w = float(np.sqrt(w2))
            a_c = self.R0 @ np.array([a_b[0], a_b[1], 0.0])
            dd.qvel[:] = 0.0
            dd.qacc[:] = 0.0
            dd.qvel[self.jadr:self.jadr + 3] = w * np.cross(Z, rh)
            dd.qvel[self.jadr + 3:self.jadr + 6] = Rb.T @ (w * Z)
            dd.qacc[self.jadr:self.jadr + 3] = a_c + alpha * np.cross(Z, rh) - w2 * rh
            dd.qacc[self.jadr + 3:self.jadr + 6] = Rb.T @ (alpha * Z)
            mujoco.mj_comVel(m, dd)                           # rne reads cvel / cdof_dot: without this the velocity terms (centrifugal) are dropped
            res = np.zeros(m.nv)
            mujoco.mj_rne(m, dd, 1, res)
        finally:
            m.body_gravcomp[:] = gc
        out = {s: res[self.da[s]].copy() for s in SIDES}
        self._id_cache[key] = out
        return out

    # ---- object wrench + lever split --------------------------------------------------------------------------------------
    def _accel_at(self, p, a_b, alpha, w2):
        rh = np.array([p[0] - self.c[0], p[1] - self.c[1], 0.0])
        return self.R0 @ np.array([a_b[0], a_b[1], 0.0]) + alpha * np.cross(Z, rh) - w2 * rh

    def claw_wrenches(self, vertex, a_b, alpha, w2):
        """{arm: (F (3,), T (3,))} the lever-split wrench each claw applies to the object (world axes, T about the claw point)."""
        mc, cobj = float(vertex[0]), np.asarray(vertex[1], float)
        pc = self.Ro @ cobj + self.o
        ms, ps = (self.m_e, mc), (self.p_e, pc)
        up = np.array([0.0, 0.0, G])
        F = np.zeros(3)
        T = np.zeros(3)
        for mi, pi in zip(ms, ps):
            acc = self._accel_at(pi, a_b, alpha, w2) + up
            F += mi * acc
            T += mi * np.cross(pi - self.o, acc)
        Mtot = self.m_e + mc
        S = self.m_e * (self.p_e - self.o) + mc * (pc - self.o)
        acc_c = F / Mtot
        FL, TL, FR, TR = E.lever_split(Mtot, acc_c, S, self.o, self.p_claw["left"], self.p_claw["right"])
        dT = T - np.cross(S, acc_c)                          # inertial moment beyond the uniform-acceleration part
        return {"left": (FL, TL + 0.5 * dT), "right": (FR, TR + 0.5 * dT)}

    def loads(self, vertex, a_b, alpha, w2) -> dict:
        idr = self.arm_id(a_b, alpha, w2)
        wr = self.claw_wrenches(vertex, a_b, alpha, w2)
        out = {}
        for s in SIDES:
            F, T = wr[s]
            out[s] = (idr[s] + self.J[s].T @ np.concatenate([F, T]), float(F[2]))
        return out

    def spread(self, vertex, alpha, w2) -> dict:
        dT = (alpha + w2) * (float(vertex[0]) * self.rho_c ** 2 + self.m_e * self.rho_e ** 2)
        return {s: 0.5 * dT * np.linalg.norm(self.J[s][3:], axis=0) for s in SIDES}


@functools.lru_cache(maxsize=4)
def envelope(task: str) -> BaseEnvelope:
    return BaseEnvelope(task)


def torque_row(task: str, contents, level: str, source: str = "measured", world=None):
    """base_torque row of `contents` (OB.Contents) on task at `level`; source "measured" admits (sampled), "command" is reported only (proof of the model)."""
    env = envelope(task) if world is None else BaseEnvelope(task, world)
    return BM.torque_row(env, contents, BT.accels(level, source))


def mug_geom(task: str = "tray6_behind") -> BM.MugGeom:
    """MODEL / ASSUMED mug of the tray scenes (objects.CUP + scenes constants).  z_c: the mug is a solid cylinder (COM H/2, scenes.py) and the fill sits on the
    inner bottom (COM CUP_BOT + h/2, h = m_f / (rho pi r_in^2)); z_c(m_f) is monotone, so the highest is at an end of the fill interval."""
    D, H = OB.CUP["D"], OB.CUP["H"]
    mc = OB.CUP["empty"]

    def zc(mf):
        h = mf / (OB.WATER * np.pi * SC.CUP_RIN ** 2)
        return (mc * H / 2 + mf * (SC.CUP_BOT + h / 2)) / (mc + mf)

    z_hi = max(zc(0.0), zc(OB.CUP["fill_max_kg"]))
    return BM.MugGeom(D, H, H - OB.CUP["freeboard_min"], MU_CUP, float(z_hi),
                      labels=dict(mug_labels=dict(geometry="ASSUMED mug D/H/fill (objects.CUP); fill up to the freeboard_min below the lip",
                                           mu_cup=f"ASSUMED {MU_CUP} (= scenes.TrayWorld.cup_mu; the sim pair rule uses 0.6 with the tray)",
                                           zc="MODEL: solid cylinder COM H/2, fill on the inner bottom; max over the fill interval")))


def mug_corners(task: str = "tray6_behind", footprint: str = "tray"):
    """mug positions [m] from the rotation centre, in B: the tray footprint corners (ASSUMED A10; 'cups' = the cup-centre rectangle, smaller)."""
    mm = BT.mass_model(task)
    if footprint == "tray":
        hx, hy = OB.TRAY["lx"] / 2, OB.TRAY["ly"] / 2
    else:
        hx, hy = OB.TRAY["lx"] / 2 - OB.CUP["D"] / 2, OB.TRAY["ly"] / 2 - OB.CUP["D"] / 2
    return np.array([(mm.obj_R @ np.array([sx * hx, sy * hy, 0.0]) + mm.obj_p)[:2] for sx in (-1, 1) for sy in (-1, 1)])


def open_row(task: str, level: str, source: str = "measured", footprint: str = "tray"):
    """base_open_content row (tray tasks only)."""
    return BM.open_content_base_row(mug_geom(task), mug_corners(task, footprint), BT.accels(level, source))


def pick_level(task: str, spec, levels=("L1", "L2", "L3"), source: str = "measured"):
    """the base level of the admission (design section 3): the HIGHEST level at which every base row passes on spec.contents (Theta_t): base_tip, base_torque and (tray
    tasks only) base_open_content, all on the acceleration box of `source` ("measured" = the base card, sampled; "command" is for comparison only).  Robot masses and
    the arm envelope are the task's default scene (MODEL: they do not depend on the hidden world; the contents enter through spec.contents).
    -> (level | None, {level: {row name: Row}}); None = no level passes -> refuse."""
    rows, best = {}, None
    for L in levels:
        r = {"base_tip": BTIP.tip_row(BT.mass_model(task), spec.contents, BT.polygons(task), BT.accels(L, source)),
             "base_torque": torque_row(task, spec.contents, L, source)}
        if TK.task(task).kind == "tray":
            r["base_open_content"] = open_row(task, L, source)
        rows[L] = r
        if all(x.ok for x in r.values()):
            best = L
    return best, rows
