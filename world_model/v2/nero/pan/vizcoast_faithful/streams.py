"""COAST-style streams on the NOMINAL (believed, known) scene: sampled continuous parameters + kinematic tests only (IK, contact-
free pre-grasp, reachable push / lift poses).  No dynamics, no load, no uncertainty -- as in COAST/VIZ-COAST.  SIMULATION ONLY.

refine(plan) samples each action's parameters in order (up to N_SAMPLES draws per stream) and returns the grounded plan or the
index + reason of the first stream that ran out of samples (the refinement failure VIZ-COAST caches).
"""
from __future__ import annotations

import numpy as np
import mujoco

from .. import covered as CV
from .. import scene as S
from .. import handle_grasp as HG
from .. import under_rim as UR
from ...democ import kin as K

N_SAMPLES = 10
CLAMP_MIN_OVERHANG = 0.02   # ASSUMED, m: the lower finger needs room under the pan bottom beyond the stove edge (geometric assumption)
L_FRAC = (0.55, 0.95)       # design (our choice; the paper gives no ranges): sampled left pinch position along the handle
R_FRAC = (0.2, 0.6)         # design
PUSH = (0.04, 0.09)         # design


class Scene:
    def __init__(self, seed=0):
        self.cp = CV.CoveredParams(placement="none")
        self.m = CV.compose(self.cp)
        self.m.opt.timestep = HG.DT
        self.d = mujoco.MjData(self.m)
        S.settle(self.m, self.d, 1.0)
        self.p = self.cp.pan
        self.rng = np.random.default_rng(seed)
        self.n_stream_calls = 0

    # --- streams ------------------------------------------------------------------------------------------------------
    def grasp_handle(self, arm):
        self.n_stream_calls += 1
        if arm == "l":
            frac = float(self.rng.uniform(*L_FRAC))
            tgt = HG.handle_point(self.m, self.d, self.p, frac)
            pre = HG.find_pregrasp(self.m, self.d, "left", tgt, HG.LEFT_APPROACH, HG.LEFT_JAW, n_sol=6, seeds=30)
            return dict(arm="l", part="handle", frac=frac, q=pre["q"])
        frac = float(self.rng.uniform(*R_FRAC))
        grip = str(self.rng.choice(sorted(HG.RIGHT_GRIPS)))
        appr, jaw = (np.array(v, float) for v in HG.RIGHT_GRIPS[grip])
        depth = 0.5 * (abs(appr[0]) * 0.024 + abs(appr[2]) * 0.014) + 0.004
        tgt = HG.handle_point(self.m, self.d, self.p, frac) + depth * appr
        pre = HG.find_pregrasp(self.m, self.d, "right", tgt, appr, jaw, n_sol=6, seeds=30)
        return dict(arm="r", part="handle", frac=frac, grip=grip, q=pre["q"])

    def grasp_lid(self, arm):
        self.n_stream_calls += 1
        knob = self.d.xpos[self.m.body("lid").id] + np.array([0, 0, 0.012])
        side = "left" if arm == "l" else "right"
        pre = HG.find_pregrasp(self.m, self.d, side, knob, np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), n_sol=6, seeds=30)
        return dict(arm=arm, part="lid", q=pre["q"])

    def push(self, grasp):
        self.n_stream_calls += 1
        dist = float(self.rng.uniform(*PUSH))
        u = self.p.handle_axis()
        pik = K.PoseIK(self.m, "left")
        p0, R0 = pik.ik.pose(grasp["q"])
        q, _ = pik.solve(p0 - dist * u, R0, grasp["q"], q_post=grasp["q"], k_post=0.1, iters=300, tol=1e-10)
        err = float(np.linalg.norm(pik.ik.pose(q)[0] - (p0 - dist * u)))
        if err > 2e-3:
            raise RuntimeError(f"push {dist:.3f} m unreachable (IK err {err*1e3:.1f} mm)")
        stove_edge = self.p.pan_centre_xy[1] - S.ASSUMED["stove_half"][1]
        rim_after = self.p.pan_centre_xy[1] - dist - (self.p.pan_radius + self.p.pan_wall_t)
        return dict(push=dist, overhang=float(stove_edge - rim_after), q_after=q)

    def clamp(self, push):
        self.n_stream_calls += 1
        if push["overhang"] < CLAMP_MIN_OVERHANG:
            raise RuntimeError(f"overhang {push['overhang']*100:.1f} cm < {CLAMP_MIN_OVERHANG*100:.0f} cm")
        d = mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        u = self.p.handle_axis()
        for b in ("pan", "lid"):
            j = self.m.body_jntadr[self.m.body(b).id]
            a = self.m.jnt_qposadr[j]
            d.qpos[a:a + 3] -= push["push"] * u
        mujoco.mj_forward(self.m, d)
        stove_edge = self.p.pan_centre_xy[1] - S.ASSUMED["stove_half"][1]
        z_mid = self.p.pan_base_z() + 0.5 * (self.p.pan_bottom_t + self.p.pan_wall_h + CV.ASSUMED_COVERED["lid_t"])
        target = np.array([self.p.pan_centre_xy[0], stove_edge - UR.EDGE_CLEAR, z_mid])
        pre = HG.find_pregrasp(self.m, d, "right", target - UR.APPROACH_IN * UR.RIGHT_APPROACH, UR.RIGHT_APPROACH, UR.RIGHT_JAW,
                               n_sol=6, seeds=30)
        return dict(arm="r", part="rim", q=pre["q"])

    def lift(self, grasps):
        self.n_stream_calls += 1
        for g in grasps:
            side = "left" if g["arm"] == "l" else "right"
            pik = K.PoseIK(self.m, side)
            q0 = g.get("q_after", g["q"])
            p0, R0 = pik.ik.pose(q0)
            q, _ = pik.solve(p0 + np.array([0, 0, HG.LIFT]), R0, q0, q_post=q0, k_post=0.1, iters=300, tol=1e-10)
            err = float(np.linalg.norm(pik.ik.pose(q)[0] - (p0 + np.array([0, 0, HG.LIFT]))))
            if err > 2e-3:
                raise RuntimeError(f"{side} lift pose unreachable (IK err {err*1e3:.1f} mm)")
        return dict(lift=HG.LIFT)

    # --- refinement -----------------------------------------------------------------------------------------------------
    def refine(self, plan):
        held, out, last_push = {}, [], None
        for i, ga in enumerate(plan):
            reason = None
            for _ in range(N_SAMPLES):
                try:
                    if ga.name == "pick":
                        v = self.grasp_handle(ga.args[0]) if ga.args[1] == "handle" else self.grasp_lid(ga.args[0])
                        held[ga.args[0]] = v
                    elif ga.name == "push":
                        v = self.push(held[ga.args[0]])
                        held[ga.args[0]] = dict(held[ga.args[0]], q_after=v["q_after"])
                        last_push = v
                    elif ga.name == "clamp":
                        v = self.clamp(last_push)
                        held[ga.args[0]] = v
                    elif ga.name in ("lift-one", "lift-two"):
                        arms = [ga.args[0]] if ga.name == "lift-one" else [ga.args[0], ga.args[2]]
                        v = self.lift([held[a] for a in arms])
                    else:
                        raise RuntimeError(f"no stream for {ga.name}")
                    reason = None
                    break
                except RuntimeError as e:
                    reason = str(e)
            if reason is not None:
                return False, out, i, reason
            out.append(dict(action=str(ga), **{k: (float(x) if isinstance(x, (float, np.floating)) else x)
                                               for k, x in v.items() if k not in ("q", "q_after")}))
        return True, out, None, None
