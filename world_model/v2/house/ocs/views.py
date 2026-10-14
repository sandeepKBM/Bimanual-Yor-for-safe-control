"""ocs/views.py -- what a Producer reads: a SensorView (+ a TruthView for ground truth).  SIMULATION ONLY.

A producer's `update(ws, view, t)` receives a view object.  Two implementations share one interface so the SAME producer code runs
live (LiveView on a TeleopSim) and offline (RecView on a Recording written by ocs/record.py):

  view.t, view.k                     time [s] / 50 Hz tick index
  view.encoders()                    dict(qL (6,), qR (6,), qdL, qdR, lift)           arm / lift encoders (noisy)
  view.odom()                        dict(pose (3,) map-frame estimate, sigma_xy, v (3,) body twist)     (house/loc PoseEstimator output)
  view.head_frame()                  head-camera capture taken THIS tick (dict) or None      (10 Hz; noisy depth points, see HeadSensor)
  view.wrist()                       dict(side -> (n_px, u, v) | None) wrist-camera red-blob track taken this tick, or None when no capture
  view.zono_obs()                    perceived static-scene result (flagship stereo zonotopes) or None
  view.fk                            ArmFK (kinematic model of the robot; constants of the robot, no state)
  view.truth                         TruthView -- PRIVILEGED: TruthProducer and the evaluation only
"""
from __future__ import annotations

import gzip
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.sets.zono import Zonotope


# ====================================================================================== robot kinematics (constants of the robot)
class ArmFK:
    """forward kinematics of the two arms + lift in the BASE frame (mocap root at identity), built from a TeleopSim-like object.
    tray(): the held tray's pose in the base frame = mean over the two hands of (hand pose o nominal weld relpose)  -- the nominal grasp offset
    is the engineering prior (ObjectFrame.nominal); the weld/grasp error is what E2 measures."""

    def __init__(self, sim):
        import mujoco
        self.mj = mujoco
        m = sim.m
        self.m = m
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = sim.d.qpos
        self.d.mocap_pos[:] = 0.0
        self.d.mocap_quat[:] = np.array([1.0, 0.0, 0.0, 0.0])
        self.lq, self.rq = np.asarray(sim.ctl.lq), np.asarray(sim.ctl.rq)
        self.slider = list(sim.slider_q)
        self.body = {s: m.body(f"{s}_arm_gripper_base").id for s in ("left", "right")}
        self.nominal = {k: np.array(v, float) for k, v in sim.of.nominal.items()}
        self.cam_id = {"left": m.camera("left_arm_egocentric").id, "right": m.camera("right_arm_egocentric").id}
        self.head_pos = None

    def _kin(self, qL, qR, lift):
        d = self.d
        d.qpos[self.lq] = qL
        d.qpos[self.rq] = qR
        for a in self.slider:
            d.qpos[a] = 0.5 * float(lift)
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_comPos(self.m, d)
        self.mj.mj_camlight(self.m, d)

    def hands(self, qL, qR, lift):
        self._kin(qL, qR, lift)
        return {s: (self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()) for s, b in self.body.items()}

    def tray(self, qL, qR, lift):
        """-> (p (3,), R (3,3)) tray pose in the base frame, FK + nominal grasp offset."""
        self._kin(qL, qR, lift)
        ps, Rs = [], []
        for s, b in self.body.items():
            nom = self.nominal[s]
            Rg = self.d.xmat[b].reshape(3, 3)
            Rn = np.zeros(9)
            self.mj.mju_quat2Mat(Rn, nom[6:10])
            ps.append(self.d.xpos[b] + Rg @ nom[3:6])
            Rs.append(Rg @ Rn.reshape(3, 3))
        M = Rs[0] + Rs[1]
        U_, _, Vt = np.linalg.svd(M)
        R = U_ @ Vt
        if np.linalg.det(R) < 0:
            U_[:, -1] *= -1
            R = U_ @ Vt
        return 0.5 * (ps[0] + ps[1]), R

    def wrist_cam(self, side, qL, qR, lift):
        """-> (origin (3,), R_cam_to_base (3,3)) of the wrist camera in the base frame."""
        self._kin(qL, qR, lift)
        c = self.cam_id[side]
        return self.d.cam_xpos[c].copy(), self.d.cam_xmat[c].reshape(3, 3).copy()


_FK_CACHE: dict = {}


def fk_for(scene: str = "h5") -> ArmFK:
    """ArmFK of a scene's rig (built once per process; building a TeleopSim takes seconds)."""
    if scene not in _FK_CACHE:
        if scene == "flagship":
            from world_model.v2.house.flagship import scene as FS
            sim = FS.new_sim(0, cafe_walker=False)
        else:
            from world_model.v2.house.teleop.sim import TeleopSim
            sim = TeleopSim(scene, grasped=True, seed=0, with_people=False)
        _FK_CACHE[scene] = ArmFK(sim)
    return _FK_CACHE[scene]


# ====================================================================================== truth views
class TruthView:
    """PRIVILEGED ground truth at one tick.  Attributes: t, base (x, y, yaw), base_v (3,), lift, qL, qR, qdL, qdR,
    tray_p, tray_R, tray_q, cup_p, cup_R, cup_q, people [(name, xy, v)], obstacles [(key, Zonotope)], occ (OccGrid | None), mu_true."""
    mu_true = 0.25


class LiveTruth(TruthView):
    def __init__(self, sim):
        self.sim = sim

    @property
    def t(self):
        return self.sim.t

    @property
    def base(self):
        b = self.sim.base
        return np.array([b.x, b.y, b.yaw])

    @property
    def base_v(self):
        return self.sim.base.v.copy()

    @property
    def lift(self):
        return self.sim.lift_actual()

    def _q(self, idx):
        return self.sim.d.qpos[idx].copy()

    qL = property(lambda s: s._q(s.sim.ctl.lq))
    qR = property(lambda s: s._q(s.sim.ctl.rq))
    qdL = property(lambda s: s.sim.d.qvel[s.sim.ctl.ld].copy())
    qdR = property(lambda s: s.sim.d.qvel[s.sim.ctl.rd].copy())
    tray_p = property(lambda s: s.sim.d.xpos[s.sim.tray].copy())
    tray_R = property(lambda s: s.sim.d.xmat[s.sim.tray].reshape(3, 3).copy())
    tray_q = property(lambda s: s.sim.d.xquat[s.sim.tray].copy())
    cup_p = property(lambda s: s.sim.d.xpos[s.sim.cup].copy())
    cup_R = property(lambda s: s.sim.d.xmat[s.sim.cup].reshape(3, 3).copy())
    cup_q = property(lambda s: s.sim.d.xquat[s.sim.cup].copy())

    @property
    def people(self):
        if not self.sim.with_people:
            return []
        return [(p.name, np.array(xy, float), np.array(v, float)) for p, (xy, v) in zip(self.sim.scene.people, self.sim.people_state)]

    @property
    def obstacles(self):
        return getattr(self.sim, "_ocs_obstacles", [])

    @property
    def occ(self):
        return self.sim.occ

    mu_true = 0.25


class RecTruth(TruthView):
    def __init__(self, rec, k):
        self.rec, self.k = rec, k
        A = rec.A
        self.t = float(A["t"][k])
        self.base, self.base_v, self.lift = A["base"][k].copy(), A["base_v"][k].copy(), float(A["lift"][k])
        self.qL, self.qR, self.qdL, self.qdR = A["tL"][k], A["tR"][k], A["tdL"][k], A["tdR"][k]
        self.tray_p, self.tray_R, self.tray_q = A["tray_p"][k].copy(), A["tray_R"][k].reshape(3, 3).copy(), A["tray_q"][k].copy()
        self.cup_p, self.cup_R, self.cup_q = A["cup_p"][k].copy(), A["cup_R"][k].reshape(3, 3).copy(), A["cup_q"][k].copy()
        names = rec.meta.get("people_names", [])
        self.people = [(n, A["ppl_xy"][k, i].copy(), A["ppl_v"][k, i].copy()) for i, n in enumerate(names)]
        self.obstacles = rec.obstacles
        self.occ = rec.occ()
        self.mu_true = rec.meta.get("mu_true", 0.25)


# ====================================================================================== recording
class Recording:
    """arrays A (per 50 Hz tick), head frames (10 Hz), wrist blobs, optional static-scene perception result.  save()/load() = npz + pickle."""

    def __init__(self, A: dict, meta: dict, frames: dict | None = None, zono_obs=None, obstacles=None):
        self.A, self.meta = A, meta
        self.frames = frames or {}                  # tick -> head frame dict
        self.zono_obs_result = zono_obs             # flagship perception result dict (cached pickle) or None
        self.obstacles = obstacles or []            # [(key, Zonotope)] static truth obstacles
        self._occ = None
        self._fk = None

    @property
    def n(self):
        return len(self.A["t"])

    @property
    def dt(self):
        return float(self.meta.get("dt", 0.02))

    def occ(self):
        if self._occ is None and self.meta.get("scene") not in (None, "flagship"):
            from world_model.v2.house.teleop import scene as SC
            self._occ = SC.load_occupancy(SC.get_scene(self.meta["scene"]))
        return self._occ

    def fk(self):
        if self._fk is None:
            self._fk = fk_for(self.meta.get("fk_scene", self.meta.get("scene", "h5")))
        return self._fk

    def view(self, k):
        return RecView(self, k)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(dict(A=self.A, meta=self.meta, frames=self.frames, zono_obs=self.zono_obs_result,
                             obstacles=[(k, z.c, z.G) for k, z in self.obstacles]), f, protocol=4)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["A"], d["meta"], d["frames"], d.get("zono_obs"), [(k, Zonotope(c, G)) for k, c, G in d.get("obstacles", [])])


class RecView:
    def __init__(self, rec: Recording, k: int):
        self.rec, self.k = rec, k
        self.t = float(rec.A["t"][k])
        self.fk = rec.fk()
        self._truth = None

    @property
    def truth(self):
        if self._truth is None:
            self._truth = RecTruth(self.rec, self.k)
        return self._truth

    def encoders(self):
        A, k = self.rec.A, self.k
        return dict(qL=A["qL"][k], qR=A["qR"][k], qdL=A["qdL"][k], qdR=A["qdR"][k], lift=float(A["lift_enc"][k]))

    def odom(self):
        A, k = self.rec.A, self.k
        return dict(pose=A["loc_pose"][k].copy(), sigma_xy=float(A["loc_sigma"][k]), v=A["loc_v"][k].copy())

    def head_frame(self):
        return self.rec.frames.get(self.k)

    def wrist(self):
        A, k = self.rec.A, self.k
        if not A["wrist_due"][k]:
            return None
        out = {}
        for i, s in enumerate(("left", "right")):
            r = A["wrist"][k, i]
            out[s] = None if not np.isfinite(r[0]) else (int(r[0]), float(r[1]), float(r[2]))
        return out

    def zono_obs(self):
        r = self.rec.zono_obs_result
        if r is None or self.t < float(self.rec.meta.get("zono_t", 0.0)):
            return None
        return r

    def visibility(self):
        """EVALUATION ONLY: per person (in_fov, visible, range) from the simulator's line-of-sight test."""
        v = self.rec.A.get("ppl_vis")
        return None if v is None else v[self.k]


# ====================================================================================== live view
class LiveView:
    """the same interface on a running TeleopSim.  `sensors` (record.LiveSensors) is needed for head_frame / wrist; encoders / odom / truth work without."""

    def __init__(self, sim, sensors=None, loc=None, fk: ArmFK | None = None, enc_rng=None):
        self.sim, self.sensors, self.loc = sim, sensors, loc
        self._fk = fk
        self.truth = LiveTruth(sim)
        self.rng = enc_rng
        self.k = sim.k
        self.t = sim.t

    @property
    def fk(self):
        if self._fk is None:
            self._fk = fk_for_sim(self.sim)
        return self._fk

    def encoders(self):
        from world_model.v2.house.teleop.arm import ENC_Q, ENC_V
        tr = self.truth
        r = self.rng
        n = (lambda s: r.normal(0, s, 6)) if r is not None else (lambda s: np.zeros(6))
        return dict(qL=tr.qL + n(ENC_Q), qR=tr.qR + n(ENC_Q), qdL=tr.qdL + n(ENC_V), qdR=tr.qdR + n(ENC_V), lift=self.sim.lift_actual())

    def odom(self):
        if self.loc is None:
            raise RuntimeError("LiveView.odom needs a PoseEstimator (loc=...)")
        return dict(pose=self.loc.pose.copy(), sigma_xy=float(self.loc.sigma_xy), v=self.loc.v_odo.copy())

    def head_frame(self):
        return None if self.sensors is None else self.sensors.head_frame(self.sim)

    def wrist(self):
        return None if self.sensors is None else self.sensors.wrist(self.sim)

    def zono_obs(self):
        return getattr(self.sim, "_ocs_zono_obs", None)


def fk_for_sim(sim) -> ArmFK:
    key = ("sim", id(sim.m))
    if key not in _FK_CACHE:
        _FK_CACHE[key] = ArmFK(sim)
    return _FK_CACHE[key]
