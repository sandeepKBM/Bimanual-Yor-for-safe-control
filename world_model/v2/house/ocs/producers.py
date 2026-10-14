"""ocs/producers.py -- sensors -> WorldState.  SIMULATION ONLY.

  TruthProducer        every field from the simulator, Source.TRUTH (ablations / evaluation).  chain=False: tray, cup, people in the WORLD frame with the
                       raw MuJoCo numbers (bit-exact shield inputs, experiment E1); chain=True: tray -> "robot", cup -> "tray" (for rollouts).
  BaseProducer         house/loc PoseEstimator output (wheel odometry + ZED VIO), Source.ODOM; encoders -> q, lift.  Falls back to TRUTH (flagged) without odom.
  TrayFKProducer       arm encoders -> FK of the two hands o the NOMINAL weld relpose (engineering prior) -> tray pose in the base frame, Source.FK.
  CupWristProducer     wrist-camera red-blob centroids (RGB colour threshold, grasp/wrist.WristCam.seg_cup_proprio) -> rays from the FK camera poses -> the
                       tray plane -> cup position in the TRAY frame, Source.WRIST.  Orientation = upright relative to the tray (a contact PRIOR; unobservable
                       from a blob centroid) with a rotation bound; z = on the tray surface (contact prior).
  PeopleProducer       head-camera depth frames -> people_track DepthDetector (background subtraction vs the house model) -> MultiTracker (KF) -> PERSON
                       objects, Source.TRACKER.  The frame is expressed in the ESTIMATED map frame (world placement of the depth points uses the
                       odometry pose, not the truth) so people, tray and cup share the same localisation error.
  ObstacleZonoProducer perceived stereo zonotopes (flagship/perception result, cached pickle or live) -> OBSTACLE / SURFACE objects with ZonoShape + the
                       pipeline's own p95 stereo inflation as `unc`, Source.STEREO.
  NamesProducer        names: ground-truth association (Source.TRUTH, FLAGGED by privileged()) because the VLM grounding (house/lang) needs a live daemon; the
                       fixed engineering names tray / cup are Source.PRIOR.
Every producer sets source / t_meas / visible / age honestly; uncertainty sets come from a `Calib` (split-conformal radii fitted on calibration seeds)
or, without one, from engineering priors (k-sigma of the estimator).
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.sets.ir import Pose, Unc, BoxShape, ZonoShape
from world_model.v2.tray_rep.sets.zono import Zonotope, ellipsoid_outer

from .state import Source, Kind, Belief, ObjectState, RobotState, FreeSpace, Relation, WorldState, Producer
from . import util as U

TICK = 0.02
EYE3 = np.eye(3)


# ====================================================================================== calibration container
def cq(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return float(s[k - 1]) if 1 <= k <= n else float("inf")


@dataclass
class Calib:
    """split-conformal radii of the OCS uncertainty sets.  scalars: dict name -> value; person: people_track ConformalMargin (or None)."""
    scalars: dict = field(default_factory=dict)
    person: object = None
    alpha: float = 0.05

    def get(self, name, default):
        return float(self.scalars.get(name, default))

    def has(self, name):
        return name in self.scalars

    def to_json(self):
        return dict(scalars=self.scalars, alpha=self.alpha, person=None if self.person is None else self.person.to_json())

    @classmethod
    def from_json(cls, j):
        pm = None
        if j.get("person"):
            from world_model.v2.house.people_track.conformal import ConformalMargin
            pm = ConformalMargin.from_json(j["person"])
        return cls(dict(j["scalars"]), pm, j.get("alpha", 0.05))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text()))


# ====================================================================================== truth
class TruthProducer(Producer):
    name = "truth"

    def __init__(self, chain: bool = False, with_free: bool = True, cup: str = "flagship"):
        self.chain, self.with_free, self.cup = chain, with_free, cup
        self._free = None

    def reset(self, view=None):
        self._free = None

    def update(self, ws: WorldState, view, t: float):
        tr = view.truth
        x, y, yaw = (float(v) for v in tr.base)
        bp = U.base_pose(x, y, yaw)
        ws.t = t
        ws.robot = RobotState(base=bp, base_unc=Unc(), base_vel=np.asarray(tr.base_v, float).copy(), base_source=Source.TRUTH, lift_h=float(tr.lift),
                              q={"left": np.asarray(tr.qL, float).copy(), "right": np.asarray(tr.qR, float).copy()},
                              qd={"left": np.asarray(tr.qdL, float).copy(), "right": np.asarray(tr.qdR, float).copy()},
                              q_source=Source.TRUTH, yaw=yaw)
        Rb, pb = np.asarray(bp.R), np.asarray(bp.p)
        tray_shape = BoxShape(np.array([U.HX, U.HY, 0.003]))
        mu = Belief(float(tr.mu_true), float(tr.mu_true), float(tr.mu_true), None, Source.TRUTH, "simulator friction")
        if self.chain:
            Pt = Pose(Rb.T @ np.asarray(tr.tray_R), Rb.T @ (np.asarray(tr.tray_p) - pb))
            Rt, pt = np.asarray(tr.tray_R), np.asarray(tr.tray_p)
            Pc = Pose(Rt.T @ np.asarray(tr.cup_R), Rt.T @ (np.asarray(tr.cup_p) - pt))
            tray = ObjectState("tray", Kind.PAYLOAD, tray_shape, Pt, parent="robot", name="tray", dynamics="attached", source=Source.TRUTH, t_meas=t,
                               extra={"quat": np.asarray(tr.tray_q, float)}, name_source=Source.PRIOR)
            cup = ObjectState("cup", Kind.CONTENT, ZonoShape(U.cup_shape(self.cup)), Pc, parent="tray", name="cup", dynamics="slip_on_support",
                              source=Source.TRUTH, t_meas=t, params={"mu": mu}, extra={"quat": np.asarray(tr.cup_q, float)}, name_source=Source.PRIOR)
        else:
            tray = ObjectState("tray", Kind.PAYLOAD, tray_shape, Pose(np.asarray(tr.tray_R, float).copy(), np.asarray(tr.tray_p, float).copy()), parent=None,
                               name="tray", dynamics="static", source=Source.TRUTH, t_meas=t, extra={"quat": np.asarray(tr.tray_q, float).copy()},
                               name_source=Source.PRIOR)
            cup = ObjectState("cup", Kind.CONTENT, ZonoShape(U.cup_shape(self.cup)), Pose(np.asarray(tr.cup_R, float).copy(), np.asarray(tr.cup_p, float).copy()),
                              parent=None, name="cup", dynamics="static", source=Source.TRUTH, t_meas=t, params={"mu": mu},
                              extra={"quat": np.asarray(tr.cup_q, float).copy()}, name_source=Source.PRIOR)
        ws.objects = {"tray": tray, "cup": cup}
        for name, xy, v in tr.people:
            ws.objects[name] = ObjectState(name, Kind.PERSON, ZonoShape(U.person_shape()), Pose(EYE3.copy(), np.array([xy[0], xy[1], 0.0])),
                                           vel=np.array([v[0], v[1], 0.0, 0.0, 0.0, 0.0]), name=name, dynamics="const_vel", source=Source.TRUTH, t_meas=t,
                                           name_source=Source.TRUTH)
        for key, Z in tr.obstacles:
            ws.objects[key] = ObjectState(key, Kind.OBSTACLE, ZonoShape(Zonotope(np.zeros(3), Z.G)), Pose(EYE3.copy(), np.asarray(Z.c, float).copy()),
                                          name=key, dynamics="static", source=Source.TRUTH, t_meas=t, name_source=Source.TRUTH)
        ws.relations = [Relation("cup", "on", "tray", 1.0, Source.TRUTH)]
        if self.with_free and tr.occ is not None:
            if self._free is None:
                occ = tr.occ
                self._free = FreeSpace(origin=np.array([occ.x0, occ.y0]), res=float(occ.res), logodds=np.where(occ.occ, 5.0, -5.0).astype(np.float32),
                                       last_seen=np.zeros(occ.occ.shape, np.float32), source=Source.TRUTH)
            ws.free = self._free
        ws.meta["producers"] = ws.meta.get("producers", []) + ["truth"]


# ====================================================================================== base
class BaseProducer(Producer):
    name = "base"

    def __init__(self, calib: Calib | None = None, k_sigma: float = 2.0, rot_prior: float = 0.03):
        self.calib, self.k_sigma, self.rot_prior = calib, k_sigma, rot_prior

    def reset(self, view=None):
        pass

    def update(self, ws, view, t):
        enc = view.encoders()
        try:
            od = view.odom()
        except Exception:
            od = None
        if od is not None:
            x, y, yaw = (float(v) for v in od["pose"])
            if self.calib is not None and self.calib.has("base_k"):
                r, rot = self.calib.get("base_k", self.k_sigma) * float(od["sigma_xy"]), self.calib.get("base_rot", self.rot_prior)
            else:
                r, rot = self.k_sigma * float(od["sigma_xy"]), self.rot_prior
            src = Source.ODOM
            vel = np.asarray(od["v"], float).copy()
            unc = Unc(U.ball_zono(r), float(rot))
        else:
            x, y, yaw = (float(v) for v in view.truth.base)
            src, vel, unc = Source.TRUTH, np.asarray(view.truth.base_v, float).copy(), Unc()
        ws.t = t
        ws.robot = RobotState(base=U.base_pose(x, y, yaw), base_unc=unc, base_vel=vel, base_source=src, lift_h=float(enc["lift"]),
                              q={"left": np.asarray(enc["qL"], float).copy(), "right": np.asarray(enc["qR"], float).copy()},
                              qd={"left": np.asarray(enc["qdL"], float).copy(), "right": np.asarray(enc["qdR"], float).copy()},
                              q_source=Source.ENCODER, yaw=yaw)
        ws.meta["producers"] = ws.meta.get("producers", []) + ["base"]


# ====================================================================================== tray (FK)
class TrayFKProducer(Producer):
    name = "tray_fk"

    def __init__(self, calib: Calib | None = None, pos_prior: float = 0.005, rot_prior: float = 0.01):
        self.calib, self.pos_prior, self.rot_prior = calib, pos_prior, rot_prior
        self.t0 = None

    def reset(self, view=None):
        self.t0 = None

    def update(self, ws, view, t):
        enc = view.encoders()
        p, R = view.fk.tray(enc["qL"], enc["qR"], enc["lift"])
        if self.t0 is None:
            self.t0 = t
        c = self.calib
        r = c.get("tray_pos", self.pos_prior) if c is not None else self.pos_prior
        rot = c.get("tray_rot", self.rot_prior) if c is not None else self.rot_prior
        ws.objects["tray"] = ObjectState("tray", Kind.PAYLOAD, BoxShape(np.array([U.HX, U.HY, 0.003])), Pose(R, p), unc=Unc(U.ball_zono(r), float(rot)),
                                         parent="robot", name="tray", name_source=Source.PRIOR, dynamics="attached", source=Source.FK, t_meas=t, visible=True,
                                         age=t - self.t0, extra={"grasp_offset": "nominal weld relpose (prior)"})
        ws.meta["producers"] = ws.meta.get("producers", []) + ["tray_fk"]


# ====================================================================================== cup (wrist camera)
def _blob_ray(spec, u, v):
    """camera-frame unit direction of pixel (u, v) (continuous pixel coordinates, the HeadCamera.project convention); the camera looks along -z."""
    d = np.array([(u + 0.5 - spec.W / 2.0) / spec.f_px, -(v + 0.5 - spec.H / 2.0) / spec.f_px, -1.0])
    return d / np.linalg.norm(d)


class CupWristProducer(Producer):
    name = "cup_wrist"

    def __init__(self, calib: Calib | None = None, plane_z: float | None = None, pos_prior: float = 0.02, rot_prior: float = 0.10,
                 mu_prior=(0.18, 0.35), hold_s: float = 0.15, min_px: int = 3000, gate=(U.HX + 0.05, U.HY + 0.05), cup: str = "flagship"):
        self.cup = cup
        if plane_z is None:
            plane_z = U.cup_dims(cup)[1] / 2          # the blob centroid height = mid-height of the cup
        from world_model.v2.tray_rep.perception import depth_sim as D
        self.spec = D.CAMERAS["wrist_l"]
        self.calib, self.plane_z, self.pos_prior, self.rot_prior = calib, plane_z, pos_prior, rot_prior
        self.mu_prior, self.hold_s, self.min_px, self.gate = mu_prior, hold_s, min_px, gate
        self.last = None            # (t, p_tray (3,), sides)
        self.first_t = None
        self.raw = {}
        self._uv = {}
        self.uv = {}

    def reset(self, view=None):
        self.last, self.first_t, self.raw, self._uv, self.uv = None, None, {}, {}, {}

    def _correct(self, side, p, uv):
        """blob-bias correction fitted on the calibration seeds: affine in the blob centroid pixel (the truncation of the blob at the image border
        makes the centroid ray error a smooth function of (u, v)), else a constant per-side bias, else nothing."""
        c = self.calib
        if c is None:
            return p
        A = c.scalars.get("cup_affine")
        if A is not None and side in A and uv is not None:
            return p - np.r_[np.asarray(A[side], float) @ np.array([1.0, uv[0], uv[1]]), 0.0]
        b = c.scalars.get("cup_bias")
        return p - (np.zeros(3) if b is None else np.asarray(b[side], float))

    def raw_estimate(self, view, wr):
        """per side: cup position in the tray frame from the blob ray (tray plane at plane_z) or None."""
        enc = view.encoders()
        pt, Rt = view.fk.tray(enc["qL"], enc["qR"], enc["lift"])
        n = Rt[:, 2]
        out = {}
        for side, blob in wr.items():
            if blob is None or blob[0] < self.min_px:
                continue
            org, Rc = view.fk.wrist_cam(side, enc["qL"], enc["qR"], enc["lift"])
            dirw = Rc @ _blob_ray(self.spec, blob[1], blob[2])
            den = float(n @ dirw)
            if abs(den) < 1e-6:
                continue
            tt = float(n @ (pt + n * self.plane_z - org)) / den
            if tt <= 0:
                continue
            x = org + tt * dirw
            self._uv[side] = (float(blob[1]), float(blob[2]))
            pl = Rt.T @ (x - pt)
            if abs(pl[0]) > self.gate[0] or abs(pl[1]) > self.gate[1]:       # off the tray: impossible for a cup standing on it (truncated / grazing blob)
                continue
            out[side] = pl
        return out

    def update(self, ws, view, t):
        if self.first_t is None:
            self.first_t = t
        wr = view.wrist()
        if wr is not None:
            est = self.raw_estimate(view, wr)
            if est:
                p = np.mean([self._correct(side, q, self._uv.get(side)) for side, q in est.items()], 0)
                self.raw = {**self.raw, **est}
                self.uv = {**self.uv, **{k_: v_ for k_, v_ in self._uv.items() if k_ in est}}                      # latest per-side blob estimate (tray frame, before bias correction): calibration input
                self.last = (t, p, sorted(est))
        if self.last is None:
            return
        tl, p, sides = self.last
        c = self.calib
        r = c.get("cup_pos", self.pos_prior) if c is not None else self.pos_prior
        rot = c.get("cup_rot", self.rot_prior) if c is not None else self.rot_prior
        age_since = t - tl
        vis = age_since <= self.hold_s
        if not vis and c is not None and c.has("cup_pos_occ"):
            r = c.get("cup_pos_occ", r)
        pc = np.array([p[0], p[1], 0.0])                           # on the tray surface (contact prior)
        mu = Belief(self.mu_prior[0], self.mu_prior[1], None, None, Source.PRIOR, "ShieldCfg prior interval")
        ws.objects["cup"] = ObjectState("cup", Kind.CONTENT, ZonoShape(U.cup_shape(self.cup)), Pose(EYE3.copy(), pc),
                                        unc=Unc(Zonotope(np.zeros(3), np.diag([r, r, 0.003])), float(rot)), parent="tray", name="cup",
                                        name_source=Source.PRIOR, params={"mu": mu}, dynamics="slip_on_support", source=Source.WRIST, t_meas=tl, visible=vis,
                                        age=t - self.first_t, extra={"sides": sides, "raw": {k: v.copy() for k, v in self.raw.items()}, "uv": dict(self.uv), "orientation": "upright-on-tray prior"})
        ws.relations = [r_ for r_ in ws.relations if not (r_.a == "cup" and r_.rel == "on")] + [Relation("cup", "on", "tray", 1.0 if vis else 0.5, Source.WRIST)]
        ws.meta["producers"] = ws.meta.get("producers", []) + ["cup_wrist"]


# ====================================================================================== people (head depth -> tracker)
class _ReplayCam:
    """the part of people_track.camera.HeadCamera that the detectors use, backed by a stored frame."""

    def __init__(self, spec):
        self.spec = spec

    def static_depth(self, frame, idx):
        return frame["zs"][idx].astype(np.float64)


def _to_est_frame(fr: dict, base_est) -> dict:
    """re-express a stored head frame (world placement with the TRUE base pose, recorded in fr['base_true']) in the estimated map frame:
    rigid planar transform T_est o T_true^-1 applied to the points, the camera origin and the camera rotation."""
    bt = np.asarray(fr["base_true"], float)
    be = np.asarray(base_est, float)
    d = be[2] - bt[2]
    c, s = math.cos(d), math.sin(d)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = fr["pts"].astype(np.float64)
    pts = (pts - np.array([bt[0], bt[1], 0.0])) @ Rz.T + np.array([be[0], be[1], 0.0])
    org = (np.asarray(fr["origin"], float) - np.array([bt[0], bt[1], 0.0])) @ Rz.T + np.array([be[0], be[1], 0.0])
    return dict(origin=org, R=Rz @ np.asarray(fr["R"], float), pts=pts, z=fr["z"].astype(np.float64), pix=fr["pix"].astype(int), zs=fr["zs"],
                person=fr["person"].astype(int), geom=None, tick=fr["tick"])


class PeopleProducer(Producer):
    name = "people"

    def __init__(self, calib: Calib | None = None, k_sigma: float = 2.0, latency_ticks: int = 3, tracker_cfg=None, vis_s: float = 0.15, r_max: float = 4.5):
        from world_model.v2.house.people_track import camera as CAM, detectors as DET
        from world_model.v2.house.people_track.tracker import MultiTracker, TrackerCfg
        self.CAM, self.DET, self.MT = CAM, DET, MultiTracker
        self.tcfg = tracker_cfg or TrackerCfg()
        self.spec = CAM.head_spec(W=160, H=90, rate_hz=10.0, latency_ticks=latency_ticks)
        self.cam = _ReplayCam(self.spec)
        self.det = DET.DepthDetector(occ=None, use_static_map=False, bg=True, r_max=r_max)
        self.latency, self.calib, self.k_sigma, self.vis_s = latency_ticks, calib, k_sigma, vis_s
        self.reset()

    def reset(self, view=None):
        self.trk = self.MT(self.tcfg)
        self.queue = deque()
        self.cam_xy = None
        self.n_frames = 0
        self.first_seen = {}

    def _unc_rel(self, tr, k, Rb2):
        """position set of the track in the ROBOT frame axes (Rb2 = 2x2 base rotation): calibrated Mahalanobis ellipse (+ floor) when a Calib with
        `person_q` is given, else the k-sigma ellipse of the track covariance.  Also returns (since_upd, age, range)."""
        since = max(0, k - tr.last_upd) * TICK
        age = max(0, k - tr.born) * TICK
        rng = None if self.cam_xy is None else float(np.hypot(tr.x[0] - self.cam_xy[0], tr.x[1] - self.cam_xy[1]))
        Prel = Rb2.T @ tr.P[:2, :2] @ Rb2
        if self.calib is not None and self.calib.has("person_q"):
            Z = U.ellipse_zono(Prel, self.calib.get("person_q", self.k_sigma))
            fl = self.calib.get("person_floor", 0.0)
            if fl > 0:
                Z = Z.minkowski(U.disc_zono(fl))
            return Unc(Z, 0.0), since, age, rng
        return Unc(U.ellipse_zono(Prel, self.k_sigma), 0.0), since, age, rng

    def update(self, ws, view, t):
        k = view.k
        self.trk.predict_to(k)
        od = view.odom()["pose"]
        fr = view.head_frame()
        if fr is not None:
            frame = _to_est_frame(fr, od)
            self.cam_xy = frame["origin"][:2].copy()
            dets = self.det.detect(frame, self.cam, None)
            self.queue.append((k + self.latency, k, dets))
            self.n_frames += 1
        delivered = False
        while self.queue and self.queue[0][0] <= k:
            _, kc, dets = self.queue.popleft()
            self.trk.process_frame(dets, kc)
            delivered = True
        if not delivered:
            self.trk.predict_frame_tick(k)
        Rb3 = U.rz(float(od[2]))
        Rb2 = Rb3[:2, :2]
        for tr in self.trk.output(confirmed_only=True):
            unc, since, age, rng = self._unc_rel(tr, k, Rb2)
            P4 = tr.P.copy()
            cov = np.zeros((3, 3))
            cov[:2, :2] = tr.P[:2, :2]
            sv = float(math.sqrt(max(tr.P[2, 2], tr.P[3, 3], 0.0)))
            prel = Rb2.T @ (np.array([tr.x[0], tr.x[1]]) - np.asarray(od[:2], float))              # person position in the ROBOT frame (what the camera measured)
            oid = tr.name
            ws.objects[oid] = ObjectState(oid, Kind.PERSON, ZonoShape(U.person_shape()), Pose(Rb3.T.copy(), np.array([prel[0], prel[1], 0.0])), unc=unc,
                                          vel=np.array([tr.x[2], tr.x[3], 0.0, 0.0, 0.0, 0.0]), cov=cov, parent="robot", dynamics="const_vel", source=Source.TRACKER,
                                          t_meas=t - since, visible=since <= self.vis_s, age=age,
                                          extra={"P4": P4, "since_upd_s": since, "age_s": age, "range_m": rng, "track_id": tr.id, "sigma_v": max(sv, 0.1)})
        ws.meta["producers"] = ws.meta.get("producers", []) + ["people"]


# ====================================================================================== static obstacles (stereo zonotopes)
class ObstacleZonoProducer(Producer):
    name = "obstacles"

    def __init__(self, kinds=("obj", "support"), pose_aware=False, loc_k=2.0, drift_xy=0.02, drift_yaw=0.003):
        """pose_aware (flag, default False = unchanged): the zonotopes were built with the ESTIMATED capture pose (flagship.perception.Twin.capture(pose_est=...):
        r["pose_est"] present, Zi already inflated by the pose uncertainty AT CAPTURE).  The producer then adds the RELATIVE odometry drift since the capture
        (the static zonotopes live in the estimated frame frozen at capture time):  k (drift_xy sqrt(s) + range * drift_yaw sqrt(s)),  s = odometry path length
        since the first use, as extra ball uncertainty.  A legacy (true-frame) result is flagged extra['frame']='true(legacy)'."""
        self.kinds = kinds
        self.pose_aware, self.loc_k, self.drift_xy, self.drift_yaw = pose_aware, loc_k, drift_xy, drift_yaw
        self._objs = None
        self._t = None
        self._s = 0.0
        self._last = None

    def reset(self, view=None):
        self._objs, self._t = None, None
        self._s, self._last = 0.0, None

    def _build(self, r, t):
        out = {}
        for z in r["zonos"]:
            if z["kind"] not in self.kinds:
                continue
            Zi = z["Zi"]
            c, G = np.asarray(Zi["c"], float), np.asarray(Zi["G"], float)
            d = float(z["delta"])
            if G.shape[1] >= 3 and np.allclose(G[:, -3:], d * np.eye(3)):
                Gn, unc = G[:, :-3], Unc(U.ball_zono(d), 0.0)           # Zi = Z (+) cube(delta): split back into shape + set
            else:
                Gn, unc = G, Unc()
            oid = f"z{z['id']}"
            kind = Kind.SURFACE if z["kind"] == "support" else Kind.OBSTACLE
            out[oid] = ObjectState(oid, kind, ZonoShape(Zonotope(np.zeros(3), Gn)), Pose(EYE3.copy(), c.copy()), unc=unc, dynamics="static",
                                   source=Source.STEREO, t_meas=float(r.get("t_sim", t)), visible=True,
                                   extra={"gt_key": z.get("gt_key"), "n_pts": z.get("n_pts"), "range_m": z.get("range_m"), "delta": d})
        return out

    def update(self, ws, view, t):
        r = view.zono_obs()
        if r is None:
            return
        if self._objs is None:
            self._objs = self._build(r, t)
        if self.pose_aware:
            try:
                p = np.asarray(view.odom()["pose"], float)[:2]
                if self._last is not None:
                    self._s += float(np.hypot(*(p - self._last)))
                self._last = p
            except Exception:
                pass
            sq = math.sqrt(self._s)
            for oid, o in self._objs.items():
                rng = float(o.extra.get("range_m") or 1.5)
                ex = self.loc_k * (self.drift_xy * sq + rng * math.tan(min(self.drift_yaw * sq, 0.3)))
                d0 = float(o.extra.get("delta", 0.0)) if (o.unc is not None and o.unc.pos is not None) else 0.0
                ws.objects[oid] = replace(o, age=t - o.t_meas, unc=Unc(U.ball_zono(d0 + ex), 0.0) if (d0 > 0 or ex > 0) else o.unc,
                                          extra=dict(o.extra, drift_extra_m=ex, frame="est(capture)" if "pose_est" in r else "true(legacy)"))
            ws.meta["producers"] = ws.meta.get("producers", []) + ["obstacles"]
            return
        for oid, o in self._objs.items():
            ws.objects[oid] = replace(o, age=t - o.t_meas)
        ws.meta["producers"] = ws.meta.get("producers", []) + ["obstacles"]


# ====================================================================================== names
class NamesProducer(Producer):
    """names for people (nearest ground-truth person within 1 m) and perceived obstacles (the pipeline's gt_key): Source.TRUTH, flagged.
    Replace with house/lang grounding (Source.VLM) when a daemon is up."""
    name = "names"

    def __init__(self, gate: float = 1.0):
        self.gate = gate

    def reset(self, view=None):
        pass

    def update(self, ws, view, t):
        truth = view.truth.people
        for o in ws.objects.values():
            if o.kind == Kind.PERSON and o.source == Source.TRACKER and truth:
                p = np.asarray(ws.world_pose(o.oid).p)[:2]
                d = [np.linalg.norm(p - np.asarray(xy)) for _, xy, _ in truth]
                i = int(np.argmin(d))
                if d[i] < self.gate:
                    o.name, o.name_conf, o.name_source = truth[i][0], 1.0, Source.TRUTH
            elif o.kind in (Kind.OBSTACLE, Kind.SURFACE) and o.source == Source.STEREO and o.extra.get("gt_key"):
                o.name, o.name_conf, o.name_source = o.extra["gt_key"], 1.0, Source.TRUTH


# ====================================================================================== pipeline
class Pipeline:
    def __init__(self, producers):
        self.producers = list(producers)

    def reset(self, view=None):
        for p in self.producers:
            p.reset(view)

    def step(self, view) -> WorldState:
        ws = WorldState(t=view.t, robot=RobotState(base=U.base_pose(0, 0, 0)))
        for p in self.producers:
            p.update(ws, view, view.t)
        return ws


def perception_pipeline(calib: Calib | None = None, people: bool = True, cup: bool = True, obstacles: bool = False, names: bool = True,
                        base: bool = True, cup_kind: str = "flagship") -> Pipeline:
    ps = []
    if base:
        ps.append(BaseProducer(calib))
    ps.append(TrayFKProducer(calib))
    if cup:
        ps.append(CupWristProducer(calib, cup=cup_kind))
    if people:
        ps.append(PeopleProducer(calib))
    if obstacles:
        ps.append(ObstacleZonoProducer())
    if names:
        ps.append(NamesProducer())
    return Pipeline(ps)
