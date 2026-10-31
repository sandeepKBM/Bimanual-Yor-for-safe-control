"""ocs/anchored.py -- general SE(3)-ANCHORED set tracker for rigid objects.  SIMULATION ONLY.

An object is tracked as
    shape   a zonotope in the object's OWN frame, fitted once (ObjectModel.from_points: hull + LP fit of an onboarding cloud) or given by a model;
            never refit per frame.  The world zonotope is the shape moved by the pose (exact: c = R c0 + t, G = R G0) -- ir / consumers place it.
    pose    SE(3) in the parent's frame (a support such as "tray", the carrier "robot", or the world).
    set     hull of a bank of motion hypotheses (+) the radius of the last measurement; rotation bound from the source (or the contact prior).

Pieces (each swappable, nothing object-specific in the producer):
  ObjectModel        what is known about the object independent of the sensor: shape, parent, motion key, physical params, contact prior.
  MeasurementSource  view -> Measurement (position [+ rotation] in the parent frame + set radius) or None.
                       BlobPlaneSource  colour-blob centroid ray of the wrist cameras -> plane parallel to the support (generalises
                                        producers.CupWristProducer.raw_estimate; same numbers, see tests).
                       FoundationPose / stereo-cluster sources implement the same two methods (reset, measure).
  Propagator         motion hypotheses in the parent frame between measurements, driven by MEASURED robot motion only (odometry twist, FK).
                       StaticProp        pose constant in the parent frame (static / attached objects).
                       SupportSlideProp  Coulomb sliding on the support (ocs.dynamics._slide_traj) under the support's tilt and acceleration;
                                         one hypothesis per (mu in {lo, hi}) x (disturbance direction) + the mu_mid nominal (index 0).
                                         Velocities persist across measurements: a body that leaves the view while sliding keeps sliding.
  AnchoredProducer   per tick: step the hypotheses; on a measurement, condition them (positions re-anchored to the measurement; a hypothesis whose
                     prediction lies farther than k_fals * r from it is inconsistent with the data and restarts from the nominal velocity);
                     emit the ObjectState (pose = nominal hypothesis, set = hull (+) r, extra["v_rel"] for the rollout).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from world_model.v2.tray_rep.sets.ir import Pose, Unc, ZonoShape
from world_model.v2.tray_rep.sets.zono import Zonotope

from .state import Source, Kind, Belief, ObjectState, Relation, WorldState, Producer
from . import util as U, dynamics as D

EYE3 = np.eye(3)
TICK = 0.02


# ====================================================================================== object model
@dataclass
class ObjectModel:
    oid: str
    kind: Kind
    shape: Zonotope                         # object frame; for an object on a support the origin is the contact point (base centre)
    parent: str | None = None               # support / carrier oid, "robot", or None (world)
    dynamics: str = "static"                # ocs.dynamics key the downstream rollout uses
    params: dict = field(default_factory=dict)
    name: str | None = None
    name_source: Source | None = Source.PRIOR
    on_support: bool = False                # position-only measurements: on the parent's top surface (z = 0) and upright (contact prior)
    z_unc: float = 0.003                    # set half-height of an on-support object [m]

    @classmethod
    def from_points(cls, oid, kind, pts_obj, order=2, **kw):
        """fit the object-frame zonotope ONCE from an onboarding cloud already expressed in the object frame (hull + LP fit)."""
        from world_model.v2.perception_zono.fit import fit_hull_zono
        return cls(oid, kind, fit_hull_zono(np.asarray(pts_obj, float), order=order), **kw)

    def height(self) -> float:
        return float(self.shape.support(np.array([0.0, 0.0, 1.0])))

    def world_zonotope(self, R, p) -> Zonotope:
        """the anchored shape at pose (R, p): exact image of the object-frame zonotope."""
        return self.shape.linear(np.asarray(R, float), np.asarray(p, float))


@dataclass
class Measurement:
    t: float
    p: np.ndarray                           # (3,) position in the parent frame
    r: float                                # set radius of the position [m] (calibrated or prior)
    R: np.ndarray | None = None             # (3, 3) parent-frame rotation, if the source measures it
    r_rot: float = 0.10                     # rotation bound [rad] (of R, or of the contact prior when R is None)
    source: Source = Source.PRIOR
    extra: dict = field(default_factory=dict)


# ====================================================================================== measurement sources
class MeasurementSource:
    name = "source"
    source = Source.PRIOR

    def reset(self, view=None):
        pass

    def measure(self, view, ws: WorldState, model: ObjectModel, t: float) -> Measurement | None:
        raise NotImplementedError


class BlobPlaneSource(MeasurementSource):
    """wrist-camera colour blobs -> rays (FK camera poses) -> plane parallel to the parent support at `plane_z` -> parent-frame position.
    `blobs(view)` returns {side: (n_px, u, v) | None} (default: view.wrist(), the recorded / live cup blob).  Calibration keys
    <prefix>_affine / <prefix>_bias (blob-bias correction), <prefix>_pos (radius), <prefix>_rot (rotation bound of the contact prior)."""
    name = "blob_plane"
    source = Source.WRIST

    def __init__(self, calib=None, prefix: str = "obj", plane_z: float | None = None, gate=None, min_px: int = 3000, cam: str = "wrist_l",
                 pos_prior: float = 0.02, rot_prior: float = 0.10, blobs=None):
        from world_model.v2.tray_rep.perception import depth_sim as DS
        self.spec = DS.CAMERAS[cam]
        self.calib, self.prefix, self.plane_z, self.gate, self.min_px = calib, prefix, plane_z, gate, min_px
        self.pos_prior, self.rot_prior = pos_prior, rot_prior
        self.blobs = blobs or (lambda view: view.wrist())
        self.raw, self.uv = {}, {}

    def reset(self, view=None):
        self.raw, self.uv = {}, {}

    def _ray(self, u, v):
        d = np.array([(u + 0.5 - self.spec.W / 2.0) / self.spec.f_px, -(v + 0.5 - self.spec.H / 2.0) / self.spec.f_px, -1.0])
        return d / np.linalg.norm(d)

    def raw_estimate(self, view, ws, model, blobs):
        enc = view.encoders()
        pt, Rt = view.fk.tray(enc["qL"], enc["qR"], enc["lift"]) if model.parent == "tray" else (None, None)
        if pt is None:
            raise NotImplementedError("BlobPlaneSource: only a tray-supported object is wired (the support pose comes from arm FK)")
        n = Rt[:, 2]
        hz = model.height() / 2.0 if self.plane_z is None else self.plane_z
        out, uv = {}, {}
        for side, blob in blobs.items():
            if blob is None or blob[0] < self.min_px:
                continue
            org, Rc = view.fk.wrist_cam(side, enc["qL"], enc["qR"], enc["lift"])
            dirw = Rc @ self._ray(blob[1], blob[2])
            den = float(n @ dirw)
            if abs(den) < 1e-6:
                continue
            tt = float(n @ (pt + n * hz - org)) / den
            if tt <= 0:
                continue
            pl = Rt.T @ (org + tt * dirw - pt)
            uv[side] = (float(blob[1]), float(blob[2]))
            if self.gate is not None and (abs(pl[0]) > self.gate[0] or abs(pl[1]) > self.gate[1]):
                continue
            out[side] = pl
        return out, uv

    def _correct(self, side, p, uv):
        c = self.calib
        if c is None:
            return p
        A = c.scalars.get(f"{self.prefix}_affine")
        if A is not None and side in A and uv is not None:
            return p - np.r_[np.asarray(A[side], float) @ np.array([1.0, uv[0], uv[1]]), 0.0]
        b = c.scalars.get(f"{self.prefix}_bias")
        return p - (np.zeros(3) if b is None else np.asarray(b[side], float))

    def measure(self, view, ws, model, t):
        bl = self.blobs(view)
        if bl is None:
            return None
        est, uv = self.raw_estimate(view, ws, model, bl)
        if not est:
            return None
        p = np.mean([self._correct(s, q, uv.get(s)) for s, q in est.items()], 0)
        self.raw = {**self.raw, **est}
        self.uv = {**self.uv, **{s: uv[s] for s in est}}
        if model.on_support:
            p = np.array([p[0], p[1], 0.0])
        c = self.calib
        r = c.get(f"{self.prefix}_pos", self.pos_prior) if c is not None else self.pos_prior
        rot = c.get(f"{self.prefix}_rot", self.rot_prior) if c is not None else self.rot_prior
        return Measurement(t, p, float(r), None, float(rot), self.source,
                           dict(sides=sorted(est), raw={k: v.copy() for k, v in self.raw.items()}, uv=dict(self.uv), blob_t=float(t)))


# ====================================================================================== propagators
@dataclass
class Hyps:
    P: np.ndarray                           # (N, 3) parent-frame positions; row 0 = nominal
    V: np.ndarray                           # (N, 2) in-plane velocities (support frame)
    mu: np.ndarray                          # (N,)
    dirs: np.ndarray                        # (N, 3) unit disturbance directions (world), zero for the nominal / undisturbed


@dataclass
class SupportCtx:
    R_prev: np.ndarray                      # world rotation of the support at the previous tick
    R_now: np.ndarray
    a_world: np.ndarray                     # (3,) world acceleration of the support point under the object
    dt: float


class StaticProp:
    """pose constant in the parent frame; the set is the measurement radius only."""
    needs_support = False

    def init(self, meas: Measurement, model: ObjectModel) -> Hyps:
        return Hyps(meas.p[None].copy(), np.zeros((1, 2)), np.zeros(1), np.zeros((1, 3)))

    def step(self, H: Hyps, ctx: SupportCtx | None) -> Hyps:
        return H


class SupportSlideProp:
    """Coulomb sliding block on the support (k_in = 0, r = 1; rolling bodies: k_in = reach.K_IN, r = the ball radius)."""
    needs_support = True

    def __init__(self, mu_key: str = "mu", mu_default=(0.18, 0.35), a_dist: float = 0.3, n_dir: int = 8, k_in: float = 0.0, r: float = 1.0):
        self.mu_key, self.mu_default, self.a_dist, self.n_dir, self.k_in, self.r = mu_key, tuple(mu_default), a_dist, n_dir, k_in, r

    def _mu(self, model):
        b = model.params.get(self.mu_key)
        lo, hi = (b.lo, b.hi) if b is not None else self.mu_default
        mid = b.mean if (b is not None and b.mean is not None) else 0.5 * (lo + hi)
        return lo, mid, hi

    def init(self, meas: Measurement, model: ObjectModel) -> Hyps:
        lo, mid, hi = self._mu(model)
        dirs = [np.zeros(3)] + [np.array([math.cos(a), math.sin(a), 0.0]) for a in np.arange(self.n_dir) * 2 * math.pi / self.n_dir]
        mus, ds = [mid], [np.zeros(3)]
        for mu in (lo, hi):
            for d in dirs:
                mus.append(mu)
                ds.append(d)
        N = len(mus)
        return Hyps(np.tile(meas.p, (N, 1)), np.zeros((N, 2)), np.array(mus, float), np.array(ds, float))

    def step(self, H: Hyps, ctx: SupportCtx | None) -> Hyps:
        if ctx is None:
            return H
        nsub = max(int(round(ctx.dt / TICK)), 1)
        dt = None if nsub == 1 else nsub * TICK
        P, V = H.P.copy(), H.V.copy()
        for i in range(len(P)):
            Ps, Vs = D._slide_traj(P[i, :2], V[i], [ctx.R_prev, ctx.R_now], [ctx.a_world + H.dirs[i] * self.a_dist], float(H.mu[i]),
                                   self.k_in, self.r, 1, dt)
            P[i, :2], V[i] = Ps[1], Vs[1]
        return Hyps(P, V, H.mu, H.dirs)


# ====================================================================================== producer
def _chain_root(ws: WorldState, oid: str | None):
    """walk the parent chain: 'robot' if the object is (transitively) carried by the robot, else None (world)."""
    seen = 0
    while oid is not None and oid != "robot" and seen < 16:
        o = ws.objects.get(oid)
        if o is None:
            return None
        oid, seen = o.parent, seen + 1
    return oid


class AnchoredProducer(Producer):
    name = "anchored"

    def __init__(self, model: ObjectModel, sources, propagator=None, hold_s: float = 0.15, k_fals: float | None = 2.0, v_window: int = 5):
        self.model, self.sources = model, list(sources)
        self.prop = propagator or StaticProp()
        self.hold_s, self.k_fals, self.v_window = hold_s, k_fals, v_window
        self.name = f"anchored_{model.oid}"
        self.reset()

    def reset(self, view=None):
        for s in self.sources:
            s.reset(view)
        self.H = None
        self.meas = None
        self.first_t = None
        self.prev = None                    # (t, support world R)
        self.vbuf = deque(maxlen=self.v_window + 1)
        self.n_reset = 0
        self.n_mismatch = 0

    # ---- support motion from the robot's own estimates (odometry twist, FK pose); no simulator state
    def _support_ctx(self, ws: WorldState, t: float):
        m = self.model
        if not self.prop.needs_support or m.parent is None or m.parent == "robot" or m.parent not in ws.objects:
            return None, None
        Ps = ws.world_pose(m.parent)
        Rs = np.asarray(Ps.R, float)
        rob = ws.robot
        if rob.base_vel is not None:
            self.vbuf.append(np.asarray(rob.base_vel, float))
        a_w = np.zeros(3)
        if _chain_root(ws, m.parent) == "robot" and len(self.vbuf) >= 2:
            from world_model.v2.house.teleop import base as BS
            vb = list(self.vbuf)
            v_now = np.mean(vb[-self.v_window:], 0)
            v_prev = np.mean(vb[-self.v_window - 1:-1], 0)
            dt_ = t - self.prev[0] if self.prev is not None and t > self.prev[0] else TICK
            c = self.H.P[0] if self.H is not None else np.zeros(3)
            pw = Rs @ c + np.asarray(Ps.p, float)
            Rb, pb = np.asarray(rob.base.R, float), np.asarray(rob.base.p, float)
            r_b = (Rb.T @ (pw - pb))[:2]
            a_b = BS.cup_accel_body(v_now, (v_now - v_prev) / dt_, r_b)
            a_xy = Rb[:2, :2] @ a_b
            a_w = np.array([a_xy[0], a_xy[1], 0.0])
        ctx = None
        if self.prev is not None and t > self.prev[0]:
            ctx = SupportCtx(self.prev[1], Rs, a_w, t - self.prev[0])
        return ctx, Rs

    def _condition(self, meas: Measurement):
        H = self.H
        z = np.asarray(meas.p, float)
        if self.k_fals is not None and len(H.P) > 1:
            d = np.linalg.norm(H.P[:, :2] - z[:2], axis=1)
            bad = d > self.k_fals * meas.r
            if bad[0] and bad.all():
                self.n_mismatch += 1                         # the whole bank contradicts the data: model mismatch; keep the velocities
            elif bad.any():
                bad[0] = False
                H.V[bad] = H.V[0]                            # inconsistent with the data: restart from the nominal velocity
                self.n_reset += int(bad.sum())
        H.P[:] = z                                           # re-anchor every hypothesis on the measurement

    def update(self, ws: WorldState, view, t: float):
        m = self.model
        if self.first_t is None:
            self.first_t = t
        ctx, Rs = self._support_ctx(ws, t)
        if self.H is not None:
            self.H = self.prop.step(self.H, ctx)
        meas, used, extra = None, None, {}
        for s in self.sources:                                # priority order: the first source that measures this tick is used
            mm = s.measure(view, ws, m, t)
            if mm is not None:
                extra.update(mm.extra)                        # every source's diagnostics (calibration inputs), whichever is used
                if meas is None:
                    meas, used = mm, s.name
        if meas is not None:
            meas.extra = {**extra, "used": used}
            if self.H is None:
                self.H = self.prop.init(meas, m)
            else:
                self._condition(meas)
            self.meas = meas
        if Rs is not None:
            self.prev = (t, Rs)
        if self.H is None:
            return
        me = self.meas
        H = self.H
        c = H.P[0].copy()
        half = np.abs(H.P - c).max(0)
        vis = (t - me.t) <= self.hold_s
        hz = m.z_unc if m.on_support else me.r + half[2]
        unc = Unc(Zonotope(np.zeros(3), np.diag([half[0] + me.r, half[1] + me.r, hz])), float(me.r_rot))
        R = np.asarray(me.R, float) if me.R is not None else EYE3.copy()
        ws.objects[m.oid] = ObjectState(m.oid, m.kind, ZonoShape(m.shape), Pose(R, c), unc=unc, parent=m.parent, name=m.name,
                                        name_source=m.name_source, params=dict(m.params), dynamics=m.dynamics, source=me.source, t_meas=me.t,
                                        visible=bool(vis), age=t - self.first_t,
                                        extra={**me.extra, "v_rel": H.V[0].copy(), "half": half.copy(), "n_hyp": len(H.P), "n_reset": self.n_reset,
                                               "n_mismatch": self.n_mismatch, "orientation": "measured" if me.R is not None else "contact prior"})
        if m.on_support and m.parent is not None:
            ws.relations = [r_ for r_ in ws.relations if not (r_.a == m.oid and r_.rel == "on")] + [Relation(m.oid, "on", m.parent, 1.0 if vis else 0.5, me.source)]
        ws.meta["producers"] = ws.meta.get("producers", []) + [self.name]


# ====================================================================================== configurations
def cup_on_tray(calib=None, cup_kind: str = "flagship", k_fals: float | None = 2.0, a_dist: float = 0.3, mu_prior=(0.18, 0.35),
                fp_npz=None) -> AnchoredProducer:
    """the cup as one configuration of the general tracker: shape = the cup model (object frame, base centre), parent = tray, Coulomb slide,
    wrist blob measurements with the cup_* calibration keys (same keys CupWristProducer reads).  fp_npz: also the FoundationPose source
    (ocs/fp_source.RecordedFPSource), FIRST in priority; the blob covers the ticks FoundationPose misses."""
    model = ObjectModel("cup", Kind.CONTENT, U.cup_shape(cup_kind), parent="tray", dynamics="slip_on_support",
                        params={"mu": Belief(mu_prior[0], mu_prior[1], None, None, Source.PRIOR, "ShieldCfg prior interval")}, name="cup",
                        name_source=Source.PRIOR, on_support=True)
    srcs = [BlobPlaneSource(calib, prefix="cup", plane_z=U.cup_dims(cup_kind)[1] / 2, gate=(U.HX + 0.05, U.HY + 0.05))]
    if fp_npz is not None:
        from .fp_source import RecordedFPSource
        srcs.insert(0, RecordedFPSource(fp_npz, calib, prefix="cup"))
    return AnchoredProducer(model, srcs, SupportSlideProp(mu_default=mu_prior, a_dist=a_dist), k_fals=k_fals)
