"""perception_bridge.py -- the tray/ball point-cloud perception stack (tray_rep/perception, perception_zono) run on the composed
MolmoSpaces house model (build.compose).  SIMULATION ONLY: ray casting on the MuJoCo model, CPU, no GL, no camera driver, no
hardware, no network.  Never imports YOR/robot or YOR/extra.  Nothing outside this file is edited.

What it adds (docs/status/molmospaces_integration_plan.md 3.5, 9.3):
  * `HouseScene`   adapter over a `build.Composed`: same `m`, `d`, `radius`, `gid_tray`, `gid_ball` fields as `depth_sim.Scene`.
                   Reassigns tray, ball and the house floor to a free geom group (G_TRAY = 5; they default to group 0, like the
                   house visuals) so a cheap group mask {3, 4, 5} sees walls + collision hulls + robot collision geoms + tray +
                   ball + floor.  `mask('all')` = no mask (every group, for cost / realism comparison).  `hide(...)` parks
                   geoms in group 1 temporarily (e.g. the whole house, to get the "no furniture" condition).
                   `place_base` moves the mocap base AND the welded tray + ball with it (the weld relpose stays satisfied),
                   without stepping the physics.
  * `HouseCamera`  subclass of `depth_sim.DepthCamera` (depth_sim itself is untouched).  Mounts: 'base' (rigid in the robot frame:
                   the head camera, pose = mocap base pose composed with depth_sim's _HEAD_POS_ROBOT/_HEAD_R), 'model' (an MJCF camera
                   of the composed model: the YOR wrist cameras left/right_arm_egocentric).  `capture` has the depth_sim interface
                   and returns a `depth_sim.Frame`, plus a geom-group mask, bodyexclude, a ray cutoff of 1.2 x z_max and an optional
                   arbitrary ROI sphere (so the same class serves the object-zonotope study).
  * `HouseTracker` depth_sim sensor clock + BallKF (pc_env.Pipeline) driven by HouseCamera frames; records per-frame cost.
  * labels         geom -> body -> metadata category (Chair, DiningTable, Doorframe ...), robot classes L/R/B, tray, ball.
  * `MeshObj`      truth object for perception_zono.score.Truth built from the g4 collision hulls of annotated house objects.
  * `capture_cloud` perception_zono.sensing.Cloud (noisy points, GT object labels, dense depth maps) for house objects.

Head camera note (VERIFIED from scene_yor_canonical.xml line 59): depth_sim's head camera sits 1.1 m AHEAD of the base origin and looks
back toward the robot (it is a world camera in the canonical scene).  Composing it with the base pose (as asked) gives a camera that
floats 1.1 m in front of the robot.  `head_aim` is the same position re-aimed at the tray (the rig's winning placement).
"""
from __future__ import annotations

import contextlib
import re
import time
import zlib
from dataclasses import dataclass, replace

import numpy as np
import mujoco

from world_model.v2.tray_rep.perception import depth_sim as D
from world_model.v2.tray_rep.perception import ball_estimator as BE
from world_model.v2.tray_rep.perception import pc_env as PC
from world_model.v2.testbeds.b_tray_ball.tray_env import HX, HY
from world_model.v2.house import build as B
from world_model.v2.house import assets as A
from world_model.v2.house.phase1 import geom_class

G_TRAY = 5            # free group for tray, ball and floor
G_HIDE = 1            # unused group: geoms parked here are invisible to every mask below
MASKS = {"cheap": (3, 4, G_TRAY),            # robot collision (g3), house collision hulls (g4), walls/tray/ball/floor (g5)
         "robot": (3,),                      # robot collision geoms only (self-filter render)
         "vis": (0, 2, 3, 4, G_TRAY),        # + house visual meshes (g0) + robot visual meshes (g2): the realistic, expensive one
         "all": None}
CUTOFF_MULT = 1.2     # ray cutoff = this x z_max
BALL_R = 0.03


def _mask_arr(groups):
    if groups is None:
        return None
    a = np.zeros(6, np.uint8)
    a[list(groups)] = 1
    return a


def _lookat_R(pos, target):
    """camera-to-parent rotation (MuJoCo camera: looks along -z) for a look-at with world-up."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd], 1)


# ------------------------------------------------------------------------------------------------------------- scene
class HouseScene:
    """Adapter over a Composed house model with depth_sim.Scene's duck-typed fields."""

    def __init__(self, C: B.Composed, idx: int | None = None):
        self.C, self.m, self.d = C, C.m, C.d
        m = self.m
        self.radius = BALL_R
        self.idx = C.info["house"] if idx is None else idx
        self.root = m.body("yor_root").id
        self.cls = geom_class(m)                                  # 'L','R','B' robot, 'T' tray, 'P' ball, 'H' house
        self.body_tray, self.body_ball = m.body("tray").id, m.body("ball").id
        self.gid_tray = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_tray]
        self.gid_ball = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_ball][0]
        # ---- group reassignment (tray, ball, floor -> G_TRAY)
        self.floor_geoms = [g for g in range(m.ngeom) if self.cls[g] == "H" and re.match(r"room_\d+_visual", m.geom(g).name or "")]
        self.wall_geoms = [g for g in range(m.ngeom) if self.cls[g] == "H" and m.geom_group[g] == 3]
        for g in (*self.gid_tray, self.gid_ball, *self.floor_geoms, *self.wall_geoms):      # walls leave g3 so g3 = robot only
            m.geom_group[g] = G_TRAY
        self.group0 = m.geom_group.copy()                         # reference groups after the reassignment
        # ---- labels
        self.md = A.metadata(self.idx)
        self.body_cat = {}
        self.body_obj = {}
        for k, v in self.md.items():
            for bn in v["name_map"]["bodies"]:
                try:
                    bid = m.body(bn).id
                except KeyError:
                    continue
                self.body_cat[bid] = v["category"]
                self.body_obj[bid] = k
        self.geom_cat = np.empty(m.ngeom, object)
        for g in range(m.ngeom):
            c = self.cls[g]
            if c in "LRB":
                self.geom_cat[g] = {"L": "arm_L", "R": "arm_R", "B": "robot_base"}[c]
            elif c == "T":
                self.geom_cat[g] = "tray"
            elif c == "P":
                self.geom_cat[g] = "ball"
            else:
                nm = m.geom(g).name or ""
                bid = int(m.geom_bodyid[g])
                if bid in self.body_cat:
                    self.geom_cat[g] = self.body_cat[bid]
                elif nm.startswith("wall"):
                    self.geom_cat[g] = "Wall"
                elif nm.startswith("room"):
                    self.geom_cat[g] = "Floor"
                else:
                    self.geom_cat[g] = "house_other"
        self.is_house = (self.cls == "H")
        self.tray_free = m.jnt_qposadr[m.joint("tray_free").id]
        self.ball_free = m.jnt_qposadr[m.joint("ball_free").id]
        self.tray_dof = m.jnt_dofadr[m.joint("tray_free").id]
        self.ball_dof = m.jnt_dofadr[m.joint("ball_free").id]

    # ---- depth_sim.Scene compatibility
    def set_pose(self, *a, **k):         # the physics model IS the scene
        pass

    def mask(self, kind):
        return _mask_arr(MASKS[kind]) if isinstance(kind, str) else kind

    # ---- hiding
    @contextlib.contextmanager
    def hide(self, what="house"):
        """Temporarily park geoms in G_HIDE.  what: 'house' (every house geom), 'furniture' (house geoms that are not walls or the
        floor), 'doors' (door leaves/handles), 'arms' (both arms), or a boolean/int geom array."""
        m = self.m
        saved = m.geom_group.copy()
        if isinstance(what, str):
            if what == "house":
                sel = self.is_house
            elif what == "furniture":
                sel = self.is_house & ~np.isin(self.geom_cat, ["Wall", "Floor"])
            elif what == "doors":
                sel = self.door_leaf_geoms()
            elif what == "arms":
                sel = np.isin(self.geom_cat, ["arm_L", "arm_R"])
            elif what == "tray":
                sel = np.isin(self.geom_cat, ["tray", "ball"])
            elif what == "robot":
                sel = np.isin(self.geom_cat, ["arm_L", "arm_R", "robot_base"])
            else:
                raise KeyError(what)
        else:
            sel = np.asarray(what)
        try:
            m.geom_group[np.flatnonzero(sel)] = G_HIDE
            yield
        finally:
            m.geom_group[:] = saved

    def door_leaf_geoms(self):
        """bool (ngeom,): door leaf + handle geoms of Doorway objects (same rule as selftest.measure_doors)."""
        m = self.m
        sel = np.zeros(m.ngeom, bool)
        for v in self.md.values():
            if v["category"] not in ("Doorway", "Doorframe"):
                continue
            for bn, asset in v["name_map"]["bodies"].items():
                a = asset.lower()
                if "_door_" in a or "handle" in a or a.endswith("_door"):
                    try:
                        bid = m.body(bn).id
                    except KeyError:
                        continue
                    sel |= (m.geom_bodyid == bid)
        return sel

    # ---- poses
    def base_pose(self):
        """(position (3,), yaw) of the mocap base as currently set."""
        q = self.d.mocap_quat[0]
        return self.d.mocap_pos[0].copy(), 2 * np.arctan2(q[3], q[0])

    def tray_pose(self):
        d = self.d
        R = d.xmat[self.body_tray].reshape(3, 3).copy()
        return d.xpos[self.body_tray].copy(), R

    def ball_in_tray(self):
        p, R = self.tray_pose()
        return R.T @ (self.d.xpos[self.body_ball] - p)

    def place_base(self, xy, yaw, ball_xy=(0.0, 0.0), forward=True):
        """Put the mocap base at (xy, yaw); tray + ball follow rigidly (tray at its hold-time pose in the robot frame, ball at
        ball_xy on the tray, tray frame).  No physics step; the weld relpose is unchanged so the constraint stays satisfied."""
        m, d, info = self.m, self.d, self.C.info
        R = B._rotz(yaw)
        base = np.array([xy[0], xy[1], 0.0])
        d.mocap_pos[0] = base
        d.mocap_quat[0] = B.yaw_quat(yaw)
        if info["has_tray"]:
            c = np.asarray(info["tray_local"], float)
            ty = yaw + (np.pi / 2 if info["tray_across"] else 0.0)
            Rt = B._rotz(ty)
            tc = base + R @ c + np.array([0.0, 0.0, info.get("lift_h", 0.0)])      # lift raises the whole arm block (lift.py)
            d.qpos[self.tray_free:self.tray_free + 3] = tc
            d.qpos[self.tray_free + 3:self.tray_free + 7] = B.yaw_quat(ty)
            bp = tc + Rt @ np.array([ball_xy[0], ball_xy[1], BALL_R + 0.0005])
            d.qpos[self.ball_free:self.ball_free + 3] = bp
            d.qpos[self.ball_free + 3:self.ball_free + 7] = [1, 0, 0, 0]
            d.qvel[self.tray_dof:self.tray_dof + 6] = 0
            d.qvel[self.ball_dof:self.ball_dof + 6] = 0
        if forward:
            mujoco.mj_forward(m, d)

    def house_contacts(self):
        """number of contacts between robot/tray/ball geoms and house geoms at the current state (mj_forward done)."""
        n = 0
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            if self.is_house[c.geom1] != self.is_house[c.geom2]:
                n += 1
        return n

    # ---- eval-only helpers
    def los(self, origin, target, mask="cheap", tol=0.012):
        """Line of sight from `origin` to `target` (the ball centre) with the BALL excluded: -> (blocked, blocker_category,
        blocker_distance, target_distance).  A first hit more than `tol` short of the target blocks (tray, arms and house all count)."""
        o = np.asarray(origin, float)
        v = np.asarray(target, float) - o
        L = float(np.linalg.norm(v))
        gid = np.zeros(1, np.int32)
        dd = mujoco.mj_ray(self.m, self.d, o, v / L, self.mask(mask), 1, self.body_ball, gid)
        if gid[0] < 0 or dd < 0 or dd > L - tol:
            return False, "", float("nan"), L
        return True, str(self.geom_cat[gid[0]]), float(dd), L


# ------------------------------------------------------------------------------------------------------------- camera
def _spec_head(scene_tray_local, aim=False, **kw):
    pos = D._HEAD_POS_ROBOT
    R = _lookat_R(pos, np.asarray(scene_tray_local, float) + [0, 0, 0.02]) if aim else D._HEAD_R
    base = D.CAMERAS["head_aim" if aim else "head"]
    return replace(base, name="head_aim" if aim else "head", mount="base", pos=tuple(pos), R=tuple(np.asarray(R).ravel()), **kw)


HEAD_NEW_POS_LIFT0 = np.array([0.05, -0.209550, 1.15])   # ASSUMED torso/mast head, robot frame at lift 0: over the lift column (y = column
                                                         # centreline), 0.05 m ahead of it, 1.15 m high = ~0.3 m above the tray at lift 0
H_MAX = 0.416
HEAD_PITCH_CURRENT = float(np.degrees(np.arcsin(0.5924)))   # pitch of the scene head_cam (xyaxes 0 1 0 -0.5924 0 0.8057): 36.3 deg down


def _spec_head_new(mount="lift", pitch_deg=HEAD_PITCH_CURRENT, **kw):
    """ASSUMED head on a torso mast, looking straight ahead (+x) and pitched `pitch_deg` down.  mount 'lift': rides the lift with the
    arms (z = 1.15 + lift); 'mast': fixed to the base on a mast whose top is 1.15 + 0.416 m (clears the fully raised arm block)."""
    pos = HEAD_NEW_POS_LIFT0.copy()
    if mount == "mast":
        pos[2] += H_MAX
    th = np.deg2rad(pitch_deg)
    R = _lookat_R(pos, pos + np.array([np.cos(th), 0.0, -np.sin(th)]))
    return replace(D.CAMERAS["head"], name="head", mount=mount, pos=tuple(pos), R=tuple(R.ravel()),
                   note=f"ASSUMED mast head ({mount}, pitch {pitch_deg:.1f} deg down)", **kw)


def _spec_wrist(side, **kw):
    return replace(D.CAMERAS[f"wrist_{side}"], mount="model", **kw)


RIG_CAMS = {"headN": ("headN",), "headN+wrist": ("headN", "wrist_l", "wrist_r"), "head": ("head",), "head_aim": ("head_aim",), "wrist_l": ("wrist_l",), "wrist_r": ("wrist_r",),
            "wrist_both": ("wrist_l", "wrist_r"), "head+wrist": ("head", "wrist_l", "wrist_r"),
            "head_aim+wrist": ("head_aim", "wrist_l", "wrist_r")}
_MODEL_CAM = {"wrist_l": "left_arm_egocentric", "wrist_r": "right_arm_egocentric"}


class HouseCamera(D.DepthCamera):
    """depth_sim.DepthCamera on the composed house model.  See the module doc for the mounts."""

    def __init__(self, spec: D.CamSpec, scene: HouseScene):
        super().__init__(spec)
        self.scene = scene
        self.cam_id = scene.m.camera(_MODEL_CAM[spec.name]).id if spec.mount == "model" else -1
        self.bodyexclude = int(scene.m.cam_bodyid[self.cam_id]) if self.cam_id >= 0 else -1

    def pose(self, tray_pos=None, tray_R=None):
        """(camera centre, camera-to-world rotation) from the CURRENT data state (tray args accepted for interface parity)."""
        d = self.scene.d
        if self.spec.mount == "world":
            return self._pos, self._R_cam
        if self.spec.mount in ("base", "lift", "mast"):
            Rb = d.xmat[self.scene.root].reshape(3, 3)
            lift = self.scene.C.info.get("lift_h", 0.0) if self.spec.mount == "lift" else 0.0
            return d.xpos[self.scene.root] + Rb @ (self._pos + np.array([0.0, 0.0, lift])), Rb @ self._R_cam
        return d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def roi_box(self, origin, R, tray_pos, tray_R, scene, pad_px=3):
        """Tight pixel box around the volume the ball tracker can use: the tray footprint grown by radius + 2 cm, from the plane up to
        2 r + z_top (the same limits as ball_estimator.select_points).  Falls back to the full image if a corner is behind the camera."""
        s = self.spec
        cfg = BE.EstCfg()
        hx, hy = HX + scene.radius + cfg.foot_margin, HY + scene.radius + cfg.foot_margin
        zt = 2 * scene.radius + cfg.z_top
        c = np.array([[i * hx, j * hy, k * zt] for i in (-1, 1) for j in (-1, 1) for k in (0, 1)], float)
        pc = (np.asarray(tray_pos) + c @ np.asarray(tray_R).T - origin) @ R               # camera frame (looks along -z)
        if (pc[:, 2] >= -0.02).any():
            return 0, s.W, 0, s.H
        zc = -pc[:, 2]
        u = s.W / 2.0 + s.f_px * pc[:, 0] / zc - 0.5
        v = s.H / 2.0 - s.f_px * pc[:, 1] / zc - 0.5
        u0, u1 = int(np.floor(u.min())) - pad_px, int(np.ceil(u.max())) + 1 + pad_px
        v0, v1 = int(np.floor(v.min())) - pad_px, int(np.ceil(v.max())) + 1 + pad_px
        u0, u1, v0, v1 = max(u0, 0), min(u1, s.W), max(v0, 0), min(v1, s.H)
        if u1 <= u0 or v1 <= v0:
            return 0, 0, 0, 0
        return u0, u1, v0, v1

    SELF_MARGIN = 0.015          # self-filter: drop a return within this (+ 3 sigma_z) of the robot's own predicted depth

    def capture(self, scene, tray_pos, tray_R, tick, seed, use_roi=True, roi_pad=0.06, mask="cheap", roi=None, want_view=False,
                tight=True, max_rays=None, self_filter=True):
        """Cast rays (inside the ROI when `use_roi`) at the house model and return a depth_sim.Frame (noisy world points,
        house geom ids, ...).  roi=(centre, rho) overrides the default ROI sphere around the tray.  want_view -> (Frame, CamView)
        where CamView carries the dense z-depth map (inf = no hit, nan = invalid) as perception_zono.sensing.capture does."""
        s = self.spec
        origin, R = self.pose()
        if roi is None:
            rho = float(np.hypot(HX, HY)) + scene.radius * 2.0 + roi_pad
            centre = np.asarray(tray_pos, float) + np.asarray(tray_R, float) @ np.array([0.0, 0.0, scene.radius])
        else:
            centre, rho = roi
        if not use_roi:
            u0, u1, v0, v1 = 0, s.W, 0, s.H
        elif roi is None and tight:
            u0, u1, v0, v1 = self.roi_box(origin, R, tray_pos, tray_R, scene)
        else:
            u0, u1, v0, v1 = self.roi(origin, R, centre, rho)
        st = 1
        if max_rays is not None and (u1 - u0) * (v1 - v0) > max_rays:
            st = int(np.ceil(np.sqrt((u1 - u0) * (v1 - v0) / max_rays)))
        tp = np.zeros(3) if tray_pos is None else np.array(tray_pos, float)
        tR = np.eye(3) if tray_R is None else np.array(tray_R, float)
        empty = D.Frame(np.zeros((0, 3)), np.zeros(0, np.int32), origin, R, tp, tR, int(tick), 0, s.name)
        if u1 <= u0 or v1 <= v0:
            return (empty, None) if want_view else empty
        dsub = self.dirs[v0:v1:st, u0:u1:st]
        dirs = dsub.reshape(-1, 3)
        cth = self.cos_th[v0:v1:st, u0:u1:st].reshape(-1)
        n = len(dirs)
        dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        nrm = np.zeros(3 * n)
        mujoco.mj_multiRay(scene.m, scene.d, origin, dw.reshape(-1), scene.mask(mask), True, self.bodyexclude, geom, dist, nrm, n,
                           s.z_max * CUTOFF_MULT)
        hit = (geom >= 0) & (dist > 0)
        idx = np.flatnonzero(hit)
        zmap = np.full(n, np.inf)
        if len(idx) == 0:
            empty.n_rays = n
            return (empty, _view(s, origin, R, (u0, u1, v0, v1), zmap)) if want_view else empty
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), zlib.crc32(s.name.encode())]))
        k = len(idx)
        z = dist[idx] * cth[idx]
        nh = nrm.reshape(-1, 3)[idx]
        cosinc = np.abs(np.einsum("ij,ij->i", nh, dw[idx]))
        e = rng.normal(size=k)
        u_drop, u_out, e_out = rng.random(k), rng.random(k), rng.uniform(0.05, 0.20, k) * np.where(rng.random(k) < 0.5, -1, 1)
        zn = z + s.sigma_z(z) * e
        zn = np.where(u_out < s.p_out, z + e_out, zn)
        if s.quant_m > 0:
            zn = np.round(zn / s.quant_m) * s.quant_m
        ok = (cosinc >= s.cos_min) & (u_drop >= s.p_drop) & (zn >= s.z_min) & (zn <= s.z_max)
        self.n_selffilt = 0
        if self_filter and ok.any() and scene.mask("robot") is not None:
            # robot self-filter: render the robot alone (known kinematics) along the surviving rays; a return that lies within
            # margin + 3 sigma of that predicted robot depth is the robot, not the scene.
            si = idx[ok]
            g2 = np.full(len(si), -1, np.int32)
            d2 = np.full(len(si), -1.0)
            mujoco.mj_multiRay(scene.m, scene.d, origin, np.ascontiguousarray(dw[si]).reshape(-1), scene.mask("robot"), True,
                               self.bodyexclude, g2, d2, None, len(si), s.z_max * CUTOFF_MULT)
            z_r = d2 * cth[si]
            rob = (g2 >= 0) & (d2 > 0) & (np.abs(zn[ok] - z_r) < self.SELF_MARGIN + 3 * s.sigma_z(zn[ok]))
            self.n_selffilt = int(rob.sum())
            okk = np.flatnonzero(ok)
            ok[okk[rob]] = False
        sel = idx[ok]
        zn = zn[ok]
        pts = origin + (zn / cth[sel])[:, None] * dw[sel]
        fr = D.Frame(pts, geom[sel].copy(), origin, R, tp, tR, int(tick), n, s.name, zn.copy(), dw[sel].copy())
        if want_view:
            zmap[idx] = np.nan
            zmap[sel] = zn
            return fr, _view(s, origin, R, (u0, u1, v0, v1), zmap)
        return fr


def _view(s, origin, R, roi, zmap):
    from world_model.v2.perception_zono.sensing import CamView
    u0, u1, v0, v1 = roi
    return CamView(s.name, origin, R, s.f_px, s.W, s.H, roi, zmap.reshape(v1 - v0, u1 - u0), s.z_min)


def make_rig(scene: HouseScene, rig="head_aim+wrist", head_mount="lift", head_pitch=HEAD_PITCH_CURRENT, **overrides):
    """-> (list[HouseCamera], list[CamSpec]) for a rig name in RIG_CAMS.  `overrides` (rate_hz, latency_ticks, W, H ...) apply to all."""
    tl = np.asarray(scene.C.info["tray_local"], float)
    out = []
    for nm in RIG_CAMS[rig]:
        if nm == "headN":
            sp = _spec_head_new(head_mount, head_pitch)
        elif nm == "head":
            sp = _spec_head(tl)
        elif nm == "head_aim":
            sp = _spec_head(tl, aim=True)
        else:
            sp = _spec_wrist(nm[-1])
        if overrides:
            sp = replace(sp, **overrides)
        out.append(HouseCamera(sp, scene))
    return out, [c.spec for c in out]


# ------------------------------------------------------------------------------------------------------------- tracker
class HouseTracker:
    """depth_sim sensor clock + delivery latency + BallKF (pc_env.Pipeline) fed by HouseCamera frames of the live house state.
    step(tick) must be called every 50 Hz tick; the scene state (base, tray, ball) is whatever the caller has set."""

    def __init__(self, scene: HouseScene, rig="head_aim+wrist", rate_hz=30.0, latency_ticks=3, mask="cheap", seed=0, cfg=None,
                 use_roi=True, max_rays=None, eval_los=False, focus=False, head_mount="lift", head_pitch=HEAD_PITCH_CURRENT, **cam_overrides):
        self.scene, self.rig, self.mask, self.seed, self.use_roi = scene, rig, mask, seed, use_roi
        self.max_rays, self.eval_los, self.focus = max_rays, eval_los, focus
        self.cams, self.specs = make_rig(scene, rig, head_mount, head_pitch, **cam_overrides)
        self.cfg = cfg or BE.EstCfg()
        self.pipe = PC.Pipeline(rate_hz, latency_ticks)
        self.cost = []          # per capture: dict(t_cam=[ms per camera], t_fit=ms, n_rays=[...], n_pts=[...], n_ball_hits=[...])
        self.last = None

    def _capture(self, tick):
        sc = self.scene
        tp, tR = sc.tray_pose()
        fr, tc = [], []
        roi = None
        if self.focus and self.pipe.kf.n_upd > 0 and self.pipe.kf.since(tick) <= 12:
            # two-stage ROI: once the track is alive, cast only a sphere around the predicted ball (KF mean, radius = ball + 3 sigma + 2 cm)
            kf = self.pipe.kf
            sig = float(np.sqrt(max(kf.P[0, 0], kf.P[1, 1])))
            roi = (tp + tR @ np.array([kf.x[0], kf.x[1], sc.radius + 0.0005]), sc.radius + 0.02 + min(3 * sig, 0.08))
        for c in self.cams:
            t0 = time.perf_counter()
            f = c.capture(sc, tp, tR, tick, self.seed, self.use_roi, mask=self.mask, max_rays=self.max_rays, roi=roi)
            tc.append((time.perf_counter() - t0) * 1e3)
            fr.append(f)
        t1 = time.perf_counter()
        pay = BE.measure(fr, self.cfg, self.specs)
        tf = (time.perf_counter() - t1) * 1e3
        rec = dict(tick=tick, focused=roi is not None, t_cam=tc, t_fit=tf, n_rays=[f.n_rays for f in fr], n_pts=[len(f.geom) for f in fr],
                   n_ball_hits=[int((f.geom == sc.gid_ball).sum()) for f in fr], valid=float(pay[BE.I_VALID]),
                   raw_xy=pay[BE.I_X:BE.I_X + 2].copy(), n_inl=float(pay[BE.I_N]), rms=float(pay[BE.I_RMS]),
                   truth_xy=sc.ball_in_tray()[:2].copy())
        if self.eval_los:                      # EVAL ONLY (geometry ids / truth): who blocks the line camera -> ball centre
            bw = sc.d.xpos[sc.body_ball]
            mk = self.mask if self.mask != "all" else "cheap"
            rec["los"] = [sc.los(c.pose()[0], bw, mk)[:2] for c in self.cams]
            with sc.hide("robot"):                 # would the HOUSE alone block the line of sight?
                rec["los_h"] = [sc.los(c.pose()[0], bw, mk)[:2] for c in self.cams]
        self.cost.append(rec)
        self.last = rec
        return pay

    def init(self, tick=0):
        self.pipe.init(tick, self._capture(tick))

    def step(self, tick):
        return self.pipe.step(tick, lambda: self._capture(tick))

    def estimate(self):
        """(KF position xy in the tray frame, covariance diag of the position block, ticks since last good measurement)."""
        kf = self.pipe.kf
        return kf.x[:2].copy(), np.diag(kf.P)[:2].copy(), kf.since(self.pipe.tick)


# ------------------------------------------------------------------------------------------------------------- objects
def object_bodies(scene: HouseScene, key):
    """body ids of the annotated object `key` (metadata key)."""
    out = []
    for bn in scene.md[key]["name_map"]["bodies"]:
        try:
            out.append(scene.m.body(bn).id)
        except KeyError:
            pass
    return out


def object_geoms(scene: HouseScene, key, groups=(4,)):
    m = scene.m
    bs = set(object_bodies(scene, key))
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bs and int(scene.group0[g]) in groups]


def _geom_hull_pts(m, d, g):
    """world-frame vertices of a collision geom (mesh / box / sphere / capsule / cylinder as a point set)."""
    typ = int(m.geom_type[g])
    R = d.geom_xmat[g].reshape(3, 3)
    p = d.geom_xpos[g]
    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        a0, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = m.mesh_vert[a0:a0 + n]
        # mesh_vert is in the mesh's inertial frame; geom_pos/quat already include mesh offset in compiled models
        return V @ R.T + p
    s = m.geom_size[g]
    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
        return c @ R.T + p
    if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
        u = np.random.default_rng(0).normal(size=(200, 3))
        return p + s[0] * u / np.linalg.norm(u, axis=1, keepdims=True)
    if typ in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ring = np.stack([s[0] * np.cos(a), s[0] * np.sin(a), np.zeros(16)], 1)
        c = np.concatenate([ring + [0, 0, s[1]], ring - [0, 0, s[1]]])
        return c @ R.T + p
    raise ValueError(f"geom type {typ}")


class MeshObj:
    """Truth object (perception_zono.geometry.Obj duck type: surface / volume / sdf / contains / aabb) for the union of the convex
    collision hulls of one annotated house object.  The rays hit exactly these hulls, so they are the truth for the point cloud.
    sdf: negative inside any hull (max plane distance, exact inside), positive outside = distance to the nearest of 30k sampled
    boundary points (sampling error about 1 mm); surface samples buried inside another hull are discarded."""

    def __init__(self, scene: HouseScene, key, oid=0, n_boundary=30000, seed=0):
        from scipy.spatial import ConvexHull, cKDTree
        m, d = scene.m, scene.d
        self.name, self.kind, self.oid = key, scene.md[key]["category"], oid
        self.hulls = []
        for g in object_geoms(scene, key):
            P = _geom_hull_pts(m, d, g)
            try:
                h = ConvexHull(P)
            except Exception:
                continue
            self.hulls.append((h.equations[:, :3].copy(), h.equations[:, 3].copy(), P[h.vertices], float(h.area)))
        if not self.hulls:
            raise ValueError(f"no hulls for {key}")
        rng = np.random.default_rng(seed)
        self._boundary = self.surface(n_boundary, rng)
        self._tree = cKDTree(self._boundary)

    def _ins(self, X):
        out = np.zeros(len(X), bool)
        for A_, b_, _, _ in self.hulls:
            out |= ((X @ A_.T + b_) <= 0).all(1)
        return out

    def contains(self, X):
        return self._ins(np.asarray(X))

    def aabb(self):
        V = np.concatenate([h[2] for h in self.hulls])
        return V.min(0), V.max(0)

    def _hull_surface(self, i, n, rng):
        A_, b_, V, area = self.hulls[i]
        from scipy.spatial import ConvexHull
        h = ConvexHull(V)
        tri = V[h.simplices]
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        k = rng.choice(len(tri), n, p=a / a.sum())
        r1, r2 = rng.random((2, n))
        s = np.sqrt(r1)[:, None]
        return (1 - s) * tri[k, 0] + s * (1 - r2)[:, None] * tri[k, 1] + s * r2[:, None] * tri[k, 2]

    def surface(self, n, rng):
        ar = np.array([h[3] for h in self.hulls])
        cnt = rng.multinomial(int(n * 1.4) + 8, ar / ar.sum())
        out = []
        for i, c in enumerate(cnt):
            if c == 0:
                continue
            P = self._hull_surface(i, c, rng)
            keep = np.ones(len(P), bool)
            for j, (A_, b_, _, _) in enumerate(self.hulls):
                if j != i:
                    keep &= ~((P @ A_.T + b_) <= -5e-5).all(1)
            out.append(P[keep])
        P = np.concatenate(out)
        return P[rng.permutation(len(P))[:n]]

    def sdf(self, X):
        X = np.asarray(X)
        d_out, _ = self._tree.query(X)
        ins = self._ins(X)
        d_in = np.full(len(X), -np.inf)
        for A_, b_, _, _ in self.hulls:
            d_in = np.maximum(d_in, np.where(((X @ A_.T + b_) <= 0).all(1), (X @ A_.T + b_).max(1), -np.inf))
        return np.where(ins, d_in, d_out)

    def volume(self, rng, n=200000):
        lo, hi = self.aabb()
        X = rng.uniform(lo, hi, (n, 3))
        return float(self._ins(X).mean() * np.prod(hi - lo))


# ------------------------------------------------------------------------------------------------------------- object clouds
@dataclass
class ObjView:
    """a virtual copy of the rig's camera set placed relative to a 'virtual tray' frame (origin on the support surface)."""
    specs: list
    tray_pos: np.ndarray
    tray_R: np.ndarray


def rig_in_frame(rig, tray_pos, yaw, W=480, H=270, cond="nominal"):
    """The tabletop rig (perception_zono.sensing views head_aim, head_aim+wrist, wrist_both, head_hi) expressed relative to a
    virtual tray frame at `tray_pos` (origin on the supporting surface, tray_R = Rz(yaw)).  Returns list[CamSpec] in WORLD coordinates
    (mount 'world').  The rig's cameras live in a testbed world where the tray home is (0,0,1); wrist cameras sit in the tray frame."""
    from world_model.v2.perception_zono import sensing as S
    Rt = B._rotz(yaw)
    out = []
    for sp in S.rig_cameras(rig, cond, W, H):
        p, Rc = np.asarray(sp.pos, float), np.asarray(sp.R, float).reshape(3, 3)
        if sp.mount == "tray":
            pw, Rw = tray_pos + Rt @ p, Rt @ Rc
        else:                                    # world camera in the testbed: relative to the tray home (0,0,1), tray_R = I
            pw, Rw = tray_pos + Rt @ (p - S.TRAY_POS), Rt @ Rc
        out.append(replace(sp, mount="world", pos=tuple(pw), R=tuple(Rw.ravel())))
    return out


def capture_cloud(scene: HouseScene, specs, centre, rho, labels, seed=0, tick=0, crop=None, mask="cheap"):
    """perception_zono.sensing.Cloud for house objects.  `labels`: dict geom id -> object label (>= 1); every other house geom is label 0
    (support surface, walls, other furniture).  Points farther than `crop` (default rho) from `centre` are dropped from the cloud (a
    crop box around the work area) but stay in the dense depth maps used for free-space carving."""
    from world_model.v2.perception_zono.sensing import Cloud
    lab_of = np.zeros(scene.m.ngeom, np.int32)
    for g, k in labels.items():
        lab_of[g] = k
    crop = rho if crop is None else crop
    P, L, C, views = [], [], [], []
    nr = 0
    for ci, sp in enumerate(specs):
        cam = sp if isinstance(sp, HouseCamera) else HouseCamera(sp, scene)      # CamSpec (mount 'world') or a robot-mounted HouseCamera
        sp = cam.spec
        fr, vw = cam.capture(scene, None, None, tick, seed + 7919 * ci, True, roi=(np.asarray(centre, float), rho), mask=mask, want_view=True)
        if vw is None:
            continue
        nr += fr.n_rays
        views.append(vw)
        if len(fr.pts_w):
            keep = np.linalg.norm(fr.pts_w - centre, axis=1) <= crop
            P.append(fr.pts_w[keep]); L.append(lab_of[fr.geom[keep]]); C.append(np.full(int(keep.sum()), len(views) - 1, np.int32))
    if P:
        return Cloud(np.concatenate(P), np.concatenate(L), np.concatenate(C), views, nr)
    return Cloud(np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0, np.int32), views, nr)


def robot_occluded_frac(scene: HouseScene, cam: HouseCamera, pts, margin=0.01):
    """EVAL ONLY: fraction of the target points whose line of sight from the camera is blocked by the robot's own geometry
    (arms, hands, base; ray cast against the robot-collision group only)."""
    pts = np.asarray(pts, float)
    o, _ = cam.pose()
    v = pts - o
    L = np.linalg.norm(v, axis=1)
    dirs = np.ascontiguousarray(v / L[:, None])
    n = len(pts)
    g = np.full(n, -1, np.int32)
    dd = np.full(n, -1.0)
    mujoco.mj_multiRay(scene.m, scene.d, o, dirs.reshape(-1), scene.mask("robot"), True, -1, g, dd, None, n, float(L.max()) + 0.1)
    return float(((g >= 0) & (dd > 0) & (dd < L - margin)).mean())
