"""pick/propose.py -- egocentric object proposals: instance segments -> 2D boxes, depth + FK -> 3D centroid / extent.  SIMULATION ONLY.

PRIVILEGED vs. NON-PRIVILEGED (read this)
  PRIVILEGED (stand-in for a perception front-end, identical in spirit to lang/web.proposals_from_sim_seg):
    * the INSTANCE MASKS: the sim's segmentation buffer (geom id -> body) restricted to the pickable bodies `pp_*` (freejoint props).  Which bodies are "candidate objects"
      and the per-pixel instance partition come from the simulator, not from a detector.  Names of the bodies are NEVER used downstream (labels come from the lang backend).
  NON-PRIVILEGED (everything after the mask):
    * depth = the rendered z-depth with the ZED-head depth noise model (ocs/fp_render.head_spec / wrist_depth_noise: stereo-law sigma, 2 % dropout, 0.2 % outliers, 1 mm
      quantisation, grazing-angle invalid) and the robot self-filter (known kinematics);
    * camera pose = the known mount (fp_render.HEAD_POS / pitch, "proposed torso mount, confirm on hardware") + lift encoder + base pose estimate input (here the truth base pose);
    * 3D points = back-projection of the masked depth pixels; centroid / OBB extents / yaw from those points only (robust 3-97 percentile, PCA of the xy footprint).
  Known estimator bias: only the visible surface is seen; the footprint comes from the top face (camera looks down 36 deg) so xy extents are good, height of a short object seen
  only from above is its top z minus the table plane (support plane z is taken from the lowest visible points of the mask).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import mujoco

from world_model.v2.house.ocs import fp_render as FR

W_DEF, H_DEF = 640, 360
TOP_SLAB = 0.015           # [m] thickness of the top-face slab used for the footprint
ISO_RATIO = 1.5            # footprint a/b below this: orientation is not observable (round / square) -> axes snapped to the robot frame
EDGE_PAD = 0.0           # [m] total: silhouette points at grazing incidence (>80 deg) are invalid in the depth model, so the visible footprint is a few mm short (bias measured on the props; see smoke est_err)
PICK_CAM = "pick_head"


class EgoCam:
    """private-model renderer of the head camera (a copy of the model: the shared model's clip planes / buffers are not touched).  Create + use in ONE thread (EGL context)."""

    def __init__(self, sim, W=W_DEF, H=H_DEF):
        self.m = copy.copy(sim.m)
        self.m.vis.global_.offwidth = max(W, int(self.m.vis.global_.offwidth))
        self.m.vis.global_.offheight = max(H, int(self.m.vis.global_.offheight))
        self.d = mujoco.MjData(self.m)
        self.rend = FR.RGBDRenderer(self.m, W, H, msaa=False)
        self.W, self.H = W, H
        self.K = FR.K_from_fovy(float(self.m.cam_fovy[self.m.camera(PICK_CAM).id]), W, H)
        self.cam_id = int(self.m.camera(PICK_CAM).id)
        self.spec = FR.head_spec(W, H)
        self.robot_geom = np.array(sim.robot_geom, bool)
        self.prop_of_geom = {}
        for g in range(self.m.ngeom):
            bn = self.m.body(int(self.m.geom_bodyid[g])).name or ""
            if bn.startswith("pp_"):
                self.prop_of_geom[g] = bn          # PRIVILEGED instance partition (names never leave this module)

    def snapshot(self, sim):
        """cheap, sim-thread: copy the physics state (+ the head pose from base pose / lift)."""
        mujoco.mj_copyData(self.d, self.m, sim.d)
        lift = sim.lift_actual()
        base = sim.pose()
        o_b, R_b = FR.head_cam_base(lift)
        cy, sy = math.cos(base[2]), math.sin(base[2])
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        org = Rz @ o_b + np.array([base[0], base[1], 0.0])
        return org, Rz @ R_b, base.copy(), float(lift)

    def render(self, snap, seed=0, noise=True):
        org, R, base, lift = snap
        self.d.cam_xpos[self.cam_id] = org
        self.d.cam_xmat[self.cam_id] = R.ravel()
        rgb, dep, geom = self.rend.render(self.d, PICK_CAM, seg=True)
        robot = (geom >= 0) & self.robot_geom[np.maximum(geom, 0)]
        if noise:
            rng = np.random.default_rng(seed)
            dn = FR.wrist_depth_noise(dep, self.K, self.spec, rng, robot)
        else:
            dn = np.where(robot, 0.0, dep).astype(np.float32)
        return dict(rgb=rgb, depth=dn, geom=geom, K=self.K, cam_pos=org, cam_R=R, base=base, lift=lift)

    def close(self):
        self.rend.close()


def to_robot_frame(P_world, base):
    c, s = math.cos(base[2]), math.sin(base[2])
    Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(P_world) - np.array([base[0], base[1], 0.0])) @ Rb


def points_world(frame, mask):
    """masked depth pixels -> world points (n,3) (OpenCV camera: R_cv = R_mj @ FLIP)."""
    v, u = np.nonzero(mask & (frame["depth"] > 0))
    z = frame["depth"][v, u].astype(float)
    K = frame["K"]
    Pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    Rcv = frame["cam_R"] @ FR.FLIP
    return Pc @ Rcv.T + frame["cam_pos"]


def estimate_obb(P):
    """robust footprint OBB of robot-frame points P (n,3): -> dict(extent=(a,b,h), yaw (dir of the a axis, rad), centroid (3,), top_z, base_z)."""
    med = np.median(P, axis=0)
    P = P[np.linalg.norm(P - med, axis=1) < 0.35]                  # gross depth outliers (0.2 % of pixels, 5-20 cm) are removed first
    lo, hi = np.percentile(P, [1, 99], axis=0)
    keep = np.all((P >= lo - 1e-9) & (P <= hi + 1e-9), axis=1)
    Q = P[keep] if keep.sum() >= 20 else P
    # footprint from the TOP FACE only (points within 1.5 cm of the top height): the camera looks down, so the top face is fully visible, whereas the visible side faces bias the
    # whole-cloud centre toward the camera by up to ~1.5 cm (measured on the props).  Falls back to the whole cloud when the top slab is thin.
    zt = np.percentile(Q[:, 2], 97)
    slab = Q[Q[:, 2] > zt - TOP_SLAB]
    F = slab if len(slab) >= 40 else Q
    xy = F[:, :2]
    mu = xy.mean(0)
    C = np.cov((xy - mu).T)
    w, V = np.linalg.eigh(C)
    ax = V[:, ::-1]                                        # columns: major, minor
    t = (xy - mu) @ ax
    ext = t.max(0) - t.min(0) + EDGE_PAD
    ctr_t = 0.5 * (t.max(0) + t.min(0))
    c_xy = mu + ax @ ctr_t
    yaw = math.atan2(ax[1, 0], ax[0, 0])
    if ext[0] < ISO_RATIO * ext[1]:                         # near-isotropic footprint: orientation is not observable -> align with the robot axes
        yaw = 0.0 if abs(math.sin(yaw)) < 0.7071 else math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        ax = np.array([[c, -s], [s, c]])
        t = (xy - mu) @ ax
        ext = t.max(0) - t.min(0) + EDGE_PAD
        c_xy = mu + ax @ (0.5 * (t.max(0) + t.min(0)))
        if ext[0] < ext[1]:
            ext = ext[::-1]
            yaw += math.pi / 2
    top, base = float(Q[:, 2].max()), float(Q[:, 2].min())
    h = top - base
    return dict(extent=(float(ext[0]), float(ext[1]), float(h)), yaw=float((yaw + math.pi / 2) % math.pi - math.pi / 2),
                centroid=np.array([c_xy[0], c_xy[1], 0.5 * (top + base)]), top_z=top, base_z=base, n_pts=int(len(Q)))


def proposals_from_frame(frame, min_px=60, min_pts=25):
    """-> list of dict(box2d, mask_px, extent (a,b,h), yaw, centroid (robot frame), top_z, base_z, n_pts, _gt (PRIVILEGED body name, EVALUATION only)).
    The privileged instance partition is applied by the caller through frame['inst'] = {gt name: bool mask}."""
    out = []
    for gt, mask in frame["inst"].items():
        npx = int(mask.sum())
        if npx < min_px:
            continue
        P = points_world(frame, mask)
        if len(P) < min_pts:
            continue
        o = estimate_obb(to_robot_frame(P, frame["base"]))
        yy, xx = np.nonzero(mask)
        o.update(box2d=[int(xx.min()) - 2, int(yy.min()) - 2, int(xx.max()) + 2, int(yy.max()) + 2], mask_px=npx, _gt=gt)
        out.append(o)
    out.sort(key=lambda o: o["box2d"][0])
    return out


def instance_masks(cam: EgoCam, frame):
    """PRIVILEGED: per-prop boolean masks from the segmentation buffer."""
    g = frame["geom"]
    inst = {}
    for gid, bn in cam.prop_of_geom.items():
        mk = g == gid
        if mk.any():
            inst[bn] = inst.get(bn, np.zeros(g.shape, bool)) | mk
    return inst


def draw_boxes(rgb, props, selected=None, labels=True):
    import cv2
    im = np.ascontiguousarray(rgb[:, :, :3]).copy()
    for p in props:
        sel = selected is not None and p["id"] == selected
        col = (255, 220, 0) if sel else (60, 230, 90) if p.get("mode") == "single" else (255, 120, 200) if p.get("mode") == "bimanual" else (180, 180, 180)
        x0, y0, x1, y1 = [int(v) for v in p["box2d"]]
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 3 if sel else 2, cv2.LINE_AA)
        if labels:
            txt = f"{p['id']} {p.get('label', '')} {p.get('mode', '')[:1].upper()}{'/' + p['arm'][0].upper() if p.get('arm') and p.get('arm') not in ('none',) else ''}"
            cv2.putText(im, txt, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return im


def fuse_estimates(frames):
    """median over per-frame estimates of the same instance (association by instance id `_gt`, i.e. by the segmenter's own instance index, not by geometry).  The box / pixel stats come
    from the first frame that has the instance."""
    by = {}
    for fr in frames:
        for o in fr:
            by.setdefault(o["_gt"], []).append(o)
    out = []
    for gt, lst in by.items():
        o = dict(lst[0])
        o["centroid"] = np.median(np.stack([x["centroid"] for x in lst]), 0)
        o["extent"] = tuple(float(v) for v in np.median(np.array([x["extent"] for x in lst]), 0))
        o["top_z"] = float(np.median([x["top_z"] for x in lst]))
        o["base_z"] = float(np.median([x["base_z"] for x in lst]))
        o["yaw"] = float(np.median([x["yaw"] for x in lst]))
        o["n_frames"] = len(lst)
        out.append(o)
    out.sort(key=lambda o: o["box2d"][0])
    return out
