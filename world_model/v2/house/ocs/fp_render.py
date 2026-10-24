"""ocs/fp_render.py -- kinematic RE-RENDER of an OCS carry recording tick as an RGB-D wrist-camera frame (for FoundationPose).  SIMULATION ONLY.

No physics: the TeleopSim "h5" scene the recording was made in is rebuilt, then for tick k the state is set from the recording's TRUE arrays
(arm joints tL/tR, lift, base pose, tray pose, cup pose; the beads ride rigidly with the cup at their initial cup-relative offsets) and mj_forward is called.
A wrist camera (MJCF `left_arm_egocentric` / `right_arm_egocentric`) is rendered with mujoco.Renderer (EGL; launch with MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=6).

    fr = FPRender(seed=4, W=320, H=180)
    fr.set_tick(rec, k)
    out = fr.render("right", k, seed=4)        # dict rgb u8 (H,W,3), depth f32 (H,W) [m; 0 = invalid], K (3,3), cam_pos, cam_R (MuJoCo convention), cup_seg [EVAL], ...

Depth = renderer z-depth + the wrist depth model of tray_rep/perception/depth_sim CamSpec wrist_l / wrist_r (stereo-law sigma_z with 2 mm floor, grazing
incidence invalid (> 80 deg, normals from the depth image), 2 % dropout, 0.2 % gross outliers, 1 mm quantisation, valid range [0.10, 2.0] m) and the same SELF-FILTER as
grasp/wrist.py (pixels whose first hit is a robot geom are dropped: the known robot kinematics).  RNG seeded from (recording seed, tick, camera).  The wrist camera has NO
documented depth on the real robot: everything built on this is a what-if on an ASSUMED depth-capable wrist camera.

Pure-geometry helpers (no GL, no GPU): cv_cam_in_tray, backproject, tray_gate_mask, canonical_yaw, K_from_fovy.
"""
from __future__ import annotations

import os
import zlib

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])            # MuJoCo camera (x right, y up, looks -z)  <->  OpenCV camera (x right, y down, looks +z): R_cv = R_mj @ FLIP
CAM_NAME = {"left": "left_arm_egocentric", "right": "right_arm_egocentric"}
SPEC_NAME = {"left": "wrist_l", "right": "wrist_r"}
SIDES = ("left", "right")


# ====================================================================================== pure geometry
def K_from_fovy(fovy_deg: float, W: int, H: int) -> np.ndarray:
    """pinhole intrinsics of a MuJoCo camera (pixel-index convention: pixel centres at integer coordinates, principal point at (W-1)/2, (H-1)/2)."""
    f = (H / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array([[f, 0.0, (W - 1) / 2.0], [0.0, f, (H - 1) / 2.0], [0.0, 0.0, 1.0]])


def T_of(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = p
    return T


def cv_cam_in_tray(Rt, pt, cam_org, cam_R_mj) -> np.ndarray:
    """4x4 pose of the OpenCV camera in the TRAY frame.  (pt, Rt) = tray pose and (cam_org, cam_R_mj) = MuJoCo camera pose, both in the same (base / world) frame."""
    Rt = np.asarray(Rt, float).reshape(3, 3)
    return T_of(Rt.T @ np.asarray(cam_R_mj, float).reshape(3, 3) @ FLIP, Rt.T @ (np.asarray(cam_org, float) - np.asarray(pt, float)))


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H,W,3) camera-frame (OpenCV) points of a z-depth image."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)


def tray_gate_mask(depth, K, T_tray_cam, hx, hy, shrink=0.02, h_lo=0.005, h_hi=0.11, min_px=1):
    """NON-privileged cup mask: depth points (camera -> tray frame by T_tray_cam, the ENCODER-FK pose) whose height above the tray top lies in [h_lo, h_hi] and which are
    inside the tray rectangle shrunk by `shrink` (the hands hold the tray edges).  Largest 8-connected component after a small closing; holes filled.
    -> (mask bool (H,W), pts_tray (n,3) of the mask pixels).  Uses no rendered segmentation / geom ids."""
    import cv2
    P = backproject(np.nan_to_num(depth, nan=0.0), K)
    Pt = P @ T_tray_cam[:3, :3].T + T_tray_cam[:3, 3]
    ok = (depth > 0.0) & (Pt[..., 2] >= h_lo) & (Pt[..., 2] <= h_hi) & (np.abs(Pt[..., 0]) <= hx - shrink) & (np.abs(Pt[..., 1]) <= hy - shrink)
    m = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    empty = np.zeros(depth.shape, bool)
    if n <= 1:
        return empty, np.zeros((0, 3))
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return empty, np.zeros((0, 3))
    out = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    mk = filled.astype(bool) & (depth > 0.0)          # filled holes keep only pixels that have a depth (FoundationPose ignores the rest anyway)
    return mk, Pt[mk]


def canonical_yaw(R, ref_x=(1.0, 0.0, 0.0)) -> np.ndarray:
    """the cup is rotationally symmetric about its axis: keep the measured z-axis, set x = ref_x (the tray x-axis) projected orthogonal to z."""
    z = np.asarray(R, float).reshape(3, 3)[:, 2]
    z = z / np.linalg.norm(z)
    x0 = np.asarray(ref_x, float)
    x = x0 - (x0 @ z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z[1] * z
        nx = np.linalg.norm(x)
    x = x / nx
    return np.stack([x, np.cross(z, x), z], 1)


def wrist_depth_noise(depth, K, spec, rng, robot_mask=None):
    """depth_sim wrist noise model on a clean z-depth image (H,W) [m] -> (H,W) float32, 0 = invalid.  `spec` = depth_sim.CamSpec."""
    z = np.asarray(depth, np.float64)
    valid = np.isfinite(z) & (z > 0)
    z0 = np.where(valid, z, 1.0)
    P = backproject(z0, K)
    gy, gx = np.gradient(P, axis=0), np.gradient(P, axis=1)                       # d/dv, d/du of the camera-frame point map
    nrm = np.cross(gx, gy)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    ray = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
    cosinc = np.abs(np.einsum("hwc,hwc->hw", nrm, ray))
    e = rng.normal(size=z.shape)
    u_drop, u_out = rng.random(z.shape), rng.random(z.shape)
    e_out = rng.uniform(0.05, 0.20, z.shape) * np.where(rng.random(z.shape) < 0.5, -1, 1)
    zn = z0 + spec.sigma_z(z0) * e
    zn = np.where(u_out < spec.p_out, z0 + e_out, zn)
    if spec.quant_m > 0:
        zn = np.round(zn / spec.quant_m) * spec.quant_m
    ok = valid & (cosinc >= spec.cos_min) & (u_drop >= spec.p_drop) & (zn >= spec.z_min) & (zn <= spec.z_max)
    if robot_mask is not None:
        ok &= ~robot_mask                                                           # self-filter (known robot kinematics), as grasp/wrist.py
    return np.where(ok, zn, 0.0).astype(np.float32)


def red_blob(rgb, min_px=20):
    """the colour threshold of grasp/wrist.WristCam.seg_cup_rgb -> (n_px, u, v) or None (pixel-index coordinates of THIS image)."""
    a = np.asarray(rgb, float) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r > 0.25) & (r > 1.8 * g) & (r > 1.8 * b)
    if m.sum() < min_px:
        return None
    v, u = np.nonzero(m)
    return int(m.sum()), float(u.mean()), float(v.mean())


# ====================================================================================== renderer
class RGBDRenderer:
    """mujoco.Renderer wrapper -> RGB u8, z-depth f32 (m, inf/far -> 0), optional geom-id segmentation."""

    def __init__(self, m, W, H, msaa=True):
        import mujoco
        self.mj = mujoco
        assert m.vis.global_.offwidth >= W and m.vis.global_.offheight >= H, "model offscreen buffer too small"
        # the house model's default clip planes (znear = 0.01 * extent ~ 0.1 m+) would delete the very close cup (it sits 5-12 cm from the right wrist camera): set an absolute 1 cm / 4 m range
        m.vis.map.znear, m.vis.map.zfar = 0.01 / m.stat.extent, 4.0 / m.stat.extent
        if not msaa:                                          # no multisampling: segmentation ids / depth edges stay exact (MSAA blends ids at 640x360 -> invalid index)
            m.vis.quality.offsamples = 0
        self.r = mujoco.Renderer(m, height=H, width=W)
        self.W, self.H = W, H
        self.far = float(m.stat.extent * m.vis.map.zfar)

    def render(self, d, camera, seg=False, groups=None):
        """groups: geom groups to draw (default = the model's default visual groups 0,1,2 = what a real camera sees, incl. the gripper's visual meshes)."""
        if groups is None:
            self.r.update_scene(d, camera=camera)
        else:
            opt = self.mj.MjvOption()
            opt.geomgroup[:] = 0
            opt.geomgroup[list(groups)] = 1
            self.r.update_scene(d, camera=camera, scene_option=opt)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        dep = self.r.render().copy()
        self.r.disable_depth_rendering()
        dep = np.where(np.isfinite(dep) & (dep < 0.98 * self.far), dep, 0.0).astype(np.float32)
        geom = None
        if seg:
            self.r.enable_segmentation_rendering()
            s = self.r.render().copy()
            self.r.disable_segmentation_rendering()
            geom = np.where(s[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)
        return rgb, dep, geom

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass


# ---- ZED head camera (proposed torso mount; people_track/camera.py head spec, zed_demo/rig._zed_cams convention) ----
HEAD_POS = np.array([0.05, -0.20955, 1.15])      # robot (base) frame at lift 0; rides the lift: z = 1.15 + lift
HEAD_PITCH_DEG = 36.3
HEAD_FOVY = 70.0
HEAD_BASELINE = 0.12
HEAD_SIGMA_DISP_PX = 0.176                        # stereo/cfg refit (coordinator-supplied)
HEAD_SIGMA_FLOOR = 4.8e-3


def head_cam_base(lift, pitch_deg=HEAD_PITCH_DEG, left=True):
    """ZED LEFT-eye pose in the BASE frame for a lift height: (origin (3,), R_cam_to_base (3,3), MuJoCo convention: columns x right, y up, z back).  Known mount + lift only.
    Camera looks along robot +x pitched down; image right = robot -y (zed_demo/rig._zed_cams); left eye = centre - baseline/2 * x."""
    p = np.deg2rad(pitch_deg)
    fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
    up = np.array([np.sin(p), 0.0, np.cos(p)])
    z = -fwd
    x = np.cross(up, z)
    pos = HEAD_POS + np.array([0.0, 0.0, float(lift)])
    if left:
        pos = pos - 0.5 * HEAD_BASELINE * x
    return pos, np.stack([x, up, z], 1)


def head_spec(W, H):
    """depth_sim.CamSpec of the ZED head: stereo law with the ZED geometry at THIS resolution (f from fovy; native_H = H so f_native = f), refit sigma_disp 0.176 px, 4.8 mm floor,
    2 % dropout, 0.2 % outliers, 1 mm quantisation, grazing > 80 deg invalid, valid range [0.3, 5] m."""
    from world_model.v2.tray_rep.perception import depth_sim as D
    return D.CamSpec(name="zed_head", mount="world", pos=(0.0, 0.0, 0.0), R=tuple(np.eye(3).ravel()), fovy=HEAD_FOVY, W=W, H=H, native_H=H, baseline=HEAD_BASELINE,
                     sigma_disp_px=HEAD_SIGMA_DISP_PX, sigma_floor=HEAD_SIGMA_FLOOR, z_min=0.30, z_max=5.0, assumed_depth=False, note="ZED head (proposed torso mount)")


def build_head_model(sim):
    """recompile the TeleopSim rig (same arguments, as stereo/twin.build_twin_model) through MjSpec with a fixed world camera `zed_left` added (pose is overwritten every frame
    from the base pose + lift); asserts the state layout equals the sim's."""
    import mujoco
    import sim_common
    from world_model.v2.house.teleop import scene as SC
    orig = sim_common._patch_real_collision_geoms

    def hooked(spec):
        orig(spec)
        spec.worldbody.add_camera(name="zed_left", pos=[0, 0, 0], fovy=HEAD_FOVY)

    m0, info = sim.m, sim.rig.info
    wp = sim.scene.waypoints
    has = lambda n: mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
    sim_common._patch_real_collision_geoms = hooked
    try:
        rig = SC.build_rig(sim.scene, info["spawn"][:2], info["yaw"], dt=info["dt"], with_people=sim.with_people, counter_at=wp["cafe_counter_pick"] if has("tp_counter") else None,
                           desk_at=wp["lab_desk_place"] if has("tp_desk") else None, cup=info["has_cup"], dynamic_base=info["dynamic_base"], finger_contact=info["finger_contact"])
    finally:
        sim_common._patch_real_collision_geoms = orig
    m = rig.m
    for k in ("nq", "nv", "nmocap", "neq", "nbody", "ngeom"):
        assert getattr(m, k) == getattr(m0, k), (k, getattr(m, k), getattr(m0, k))
    return m


def scene_constants(sim) -> dict:
    """cup / bead / tray constants read off a freshly built TeleopSim (initial state): bead centres in the cup frame, bead radius, the tray top colour."""
    m, d = sim.m, sim.d
    names = [f"ball_{i}" for i in range(sim.rig.n_beads)]
    Rc0, pc0 = d.xmat[sim.cup].reshape(3, 3), d.xpos[sim.cup]
    rel = np.array([Rc0.T @ (d.xpos[m.body(n).id] - pc0) for n in names])
    g = [g for g in range(m.ngeom) if (m.body(m.geom_bodyid[g]).name or "") == "ball_0"][0]
    gt = [g for g in range(m.ngeom) if m.geom(g).name == "tray_top"][0]
    return dict(bead_rel=rel, bead_r=float(m.geom_size[g][0]), tray_rgba=m.geom_rgba[gt].copy(), n_beads=len(names))


class FPRender:
    def __init__(self, seed: int = 0, W: int = 320, H: int = 180, scene: str = "h5", cam: str = "wrist"):
        assert cam in ("wrist", "head")
        self.cam = cam
        self.sides = SIDES if cam == "wrist" else ("head",)
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        from world_model.v2.house.teleop.sim import TeleopSim
        from world_model.v2.tray_rep.perception import depth_sim as D
        self.mj, self.W, self.H = mujoco, W, H
        sim = TeleopSim(scene, grasped=True, seed=seed, with_people=False)          # same construction as ocs.record.record_carry (people are irrelevant to a wrist view)
        self.sim = sim
        m = build_head_model(sim) if cam == "head" else sim.m
        self.m = m
        self._base, self._lift = np.zeros(3), 0.0
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = sim.d.qpos
        self.d.mocap_pos[:] = sim.d.mocap_pos
        self.d.mocap_quat[:] = sim.d.mocap_quat
        mujoco.mj_forward(m, self.d)
        self.lq, self.rq = np.asarray(sim.ctl.lq), np.asarray(sim.ctl.rq)
        self.slider = list(sim.slider_q)
        self.tray_adr = int(m.jnt_qposadr[m.joint("tray_free").id])
        self.cup_adr = int(m.jnt_qposadr[m.body_jntadr[sim.cup]])
        self.cup = sim.cup
        names = [f"ball_{i}" for i in range(sim.rig.n_beads)]
        self.ball_adr = [int(m.jnt_qposadr[m.body_jntadr[m.body(n).id]]) for n in names]
        self.consts = scene_constants(sim)
        self.bead_rel = self.consts["bead_rel"]                                                     # cup-frame bead centres at the initial state
        if cam == "head":
            self.spec = {"head": head_spec(W, H)}
            self.cam_id = {"head": int(m.camera("zed_left").id)}
        else:
            self.spec = {s: D.CAMERAS[SPEC_NAME[s]] for s in SIDES}
            self.cam_id = {s: int(m.camera(CAM_NAME[s]).id) for s in SIDES}
        self.fovy = {s: float(m.cam_fovy[self.cam_id[s]]) for s in self.sides}
        self.K = {s: K_from_fovy(self.fovy[s], W, H) for s in self.sides}
        root = int(m.body("yor_root").id)
        anc = np.zeros(m.nbody, bool)
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == root:
                    anc[b] = True
                    break
                x = int(m.body_parentid[x])
        self.is_robot_geom = anc[m.geom_bodyid]
        self.is_cup_geom = np.array([m.geom_bodyid[g] == self.cup for g in range(m.ngeom)])
        self.is_bead_geom = np.array([(m.body(m.geom_bodyid[g]).name or "").startswith("ball_") for g in range(m.ngeom)])
        self.rend = RGBDRenderer(m, W, H, msaa=(cam != "head"))

    def close(self):
        self.rend.close()

    # ------------------------------------------------------------------ state
    def set_state(self, base, lift, tL, tR, tray_p, tray_q, cup_p, cup_q):
        d = self.d
        d.qpos[self.lq] = tL
        d.qpos[self.rq] = tR
        for a in self.slider:
            d.qpos[a] = 0.5 * float(lift)
        d.qpos[self.tray_adr:self.tray_adr + 3] = tray_p
        d.qpos[self.tray_adr + 3:self.tray_adr + 7] = tray_q
        d.qpos[self.cup_adr:self.cup_adr + 3] = cup_p
        d.qpos[self.cup_adr + 3:self.cup_adr + 7] = cup_q
        Rc = np.zeros(9)
        self.mj.mju_quat2Mat(Rc, np.asarray(cup_q, float))
        pb = np.asarray(cup_p, float) + self.bead_rel @ Rc.reshape(3, 3).T                   # beads move rigidly with the cup
        for a, p in zip(self.ball_adr, pb):
            d.qpos[a:a + 3] = p
            d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[:] = 0.0
        d.mocap_pos[0] = (float(base[0]), float(base[1]), 0.0)
        c, s = np.cos(0.5 * base[2]), np.sin(0.5 * base[2])
        d.mocap_quat[0] = (c, 0.0, 0.0, s)
        self._base, self._lift = np.asarray(base, float).copy(), float(lift)
        self.mj.mj_forward(self.m, d)

    def set_tick(self, rec, k):
        A = rec.A
        self.set_state(A["base"][k], A["lift"][k], A["tL"][k], A["tR"][k], A["tray_p"][k], A["tray_q"][k], A["cup_p"][k], A["cup_q"][k])

    # ------------------------------------------------------------------ render
    def cam_pose(self, side):
        c = self.cam_id[side]
        if side == "head":                                   # known mount + TRUE lift (what the camera physically is), rides the base pose
            o_b, R_b = head_cam_base(self._lift)
            cy, sy = np.cos(self._base[2]), np.sin(self._base[2])
            Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            return Rz @ o_b + np.array([self._base[0], self._base[1], 0.0]), Rz @ R_b
        return self.d.cam_xpos[c].copy(), self.d.cam_xmat[c].reshape(3, 3).copy()

    def render(self, side, tick, seed=0, noise=True, seg=True, groups=None):
        if side == "head":                                   # fixed world camera: overwrite its pose in the data (as stereo/sensor._place_cams)
            o, R = self.cam_pose("head")
            self.d.cam_xpos[self.cam_id["head"]] = o
            self.d.cam_xmat[self.cam_id["head"]] = R.ravel()
        rgb, dep, geom = self.rend.render(self.d, "zed_left" if side == "head" else CAM_NAME[side], seg=True, groups=groups)
        robot = np.zeros(dep.shape, bool) if geom is None else (geom >= 0) & self.is_robot_geom[np.maximum(geom, 0)]
        cup = None if geom is None else (geom >= 0) & (self.is_cup_geom | self.is_bead_geom)[np.maximum(geom, 0)]
        if noise:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFF, int(tick), zlib.crc32(self.spec[side].name.encode())]))
            dn = wrist_depth_noise(dep, self.K[side], self.spec[side], rng, robot)
        else:
            dn = np.where(robot, 0.0, dep).astype(np.float32)
        org, R = self.cam_pose(side)
        out = dict(rgb=rgb, depth=dn, depth_clean=dep, K=self.K[side].copy(), cam_pos=org, cam_R=R)
        if seg:
            out.update(cup_seg=cup, robot_seg=robot, geom=geom)
        return out

    def blob_check(self, rec, k, side):
        """sanity check vs the recorded 160x90 blob of tick k.  The recording's blob is a RAY-CAST colour blob over geom groups (0,3,4) (it does not see the gripper's visual meshes),
        so the like-for-like comparison renders those groups only: cup+bead pixel centroid of the render (seg buffer), rescaled to the 160x90 grid.
        Also reports the same for the default visual groups (what a real camera sees: the gripper occludes part of the cup).
        -> dict(n_rec, u_rec, v_rec, like_for_like=dict(n, du, dv), visual=dict(n, du, dv)) ; n in 160x90-pixel units; None entries if nothing seen."""
        bl = rec.A["wrist"][k, SIDES.index(side)]
        res = dict(n_rec=None if not np.isfinite(bl[0]) else int(bl[0]), u_rec=float(bl[1]), v_rec=float(bl[2]))
        sx, sy = 160.0 / self.W, 90.0 / self.H
        for name, groups in (("like_for_like", (0, 3, 4)), ("visual", None)):
            o = self.render(side, k, seed=int(rec.meta.get("seed", 0)), noise=False, groups=groups)
            v, u = np.nonzero(o["cup_seg"])
            if len(u) < 20:
                res[name] = dict(n=float(len(u)) * sx * sy, du=None, dv=None)
                continue
            u160, v160 = (u.mean() + 0.5) * sx - 0.5, (v.mean() + 0.5) * sy - 0.5
            res[name] = dict(n=float(len(u)) * sx * sy, du=float(u160 - bl[1]) if np.isfinite(bl[1]) else None, dv=float(v160 - bl[2]) if np.isfinite(bl[2]) else None)
        return res
