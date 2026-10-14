"""onboard_capture.py -- capture the REFERENCE VIEWS for FoundationPose model-free onboarding from the robot's own simulated head ZED.  SIMULATION ONLY.

Scenario ("show me the object"): the object rests alone on the floor in front of the robot; the head camera (ZED intrinsics, 1280x720, fovy 70) visits a ring of
viewpoints around it (in reality: the base / lift / head pitch moving; the camera pose of every view then comes from odometry + FK -- here the commanded pose).
For every view we store RGB, depth, the camera-to-world pose, and (EVALUATION ONLY, never fed to onboarding) the rendered segmentation.

The MASKS used for onboarding are built later (masks.floor_gate_mask) from depth + the known floor plane + a 2D/3D seed -- NOT from the rendered segmentation (privileged).

    python -m world_model.v2.house.foundation.onboard_capture --obj tray --obj cup --n 24
Output  <OUT>/onboard/<obj>.npz : rgb (N,H,W,3) u8, depth (N,H,W) f32 [m] (MuJoCo buffer + sigma noise), seg (N,H,W) i16 [EVAL], cam_in_world (N,4,4) [OpenCV camera convention], K (3,3),
                                  obj_world (4,4) [EVAL: true pose of the object body frame at scan time], seed_xy (2,) [the robot's placement point, ~3 cm prior]
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

from .capture import NullBackend, OUT, SNAP

DEPTH_SIGMA = 0.0005         # m, simulated depth noise added to the MuJoCo buffer (onboarding views only; close-range, multi-frame-averaged ZED depth)


def lookat_R(eye, target, up=(0, 0, 1)):
    """camera-to-world rotation, columns (right, up, back) -- MuJoCo camera convention (looks along -z)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    u = np.cross(back, right)
    return np.stack([right, u, back], 1)


def ring_views(center, radius, n, elevs=(25.0, 45.0, 65.0)):
    out = []
    for i in range(n):
        az = 2 * math.pi * i / n + 0.3 * (i % 3)
        el = math.radians(elevs[i % len(elevs)])
        eye = np.asarray(center) + radius * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        out.append((eye, lookat_R(eye, center)))
    return out


def set_free(qpos, m, body, p, quat=(1, 0, 0, 0)):
    j = int(m.body_jntadr[m.body(body).id])
    a = int(m.jnt_qposadr[j])
    qpos[a:a + 3] = p
    qpos[a + 3:a + 7] = quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", action="append", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--W", type=int, default=1280)
    ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--dist", type=float, default=None)
    a = ap.parse_args()
    objs = a.obj or ["tray", "cup"]
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.stereo.cfg import StereoCfg
    from world_model.v2.house.stereo import twin as TW
    from world_model.v2.house.stereo.sensor import StereoSensor
    snap = np.load(SNAP / f"carry_s{a.seed}.npz", allow_pickle=True)
    sim = TeleopSim("h5", grasped=True, seed=a.seed, with_people=True, people_clock="time")
    cfg = StereoCfg(W=a.W, H=a.H, tex="speckle", gt=True, keep_rgb=True, self_mask=True, seed=a.seed)
    m = TW.build_twin_model(sim, cfg)
    s = StereoSensor(m, cfg, backend=NullBackend())
    rng = np.random.default_rng(5)
    bx, by, byaw = snap["base"][0]
    fwd = np.array([math.cos(byaw), math.sin(byaw)])
    spot = np.array([bx, by]) + 1.4 * fwd
    K = np.array([[cfg.f, 0, (a.W - 1) / 2], [0, cfg.f, (a.H - 1) / 2], [0, 0, 1]])
    od = OUT / "onboard"
    od.mkdir(parents=True, exist_ok=True)
    balls = [m.body(i).name for i in range(m.nbody) if (m.body(i).name or "").startswith("ball_")]
    for obj in objs:
        q = snap["qpos"][0].copy()
        other = "cup" if obj == "tray" else "tray"
        # hide the other object and the beads below the floor; put the scanned object upright on the floor at the spot
        set_free(q, m, other, [spot[0], spot[1], -5.0])
        for b in balls:
            set_free(q, m, b, [spot[0], spot[1], -5.0])
        z0 = 0.006 if obj == "tray" else 0.0           # tray body frame sits on its top face (plate is 6 mm thick); cup body frame on its base
        yaw = 0.4
        quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        set_free(q, m, obj, [spot[0], spot[1], z0], quat)
        center = np.array([spot[0], spot[1], 0.04 if obj == "cup" else 0.0])
        dist = a.dist or (0.55 if obj == "cup" else 0.85)
        views = ring_views(center, dist, a.n, elevs=(30.0, 50.0, 70.0) if obj == "cup" else (25.0, 45.0, 65.0))
        rec = {k: [] for k in ("rgb", "depth", "seg", "cam_in_world")}
        t0 = time.time()
        s.load(q, snap["mpos"][0], snap["mquat"][0], snap["eq"][0])
        obj_g = np.flatnonzero(m.geom_bodyid == m.body(obj).id)
        for eye, R in views:
            s.set_center_pose(eye, R)                    # the cyclopean centre is the stereo pair's mid-point; use the LEFT eye as the RGB-D frame origin
            fr = s.capture(0, 0.0)
            oL = fr["origin"]
            Tcw = np.eye(4)
            Tcw[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])      # -> OpenCV camera convention (x right, y down, z forward)
            Tcw[:3, 3] = oL
            d = fr["gt_depth"].astype(np.float32)
            d = d + rng.normal(0, DEPTH_SIGMA, d.shape).astype(np.float32) * (d > 0)
            rec["rgb"].append(fr["left"])
            rec["depth"].append(d)
            rec["seg"].append(fr["gt_seg"].astype(np.int16))
            rec["cam_in_world"].append(Tcw)
        Tobj = np.eye(4)
        Tobj[:3, 3] = [spot[0], spot[1], z0]
        Tobj[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        A = {k: np.asarray(v) for k, v in rec.items()}
        np.savez(od / f"{obj}.npz", K=K, obj_world=Tobj, seed_xy=spot + rng.normal(0, 0.03, 2), obj_geoms=obj_g, floor_z=0.0, **A)
        print(f"[onboard_capture] {obj}: {len(views)} views {a.W}x{a.H} in {time.time() - t0:.0f}s -> {od / (obj + '.npz')}", flush=True)
    s.close()
    os._exit(0)


if __name__ == "__main__":
    main()
