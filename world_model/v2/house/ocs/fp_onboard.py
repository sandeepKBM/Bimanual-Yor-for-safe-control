"""ocs/fp_onboard.py -- model-free FoundationPose onboarding of the SIM CUP (+ beads), ONCE.  SIMULATION ONLY.

    CUDA_VISIBLE_DEVICES=6 MUJOCO_GL=egl python -m world_model.v2.house.ocs.fp_onboard [--views 20] [--W 320 --H 180] [--n-step N]

Scene: an isolated MJCF with the sim cup (the cup body of scenes/scene_yor_cup_collect.xml, same materials) + its 8 beads (initial cup-relative layout of the h5 TeleopSim)
standing on a flat support (tray colour, top face z = 0 = the cup base plane), headlight + directional light like the house scenes.  A ring of virtual views (ring_views
style: ~0.30-0.40 m, elevations 25-65 deg, even azimuth + jitter) is rendered with the SAME renderer / depth model as the wrist re-render (fp_render: clip planes, wrist_l depth
noise: sigma floor 2 mm + stereo law, dropout, outliers, 1 mm quantisation).  Masks are NON-PRIVILEGED (foundation.masks.floor_gate_mask: depth + the known support plane + the
placement seed ~1 cm prior); the rendered segmentation is used only to SCORE them (IoU).  cam_in_obj = OpenCV camera pose in the CUP BASE frame (R_mj @ diag(1,-1,-1)), so the
FoundationPose mesh frame = cup base frame and ob_in_cam returned later has its origin at the cup base centre.
Outputs  <OUT>/cup_sim_mesh/cup_sim.obj (+mtl/png)  <OUT>/onboard_views.npz  <OUT>/onboard_info.json   (OUT = /data/local/ss5772/molmospaces/ocs/fp)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np

from world_model.v2.house.ocs import fp_render as FR

OUT = Path(os.environ.get("OCS_FP_OUT", "/data/local/ss5772/molmospaces/ocs/fp"))
MESH_DIR = OUT / "cup_sim_mesh"
MESH_PATH = MESH_DIR / "cup_sim.obj"


def lookat_R(eye, target, up=(0, 0, 1)):
    """camera-to-world rotation (columns right, up, back) -- MuJoCo camera convention (looks along -z)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back], 1)


def ring_views(center, dists, n, elevs=(25.0, 45.0, 65.0)):
    out = []
    for i in range(n):
        az = 2 * math.pi * i / n + 0.3 * (i % 3)
        el = math.radians(elevs[i % len(elevs)])
        r = dists[i % len(dists)]
        eye = np.asarray(center) + r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        out.append((eye, lookat_R(eye, center)))
    return out


def build_model(consts, fovy, rgba_cup="0.85 0.15 0.15 0.95", rgba_ball="0.15 0.55 0.9 1"):
    """isolated MJCF: static cup + beads standing on a flat support (top face z = 0), one mocap camera `cam`."""
    import re
    import mujoco
    from world_model.v2.data import contents as C
    cup = C._extract_body(C.SCENE.read_text(), "cup")
    cup = re.sub(r'<body name="cup" pos="[^"]+"', '<body name="cup" pos="0 0 0"', cup, count=1)
    cup = re.sub(r"<freejoint[^>]*/>", "", cup)
    cup = re.sub(r"<inertial[^>]*/>", "", cup)
    beads = "".join(f'<geom name="bead_{i}" type="sphere" size="{consts["bead_r"]:.6f}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" material="ball_mat" contype="0" conaffinity="0"/>'
                    for i, p in enumerate(consts["bead_rel"]))
    cup = cup.replace("</body>", beads + "</body>")
    tr = " ".join(f"{v:.4f}" for v in consts["tray_rgba"])
    xml = f"""<mujoco model="fp_cup_onboard">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <material name="cup_mat" rgba="{rgba_cup}"/>
    <material name="ball_mat" rgba="{rgba_ball}"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.0" directional="true"/>
    <geom name="support" type="box" size="0.6 0.6 0.01" pos="0 0 -0.01" rgba="{tr}" contype="0" conaffinity="0"/>
    {cup}
    <body name="cam" mocap="true" pos="0 0 1"><camera name="v" fovy="{fovy}"/></body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def capture_views(consts, fovy, W, H, n, dists, seed=0, noise=True):
    import mujoco
    from world_model.v2.tray_rep.perception import depth_sim as D
    m = build_model(consts, fovy)
    d = mujoco.MjData(m)
    rend = FR.RGBDRenderer(m, W, H)
    K = FR.K_from_fovy(fovy, W, H)
    spec = D.CAMERAS["wrist_l"]
    cupg = np.array([m.body(m.geom_bodyid[g]).name == "cup" for g in range(m.ngeom)])
    cid = m.body("cam").mocapid[0]
    views = ring_views(np.array([0.0, 0.0, 0.04]), dists, n)
    out = dict(rgb=[], depth=[], seg=[], cam_in_obj=[])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 17]))
    for i, (eye, R) in enumerate(views):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        d.mocap_pos[cid], d.mocap_quat[cid] = eye, q
        mujoco.mj_forward(m, d)
        rgb, dep, geom = rend.render(d, "v", seg=True)
        dn = FR.wrist_depth_noise(dep, K, spec, rng) if noise else dep
        T = np.eye(4)
        T[:3, :3] = R @ FR.FLIP
        T[:3, 3] = eye
        out["rgb"].append(rgb)
        out["depth"].append(dn)
        out["seg"].append((geom >= 0) & cupg[np.maximum(geom, 0)])             # EVALUATION ONLY
        out["cam_in_obj"].append(T)
    rend.close()
    return {k: np.asarray(v) for k, v in out.items()}, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--W", type=int, default=320)
    ap.add_argument("--H", type=int, default=180)
    ap.add_argument("--dist", type=float, nargs="+", default=[0.30, 0.35, 0.40])
    ap.add_argument("--n-step", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--h-min", type=float, default=0.008, help="support-plane gate: min height above the plane [m] (4 sigma of the 2 mm depth noise)")
    ap.add_argument("--radius", type=float, default=0.06, help="support-plane gate: max horizontal distance from the placement seed [m]")
    ap.add_argument("--views-only", action="store_true", help="render + mask the views and report mask IoU, skip the NOF (no GPU server)")
    a = ap.parse_args()
    assert a.gpu == 6, "this job is restricted to GPU 6"
    from world_model.v2.house.foundation import masks as M
    from world_model.v2.house.teleop.sim import TeleopSim
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim = TeleopSim("h5", grasped=True, seed=a.seed, with_people=False)
    consts = FR.scene_constants(sim)
    fovy = float(sim.m.cam_fovy[sim.m.camera("left_arm_egocentric").id])
    del sim
    V, K = capture_views(consts, fovy, a.W, a.H, a.views, a.dist, seed=a.seed, noise=not a.no_noise)
    print(f"[fp_onboard] rendered {a.views} views {a.W}x{a.H} in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(a.seed + 5)
    seed_xy = rng.normal(0, 0.01, 2)                                  # the robot's placement point: ~1 cm prior (a known drop point), NOT the segmentation
    mks, ious = [], []
    for i in range(len(V["rgb"])):
        dm = cv2.medianBlur(V["depth"][i].astype(np.float32), 5)           # 5x5 median: suppresses the 2 mm depth speckle / dropout before the plane gate (the gate is on the noisy depth, not the clean one)
        mk = M.floor_gate_mask(dm, K, V["cam_in_obj"][i], seed_xy, floor_z=0.0, h_min=a.h_min, h_max=0.40, radius=a.radius, min_px=30)
        mks.append(mk)
        ious.append(M.iou(mk, V["seg"][i]))
    mks = np.asarray(mks)
    print(f"[fp_onboard] mask IoU vs rendered seg (scoring only): mean {np.mean(ious):.3f} min {np.min(ious):.3f}; mask px median {int(np.median(mks.sum((1, 2))))}", flush=True)
    np.savez_compressed(OUT / "onboard_views.npz", rgb=V["rgb"], depth=V["depth"], mask=mks, seg=V["seg"], cam_in_obj=V["cam_in_obj"], K=K, seed_xy=seed_xy)
    info = dict(views=a.views, W=a.W, H=a.H, dists=a.dist, fovy=fovy, K=K.tolist(), mask_iou_mean=float(np.mean(ious)), mask_iou_min=float(np.min(ious)),
                mask_iou=[float(x) for x in ious], seed_xy=seed_xy.tolist(), noise=not a.no_noise, cup_dims=[float(v) for v in __import__("world_model.v2.house.ocs.util", fromlist=["x"]).cup_dims("sim")])
    if a.views_only:
        (OUT / "onboard_info.json").write_text(json.dumps(info, indent=1))
        return
    from world_model.v2.house.foundation.pose_fp import Client
    cl = Client(sizes=[f"{a.W}x{a.H}"], gpu=a.gpu)
    try:
        cl.warmup()
        MESH_DIR.mkdir(parents=True, exist_ok=True)
        r = cl.onboard("cup_sim", V["rgb"], V["depth"], mks, V["cam_in_obj"], K, n_step=a.n_step, save_mesh=str(MESH_PATH))
        info.update(onboard_s=r["onboard_s"], mesh_s=r.get("mesh_s"), extents=r["extents"], bounds=r["bounds"], n_vert=r["n_vert"], mesh=str(MESH_PATH))
        print(f"[fp_onboard] NOF {r['onboard_s']:.0f}s extents {np.round(r['extents'], 4)} bounds {np.round(r['bounds'], 4).tolist()} (true cup r/h {info['cup_dims']})", flush=True)
    finally:
        cl.close()
    (OUT / "onboard_info.json").write_text(json.dumps(info, indent=1))
    print(f"[fp_onboard] mesh -> {MESH_PATH}  total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
