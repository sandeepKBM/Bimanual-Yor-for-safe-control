"""pan/graspnet_sim.py -- graspnet-baseline (borrowed proposer) on the covered-pan scene, then OUR checks.  SIMULATION ONLY.

1. settle the scene, render a depth image from a virtual head-like camera (MuJoCo free camera; the real robot's camera mount is
   not modelled -- AGENTS.md: ZED torso mount is a proposal), unproject to an OpenCV camera-frame point cloud (workspace-cropped);
2. run pan/graspnet_proposer.py in the graspnet env on westeros GPU 6 (subprocess; the caller must hold the GPU 6 lock);
3. map grasps to the world frame and to our claw: approach = R[:,0], jaw = R[:,1], fingertip target = t + depth * approach
   (graspnetAPI gripper model: fingertips at +depth along the approach from the grasp centre);
4. label each grasp by the nearest pan part (handle / wall / lid / knob / other);
5. geometric admission with OUR claw (80 mm fingers, not graspnet's 60 mm): left-arm IK + contact-free pre-grasp (handle_grasp.find_pregrasp);
6. --lift K: physics lift test (handle_grasp.run with left_grasp=...) for the top K admitted grasps.

    python -m world_model.v2.nero.pan.graspnet_sim --placement far_centre --out DIR [--top 30] [--lift 5]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import mujoco

from . import covered as CV
from . import scene as S
from . import handle_grasp as H

GN_PY = "/data/local/ss5772/graspnet/env/bin/python"
GN_SCRIPT = str(Path(__file__).with_name("graspnet_proposer.py"))
CAM = dict(distance=0.75, azimuth=0.0, elevation=-45.0, width=640, height=480)   # looks along +x from behind the robot's hands
CROP = dict(x=(0.15, 0.70), y=(-0.60, 0.10), z=(0.77, 1.05))                    # world box kept in the cloud (stove, pan, counter)


def render_cloud(m, d, lookat):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    r = mujoco.Renderer(m, CAM["height"], CAM["width"])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    depth = r.render().copy()
    gl = r.scene.camera[0]
    pos = np.array(gl.pos, float)
    fwd = np.array(gl.forward, float)
    up = np.array(gl.up, float)
    fwd /= np.linalg.norm(fwd)
    up /= np.linalg.norm(up)
    right = np.cross(fwd, up)
    R_wc = np.stack([right, -up, fwd], 1)                     # OpenCV camera axes (x right, y down, z forward) in world
    fovy = np.radians(m.vis.global_.fovy)
    f = 0.5 * CAM["height"] / np.tan(0.5 * fovy)
    v, u = np.mgrid[0:CAM["height"], 0:CAM["width"]]
    z = depth
    x = (u - 0.5 * CAM["width"] + 0.5) * z / f
    y = (v - 0.5 * CAM["height"] + 0.5) * z / f
    pc = np.stack([x, y, z], -1).reshape(-1, 3)               # renderer image row 0 = top (MuJoCo Renderer flips for us)
    ok = np.isfinite(pc).all(1) & (pc[:, 2] > 0.05) & (pc[:, 2] < 3.0)
    pc = pc[ok]
    pw = pc @ R_wc.T + pos
    keep = ((pw[:, 0] > CROP["x"][0]) & (pw[:, 0] < CROP["x"][1]) & (pw[:, 1] > CROP["y"][0]) & (pw[:, 1] < CROP["y"][1])
            & (pw[:, 2] > CROP["z"][0]) & (pw[:, 2] < CROP["z"][1]))
    return pc[keep].astype(np.float32), pw[keep], R_wc, pos, f


def _box_dist(p, c, R, half):
    q = np.abs(R.T @ (p - c)) - half
    return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))


def label(m, d, p):
    best = ("other", 0.03)
    for g in range(m.ngeom):
        nm = m.geom(g).name
        if not (nm.startswith("pan_") or nm.startswith("lid")) or "table" in nm:
            continue
        sz = m.geom_size[g].copy()
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            sz = np.array([sz[0], sz[0], sz[1]])
        dist = _box_dist(p, d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), sz[:3])
        if dist < best[1]:
            part = ("handle" if "handle" in nm else "knob" if "knob" in nm else "lid" if nm.startswith("lid") else
                    "wall" if "wall" in nm else "bottom" if "bottom" in nm else nm)
            best = (part, dist)
    return best[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="far_centre", choices=list(CV.ASSUMED_COVERED["placements"]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--lift", type=int, default=0)
    ap.add_argument("--gpu", default="6")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cp = CV.CoveredParams(placement=a.placement)
    m = CV.compose(cp)
    m.opt.timestep = H.DT
    d = mujoco.MjData(m)
    S.settle(m, d, 1.0)
    lookat = d.site_xpos[m.site("pan_centre").id].copy()
    pc, pw, R_wc, cam_pos, f = render_cloud(m, d, lookat)
    np.savez(out / "cloud.npz", points=pc)
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    pr = subprocess.run([GN_PY, GN_SCRIPT, "--in", str(out / "cloud.npz"), "--out", str(out / "grasps_cam.json"), "--top", str(a.top)],
                        env=env, capture_output=True, text=True)
    if pr.returncode:
        raise SystemExit(pr.stderr[-2000:])
    grasps = json.load(open(out / "grasps_cam.json"))
    rows = []
    PAN_PARTS = ("handle", "wall", "lid", "knob", "bottom")
    for i, g in enumerate(grasps):
        Rc, tc = np.array(g["rotation"]), np.array(g["translation"])
        Rw, tw = R_wc @ Rc, R_wc @ tc + cam_pos
        appr, jaw = Rw[:, 0], Rw[:, 1]
        tip = tw + g["depth"] * appr
        row = dict(rank=i, score=g["score"], width=g["width"], depth=g["depth"], part=label(m, d, tw),
                   centre_w=tw.tolist(), approach_w=appr.tolist(), jaw_w=jaw.tolist(), target_w=tip.tolist(),
                   width_ok=bool(g["width"] < 0.098))                       # our claw: 100 mm max, keep 2 mm margin
        if row["part"] not in ("handle", "wall", "lid", "knob", "bottom"):
            row.update(left_reachable_clean=None)
            rows.append(row)
            continue
        try:
            pre = H.find_pregrasp(m, d, "left", tip, appr, jaw, n_sol=6, seeds=40)
            row.update(left_reachable_clean=True, margin=pre["margin"])
        except RuntimeError:
            row.update(left_reachable_clean=False)
        rows.append(row)
    n_all = len(rows)
    by_part_all = {p: sum(r["part"] == p for r in rows) for p in sorted({r["part"] for r in rows})}
    rows = [r for r in rows if r["part"] in PAN_PARTS]          # only grasps on the pan itself (stove/counter edges dropped)
    summ = dict(placement=a.placement, n_points=int(len(pc)), camera=dict(CAM, pos=cam_pos.tolist(), f_px=f),
                proposer_s=round(time.time() - t0, 2), proposer_stdout=pr.stdout.strip()[-300:],
                n_proposals=n_all, by_part_all=by_part_all, n_on_pan=len(rows),
                n_admitted_geom=sum(r["left_reachable_clean"] and r["width_ok"] for r in rows), grasps=rows)
    lifts = []
    for r in [r for r in rows if r["left_reachable_clean"] and r["width_ok"]][:a.lift]:
        res = H.run(a.placement, grip="force", grip_force=10.0,
                    left_grasp=dict(target=r["target_w"], approach=r["approach_w"], jaw=r["jaw_w"]))
        lifts.append(dict(rank=r["rank"], part=r["part"], score=r["score"], held=res["held"], lift_cm=100 * res["lift_achieved_m"],
                          tilt_deg=res["final_tilt_deg"], support_N=res["hold_last1s_support_Fz_N"],
                          slip_deg=res["slip_final_deg"], slip_mm=res["slip_final_mm"], n_fallback=res["n_fallback"]))
    summ["lift_tests"] = lifts
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "grasps"}, indent=1, default=float))


if __name__ == "__main__":
    main()
