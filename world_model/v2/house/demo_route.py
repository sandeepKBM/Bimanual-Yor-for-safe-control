"""demo_route.py -- short video of one Phase 2 route carry (SIMULATION; new script, flagship demo not imported/edited).

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free GPU> python -m world_model.v2.house.demo_route --route h5 --tag a0.25_v0.3 \
        --seed 0 --out /data/local/ss5772/molmospaces/p2/demo_h5.mp4

Re-runs the same closed loop as phase2.run_episode (same seed => same motion) with a render hook.  HUD pattern copied
from scripts/make_cup_collect_safety_demo.build_hud_and_encode (PIL text over RGB frames, imageio libx264).
The textures of the house are stripped (plain colours): physics/rays do not need them.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import mujoco

from . import phase2 as P2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="h5")
    ap.add_argument("--tag", default="a0.25_v0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=8, help="render every N physics steps (dt 5 ms => 25 fps at 8)")
    ap.add_argument("--out", default=str(P2.RUNS / "demo_h5.mp4"))
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio

    W, H = 1280, 720
    font = lambda sz: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    F1, F2, F3 = font(26), font(19), font(14)
    state = dict(r=None, cam=None, frames=[], prev=None)
    traj = P2.load_traj(a.route, a.tag)
    ref = P2.ROUTES[a.route]
    info = dict(house=ref["house"])
    dt = P2.DT

    def cb(k, kk, m, d, st):
        if k % a.every:
            return
        if state["r"] is None:
            m.vis.global_.offwidth, m.vis.global_.offheight = W, H
            state["r"] = mujoco.Renderer(m, H, W)
            state["cam"] = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(state["cam"])
        cam = state["cam"]
        bx, by = d.mocap_pos[0, 0], d.mocap_pos[0, 1]
        cam.lookat[:] = (bx, by, 0.6)
        cam.distance, cam.elevation, cam.azimuth = 4.2, -72.0, 90.0
        state["r"].update_scene(d, camera=cam)
        img = Image.fromarray(state["r"].render())
        dr = ImageDraw.Draw(img, "RGBA")
        t = kk * dt
        sp = 0.0
        if state["prev"] is not None:
            sp = math.hypot(bx - state["prev"][0], by - state["prev"][1]) / (dt * a.every)
        state["prev"] = (bx, by)
        dr.rectangle([0, 0, W, 44], fill=(0, 0, 0, 150))
        dr.text((14, 7), f"MolmoSpaces house {ref['house']}: bimanual YOR carries a tray + ball through a door (FL-MPC, moving base)", font=F2, fill=(255, 255, 255))
        dr.rectangle([10, H - 128, 520, H - 30], fill=(0, 0, 0, 150))
        rel = st["rel"]
        dr.text((18, H - 124), f"t = {max(t, 0):5.1f} s   base speed {sp:4.2f} m/s", font=F2, fill=(255, 255, 255))
        dr.text((18, H - 100), f"ball offset on tray ({rel[0]*100:+4.1f}, {rel[1]*100:+4.1f}) cm   tilt {st['tilt']:.2f} deg", font=F2, fill=(255, 255, 255))
        dd = st["d_rh"]
        dr.text((18, H - 76), f"robot-house min dist {dd*100 if dd is not None else float('nan'):5.1f} cm   QP infeasible: {st['infeasible']}", font=F2, fill=(255, 255, 255))
        dr.text((18, H - 52), f"FL-MPC {st['ctl_ms']:4.1f} ms/call (python)   |tau|max {np.abs(st['tau']).max():.2f} Nm", font=F2, fill=(255, 255, 255))
        dr.rectangle([0, H - 22, W, H], fill=(0, 0, 0, 150))
        dr.text((10, H - 20), "SIMULATION. Base = kinematic mocap (no swerve/wheel model). Tray held by two weld constraints. Textures stripped.", font=F3, fill=(255, 255, 255))
        state["frames"].append(np.asarray(img.convert("RGB")))

    summ, _ = P2.run_episode(ref["house"], traj, a.seed, door_bodies=[], frame_cb=cb, max_steps=a.max_steps)
    fps = 1.0 / (dt * a.every)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=None)
    for f in state["frames"]:
        w.append_data(f)
    w.close()
    print(dict(out=str(out), frames=len(state["frames"]), fps=fps, ball_lost=summ["ball_lost"],
               ball_drift_max_m=summ["ball_drift_max_m"], tilt_max=summ["tray_tilt_max_deg"]))


if __name__ == "__main__":
    main()
