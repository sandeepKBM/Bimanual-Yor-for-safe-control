"""Render artifacts/cup_obs_check.png: proof that cup_cam_a/cup_cam_b (the
fixed, close-in world-model observation cameras added to
scenes/scene_bimanual_cup.xml) clearly resolve the ball level inside the cup
at the 64x64 render resolution collection/cup_collect.py actually records.
Shows both cams side-by-side, before and after an upright-shake-induced spill.
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(__file__))
from cup_task_utils import load_cup_model, count_spilled

SCENE = "/common/users/ss5772/bimanual/scenes/scene_bimanual_cup.xml"
OUT = "/common/users/ss5772/bimanual/artifacts/cup_obs_check.png"


def render(model, data, cam, size=64):
    r = mujoco.Renderer(model, height=size, width=size)
    r.update_scene(data, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def upscale(img, factor=4):
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def main():
    model, data = load_cup_model(SCENE)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    before_a = render(model, data, "cup_cam_a")
    before_b = render(model, data, "cup_cam_b")
    spilled_before = count_spilled(model, data)

    cup_bid = model.body("cup").id
    cup_joint = model.joint("cup_free")
    qadr, dadr = int(cup_joint.qposadr[0]), int(cup_joint.dofadr[0])
    base_pos = data.xpos[cup_bid].copy()
    dt = model.opt.timestep
    amp, freq, shake_dur, settle = 0.04, 6.0, 1.2, 0.5
    ident = np.array([1.0, 0, 0, 0])
    t = 0.0
    while t < shake_dur + settle:
        xo = amp * np.sin(2 * np.pi * freq * t) if t < shake_dur else 0.0
        data.qpos[qadr:qadr + 3] = base_pos + np.array([xo, 0, 0])
        data.qpos[qadr + 3:qadr + 7] = ident
        data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_step(model, data)
        t += dt

    after_a = render(model, data, "cup_cam_a")
    after_b = render(model, data, "cup_cam_b")
    spilled_after = count_spilled(model, data)

    row1 = np.concatenate([upscale(before_a), upscale(before_b)], axis=1)
    row2 = np.concatenate([upscale(after_a), upscale(after_b)], axis=1)
    pad = np.full((8, row1.shape[1], 3), 255, dtype=np.uint8)
    montage = np.concatenate([row1, pad, row2], axis=0)
    imageio.imwrite(OUT, montage)
    print(f"wrote {OUT}  shape={montage.shape}")
    print(f"spilled_before={spilled_before} (top row: cup_cam_a | cup_cam_b, resting)")
    print(f"spilled_after={spilled_after} (bottom row: cup_cam_a | cup_cam_b, after upright shake)")


if __name__ == "__main__":
    main()
