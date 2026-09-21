"""
Headless verification for scenes/scene_bimanual_cup.xml ("don't-spill-the-cup" task).

Run with:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \
        python /common/users/ss5772/bimanual/scripts/test_cup_scene.py

Checks (see PASS/FAIL summary at the end):
  1. scene loads (with wrist camera attached via cup_task_utils.load_cup_model)
  2. both cameras (head_cam, right/wrist_cam) render non-degenerate RGB
  3. balls rest inside the cup (count_spilled == 0) after a settle period
  4. a scripted open-loop tilt of the cup spills balls out (count_spilled goes
     from 0 -> >0), confirmed by cup_task_utils.count_spilled

Also runs a secondary experiment: keep the cup UPRIGHT (no tilt) and shake it
laterally with a fast, high-amplitude schedule, to see whether pure
acceleration (no tipping) can eject balls. Reported but not part of PASS/FAIL.

Scripting note: the cup is a free body (not grasped by a gripper -- no IK/reach
controller exists in this repo). Both the tilt test and the upright-shake test
move the cup by directly overwriting its freejoint qpos each step to a
prescribed pose(t) (position + quaternion), zeroing its qvel, then calling
mj_step -- the same "teleport" style open-loop scripting used in
scripts/test_pickup_scene.py for the box. The balls are fully dynamic and are
pushed via contact against the moving cup.
"""

import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from cup_task_utils import load_cup_model, count_spilled, ball_body_ids, N_BALLS_DEFAULT

SCENE = "/common/users/ss5772/bimanual/scenes/scene_bimanual_cup.xml"
ART_DIR = "/common/users/ss5772/bimanual/artifacts"
HEADCAM_PNG = os.path.join(ART_DIR, "cup_headcam.png")
WRISTCAM_PNG = os.path.join(ART_DIR, "cup_wristcam.png")
MONTAGE_PNG = os.path.join(ART_DIR, "cup_spill_montage.png")

os.makedirs(ART_DIR, exist_ok=True)

results = {}  # name -> (bool, evidence string)


def render_cam(model, data, cam_name, width=640, height=480):
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam_name)
    pixels = renderer.render()
    renderer.close()
    return pixels


def render_free_cam(model, data, lookat, distance, azimuth, elevation, width=640, height=480):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array(lookat)
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    renderer.close()
    return pixels


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return 3 * t * t - 2 * t * t * t


def set_cup_pose(data, cup_qposadr, cup_dofadr, pos, quat):
    data.qpos[cup_qposadr:cup_qposadr + 3] = pos
    data.qpos[cup_qposadr + 3:cup_qposadr + 7] = quat
    data.qvel[cup_dofadr:cup_dofadr + 6] = 0.0


def run_tilt_test(model, data, base_pos, cup_qposadr, cup_dofadr, dt):
    """Lift the cup ~10cm clear of the table, tilt it ~100 deg about the world
    x-axis, hold, then let spilled balls settle. Returns (spilled_after, frames)
    where frames is a dict of a few snapshots for the montage."""
    lift_dur, tilt_dur, hold_dur, settle_dur = 0.3, 1.0, 0.4, 0.6
    lift_h = 0.10
    tilt_max = np.deg2rad(100)
    t_lift_end = lift_dur
    t_tilt_end = t_lift_end + tilt_dur
    t_hold_end = t_tilt_end + hold_dur
    t_total = t_hold_end + settle_dur

    frames = {}
    t = 0.0
    mid_captured = False
    while t < t_total:
        if t < t_lift_end:
            lift = lift_h * smoothstep(t / lift_dur)
            angle = 0.0
        elif t < t_tilt_end:
            lift = lift_h
            angle = tilt_max * smoothstep((t - t_lift_end) / tilt_dur)
        else:
            lift = lift_h
            angle = tilt_max

        pos = base_pos + np.array([0.0, 0.0, lift])
        quat = axis_angle_quat([1, 0, 0], angle)
        set_cup_pose(data, cup_qposadr, cup_dofadr, pos, quat)
        mujoco.mj_step(model, data)
        t += dt

        if not mid_captured and t >= t_tilt_end:
            frames["mid_tilt"] = render_free_cam(
                model, data, lookat=[0.32, 0.13, 0.28], distance=0.55, azimuth=110, elevation=-15)
            mid_captured = True

    frames["after"] = render_free_cam(
        model, data, lookat=[0.32, 0.13, 0.22], distance=0.55, azimuth=110, elevation=-15)
    spilled_after = count_spilled(model, data)
    return spilled_after, frames


def run_upright_shake_test(model, data, base_pos, cup_qposadr, cup_dofadr, dt):
    """Keep the cup upright (identity orientation) at its resting height and
    shake it laterally (fast sinusoid) to see if acceleration alone (no
    tipping) can eject balls over the rim."""
    amp = 0.06       # m, lateral shake amplitude
    freq = 8.0       # Hz
    shake_dur = 1.2
    settle_dur = 0.5
    t_total = shake_dur + settle_dur

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    t = 0.0
    while t < t_total:
        if t < shake_dur:
            x_off = amp * np.sin(2 * np.pi * freq * t)
        else:
            x_off = 0.0
        pos = base_pos + np.array([x_off, 0.0, 0.0])
        set_cup_pose(data, cup_qposadr, cup_dofadr, pos, identity_quat)
        mujoco.mj_step(model, data)
        t += dt

    return count_spilled(model, data)


def hstack_images(imgs):
    h = min(im.shape[0] for im in imgs)
    resized = [im[:h] for im in imgs]
    return np.concatenate(resized, axis=1)


def main():
    # ---- Step 1: load (with wrist cam attached) ----
    try:
        model, data = load_cup_model(SCENE)
        results["scene_loads"] = (True, f"nq={model.nq} nu={model.nu} nbody={model.nbody}")
    except Exception as e:
        results["scene_loads"] = (False, repr(e))
        print_summary()
        return

    print(f"nq = {model.nq}, nu = {model.nu}, timestep = {model.opt.timestep}")
    cam_names = [model.camera(i).name for i in range(model.ncam)]
    print("cameras:", cam_names)
    balls = ball_body_ids(model)
    print(f"ball count = {len(balls)} (expected {N_BALLS_DEFAULT})")

    key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    cup_bid = model.body("cup").id
    print(f"cup pose (home): pos={np.round(data.xpos[cup_bid], 4)} "
          f"quat={np.round(data.xquat[cup_bid], 4)}")

    cameras_ok = "head_cam" in cam_names and "right/wrist_cam" in cam_names
    results["cameras_present"] = (cameras_ok, f"found: {cam_names}")

    # ---- Step 2: settle (let balls come to rest inside the cup under gravity) ----
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    spilled_before = count_spilled(model, data)
    print(f"count_spilled after settle (should be 0) = {spilled_before}")
    results["balls_rest_before_shake"] = (
        spilled_before == 0, f"count_spilled = {spilled_before}")

    # ---- Step 3: render both cameras, save PNGs ----
    import imageio.v2 as imageio
    try:
        head_rgb = render_cam(model, data, "head_cam")
        wrist_rgb = render_cam(model, data, "right/wrist_cam")
        imageio.imwrite(HEADCAM_PNG, head_rgb)
        imageio.imwrite(WRISTCAM_PNG, wrist_rgb)
        cams_render_ok = (
            head_rgb.ndim == 3 and head_rgb.shape[-1] == 3 and head_rgb.std() > 1.0
            and wrist_rgb.ndim == 3 and wrist_rgb.shape[-1] == 3 and wrist_rgb.std() > 1.0
        )
        results["cameras_render_rgb"] = (
            cams_render_ok,
            f"head_cam shape={head_rgb.shape} std={head_rgb.std():.2f}; "
            f"wrist_cam shape={wrist_rgb.shape} std={wrist_rgb.std():.2f}"
        )
        print(f"wrote {HEADCAM_PNG}, {WRISTCAM_PNG}")
    except Exception as e:
        results["cameras_render_rgb"] = (False, repr(e))

    before_frame = render_free_cam(
        model, data, lookat=[0.32, 0.13, 0.22], distance=0.55, azimuth=110, elevation=-15)

    # ---- Step 4: scripted tilt/shake -> spill ----
    cup_joint = model.joint("cup_free")
    cup_qposadr = int(cup_joint.qposadr[0])
    cup_dofadr = int(cup_joint.dofadr[0])
    base_pos = data.xpos[cup_bid].copy()
    dt = model.opt.timestep

    spilled_after, frames = run_tilt_test(model, data, base_pos, cup_qposadr, cup_dofadr, dt)
    print(f"count_spilled after tilt (should be > 0) = {spilled_after}")
    results["tilt_spills_balls"] = (
        spilled_before == 0 and spilled_after > 0,
        f"spilled_before={spilled_before}, spilled_after={spilled_after}"
    )

    montage = hstack_images([before_frame, frames["mid_tilt"], frames["after"]])
    imageio.imwrite(MONTAGE_PNG, montage)
    print(f"wrote {MONTAGE_PNG} (before | mid-tilt | after)")

    # ---- Step 5 (experiment, not gating PASS/FAIL): upright fast shake ----
    model2, data2 = load_cup_model(SCENE)
    mujoco.mj_resetDataKeyframe(model2, data2, model2.key("home").id)
    mujoco.mj_forward(model2, data2)
    for _ in range(int(1.0 / model2.opt.timestep)):
        mujoco.mj_step(model2, data2)
    cup_bid2 = model2.body("cup").id
    base_pos2 = data2.xpos[cup_bid2].copy()
    cup_joint2 = model2.joint("cup_free")
    spilled_upright = run_upright_shake_test(
        model2, data2, base_pos2, int(cup_joint2.qposadr[0]), int(cup_joint2.dofadr[0]),
        model2.opt.timestep)
    print(f"count_spilled after UPRIGHT fast lateral shake (amp=0.06m, 8Hz) = {spilled_upright}")

    print_summary()
    print(f"\nupright-shake experiment: spilled={spilled_upright} "
          f"({'achieved' if spilled_upright > 0 else 'NOT achieved'} spill while cup stayed upright)")


def print_summary():
    print("\n=== PASS/FAIL SUMMARY ===")
    order = ["scene_loads", "cameras_present", "cameras_render_rgb",
             "balls_rest_before_shake", "tilt_spills_balls"]
    for key in order:
        if key not in results:
            print(f"{key}: SKIPPED (earlier failure)")
            continue
        ok, evidence = results[key]
        status = "PASS" if ok else "FAIL"
        print(f"{key}: {status}  -- {evidence}")


if __name__ == "__main__":
    main()
