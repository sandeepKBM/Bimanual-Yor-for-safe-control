"""
Headless verification for scenes/scene_bimanual_pickup.xml.

Run with:
    MUJOCO_GL=egl /common/users/ss5772/miniforge3/bin/conda run -n safe_env \
        python /common/users/ss5772/bimanual/scripts/test_pickup_scene.py

Checks (see PASS/FAIL summary at the end):
  1. scene loads
  2. gripper actuators present (left/right joint7_pos, joint8_pos)
  3. fingers actuate (commanded qpos delta on joint7/joint8)
  4. box graspable (nonzero contact force between a fingertip pad and a box,
     AND the box rises with the arm when the arm is commanded upward while
     the gripper stays closed)

Control-assumption notes for readers of this script:
  - All 16 actuators are <position> actuators (PD position servos), NOT torque/motor
    actuators. ctrl units for the 6 arm joints (joint1..joint6, per side) are RADIANS
    (revolute joints); ctrl units for the 2 finger joints (joint7, joint8, per side)
    are METERS (prismatic/slide joints).
  - Finger convention: joint7 in [0, 0.035] m, joint8 in [-0.035, 0] m (mirrored axes).
    ctrl = (0, 0)      -> gripper CLOSED (pad faces touching/overlapping, verified by FK
                           probe: pad-to-pad center distance ~0.005 m at this setting).
    ctrl = (0.035, -0.035) -> gripper fully OPEN (pad-to-pad center distance ~0.075 m,
                           i.e. ~0.045 m clear gap between the two pad faces).
  - Everything here is a POSITION command; there is no velocity/torque control exposed.
"""

import os
import numpy as np
import mujoco

SCENE = "/common/users/ss5772/bimanual/scenes/scene_bimanual_pickup.xml"
ART_DIR = "/common/users/ss5772/bimanual/artifacts"
INITIAL_PNG = os.path.join(ART_DIR, "scene_bimanual_pickup_initial.png")
GRASP_PNG = os.path.join(ART_DIR, "scene_bimanual_pickup_grasp.png")

os.makedirs(ART_DIR, exist_ok=True)

results = {}  # name -> (bool, evidence string)


def render(model, data, path, width=960, height=720):
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.28, 0.0, 0.22])
    cam.distance = 1.35
    cam.azimuth = 140
    cam.elevation = -25
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    renderer.close()
    import imageio.v2 as imageio
    imageio.imwrite(path, pixels)
    return path


def main():
    # ---- Step 1: load ----
    try:
        model = mujoco.MjModel.from_xml_path(SCENE)
        data = mujoco.MjData(model)
        results["scene_loads"] = (True, f"nq={model.nq} nu={model.nu} nbody={model.nbody}")
    except Exception as e:
        results["scene_loads"] = (False, repr(e))
        print_summary()
        return

    print(f"nq = {model.nq}")
    print(f"nu = {model.nu}")
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    print("actuators:", actuator_names)

    gripper_actuators = [n for n in actuator_names if "joint7_pos" in n or "joint8_pos" in n]
    results["gripper_actuators_present"] = (
        len(gripper_actuators) == 4,
        f"found: {gripper_actuators}",
    )

    # Table top z: table body is at pos z=0.10 with a box geom half-size 0.10 -> top = 0.20
    table_body = model.body("table")
    table_geom = model.geom("table")
    table_top_z = table_body.pos[2] + table_geom.size[2]
    print(f"table top z = {table_top_z:.3f} m (expected 0.20 m)")

    # ---- reset to home keyframe, report initial box poses ----
    key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    box_names = ["box_center", "box_left", "box_right"]
    print("initial box poses (from home keyframe, resting on table):")
    for name in box_names:
        b = model.body(name)
        print(f"  {name}: pos={np.round(b.pos, 4)}")

    # ---- Step 2: render initial PNG (untouched home state) ----
    render(model, data, INITIAL_PNG)
    print(f"wrote {INITIAL_PNG}")

    # ---- Step 3: graspability check on LEFT arm + box_center ----
    j7_id = model.joint("left/joint7").id
    j8_id = model.joint("left/joint8").id
    j7_qposadr = model.jnt_qposadr[j7_id]
    j8_qposadr = model.jnt_qposadr[j8_id]

    j7_open = data.qpos[j7_qposadr]
    j8_open = data.qpos[j8_qposadr]
    print(f"left finger joints before closing: joint7={j7_open:.4f} joint8={j8_open:.4f}")

    # Compute the midpoint between the two fingertip pads while the gripper is OPEN
    # at the current (home) arm pose, then teleport box_center there so it sits
    # between the open jaws. This "cheats" past inverse kinematics per the task's
    # allowance -- we are not solving IK, just staging a box for the gripper to close on.
    pad7_id = model.geom("left/link7_pad").id
    pad8_id = model.geom("left/link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[pad7_id] + data.geom_xpos[pad8_id])
    print(f"fingertip pad midpoint (pre-grasp target) = {np.round(pad_mid, 4)}")

    box_center_jid = model.joint("box_center_free").id
    box_qposadr = model.jnt_qposadr[box_center_jid]
    box_qveladr = model.jnt_dofadr[box_center_jid]

    box_z_before_teleport = data.qpos[box_qposadr + 2]
    data.qpos[box_qposadr:box_qposadr + 3] = pad_mid
    data.qpos[box_qposadr + 3:box_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[box_qveladr:box_qveladr + 6] = 0.0
    mujoco.mj_forward(model, data)
    print(f"box_center teleported from table (z={box_z_before_teleport:.3f}) to "
          f"pre-grasp position {np.round(data.qpos[box_qposadr:box_qposadr+3], 4)} "
          "(between the open left fingers)")

    # We are not running a real reach controller (no vision/feedback loop, no IK) -- this
    # is an open-loop scripted test. To avoid an artificial race between "how fast the
    # fingers close" and "how fast the teleported box free-falls" (the box has no support
    # the instant it is teleported mid-air), gravity is switched off for the CLOSE phase
    # only, so we can cleanly verify "can the fingers make contact and squeeze the box"
    # in isolation. Gravity is restored before the LIFT phase below, which is exactly
    # where we want gravity active: a weak/incorrect grasp will drop the box instead of
    # lifting it with the arm.
    original_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0

    # Command the left gripper to CLOSE (ctrl -> 0, 0) while all other actuators
    # keep whatever ctrl the home keyframe set (arms hold their pose).
    j7_act = model.actuator("left/joint7_pos").id
    j8_act = model.actuator("left/joint8_pos").id
    data.ctrl[j7_act] = 0.0
    data.ctrl[j8_act] = 0.0

    for _ in range(500):
        mujoco.mj_step(model, data)

    model.opt.gravity[:] = original_gravity

    j7_closed = data.qpos[j7_qposadr]
    j8_closed = data.qpos[j8_qposadr]
    j7_delta = j7_open - j7_closed
    j8_delta = j8_closed - j8_open  # j8 goes from -0.035 towards 0, so closed-open is positive
    print(f"left finger joints after closing: joint7={j7_closed:.4f} joint8={j8_closed:.4f} "
          f"(delta7={j7_delta:.4f} m, delta8={j8_delta:.4f} m)")

    results["fingers_actuate"] = (
        j7_delta > 0.005 and j8_delta > 0.005,
        f"joint7 moved {j7_delta:.4f} m toward closed, joint8 moved {j8_delta:.4f} m toward closed",
    )

    # Check contact between the box and either fingertip pad, sum contact force magnitude.
    box_geom_id = model.geom("box_center").id

    def contact_force_sum():
        total = 0.0
        ncontacts = 0
        for i in range(data.ncon):
            con = data.contact[i]
            geoms = {con.geom1, con.geom2}
            if box_geom_id in geoms and (pad7_id in geoms or pad8_id in geoms):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total += np.linalg.norm(force[:3])
                ncontacts += 1
        return total, ncontacts

    force_after_close, ncontacts_after_close = contact_force_sum()
    print(f"contact force sum after closing = {force_after_close:.4f} N over {ncontacts_after_close} contact(s)")

    box_z_after_close = data.qpos[box_qposadr + 2]

    # ---- lift check: raise the arm a few cm via joint3 (elbow) while fingers stay closed ----
    # FK probe (see scratchpad/probe_lift.py) showed left/joint3 -= 0.10 rad raises
    # left/gripper_base by ~3.4 cm with <4 mm horizontal drift, at the home pose.
    j3_act = model.actuator("left/joint3_pos").id
    data.ctrl[j3_act] = data.ctrl[j3_act] - 0.10

    for _ in range(500):
        mujoco.mj_step(model, data)

    box_z_after_lift = data.qpos[box_qposadr + 2]
    force_after_lift, ncontacts_after_lift = contact_force_sum()
    lift_delta = box_z_after_lift - box_z_after_close
    print(f"box_center z: after close={box_z_after_close:.4f}  after lift={box_z_after_lift:.4f}  "
          f"delta={lift_delta:+.4f} m")
    print(f"contact force sum after lift = {force_after_lift:.4f} N over {ncontacts_after_lift} contact(s)")

    graspable = (
        (force_after_close > 1e-4 or force_after_lift > 1e-4)
        and lift_delta > 0.01
    )
    results["box_graspable"] = (
        graspable,
        f"contact force after close={force_after_close:.4f} N, after lift={force_after_lift:.4f} N; "
        f"box rose {lift_delta*100:.2f} cm with the arm",
    )

    # ---- Step 3 (cont'd): render after-grasp PNG ----
    render(model, data, GRASP_PNG)
    print(f"wrote {GRASP_PNG}")

    print_summary()


def print_summary():
    print("\n=== PASS/FAIL SUMMARY ===")
    order = ["scene_loads", "gripper_actuators_present", "fingers_actuate", "box_graspable"]
    for key in order:
        if key not in results:
            print(f"{key}: SKIPPED (earlier failure)")
            continue
        ok, evidence = results[key]
        status = "PASS" if ok else "FAIL"
        print(f"{key}: {status}  -- {evidence}")


if __name__ == "__main__":
    main()
