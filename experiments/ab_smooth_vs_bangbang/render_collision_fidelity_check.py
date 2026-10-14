"""Task 5: render one colliding-config frame (C0, no-safety) using the
CORRECTED collision proxy/geometry, to artifacts/collision_fidelity_check.png.
Re-runs C0 exactly as run_ab.run_condition does but also snapshots full qpos
at the step of minimum barrier h (deepest true intrusion), then renders that
state with the real per-link collision geoms visible (group 3) overlaid on
the normal visual mesh, so the actual penetrating capsules are visible.
"""
import sys
sys.path.insert(0, "/common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang")
import numpy as np
import mujoco
import sim_common as sc

OUT_PATH = "/common/users/ss5772/bimanual/artifacts/collision_fidelity_check.png"


def main():
    model = sc.load_model()
    data = sc.init_grasped_state(model)
    q0_left = data.qpos[sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)[1]].copy()
    sc.disable_left_arm_position_servos(model)
    ctrl = sc.EgoController(model, q0_left=q0_left)

    right_act_ids = [model.actuator(f"right_arm_joint{i}_pos").id for i in range(1, 7)]
    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id

    best_h = np.inf
    best_qpos = None
    best_t = None

    for k in range(sc.N_STEPS):
        t = k * sc.DT
        data.ctrl[right_act_ids] = sc.intruder_ctrl(t)
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)

        tau_task, M_ll, bias_l = ctrl.task_torque(data)
        bar = ctrl.barrier(data)
        data.qfrc_applied[ctrl.left_dofadr] = np.clip(tau_task, -sc.TAU_MAX, sc.TAU_MAX)

        if bar["h"] < best_h:
            best_h = bar["h"]
            best_qpos = data.qpos.copy()
            best_t = t

        mujoco.mj_step(model, data)

    print(f"min h over C0 rollout: {best_h:.4f} at t={best_t:.3f}s")

    # Re-set to the worst-h snapshot and confirm via BOTH the proxy and the
    # real-geom ground truth (mj_geomDistance) that this is a genuine
    # penetrating configuration.
    data.qpos[:] = best_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    bar_check = ctrl.barrier(data)
    print(f"re-check proxy h at snapshot: {bar_check['h']:.4f} (dist={bar_check['dist']:.4f})")

    left_geoms = [g for g in range(model.ngeom)
                  if model.body(model.geom_bodyid[g]).name.startswith("left_arm_")
                  and model.geom_group[g] == 3]
    right_geoms = [g for g in range(model.ngeom)
                   if model.body(model.geom_bodyid[g]).name.startswith("right_arm_")
                   and model.geom_group[g] == 3]
    fromto = np.zeros(6)
    true_min = min(mujoco.mj_geomDistance(model, data, g1, g2, 10.0, fromto)
                    for g1 in left_geoms for g2 in right_geoms)
    print(f"ground-truth mj_geomDistance min at snapshot: {true_min:.4f}m "
          f"({'PENETRATING' if true_min < 0 else 'not touching'})")

    # Render: enable collision-geom group (3) visibility alongside visual (2)
    # so the actual overlapping capsules are visible, not just the meshes.
    renderer = mujoco.Renderer(model, height=960, width=1280)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[3] = 1  # show "collision" class geoms
    scene_option.geomgroup[2] = 1  # keep visual meshes too
    renderer.update_scene(data, camera="head_cam", scene_option=scene_option)
    img = renderer.render()

    import imageio.v2 as imageio
    imageio.imwrite(OUT_PATH, img)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
