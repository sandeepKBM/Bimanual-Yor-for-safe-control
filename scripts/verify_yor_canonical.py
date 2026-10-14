"""
Headless verification for scenes/scene_yor_canonical.xml (the canonical YOR
bimanual scene). Checks:
  1. Model loads; prints nq/nv/nu, joint/actuator/camera inventory.
  2. Left-arm gripper can grasp the test box and lift it (IK to a grasp pose,
     close fingers, IK to a point 10cm higher, check contact forces + box
     height delta).
  3. All cameras (both wrist cams + head_cam) render non-degenerate RGB.

Run: MUJOCO_GL=egl python scripts/verify_yor_canonical.py

Requires `mujoco` + `PIL` (conda env `safe_env` in this repo).
"""
import mujoco
import numpy as np
from PIL import Image

SCENE = "/common/users/ss5772/bimanual/scenes/scene_yor_canonical.xml"
ARTIFACTS = "/common/users/ss5772/bimanual/artifacts"


def coarse_seed(m, d, site_id, gripper_body_id, target, joint_slice, n=40000, seed=0):
    """Random-sample the joint_slice's range to find a good IK initial guess.
    Position-only IK on a 6-DOF arm is under-constrained (3 constraints, 6
    DOF): gradient descent from an arbitrary seed (e.g. the resting/home pose)
    can converge to a reachable-but-badly-oriented elbow configuration (e.g.
    the finger open/close axis pointing vertically instead of straddling the
    box horizontally). Among samples that land near the target, this also
    scores orientation -- preferring configs where the gripper approaches
    from straight above (local +Z axis pointing down) with the finger
    open/close axis (local +Y) roughly horizontal -- so the seeded IK
    reliably lands in a "reach down, straddle sideways" grasp configuration."""
    rng = np.random.default_rng(seed)
    jr = m.jnt_range[joint_slice].copy()
    best = (1e9, None)
    for _ in range(n):
        q = rng.uniform(jr[:, 0], jr[:, 1])
        d.qpos[joint_slice] = q
        mujoco.mj_forward(m, d)
        pos_err = np.linalg.norm(d.site_xpos[site_id] - target)
        if pos_err > 0.10:
            continue
        xmat = d.xmat[gripper_body_id].reshape(3, 3)
        approach_z, open_y = xmat[:, 2], xmat[:, 1]
        vert_penalty = abs(open_y[2])       # want open axis horizontal (~0)
        down_penalty = 1.0 + approach_z[2]  # want approach axis pointing down (~-1)
        cost = pos_err * 2.0 + vert_penalty + down_penalty
        if cost < best[0]:
            best = (cost, q.copy())
    return best[1]


def ik_to_target(m, d, site_id, target, q0, joint_slice, iters=400, damp=1e-2):
    """Damped-least-squares position IK over the 6 revolute joints in
    joint_slice (e.g. slice(0,6) for the left arm)."""
    q = q0.copy()
    jr = m.jnt_range[joint_slice].copy()
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    err = np.zeros(3)
    for _ in range(iters):
        d.qpos[joint_slice] = q
        mujoco.mj_forward(m, d)
        p = d.site_xpos[site_id].copy()
        err = target - p
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        Jp = jacp[:, joint_slice]
        dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + damp * np.eye(3), err)
        q = np.clip(q + dq, jr[:, 0], jr[:, 1])
    return q, np.linalg.norm(err)


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ncam={m.ncam}")
    print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)])
    print("cameras:  ", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)])

    box_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "test_box")
    box_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "test_box_geom")
    lpad7 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_arm_link7_pad")
    lpad8 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_arm_link8_pad")
    grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_arm_ee_grasp")
    gripper_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_arm_gripper_base")

    box_pos0 = d.xpos[box_bid].copy()
    print("box initial pos:", box_pos0)

    # --- IK: left-arm grasp pose right at the box top ---
    box_half = 0.02
    grasp_tgt = box_pos0 + np.array([0.0, 0.0, box_half - 0.005])  # near box top
    q_seed = coarse_seed(m, d, grasp_site, gripper_body, grasp_tgt, slice(0, 6))
    grasp_q, err = ik_to_target(m, d, grasp_site, grasp_tgt, q_seed, slice(0, 6))
    print(f"grasp IK error: {err * 1000:.3f} mm")

    # --- Simulate: teleport to grasp pose (fingers open), settle, close, lift ---
    d2 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d2, 0)
    d2.qpos[0:6] = grasp_q
    d2.qpos[6:8] = [0.03, -0.03]  # left fingers open
    d2.qvel[:] = 0
    mujoco.mj_forward(m, d2)

    d2.ctrl[:] = m.key_ctrl[0]
    d2.ctrl[0:6] = grasp_q
    d2.ctrl[12], d2.ctrl[13] = 0.03, -0.03
    for _ in range(300):
        mujoco.mj_step(m, d2)

    d2.ctrl[12], d2.ctrl[13] = 0.0, 0.0  # command fingers closed
    for _ in range(500):
        mujoco.mj_step(m, d2)

    def grip_contacts(dd):
        n, fmax = 0, 0.0
        for i in range(dd.ncon):
            c = dd.contact[i]
            names = {c.geom1, c.geom2}
            if box_geom in names and (lpad7 in names or lpad8 in names):
                n += 1
                f = np.zeros(6)
                mujoco.mj_contactForce(m, dd, i, f)
                fmax = max(fmax, np.linalg.norm(f[:3]))
        return n, fmax

    n_close, f_close = grip_contacts(d2)
    box_z_before = d2.xpos[box_bid][2].copy()
    print(f"after close: contacts={n_close} max_force={f_close:.3f}N box_z={box_z_before:.4f} "
          f"finger_qpos={d2.qpos[6:8]}")

    lift_tgt = d2.site_xpos[grasp_site].copy() + np.array([0.0, 0.0, 0.10])
    lift_q, lift_err = ik_to_target(m, d2, grasp_site, lift_tgt, d2.qpos[0:6].copy(), slice(0, 6))
    print(f"lift IK error: {lift_err * 1000:.3f} mm")
    # ik_to_target scribbled on d2.qpos while solving; rewind to the settled grasp state.
    d2.qpos[0:6] = grasp_q
    mujoco.mj_forward(m, d2)

    d2.ctrl[0:6] = lift_q
    for _ in range(1500):
        mujoco.mj_step(m, d2)

    box_z_after = d2.xpos[box_bid][2].copy()
    n_lift, f_lift = grip_contacts(d2)
    lifted_cm = (box_z_after - box_z_before) * 100
    grasp_pass = (n_lift >= 1 and f_lift > 0.01 and lifted_cm > 2.0)
    print(f"after lift: contacts={n_lift} max_force={f_lift:.3f}N box_z={box_z_after:.4f} "
          f"lifted={lifted_cm:.2f}cm")
    print("GRASP+LIFT TEST:", "PASS" if grasp_pass else "FAIL")

    # --- Camera checks: render + report std (0 std == blank/degenerate) ---
    r = mujoco.Renderer(m, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.lookat = [0.42, -0.21, 0.65]
    cam.distance = 0.7
    cam.azimuth = 40
    cam.elevation = -15
    r.update_scene(d2, camera=cam)
    Image.fromarray(r.render()).save(f"{ARTIFACTS}/yor_canonical_grasp.png")

    cam_stds = {}
    cam_imgs = {}
    for cname in ["left_arm_egocentric", "right_arm_egocentric", "head_cam"]:
        r.update_scene(d2, camera=cname)
        img = r.render()
        cam_imgs[cname] = img
        cam_stds[cname] = float(img.std())
    wrist_montage = np.concatenate(
        [cam_imgs["left_arm_egocentric"], cam_imgs["right_arm_egocentric"]], axis=1)
    Image.fromarray(wrist_montage).save(f"{ARTIFACTS}/yor_canonical_wristcam.png")
    Image.fromarray(cam_imgs["head_cam"]).save(f"{ARTIFACTS}/yor_canonical_headcam.png")
    print("camera RGB std:", cam_stds)
    cams_pass = all(v > 1.0 for v in cam_stds.values())
    print("CAMERA TEST:", "PASS" if cams_pass else "FAIL")

    print("\nOVERALL:", "PASS" if (grasp_pass and cams_pass) else "FAIL")


if __name__ == "__main__":
    main()
