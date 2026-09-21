"""One-off probe (v3 re-point): find LEFT-arm grasp IK + a RIGHT-arm intrusion
pose that produces a genuine near/actual collision between the two arms'
collision-proxy spheres on the new scenes/scene_yor_canonical.xml (arms 0.185m
apart, lift-mounted). Prints numbers to hardcode into sim_common.py (same
convention as v1/v2's probe2.py) -- NOT imported at runtime.
"""
import numpy as np
import mujoco

SCENE = "/common/users/ss5772/bimanual/scenes/scene_yor_canonical.xml"


def coarse_seed(m, d, site_id, gripper_body_id, target, joint_slice, n=40000, seed=0):
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
        vert_penalty = abs(open_y[2])
        down_penalty = 1.0 + approach_z[2]
        cost = pos_err * 2.0 + vert_penalty + down_penalty
        if cost < best[0]:
            best = (cost, q.copy())
    return best[1]


def ik_to_target(m, d, site_id, target, q0, joint_slice, iters=400, damp=1e-2):
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

    box_bid = m.body("test_box").id
    box_pos0 = d.xpos[box_bid].copy()
    print("box pos:", box_pos0)

    left_site = m.site("left_arm_ee_grasp").id
    left_gbody = m.body("left_arm_gripper_base").id
    right_site = m.site("right_arm_ee_grasp").id
    right_gbody = m.body("right_arm_gripper_base").id

    box_half = 0.02
    grasp_tgt = box_pos0 + np.array([0.0, 0.0, box_half - 0.005])
    q_seed = coarse_seed(m, d, left_site, left_gbody, grasp_tgt, slice(0, 6))
    grasp_q, err = ik_to_target(m, d, left_site, grasp_tgt, q_seed, slice(0, 6))
    print(f"LEFT grasp IK error: {err*1000:.3f} mm  q={grasp_q}")

    # left arm frozen at grasp_q; find left gripper world pos there
    d.qpos[0:6] = grasp_q
    d.qpos[6:8] = [0.0, 0.0]
    mujoco.mj_forward(m, d)
    left_gpos = d.xpos[left_gbody].copy()
    left_site_pos = d.site_xpos[left_site].copy()
    print("left gripper pos @ grasp:", left_gpos, " left grasp SITE pos:", left_site_pos)

    # RIGHT home ctrl = keyframe ready pose
    right_home_q = np.array(m.key_ctrl[0][6:12])
    print("right home q:", right_home_q)

    # RIGHT intrude target: aim the right grasp SITE at the LEFT grasp SITE's
    # position (site-to-site, not site-to-body -- avoids the fixed ~0.10m
    # site->gripper_base offset dominating the resulting body-body distance).
    intrude_tgt = left_site_pos.copy()
    d.qpos[0:6] = grasp_q  # keep left frozen while solving right IK
    q_seed_r = coarse_seed(m, d, right_site, right_gbody, intrude_tgt, slice(8, 14))
    if q_seed_r is None:
        print("coarse_seed found nothing within 10cm; widening target")
        q_seed_r = right_home_q.copy()
    right_intrude_q, err_r = ik_to_target(m, d, right_site, intrude_tgt, q_seed_r, slice(8, 14))
    print(f"RIGHT intrude IK error: {err_r*1000:.3f} mm  q={right_intrude_q}")

    # Evaluate min pairwise distance between collision-proxy spheres at the
    # fully-intruded pose (left frozen at grasp_q, right at right_intrude_q).
    d.qpos[0:6] = grasp_q
    d.qpos[8:14] = right_intrude_q
    mujoco.mj_forward(m, d)

    EGO = ["left_arm_link3", "left_arm_link5", "left_arm_gripper_base"]
    INT = ["right_arm_link3", "right_arm_link5", "right_arm_gripper_base"]
    best = 1e9
    best_pair = None
    for eb in EGO:
        pe = d.xpos[m.body(eb).id]
        for ib in INT:
            pi = d.xpos[m.body(ib).id]
            dist = float(np.linalg.norm(pe - pi))
            if dist < best:
                best = dist
                best_pair = (eb, ib)
    print(f"min proxy-sphere distance at full intrusion: {best:.4f} m  pair={best_pair}")
    print("(compare to COMBINED_RADIUS = R_EGO+R_INT+D_MIN; need best < that for a genuine near-collision)")

    # Also report right gripper pos actually reached, and left/right elbow (link3) sep at home (no intrusion)
    d.qpos[8:14] = right_home_q
    mujoco.mj_forward(m, d)
    home_min = 1e9
    for eb in EGO:
        pe = d.xpos[m.body(eb).id]
        for ib in INT:
            pi = d.xpos[m.body(ib).id]
            dist = float(np.linalg.norm(pe - pi))
            home_min = min(home_min, dist)
    print(f"min proxy-sphere distance at right-HOME (no intrusion): {home_min:.4f} m")


if __name__ == "__main__":
    main()
