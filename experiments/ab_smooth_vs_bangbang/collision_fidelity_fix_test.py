"""Task 3/4 support: test the corrected multi-keypoint proxy's false-safe rate
against the same ground-truth oracle as collision_fidelity_assess.py, for two
candidate keypoint sets, before committing the final one to sim_common.py."""
import sys
sys.path.insert(0, "/common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang")
import numpy as np
import mujoco
from collision_fidelity_assess import (
    SCENE, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, JOINT_RANGES, LEFT_GRASP_Q,
    RIGHT_HOME_CTRL, RIGHT_INTRUDE_CTRL, D_MIN, load_model_patched,
    joint_qposadr, collision_geom_ids, link_of, sample_configs,
)

R_LINK_MARGIN = 0.0  # no extra per-link margin beyond D_MIN global clearance


def quat_rotate(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def keypoints_for_body(m, bid):
    kps = []
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != bid or m.geom_group[gid] != 3:
            continue
        gtype = m.geom_type[gid]
        pos = m.geom_pos[gid].copy()
        size = m.geom_size[gid].copy()
        quat = m.geom_quat[gid].copy()
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = quat_rotate(quat, np.array([0., 0., 1.]))
            hl, r = size[1], size[0]
            kps.append((pos + hl * axis, r))
            kps.append((pos - hl * axis, r))
            if hl > 0.05:  # long link: add a midpoint sphere too (covers the
                           # gap a 2-endpoint-only approximation leaves along
                           # the shaft of long links like link2/link3)
                kps.append((pos, r))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            r = float(np.linalg.norm(size))
            kps.append((pos, r))
        else:
            kps.append((pos, float(size[0])))
    return kps


def build_keypoint_set(model, link_names):
    out = []
    for side in ["left_arm_", "right_arm_"]:
        side_kps = []
        for ln in link_names:
            bid = model.body(side + ln).id
            for offset, r in keypoints_for_body(model, bid):
                side_kps.append((bid, offset, r))
        out.append(side_kps)
    return out[0], out[1]  # ego (left), int (right)


def proxy_min_h(data, ego_kps, int_kps, d_min):
    best = np.inf
    for bid_e, off_e, r_e in ego_kps:
        pe = data.xpos[bid_e] + data.xmat[bid_e].reshape(3, 3) @ off_e
        for bid_i, off_i, r_i in int_kps:
            pi = data.xpos[bid_i] + data.xmat[bid_i].reshape(3, 3) @ off_i
            d = float(np.linalg.norm(pe - pi))
            h = d - (r_e + r_i + d_min)
            if h < best:
                best = h
    return best


def run_variant(name, link_names):
    model = load_model_patched()
    data = mujoco.MjData(model)
    left_qposadr = joint_qposadr(model, LEFT_ARM_JOINTS)
    right_qposadr = joint_qposadr(model, RIGHT_ARM_JOINTS)
    left_geoms = collision_geom_ids(model, "left_arm_")
    right_geoms = collision_geom_ids(model, "right_arm_")
    ego_kps, int_kps = build_keypoint_set(model, link_names)

    configs = sample_configs()
    fromto = np.zeros(6)
    n_true_collide = 0
    n_false_safe = 0
    miss_pairs = {}
    for q_left, q_right, tag in configs:
        data.qpos[left_qposadr] = q_left
        data.qpos[right_qposadr] = q_right
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        true_min = np.inf
        true_pair = None
        for g1 in left_geoms:
            for g2 in right_geoms:
                d = mujoco.mj_geomDistance(model, data, g1, g2, 10.0, fromto)
                if d < true_min:
                    true_min = d
                    true_pair = (link_of(model, g1), link_of(model, g2))
        true_collide = true_min < 0.0
        if true_collide:
            n_true_collide += 1
        proxy_h = proxy_min_h(data, ego_kps, int_kps, D_MIN)
        if proxy_h >= 0.0 and true_collide:
            n_false_safe += 1
            miss_pairs[true_pair] = miss_pairs.get(true_pair, 0) + 1

    n = len(configs)
    print(f"\n=== variant: {name} ({len(ego_kps)} kp/arm) ===")
    print(f"true collisions: {n_true_collide}/{n} ({100*n_true_collide/n:.2f}%)")
    print(f"FALSE-SAFE: {n_false_safe}/{n_true_collide} true-collisions "
          f"({100*n_false_safe/max(n_true_collide,1):.2f}%), = {100*n_false_safe/n:.3f}% of all configs")
    if miss_pairs:
        print("miss link pairs:", sorted(miss_pairs.items(), key=lambda kv: -kv[1])[:10])


if __name__ == "__main__":
    run_variant("link1-6+gripper (14kp/arm)",
                ["link1", "link2", "link3", "link4", "link5", "link6", "gripper_base"])
    run_variant("link1-6+gripper+link7/8+iphone (20kp/arm)",
                ["link1", "link2", "link3", "link4", "link5", "link6", "gripper_base",
                 "link7", "link8", "iphone_body"])
    run_variant("link0-6+gripper+link7/8+iphone (22kp/arm)",
                ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base",
                 "link7", "link8", "iphone_body"])
