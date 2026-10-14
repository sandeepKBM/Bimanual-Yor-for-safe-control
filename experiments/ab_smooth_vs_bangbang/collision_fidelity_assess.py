"""Task 2+3: ground-truth cross-arm collision oracle + 3-sphere proxy fidelity
assessment on scenes/scene_yor_canonical.xml.

Ground truth = mj_geomDistance over ALL cross-arm "collision"-class geom pairs
(group==3: the fitted capsules per link + finger-pad boxes), which
diag_contact_drop.py verified matches MuJoCo's own contact-generation exactly
(0/10 missed) once contype/conaffinity is patched correctly on the REAL geoms
(not just the 3-sphere proxy add-on).

Proxy = the experiment's existing EGO_BARRIER_BODIES/INT_BARRIER_BODIES
3-sphere-per-arm set (link3, link5, gripper_base), body-origin-to-body-origin
distance minus COMBINED_RADIUS (sim_common.min_pair_distance / .barrier()).

Sampling mix (~2000 total):
  - 1000 uniform: both arms' 6 joints uniform in their full jnt_range.
  - 1000 task-region: left arm at LEFT_GRASP_Q + small noise (the grasp pose
    used throughout the experiment), right arm interpolated along the
    home->intrude schedule (probe_v3-derived RIGHT_HOME_CTRL/RIGHT_INTRUDE_CTRL,
    the "both grippers try to occupy the shared workspace point (0.42,-0.21)"
    IK solution) at a random factor in [-0.1, 1.15] + noise -- this is exactly
    the region where cross-arm contact actually occurs, so it is where proxy
    fidelity matters most.
"""
import sys
sys.path.insert(0, "/common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang")
import numpy as np
import mujoco

SCENE = "/common/users/ss5772/bimanual/scenes/scene_yor_canonical.xml"

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 7)]

JOINT_RANGES = np.array([
    [-2.61, 2.61], [0.0, 3.13], [-2.965, 0.0], [-1.74, 1.74], [-1.2, 1.2], [-3.0, 3.0],
])

LEFT_GRASP_Q = np.array([-0.00944173, 2.69291536, -1.83582068, -0.47028722, 1.03733985, -1.30204097])
RIGHT_HOME_CTRL = np.array([0.0, 1.58065, -0.578175, 0.0, -0.912, -0.78])
RIGHT_INTRUDE_CTRL = np.array([-0.15229176, 2.56294267, -1.48373594, 0.93891875, 1.01939931, 0.80796997])

EGO_BARRIER_BODIES = ["left_arm_link3", "left_arm_link5", "left_arm_gripper_base"]
INT_BARRIER_BODIES = ["right_arm_link3", "right_arm_link5", "right_arm_gripper_base"]
R_EGO, R_INT, D_MIN = 0.035, 0.035, 0.03
COMBINED_RADIUS = R_EGO + R_INT + D_MIN  # 0.10

RNG = np.random.default_rng(0)
N_UNIFORM = 1000
N_TASK = 1000


def load_model_patched():
    """Real-geometry model with contype/conaffinity patched on ALL left_arm_*/
    right_arm_* collision-class (group==3) geoms so cross-arm pairs can be
    queried/compiled consistently. No sphere proxies added -- geometry only."""
    spec = mujoco.MjSpec.from_file(SCENE)
    LEFT_BIT, RIGHT_BIT = 0b0100, 0b1000
    for b in spec.bodies:
        if b.name.startswith("left_arm_"):
            for g in b.geoms:
                if g.contype == 0 and g.conaffinity == 1:
                    g.contype, g.conaffinity = LEFT_BIT, RIGHT_BIT
        elif b.name.startswith("right_arm_"):
            for g in b.geoms:
                if g.contype == 0 and g.conaffinity == 1:
                    g.contype, g.conaffinity = RIGHT_BIT, LEFT_BIT
    return spec.compile()


def joint_qposadr(model, names):
    return np.array([model.jnt_qposadr[model.joint(n).id] for n in names])


def collision_geom_ids(model, prefix):
    ids = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        if model.body(bid).name.startswith(prefix) and model.geom_group[gid] == 3:
            ids.append(gid)
    return ids


def link_of(model, gid):
    """Map a geom id to its owning link/body name (for the per-link-pair miss
    breakdown)."""
    return model.body(model.geom_bodyid[gid]).name


def sample_configs():
    configs = []
    for _ in range(N_UNIFORM):
        q_left = RNG.uniform(JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])
        q_right = RNG.uniform(JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])
        configs.append((q_left, q_right, "uniform"))
    for _ in range(N_TASK):
        q_left = LEFT_GRASP_Q + RNG.normal(0, 0.12, size=6)
        q_left = np.clip(q_left, JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])
        s = RNG.uniform(-0.1, 1.15)
        q_right = RIGHT_HOME_CTRL + s * (RIGHT_INTRUDE_CTRL - RIGHT_HOME_CTRL)
        q_right = q_right + RNG.normal(0, 0.05, size=6)
        q_right = np.clip(q_right, JOINT_RANGES[:, 0], JOINT_RANGES[:, 1])
        configs.append((q_left, q_right, "task"))
    return configs


def main():
    model = load_model_patched()
    data = mujoco.MjData(model)

    left_qposadr = joint_qposadr(model, LEFT_ARM_JOINTS)
    right_qposadr = joint_qposadr(model, RIGHT_ARM_JOINTS)

    left_geoms = collision_geom_ids(model, "left_arm_")
    right_geoms = collision_geom_ids(model, "right_arm_")

    ego_body_ids = [model.body(b).id for b in EGO_BARRIER_BODIES]
    int_body_ids = [model.body(b).id for b in INT_BARRIER_BODIES]

    configs = sample_configs()
    fromto = np.zeros(6)

    n_true_collide = 0
    n_false_safe = 0
    n_proxy_unsafe = 0
    n_proxy_unsafe_true_safe = 0  # "false unsafe" (conservative, not a correctness bug)
    miss_pair_counts = {}
    worst_false_safe = None  # (true_dist, proxy_h)

    for q_left, q_right, tag in configs:
        data.qpos[left_qposadr] = q_left
        data.qpos[right_qposadr] = q_right
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # ground truth: true min distance (raw geometry, no margin) + which
        # link pair achieves it
        true_min = np.inf
        true_pair = None
        for g1 in left_geoms:
            for g2 in right_geoms:
                d = mujoco.mj_geomDistance(model, data, g1, g2, 10.0, fromto)
                if d < true_min:
                    true_min = d
                    true_pair = (link_of(model, g1), link_of(model, g2))
        true_collide = true_min < 0.0

        # proxy: 3-sphere body-origin distance
        best = np.inf
        for eid in ego_body_ids:
            pe = data.xpos[eid]
            for iid in int_body_ids:
                pi = data.xpos[iid]
                d = float(np.linalg.norm(pe - pi))
                if d < best:
                    best = d
        proxy_h = best - COMBINED_RADIUS
        proxy_safe = proxy_h >= 0.0

        if true_collide:
            n_true_collide += 1
        if proxy_safe and true_collide:
            n_false_safe += 1
            miss_pair_counts[true_pair] = miss_pair_counts.get(true_pair, 0) + 1
            if worst_false_safe is None or true_min < worst_false_safe[0]:
                worst_false_safe = (true_min, proxy_h, true_pair, tag)
        if not proxy_safe:
            n_proxy_unsafe += 1
            if true_min >= 0.0:
                n_proxy_unsafe_true_safe += 1

    n = len(configs)
    print(f"N configs: {n} ({N_UNIFORM} uniform + {N_TASK} task-region)")
    print(f"True collisions (real geom overlap, dist<0): {n_true_collide} ({100*n_true_collide/n:.2f}%)")
    print(f"Proxy says 'safe' (h>=0, i.e. dist >= {COMBINED_RADIUS}m): {n - n_proxy_unsafe} ({100*(n-n_proxy_unsafe)/n:.2f}%)")
    print()
    print(f"FALSE-SAFE (proxy h>=0 AND true penetration): {n_false_safe} / {n_true_collide} true-collisions "
          f"({100*n_false_safe/max(n_true_collide,1):.1f}% of true collisions missed by proxy)")
    print(f"  = {100*n_false_safe/n:.3f}% of all sampled configs")
    print(f"Proxy 'unsafe' but truly non-penetrating (conservative, not a correctness bug): "
          f"{n_proxy_unsafe_true_safe} / {n_proxy_unsafe}")
    print()
    print("Which link pairs the proxy misses (false-safe cases), by count:")
    for pair, cnt in sorted(miss_pair_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {pair}: {cnt}")
    if worst_false_safe:
        print(f"\nWorst false-safe: true_dist={worst_false_safe[0]:.4f}m (penetration "
              f"{-worst_false_safe[0]*100:.2f}cm), proxy_h={worst_false_safe[1]:.4f}m, "
              f"link_pair={worst_false_safe[2]}, sample={worst_false_safe[3]}")


if __name__ == "__main__":
    main()
