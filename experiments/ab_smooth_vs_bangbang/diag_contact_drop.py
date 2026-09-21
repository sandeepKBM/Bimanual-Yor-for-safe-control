"""One-off diagnostic: why does MuJoCo drop a verified-penetrating cross-arm
mesh/capsule contact on the full scene_yor_canonical.xml model?

Read-only w.r.t. scene files; patches contype/conaffinity in-memory via MjSpec
(same technique as sim_common.py) but on the REAL per-link collision geoms
(not the sphere-proxy workaround), so we can see what the full mesh/capsule
geometry actually does once the contype=0 filter bug is fixed.
"""
import sys
sys.path.insert(0, "/common/users/ss5772/bimanual/experiments/ab_smooth_vs_bangbang")
import numpy as np
import mujoco

SCENE = "/common/users/ss5772/bimanual/scenes/scene_yor_canonical.xml"

LEFT_GRASP_Q = np.array([-0.00944173, 2.69291536, -1.83582068, -0.47028722, 1.03733985, -1.30204097])
RIGHT_INTRUDE_CTRL = np.array([-0.15229176, 2.56294267, -1.48373594, 0.93891875, 1.01939931, 0.80796997])

_LEFT_BIT, _RIGHT_BIT = 0b0100, 0b1000


def build_model(patch_real_geoms=True):
    spec = mujoco.MjSpec.from_file(SCENE)
    left_geoms, right_geoms = [], []
    for g in spec.geoms:
        bname = g.name  # geoms often unnamed; use body instead
    # iterate bodies instead, patch all geoms belonging to left_arm_*/right_arm_* bodies
    for b in spec.bodies:
        if b.name.startswith("left_arm_"):
            for g in b.geoms:
                left_geoms.append(g)
        elif b.name.startswith("right_arm_"):
            for g in b.geoms:
                right_geoms.append(g)
    if patch_real_geoms:
        for g in left_geoms:
            if g.contype == 0 and g.conaffinity == 1:  # "collision" class real geoms only
                g.contype = _LEFT_BIT
                g.conaffinity = _RIGHT_BIT
        for g in right_geoms:
            if g.contype == 0 and g.conaffinity == 1:
                g.contype = _RIGHT_BIT
                g.conaffinity = _LEFT_BIT
    model = spec.compile()
    return model


def set_intrusion_pose(model, data):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for i in range(1, 7):
        data.qpos[model.joint(f"left_arm_joint{i}").id + 0] = 0.0  # placeholder, fixed below
    # proper qpos addressing
    for i, jn in enumerate([f"left_arm_joint{i}" for i in range(1, 7)]):
        data.qpos[model.jnt_qposadr[model.joint(jn).id]] = LEFT_GRASP_Q[i]
    for i, jn in enumerate([f"right_arm_joint{i}" for i in range(1, 7)]):
        data.qpos[model.jnt_qposadr[model.joint(jn).id]] = RIGHT_INTRUDE_CTRL[i]
    mujoco.mj_forward(model, data)


def collision_geom_ids(model, prefix):
    ids = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = model.body(bid).name
        # restrict to the REAL "collision" class geoms (group==3): the fitted
        # capsules + finger pad boxes. Excludes group==2 visual meshes (which
        # always geometrically overlap and are contype=conaffinity=0 anyway)
        # and group==0 camera-marker boxes.
        if bname.startswith(prefix) and model.geom_group[gid] == 3:
            ids.append(gid)
    return ids


def main():
    model = build_model(patch_real_geoms=True)
    data = mujoco.MjData(model)
    set_intrusion_pose(model, data)

    print("=== model summary ===")
    print("ngeom:", model.ngeom, "nbody:", model.nbody)
    print("ncon (mj_forward, real geoms, filter patched):", data.ncon)
    def gdesc0(gid):
        bid = model.geom_bodyid[gid]
        return f"{model.body(bid).name}[geom{gid}:{model.geom(gid).name or '(unnamed)'}]"

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        print(f"  contact {i}: {gdesc0(g1)} <-> {gdesc0(g2)} dist={c.dist:.5f}")

    left_ids = collision_geom_ids(model, "left_arm_")
    right_ids = collision_geom_ids(model, "right_arm_")
    print(f"\nleft_arm geoms: {len(left_ids)}, right_arm geoms: {len(right_ids)}")

    print("\n=== mj_geomDistance ground truth over ALL cross-arm geom pairs ===")
    def gdesc(gid):
        bid = model.geom_bodyid[gid]
        return f"{model.body(bid).name}[geom{gid}:{model.geom(gid).name or '(unnamed)'}]"

    penetrating = []
    fromto = np.zeros(6)
    for g1 in left_ids:
        for g2 in right_ids:
            dist = mujoco.mj_geomDistance(model, data, g1, g2, 10.0, fromto)
            if dist < 0.0:
                penetrating.append((g1, g2, dist))
    print(f"# geom pairs with mj_geomDistance < 0 (true penetration): {len(penetrating)}")
    for g1, g2, d in penetrating:
        print(f"  PENETRATING: {gdesc(g1)} <-> {gdesc(g2)} dist={d:.5f}")

    # cross check: are all penetrating pairs present in data.contact?
    contact_pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        contact_pairs.add(frozenset((c.geom1, c.geom2)))
    missed = [p for p in penetrating if frozenset((p[0], p[1])) not in contact_pairs]
    print(f"\n# true-penetrating pairs MISSING from data.contact: {len(missed)} / {len(penetrating)}")
    for g1, g2, d in missed:
        print(f"  MISSED: {gdesc(g1)} <-> {gdesc(g2)} dist={d:.5f}")

    print("\nmodel.opt.disableflags:", model.opt.disableflags)
    print("nconmax:", model.nconmax, "njmax:", model.njmax)


if __name__ == "__main__":
    main()
