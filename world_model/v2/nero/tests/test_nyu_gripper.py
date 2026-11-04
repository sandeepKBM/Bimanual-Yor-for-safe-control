"""NERO + NYU gripper: compile, jaw range, mass, wrist camera sees the jaw tips (all CPU, no rendering)."""
import mujoco
import numpy as np
import pytest

from world_model.v2.nero import config as C
from world_model.v2.nero import nyu_gripper as G

XML = C.ASSETS / "nero_arm_nyugrip.xml"


@pytest.fixture(scope="module")
def mdl():
    G.write_standalone()
    m = mujoco.MjModel.from_xml_path(str(XML))
    return m, mujoco.MjData(m)


def _geoms(m, pre):
    return [i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(pre)]


def test_nyu_is_the_only_gripper():
    assert C.GRIPPER == "nyu" and not hasattr(C, "GRIPPER_VARIANT")


def test_compiles_and_mass(mdl):
    m, _ = mdl
    assert m.nu == 8 and m.neq == 2
    mass = sum(m.body_mass[m.body(n).id] for n in ("gripper_base", "nut", "finger1", "finger2"))
    assert mass == pytest.approx(G.YOR_MASS, rel=1e-3)                                  # YOR MJCF 0.45 kg
    assert m.body_subtreemass[m.body("gripper_base").id] == pytest.approx(G.YOR_MASS, rel=1e-3)   # cameras add no mass
    assert {"wrist_rgb", "wrist_depth"} <= {m.camera(i).name for i in range(m.ncam)}


def test_jaw_range_no_collision(mdl):
    """mj_geomDistance / contacts on the jaw hull pieces were unreliable (spurious 0.0), so measure the true jaw surfaces:
    40k sampled surface points per jaw arm (beyond the hinge), KD-tree distance, over the closing range."""
    import trimesh
    from scipy.spatial import cKDTree
    m, d = mdl
    pts = []
    for k, (nm, h) in enumerate((("finger_a", G.HINGE_A_MM), ("finger_b", G.HINGE_B_MM))):
        P, _ = trimesh.sample.sample_surface(trimesh.load(str(G.OUT_DIR / f"{nm}.stl")), 40000, seed=0)
        keep = (P[:, :2] - np.array(h) * 1e-3) @ G._jaw_dir(k)[:2] >= 0
        pts.append(P[keep] - np.array([h[0] * 1e-3, h[1] * 1e-3, G.JAW_Z_MID_MM * 1e-3]))
    j1, j2 = m.joint("gripper_joint1").qposadr[0], m.joint("gripper_joint2").qposadr[0]
    lo = 9.0
    for q in np.linspace(0, G.THETA_MAX_RAD, 30):
        d.qpos[j1] = d.qpos[j2] = q
        mujoco.mj_forward(m, d)
        W = [pts[k] @ d.xmat[m.body(b).id].reshape(3, 3).T + d.xpos[m.body(b).id] for k, b in enumerate(("finger1", "finger2"))]
        lo = min(lo, cKDTree(W[0]).query(W[1])[0].min())
    print("min jaw-jaw surface distance over 0..%.3f rad [m]: %.4f" % (G.THETA_MAX_RAD, lo))
    assert lo > 0.0


def test_nut_drives_jaws_kinematically(mdl):
    m, d = mdl
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    for _ in range(600):
        d.ctrl[7] = 0.9 * G.NUT_TRAVEL_M
        mujoco.mj_step(m, d)
    u = d.qpos[m.joint("gripper_nut").qposadr[0]]
    th = d.qpos[m.joint("gripper_joint1").qposadr[0]]
    assert abs(u - 0.9 * G.NUT_TRAVEL_M) < 5e-4
    assert abs(th - np.polyval(G.THETA_POLY[::-1], u)) < 0.01     # rad


def test_camera_sees_jaw_tips(mdl):
    """true jaw-tip vertices (mesh x < -130 mm) projected into the 640x360 RGB frame, jaws open and fully closed."""
    import trimesh
    m, d = mdl
    IW, IH = G.CAM_RES
    f = 0.5 * IH / np.tan(np.deg2rad(G.RGB_FOVY) / 2)
    tips = []
    for nm, h in (("finger_a", G.HINGE_A_MM), ("finger_b", G.HINGE_B_MM)):
        v = trimesh.load(str(G.OUT_DIR / f"{nm}.stl")).vertices
        tips.append(v[v[:, 0] < -0.130] - np.array([h[0] * 1e-3, h[1] * 1e-3, G.JAW_Z_MID_MM * 1e-3]))
    for q in (0.0, G.THETA_MAX_RAD):
        d.qpos[m.joint("gripper_joint1").qposadr[0]] = d.qpos[m.joint("gripper_joint2").qposadr[0]] = q
        mujoco.mj_forward(m, d)
        cam = m.camera("wrist_rgb")
        pos, R = d.cam_xpos[cam.id], d.cam_xmat[cam.id].reshape(3, 3)
        for T, b in zip(tips, ("finger1", "finger2")):
            Wp = T @ d.xmat[m.body(b).id].reshape(3, 3).T + d.xpos[m.body(b).id]
            pc = (Wp - pos) @ R                      # camera frame: x right, y up, -z forward
            assert (pc[:, 2] < 0).all()
            u, v = IW / 2 + f * pc[:, 0] / -pc[:, 2], IH / 2 - f * pc[:, 1] / -pc[:, 2]
            assert (u >= 0).all() and (u < IW).all() and (v >= 0).all() and (v < IH).all(), (q, b, u.min(), u.max(), v.min(), v.max())


def test_depth_camera_sees_fingers(mdl):
    """depth rendered from wrist_depth (left IR imager): finite everywhere and a visible share of pixels is near (< 0.25 m = the jaw)."""
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    m, d = mdl
    d.qpos[m.joint("gripper_nut").qposadr[0]] = 0.0
    d.qpos[m.joint("gripper_joint1").qposadr[0]] = d.qpos[m.joint("gripper_joint2").qposadr[0]] = 0.0
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, G.CAM_RES[1], G.CAM_RES[0])
    r.enable_depth_rendering()
    r.update_scene(d, "wrist_depth")
    dep = r.render()
    assert np.isfinite(dep).all()
    assert (dep < 0.25).mean() > 0.01
