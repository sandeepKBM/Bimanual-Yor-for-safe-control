"""NERO port tests.  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/nero/tests   (~1.5 min: one hold IK + 1 s FL-MPC)"""
import numpy as np
import pytest
import mujoco

from world_model.v2.nero import config as C
from world_model.v2.nero import adapters as A
from world_model.v2.nero import ik as IK
from world_model.v2.nero import verify_arm as V
from world_model.v2.nero import proxy as P
from world_model.v2.nero import urdf_to_mjcf as X


@pytest.fixture(scope="module")
def arm():
    return V.load()


@pytest.fixture(scope="module")
def hold():
    m = A.compose_scene()
    d = mujoco.MjData(m)
    info = IK.set_hold_state(m, d)
    return m, d, info


def test_fk_matches_urdf_chain(arm):
    m, d = arm
    r = V.fk_check(m, d, n=25)
    assert r["n_configs"] >= 20
    assert r["max_pos_err_m"] < 1e-9
    assert r["max_rot_err_rad"] < 1e-6


def test_mass_inertia_carried(arm):
    m, _ = arm
    r = V.inertia_check(m)
    assert r["worst"]["mass"] < 1e-9 and r["worst"]["com"] < 1e-9 and r["worst"]["inertia_rel"] < 1e-5
    assert abs(r["total_mass_urdf"] - r["total_mass_mjcf"]) < 1e-9


def test_joint_ranges_equal_urdf(arm):
    m, _ = arm
    jr = np.array([m.jnt_range[m.joint(f"joint{i+1}").id] for i in range(7)])
    assert np.abs(jr - C.URDF_LIMITS).max() < 1e-9
    assert np.abs(C.SDK_DOC_LIMITS - C.URDF_LIMITS).max() < 0.02         # URDF is the rounded version of the SDK doc table


def test_gravity_torque_sane(arm):
    m, d = arm
    g = V.gravity_report(m, d, n_rand=50)
    assert max(g["sampled_peak_abs_tau_g"]) < 30.0
    assert g["sampled_peak_abs_tau_g"][0] < 1e-6      # joint 1 is vertical: no gravity torque


def test_proxy_covers_collision_mesh_vertices():
    spec = P.load()
    links, _ = __import__("world_model.v2.nero.urdf_fk", fromlist=["x"]).parse()
    from world_model.v2.nero import urdf_fk as U
    for un, body in P.BODY.items():
        L = links[un]
        V_ = P.load_stl(C.MESH_DIR["agx_gripper" if "agx_gripper" in L.coll_mesh else "nero"] / (L.coll_mesh.split("/")[-1]))
        V_ = V_ @ U.rpy_to_R(*L.coll_rpy).T + L.coll_xyz
        c, r = np.array(spec[body]["centers"]), np.array(spec[body]["radii"])
        inside = (np.linalg.norm(V_[:, None, :] - c[None], axis=-1) <= r[None] + 1e-9).any(1)
        assert inside.all(), body


def test_variant_hold_pose(hold):
    m, d, info = hold
    assert m.nu == 2 * (7 + 1)
    assert info["ik_err"]["left"][0] < 1e-4 and info["ik_err"]["right"][0] < 1e-4
    assert d.ncon == 0                                  # zero self/arm-arm contacts at the hold pose
    lo, hi = C.URDF_LIMITS[:, 0], C.URDF_LIMITS[:, 1]
    for s in C.SIDES:
        q = np.array(info["hold_q"][s])
        assert (q > lo).all() and (q < hi).all()


def test_pair_barrier_format(hold):
    m, d, info = hold
    pb = A.PairBarriers(m)
    pairs = pb(d)
    assert len(pairs) == 16 and pairs[0]["Je"].shape == (3, 7) and pairs[0]["Ji"].shape == (3, 7)
    assert abs(np.linalg.norm(pairs[0]["n"]) - 1) < 1e-9
    assert pairs[0]["h"] <= pairs[-1]["h"]


def test_flmpc_nv7_static_hold_1s():
    from world_model.v2.nero import hold_test as H
    r = H.run(seconds=1.0)
    assert r["config"]["NV"] == 7
    assert r["n_infeasible"] == 0
    assert r["ball_stays"]
    assert r["tau_peak_over_limit"] < 1.0


def test_nullspace_keeps_hand_pose(hold):
    from world_model.v2.nero import nullspace as N
    m, d, info = hold
    sm = N.SelfMotion(m, "left", info["hold_q"]["left"])
    path = sm.branch(+1, max_shift=0.06)
    assert path[-1][0] >= 0.05
    for s, q in path[::5]:
        ep, er = sm.pose_err(q)
        assert np.linalg.norm(ep) < 1e-6 and np.linalg.norm(er) < 1e-6          # << 1 mm


def test_proxy_conservative_vs_hull_distance():
    from world_model.v2.nero import variant_checks as VC
    m = A.compose_scene(with_tray=False)
    d = mujoco.MjData(m)
    info = IK.set_hold_state(A.compose_scene(), mujoco.MjData(A.compose_scene()))
    r = VC.proxy_conservative(m, info["hold_q"], n=200)
    assert r["false_safe_count"] == 0 and r["max_proxy_minus_true_m"] < 0.0
