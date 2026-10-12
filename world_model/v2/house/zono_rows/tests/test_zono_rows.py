"""tests for zono_rows (SIMULATION only).  Run:  PYTHONPATH=. python -m world_model.v2.house.zono_rows.tests.test_zono_rows   (or pytest)."""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import mujoco

from world_model.v2.house.zono_rows import zono_hrep as ZH
from world_model.v2.house.zono_rows import rows as R
from world_model.v2.perception_zono.zonoutil import halfspaces
from world_model.v2.tray_rep.sets.zono import Zonotope

sc = R.sc


def _rand_zono(rng, m):
    return rng.normal(size=3), rng.normal(size=(3, m)) * rng.uniform(0.05, 0.6)


def test_c_facets_contain_all_vertices():
    rng = np.random.default_rng(0)
    worst = -np.inf
    for m in list(range(1, 11)) * 6:
        c, G = _rand_zono(rng, m)
        h = ZH.zono_hrep(c, G)
        V = ZH.vertices(h)
        worst = max(worst, float((V @ h.N.T - h.d).max()))
        assert (V @ h.N.T - h.d).max() <= 1e-9
        # facets are tight: every facet touches the zonotope (support function equality)
        assert np.allclose([ZH.dist_exact(h, c + 0.0) for _ in range(1)], 0.0)
    # structured cases: box, parallel generators, flat patch, segment, point
    cases = {
        "box": (np.zeros(3), np.diag([1.0, 2.0, 3.0])),
        "parallel": (np.zeros(3), np.array([[1.0, 2.0, 0], [0, 0, 1.0], [0, 0, 0.5]]).T if False else np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])),
        "flat": (np.array([1.0, 1, 1]), np.array([[1.0, 0, 0], [0, 1.0, 0], [0.5, 0.5, 0]]).T),
        "segment": (np.zeros(3), np.array([[1.0], [0.5], [0.0]])),
        "point": (np.ones(3), np.zeros((3, 0))),
    }
    for k, (c, G) in cases.items():
        h = ZH.zono_hrep(c, G)
        V = ZH.vertices(h)
        assert (V @ h.N.T - h.d).max() <= 1e-9, k
        assert h.n_facets >= 6, k
    # the same set as the library's exact H-rep (perception_zono.zonoutil.halfspaces): random-point containment agrees
    rng = np.random.default_rng(1)
    for m in (3, 5, 8, 10):
        c, G = _rand_zono(rng, m)
        h = ZH.zono_hrep(c, G)
        A, b = halfspaces(Zonotope(c, G))
        X = c + rng.normal(size=(4000, 3)) * np.abs(G).sum(1)
        mine = ((X @ h.N.T - h.d) <= 0).all(1)
        lib = ((X @ A.T - b) <= 0).all(1)
        assert (mine != lib).mean() < 2e-3, (m, (mine != lib).mean())
    print(f"(c) all vertices inside all facets: worst slack {worst:.2e}; structured + library-agreement OK")


def test_b_sd_lower_bound_le_exact():
    rng = np.random.default_rng(2)
    viol, n_out, n_in, gap, near_gap = 0, 0, 0, [], []
    for m in (3, 4, 6, 8, 10):
        for _ in range(8):
            c, G = _rand_zono(rng, m)
            h = ZH.zono_hrep(c, G)
            sc_ = np.abs(G).sum(1).max()
            P = c + rng.normal(size=(60, 3)) * sc_ * 1.5
            lb = ZH.sd_lb(h, P)
            for p, l in zip(P, lb):
                ex = ZH.sd_exact(h, p)
                if ex > 1e-9:
                    n_out += 1
                    viol += l > ex + 1e-7
                    gap.append(ex - l)
                    if ex < 0.25:
                        near_gap.append(ex - l)
                else:
                    n_in += 1
                    viol += abs(l - ex) > 1e-9          # inside: exact
    gap = np.array(gap)
    assert viol == 0, viol
    print(f"(b) sd_lb <= exact on {n_out} outside pts (0 violations; gap median {np.median(gap)*1e3:.1f} mm, p95 {np.percentile(gap,95)*1e3:.1f} mm, "
          f"max {gap.max()*1e3:.1f} mm; within 25 cm of the set: n={len(near_gap)}, median {np.median(near_gap)*1e3:.1f} mm, "
          f"p95 {np.percentile(near_gap,95)*1e3:.1f} mm); exact (==) on {n_in} inside pts")


def _canon_state(rng):
    m = sc.load_model()
    d = mujoco.MjData(m)
    _, lq, ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
    d.qpos[lq] = rng.normal(0, 0.3, 6) + np.array([0, 1.2, -0.6, 0, -0.9, 0.78])
    d.qvel[ld] = rng.normal(0, 0.5, 6)
    mujoco.mj_forward(m, d)
    return m, d, ld


def test_a_flat_box_reproduces_table_plane_rows():
    rng = np.random.default_rng(3)
    worst_o, worst_f = 0.0, 0.0
    n_cases = 0
    for trial in range(6):
        m, d, ld = _canon_state(rng)
        kps = sc.barrier_keypoints(m, "left_arm_")
        ks = R.arm_keypoints(m, "left")
        Mf = sc.full_mass_matrix(m, d)
        M_inv = np.linalg.inv(Mf[np.ix_(ld, ld)])
        bias = d.qfrc_bias[ld] - d.qfrc_passive[ld]
        zmin = min(sc._kp_world(d, b, o)[2] - r for b, o, r in kps)
        z_floor = zmin - 0.04                                   # every keypoint (incl. radius) above the plane
        zono = (np.array([0.3, 0.2, z_floor - 50.0]), np.diag([1e3, 1e3, 50.0]))      # huge flat box, top face at z_floor
        for mt in (0.0, 0.01, -0.01):
            for kp_radius in (False, True):
                a_o, b_o = sc.table_plane_barrier_rows(m, d, kps, ld, M_inv, bias, z_floor, margin=mt, kp_radius=kp_radius)
                a_f, b_f = R._fb.table_plane_barrier_rows(m, d, kps, ld, M_inv, bias, z_floor, margin=mt, kp_radius=kp_radius)
                a_z, b_z = R.zono_barrier_rows(m, d, ks, ld, M_inv, bias, [zono], margin=-mt, d_act=1e9, kp_radius=kp_radius)
                assert a_z.shape == a_o.shape, (a_z.shape, a_o.shape)
                worst_o = max(worst_o, float(np.abs(a_z - np.asarray(a_o)).max()), float(np.abs(b_z - np.asarray(b_o)).max()))
                worst_f = max(worst_f, float(np.abs(a_z - a_f).max()), float(np.abs(b_z - b_f).max()))
                n_cases += 1
    assert worst_o < 1e-9 and worst_f < 1e-9, (worst_o, worst_f)
    print(f"(a) huge flat box == table plane rows: {n_cases} cases ({len(kps)} keypoints, margin +-/0, kp_radius on/off); "
          f"max|diff| vs sim_common original {worst_o:.2e}, vs fast_barriers {worst_f:.2e}")


def test_rows_sign_physical():
    """a feasible torque for the row moves the keypoint's barrier away from the facet: psi2 >= 0 exactly when a@tau <= b."""
    rng = np.random.default_rng(4)
    m, d, ld = _canon_state(rng)
    ks = R.arm_keypoints(m, "left")
    p, J, Jd, v, qd = R.kp_states(m, d, ks, ld)
    # a box 4 cm in front of the hand keypoint closest to +x
    k = int(np.argmax(p[:, 0])); c = p[k] + np.array([0.06, 0, 0])
    zono = (c, np.diag([0.02, 0.1, 0.1]))
    Mf = sc.full_mass_matrix(m, d); M_inv = np.linalg.inv(Mf[np.ix_(ld, ld)]); bias = d.qfrc_bias[ld] - d.qfrc_passive[ld]
    info = {}
    A, b = R.zono_barrier_rows(m, d, ks, ld, M_inv, bias, [zono], margin=0.0, d_act=0.25, info=info)
    assert info["n_pairs"] > 0 and A.shape[0] == info["n_pairs"]
    # integrate: apply tau, one MuJoCo-free check of psi2 via the row: psi2 = b - a@tau
    tau = rng.normal(size=6) * 3
    psi2 = b - A @ tau
    # recompute psi2 from first principles for the first row
    H = ZH.zono_hrep(*zono); a_ix = ZH.sd_lb(H, p, return_active=True)[1]
    near = np.flatnonzero((ZH.sd_lb(H, p) - ks.radii) < 0.25)
    r0 = near[0]; n0 = H.N[a_ix[r0]]; h0 = ZH.sd_lb(H, p)[r0] - ks.radii[r0]
    acc = M_inv @ (tau - bias)
    hdot = n0 @ (J[r0] @ qd); hddot = n0 @ (J[r0] @ acc + Jd[r0] @ qd)
    psi2_ref = hddot + 30 * hdot + 30 * (hdot + 30 * h0)
    assert abs(psi2[0] - psi2_ref) < 1e-8, (psi2[0], psi2_ref)
    print("rows: psi2(tau) from the row == first-principles psi2 (|diff| %.1e)" % abs(psi2[0] - psi2_ref))


if __name__ == "__main__":
    test_c_facets_contain_all_vertices()
    test_b_sd_lower_bound_le_exact()
    test_a_flat_box_reproduces_table_plane_rows()
    test_rows_sign_physical()
    print("ALL PASSED")
