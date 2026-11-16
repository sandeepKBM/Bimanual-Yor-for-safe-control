#!/usr/bin/env python
"""experiments/residual_dynamics/collect_residual_data.py

Generates the residual-dynamics dataset described in
experiments/residual_dynamics/perturbed_plant.py's module docstring.

Per sample:
  1. Sample a collision-free (q, qdot) for both arms (uniform in joint
     range / ~N(0, qd_std^2) clipped) -- rejecting arm-involved-contact
     configs (perturbed_plant.sample_collision_free_state), since a
     contact/constraint force would corrupt the residual with something
     unrelated to the nominal-vs-perturbed DYNAMICS mismatch this dataset
     targets. Set on the NOMINAL model.
  2. Sample a commanded accel a (12-D) ~ N(0, A_STD^2), rescaled per-arm so
     |M_nom @ a + c_nom| <= TAU_MAX (component-wise) -- if a raw sample
     would exceed TAU_MAX on either arm, its `a` is rescaled down (not
     resampled) so its DIRECTION is kept.
  3. tau = M_nom @ a + c_nom per arm (perturbed_plant.fl_torque, mpc_fl.py's
     exact FL convention).
  4. Apply tau to the PERTURBED plant at the SAME (q, qdot) via
     qfrc_applied + mj_forward (instantaneous qacc, no integration/
     finite-difference error). qacc_realized = data_real.qacc at the arm
     dofs.
  5. residual = qacc_realized - a.

Output (--out, .npz): state (N,24)=[q(12),qd(12)], a_cmd (N,12),
residual (N,12), qacc_realized (N,12).

Threads pinned (OMP/OPENBLAS/MKL=1) BEFORE numpy/mujoco import, since a
tight per-sample Python loop over many small dense linalg ops is faster
single-threaded (no thread-pool spin-up overhead) and this script is not
meant to be run multi-process-parallel-unsafe.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ab_smooth_vs_bangbang"))

import perturbed_plant as pp  # noqa: E402
import sim_common as sc  # noqa: E402

QD_STD = 1.0          # [rad/s] per-dof qdot sample std
QD_CLIP = 3.0          # [rad/s] per-dof qdot clip
A_STD = 3.0            # [rad/s^2] per-dof commanded-accel sample std ("a few rad/s^2")
MAX_CONTACT_TRIES = 200


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    # sanity check FIRST -- fail fast if FL/model wiring is broken.
    max_res = pp.sanity_check_nominal_residual(seed=args.seed)
    assert max_res < pp.SANITY_NOMINAL_RESIDUAL_THRESHOLD, \
        f"nominal-plant sanity check FAILED: max|residual|={max_res:.3e}"
    print(f"[collect] sanity OK: nominal-plant residual max|.|={max_res:.3e} "
          f"(< {pp.SANITY_NOMINAL_RESIDUAL_THRESHOLD})")

    m_nom = pp.load_nominal()
    m_real = pp.make_perturbed_model()
    d_nom = mujoco.MjData(m_nom)
    d_real = mujoco.MjData(m_real)
    key_nom = m_nom.key("home").id
    key_real = m_real.key("home").id

    left_qposadr, left_dofadr, right_qposadr, right_dofadr = pp.get_arm_addrs(m_nom)
    ranges = pp.arm_joint_ranges(m_nom)

    N = args.num_samples
    state = np.zeros((N, 24), dtype=np.float64)
    a_cmd = np.zeros((N, 12), dtype=np.float64)
    residual = np.zeros((N, 12), dtype=np.float64)
    qacc_realized = np.zeros((N, 12), dtype=np.float64)

    n_clipped = 0
    n_contact_retries = 0
    max_tries_seen = 1
    t0 = time.time()

    for i in range(N):
        q, qd, tries = pp.sample_collision_free_state(
            m_nom, d_nom, key_nom, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
            ranges, rng, qd_std=QD_STD, qd_clip=QD_CLIP, max_tries=MAX_CONTACT_TRIES)
        n_contact_retries += (tries - 1)
        max_tries_seen = max(max_tries_seen, tries)

        M_full = sc.full_mass_matrix(m_nom, d_nom)
        a = rng.normal(0.0, A_STD, size=12)

        tau_left = pp.fl_torque(M_full, d_nom, left_dofadr, a[:6])
        tau_right = pp.fl_torque(M_full, d_nom, right_dofadr, a[6:])

        # clip so |tau| <= TAU_MAX per arm, by rescaling `a`'s magnitude
        # (direction preserved) -- matches the task's "clipped so
        # |M_nom.a + c_nom| <= TAU_MAX" spec.
        clipped = False
        max_tau_left = np.max(np.abs(tau_left))
        if max_tau_left > sc.TAU_MAX:
            scale = sc.TAU_MAX / max_tau_left * 0.999
            a[:6] *= scale
            tau_left = pp.fl_torque(M_full, d_nom, left_dofadr, a[:6])
            clipped = True
        max_tau_right = np.max(np.abs(tau_right))
        if max_tau_right > sc.TAU_MAX:
            scale = sc.TAU_MAX / max_tau_right * 0.999
            a[6:] *= scale
            tau_right = pp.fl_torque(M_full, d_nom, right_dofadr, a[6:])
            clipped = True
        if clipped:
            n_clipped += 1

        # apply tau to the PERTURBED plant at the SAME (q, qd).
        mujoco.mj_resetDataKeyframe(m_real, d_real, key_real)
        d_real.qpos[left_qposadr] = q[:6]
        d_real.qpos[right_qposadr] = q[6:]
        d_real.qvel[left_dofadr] = qd[:6]
        d_real.qvel[right_dofadr] = qd[6:]
        d_real.qfrc_applied[:] = 0.0
        d_real.qfrc_applied[left_dofadr] = tau_left
        d_real.qfrc_applied[right_dofadr] = tau_right
        mujoco.mj_forward(m_real, d_real)

        qacc_r = np.concatenate([d_real.qacc[left_dofadr], d_real.qacc[right_dofadr]])

        state[i] = np.concatenate([q, qd])
        a_cmd[i] = a
        qacc_realized[i] = qacc_r
        residual[i] = qacc_r - a

        if (i + 1) % max(1, N // 10) == 0:
            print(f"[collect] {i + 1}/{N} samples "
                  f"(elapsed {time.time() - t0:.1f}s)")

    dt = time.time() - t0
    print(f"[collect] done: N={N}, elapsed={dt:.1f}s ({dt / max(N, 1) * 1000:.2f} ms/sample)")
    print(f"[collect] clip stats: {n_clipped}/{N} samples had `a` rescaled "
          f"to respect TAU_MAX ({100.0 * n_clipped / N:.1f}%)")
    print(f"[collect] contact-rejection stats: {n_contact_retries} extra resamples "
          f"across {N} samples (max tries for a single sample: {max_tries_seen})")
    print(f"[collect] residual norm stats: mean|residual|={np.abs(residual).mean():.4f}, "
          f"rms={np.sqrt(np.mean(residual ** 2)):.4f}, max|.|={np.abs(residual).max():.4f} rad/s^2")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, state=state, a_cmd=a_cmd, residual=residual,
             qacc_realized=qacc_realized, seed=args.seed)
    print(f"[collect] wrote {out_path} "
          f"(state{state.shape}, a_cmd{a_cmd.shape}, residual{residual.shape}, "
          f"qacc_realized{qacc_realized.shape})")


if __name__ == "__main__":
    main()
