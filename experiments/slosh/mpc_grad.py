#!/usr/bin/env python
"""mpc_grad.py -- is the fitted slosh model usable as an FL-MPC CONSTRAINT row?
SIM DATA ONLY. Measures, on a real rig trajectory window:
  1. (b3) world-frame LTI filter: analytic Jacobian d phi_k / d a_j over an MPC
     horizon (N=24 steps x 5 ms = FL-MPC's NC*L), checked vs finite
     differences, timed; first-order (linearized-constraint) accuracy under
     realistic accel perturbations.
  2. (b1) nonlinear pendulum and (c) pendulum+residual: torch autograd
     Jacobian over the same horizon, timed (CPU, 1 thread).
  3. per-step causal update cost (observer mode) of b1/b3 (numpy, 1 episode).

  python experiments/slosh/mpc_grad.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import slosh_models as sm  # noqa: E402
from data_prep import load_rig  # noqa: E402

DT, G = sm.DT, sm.G
GV = sm.GVEC
N = 24


def lti_mats(w, zeta, dt=DT):
    """semi-implicit Euler of x''=w^2(e-x)-c x' as z'=A z + B e (per axis; block-
    scalar => the 3-D system is kron(., I3))."""
    c = 2 * zeta * w
    A = np.array([[1 - dt * dt * w * w, dt * (1 - c * dt)], [-dt * w * w, 1 - c * dt]])
    B = np.array([dt * dt * w * w, dt * w * w])
    return A, B


def b3_rollout(z0x, z0v, a, n, w, zeta):
    """a (N,3) planned cup accel, n (N,3) planned cup axis -> phi (N,), X (N,3)."""
    c = 2 * zeta * w
    x, xd = z0x.copy(), z0v.copy(); X = np.empty((len(a), 3))
    for k in range(len(a)):
        f = a[k] - GV; e = f / np.linalg.norm(f)
        xd = xd + DT * (w * w * (e - x) - c * xd); x = x + DT * xd; X[k] = x
    cs = np.sum(X * n, 1) / np.linalg.norm(X, axis=1)
    return np.arccos(np.clip(cs, -1, 1)), X


def b3_jacobian(z0x, z0v, a, n, w, zeta, markov=None):
    """Analytic d phi_k / d a_j  -> (N, N, 3), lower-triangular in (k, j).
    x_k = (free response) + sum_{j<=k} m_{k-j} e_j,  m_d = [A^d B]_x (scalar)."""
    Nn = len(a)
    if markov is None:
        A, B = lti_mats(w, zeta); m = np.empty(Nn); v = B.copy()
        for d in range(Nn):
            m[d] = v[0]; v = A @ v
    else:
        m = markov
    phi, X = b3_rollout(z0x, z0v, a, n, w, zeta)
    f = a - GV; fn = np.linalg.norm(f, axis=1); e = f / fn[:, None]
    De = (np.eye(3)[None] - e[:, :, None] * e[:, None, :]) / fn[:, None, None]    # (N,3,3) de/da
    xn = np.linalg.norm(X, axis=1); cs = np.sum(X * n, 1) / xn
    sn = np.sqrt(np.clip(1 - cs * cs, 1e-12, None))
    dphidx = -(n / xn[:, None] - (cs / xn)[:, None] * X / xn[:, None]) / sn[:, None]   # (N,3)
    kk, jj = np.meshgrid(np.arange(Nn), np.arange(Nn), indexing="ij")
    Mt = np.where(jj <= kk, m[np.clip(kk - jj, 0, Nn - 1)], 0.0)                     # Toeplitz m_{k-j}
    J = Mt[:, :, None] * np.einsum("ki,jil->kjl", dphidx, De)
    return phi, J, m


def main():
    D = load_rig(str(HERE / "data" / "rig_v1.pkl"))
    P = json.loads(str(np.load(HERE / "data" / "scores_phys.npz")["params"]))
    p3 = P["b3_world_lti"]; w, z = p3["w"], p3["zeta"]
    print(f"b3 params: w={w} zeta={z} kappa={p3['kappa']}")
    # nominal window: a test 'mixed' spill episode, horizon ending ~at onset
    i = [j for j in np.where(D["test"] & (D["onset"] >= 0) & (D["family"] == "mixed"))[0]][0]
    K = sm.kinematics(D["pos"][i:i + 1], D["n"][i:i + 1])
    on = D["onset"][i]; k1 = on - N
    phi_full, Xf = sm.lti_world(K, w, z, return_state=True)
    # observer state at k1-1 (x from the causal run; x' by backward diff of x)
    x0 = Xf[0, k1 - 1]; xd0 = (Xf[0, k1 - 1] - Xf[0, k1 - 2]) / DT
    a = K["a"][0, k1:k1 + N]; n = K["n"][0, k1:k1 + N]
    phi, J, m = b3_jacobian(x0, xd0, a, n, w, z)
    print(f"episode {i} onset step {on}: rollout phi over horizon (deg) {np.degrees(phi[[0, 8, 16, 23]]).round(1)}"
          f"  vs causal-run phi {np.degrees(phi_full[0, k1:k1 + N][[0, 8, 16, 23]]).round(1)}")
    # finite-difference check
    eps = 1e-5; Jfd = np.zeros_like(J)
    for j in range(N):
        for d in range(3):
            ap = a.copy(); ap[j, d] += eps; am = a.copy(); am[j, d] -= eps
            Jfd[:, j, d] = (b3_rollout(x0, xd0, ap, n, w, z)[0] - b3_rollout(x0, xd0, am, n, w, z)[0]) / (2 * eps)
    print(f"analytic vs FD Jacobian: max|diff| = {np.abs(J - Jfd).max():.2e}  (max|J| = {np.abs(J).max():.2e} rad per m/s^2)")
    # timing (markov params precomputed once per (w,zeta), as they would be in the controller)
    reps = 2000; t0 = time.perf_counter()
    for _ in range(reps):
        b3_jacobian(x0, xd0, a, n, w, z, markov=m)
    t_b3 = (time.perf_counter() - t0) / reps * 1e3
    print(f"b3 rollout + full (N x N x 3) Jacobian, numpy, 1 thread: {t_b3:.3f} ms per MPC solve")
    # linearization accuracy: perturb planned accel by realistic deltas
    rng = np.random.default_rng(0)
    for sig in (1.0, 3.0, 10.0):
        errs = []
        for _ in range(200):
            da = rng.normal(scale=sig, size=a.shape)
            true = b3_rollout(x0, xd0, a + da, n, w, z)[0]
            lin = phi + np.einsum("kjd,jd->k", J, da)
            errs.append(np.degrees(np.abs(true - lin)).max())
        print(f"  linearized-constraint error, accel perturbation sigma={sig:4.1f} m/s^2/step: "
              f"max over horizon median {np.median(errs):.2f} deg, p95 {np.percentile(errs, 95):.2f} deg")
    # --- torch autograd for b1 and c over the horizon
    import torch
    torch.set_num_threads(1)
    from train_nets import PendulumResidual
    p1 = P["b1_world_pendulum"]
    for name, mdl in (("b1 pendulum (torch)", PendulumResidual(p1["w"], p1["zeta"])),
                      ("c pendulum+residual (torch)", None)):
        if mdl is None:
            pt = HERE / "data" / "c_pendulum_residual.pt"
            if not pt.exists():
                print("  (c model not trained yet; skip)"); continue
            mdl = PendulumResidual(p1["w"], p1["zeta"]); mdl.load_state_dict(torch.load(pt))
        at = torch.tensor(a, dtype=torch.float64, requires_grad=True)
        nt = torch.tensor(n, dtype=torch.float64)[None]
        mdl = mdl.double()
        xc = torch.zeros(1, N, 12, dtype=torch.float64)

        def f_of(at_):
            f = (at_ - torch.tensor(GV))[None]
            if name.startswith("b1"):
                return mdl.pendulum(f, nt)[0]
            return mdl(f, nt, xc)[0]
        t0 = time.perf_counter(); reps = 20
        for _ in range(reps):
            Jt = torch.autograd.functional.jacobian(f_of, at, vectorize=True)
        dtm = (time.perf_counter() - t0) / reps * 1e3
        print(f"{name}: N={N} horizon Jacobian (N x N x 3) via autograd: {dtm:.2f} ms  "
              f"(nonzero: {bool(Jt.abs().sum() > 0)})")
    # per-step observer cost
    Kone = sm.kinematics(D["pos"][i:i + 1], D["n"][i:i + 1])
    for nm, fn, pp in (("b1", sm.pendulum_world, p1), ("b3", sm.lti_world, p3)):
        t0 = time.perf_counter(); fn(Kone, pp["w"], pp["zeta"]); dtm = (time.perf_counter() - t0) / D["T"] * 1e6
        print(f"{nm} causal observer update (numpy, unoptimized, 1 episode): {dtm:.1f} us/step")


if __name__ == "__main__":
    main()
