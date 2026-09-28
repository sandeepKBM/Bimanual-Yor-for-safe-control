"""Replay the FLAGSHIP left-EE (cup) motion (lift + crossing, logged in Part A under the faked grasp)
on the REAL friction cup grasp, at time-compression factors s (accel x s^2), and measure slip/drop.
SIMULATION ONLY.  usage: python c_replay.py A_seedN.npz VARIANT OUT.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import mujoco

import c_grasp_stress as C
from common import sc, quat_mul, quat_conj, rel_pose, jaw_geoms, contact_stats


def main():
    npz, vs, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.load(npz)
    dt = float(z["dt"])
    P, Q = z["l_pos"], z["l_quat"]
    v = C.parse_variant(vs)
    model, data, side, ob, fingers, grip = C.setup_cup(v)
    jaw7, jaw8 = jaw_geoms(model, side)
    C.TRUE_RANGE.clear(); C.TRUE_RANGE.update({j: model.jnt_range[j].copy() for j in range(model.njnt)})
    C.zero_arm_fricloss(model, side)
    sc.disable_left_arm_position_servos(model)
    for jn in sc.LEFT_ARM_JOINTS:
        model.jnt_range[model.joint(jn).id] = (-6.5, 6.5)
    snap = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, snap, mujoco.mjtState.mjSTATE_INTEGRATION)
    n = len(P)
    tgrid = np.arange(n) * dt
    # relative trajectory in the flagship: p(t)-p(0), q(t) q(0)^-1 (world)
    dP = P - P[0]
    dQ = np.array([quat_mul(Q[i], quat_conj(Q[0])) for i in range(n)])
    V = np.gradient(dP, dt, axis=0); A = np.gradient(V, dt, axis=0)
    W = z["l_w"]; AL = np.gradient(W, dt, axis=0)
    res = []
    import os
    for s in [float(x) for x in os.environ.get('SPEEDUPS', '1,2,3,4,6').split(',')]:
        T = tgrid[-1] / s

        def prof(t, p0, q0, c0, pinch, s=s):
            tt = min(t * s, tgrid[-1])
            i = min(int(tt / dt), n - 1)
            return (p0 + dP[i], quat_mul(dQ[i], q0), V[i] * s, W[i] * s, A[i] * s * s, AL[i] * s * s)
        r = C.run_test(model, data, side, ob, fingers, jaw7, jaw8, snap, prof, T + 0.3, ob)
        r.update(speedup=s, demand_a_max=float(np.max(np.linalg.norm(A, axis=1)) * s * s),
                 demand_al_max=float(np.max(np.linalg.norm(AL, axis=1)) * s * s))
        res.append(r)
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}), flush=True)
    json.dump(dict(npz=npz, variant=v, results=res), open(outp, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
