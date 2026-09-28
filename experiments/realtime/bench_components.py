"""Micro-benchmark: original vs fast per-component wall time on states saved by
gate_equivalence.py (<scratch>/gate_states.npz). NEW file. SIMULATION ONLY.
Each state: restore -> mj_forward -> time each function (orig, fast)."""
from __future__ import annotations

import time

import numpy as np
import mujoco

import _demo_env as env
import fast_barriers as fb

demo, sc = env.demo, env.sc


def main(n_max=600):
    info = demo.build_episode(seed=0)
    model = info["model"]
    data = mujoco.MjData(model)
    S = np.load(env.SCRATCH / "gate_states.npz")["states"]
    S = S[np.linspace(0, len(S) - 1, min(n_max, len(S))).astype(int)]
    ld, rd = info["left_dofadr"], info["right_dofadr"]
    cup = info["cup_barrier_keypoints"]
    lk = sc.barrier_keypoints(model, sc.EGO_PREFIX)
    rk = list(sc.barrier_keypoints(model, sc.INT_PREFIX))
    rk_tab = rk + [(model.body("right_arm_gripper_base").id, np.zeros(3), 0.01)] * 3
    Minv = np.eye(6); bias = np.zeros(6)
    O = fb._ORIG
    cases = {
        "all_pair_barriers (45x26, +cup)": (
            lambda: O["all_pair_barriers"](model, data, ld, rd, extra_ego_keypoints=cup),
            lambda: fb.all_pair_barriers(model, data, ld, rd, extra_ego_keypoints=cup)),
        "  ...+ FLMPC top-4 gather": (
            lambda: sorted(O["all_pair_barriers"](model, data, ld, rd, extra_ego_keypoints=cup),
                           key=lambda p: p["h"])[:4],
            lambda: fb.topk_pairs(fb.all_pair_barriers(model, data, ld, rd, extra_ego_keypoints=cup), 4)),
        "  ...+ materialize all dicts (v4c consumers)": (
            lambda: O["all_pair_barriers"](model, data, ld, rd, extra_ego_keypoints=cup),
            lambda: fb.all_pair_barriers(model, data, ld, rd, extra_ego_keypoints=cup).as_list()),
        "table_plane_barrier_rows x2 (26 + 29 kps)": (
            lambda: (O["table_plane_barrier_rows"](model, data, lk, ld, Minv, bias, 0.75, 0.01),
                     O["table_plane_barrier_rows"](model, data, rk_tab, rd, Minv, bias, 0.75, 0.01)),
            lambda: (fb.table_plane_barrier_rows(model, data, lk, ld, Minv, bias, 0.75, 0.01),
                     fb.table_plane_barrier_rows(model, data, rk_tab, rd, Minv, bias, 0.75, 0.01))),
        "min_pair_distance (log)": (
            lambda: O["min_pair_distance"](model, data), lambda: fb.min_pair_distance(model, data)),
        "right_arm_to_cup_clearance (log)": (
            lambda: demo.right_arm_to_cup_clearance(model, data, cup, rk, info["left_ee_body"]),
            lambda: fb.right_arm_to_cup_clearance(model, data, cup, rk, info["left_ee_body"])),
    }
    T = {k: [[], []] for k in cases}
    for s in S:
        mujoco.mj_setState(model, data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        for k, (fo, ff) in cases.items():
            for j, fn in enumerate((fo, ff)):
                t0 = time.perf_counter(); fn(); T[k][j].append(time.perf_counter() - t0)
    print(f"{len(S)} saved episode states; median per call (ms)")
    print(f"{'component':48s} {'orig':>8s} {'fast':>8s} {'speedup':>8s}")
    for k, (a, b) in T.items():
        a, b = np.median(a) * 1e3, np.median(b) * 1e3
        print(f"{k:48s} {a:8.3f} {b:8.3f} {a/b:7.1f}x")


if __name__ == "__main__":
    main()
