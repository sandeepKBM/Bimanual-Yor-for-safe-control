"""G4 TIMING for rt_flmpc vs the current fast path. SIMULATION ONLY.

Run it PINNED to one core (taskset -c N) at nice 0; threads pinned to 1.

  --impl py : flagship episode (fast path). Per FL step the CONTROLLER time is
              full_mass_matrix + payload_reflected_terms + all_pair_barriers + 2x
              table_plane_barrier_rows + FLMPC.step (the scene mj_forward they read is
              timed separately). Then a fixed-state microbenchmark (N calls cycling over the
              logged states; scene state restored + mj_forward untimed before each call).
  --impl rt : same episode with --controller-impl rt; controller time = RTFLMPC.step_lean
              (lean sync+forward + ONE compiled call). Microbenchmark: step_lean on the
              logged states (robot state via `sensing`). Also: lean dynamics vs full-scene
              mj_forward on the same states, iteration histograms, and the DETERMINISTIC
              WORST CASE: synthetic problems with the real dimensions that drive the hard QP
              AND the relaxed QP to their iteration caps (then a=0 + torque scale), timed
              through rt_core (rows + both capped solves + backstop), max over trials.
Writes <out>/g4_<impl>.json (+ npz of per-step times).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import mujoco

import rt_harness as H
import fast_barriers as fb
import fast_qp as fq
import rt_kernels as K
import rt_qp

demo, sc, mpc_fl = H.demo, H.sc, H.mpc_fl
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS


def pct(v):
    v = np.asarray(v, float) * 1e3
    return dict(n=int(v.size), mean=float(v.mean()), p50=float(np.median(v)), p99=float(np.percentile(v, 99)),
                p999=float(np.percentile(v, 99.9)), max=float(v.max()))


def worst_case(fl, trials=200, seed=0):
    """DETERMINISTIC WORST CASE, measured: synthetic inputs with the REAL dimensions (same
    K, horizon, row blocks, table-row counts) through rt_core -- rows + capped hard QP +
    capped relaxed QP + torque backstop -- constructed so that the clearance / EE-speed rows
    are deeply violated in many directions (hard QP infeasible, relaxed QP needing many
    add/drop iterations). Row building is data-independent; the solve is bounded by
    cap x (max per-iteration cost). Reports the time distribution, how many trials hit each
    cap, and the max (the empirical worst case of this code path at these caps)."""
    rng = np.random.default_rng(seed)
    NV, NP, nb = fl.NV, fl.NP, fl.n_blk
    ws = fl._ws
    Kp = fl.K_PAIRS
    z6 = np.zeros(NV)
    times, statuses = [], []
    ntl = len(fl._lean_cfg["ltab"]) if fl._lean_cfg is not None else 30
    ntr = len(fl._lean_cfg["rtab"]) if fl._lean_cfg is not None else 30
    for tr in range(trials):
        sev = 0.02 + 0.3 * rng.random()
        q = rng.normal(0, 0.3, NV)
        qd = rng.normal(0, 2.0, NV)
        B = rng.normal(size=(NV, NV))
        M = B @ B.T * 0.05 + 0.1 * np.eye(NV)
        c = rng.normal(0, 5, NV)
        refs = [np.ascontiguousarray(rng.normal(0, 0.5, (NP + 1, NV))) for _ in range(4)]
        n_arr = rng.normal(size=(Kp, 3))
        n_arr /= np.linalg.norm(n_arr, axis=1, keepdims=True)
        Je = rng.normal(0, 0.5, (Kp, 3, NV))
        Ji = rng.normal(0, 0.5, (Kp, 3, NV))
        h0 = -np.abs(rng.normal(sev, 0.02, Kp))
        a_te = np.ascontiguousarray(rng.normal(0, 1, (ntl, NV)))
        b_te = np.abs(rng.normal(0, 50, ntl)) + 1.0          # hard table rows satisfiable
        a_ti = np.ascontiguousarray(rng.normal(0, 1, (ntr, NV)))
        b_ti = np.abs(rng.normal(0, 50, ntr)) + 1.0
        Jr = rng.normal(0, 0.3, (3, NV))
        Jp = rng.normal(0, 0.5, (3, NV))
        fl._set_params(True, 8, True, True, True, ntl, ntr, True, 0.0, fl.H_MARGIN, 0.5, 0.5, 0.02)
        J0 = fl._J0_for(0.0)
        tl = np.full(NV, sc.TAU_MAX)
        t0 = time.perf_counter()
        K.rt_core(fl._ci, fl._cf, fl._pi, fl._pf, fl._cq, fl._cqd, fl._tvec, fl._coll_steps, fl._tilt_steps,
                  fl._speed_steps, q, qd, -q, -qd, M, c, M, c, refs[0], refs[1], refs[2], refs[3], z6, z6,
                  n_arr, Je, Ji, h0, tl, tl, np.array([0.05, -0.05]), Jr, Jp, Jp, a_te, b_te, a_ti, b_ti, Jp,
                  J0, ws.f, ws.A, ws.b, ws.slack_rows, ws.J, ws.R, ws.act, ws.u, ws.y, ws.iact, ws.d, ws.z, ws.r,
                  ws.sidx, ws.g_aug, ws.lo, ws.hi, ws.stats, ws.oi, ws.xout, ws.a_e0, ws.a_i0, ws.tau_e, ws.tau_i,
                  ws.scr, ws.rn)
        times.append(time.perf_counter() - t0)
        statuses.append((int(ws.oi[K.OI_HARD_STATUS]), int(ws.oi[K.OI_HARD_IT]), int(ws.oi[K.OI_RELAX_STATUS]),
                         int(ws.oi[K.OI_RELAX_IT]), int(ws.oi[K.OI_M]), int(ws.oi[K.OI_NS])))
    S = np.array(statuses)
    T = np.array(times)
    hc = S[:, 0] == rt_qp.ST_ITERCAP
    rc = S[:, 2] == rt_qp.ST_ITERCAP
    return dict(trials=trials, caps=[fl.it_hard, fl.it_relax], m=int(S[0, 4]), ns=int(S[0, 5]),
                hard_status_counts={int(k): int(v) for k, v in zip(*np.unique(S[:, 0], return_counts=True))},
                relax_status_counts={int(k): int(v) for k, v in zip(*np.unique(S[:, 2], return_counts=True))},
                hard_it_max=int(S[:, 1].max()), relax_it_max=int(S[:, 3].max()),
                n_hard_cap=int(hc.sum()), n_relax_cap=int(rc.sum()),
                t_all_ms=pct(T), t_relax_cap_ms=(pct(T[rc]) if rc.any() else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=("py", "rt"), required=True)
    ap.add_argument("--profile", default="flagship")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--micro", type=int, default=10000)
    ap.add_argument("--micro-py", type=int, default=2000)
    ap.add_argument("--worst-trials", type=int, default=300)
    ap.add_argument("--out", default=H.RUN_DIR + "/g4")
    ap.add_argument("--gc", choices=("default", "disable"), default="default",
                    help="disable Python's cyclic GC for the measurement (a gen-2 collection of the big sim heap "
                         "is a multi-ms pause unrelated to the controller; deployment would gc.freeze()/disable)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.gc == "disable":
        import gc
        gc.collect()
        gc.disable()
    os.makedirs(args.out, exist_ok=True)
    fb.install(demo)
    fq.install()
    res = dict(impl=args.impl, profile=args.profile, seed=args.seed, gc=args.gc, cpu_affinity=sorted(os.sched_getaffinity(0)),
               nice=os.nice(0), host=os.uname().nodename)
    cur = {"t": 0.0}
    per_step = []
    saved = []
    fwd_cmp = []
    ctl = {}

    def timed(obj, attr):
        f = getattr(obj, attr)

        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                cur["t"] += time.perf_counter() - t0
        setattr(obj, attr, w)

    for attr in ("full_mass_matrix", "payload_reflected_terms", "all_pair_barriers", "table_plane_barrier_rows"):
        timed(sc, attr)
    mj_step0 = mujoco.mj_step

    def mj_step_w(m, d, *a):
        cur["t"] = 0.0
        return mj_step0(m, d, *a)
    mujoco.mj_step = mj_step_w

    if args.impl == "py":
        step0 = mpc_fl.FLMPC.step

        def step_w(self, model, data, *a, **k):
            if "scratch" not in ctl:
                ctl["scratch"] = mujoco.MjData(model)
                ctl["fl"], ctl["model"] = self, model
            helpers = cur["t"]
            st = np.empty(mujoco.mj_stateSize(model, STATE))
            mujoco.mj_getState(model, data, st, STATE)
            snap = (self.a_ego_prev.copy(), self.a_int_prev.copy(),
                    None if self._cup_axis_local is None else self._cup_axis_local.copy())
            t0 = time.perf_counter()
            out = step0(self, model, data, *a, **k)
            t_step = time.perf_counter() - t0
            per_step.append((helpers + t_step, t_step, helpers))
            saved.append((st, a, k, snap))
            return out
        mpc_fl.FLMPC.step = step_w
    else:
        import rt_flmpc
        sl0 = rt_flmpc.RTFLMPC.step_lean

        def sl_w(self, scene_data, *a, **k):
            if scene_data is None:           # configure-time JIT warmup / microbenchmark calls: not timed here
                return sl0(self, scene_data, *a, **k)
            self.record = True
            if "scratch" not in ctl:
                ctl["fl"] = self
                ctl["model"] = self._lean_cfg["scene_model"]
                ctl["scratch"] = mujoco.MjData(ctl["model"])
                self.record = True
            lean = self._lean_cfg["lean"]
            t0 = time.perf_counter()
            out = sl0(self, scene_data, *a, **k)
            t = time.perf_counter() - t0
            per_step.append((t, self.records[-1][7], self.records[-1][6]))
            saved.append((lean.data.qpos.copy(), lean.data.qvel.copy(), a, k,
                          (self.a_ego_prev.copy(), self.a_int_prev.copy()), int(self.records[-1][1])))
            if len(per_step) % 10 == 0:
                sd = ctl["scratch"]
                mujoco.mj_copyData(sd, ctl["model"], scene_data)
                t1 = time.perf_counter()
                mujoco.mj_forward(ctl["model"], sd)
                t2 = time.perf_counter()
                lean.sync_from_scene(scene_data)
                lean.forward()
                lean.full_M()
                t3 = time.perf_counter()
                fwd_cmp.append((t2 - t1, t3 - t2))
            return out
        rt_flmpc.RTFLMPC.step_lean = sl_w

    info = H.build(args.profile, args.seed)
    t0 = time.perf_counter()
    log, summ = H.run(info, args.profile, max_steps=args.max_steps, controller_impl=args.impl)
    res["episode_wall_s"] = time.perf_counter() - t0
    res["summary"] = dict(collision=summ["collision"], placed_final=summ["placed_final"], h_min=summ["h_min"])
    P = np.array(per_step)
    res["inloop_controller_ms"] = pct(P[:, 0])
    if args.impl == "py":
        res["inloop_flmpc_step_ms"] = pct(P[:, 1])
        res["inloop_helpers_ms"] = pct(P[:, 2])
    else:
        res["inloop_kernel_ms"] = pct(P[:, 1])
        res["inloop_lean_dyn_ms"] = pct(P[:, 2])
        F = np.array(fwd_cmp)
        res["scene_mj_forward_ms"] = pct(F[:, 0])
        res["lean_sync_forward_fullM_ms"] = pct(F[:, 1])
        fl = ctl["fl"]
        res["hard_it_hist"] = np.trim_zeros(fl.hard_it_hist, "b").tolist()
        res["relax_it_hist"] = np.trim_zeros(fl.relax_it_hist, "b").tolist()
        res["counters"] = {k: (np.asarray(v).tolist()) for k, v in fl.rt_log_fields().items()}
        res["jit_warmup_s"] = getattr(fl, "jit_warmup_s", None)
    np.savez_compressed(f"{args.out}/g4_{args.impl}{args.tag}_perstep.npz", per_step=P)
    # ---------------- fixed-state microbenchmark ----------------
    fl = ctl["fl"]
    model = ctl["model"]
    if args.impl == "rt":
        import rt_flmpc
        sl = rt_flmpc.RTFLMPC.step_lean = sl0
        fl.record = False
        N = args.micro
        T = np.empty(N)
        its = np.empty(N, np.int64)
        for i in range(N):
            qp_, qv_, a, k, (ae, ai), _ = saved[i % len(saved)]
            fl.a_ego_prev, fl.a_int_prev = ae, ai
            t0 = time.perf_counter()
            sl(fl, None, *a, sensing=(qp_, qv_), **k)
            T[i] = time.perf_counter() - t0
            its[i] = fl.last_oi[K.OI_HARD_IT]
        res["micro_step_lean_ms"] = pct(T)
        res["micro_hard_it"] = dict(mean=float(its.mean()), max=int(its.max()))
        # lean dynamics alone vs scene forward on the same states
        lean = fl._lean_cfg["lean"]
        sd = ctl["scratch"]
        TL = []
        for i in range(min(N, 3000)):
            qp_, qv_ = saved[i % len(saved)][:2]
            t0 = time.perf_counter()
            lean.set_state(qp_, qv_)
            lean.forward()
            lean.full_M()
            TL.append(time.perf_counter() - t0)
        res["micro_lean_dyn_ms"] = pct(TL)
        worst_case(fl, trials=10, seed=99)      # warm the Python-level rt_core entry (cache load), untimed
        res["worst_case"] = worst_case(fl, trials=args.worst_trials)
    else:
        mpc_fl.FLMPC.step = step0
        N = min(args.micro_py, len(saved) * 10)
        sd = ctl["scratch"]
        T, TS = np.empty(N), np.empty(N)
        info_m = info
        ld, rd = info_m["left_dofadr"], info_m["right_dofadr"]
        for i in range(N):
            st, a, k, (ae, ai, ax) = saved[i % len(saved)]
            mujoco.mj_setState(model, sd, st, STATE)
            mujoco.mj_forward(model, sd)
            fl.a_ego_prev, fl.a_int_prev, fl._cup_axis_local = ae.copy(), ai.copy(), (None if ax is None else ax.copy())
            t0 = time.perf_counter()
            # the demo's controller-side inputs for this state (same helper calls as its loop)
            Mf = sc.full_mass_matrix(model, sd)
            Mb, bb = sc.payload_reflected_terms(model, sd, info_m["left_ee_body"], info_m["cup_body"], ld)
            pairs = sc.all_pair_barriers(model, sd, ld, rd, extra_ego_keypoints=info_m["cup_barrier_keypoints"])
            tr = k.get("table_rows")
            t1 = time.perf_counter()
            fl.step(model, sd, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], pairs, **k)
            t2 = time.perf_counter()
            T[i] = (t2 - t0)
            TS[i] = (t2 - t1)
        res["micro_py_controller_ms_excl_table_rows"] = pct(T)
        res["micro_py_flmpc_step_ms"] = pct(TS)
    json.dump(res, open(f"{args.out}/g4_{args.impl}{args.tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
