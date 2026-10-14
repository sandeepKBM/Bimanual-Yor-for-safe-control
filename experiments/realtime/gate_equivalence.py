"""STRICT EQUIVALENCE GATE for experiments/realtime/fast_barriers.py. NEW file,
no edits to shared code. SIMULATION ONLY.

--mode component : run the real seed-0 `flmpc` episode driven by the ORIGINAL
    functions; at EVERY call site (all_pair_barriers, table_plane_barrier_rows,
    min_pair_distance, right_arm_to_cup_clearance, FLMPC top-K gather) also
    evaluate the fast version on the very same MjData and compare every output
    field (exact equality count + max |abs diff|). Also snapshots the full
    physics state at every all_pair_barriers call to <scratch>/gate_states.npz
    and (--replay) re-checks the fast vs original functions offline on those
    saved states (restore -> mj_forward -> compare).
--mode e2e : run the seed-0 `flmpc` episode twice in-process-independent
    fashion (original, then fast.install()) and compare every logged series
    bit-for-bit + the summary metrics vs the known baseline.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import mujoco

import _demo_env as env
import fast_barriers as fb
import fast_qp as fq

demo, sc, mpc_fl = env.demo, env.sc, env.mpc_fl

ORIG = dict(fb._ORIG)
ORIG["right_arm_to_cup_clearance"] = demo.right_arm_to_cup_clearance


class Stat:
    def __init__(self):
        self.calls = 0
        self.exact_calls = 0
        self.max_abs = 0.0
        self.notes = set()

    def add(self, exact, dev, note=None):
        self.calls += 1
        self.exact_calls += int(exact)
        self.max_abs = max(self.max_abs, float(dev))
        if note:
            self.notes.add(note)

    def __repr__(self):
        return (f"calls={self.calls} bit-identical={self.exact_calls}/{self.calls} "
                f"max|diff|={self.max_abs:.3e}" + (f" notes={sorted(self.notes)}" if self.notes else ""))


STATS = {k: Stat() for k in ("all_pair_barriers", "table_plane_barrier_rows", "min_pair_distance",
                             "right_arm_to_cup_clearance", "flmpc_topk", "solve_constrained_box_qp")}


def _cmp_val(a, b):
    """(exact, maxabs) for scalars/arrays/ints."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            return False, np.inf
        ex = bool(np.array_equal(a, b))
        return ex, (0.0 if ex else float(np.max(np.abs(a - b))))
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b), (0.0 if int(a) == int(b) else np.inf)
    ex = (a == b)
    return bool(ex), (0.0 if ex else abs(float(a) - float(b)))


def cmp_pairs(po, pf):
    if len(po) != len(pf):
        return False, np.inf, "len"
    ex_all, dev = True, 0.0
    for a, b in zip(po, pf):
        if set(a) != set(b):
            return False, np.inf, "keys"
        for k in a:
            ex, d = _cmp_val(a[k], b[k])
            ex_all &= ex
            dev = max(dev, d)
            if type(a[k]) is not type(b[k]) and not (isinstance(a[k], np.ndarray) and isinstance(b[k], np.ndarray)):
                return ex_all, dev, f"type:{k}:{type(a[k]).__name__}->{type(b[k]).__name__}"
            # memory layout matters for downstream bit-identity (numpy picks the
            # cblas variant from the strides) -> require identical strides too.
            if isinstance(a[k], np.ndarray) and a[k].strides != b[k].strides:
                return False, dev, f"strides:{k}:{a[k].strides}->{b[k].strides}"
    return ex_all, dev, None


def cmp_tuple(to, tf):
    ex_all, dev = True, 0.0
    for a, b in zip(to, tf):
        ex, d = _cmp_val(a, b)
        ex_all &= ex
        dev = max(dev, d)
    return ex_all, dev


# ---------------------------------------------------------------------------
_STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
_saved_states = []
_saved_args = []


def install_checkers(save_states=True):
    def apb(model, data, ego_dofadr, int_dofadr, ego_prefix=sc.EGO_PREFIX, int_prefix=sc.INT_PREFIX,
            extra_ego_keypoints=None):
        po = ORIG["all_pair_barriers"](model, data, ego_dofadr, int_dofadr, ego_prefix, int_prefix,
                                       extra_ego_keypoints)
        pf = fb.all_pair_barriers(model, data, ego_dofadr, int_dofadr, ego_prefix, int_prefix,
                                  extra_ego_keypoints)
        ex, dev, note = cmp_pairs(po, list(pf))
        STATS["all_pair_barriers"].add(ex, dev, note)
        # FLMPC top-K gather: fast (from arrays, fresh PairList) vs original sorted()[:K]
        pf2 = fb.all_pair_barriers(model, data, ego_dofadr, int_dofadr, ego_prefix, int_prefix,
                                   extra_ego_keypoints)
        tk_f = fb.topk_pairs(pf2, 4)
        tk_o = sorted(po, key=lambda p: p["h"])[:4]
        ex2, dev2, note2 = cmp_pairs(tk_o, tk_f)
        STATS["flmpc_topk"].add(ex2, dev2, note2)
        if save_states:
            st = np.empty(mujoco.mj_stateSize(model, _STATE_SPEC))
            mujoco.mj_getState(model, data, st, _STATE_SPEC)
            _saved_states.append(st)
        return po

    def tpr(model, data, kps, dofadr, M_inv, bias, z_floor, margin=0.0, alpha1=sc.ALPHA1, alpha2=sc.ALPHA2):
        ro = ORIG["table_plane_barrier_rows"](model, data, kps, dofadr, M_inv, bias, z_floor, margin, alpha1, alpha2)
        rf = fb.table_plane_barrier_rows(model, data, kps, dofadr, M_inv, bias, z_floor, margin, alpha1, alpha2)
        ex, dev = cmp_tuple(ro, rf)
        STATS["table_plane_barrier_rows"].add(ex, dev)
        return ro

    def mpd(model, data, ego_prefix=sc.EGO_PREFIX, int_prefix=sc.INT_PREFIX):
        ro = ORIG["min_pair_distance"](model, data, ego_prefix, int_prefix)
        rf = fb.min_pair_distance(model, data, ego_prefix, int_prefix)
        ex, dev = cmp_tuple(ro, rf)
        STATS["min_pair_distance"].add(ex, dev)
        return ro

    def rcc(model, data, cup_kps, right_kps, left_ee_body):
        ro = ORIG["right_arm_to_cup_clearance"](model, data, cup_kps, right_kps, left_ee_body)
        rf = fb.right_arm_to_cup_clearance(model, data, cup_kps, right_kps, left_ee_body)
        ex, dev = _cmp_val(ro, rf)
        STATS["right_arm_to_cup_clearance"].add(ex, dev)
        return ro

    qp_orig = sc.solve_constrained_box_qp
    fq.warmup()

    def qp(H, f, lo, hi, a=None, b=None, **kw):
        ro = qp_orig(H, f, lo, hi, a, b, **kw)
        rf = fq.solve_constrained_box_qp(H, f, lo, hi, a, b, **kw)
        ex, dev = cmp_tuple(ro[:2], rf[:2])
        ex = ex and (ro[2] == rf[2])
        STATS["solve_constrained_box_qp"].add(ex, dev if ro[2] == rf[2] else np.inf)
        return ro

    sc.solve_constrained_box_qp = qp
    sc.all_pair_barriers = apb
    sc.table_plane_barrier_rows = tpr
    sc.min_pair_distance = mpd
    demo.right_arm_to_cup_clearance = rcc


def replay(info, states, n_max=None):
    """Offline: restore each saved state, mj_forward, compare orig vs fast."""
    model = info["model"]
    data = mujoco.MjData(model)
    st_rep = {k: Stat() for k in ("all_pair_barriers", "table_plane_barrier_rows", "min_pair_distance")}
    ld, rd = info["left_dofadr"], info["right_dofadr"]
    cup = info["cup_barrier_keypoints"]
    lk = sc.barrier_keypoints(model, sc.EGO_PREFIX)
    rk = sc.barrier_keypoints(model, sc.INT_PREFIX)
    rng = np.random.default_rng(0)
    for s in states[:n_max]:
        mujoco.mj_setState(model, data, s, _STATE_SPEC)
        mujoco.mj_forward(model, data)
        po = ORIG["all_pair_barriers"](model, data, ld, rd, extra_ego_keypoints=cup)
        pf = fb.all_pair_barriers(model, data, ld, rd, extra_ego_keypoints=cup)
        ex, dev, note = cmp_pairs(po, list(pf))
        st_rep["all_pair_barriers"].add(ex, dev, note)
        Minv = np.linalg.inv(rng.standard_normal((6, 6)) + 6 * np.eye(6))
        bias = rng.standard_normal(6)
        for kps, dof in ((lk, ld), (rk, rd)):
            ro = ORIG["table_plane_barrier_rows"](model, data, kps, dof, Minv, bias, 0.75, 0.01)
            rf = fb.table_plane_barrier_rows(model, data, kps, dof, Minv, bias, 0.75, 0.01)
            st_rep["table_plane_barrier_rows"].add(*cmp_tuple(ro, rf))
        st_rep["min_pair_distance"].add(*cmp_tuple(ORIG["min_pair_distance"](model, data),
                                                     fb.min_pair_distance(model, data)))
    return st_rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("component", "e2e"), default="component")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--only", choices=("orig", "fast", "both"), default="both")
    args = ap.parse_args()

    if args.mode == "component":
        install_checkers()
        info = env.demo.build_episode(seed=0)
        t0 = time.perf_counter()
        log, _ = demo.run_episode(info, controller="flmpc", capture=False, max_steps=args.max_steps)
        summ = demo.summarize(log)
        print(f"[component] episode {len(log['t'])} steps in {time.perf_counter()-t0:.1f}s; "
              f"collision={summ['collision']} placed_final={summ['placed_final']} h_min={summ['h_min']!r}")
        for k, v in STATS.items():
            print(f"  {k:28s} {v}")
        states = np.stack(_saved_states)
        np.savez_compressed(env.SCRATCH / "gate_states.npz", states=states)
        print(f"  saved {len(states)} states -> {env.SCRATCH / 'gate_states.npz'}")
        if not args.no_replay:
            info2 = env.demo.build_episode(seed=0)
            rep = replay(info2, states)
            print("[replay on saved states, fresh model]")
            for k, v in rep.items():
                print(f"  {k:28s} {v}")
    else:
        res = {}
        labels = ("orig", "fast") if args.only == "both" else (args.only,)
        if args.only == "fast":
            # reference = the ORIGINAL-path log saved by an earlier plain run of the
            # same demo version (<scratch>/base_log.npz, see report)
            z = np.load(env.SCRATCH / "base_log.npz")
            lo_ref = {k: z[k] for k in z.files}
            if args.max_steps is not None:
                lo_ref = {k: (v[:args.max_steps] if v.ndim >= 1 and len(v) > 1 else v) for k, v in lo_ref.items()}
            res["orig"] = (lo_ref, demo.summarize(lo_ref), float("nan"))
        for label in labels:
            if label == "fast":
                fb.install(demo)
                fq.install()
            info = env.demo.build_episode(seed=0)
            t0 = time.perf_counter()
            log, _ = demo.run_episode(info, controller="flmpc", capture=False, max_steps=args.max_steps)
            wall = time.perf_counter() - t0
            res[label] = (log, demo.summarize(log), wall)
            if label == "fast":
                fb.uninstall()
                fq.uninstall()
            print(f"[{label}] {len(log['t'])} steps, wall {wall:.1f}s ({wall/len(log['t'])*1e3:.2f} ms/step)")
        lo, so, _ = res["orig"]
        if "fast" not in res:
            return
        lf, sf, _ = res["fast"]
        n_diff = 0
        worst = 0.0
        for k in lo:
            a, b = np.asarray(lo[k], float), np.asarray(lf[k], float)
            if k == "solve_time":
                continue   # wall-clock, not a sim output
            if not np.array_equal(a, b, equal_nan=True):
                n_diff += 1
                d = float(np.nanmax(np.abs(a - b)))
                worst = max(worst, d)
                print(f"  DIFF {k}: max|diff|={d:.3e}")
        print(f"[e2e] logged series differing (excl. solve_time): {n_diff}; worst |diff| {worst:.3e}")
        for k in so:
            if k.startswith("solve_time"):
                continue
            if so[k] != sf[k] and not (isinstance(so[k], float) and np.isnan(so[k]) and np.isnan(sf[k])):
                print(f"  SUMMARY DIFF {k}: {so[k]!r} vs {sf[k]!r}")
        b = env.BASELINE_SEED0_FLMPC
        for lab, s in (("orig", so), ("fast", sf)):
            ok = all(s[k] == v for k, v in b.items())
            print(f"  {lab}: collision={s['collision']} placed_final={s['placed_final']} h_min={s['h_min']!r} "
                  f"-> matches known baseline: {ok}")
        json.dump(dict(orig=so, fast=sf, wall_orig=res["orig"][2], wall_fast=res["fast"][2]),
                  open(env.SCRATCH / "gate_e2e.json", "w"), indent=1)


if __name__ == "__main__":
    main()
