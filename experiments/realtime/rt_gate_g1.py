"""G1 NUMERICS gate for rt_flmpc. SIMULATION ONLY.

Drives a real episode with the CURRENT fast path (fast_barriers + fast_qp, --controller-impl
py semantics) and, at EVERY FLMPC.step call, evaluates on the very same inputs:
  (a) rt DROP-IN   RTFLMPC.step (same scene inputs as FLMPC.step): H, f, A, b vs the py QP,
      QP solution (max |dx|, |da0|), active set, feasibility / fallback decisions, torque;
  (b) rt LEAN      RTFLMPC.step_lean (robot-only dynamics + in-kernel geometry): M/c,
      top-K pairs, table rows, rows and solution vs (a) and vs py;
  (c) every --stress-every steps, a STRESS pair (py FLMPC vs rt drop-in, H_MARGIN raised so
      the clearance rows go infeasible near contact): fallback decisions + relaxed solution.
Both rt controllers are re-synced to the py controller's state (a_prev, cup axis, delay
queue) before every call, i.e. open-loop comparisons along the py trajectory.
Writes <out>/g1_<profile>_s<seed>.json (+ .npz with per-step series).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import rt_harness as H
import fast_barriers as fb
import fast_qp as fq
import rt_flmpc
import rt_qp
import rt_kernels as K
from rt_dynamics import LeanRobotDynamics

demo, sc, mpc_fl = H.demo, H.sc, H.mpc_fl


def mad(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


class Acc:
    def __init__(self):
        self.d = {}

    def mx(self, k, v):
        self.d[k] = max(self.d.get(k, 0.0), float(v))

    def cnt(self, k, v=1):
        self.d[k] = self.d.get(k, 0) + int(v)

    def lst(self, k, v):
        self.d.setdefault(k, []).append(float(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="flagship")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stress-every", type=int, default=10)
    ap.add_argument("--stress-hmargin", type=float, default=0.06)
    ap.add_argument("--out", default=H.RUN_DIR + "/g1")
    ap.add_argument("--delay-steps", type=int, default=0, help="ctrlfix F5 delaycomp steps (exercises the prediction path)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fb.install(demo)
    fq.install()
    acc = Acc()
    series = {k: [] for k in ("dx_drop", "da0_drop", "dx_lean_drop", "it_hard", "m", "act_same", "old_viol",
                              "t_rt_drop", "t_rt_lean", "t_py")}
    last = {}
    qp_inner = sc.solve_constrained_box_qp
    cap = {"calls": []}

    def qp_rec(Hm, f, lo, hi, A=None, b=None, **kw):
        x, lam, feas = qp_inner(Hm, f, lo, hi, A, b, **kw)
        cap["calls"].append((np.array(Hm), np.array(f), np.array(A), np.array(b), x.copy(), lam.copy(), bool(feas)))
        return x, lam, feas

    # record the demo-side inputs the lean path needs (called right before fl.step)
    for name in ("payload_reflected_terms", "all_pair_barriers", "table_plane_barrier_rows"):
        orig = getattr(sc, name)

        def wrap(*a, _o=orig, _n=name, **k):
            if _n == "table_plane_barrier_rows":
                last.setdefault("tab", []).append((a, k))
                last["tab"] = last["tab"][-2:]
            elif _n == "payload_reflected_terms":
                last.setdefault("pay", []).append(a)
                last["pay"] = last["pay"][-3:]
            else:
                last["apb"] = k.get("extra_ego_keypoints", a[6] if len(a) > 6 else None)
            return _o(*a, **k)
        setattr(sc, name, wrap)

    ctl = {}
    base_step = mpc_fl.FLMPC.step          # FastFLMPC.step

    def sync(dst, src):
        dst.a_ego_prev = src.a_ego_prev.copy()
        dst.a_int_prev = src.a_int_prev.copy()
        dst._cup_axis_local = None if src._cup_axis_local is None else src._cup_axis_local.copy()
        dst._a_queue = [(a.copy(), b.copy()) for a, b in src._a_queue]

    def step_chk(self, model, data, lq, ld, rq, rd, qer, qdr, qir, qdir, pairs_all, *a, **kw):
        if "info" not in ctl:
            raise RuntimeError("no info")
        info = ctl["info"]
        if "drop" not in ctl:
            ctl["kw"] = dict(ctl_kwargs=self._rt_kwargs)
            ctl["drop"] = rt_flmpc.RTFLMPC(model, **self._rt_kwargs)
            ctl["lean"] = rt_flmpc.RTFLMPC(model, **self._rt_kwargs)
            ctl["lean_dyn"] = LeanRobotDynamics(model)
            ctl["ln"] = ctl["lean"]
            skw = dict(self._rt_kwargs, H_MARGIN=args.stress_hmargin)
            ctl["s_py"] = fb.FastFLMPC(model, **skw)
            ctl["s_rt"] = rt_flmpc.RTFLMPC(model, **skw)
        snap = rt_flmpc.RTFLMPC.__new__(rt_flmpc.RTFLMPC)
        sync(snap, self)
        # ---------------- py (reference) ----------------
        cap["calls"] = []
        sc.solve_constrained_box_qp = qp_rec
        t0 = time.perf_counter()
        try:
            out_py = base_step(self, model, data, lq, ld, rq, rd, qer, qdr, qir, qdir, pairs_all, *a, **kw)
        finally:
            sc.solve_constrained_box_qp = qp_inner
        t_py = time.perf_counter() - t0
        py_calls = cap["calls"]
        Hp, fp, Ap, bp, xp, lamp, feasp = py_calls[0]
        m = Ap.shape[0]
        n = Hp.shape[0]
        nb = self.n_blk
        # ---------------- rt drop-in ----------------
        dr = ctl["drop"]
        sync(dr, snap)
        t0 = time.perf_counter()
        out_dr = dr.step(model, data, lq, ld, rq, rd, qer, qdr, qir, qdir, pairs_all, *a, **kw)
        t_dr = time.perf_counter() - t0
        ws = dr._ws
        mr = int(dr.last_oi[K.OI_M])
        W_pe_eff = self.last_W_pos_ego_eff
        acc.mx("H_maxabs", mad(dr._H_cache[float(W_pe_eff)], Hp))
        acc.mx("f_maxabs", mad(ws.f, fp))
        acc.mx("f_rel", mad(ws.f, fp) / max(1e-300, float(np.max(np.abs(fp)))))
        acc.cnt("rows_count_equal", int(mr == m))
        if mr == m:
            acc.mx("A_maxabs", mad(ws.A[:m], Ap))
            acc.mx("b_maxabs", mad(ws.b[:m], bp))
            acc.cnt("A_bitident", int(np.array_equal(ws.A[:m], Ap)))
            acc.cnt("b_bitident", int(np.array_equal(ws.b[:m], bp)))
        acc.cnt("f_bitident", int(np.array_equal(ws.f, fp)))
        acc.cnt("steps")
        st = int(dr.last_oi[K.OI_HARD_STATUS])
        acc.cnt("py_feasible", int(feasp))
        acc.cnt("rt_optimal", int(st == rt_qp.ST_OPTIMAL))
        acc.cnt("rt_cap_hard", int(st == rt_qp.ST_ITERCAP))
        acc.cnt("feas_decision_agree", int(feasp == (st == rt_qp.ST_OPTIMAL)))
        acc.cnt("fallback_agree", int(self.last_fallback == dr.last_fallback))
        x_final_py = np.concatenate([self.a_ego_prev, self.a_int_prev])
        x_final_rt = np.concatenate([dr.a_ego_prev, dr.a_int_prev])
        da0 = mad(x_final_rt, x_final_py)
        dx = mad(ws.xout, xp) if feasp and st == 0 else np.nan
        series["dx_drop"].append(dx)
        series["da0_drop"].append(da0)
        acc.mx("tau_maxabs_drop", max(mad(out_dr[0], out_py[0]), mad(out_dr[1], out_py[1])))
        # active sets (general rows only): py lam>0 vs rt active ids < m
        q = int(ws.stats[1])
        act_rt = set(int(i) for i in ws.act[:q] if i < m) if st == 0 else set()
        act_py = set(np.nonzero(lamp > 0)[0].tolist())
        same = int(act_rt == act_py)
        series["act_same"].append(same)
        series["it_hard"].append(int(dr.last_oi[K.OI_HARD_IT]))
        series["m"].append(m)
        series["old_viol"].append(float(np.max(Ap @ xp - bp)))
        if st == 0:
            v, umin, stat = rt_qp.kkt_residuals(Hp, fp, -self.a_box * np.ones(n), self.a_box * np.ones(n), Ap, bp,
                                                ws.xout.copy(), ws.act[:q].copy(), ws.u[:q].copy())
            acc.mx("rt_kkt_viol", v)
            acc.mx("rt_kkt_stat", stat)
            acc.mx("rt_kkt_neg_u", max(0.0, -umin))
        # ---------------- rt lean ----------------
        ln = ctl["lean"]
        if ln._lean_cfg is None:
            tab = last.get("tab", [])
            ta, tk = tab[-1] if tab else ((), {})
            lkps = tab[-2][0][2] if len(tab) >= 2 else None
            ln.configure_lean(ctl["lean_dyn"], model, lq, ld, rq, rd, kw.get("left_ee_body"),
                              cup_body=info["cup_body"], right_ee_body=info["right_ee_body"],
                              speed_body=self.ee_speed_body_int,
                              ego_extra_kps=last.get("apb"), left_table_kps=lkps,
                              right_table_kps=ta[2] if ta else None,
                              table_z=(ta[6] if len(ta) > 6 else None) if ta else None,
                              table_margin=tk.get("margin", 0.0), table_kp_radius=bool(tk.get("kp_radius", False)),
                              table_gravfix=bool(demo._cf(info, "gravity")))
            ctl["cup_body"] = info["cup_body"]
        sync(ln, snap)
        cube = None
        for p in last.get("pay", []):
            if int(p[3]) != int(info["cup_body"]):
                assert int(p[2]) == int(info["right_ee_body"]), (p[2], info["right_ee_body"])
                cube = (int(p[3]), data.xpos[int(p[3])].copy(), data.xmat[int(p[3])].reshape(3, 3).copy())
        tab = last.get("tab", [])
        rtk = tab[-1][0][2] if (tab and kw.get("table_rows") is not None) else None
        cb = ctl["cup_body"]
        t0 = time.perf_counter()
        out_ln = ln.step_lean(data, qer, qdr, qir, qdir, left_ee_body=kw.get("left_ee_body"),
                              tilt_max_rad=kw.get("tilt_max_rad"), leash_ee_body=kw.get("leash_ee_body"),
                              leash_max_m=kw.get("leash_max_m"), cup_cone_facets=kw.get("cup_cone_facets"),
                              ego_hold=kw.get("ego_hold", True), table_floor=kw.get("table_rows") is not None,
                              cup_pose=(data.xpos[cb].copy(), data.xmat[cb].reshape(3, 3).copy())
                              if (cb is not None and kw.get("M_ego_extra") is not None) else None,
                              cube=cube, ego_extra_kps=last.get("apb"),
                              right_table_kps=rtk if rtk is not None else "cfg")
        t_ln = time.perf_counter() - t0
        wl = ln._ws
        acc.mx("lean_Me", mad(wl.M_e, self.last_M_e))
        acc.mx("lean_ce", mad(wl.c_e, self.last_c_e))
        acc.mx("lean_Mi", mad(wl.M_i, self.last_M_i))
        acc.mx("lean_ci", mad(wl.c_i, self.last_c_i))
        pk = rt_flmpc._topk_pairs(pairs_all, self.K_PAIRS)
        Kk = len(pk)
        acc.mx("lean_pair_h", mad(wl.h0_arr[:Kk], [p["h"] for p in pk]))
        acc.mx("lean_pair_n", mad(wl.n_arr[:Kk], np.stack([p["n"] for p in pk])))
        acc.mx("lean_pair_Je", mad(wl.Je_arr[:Kk], np.stack([p["Je"] for p in pk])))
        acc.mx("lean_pair_Ji", mad(wl.Ji_arr[:Kk], np.stack([p["Ji"] for p in pk])))
        if kw.get("table_rows") is not None:
            a_te, b_te, a_ti, b_ti = kw["table_rows"]
            acc.mx("lean_tab_a", max(mad(wl.a_tab_e[:len(b_te)], a_te), mad(wl.a_tab_i[:len(b_ti)], a_ti)))
            acc.mx("lean_tab_b", max(mad(wl.b_tab_e[:len(b_te)], b_te), mad(wl.b_tab_i[:len(b_ti)], b_ti)))
        ml = int(ln.last_oi[K.OI_M])
        acc.cnt("lean_rows_count_equal", int(ml == m))
        if ml == m:
            acc.mx("lean_A_maxabs", mad(wl.A[:m], Ap))
            acc.mx("lean_b_maxabs", mad(wl.b[:m], bp))
        acc.mx("lean_f_maxabs", mad(wl.f, fp))
        acc.cnt("lean_fallback_agree_drop", int(ln.last_fallback == dr.last_fallback))
        series["dx_lean_drop"].append(mad(wl.xout, ws.xout))
        acc.mx("lean_tau_vs_drop", max(mad(out_ln[0], out_dr[0]), mad(out_ln[1], out_dr[1])))
        series["t_rt_drop"].append(t_dr)
        series["t_rt_lean"].append(t_ln)
        series["t_py"].append(t_py)
        # ---------------- stress pair (forced-infeasible fallback coverage) ----------------
        if args.stress_every > 0 and (acc.d["steps"] % args.stress_every == 0):
            sp, sr = ctl["s_py"], ctl["s_rt"]
            sync(sp, snap)
            sync(sr, snap)
            cap["calls"] = []
            sc.solve_constrained_box_qp = qp_rec
            try:
                o1 = base_step(sp, model, data, lq, ld, rq, rd, qer, qdr, qir, qdir, pairs_all, *a, **kw)
            finally:
                sc.solve_constrained_box_qp = qp_inner
            scalls = cap["calls"]
            o2 = sr.step(model, data, lq, ld, rq, rd, qer, qdr, qir, qdir, pairs_all, *a, **kw)
            acc.cnt("stress_steps")
            acc.cnt("stress_py_infeasible", int(not scalls[0][6]))
            acc.cnt("stress_rt_not_optimal", int(sr.last_oi[K.OI_HARD_STATUS] != 0))
            acc.cnt("stress_feas_agree", int(scalls[0][6] == (sr.last_oi[K.OI_HARD_STATUS] == 0)))
            acc.cnt("stress_fallback_agree", int(sp.last_fallback == sr.last_fallback))
            acc.cnt(f"stress_py_fb{sp.last_fallback}")
            acc.cnt(f"stress_rt_fb{sr.last_fallback}")
            if sr.last_oi[K.OI_HARD_STATUS] != 0:
                acc.lst("stress_relax_it", sr.last_oi[K.OI_RELAX_IT])
                acc.cnt("stress_rt_relax_cap", int(sr.last_oi[K.OI_RELAX_STATUS] == rt_qp.ST_ITERCAP))
            if sp.last_fallback == 1 and sr.last_fallback == 1:
                acc.lst("stress_relaxed_da0", mad(np.concatenate([sr.a_ego_prev, sr.a_int_prev]),
                                                  np.concatenate([sp.a_ego_prev, sp.a_int_prev])))
                # clearance violation of both relaxed answers on the HARD collision rows
                Hs, fs, As, bs = scalls[0][0], scalls[0][1], scalls[0][2], scalls[0][3]
                mcoll = sp.K_PAIRS * len(sp.coll_steps)
                acc.lst("stress_py_relaxed_collviol", float(np.max(As[:mcoll] @ scalls[1][4][:n] - bs[:mcoll])))
                acc.lst("stress_rt_relaxed_collviol", float(np.max(As[:mcoll] @ sr._ws.xout - bs[:mcoll])))
                acc.lst("stress_py_relaxed_hardviol", float(np.max(As[mcoll:] @ scalls[1][4][:n] - bs[mcoll:])))
                acc.lst("stress_rt_relaxed_hardviol", float(np.max(As[mcoll:] @ sr._ws.xout - bs[mcoll:])))
            for nm, o in (("py", o1), ("rt", o2)):
                tl = sc.TAU_MAX
                acc.mx(f"stress_{nm}_tau_over", max(0.0, float(np.max(np.abs(np.concatenate([o[0], o[1]])))) - tl))
        last.pop("pay", None)
        last.pop("tab", None)
        return out_py

    orig_init = fb.FastFLMPC.__init__

    def init_rec(self, model, *a, **kw):
        orig_init(self, model, *a, **kw)
        self._rt_kwargs = dict(kw)

    fb.FastFLMPC.__init__ = init_rec
    mpc_fl.FLMPC.step = step_chk
    if args.delay_steps:
        H.configure(args.profile)
        demo.CTRLFIX_DELAY_STEPS = int(args.delay_steps)
        _cfg = H.configure
        H.configure = lambda prof: (_cfg(prof), setattr(demo, "CTRLFIX_DELAY_STEPS", int(args.delay_steps)))
    info = H.build(args.profile, args.seed)
    ctl["info"] = info
    t0 = time.perf_counter()
    log, summ = H.run(info, args.profile, max_steps=args.max_steps)
    wall = time.perf_counter() - t0
    out = dict(profile=args.profile, seed=args.seed, delay_steps=args.delay_steps, wall_s=wall, summary_py=dict(
        collision=summ["collision"], placed_final=summ["placed_final"], h_min=summ["h_min"]),
        counts={k: v for k, v in acc.d.items() if not isinstance(v, list)})
    if isinstance(acc.d.get("stress_relax_it"), list):
        out["stress_relax_it_hist"] = np.bincount(np.array(acc.d["stress_relax_it"], int)).tolist()
    for k in ("ln", "drop"):
        if k in ctl:
            out[f"jit_warmup_s_{k}"] = getattr(ctl[k], "jit_warmup_s", None)
    for k, v in acc.d.items():
        if isinstance(v, list):
            vv = np.array(v, float)
            out["counts"][k] = dict(n=len(vv), mean=float(np.nanmean(vv)) if len(vv) else None,
                                    max=float(np.nanmax(vv)) if len(vv) else None,
                                    min=float(np.nanmin(vv)) if len(vv) else None)
    S = {k: np.array(v, float) for k, v in series.items()}
    ok = np.isfinite(S["dx_drop"])
    out["dx_drop"] = dict(median=float(np.median(S["dx_drop"][ok])), p99=float(np.percentile(S["dx_drop"][ok], 99)),
                          max=float(np.max(S["dx_drop"][ok])), frac_below_1e6=float(np.mean(S["dx_drop"][ok] < 1e-6)))
    out["da0_drop"] = dict(median=float(np.median(S["da0_drop"])), p99=float(np.percentile(S["da0_drop"], 99)),
                           max=float(np.max(S["da0_drop"])))
    out["dx_lean_vs_drop_max"] = float(np.max(S["dx_lean_drop"]))
    out["act_same_frac"] = float(np.mean(S["act_same"]))
    out["it_hard"] = dict(mean=float(S["it_hard"].mean()), p99=float(np.percentile(S["it_hard"], 99)),
                          max=int(S["it_hard"].max()), hist=np.bincount(S["it_hard"].astype(int)).tolist())
    out["old_point_rowviol"] = dict(median=float(np.median(S["old_viol"])), p99=float(np.percentile(S["old_viol"], 99)),
                                    max=float(np.max(S["old_viol"])))
    for k in ("t_rt_drop", "t_rt_lean", "t_py"):
        v = S[k] * 1e3
        out[k + "_ms"] = dict(mean=float(v.mean()), p50=float(np.median(v)), p99=float(np.percentile(v, 99)),
                              max=float(v.max()))
    fn = f"{args.out}/g1_{args.profile}_s{args.seed}" + (f"_delay{args.delay_steps}" if args.delay_steps else "")
    json.dump(out, open(fn + ".json", "w"), indent=1)
    np.savez_compressed(fn + ".npz", **S)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
