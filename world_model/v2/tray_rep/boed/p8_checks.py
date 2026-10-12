"""boed/p8_checks.py -- checks of P8 (boed/p8.py). VAL ONLY. SIMULATION / SAVED DATA ONLY. CPU. No hardware, no network.

  python -m world_model.v2.tray_rep.boed.p8_checks c1      # null belief == P6 on 6 VAL carries per setup (dropped, n_ticks, t_complete_s, n_interv)
  python -m world_model.v2.tray_rep.boed.p8_checks c2      # determinism (1 vs 2 workers) and resume == fresh
  python -m world_model.v2.tray_rep.boed.p8_checks c3      # no hidden-mu read (grep + poison test), no hardware module imported
  python -m world_model.v2.tray_rep.boed.p8_checks c4      # belief sanity on 40 VAL carries per setup (coverage; mu_q vs the table thresholds)
  python -m world_model.v2.tray_rep.boed.p8_checks smoke   # 20 VAL carries piper_trk seed 0: P8 vs P6 (uses the c4 P8 records)
Everything goes to <OUT>/tmp/* and <OUT>/checks/*.json (OUT = bimanual_data/tray_rep/boed_p8). P6 is run by the unmodified boed.run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.boed import policy as PO
from world_model.v2.tray_rep.boed import run as BR

OUT = K.ROOT / "boed_p8"
TBL = K.ROOT / "boed" / "main"
PY = sys.executable
SETUPS = ("piper_trk", "piper")
KEYS = ("dropped", "n_ticks", "t_complete_s", "n_interv")
HERE = Path(__file__).resolve().parent


def sh(args, log):
    """start `python -m ...` (inherits the thread pins / TMPDIR env of the caller); -> Popen"""
    f = open(log, "w")
    return subprocess.Popen([PY, "-m", *args], stdout=f, stderr=subprocess.STDOUT)


def wait_all(ps):
    for p in ps:
        p.wait()
    bad = [p.args for p in ps if p.returncode != 0]
    if bad:
        raise SystemExit(f"failed: {bad}")


def p8_cmd(setup, split, n_eps, out, workers=1, belief="online", seed=0, shard="0/1", ep_start=0):
    return ["world_model.v2.tray_rep.boed.p8", "--setup", setup, "--seed", str(seed), "--split", split, "--n-eps", str(n_eps), "--ep-start", str(ep_start),
            "--shard", shard, "--workers", str(workers), "--out", str(out), "--table-dir", str(TBL), "--n-nodes", "8", "--eps", "0.02", "--belief", belief]


def p6_cmd(setup, split, n_eps, out, seed=0):
    return ["world_model.v2.tray_rep.boed.run", "--policy", "P6", "--setup", setup, "--seed", str(seed), "--split", split, "--n-eps", str(n_eps),
            "--table-dir", str(TBL), "--n-nodes", "8", "--eps", "0.02", "--out", str(out)]


def recs_of(d, setup, stem="P8__s0"):
    p = Path(d) / "main" / "val" / setup / f"{stem}.jsonl"
    return {r["ep"]: r for r in BR.read_jsonl(p)}


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def c1():
    t = OUT / "tmp"
    ps = []
    for s in SETUPS:
        ps.append(sh(p6_cmd(s, "val", 6, t / "c1_p6"), t / f"c1_p6_{s}.log"))
        ps.append(sh(p8_cmd(s, "val", 6, t / "c1_p8", belief="null"), t / f"c1_p8_{s}.log"))
    wait_all(ps)
    res = {}
    for s in SETUPS:
        a, b = recs_of(t / "c1_p6", s, "P6__s0"), recs_of(t / "c1_p8", s)
        rows, ok = [], True
        for ep in sorted(a):
            d = {k: (a[ep][k], b[ep][k], same(a[ep][k], b[ep][k])) for k in KEYS}
            extra = {k: same(a[ep][k], b[ep][k]) for k in ("drop_tick", "progress", "goal_reached", "interv_rate", "I_feas")}
            ok &= all(v[2] for v in d.values()) and all(extra.values())
            rows.append(dict(ep=ep, fields=d, extra=extra, I_hist_p6=a[ep]["I_hist"], I_hist_p8=b[ep]["I_hist"]))
        res[s] = dict(n=len(a), identical=bool(ok and len(a) == len(b) == 6), rows=rows)
        print(f"c1 {s}: n={len(a)}/{len(b)} identical={res[s]['identical']}")
    (OUT / "checks").mkdir(exist_ok=True)
    (OUT / "checks" / "c1.json").write_text(json.dumps(res, indent=1, default=float))
    return all(v["identical"] for v in res.values())


def strip(r):
    return {k: v for k, v in r.items() if k != "wall_s"}


def c2():
    t = OUT / "tmp"
    ps = []
    for s in SETUPS:
        ps.append(sh(p8_cmd(s, "val", 6, t / "c2_w1", workers=1), t / f"c2_w1_{s}.log"))
        ps.append(sh(p8_cmd(s, "val", 6, t / "c2_w2", workers=2), t / f"c2_w2_{s}.log"))
        ps.append(sh(p8_cmd(s, "val", 3, t / "c2_res", workers=1), t / f"c2_res_a_{s}.log"))
    wait_all(ps)
    ps = [sh(p8_cmd(s, "val", 6, t / "c2_res", workers=1), t / f"c2_res_b_{s}.log") for s in SETUPS]     # resume: the first 3 are kept, 3 more run
    wait_all(ps)
    res = {}
    for s in SETUPS:
        w1, w2, rs = (recs_of(t / d, s) for d in ("c2_w1", "c2_w2", "c2_res"))
        res[s] = dict(n=len(w1), w1_eq_w2=bool(set(w1) == set(w2) and all(strip(w1[e]) == strip(w2[e]) for e in w1)),
                      resume_eq_fresh=bool(set(w1) == set(rs) and all(strip(w1[e]) == strip(rs[e]) for e in w1)))
        print(f"c2 {s}: {res[s]}")
    (OUT / "checks" / "c2.json").write_text(json.dumps(res, indent=1))
    return all(v["w1_eq_w2"] and v["resume_eq_fresh"] and v["n"] == 6 for v in res.values())


def c3():
    out = {}
    # (a) grep: the P8 module touches the hidden mu in exactly the marked places
    src = (HERE / "p8.py").read_text().splitlines()
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(src) if re.search(r"\.mu\b|mu_true|\bhid\b|\.hidden|mu_roll", ln) and not ln.lstrip().startswith(("#", '"""'))]
    allowed = [h for h in hits if "mu_true" in h[1] or "run.mu = HiddenMu()" in h[1] or "hid=None" in h[1] or "poison" in h[1]]
    out["grep_hits"] = hits
    out["grep_unexpected"] = [h for h in hits if h not in allowed and "run.mu" not in h[1] and "mu_roll" not in h[1]]
    # (b) poison test: a carry with Run.mu poisoned runs; reading the poison raises; the logged coverage uses the true mu
    from world_model.v2.tray_rep.boed import episode as EP
    from world_model.v2.tray_rep.boed import p8 as P8
    P8.G.clear()
    P8._init(dict(setup="piper_trk", seed=0, split="val", table_dir=str(TBL), n_nodes=8, eps=0.02, belief="online"))
    ctx, bel = P8.G["ctx"], P8.G["bel"]
    seen = []
    orig_begin = ctx.filt.begin

    def spy(c):
        seen.append(c)
        return orig_begin(c)
    ctx.filt.begin = spy
    r = EP.Run(ctx, ctx.store.ep_info(0)["seed"], None)
    mu_true = r.mu
    # monitor: after run_p8 starts, Run.mu must be the poison; checked from the F3 call (the carry loop's callee)
    orig_call = ctx.filt.__class__.__call__
    flags = []

    def call(self, *a, **k):
        flags.append(type(r.mu).__name__)
        return orig_call(self, *a, **k)
    ctx.filt.__class__.__call__ = call
    try:
        P8.run_p8(r, bel)
    finally:
        ctx.filt.__class__.__call__ = orig_call
    rec = r.record("P8")
    out["poison_active_all_ticks"] = bool(flags) and set(flags) == {"HiddenMu"}
    out["begin_hid_is_none"] = all(c.get("hid", 1) is None for c in seen)
    out["mu_restored_for_record"] = bool(r.mu == mu_true and rec["mu"] == mu_true)
    try:
        float(P8.HiddenMu())
        out["poison_raises"] = False
    except RuntimeError:
        out["poison_raises"] = True
    # (c) the belief class has no access to the Run
    import inspect
    out["belief_api_args"] = [str(inspect.signature(P8.OnlineBelief.feed)), str(inspect.signature(P8.OnlineBelief.mu_quantile))]
    # (d) no hardware modules imported
    bad = sorted(m for m in sys.modules if m.split(".")[0] in ("YOR", "xarm", "piper_sdk", "rospy", "rclpy", "serial", "can") or ".robot." in m or m.endswith(".robot")
                 or "pyrealsense" in m)
    out["hardware_modules_imported"] = bad
    ok = (not out["grep_unexpected"] and out["poison_active_all_ticks"] and out["begin_hid_is_none"] and out["mu_restored_for_record"]
          and out["poison_raises"] and not bad)
    out["ok"] = bool(ok)
    print("c3:", json.dumps(out, indent=1, default=str))
    (OUT / "checks" / "c3.json").write_text(json.dumps(out, indent=1, default=str))
    return ok


def thresholds(table, fracs):
    """smallest mu (population grid) at which P8's rule picks I > the slowest grid intensity, for each remaining-path fraction."""
    from world_model.v2.tray_rep.boed.p7 import pick_worst_case
    grid = np.exp(np.linspace(np.log(5e-4), np.log(6e-3), 400))
    n_slow = len(table.igrid) - 1
    out = {}
    for fr in fracs:
        idx = [pick_worst_case(table, m, 0.02, fr) for m in grid]
        fast = [m for m, i in zip(grid, idx) if i < n_slow]
        out[f"{fr:.2f}"] = float(min(fast)) if fast else None
    return out


def thr_for(thr, frc):
    k = min(thr, key=lambda x: abs(float(x) - frc))
    return thr[k]


def c4(workers=6):
    t = OUT / "tmp"
    ps = []
    for s in SETUPS:
        for k in range(workers // 2):
            ps.append(sh(p8_cmd(s, "val", 40, t / "c4", shard=f"{k}/{workers // 2}"), t / f"c4_{s}_{k}.log"))
    wait_all(ps)
    res = {}
    for s in SETUPS:
        d = Path(t / "c4" / "main" / "val" / s)
        recs = {}
        for p in sorted(d.glob("P8__s0__p*of*.jsonl")):
            recs.update({r["ep"]: r for r in BR.read_jsonl(p)})
        R = [recs[e] for e in sorted(recs)]
        tbl = PO.Table.load(TBL / f"table_{s}.json")
        fr = [x for r in R for x in r["p8_frac_hist"] if x is not None]
        thr = thresholds(tbl, sorted({round(x, 2) for x in fr} | {0.3, 0.5, 0.7}))
        bnd = {}
        for b in (0, 1):
            rows = [(r["p8_mu_q_hist"][b], r["p8_frac_hist"][b], r["mu"]) for r in R if r["p8_mu_q_hist"][b] is not None]
            allows = []
            for mq, frc, mu in rows:
                from world_model.v2.tray_rep.boed.p7 import pick_worst_case
                allows.append(pick_worst_case(tbl, mq, 0.02, frc) < len(tbl.igrid) - 1)
            mus = np.array([x[0] for x in rows]) if rows else np.array([])
            bnd[f"boundary{b + 2}"] = dict(n=len(rows), frac_allows_I_gt_0p4=float(np.mean(allows)) if allows else None,
                                           mu_q_median=float(np.median(mus)) if len(mus) else None, mu_q_min=float(mus.min()) if len(mus) else None,
                                           mu_q_max=float(mus.max()) if len(mus) else None,
                                           true_mu_above_thr_frac=float(np.mean([thr_for(thr, frc) is not None and mu >= thr_for(thr, frc) for _, frc, mu in rows])) if rows else None)
        cov = [r for r in R if "bel_cover90" in r]
        res[s] = dict(n=len(R), cover90=float(np.mean([r["bel_cover90"] for r in cov])), cover90_n=len(cov),
                      cover50=float(np.mean([r["bel_cover50"] for r in cov])),
                      cover90_not_dropped=float(np.mean([r["bel_cover90"] for r in cov if not r["dropped"]])),
                      bel_sd_log=float(np.mean([r["bel_sd_log"] for r in cov])), bel_rmse_log=float(np.sqrt(np.mean([r["bel_err_log"] ** 2 for r in cov]))),
                      prior_sd_log=float(np.log(6e-3 / 5e-4) / np.sqrt(12)), boundaries=bnd, thresholds_mu_by_frac=thr,
                      drop=float(np.mean([r["dropped"] for r in R])), I_hist_counts={str(r["I_hist"]): 0 for r in R})
        for r in R:
            res[s]["I_hist_counts"][str(r["I_hist"])] += 1
        print(f"c4 {s}: {json.dumps({k: v for k, v in res[s].items() if k != 'I_hist_counts'}, default=float)}")
        print("   I_hist:", res[s]["I_hist_counts"])
    (OUT / "checks" / "c4.json").write_text(json.dumps(res, indent=1, default=float))
    return True


def smoke(workers=4):
    t = OUT / "tmp"
    ps = [sh(p6_cmd("piper_trk", "val", 20, t / "smoke_p6"), t / "smoke_p6.log")]
    wait_all(ps)
    s = "piper_trk"
    p8 = {}
    for p in sorted((t / "c4" / "main" / "val" / s).glob("P8__s0__p*of*.jsonl")):
        p8.update({r["ep"]: r for r in BR.read_jsonl(p)})
    p6 = recs_of(t / "smoke_p6", s, "P6__s0")
    eps = sorted(e for e in p6 if e in p8 and e < 20)
    d6 = np.array([p6[e]["dropped"] for e in eps]); d8 = np.array([p8[e]["dropped"] for e in eps])
    both = ~(d6 | d8)
    t6 = np.array([p6[e]["t_complete_s"] for e in eps]); t8 = np.array([p8[e]["t_complete_s"] for e in eps])
    hist = {}
    for e in eps:
        for k, I in enumerate(p8[e]["I_hist"]):
            hist.setdefault(k + 1, {}).setdefault(I, 0)
            hist[k + 1][I] += 1
    out = dict(n=len(eps), drops_p6=int(d6.sum()), drops_p8=int(d8.sum()), n_neither_dropped=int(both.sum()),
               t_complete_p6=float(t6[both].mean()) if both.any() else None, t_complete_p8=float(t8[both].mean()) if both.any() else None,
               I_hist_per_transport={str(k): v for k, v in hist.items()},
               mean_wall_s_p6=float(np.mean([p6[e]["wall_s"] for e in eps])), mean_wall_s_p8=float(np.mean([p8[e]["wall_s"] for e in eps])))
    print("smoke:", json.dumps(out, indent=1, default=float))
    (OUT / "checks" / "smoke.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    fn = dict(c1=c1, c2=c2, c3=c3, c4=c4, smoke=smoke)[sys.argv[1]]
    t0 = time.perf_counter()
    r = fn()
    print(f"[{sys.argv[1]}] {'OK' if r in (True, None) else 'FAIL'} [{time.perf_counter() - t0:.0f}s]")
