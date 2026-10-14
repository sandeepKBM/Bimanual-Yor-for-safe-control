"""vdirect/analyze_cb.py -- CB arm, the conservatism metric for every arm, and criterion K6 (PREREG_ADDENDUM_CB).
Imports the pre-registered analyze.py unchanged. SAVED DATA ONLY.

  python -m world_model.v2.vdirect.analyze_cb [--grid ILAB_ROOT] [--cb CB_ROOT] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.vdirect import analyze as A

GRID = Path("/common/users/ss5772/bimanual_data/vdirect_ilab")
CBR = Path("/data/local/ss5772/bimanual_runs/vdirect/cb")
SAFE_MARGIN = 0.1
NB = 10000


def load_cb_spill(cbroot, policy, n_states=2000):
    out = {}
    for s in (0, 1, 2):
        parts = sorted((cbroot / "cl").glob(f"{policy}_s{s}_part*.npz"))
        if not parts or not all(Path(str(p) + "._DONE").exists() for p in parts):
            continue
        sp = np.full(n_states, -1, int); r0 = np.zeros(n_states)
        acc = np.zeros(4)
        for p in parts:
            f = np.load(p)
            sp[f["idx"]] = f["spill"].astype(int); r0[f["idx"]] = f["r0"]
            acc += [f["cl_n"].sum(), f["cl_err"].sum(), f["cl_cov_bar"].sum(), f["cl_r_sum"].sum()]
        if (sp < 0).any():
            continue
        out[s] = dict(spill_all=sp.astype(bool), r0=r0, cl_aci=dict(miscoverage_q=acc[1] / acc[0],
                                                                      coverage_qbar=acc[2] / acc[0],
                                                                      mean_radius=acc[3] / acc[0], n=int(acc[0])))
    return out


def cons_arrays(cbroot):
    """-> dict[(arm, b)] -> dict[seed] -> dict(flag_now, flag_1step) + common state info."""
    res, common = {}, None
    for f in sorted((cbroot / "cons").glob("*.npz")):
        if not Path(str(f) + "._DONE").exists():
            continue
        name = f.stem
        arm, b, s = name.split("_")
        b, s = int(b[1:]), int(s[1:])
        d = np.load(f)
        if common is None:
            common = dict(ep=d["ep"], l=d["l"], Vp=d["Vp"])
        kinds = [("std", arm)] if arm != "B0" else [("std", "N0"), ("M", "M"), ("P", "P")]
        for k, nm in kinds:
            res.setdefault((nm, b), {})[s] = dict(now=d[f"V_now_{k}"] <= 0, one=d[f"V_1step_{k}"] <= 0)
        if arm == "B0":
            for k, nm in (("Vbar_CB", "CB"), ("Vbar_CBo", "CBo")):
                res.setdefault((nm, 0), {})[s] = dict(now=d[k] <= 0, one=d[k] <= 0)
            res.setdefault(("CB", 0), {})[s]["meta"] = json.loads(str(d["cb_meta"]))
            res[("CB", 0)][s]["r"] = d["r_CB"]
    return res, common


def cluster_boot(res, common, which="now", nb=NB, seed=0):
    """paired episode-cluster bootstrap (one episode draw shared by all methods) + seed resampling per arm.
    -> dict[key] -> dict(metric -> (point, samples))"""
    rng = np.random.default_rng(seed)
    ep = common["ep"]; E = int(ep.max()) + 1
    safe = (common["l"] > 0) & (common["Vp"] >= SAFE_MARGIN)
    doomed = (common["l"] > 0) & (common["Vp"] <= 0)
    unsafe = common["Vp"] <= 0
    W = np.zeros((nb, E), np.float32)
    draws = rng.integers(0, E, (nb, E))
    np.add.at(W, (np.repeat(np.arange(nb), E), draws.ravel()), 1.0)
    den = {m: np.bincount(ep[mask], minlength=E).astype(np.float32)
           for m, mask in (("cons", safe), ("recall", doomed), ("false_safe", unsafe))}
    out = {}
    for key, bys in res.items():
        seeds = sorted(bys)
        met = {}
        for m, mask in (("cons", safe), ("recall", doomed), ("false_safe", unsafe)):
            per_seed, pts = [], []
            for s in seeds:
                flag = bys[s][which]
                hit = flag if m != "false_safe" else ~flag
                num = np.bincount(ep[mask & hit], minlength=E).astype(np.float32)
                pts.append(num.sum() / den[m].sum())
                per_seed.append((W @ num) / np.maximum(W @ den[m], 1.0))
            X = np.stack(per_seed, 1)
            pick = rng.integers(0, len(seeds), (nb, len(seeds)))
            met[m] = (float(np.mean(pts)), np.take_along_axis(X, pick, 1).mean(1), [float(p) for p in pts])
        out[key] = met
    return out, dict(n_safe=int(safe.sum()), n_doomed=int(doomed.sum()), n_unsafe=int(unsafe.sum()), n_eps=E)


def k6(B, CBs, cbkey, budgets=(3000, 1000)):
    ci = A.ci
    out = {}
    for b in budgets:
        if ("V", b) not in B or (cbkey, 0) not in B or ("V", b) not in CBs or (cbkey, 0) not in CBs:
            out[b] = dict(status="MISSING"); continue
        dsp = B[(cbkey, 0)][1] - B[("V", b)][1]                 # CB - V spill
        dcons = CBs[(cbkey, 0)]["cons"][1] - CBs[("V", b)]["cons"][1]
        a_ = ci(dsp)[0] > 0
        not_worse = not (ci(-dsp)[0] > 0)
        b_ = not_worse and ci(dcons)[0] > 0
        out[b] = dict(spill_V=B[("V", b)][0], spill_CB=B[(cbkey, 0)][0], ci_spill_CB_minus_V=ci(dsp),
                      cons_V=CBs[("V", b)]["cons"][0], cons_CB=CBs[(cbkey, 0)]["cons"][0],
                      ci_cons_CB_minus_V=ci(dcons), cond_a=bool(a_), cond_b=bool(b_), pass_=bool(a_ or b_))
    v3 = out.get(3000, {})
    verdict = "INCOMPLETE" if v3.get("status") == "MISSING" else ("PASS" if v3.get("pass_") else "KILL")
    if verdict == "PASS" and not out.get(1000, {}).get("pass_", False):
        verdict = "PASS (qualified: >= 3k branches)"
    return dict(cb=cbkey, per_budget=out, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--cb", default=str(CBR))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    grid, cbroot = Path(a.grid), Path(a.cb)
    out = Path(a.out) if a.out else cbroot / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = A.load(grid)
    st = np.load(grid / "shared" / "cl_states.npz")
    sav = st["Vp0"] > 0
    cbinfo = {}
    for pol, key in (("nominal", "CB"), ("robust", "CBr")):
        sp = load_cb_spill(cbroot, pol)
        for s, d in sp.items():
            runs.setdefault((key, 0), {})[s] = dict(spill=d["spill_all"][sav].astype(float))
            cbinfo.setdefault(key, {})[s] = dict(cl_aci=d["cl_aci"], r0_mean=float(d["r0"].mean()))
    B, ns = A.boot(runs)
    res, common = cons_arrays(cbroot)
    CBs, counts = cluster_boot(res, common, "now")
    CB1, _ = cluster_boot(res, common, "one")
    ci = A.ci
    L = [f"Conservatism = false-intervention rate on clearly physics-safe TEST states (l > 0, V_phys >= {SAFE_MARGIN}; "
         f"n = {counts['n_safe']} states in {counts['n_eps']} episodes), threshold 0. Recall on {counts['n_doomed']} "
         f"doomed states; false-safe on {counts['n_unsafe']} V_phys <= 0 states. 95% paired episode-cluster bootstrap.",
         "", "| arm | budget | seeds | spill (savable) [95% CI] | conservatism [95% CI] | doomed recall | false-safe | "
         "conservatism, one-step trigger V(f(z,a)) <= 0 |", "|---|---|---|---|---|---|---|---|"]
    order = ["CB", "CBr", "CBo", "N0", "M", "P", "V", "R", "U", "Vn", "D"]
    keys = sorted(set(B) | set(CBs), key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))
    for k in keys:
        sp = A.fmt_pct(B[k][0], ci(B[k][1])) if k in B else "n/a (trigger only)"
        cs = CBs.get(k)
        c1 = CB1.get(k)
        f = lambda t: f"{100*t[0]:.2f}% [{100*ci(t[1])[0]:.2f}, {100*ci(t[1])[1]:.2f}]"
        L.append(f"| {k[0]} | {k[1]} | {B[k][2] if k in B else len(res.get(k, {}))} | {sp} | "
                 f"{f(cs['cons']) if cs else 'n/a'} | {f(cs['recall']) if cs else 'n/a'} | "
                 f"{f(cs['false_safe']) if cs else 'n/a'} | {f(c1['cons']) if c1 else 'n/a'} |")
    K6 = k6(B, CBs, "CB"); K6r = k6(B, CBs, "CBr") if ("CBr", 0) in B else dict(verdict="INCOMPLETE")
    cbm = {s: d.get("meta") for s, d in res.get(("CB", 0), {}).items()}
    summ = dict(counts=counts, K6=K6, K6_secondary_robust_fallback=K6r, cb_meta=cbm, cb_closed_loop=cbinfo,
                arms={f"{k[0]}_b{k[1]}": dict(spill=(B[k][0], ci(B[k][1])) if k in B else None,
                                             **({m: (v[0], ci(v[1]), v[2]) for m, v in CBs[k].items()} if k in CBs else {}),
                                             cons_one_step=(CB1[k]["cons"][0], ci(CB1[k]["cons"][1])) if k in CB1 else None)
                      for k in keys})
    (out / "tables_cb.md").write_text("\n".join(L) + "\n")
    (out / "summary_cb.json").write_text(json.dumps(summ, indent=1, default=str))
    print("\n".join(L)); print(json.dumps(dict(K6=K6, K6r=K6r, cb_meta=cbm, cb_closed_loop=cbinfo), indent=1, default=str))


if __name__ == "__main__":
    main()
