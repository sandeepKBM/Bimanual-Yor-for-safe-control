"""boed/report.py -- tables, belief calibration and the paired-bootstrap criteria. SIMULATION / SAVED DATA ONLY.
Reads <out>/<tag>/<split>/<setup>/<policy_tag>__s<seed>.jsonl written by boed.run.

  python -m world_model.v2.tray_rep.boed.report table   --tag main --split test    # per policy x setup (pooled over value seeds, and per seed)
  python -m world_model.v2.tray_rep.boed.report primary --tag main --split test    # the criteria of the pre-registration (paired bootstrap)
  python -m world_model.v2.tray_rep.boed.report all     --tag main

Criteria (PREREG_BOED_DRAFT.md). All on the tracker setup (piper_trk), the three value seeds pooled, paired over the same carries:
  drop      Delta = drop(A) - drop(B); non-inferior if the upper 95% bound of Delta is <= margin (2 pp).
  time      ratio = mean t_complete(A) / mean t_complete(B) over the carries neither policy dropped; "A faster by x" if the upper 95% bound
            of the ratio is <= 1 - x.
t_complete_s = seconds to the goal (probe and hold time included), or the pace-extrapolated time if the plan ended short of the goal.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep.bridge import metrics as MT
from world_model.v2.tray_rep.bridge.filters import spec_tag
from world_model.v2.tray_rep.boed.run import read_jsonl, OUT

NAMES = {"P1": "P1 conservative", "P2": "P2 reactive (stored carry)", "P3": "P3 probe-at-pickup", "P4": "P4 probe-on-trigger",
         "P5": "P5 oracle-mu", "P6": "P6 population intensity"}
ORDER = ["P1", "P2", "P6", "P3", "P4", "P5"]
MARGIN, B_N, B_SEED = 0.02, 2000, 20261005


def load(root, tag, split):
    """-> {setup: {policy: {vseed: {ep: rec}}}} for the P1..P6 runs (table cells 'FIX' are skipped)."""
    out = defaultdict(lambda: defaultdict(dict))
    for pj in sorted(glob.glob(str(Path(root) / tag / split / "*" / "*.jsonl"))):
        pj = Path(pj)
        setup = pj.parent.name
        stem = pj.stem
        m = re.match(r"^(.*)__s(\d+)(?:__p\d+of\d+)?$", stem)
        if not m:
            continue
        pol, sd = m.group(1).split("__")[0], int(m.group(2))
        if not pol.startswith("P"):
            continue
        out[setup][pol].setdefault(sd, {}).update({r["ep"]: r for r in read_jsonl(pj)})
    return out


def _nm(a):
    return float(np.nanmean(a)) if len(a) and np.isfinite(a).any() else float("nan")


def row(recs_by_seed):
    allr = [r for d in recs_by_seed.values() for r in d.values()]
    n = len(allr)
    g = lambda k: np.array([np.nan if r.get(k) is None else float(r[k]) for r in allr])
    dr = np.array([bool(r["dropped"]) for r in allr])
    per = [np.mean([bool(r["dropped"]) for r in d.values()]) for d in recs_by_seed.values()]
    out = dict(n=n, seeds=sorted(recs_by_seed), drop=float(dr.mean()), drop_ci=MT.wilson(int(dr.sum()), n), drop_sd_seed=float(np.std(per, ddof=1)) if len(per) > 1 else float("nan"),
               t_complete=_nm(g("t_complete_s")), total_s=_nm(g("total_s")), goal=float(np.mean([bool(r["goal_reached"]) for r in allr])),
               progress=_nm(g("progress")), interv=_nm(g("interv_rate")), I_rel=_nm(g("I_rel")),
               probe_run=int(sum(bool(r.get("probe_run")) for r in allr)), probe_drop=int(sum(bool(r.get("probe_drop")) for r in allr)),
               probe_skip=int(sum(bool(r.get("probe_skipped")) for r in allr)), triggered=int(sum(r.get("p4_trigger_tick", -1) >= 0 for r in allr)),
               probe_s=_nm(np.array([r["probe_ticks"] * 0.02 for r in allr if r.get("probe_run")])),
               hold_s=_nm(np.array([r["hold_ticks"] * 0.02 for r in allr if r.get("probe_run")])))
    b = [r for r in allr if "bel_cover90" in r]
    if b:
        out.update(bel_n=len(b), cover90=float(np.mean([r["bel_cover90"] for r in b])), cover50=float(np.mean([r["bel_cover50"] for r in b])),
                   bel_sd=float(np.mean([r["bel_sd_log"] for r in b])), bel_rmse=float(np.sqrt(np.mean([r["bel_err_log"] ** 2 for r in b]))))
    return out


def fmt_table(data):
    L = []
    prior_sd = np.log(6e-3 / 5e-4) / np.sqrt(12)
    for setup in sorted(data):
        L.append(f"\n### setup: {setup}\n")
        L.append("| policy | n (carries x seeds) | drop (95% CI) | sd over seeds | goal reached | path progress | t_complete s (non-dropped) | total s | I_rel mean | "
                 "F3 interv rate | probes run / dropped / skipped | P4 triggered | probe s (incl. hold) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in ORDER:
            if p not in data[setup]:
                continue
            r = row(data[setup][p])
            L.append(f"| {NAMES[p]} | {r['n']} ({len(r['seeds'])} seeds) | {r['drop']:.3f} ({r['drop_ci'][0]:.3f}, {r['drop_ci'][1]:.3f}) | {r['drop_sd_seed']:.3f} | "
                     f"{r['goal']:.2f} | {r['progress']:.2f} | {r['t_complete']:.1f} | {r['total_s']:.1f} | {r['I_rel']:.2f} | {r['interv']:.3f} | "
                     f"{r['probe_run']} / {r['probe_drop']} / {r['probe_skip']} | {r['triggered'] if p == 'P4' else '-'} | "
                     f"{(r['probe_s'] + 0) if np.isfinite(r['probe_s']) else float('nan'):.1f} |")
        L.append(f"\nBelief calibration (probe policies; 90% and 50% central intervals should cover about 0.90 and 0.50; prior sd of log mu = {prior_sd:.2f}):\n")
        L.append("| policy | n beliefs | coverage 90% | coverage 50% | mean posterior sd (log mu) | rmse (log mu) |")
        L.append("|---|---|---|---|---|---|")
        for p in ("P3", "P4"):
            if p in data[setup]:
                r = row(data[setup][p])
                if "bel_n" in r:
                    L.append(f"| {NAMES[p]} | {r['bel_n']} | {r['cover90']:.2f} | {r['cover50']:.2f} | {r['bel_sd']:.2f} | {r['bel_rmse']:.2f} |")
        L.append("\nBy hidden mu_roll band (all policies pooled over the carries): drop rate\n")
        bands = ((0.0005, 0.001), (0.001, 0.002), (0.002, 0.0036), (0.0036, 0.0061))
        L.append("| policy | " + " | ".join(f"{lo:.4f}-{hi:.4f}" for lo, hi in bands) + " |")
        L.append("|---|" + "---|" * len(bands))
        for p in ORDER:
            if p not in data[setup]:
                continue
            allr = [r for d in data[setup][p].values() for r in d.values()]
            cells = []
            for lo, hi in bands:
                s = [r for r in allr if lo <= r["mu"] < hi]
                cells.append(f"{np.mean([bool(r['dropped']) for r in s]):.3f} (n={len(s)})" if s else "-")
            L.append(f"| {NAMES[p]} | " + " | ".join(cells) + " |")
    return "\n".join(L)


# ====================================================================================== paired bootstrap
def paired(data_setup, a, b):
    """aligned arrays over the (vseed, ep) keys both policies have: drop_a, drop_b, t_a, t_b."""
    keys = sorted(set(k for s, d in data_setup[a].items() for k in ((s, e) for e in d)) &
                  set(k for s, d in data_setup[b].items() for k in ((s, e) for e in d)))
    A = [data_setup[a][s][e] for s, e in keys]
    B = [data_setup[b][s][e] for s, e in keys]
    return keys, np.array([r["dropped"] for r in A], bool), np.array([r["dropped"] for r in B], bool), \
        np.array([r["t_complete_s"] for r in A], float), np.array([r["t_complete_s"] for r in B], float)


def boot(data_setup, a, b, n_boot=B_N, seed=B_SEED):
    keys, da, db, ta, tb = paired(data_setup, a, b)
    n = len(keys)
    if n == 0:
        return None
    # resample CARRIES (episode ids), keeping the value seeds together: one draw = one bootstrap sample of carry ids
    eps = np.array([e for _, e in keys])
    uniq = np.unique(eps)
    idx_of = {u: np.flatnonzero(eps == u) for u in uniq}
    rng = np.random.default_rng(seed)
    dd, tr = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_of[u] for u in pick])
        dd.append(da[ii].mean() - db[ii].mean())
        ok = ~da[ii] & ~db[ii]
        tr.append(np.nanmean(ta[ii][ok]) / np.nanmean(tb[ii][ok]) if ok.any() else np.nan)
    ok0 = ~da & ~db
    q = lambda x, p: float(np.nanquantile(x, p))
    return dict(a=a, b=b, n=n, n_paired_ok=int(ok0.sum()), drop_a=float(da.mean()), drop_b=float(db.mean()), delta=float(da.mean() - db.mean()),
                delta_ci=(q(dd, 0.025), q(dd, 0.975)), t_a=float(np.nanmean(ta[ok0])) if ok0.any() else float("nan"),
                t_b=float(np.nanmean(tb[ok0])) if ok0.any() else float("nan"),
                ratio=float(np.nanmean(ta[ok0]) / np.nanmean(tb[ok0])) if ok0.any() else float("nan"), ratio_ci=(q(tr, 0.025), q(tr, 0.975)))


def primary(data, setup="piper_trk"):
    if setup not in data:
        return f"(no data for setup {setup})", {}
    D = data[setup]
    out, L = {}, [f"## Criteria on setup {setup} (value seeds pooled; paired bootstrap over carries, {B_N} draws)\n"]
    need = [p for p in ("P1", "P2", "P3") if p not in D]
    if need:
        return f"missing policies for the primary criterion: {need}", {}
    r31 = boot(D, "P3", "P1")
    r32 = boot(D, "P3", "P2")
    ni = r32["delta_ci"][1] <= MARGIN
    f1 = r31["ratio_ci"][1] <= 0.85
    f2 = r32["ratio_ci"][1] <= 0.95
    L.append("| contrast (A vs B) | drop A | drop B | Delta (A-B) 95% CI | non-inferior (upper <= 2 pp) | mean time A / B (paired, both not dropped) | ratio 95% CI | faster by required margin |")
    L.append("|---|---|---|---|---|---|---|---|")

    def line(r, req):
        ok_t = r["ratio_ci"][1] <= 1 - req if req is not None else None
        return (f"| {NAMES[r['a']]} vs {NAMES[r['b']]} (n={r['n']}, paired-ok {r['n_paired_ok']}) | {r['drop_a']:.3f} | {r['drop_b']:.3f} | "
                f"{r['delta']:+.3f} ({r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}) | {r['delta_ci'][1] <= MARGIN} | "
                f"{r['t_a']:.1f} / {r['t_b']:.1f} s | {r['ratio']:.3f} ({r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}) | "
                f"{'-' if req is None else ('YES (>= %d%%)' % round(100 * req) if ok_t else 'NO (needs >= %d%%)' % round(100 * req))} |")
    L += [line(r31, 0.15), line(r32, 0.05)]
    out["P3_vs_P1"], out["P3_vs_P2"] = r31, r32
    passed = bool(ni and f1 and f2)
    L.append(f"\n**PRIMARY: {'PASS' if passed else 'FAIL'}**  (P3 non-inferior to P2 on drops: {ni}; P3 >= 15% faster than P1: {f1}; P3 >= 5% faster than P2: {f2})")
    out["primary_pass"] = passed
    L.append("\nExtra contrasts (not in the primary criterion; they separate the value of the PROBE from the value of a feasible carry speed):\n")
    L.append("| contrast | drop A | drop B | Delta 95% CI | mean time A / B | ratio 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for a, b in (("P3", "P6"), ("P3", "P5"), ("P6", "P5"), ("P4", "P3"), ("P4", "P2"), ("P4", "P6")):
        if a in D and b in D:
            r = boot(D, a, b)
            if r:
                out[f"{a}_vs_{b}"] = r
                L.append(f"| {NAMES[a]} vs {NAMES[b]} (n={r['n']}) | {r['drop_a']:.3f} | {r['drop_b']:.3f} | {r['delta']:+.3f} ({r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}) | "
                         f"{r['t_a']:.1f} / {r['t_b']:.1f} s | {r['ratio']:.3f} ({r['ratio_ci'][0]:.3f}, {r['ratio_ci'][1]:.3f}) |")
    if all(p in D for p in ("P3", "P4")):
        r43 = out.get("P4_vs_P3")
        if r43:
            sec = r43["delta_ci"][1] <= MARGIN and r43["ratio_ci"][1] <= 1.0
            L.append(f"\n**SECONDARY (P4 vs P3): P4 non-inferior on drops (upper <= 2 pp) and not slower (upper ratio <= 1): {sec}**")
            out["secondary_pass"] = bool(sec)
    return "\n".join(L), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["table", "primary", "all"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--setup", default="piper_trk")
    a = ap.parse_args()
    data = load(a.out, a.tag, a.split)
    base = Path(a.out) / a.tag
    if a.cmd in ("table", "all"):
        txt = fmt_table(data)
        print(txt, flush=True)
        (base / f"table_{a.split}.md").write_text(txt + "\n")
    if a.cmd in ("primary", "all"):
        txt, js = primary(data, a.setup)
        print(txt, flush=True)
        (base / f"primary_{a.split}_{a.setup}.md").write_text(txt + "\n")
        (base / f"primary_{a.split}_{a.setup}.json").write_text(json.dumps(js, indent=1, default=float))


if __name__ == "__main__":
    main()
