"""analyze.py -- pre-registered K1-K4 verdicts, decision table, secondary analyses, plots.

    python -m world_model.v2.g3_killtest.analyze --main main_TS,main_P --mm mm
Writes summary.json, tables.md and PNGs into /common/users/ss5772/bimanual_data/g3_killtest/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.g3_killtest import config as K  # noqa: E402

OUT = K.OUT
B_BOOT = 2000
RNG = np.random.default_rng(12345)
PRIMARY_SIG = {c: K.CLASSES[c]["sigma_labels"][K.CLASSES[c]["primary_sigma"]] for c in K.CLASSES}
THETA_TAU = ["tvar", "tvar@0.25", "tvar@0.75", "eig", "eig@0.25", "eig@0.75"]
TABLE_POLS = ["oracle", "never", "always_robust", "always_probe", "g3", "g3_tau", "tvar", "eig", "g3_cost",
              "dg", "safeopt", "prmppi", "shield", "tvar_ug", "eig_ug"]
COL = {"T": "#2a78d6", "S": "#eb6834", "P": "#1baf7a"}


def load(tags):
    rows = []
    for t in tags:
        p = OUT / "results" / f"{t}.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                      # partial last line while a sweep is still writing
    return rows


def boot_ci(vals, fn=np.mean, B=B_BOOT):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(vals), (B, len(vals)))
    bs = np.array([fn(vals[i]) for i in idx])
    return (float(fn(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def cluster_boot(groups, stat, B=B_BOOT):
    """groups: list of per-episode arrays/tuples; stat(list_of_groups) -> float."""
    n = len(groups)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    pt = stat(groups)
    bs = []
    for _ in range(B):
        ii = RNG.integers(0, n, n)
        try:
            bs.append(stat([groups[i] for i in ii]))
        except Exception:
            pass
    bs = np.array([b for b in bs if np.isfinite(b)])
    return (float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fmt(ci, d=3):
    if ci is None or not np.isfinite(ci[0]):
        return "n/a"
    return f"{ci[0]:.{d}f} [{ci[1]:.{d}f}, {ci[2]:.{d}f}]"


# ------------------------------------------------------------------------------ calibration helpers
def calib_stats(pred, real, nbins=10):
    pred = np.asarray(pred, float)
    real = np.asarray(real, float)
    if len(pred) < 5 or np.std(pred) < 1e-12:
        return dict(slope=np.nan, intercept=np.nan, ece=np.nan, rho=np.nan, n=len(pred))
    A = np.vstack([pred, np.ones_like(pred)]).T
    slope, icpt = np.linalg.lstsq(A, real, rcond=None)[0]
    order = np.argsort(pred, kind="stable")
    bins = np.array_split(order, nbins)
    ece = sum(len(b) * abs(pred[b].mean() - real[b].mean()) for b in bins if len(b)) / len(pred)
    rho = spearmanr(pred, real).correlation
    return dict(slope=float(slope), intercept=float(icpt), ece=float(ece), rho=float(rho), n=int(len(pred)))


def reliability(pred, real, nbins=10):
    pred = np.asarray(pred, float)
    real = np.asarray(real, float)
    order = np.argsort(pred, kind="stable")
    bins = [b for b in np.array_split(order, nbins) if len(b)]
    return [(float(pred[b].mean()), float(real[b].mean()), float(real[b].std() / np.sqrt(len(b)))) for b in bins]


def pair_groups(rows, cls, sig, arm, key_pred, key_real, filt=None):
    """per-episode lists of (pred, real) calibration pairs"""
    g = []
    for r in rows:
        if r["cls"] != cls or r["sigma"] != sig or r["arm"] != arm:
            continue
        pr = []
        for c in r["calib"]:
            if filt is not None and not filt(c):
                continue
            p = key_pred(c)
            q = key_real(c)
            if p is None or q is None:
                continue
            pr.append((p, q))
        if pr:
            g.append(np.array(pr))
    return g


def cal_ci(groups):
    def f(field):
        return lambda gs: calib_stats(np.concatenate(gs)[:, 0], np.concatenate(gs)[:, 1])[field]
    allp = np.concatenate(groups) if groups else np.zeros((0, 2))
    st = calib_stats(allp[:, 0], allp[:, 1]) if len(allp) else {}
    out = dict(point=st)
    for field in ("slope", "ece", "rho"):
        out[field] = cluster_boot(groups, f(field), B=500)
    return out


# ------------------------------------------------------------------------------ main analyses
def recmap(r):
    return {x["policy"]: x for x in r["recs"]}


def analyze(rows_main, rows_mm):
    S = dict(prereg_sha256=(OUT / "PREREG.sha256").read_text().split()[0], classes={})
    md = []
    for cls in ("T", "S", "P"):
        R = [r for r in rows_main if r["cls"] == cls]
        if not R:
            continue
        C = dict()
        S["classes"][cls] = C
        psig = PRIMARY_SIG[cls]
        sigs = K.CLASSES[cls]["sigma_labels"]
        Rp = [r for r in R if r["sigma"] == psig]
        # ---------------- K2
        U0 = np.array([r["U0"] for r in Rp])
        C["K2"] = dict(n=len(U0), frac_U0_ge_gate=boot_ci(U0 >= 0.05), mean_U0=boot_ci(U0))
        C["K2"]["fires"] = bool(C["K2"]["frac_U0_ge_gate"][0] < 0.10 or C["K2"]["mean_U0"][0] < 0.05)
        # ---------------- K1 per sigma
        C["K1"] = {}
        for sg in sigs:
            Rs = [r for r in R if r["sigma"] == sg and r["U0"] >= 0.05]
            tol, strict, cost, cost_all = [], [], [], []
            for r in Rs:
                m = recmap(r)
                segs = m["never"]["segs"]
                tol.append(not any(s.get("undec_tol", False) for s in segs))
                strict.append(not any(s.get("undec", False) for s in segs))
            for r in [r for r in R if r["sigma"] == sg]:
                m = recmap(r)
                cost_all.append(m["never"]["ratio"] - m["oracle"]["ratio"])
                if r["U0"] >= 0.05:
                    cost.append(m["never"]["ratio"] - m["oracle"]["ratio"])
            k1 = boot_ci(np.array(tol, float))
            C["K1"][sg] = dict(n_nontrivial=len(Rs), nominal_decided_tol=k1,
                               nominal_decided_strict=boot_ci(np.array(strict, float)),
                               uncertainty_cost_nontrivial=boot_ci(cost), uncertainty_cost_all=boot_ci(cost_all),
                               fires=bool(k1[0] >= 0.90), clearly_not=bool(k1[2] < 0.90))
            # by route type
            bt = defaultdict(list)
            for r, t in zip(Rs, tol):
                bt[r["route_type"]].append(t)
            C["K1"][sg]["by_route_type"] = {k: (float(np.mean(v)), len(v)) for k, v in bt.items()}
        # ---------------- K4 + K3(A): calibration pairs, primary sigma, main arm
        C["K4"] = {}
        C["K3"] = {}
        for sg in sigs:
            g_dU = pair_groups(rows_main, cls, sg, "main", lambda c: c["pred"]["dU"], lambda c: c["real"]["dU"])
            C["K4"][sg] = cal_ci(g_dU)
            C["K4"][sg]["reliability"] = reliability(*np.concatenate(g_dU).T) if g_dU else []
            C["K4"][sg]["fires"] = bool(not (0.7 <= C["K4"][sg]["point"]["slope"] <= 1.3)
                                        or C["K4"][sg]["point"]["ece"] > 0.05)
            # by action type
            for at in ("a", "p"):
                ga = pair_groups(rows_main, cls, sg, "main", lambda c: c["pred"]["dU"], lambda c: c["real"]["dU"],
                                 filt=lambda c, at=at: c["action"] == at)
                if ga:
                    C["K4"][sg][f"point_{at}"] = calib_stats(*np.concatenate(ga).T)
        # K3 (A): spearman vs realized dU of each predictor, same pairs (primary sigma)
        def trip(c):
            return (c["pred"]["dU"], c["pred"]["r_tvar"], c["pred"]["r_eig"], c["pred"]["r_width"],
                    c["pred"]["so"]["1.0"], c["real"]["dU"])
        groups = []
        for r in Rp:
            arr = [trip(c) for c in r["calib"]]
            if arr:
                groups.append(np.array(arr, float))
        allp = np.concatenate(groups)

        def rho_col(j):
            return lambda gs: spearmanr(np.concatenate(gs)[:, j], np.concatenate(gs)[:, 5]).correlation

        def rho_gap(gs):
            a = np.concatenate(gs)
            rg = spearmanr(a[:, 0], a[:, 5]).correlation
            rb = max(spearmanr(a[:, 1], a[:, 5]).correlation, spearmanr(a[:, 2], a[:, 5]).correlation)
            return rb - rg
        C["K3"]["n_pairs"] = int(len(allp))
        C["K3"]["rho"] = {nm: cluster_boot(groups, rho_col(j), B=500)
                          for j, nm in enumerate(["g3_dU", "tvar", "eig", "width", "safeopt_L1"])}
        C["K3"]["rho_gap_best_theta_minus_g3"] = cluster_boot(groups, rho_gap, B=500)
        condA = bool(C["K3"]["rho_gap_best_theta_minus_g3"][0] >= -0.05)
        # K3 (B): decisions: g3_tau vs best theta-level tau-rule policy
        def pol_stats(pol, R_=Rp):
            rat = np.array([recmap(r)[pol]["ratio"] for r in R_])
            vio = np.array([recmap(r)[pol]["violated"] for r in R_], float)
            return rat, vio
        best, best_m = None, 1e9
        for p in THETA_TAU:
            rat, vio = pol_stats(p)
            if rat.mean() < best_m:
                best, best_m = p, rat.mean()
        rg, vg = pol_stats("g3_tau")
        rb, vb = pol_stats(best)
        ok = (vg == 0) & (vb == 0)
        dt = boot_ci(rb[ok] - rg[ok])
        dv = boot_ci(vb - vg)
        same_t = bool((dt[1] <= 0 <= dt[2]) or (dt[1] >= -0.02 and dt[2] <= 0.02))
        same_v = bool(dv[1] <= 0 <= dv[2])
        C["K3"]["decision"] = dict(best_theta_policy=best, dtime_best_minus_g3tau=dt, dviol_best_minus_g3tau=dv,
                                   n_both_safe=int(ok.sum()), same_time=same_t, same_viol=same_v)
        rv, vv = pol_stats("g3")
        ok2 = (vv == 0) & (vb == 0)
        C["K3"]["decision_voi"] = dict(dtime_best_theta_minus_g3voi=boot_ci(rb[ok2] - rv[ok2]),
                                       dviol=boot_ci(vb - vv))
        # action agreement g3_tau vs best
        agree, tot = 0, 0
        for r in Rp:
            m = recmap(r)
            a1 = [(d["k"], d["act"], d.get("b")) for d in m["g3_tau"]["decisions"]]
            a2 = [(d["k"], d["act"], d.get("b")) for d in m[best]["decisions"]]
            tot += 1
            agree += int(a1 == a2)
        C["K3"]["decision"]["identical_decision_sequences"] = agree / max(tot, 1)
        C["K3"]["condA"] = condA
        C["K3"]["condB"] = bool(same_t and same_v)
        C["K3"]["fires"] = bool(condA and same_t and same_v)
        # ---------------- decision table (all sigmas)
        C["table"] = {}
        for sg in sigs:
            Rs = [r for r in R if r["sigma"] == sg]
            tab = {}
            orc = np.array([recmap(r)["oracle"]["ratio"] for r in Rs])
            for p in TABLE_POLS:
                if p not in recmap(Rs[0]):
                    continue
                rat = np.array([recmap(r)[p]["ratio"] for r in Rs])
                vio = np.array([recmap(r)[p]["violated"] for r in Rs], float)
                prb = np.array([recmap(r)[p]["n_probes"] for r in Rs], float)
                safe = vio == 0
                sw = [sum(1 for s in recmap(r)[p]["segs"] if s.get("switched")) for r in Rs]
                tab[p] = dict(viol=boot_ci(vio), ratio=boot_ci(rat[safe]), regret=boot_ci((rat - orc)[safe]),
                              probes=float(prb.mean()), switches=float(np.mean(sw)), n=len(Rs))
            C["table"][sg] = tab
        # ---------------- POST-HOC (not pre-registered): probe break-even bound
        C["posthoc_breakeven"] = {}
        for sg in sigs:
            Rs = [r for r in R if r["sigma"] == sg]
            cost_s, g3gain, g3probed = [], [], []
            for r in Rs:
                m = recmap(r)
                if m["never"]["violated"] or m["oracle"]["violated"]:
                    continue
                cost_s.append(m["never"]["time"] - m["oracle"]["time"])
                if not m["g3"]["violated"]:
                    g3gain.append(m["never"]["time"] - m["g3"]["time"])
                    g3probed.append(m["g3"]["n_probes"] > 0)
            # VOI calibration (rough): episodes whose ONLY g3 probe was chosen at the first decision
            # (k=0, before any execution): predicted saving Ta-Tb vs realized never.time - g3.time
            vp, vr = [], []
            for r in Rs:
                m = recmap(r)
                dec = m["g3"]["decisions"]
                if (m["g3"]["n_probes"] == 1 and dec and dec[0]["act"] == "probe" and dec[0]["k"] == 0
                        and "Tb" in dec[0] and not m["g3"]["violated"] and not m["never"]["violated"]):
                    vp.append(dec[0]["Ta"] - dec[0]["Tb"])
                    vr.append(m["never"]["time"] - m["g3"]["time"])
            cost_s = np.array(cost_s)
            g3gain = np.array(g3gain)
            g3probed = np.array(g3probed, bool)
            C["posthoc_breakeven"][sg] = dict(
                uncertainty_cost_s=boot_ci(cost_s), frac_cost_gt_min_probe=float(np.mean(cost_s > 0.75)),
                min_probe_s=0.75, g3_time_saved_vs_never_s=boot_ci(g3gain),
                g3_time_saved_when_probed_s=boot_ci(g3gain[g3probed]) if g3probed.any() else None,
                g3_frac_probed=float(g3probed.mean()) if len(g3probed) else None,
                voi_first_probe_pred_saving_s=boot_ci(vp) if vp else None,
                voi_first_probe_real_saving_s=boot_ci(vr) if vr else None, voi_n=len(vp))
        # ---------------- S1 SafeOpt expander calibration vs L
        C["S1"] = {}
        for L in ("0.25", "0.5", "1.0", "2.0", "4.0", "8.0"):
            g = pair_groups(rows_main, cls, psig, "main", lambda c, L=L: c["pred"]["so"][L], lambda c: c["real"]["dU"])
            g2 = pair_groups(rows_main, cls, psig, "main", lambda c, L=L: c["pred"]["so"][L],
                             lambda c, L=L: c["real"]["so"][L])
            a = np.concatenate(g)
            a2 = np.concatenate(g2)
            C["S1"][L] = dict(vs_real_dU=calib_stats(a[:, 0], a[:, 1]), vs_own_frontier=calib_stats(a2[:, 0], a2[:, 1]),
                              mean_pred=float(a[:, 0].mean()), mean_real_dU=float(a[:, 1].mean()),
                              reliability=reliability(a[:, 0], a[:, 1]))
        # ---------------- S3 near-miss (never runs; T, S)
        C["S3"] = {}
        for sg in sigs:
            Rs = [r for r in R if r["sigma"] == sg]
            nseg = nmiss = ndet = ep_any = ep_det = 0
            for r in Rs:
                segs = recmap(r)["never"]["segs"]
                a_ = d_ = False
                for s in segs:
                    if "nm_moved" not in s:
                        continue
                    nseg += 1
                    if s["nm_moved"] and not s.get("nm_failed", False):
                        nmiss += 1
                        a_ = True
                        if s["nm_snr2"] >= 4:
                            ndet += 1
                            d_ = True
                ep_any += a_
                ep_det += d_
            # realized in-time dU of (a) pairs (never runs) with / without near-miss
            wm, wo = [], []
            for r in Rs:
                for c in r["calib"]:
                    if c["src"] != "never" or c["action"] != "a":
                        continue
                    (wm if (c["real"]["moved"] and c["real"]["snr2"] >= 4) else wo).append(c["real"]["dU"])
            tot = sum(wm) + sum(wo)
            C["S3"][sg] = dict(frac_segments_nearmiss=nmiss / max(nseg, 1), frac_segments_detectable=ndet / max(nseg, 1),
                               frac_episodes_any_nearmiss=ep_any / max(len(Rs), 1),
                               frac_episodes_detectable_nearmiss=ep_det / max(len(Rs), 1),
                               dU_a_with_detectable_nearmiss=boot_ci(wm), dU_a_without=boot_ci(wo),
                               n_with=len(wm), n_without=len(wo),
                               share_of_nominal_dU_from_nearmiss=(sum(wm) / tot) if tot > 0 else np.nan)
        # ---------------- S5 theta-width vs U divergence (primary sigma)
        W_noU = U_noW = both = n = 0
        per_act = defaultdict(lambda: [0, 0, 0])
        for r in Rp:
            for c in r["calib"]:
                U = c["U"]
                if U <= 0:
                    continue
                if c["width0"] < 1e-9:      # posterior already collapsed to one grid point: relative width undefined
                    continue
                n += 1
                rw = c["real"]["r_width"]
                ru = c["real"]["dU"] / U
                a = rw >= 0.25 and ru < 0.10
                b = ru >= 0.25 and rw < 0.10
                W_noU += a
                U_noW += b
                both += (rw >= 0.25 and ru >= 0.25)
                pa = per_act[c["action"]]
                pa[0] += 1
                pa[1] += a
                pa[2] += b
        C["S5"] = dict(n=n, width_falls_U_not=W_noU / max(n, 1), U_falls_width_not=U_noW / max(n, 1),
                       both_fall=both / max(n, 1),
                       by_action={k: dict(n=v[0], width_falls_U_not=v[1] / max(v[0], 1),
                                          U_falls_width_not=v[2] / max(v[0], 1)) for k, v in per_act.items()})
        # ---------------- S6 mismatch arm: calibration of every predictor, both arms
        C["S6"] = {}
        for arm, rows in (("main", rows_main), ("mismatch", rows_mm)):
            if not rows:
                continue
            d = {}
            for nm, kp, kr in (("g3_dU_vs_real_dU", lambda c: c["pred"]["dU"], lambda c: c["real"]["dU"]),
                               ("tvar_vs_real_tvar", lambda c: c["pred"]["r_tvar"], lambda c: c["real"]["r_tvar"]),
                               ("width_vs_real_width", lambda c: c["pred"]["r_width"], lambda c: c["real"]["r_width"]),
                               ("eig_vs_real_dH", lambda c: c["pred"]["eig"], lambda c: c["real"]["dH"]),
                               ("tvar_vs_real_dU", lambda c: c["pred"]["r_tvar"], lambda c: c["real"]["dU"]),
                               ("eig_vs_real_dU", lambda c: c["pred"]["r_eig"], lambda c: c["real"]["dU"])):
                flt = None
                if nm.startswith("width"):
                    flt = lambda c: c["width0"] > 1e-9
                elif nm.startswith("tvar"):
                    flt = lambda c: c["var0"] > 1e-9
                g = pair_groups(rows, cls, psig, arm, kp, kr, filt=flt)
                if g:
                    a = np.concatenate(g)
                    d[nm] = calib_stats(a[:, 0], a[:, 1])
                    d[nm]["reliability"] = reliability(a[:, 0], a[:, 1])
            if arm == "mismatch":
                Rm = [r for r in rows if r["cls"] == cls]
                d["table"] = {}
                for p in TABLE_POLS:
                    if Rm and p in recmap(Rm[0]):
                        vio = np.array([recmap(r)[p]["violated"] for r in Rm], float)
                        rat = np.array([recmap(r)[p]["ratio"] for r in Rm])
                        d["table"][p] = dict(viol=boot_ci(vio), ratio=boot_ci(rat[vio == 0]), n=len(Rm))
            C["S6"][arm] = d
    return S


# ------------------------------------------------------------------------------ plots
def plots(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "grid.color": "#e4e3df"})
    # Fig 1: G3 reliability per class (primary sigma, main vs mismatch)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, cls in zip(axs, [c for c in ("T", "S", "P") if c in S["classes"]]):
        C = S["classes"][cls]
        psig = PRIMARY_SIG[cls]
        lim = 0
        for arm, col, mk in (("main", COL[cls], "o"), ("mismatch", "#52514e", "s")):
            rel = C["S6"].get(arm, {}).get("g3_dU_vs_real_dU", {}).get("reliability", [])
            if not rel:
                continue
            x, y, e = np.array(rel).T
            lim = max(lim, x.max(), y.max())
            st = C["S6"][arm]["g3_dU_vs_real_dU"]
            ax.errorbar(x, y, yerr=1.96 * e, color=col, marker=mk, ms=4, lw=1.5, capsize=2,
                        label=f"{arm}: slope {st['slope']:.2f}, ECE {st['ece']:.3f}")
        lim = max(lim * 1.08, 0.05)
        ax.plot([0, lim], [0, lim], color="#8a8984", lw=1, ls="--", label="perfect calibration")
        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.05, lim)
        ax.set_title(f"Class {cls} ({psig})", loc="left")
        ax.set_xlabel("predicted ΔU (G3 preposterior)")
        ax.grid(True, lw=0.5)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axs[0].set_ylabel("realized ΔU (bin mean ± 95%)")
    fig.suptitle("G3 value-level predictor: calibration against realized in-time shrinkage of U", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "calib_g3.png", dpi=150)
    plt.close(fig)
    # Fig 2: execution-aware calibration of theta-level predictors (main arm)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (nm, lab) in zip(axs, (("tvar_vs_real_tvar", "θ-variance reduction (DG-style)"),
                                   ("width_vs_real_width", "θ-width reduction"),
                                   ("eig_vs_real_dH", "EIG vs realized entropy drop [nats]"))):
        lim_lo, lim_hi = 0, 0
        for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
            rel = S["classes"][cls]["S6"]["main"].get(nm, {}).get("reliability", [])
            if not rel:
                continue
            x, y, e = np.array(rel).T
            lim_hi = max(lim_hi, x.max(), y.max())
            lim_lo = min(lim_lo, x.min(), y.min())
            st = S["classes"][cls]["S6"]["main"][nm]
            ax.errorbar(x, y, yerr=1.96 * e, color=COL[cls], marker="o", ms=4, lw=1.5, capsize=2,
                        label=f"{cls}: slope {st['slope']:.2f}")
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#8a8984", lw=1, ls="--")
        ax.set_title(lab, loc="left")
        ax.set_xlabel("predicted")
        ax.grid(True, lw=0.5)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axs[0].set_ylabel("realized (bin mean ± 95%)")
    fig.suptitle("θ-level predictors against their own realized quantities (main arm, primary σ)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "calib_theta_level.png", dpi=150)
    plt.close(fig)
    # Fig 3: SafeOpt expander calibration slope vs L
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    Ls = ["0.25", "0.5", "1.0", "2.0", "4.0", "8.0"]
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        s1 = S["classes"][cls]["S1"]
        axs[0].plot([float(L) for L in Ls], [s1[L]["mean_pred"] for L in Ls], color=COL[cls], marker="o", lw=2,
                    label=f"{cls} predicted")
        axs[0].axhline(s1["1.0"]["mean_real_dU"], color=COL[cls], lw=1, ls=":")
        axs[1].plot([float(L) for L in Ls], [s1[L]["vs_real_dU"]["rho"] for L in Ls], color=COL[cls], marker="o",
                    lw=2, label=cls)
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_title("SafeOpt expander: mean predicted ΔU (dotted = realized)", loc="left", fontsize=9)
    axs[1].set_title("Spearman ρ with realized ΔU", loc="left", fontsize=9)
    for ax in axs:
        ax.set_xlabel("Lipschitz L [margin per m/s²]")
        ax.grid(True, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "safeopt_L_sensitivity.png", dpi=150)
    plt.close(fig)


def tables_md(S):
    L = []
    L.append("## Verdicts (primary σ)\n")
    L.append("| class | K2 frac U0≥0.05 | K2 mean U0 | K1 nominal-decided (tol) | K1 strict | K3 ρ gap (bestθ−G3) | "
             "K3 Δtime best-θ − g3_tau | K4 slope | K4 ECE | K4 ρ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        C = S["classes"][cls]
        ps = PRIMARY_SIG[cls]
        k1 = C["K1"][ps]
        k4 = C["K4"][ps]
        L.append(f"| {cls} | {fmt(C['K2']['frac_U0_ge_gate'], 2)} | {fmt(C['K2']['mean_U0'], 2)} | "
                 f"{fmt(k1['nominal_decided_tol'], 2)} (n={k1['n_nontrivial']}) | {fmt(k1['nominal_decided_strict'], 2)} | "
                 f"{fmt(C['K3']['rho_gap_best_theta_minus_g3'], 3)} | {fmt(C['K3']['decision']['dtime_best_minus_g3tau'], 3)} "
                 f"({C['K3']['decision']['best_theta_policy']}) | {fmt(k4['slope'], 2)} | {fmt(k4['ece'], 3)} | "
                 f"{fmt(k4['rho'], 2)} |")
    L.append("\n| class | K2 fires | K1 fires | K3 condA | K3 condB | K3 fires | K4 fires |")
    L.append("|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        C = S["classes"][cls]
        ps = PRIMARY_SIG[cls]
        L.append(f"| {cls} | {C['K2']['fires']} | {C['K1'][ps]['fires']} | {C['K3']['condA']} | {C['K3']['condB']} | "
                 f"{C['K3']['fires']} | {C['K4'][ps]['fires']} |")
    L.append("\n## K1 and K4 across σ\n")
    L.append("| class | σ | K1 nominal-decided (tol) | K1 strict | uncertainty cost never−oracle (nontrivial) | "
             "K4 slope | K4 ECE | K4 ρ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        C = S["classes"][cls]
        for sg in K.CLASSES[cls]["sigma_labels"]:
            k1 = C["K1"][sg]
            k4 = C["K4"][sg]
            L.append(f"| {cls} | {sg} | {fmt(k1['nominal_decided_tol'], 2)} | {fmt(k1['nominal_decided_strict'], 2)} | "
                     f"{fmt(k1['uncertainty_cost_nontrivial'], 3)} | {fmt(k4['slope'], 2)} | {fmt(k4['ece'], 3)} | "
                     f"{fmt(k4['rho'], 2)} |")
    L.append("\n## Spearman ρ with realized ΔU (primary σ, same pairs)\n")
    L.append("| class | n pairs | G3 ΔU | θ-var (DG-style) | EIG | θ-width | SafeOpt L=1 |")
    L.append("|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        C = S["classes"][cls]["K3"]
        r = C["rho"]
        L.append(f"| {cls} | {C['n_pairs']} | {fmt(r['g3_dU'], 2)} | {fmt(r['tvar'], 2)} | {fmt(r['eig'], 2)} | "
                 f"{fmt(r['width'], 2)} | {fmt(r['safeopt_L1'], 2)} |")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        ps = PRIMARY_SIG[cls]
        L.append(f"\n## Decision-policy table: class {cls}, σ={ps} (N=200; ratio/regret over non-violating episodes)\n")
        L.append("| policy | violation rate | time ratio vs nominal | regret vs oracle | probes/ep | switches/ep |")
        L.append("|---|---|---|---|---|---|")
        for p, v in S["classes"][cls]["table"][ps].items():
            L.append(f"| {p} | {fmt(v['viol'], 3)} | {fmt(v['ratio'], 3)} | {fmt(v['regret'], 3)} | "
                     f"{v['probes']:.2f} | {v['switches']:.2f} |")
    L.append("\n## POST-HOC (not pre-registered): probe break-even bound\n")
    L.append("| class | σ | uncertainty cost never−oracle [s] | frac episodes cost > 0.75 s (cheapest probe) | "
             "g3 time saved vs never [s] | g3 saved when it probed [s] | g3 frac episodes probed | "
             "VOI predicted saving, single k=0 probe [s] | realized saving [s] |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        for sg in K.CLASSES[cls]["sigma_labels"]:
            b = S["classes"][cls]["posthoc_breakeven"][sg]
            L.append(f"| {cls} | {sg} | {fmt(b['uncertainty_cost_s'], 3)} | {b['frac_cost_gt_min_probe']:.3f} | "
                     f"{fmt(b['g3_time_saved_vs_never_s'], 3)} | {fmt(b['g3_time_saved_when_probed_s'], 3)} | "
                     f"{b['g3_frac_probed']:.3f} | {fmt(b['voi_first_probe_pred_saving_s'], 3)} (n={b['voi_n']}) | "
                     f"{fmt(b['voi_first_probe_real_saving_s'], 3)} |")
    L.append("\n## S3 near-miss observability (never policy)\n")
    L.append("| class | σ | segments w/ near-miss | detectable (SNR²≥4) | episodes w/ detectable | ΔU(a) with det. near-miss | "
             "ΔU(a) without | share of nominal ΔU |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        for sg in K.CLASSES[cls]["sigma_labels"]:
            s3 = S["classes"][cls]["S3"][sg]
            L.append(f"| {cls} | {sg} | {s3['frac_segments_nearmiss']:.3f} | {s3['frac_segments_detectable']:.3f} | "
                     f"{s3['frac_episodes_detectable_nearmiss']:.3f} | {fmt(s3['dU_a_with_detectable_nearmiss'], 3)} "
                     f"(n={s3['n_with']}) | {fmt(s3['dU_a_without'], 3)} (n={s3['n_without']}) | "
                     f"{s3['share_of_nominal_dU_from_nearmiss']:.2f} |")
    L.append("\n## S5 θ-width vs U divergence (realized, primary σ)\n")
    L.append("| class | n pairs | width falls ≥25%, U falls <10% | U falls ≥25%, width falls <10% | both fall ≥25% |")
    L.append("|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        s5 = S["classes"][cls]["S5"]
        L.append(f"| {cls} | {s5['n']} | {s5['width_falls_U_not']:.3f} | {s5['U_falls_width_not']:.3f} | {s5['both_fall']:.3f} |")
    L.append("\n## S1 SafeOpt expander vs realized ΔU (primary σ)\n")
    L.append("| class | L | mean pred | mean real ΔU | slope | ECE | ρ | ρ vs own frontier |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        for Lk, v in S["classes"][cls]["S1"].items():
            L.append(f"| {cls} | {Lk} | {v['mean_pred']:.3f} | {v['mean_real_dU']:.3f} | {v['vs_real_dU']['slope']:.2f} | "
                     f"{v['vs_real_dU']['ece']:.3f} | {v['vs_real_dU']['rho']:.2f} | {v['vs_own_frontier']['rho']:.2f} |")
    L.append("\n## S6 calibration under planned-vs-executed mismatch (primary σ)\n")
    L.append("| class | predictor vs target | main slope / ECE / ρ | mismatch slope / ECE / ρ |")
    L.append("|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        s6 = S["classes"][cls]["S6"]
        for nm in ("g3_dU_vs_real_dU", "tvar_vs_real_tvar", "width_vs_real_width", "eig_vs_real_dH",
                   "tvar_vs_real_dU", "eig_vs_real_dU"):
            a = s6.get("main", {}).get(nm)
            b = s6.get("mismatch", {}).get(nm)

            def f3(x):
                return f"{x['slope']:.2f} / {x['ece']:.3f} / {x['rho']:.2f}" if x else "n/a"
            L.append(f"| {cls} | {nm} | {f3(a)} | {f3(b)} |")
    L.append("\n## S6 mismatch-arm policy outcomes (primary σ)\n")
    L.append("| class | policy | violation rate | time ratio |")
    L.append("|---|---|---|---|")
    for cls in [c for c in ("T", "S", "P") if c in S["classes"]]:
        for p, v in S["classes"][cls]["S6"].get("mismatch", {}).get("table", {}).items():
            L.append(f"| {cls} | {p} | {fmt(v['viol'], 3)} | {fmt(v['ratio'], 3)} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main_TS,main_P")
    ap.add_argument("--mm", default="mm")
    a = ap.parse_args()
    rows_main = load(a.main.split(","))
    rows_mm = load(a.mm.split(",")) if a.mm and (OUT / "results" / f"{a.mm.split(',')[0]}.jsonl").exists() else []
    S = analyze(rows_main, rows_mm)
    S["n_rows_main"] = len(rows_main)
    S["n_rows_mm"] = len(rows_mm)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    (OUT / "summary.json").write_text(json.dumps(clean(S), indent=1))
    (OUT / "tables.md").write_text(tables_md(S))
    plots(S)
    print(tables_md(S))


if __name__ == "__main__":
    main()
