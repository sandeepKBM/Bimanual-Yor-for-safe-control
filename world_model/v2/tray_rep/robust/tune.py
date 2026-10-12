"""robust/tune.py -- TEST 2 plumbing: match the conservatism of the five filters on VAL, then run them on TEST.
SIMULATION / SAVED DATA ONLY (MuJoCo CPU + torch CPU; no hardware, no network).

One CELL = (setup, arm, model seed, condition). `tune` does, on VAL carries (--n-val, default 100):
  1. rho*      closed-loop intervention rate (interventions / ticks) of the PLAIN filter at the bridge's VAL-chosen tau.
  2. profile   each filter run open loop (thr = -inf, it never intervenes) on the nominal carries under the condition, recording its
               score every tick (and the WM residuals for ACI). Pooled scores give the threshold that would intervene on a
               fraction q of ticks (open loop).
  3. match     for conf / epi / aci / rob: start at q = rho*, run the filter closed loop on VAL, move q by q <- q rho*/rho
               (at most 3 moves), keep the run whose rho is closest to rho*. thr = the pooled-score quantile at q.
  4. scale     rob's set scale = the 90% coverage factor of the tracker set on VAL (calibrate_scale: the factor s such that
               |true ball - observed ball| <= s (sigma_p + extra) for 90% of ticks, over all ticks and over dropout ticks, the larger). ACI's calibration pool = the profile residuals.
  5. conf eps  if Test-1 VAL records exist: tau_c(eps) with P(fail | V >= tau) <= eps (test1.conformal_tau), run once for the record.
Output <OUT>/test2/chosen/<setup>__<arm>_s<seed>__<cond>.json  (specs + achieved rates). Epistemic (iii) always uses the 3 seed models; it is tuned and run in every seed cell (the cell seed only shifts the corruption stream).
`test` runs the chosen specs (and F0) on the TEST carries -> <OUT>/test2/main/...

  python -m world_model.v2.tray_rep.robust.tune tune --setup piper_trk --arm Rc3k --seed 0 --cond trkB10f --n-val 100
  python -m world_model.v2.tray_rep.robust.tune test --setup piper_trk --arm Rc3k --seed 0 --cond trkB10f --workers 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.robust import corrupt as CR
from world_model.v2.tray_rep.robust import rfilters as RF
from world_model.v2.tray_rep.robust import run2
from world_model.v2.tray_rep.robust import util as RU

KINDS = ("plain", "conf", "epi", "aci", "rob")
PROFILE_THR = -1e9


def cell_name(setup, arm, seed, cond):
    return f"{setup}__{arm}_s{int(seed)}__{cond}"


def chosen_path(out_root, tag, setup, arm, seed, cond):
    return Path(out_root) / "test2" / "chosen" / f"{tag}__{cell_name(setup, arm, seed, cond)}.json"


def calib_path(out_root, setup, arm, seed, cond):
    return Path(out_root) / "calib" / f"{cell_name(setup, arm, seed, cond)}_resid.npz"


# ===================================================================================== set calibration
def calibrate_scale(setup, cond_name, n_eps=100, q=0.9, split="val", run_seed=0, tset_kw=None):
    """coverage scale of the tracker set (split-conformal style). Runs the Corruptor over the nominal TRUE ball of `n_eps` VAL
    carries (bursts as in the condition), only while the ball is on the tray. The set radius is scale * (sigma_p + extra) (kappa = 1).
    scale = the q-quantile of |true ball - observed ball| / (sigma_p + extra), taken separately over ALL ticks and over DROPOUT-phase
    ticks; the larger of the two is returned, so the set covers q of the ticks in both regimes."""
    st = Store(RU.data_dir(setup), split)
    cond = CR.parse_cond(cond_name)
    from world_model.v2.tray_rep.robust.episode import carry_key
    r_all, r_drop, errs = [], [], []
    for ep in range(min(n_eps, st.n_eps)):
        info0 = st.ep_info(ep)
        priv = st.rows("priv", ep, 0, info0["lens"])
        key = carry_key(info0["seed"], run_seed)
        bs = CR.burst_start(cond, key, t_fail=info0["onset"], T=info0["lens"])
        cr = CR.Corruptor(cond, key, (bs, cond.blen) if bs is not None else None)
        ts = CR.TrackerSet(cond, **(dict(tset_kw or {}, scale=1.0, kappa=1.0)))
        inf = cr.reset(priv[0, 0:2], priv[0, 3:5])
        kmax = info0["onset"] if info0["onset"] >= 0 else len(priv)          # ticks while the ball is still on the tray
        for k in range(1, len(priv)):
            inf = cr.update(priv[k, 0:2], priv[k, 3:5])
            ts.step(inf)
            if k >= kmax:
                continue
            e = float(np.hypot(*(priv[k, 0:2] - inf["ball"][0:2])))
            r = e / ts.radius(inf["age"])
            errs.append(e)
            r_all.append(r)
            if inf["phase"] == CR.PH_DROP:
                r_drop.append(r)
    qa = float(np.quantile(r_all, q))
    qd = float(np.quantile(r_drop, q)) if len(r_drop) >= 30 else float("nan")
    return dict(scale=float(max(qa, 0.0 if np.isnan(qd) else qd)), scale_all=qa, scale_dropout=qd, n_dropout_ticks=len(r_drop), q=q,
                n=len(r_all), err_p50=float(np.median(errs)), err_p90=float(np.quantile(errs, 0.9)), err_p99=float(np.quantile(errs, 0.99)))


# ===================================================================================== tuning
def _run_val(spec, setup, seed, cond, n_val, out_root, record=False, tag="tune", workers=1):
    summ, recs = run2.run(spec, setup, seed, cond, "val", n_val, workers, tag, out_root, None, record, quiet=True)
    return summ, recs


def profile_scores(kind, arm, setup, seed, cond, n_val, out_root, extra="", workers=1):
    spec = RF.spec_str(kind, arm, PROFILE_THR) + extra
    summ, recs = _run_val(spec, setup, seed, cond, n_val, out_root, record=True, tag="profile", workers=workers)
    sc = np.concatenate([np.array(r["x_scores"]) for r in recs])
    rs = np.concatenate([np.array(r.get("x_resid", [])) for r in recs]) if kind == "aci" else None
    return sc, rs


def match(kind, arm, setup, seed, cond, rho_star, scores, n_val, out_root, extra="", max_moves=3, tol=0.1, workers=1, log=RU.log):
    q = float(rho_star)
    best, trace = None, []
    for it in range(max_moves + 1):
        qq = min(max(q, 1e-4), 0.95)
        thr = float(np.quantile(scores, qq))
        thr = max(thr, RF.NEG_INF)
        spec = RF.spec_str(kind, arm, thr) + extra
        summ, _ = _run_val(spec, setup, seed, cond, n_val, out_root, workers=workers)
        rho, drop = float(summ["interv_rate"]), float(summ["drop"]["rate"])
        rel = abs(rho - rho_star) / max(rho_star, 1e-6)
        trace.append(dict(q=qq, thr=thr, rho=rho, drop=drop, rel_err=rel))
        log(f"  [match {kind}] it={it} q={qq:.4f} thr={thr:.4f} rho={rho:.4f} (target {rho_star:.4f}) drop={drop:.3f}")
        if best is None or rel < best["rel_err"]:
            best = dict(spec=spec, thr=thr, rho=rho, drop_val=drop, rel_err=rel)
        if rel <= tol or thr <= RF.NEG_INF + 1:
            break
        pts = [(t["q"], t["rho"]) for t in trace if t["rho"] > 0]
        lo = [p for p in pts if p[1] < rho_star]
        hi = [p for p in pts if p[1] > rho_star]
        if lo and hi:                                     # bracketed: interpolate log q against log rho between the closest pair
            (q0, r0), (q1, r1) = max(lo, key=lambda p: p[1]), min(hi, key=lambda p: p[1])
            w = (np.log(rho_star) - np.log(r0)) / (np.log(r1) - np.log(r0))
            q = float(np.exp(np.log(q0) + w * (np.log(q1) - np.log(q0))))
        else:
            q = qq * (rho_star / rho) if rho > 0 else qq * 2.0
    best["trace"] = trace
    best["rho_target"] = rho_star
    return best


def tune_cell(setup, arm, seed, cond, n_val=100, out_root=RU.OUT, tag="main", kinds=KINDS, cover_q=0.9, workers=1,
              conf_eps=0.1, test1_val=None, alpha=0.1, gamma=0.02, stale="skip", log=RU.log):
    out_root = Path(out_root)
    cp = chosen_path(out_root, tag, setup, arm, seed, cond)
    ch = json.loads(cp.read_text()) if cp.exists() else {}
    log(f"[tune] cell {cell_name(setup, arm, seed, cond)} n_val={n_val} kinds={kinds}")
    # 1. rho*
    tau_ref = RU.chosen_tau(setup, arm)
    if "plain" not in ch:
        spec = RF.spec_str("plain", arm, tau_ref)
        summ, _ = _run_val(spec, setup, seed, cond, n_val, out_root, workers=workers)
        ch["plain"] = dict(spec=spec, thr=tau_ref, rho=float(summ["interv_rate"]), drop_val=float(summ["drop"]["rate"]),
                           note="bridge VAL-chosen tau (not re-tuned)")
    rho_star = ch["plain"]["rho"]
    log(f"  rho* = {rho_star:.4f} (plain at tau {tau_ref})  VAL drop {ch['plain']['drop_val']:.3f}")
    ch["rho_star"] = rho_star
    # 4. kappa (rob) and the ACI pool
    if "scale" not in ch:
        ch["scale"] = calibrate_scale(setup, cond, min(n_val, 100), cover_q)
    scale = ch["scale"]["scale"]
    for kind in kinds:
        if kind == "plain" or kind in ch and "spec" in ch[kind]:
            continue
        extra = ""
        if kind == "aci":
            extra = f",alpha={alpha:g},gamma={gamma:g},stale={stale}"
        if kind == "rob":
            extra = f",scale={scale:.4g}"
        sc, rs = profile_scores(kind, arm, setup, seed, cond, n_val, out_root, extra)
        if kind == "aci":
            rp = calib_path(out_root, setup, arm, seed, cond)
            rp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(rp, resid=rs[rs > 0])
            extra += f",calib={rp}"
            sc, _ = profile_scores(kind, arm, setup, seed, cond, n_val, out_root, extra)      # with the pool in place
        ch[kind] = match(kind, arm, setup, seed, cond, rho_star, sc, n_val, out_root, extra, workers=workers, log=log)
        RU.atomic_json(ch, cp)
    # 5. conformal tau at eps (reported, not used for the matched comparison)
    if test1_val and Path(test1_val).exists():
        from world_model.v2.tray_rep.robust import test1 as T1
        recs = RU.read_jsonl(test1_val)
        tau_c, n_safe, r = T1.conformal_tau(recs, conf_eps)
        spec = RF.spec_str("conf", arm, tau_c)
        summ, _ = _run_val(spec, setup, seed, cond, n_val, out_root, workers=workers)
        ch["conf_eps"] = dict(eps=conf_eps, tau=tau_c, n_safe=n_safe, emp_rate=r, spec=spec, rho=float(summ["interv_rate"]),
                              drop_val=float(summ["drop"]["rate"]))
    RU.atomic_json(ch, cp)
    log(f"[tune] wrote {cp}")
    return ch


def test_cell(setup, arm, seed, cond, n_test=None, out_root=RU.OUT, tag="main", kinds=KINDS + ("f0",), workers=1, split="test"):
    out_root = Path(out_root)
    ch = json.loads(chosen_path(out_root, tag, setup, arm, seed, cond).read_text())
    res = {}
    for kind in kinds:
        if kind == "f0":
            spec = RF.spec_str("plain", arm, PROFILE_THR)
        elif kind not in ch:
            continue
        else:
            spec = ch[kind]["spec"]
        summ, _ = run2.run(spec, setup, seed, cond, split, n_test, workers, tag, out_root, None, False)
        res[kind] = summ
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tune", "test", "scale"])
    ap.add_argument("--setup", required=True, choices=RU.SETUPS)
    ap.add_argument("--arm", default="Rc3k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="trk")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(RU.OUT))
    ap.add_argument("--test1-val", default="")
    a = ap.parse_args()
    kinds = tuple(a.kinds.split(","))
    if a.cmd == "scale":
        print(json.dumps(calibrate_scale(a.setup, a.cond, a.n_val), indent=1))
    elif a.cmd == "tune":
        tune_cell(a.setup, a.arm, a.seed, a.cond, a.n_val, a.out, a.tag, kinds, workers=a.workers, test1_val=a.test1_val or None)
    else:
        test_cell(a.setup, a.arm, a.seed, a.cond, a.n_test, a.out, a.tag, kinds + ("f0",) if "f0" not in kinds else kinds,
                  a.workers)


if __name__ == "__main__":
    main()
