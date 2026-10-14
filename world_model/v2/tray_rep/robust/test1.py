"""robust/test1.py -- TEST 1: the false-safe map. SIMULATION / SAVED DATA ONLY (MuJoCo CPU + torch CPU; no hardware, no network).

For each certified-savable state (oracle label 1) and each hold-ok state (label 0) of a split, under one corruption condition
and one trained model (setup, arm, seed):
  1. rebuild the observation stream of ticks 0..t: tray part = stored (exact), ball part = corrupt(TRUE ball from `priv`)
     with the `Corruptor` (so the stored tracker noise of the trk data is NOT reused; the corruption is the condition's own);
  2. filter the WM latent on it, read V(z) (also: max_u Q, and the tracker-set robust value v_rob = min of V over the 9 support
     points of the tracker set);
  3. restore MuJoCo from the snapshot and run the monitor's own FALLBACK closed loop for T = 50 ticks (cl_eval_mj semantics:
     u = argmax_u V(f(z, u)) over the scaled 85-set; the observations it sees keep being corrupted; fail = ball centre off the tray);
  4. log distance to the nearest tray edge d0 (oracle), ball speed, the phase at the decision tick (steady / dropout / reacq).
False-safe = (V >= tau) AND (fallback fails), among certified-savable states. The report bins it by d0.

  python -m world_model.v2.tray_rep.robust.test1 run --setup piper_trk --arm N0 --seed 0 --cond trk --n-sav 20 --n-hold 10
  python -m world_model.v2.tray_rep.robust.test1 report --tag smoke
Output: <OUT>/test1/<tag>/<split>/<setup>/<arm>_s<seed>__<cond>.jsonl (one line per state; rerun = resume).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.oracle import load_oracle, SAVABLE, HOLD_OK
from world_model.v2.tray_rep.robust import corrupt as CR
from world_model.v2.tray_rep.robust import lat as LT
from world_model.v2.tray_rep.robust import util as RU

T_CL = 50
DBINS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.11)         # distance-to-edge bins [m]; inner = [0, 0.02), outer = [0.06, 0.11)
G = {}


# ===================================================================================== population
def population(oracle, n_sav=None, n_hold=None):
    """indices into the oracle arrays: every certified-savable state and every hold-ok state (or an evenly spaced subset)."""
    def pick(lab, n):
        sel = np.flatnonzero(oracle["label"] == lab)
        if n is not None and n < len(sel):
            sel = sel[np.linspace(0, len(sel) - 1, n).astype(int)]
        return sel
    return np.concatenate([pick(SAVABLE, n_sav), pick(HOLD_OK, n_hold)])


def oracle_path(setup, split):
    return RU.data_dir(setup) / "oracle" / "test_v0.npz" if split == "test" else RU.OUT / "oracle" / f"{split}_v0.npz"


# ===================================================================================== one state
def state_key(e, t, run_seed):
    return (int(e) * 100003 + int(t) * 101 + int(run_seed) * 7919 + 17) & 0x7FFFFFFF


def run_state(renv, st, orc, j, lt, cond, run_seed=0, T=T_CL, tset_kw=None):
    e, t = int(orc["ep"][j]), int(orc["t"][j])
    snap, hid = st.snapshot_at(e, t)
    renv.restore(snap, mu_roll=hid)
    priv = st.rows("priv", e, 0, t + 1)
    obs_x = st.rows("obs", e, 0, t + 1)
    key = state_key(e, t, run_seed)
    hold_ftk = int(orc["hold_ftk"][j])
    bs = CR.burst_start(cond, key, t_state=t, t_fail=(t + hold_ftk if hold_ftk >= 0 else -1), T=t + T)
    burst = (bs, cond.blen) if bs is not None else None
    ball, infos, cr = CR.corrupt_prefix(cond, key, burst, priv)
    ts = CR.TrackerSet(cond, **(tset_kw or {}))
    for i in infos:
        ts.step(i)
    obs = np.zeros((t + 1 + T, 13), np.float32)
    act = np.zeros((t + T, 6), np.float32)
    obs[:t + 1, :9] = obs_x[:, :9]
    obs[:t + 1, 9:] = ball
    act[:t] = st.rows("act", e, 0, t)
    n = t + 1
    lt.begin()
    lt.sync(obs[:n], act[:n - 1])
    info = infos[-1]
    sup = ts.support(info["ball"], info["age"])
    ov = lt.ball_variants(sup)
    v_set = lt.v_state(ov)
    q_set = lt.q(ov)                                        # (9, 85)
    q0 = q_set[0]
    sig_p, _ = ts.sigma_now(info["age"])
    rec = dict(j=int(j), ep=e, t=t, label=int(orc["label"][j]), d0=float(orc["d0"][j]), hold_ftk=hold_ftk,
               speed=float(np.hypot(priv[-1, 3], priv[-1, 4])), phase=int(info["phase"]), age=int(info["age"]),
               v0=float(v_set[0]), qmax=float(q0.max()), v_rob=float(v_set.min()), qmax_rob=float(q_set.min(0).max()),
               sig_p=sig_p, radius=ts.radius(info["age"]), burst=(list(burst) if burst else None),
               err_now=float(np.hypot(*(priv[-1, 0:2] - info["ball"][0:2]))))
    # --- the monitor's own fallback, closed loop
    fail, ftk, dmin = False, -1, renv.env.dist()
    for k in range(T):
        u = lt.U_np[int(np.argmax(q0))] if k == 0 else lt.U_np[int(np.argmax(lt.q()[0]))]
        r = renv.tick(np.asarray(u, np.float64))
        info = cr.update(r["priv"][0:2], r["priv"][3:5])
        obs[n, :9] = r["obs"][:9]
        obs[n, 9:] = info["ball"]
        act[n - 1] = r["act"]
        n += 1
        dmin = min(dmin, r["d"])
        if r["failed"] or r["tunnel"]:
            fail, ftk = True, k
            break
        lt.sync(obs[:n], act[:n - 1])
    rec.update(fail=bool(fail), fail_tick=int(ftk), dmin=float(dmin))
    return rec


# ===================================================================================== runner
def _init(cfg):
    G["cfg"] = cfg
    G["st"] = Store(cfg["data"], cfg["split"])
    G["renv"] = RU.make_clean_env(G["st"].meta)
    G["orc"] = load_oracle(cfg["oracle"])
    G["cond"] = CR.parse_cond(cfg["cond"])
    G["lt"] = LT.load_tracks(cfg["runs_root"], cfg["model_setup"], cfg["arm"], [cfg["seed"]], G["renv"].U, G["renv"].alpha)[0]


def _work(j):
    t0 = time.perf_counter()
    r = run_state(G["renv"], G["st"], G["orc"], j, G["lt"], G["cond"], G["cfg"]["run_seed"], T_CL, G["cfg"].get("tset"))
    r["wall_s"] = time.perf_counter() - t0
    return r


def out_path(out_root, tag, split, setup, arm, seed, cond):
    d = Path(out_root) / "test1" / tag / split / setup
    return d / f"{arm}_s{int(seed)}__{cond}.jsonl"


def run(setup, arm, seed, cond, split="test", n_sav=None, n_hold=None, workers=1, tag="main", out_root=RU.OUT,
        model_setup=None, run_seed=0, shard=None, tset=None, oracle=None):
    model_setup = model_setup or setup
    data = RU.data_dir(setup)
    orc_p = Path(oracle) if oracle else oracle_path(setup, split)
    orc = load_oracle(orc_p)
    sel = population(orc, n_sav, n_hold)
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        sel = sel[i::n]
    pj = out_path(out_root, tag, split, setup, arm, seed, cond)
    if shard:
        pj = pj.with_name(pj.stem + f"__part{shard.replace('/', 'of')}.jsonl")
    pj.parent.mkdir(parents=True, exist_ok=True)
    done = {r["j"] for r in RU.read_jsonl(pj)}
    todo = [int(j) for j in sel if int(j) not in done]
    RU.log(f"[test1] {setup}/{arm}_s{seed} cond={cond} split={split} n={len(sel)} done={len(done)} todo={len(todo)} -> {pj}")
    cfg = dict(data=str(data), split=split, oracle=str(orc_p), cond=cond, runs_root=RU.runs_root(), model_setup=model_setup,
               arm=arm, seed=int(seed), run_seed=int(run_seed), tset=tset)
    t0 = time.perf_counter()
    if todo:
        with open(pj, "a") as f:
            if workers <= 1:
                _init(cfg)
                for i, j in enumerate(todo):
                    f.write(json.dumps(_work(j)) + "\n"); f.flush()
                    if (i + 1) % 25 == 0:
                        RU.log(f"  {i + 1}/{len(todo)}")
            else:
                with Pool(workers, initializer=_init, initargs=(cfg,)) as P:
                    for i, r in enumerate(P.imap_unordered(_work, todo, chunksize=1)):
                        f.write(json.dumps(r) + "\n"); f.flush()
                        if (i + 1) % 25 == 0:
                            RU.log(f"  {i + 1}/{len(todo)}")
    recs = [r for r in RU.read_jsonl(pj) if r["j"] in set(int(x) for x in sel)]
    RU.log(f"[test1] done {len(recs)} states in {time.perf_counter() - t0:.0f}s "
           f"({np.mean([r['wall_s'] for r in recs]) if recs else float('nan'):.2f} s/state)")
    return recs


# ===================================================================================== analysis
def conformal_tau(recs, eps, key="v0"):
    """smallest tau such that the empirical P(fallback fails | V >= tau) over `recs` (savable + hold-ok) is <= eps, with the
    finite-sample (k + 1) / (n + 1) correction (split-conformal style upper bound). Returns (tau, n_safe, rate)."""
    v = np.array([r[key] for r in recs]); f = np.array([r["fail"] for r in recs], float)
    order = np.argsort(v)
    vs, fs = v[order], f[order]
    # for threshold = vs[i]: states with V >= vs[i] are vs[i:]
    k = np.cumsum(fs[::-1])[::-1]
    n = np.arange(len(vs), 0, -1)
    ok = (k + 1.0) / (n + 1.0) <= eps
    if not ok.any():
        return float("inf"), 0, float("nan")
    i = int(np.flatnonzero(ok)[0])
    return float(vs[i]), int(n[i]), float(k[i] / n[i])


def bin_index(d0):
    return np.clip(np.digitize(d0, DBINS) - 1, 0, len(DBINS) - 2)


def false_safe_table(recs, tau, key="v0"):
    """rows per distance bin for the certified-savable states: n, safe (V>=tau), fs (safe and fail), fs rate, P(fail | safe);
    for hold-ok states: false-intervention rate (V < tau)."""
    sav = [r for r in recs if r["label"] == SAVABLE]
    hok = [r for r in recs if r["label"] == HOLD_OK]
    out = []
    for b in range(len(DBINS) - 1):
        s = [r for r in sav if bin_index(r["d0"]) == b]
        h = [r for r in hok if bin_index(r["d0"]) == b]
        n = len(s)
        safe = sum(r[key] >= tau for r in s)
        fs = sum((r[key] >= tau) and r["fail"] for r in s)
        fail = sum(r["fail"] for r in s)
        out.append(dict(bin=f"[{DBINS[b]:.2f},{DBINS[b + 1]:.2f})", n=n, fail=fail, safe=safe, fs=fs,
                        fs_rate=RU.rate(fs, n), p_fail_given_safe=RU.rate(fs, safe),
                        n_hold=len(h), false_interv=RU.rate(sum(r[key] < tau for r in h), len(h))))
    return out


def inner_outer(recs, tau, key="v0", n_boot=2000, seed=0, inner=(0.0, 0.02), outer=(0.06, 0.11)):
    """R1 statistic: false-safe rate (per savable state) in the inner bin minus the outer bin, bootstrap CI (resampling episodes)."""
    sav = [r for r in recs if r["label"] == SAVABLE]
    def sel(lo, hi):
        return [r for r in sav if lo <= r["d0"] < hi]
    a, b = sel(*inner), sel(*outer)
    if not a or not b:
        return dict(n_inner=len(a), n_outer=len(b), diff=float("nan"), ci95=(float("nan"), float("nan")))
    fs = lambda r: float((r[key] >= tau) and r["fail"])
    eps = sorted({r["ep"] for r in a + b})
    ia = {e: [fs(r) for r in a if r["ep"] == e] for e in eps}
    ib = {e: [fs(r) for r in b if r["ep"] == e] for e in eps}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(eps, len(eps))
        xa = [x for e in pick for x in ia[e]]; xb = [x for e in pick for x in ib[e]]
        if xa and xb:
            diffs.append(np.mean(xa) - np.mean(xb))
    d0 = np.mean([fs(r) for r in a]) - np.mean([fs(r) for r in b])
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"),) * 2
    return dict(n_inner=len(a), n_outer=len(b), rate_inner=float(np.mean([fs(r) for r in a])),
                rate_outer=float(np.mean([fs(r) for r in b])), diff=float(d0), ci95=(float(lo), float(hi)))


def excess_inner_outer(recs, ref, tau, key="v0", n_boot=2000, seed=0, inner=(0.0, 0.02), outer=(0.06, 0.11)):
    """R1b: difference in differences. Per savable state (paired by (seed, j) with the reference condition `ref`, normally `off`)
    e = false-safe(condition) - false-safe(reference). Returns mean(e | inner) - mean(e | outer) with a bootstrap CI over episodes."""
    fs = lambda r: float((r[key] >= tau) and r["fail"])
    rb = {(r["seed"], r["j"]): r for r in ref if r["label"] == SAVABLE}
    pairs = [(r, rb[(r["seed"], r["j"])]) for r in recs if r["label"] == SAVABLE and (r["seed"], r["j"]) in rb]
    def pick(lo, hi):
        return [(a, b) for a, b in pairs if lo <= a["d0"] < hi]
    A, B = pick(*inner), pick(*outer)
    if not A or not B:
        return dict(n_inner=len(A), n_outer=len(B), diff=float("nan"), ci95=(float("nan"), float("nan")))
    eps = sorted({a["ep"] for a, _ in A + B})
    ea = {e: [fs(a) - fs(b) for a, b in A if a["ep"] == e] for e in eps}
    eb = {e: [fs(a) - fs(b) for a, b in B if a["ep"] == e] for e in eps}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pk = rng.choice(eps, len(eps))
        xa = [x for e in pk for x in ea[e]]; xb = [x for e in pk for x in eb[e]]
        if xa and xb:
            diffs.append(np.mean(xa) - np.mean(xb))
    mean = float(np.mean([fs(a) - fs(b) for a, b in A]) - np.mean([fs(a) - fs(b) for a, b in B]))
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"),) * 2
    return dict(n_inner=len(A), n_outer=len(B), excess_inner=float(np.mean([fs(a) - fs(b) for a, b in A])),
                excess_outer=float(np.mean([fs(a) - fs(b) for a, b in B])), diff=mean, ci95=(float(lo), float(hi)))


def by_phase(recs, tau, key="v0"):
    sav = [r for r in recs if r["label"] == SAVABLE]
    out = {}
    for p, nm in enumerate(CR.PHASES):
        s = [r for r in sav if r["phase"] == p]
        out[nm] = dict(n=len(s), fs=RU.rate(sum((r[key] >= tau) and r["fail"] for r in s), len(s)),
                       fail=RU.rate(sum(r["fail"] for r in s), len(s)))
    return out


def summarize_file(recs, taus, keys=("v0", "v_rob")):
    out = dict(n=len(recs), n_sav=sum(r["label"] == SAVABLE for r in recs), n_hold=sum(r["label"] == HOLD_OK for r in recs),
               fail_rate_sav=RU.rate(sum(r["fail"] for r in recs if r["label"] == SAVABLE),
                                     sum(r["label"] == SAVABLE for r in recs)))
    for key in keys:
        for tau in taus:
            out[f"{key}@{tau:g}"] = dict(bins=false_safe_table(recs, tau, key), inner_outer=inner_outer(recs, tau, key),
                                         phase=by_phase(recs, tau, key))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--setup", required=True, choices=RU.SETUPS)
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cond", required=True)
    r.add_argument("--split", default="test", choices=["val", "test"])
    r.add_argument("--n-sav", type=int, default=None)
    r.add_argument("--n-hold", type=int, default=None)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--tag", default="main")
    r.add_argument("--out", default=str(RU.OUT))
    r.add_argument("--model-setup", default="")
    r.add_argument("--run-seed", type=int, default=0)
    r.add_argument("--shard", default="")
    r.add_argument("--oracle", default="", help="oracle npz (default: TEST = the dataset's test_v0; VAL = <OUT>/oracle/val_v0.npz)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.setup, a.arm, a.seed, a.cond, a.split, a.n_sav, a.n_hold, a.workers, a.tag, a.out, a.model_setup or None,
            a.run_seed, a.shard or None, None, a.oracle or None)


if __name__ == "__main__":
    main()
