"""Feasibility probe (SIM ONLY, CPU only). See README.md.
  python -m world_model.v2.testbeds.a_loose_items.probe --beta 45 --h 0.025 --n 10 --episodes 60 --workers 6 --out <json>
Per failing nominal episode: (a) failure rate, (b) certify 4 random pre-failure states (window = 100 ticks before onset),
(c) backward scan (stride 5 ticks) from the onset until the first certified state -> lead time."""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.a_loose_items.tray_env import TrayEnv, TrayCfg, fallback_U, brake_action, DT_WM
from world_model.v2.testbeds.a_loose_items import carry

T_BR, SNAP_EVERY, N_RAND = 100, 5, 40


def run_branch(env, snap, seq_fn, T=T_BR):
    """True if no rim spill and no tunnel artifact for T ticks. seq_fn(env, t) -> u."""
    env.restore(snap)
    for t in range(T):
        r = env.tick(seq_fn(env, t))
        if r["n_spilled"] > 0 or r["tunnel"]:
            return False, r["tunnel"], t
    return True, False, T


def certify(env, snap, rng, n_rand=N_RAND):
    """hold(u=0) -> brake -> 84 held constants -> n_rand random 2-segment sequences + brake tail (bead_rep/oracle.py recipe,
    smaller n_rand). Returns dict(certified, hold_ok, brake_ok, kind, n_eval)."""
    U = fallback_U(); zero = np.zeros(6)
    ne = 1
    hold_ok, _, _ = run_branch(env, snap, lambda e, t: zero)
    out = dict(hold_ok=bool(hold_ok), brake_ok=False, certified=bool(hold_ok), kind=1 if hold_ok else 0)
    if hold_ok:
        out["n_eval"] = ne; return out
    brake = lambda e, t: brake_action(e.v, e.axis())
    ne += 1
    ok, _, _ = run_branch(env, snap, brake)
    out["brake_ok"] = bool(ok)
    if ok:
        out.update(certified=True, kind=2, n_eval=ne); return out
    nz = [u for u in U if np.abs(u).sum() > 0]
    for u in nz:
        ne += 1
        ok, _, _ = run_branch(env, snap, lambda e, t, u=u: u)
        if ok:
            out.update(certified=True, kind=3, n_eval=ne); return out
    for _ in range(n_rand):
        ne += 1
        k = int(rng.integers(3, 16)); u1, u2 = U[rng.integers(len(U))], U[rng.integers(len(U))]
        k2 = int(rng.integers(3, 16))
        def f(e, t, u1=u1, u2=u2, k=k, k2=k2):
            return u1 if t < k else (u2 if t < k + k2 else brake_action(e.v, e.axis()))
        ok, _, _ = run_branch(env, snap, f)
        if ok:
            out.update(certified=True, kind=4, n_eval=ne); return out
    out["n_eval"] = ne
    return out


def episode(args):
    (cfgd, ep, carry_kind, seed0, do_cert, n_states) = args
    cfg = TrayCfg(**cfgd)
    env = TrayEnv(cfg)
    rng = np.random.default_rng(seed0 * 100003 + ep)
    if carry_kind == "replay":
        from world_model.v2.wm_audit import common as WC
        D = WC.load_split("test")
        acts = carry.replay(D, int(rng.integers(len(D["lens"]))), T=300)
    else:
        acts = carry.synthetic(rng, T=300)
    env.reset(int(seed0 * 100003 + ep))
    snaps, onset, tun, lh, ls = {}, -1, False, [], []
    t0 = time.perf_counter()
    for t, u in enumerate(acts):
        if t % SNAP_EVERY == 0:
            snaps[t] = env.snapshot()
        r = env.tick(u)
        lh.append(r["l"]); ls.append(env.softmin_margin())
        if r["tunnel"]:
            tun = True
        if r["n_spilled"] > 0 and onset < 0:
            onset = t + 1                       # state index after this tick contains the exit
            break
    res = dict(ep=ep, fail=onset >= 0, onset=onset, tunnel=tun, ticks=t + 1, sim_s=time.perf_counter() - t0, n_cert=[])
    res["l_hard_tail"] = [float(x) for x in lh[-60:]]; res["l_soft_tail"] = [float(x) for x in ls[-60:]]
    if do_cert and not tun:
        # unconditional hold-still spill rate: 2 random snapshot ticks per episode (failing or not)
        ks_all = [k for k in snaps if k >= 20]
        res["hold_all"] = [int(not run_branch(env, snaps[int(k)], lambda e, t: np.zeros(6))[0]) for k in rng.choice(ks_all, size=min(2, len(ks_all)), replace=False)] if ks_all else []
    if onset >= 0 and do_cert and not tun:
        # (b) random pre-failure states in [onset-100, onset-1] on the snapshot grid
        cand = [k for k in snaps if onset - T_BR <= k <= onset - 1]
        t1 = time.perf_counter()
        for k in rng.choice(cand, size=min(n_states, len(cand)), replace=False) if cand else []:
            c = certify(env, snaps[int(k)], rng)
            c.update(tick=int(k), dt_to_onset=int(onset - k)); res["n_cert"].append(c)
        # (c) backward scan for the first certified state
        ks = sorted([k for k in snaps if k <= onset - 1], reverse=True)
        lead = None
        for k in ks:
            c = certify(env, snaps[k], rng)
            if c["certified"]:
                lead = dict(last_cert_tick=int(k), kind=int(c["kind"]), lead_ticks=float(onset - k - SNAP_EVERY / 2)); break
        res["lead"] = lead if lead is not None else dict(last_cert_tick=None, lead_ticks=None)
        res["cert_s"] = time.perf_counter() - t1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=45.0); ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--carry", default="replay"); ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--no-cert", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfgd = dict(n_items=a.n, beta_deg=a.beta, h=a.h)
    jobs = [(cfgd, e, a.carry, a.seed, not a.no_cert, a.states) for e in range(a.episodes)]
    t0 = time.perf_counter()
    with Pool(a.workers, maxtasksperchild=3) as p:
        R = list(p.imap_unordered(episode, jobs))
    wall = time.perf_counter() - t0
    fail = np.array([r["fail"] for r in R]); cpu = sum(r["sim_s"] + r.get("cert_s", 0.0) for r in R)
    cs = [c for r in R for c in r["n_cert"]]
    leads = [r["lead"]["lead_ticks"] for r in R if r.get("lead") and r["lead"]["lead_ticks"] is not None]
    ha = [x for r in R for x in r.get("hold_all", [])]
    def binfrac(lo, hi):
        b = [c for c in cs if lo <= c["dt_to_onset"] <= hi]
        return dict(n=len(b), hold_ok=float(np.mean([c["hold_ok"] for c in b])) if b else None,
                    uncert=float(np.mean([not c["certified"] for c in b])) if b else None,
                    savable_not_hold=float(np.mean([c["certified"] and not c["hold_ok"] for c in b])) if b else None)
    # margin warning: ticks before onset at which the margin first drops below 0.3 and stays under (hard vs soft)
    def warn(key, thr=0.3):
        w = []
        for r in R:
            tl = r.get(key)
            if r["fail"] and tl:
                x = np.array(tl); idx = np.where(x >= thr)[0]
                w.append(len(x) - (idx[-1] + 1 if len(idx) else 0))
        return float(np.median(w)) if w else None
    nolead = sum(1 for r in R if r.get("lead") and r["lead"]["lead_ticks"] is None)
    summ = dict(cfg=cfgd, carry=a.carry, episodes=len(R), fail_rate=float(fail.mean()), tunnel_eps=int(sum(r["tunnel"] for r in R)),
                n_pre_failure_states=len(cs),
                frac_hold_ok=float(np.mean([c["hold_ok"] for c in cs])) if cs else None,
                frac_brake_ok=float(np.mean([c["brake_ok"] for c in cs])) if cs else None,
                frac_savable_not_hold=float(np.mean([c["certified"] and not c["hold_ok"] for c in cs])) if cs else None,
                frac_uncertified=float(np.mean([not c["certified"] for c in cs])) if cs else None,
                kinds={int(k): int(sum(c["kind"] == k for c in cs)) for k in range(5)} if cs else None,
                lead_ticks_median=float(np.median(leads)) if leads else None,
                lead_ticks_pcts=[float(x) for x in np.percentile(leads, [10, 25, 50, 75, 90])] if leads else None,
                n_lead=len(leads), n_never_certified_in_scan=nolead,
                hold_spill_rate_all_states=float(np.mean(ha)) if ha else None, n_hold_all=len(ha),
                bins={"1-20":binfrac(1,20),"21-60":binfrac(21,60),"61-100":binfrac(61,100)},
                margin_below_0p3_ticks_before_onset=dict(hard=warn("l_hard_tail"), soft=warn("l_soft_tail")), cpu_s=cpu, wall_s=wall)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summ, episodes=R, certs=cs), open(a.out, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(summ), flush=True)

if __name__ == "__main__":
    main()
