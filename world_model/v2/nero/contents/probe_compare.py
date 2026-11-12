"""nero/contents/probe_compare.py -- probe-only comparison of the weigh-probe methods on ONE world list (Stage 0b, 11-12).  SIMULATION ONLY, offline.

Question: does poslift_ff / claw_guard resolve the loads poslift cannot (bug B6), with the truth inside Theta and no U during the probe?
Worlds (the same for every method; bounds rule 8):
  behind  box_behind, tray6_behind, i = 0..19: nero/base/smoke.behind_world(task, i) (the run_n20_v2b / forced_admit worlds); episode as forced_admit.job
          (swerve base standing still, the proposer's FIRST accepted plan from the run_n20_v2b cache, offline); M_stop = empty + behind_job.c_admit_max(task, plan)
  calib   box, tray6, pot, i = 0..19: probe._world(task, probe._seed("calib", task, i)) (the kstep_calib_2026-11-09 calibration worlds); episode probe._episode;
          M_stop = empty + probe.c_admit_max(task).  NOTE: the eps of each method are calibrated on these worlds, so truth-in-Theta here is IN-SAMPLE.
Per (world, method) ONLY the probe runs (no carry): probe.make_probe(method, rng = default_rng(seed), noise=True, adaptive=True).run(M_stop), U incl. the per-claw
payload row (same_worlds._init_worker), every U kind counted (ep._viol wrapped as forced_admit does).  Recorded: Theta, truth inside (sampled), width, why,
ceiling (last read L, u_read_max), peak_u (sim truth), U, probe_time, static verdict on Theta and on the TRUE contents point (behind_job._static + base level as
forced_admit / probe.static_interval), the eps used and their source.  A method whose own CALIB.json is missing refuses to run unless --provisional-eps
(then it borrows poslift's eps and every record says so).

python -m world_model.v2.nero.contents.probe_compare run --out DIR [--methods poslift,poslift_ff,claw_guard] [--sets behind,calib] [--procs N] [--shard k/K]
       [--only box_behind:4,tray6_behind:0] [--calib poslift_ff=PATH,claw_guard=PATH] [--provisional-eps]
python -m world_model.v2.nero.contents.probe_compare summary --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

METHODS = ("poslift", "poslift_ff", "claw_guard")
SETS = {"behind": ("box_behind", "tray6_behind"), "calib": ("box", "tray6", "pot")}
N_WORLDS = 20


def world_list(sets, n=N_WORLDS):
    return [(s, t, i) for s in sets for t in SETS[s] for i in range(n)]


def _eps(method, calib_over, provisional):
    """(eps_lo, eps_free, source).  The method's own CALIB (load_calib asserts its method field); --provisional-eps falls back to poslift's for a new method."""
    from world_model.v2.nero.contents import probe as P
    path = calib_over.get(method) or P.default_calib(method)
    if Path(path).exists():
        c = P.load_calib(path, method)["eps"]
        return float(c["EPS_LO"]), float(c["EPS_FREE"]), f"MEASURED {path}"
    if not provisional or method == "poslift":
        raise FileNotFoundError(f"{method}: no CALIB at {path} (run the PROBE_FIX_RUN.md calibration, or pass --provisional-eps)")
    c = P.load_calib(P.default_calib("poslift"), "poslift")["eps"]
    return float(c["EPS_LO"]), float(c["EPS_FREE"]), f"PROVISIONAL: poslift eps ({P.default_calib('poslift')}); {method} not calibrated"


def job(args):
    wset, task, i, method, calib_over, provisional = args
    t0 = time.time()
    try:
        from world_model.v2.nero.contents import same_worlds as SW
        SW._init_worker()                                                   # strict_claw=True: the per-claw payload row is in U (as the source runs)
        from world_model.v2.nero.contents import carry as CR, carry_cert as CC, probe as P
        eps_lo, eps_free, eps_src = _eps(method, calib_over, provisional)
        if wset == "behind":
            from world_model.v2.adapters import base_envelope as BE
            from world_model.v2.nero.base import smoke as SM
            from world_model.v2.nero.contents import behind_job as BJ, forced_admit as FA
            from world_model.v2.stack import tasks as TK
            bt = BJ.base_task(task)
            plan, cp = FA.first_plan(task)                                  # offline cache of the source run
            wp0 = cp.wp
            seed, w = SM.behind_world(task, i)
            ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp0, base="swerve", goal_support=TK.task(task).goal_support)
            cm = BJ.c_admit_max(task, wp0)

            def verdict(lo, hi):
                st = BJ._static(task, lo, hi, w, wp0)
                lvl = None
                if st["a_admit"] is not None:
                    lvl, _ = BE.pick_level(task, P.interval_spec(bt, lo, hi), BJ.LEVELS, BJ.SOURCE)
                return dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], base_level=lvl,
                            admitted=st["a_admit"] is not None and lvl is not None)
            plan_id = plan.get("plan_id")
        else:
            bt = task
            seed = P._seed("calib", task, i)
            w = P._world(task, seed)
            ep = P._episode(w)
            cm = P.c_admit_max(task)

            def verdict(lo, hi):
                st = P.static_interval(task, lo, hi, w)
                return dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], base_level=None, admitted=st["a_admit"] is not None)
            plan_id = None
        allv = {}
        orig = ep._viol

        def _viol_all(kind, t, **kw):                                       # every U entry, not only the first
            e = allv.setdefault(kind, dict(n=0, t_first=round(float(t), 3)))
            e["n"] += 1
            return orig(kind, t, **kw)
        ep._viol = _viol_all
        e = P.empty_mass(bt)
        M = P.true_mass(ep)
        c_true = M - e
        pr = P.make_probe(ep, bt, eps_lo, eps_free, rng=np.random.default_rng(seed), noise=True, method=method, adaptive=True)
        tp = time.time()
        why = pr.run(M_stop=None if cm is None else e + cm)
        wall_probe = round(time.time() - tp, 1)
        c_lo, c_hi = pr.M_lo - e, pr.M_hi - e
        rec = dict(set=wset, task=task, i=i, seed=seed, method=method, plan_id=plan_id, eps_lo=eps_lo, eps_free=eps_free, eps_source=eps_src,
                   probe_assumptions=P.method_assumptions(method), M_true=M, c_true=c_true, Theta=[pr.M_lo, pr.M_hi],
                   theta_sound=bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9), width=pr.M_hi - pr.M_lo, why=why, n_rungs=len(pr.rungs),
                   n_ff_rungs=sum(1 for r in pr.rungs if r.get("phase") == "ff"), last_L=pr.rungs[-1]["L"] if pr.rungs else None,
                   u_read_max=round(float(getattr(pr, "u_read_max", 0.0)), 4), peak_u=round(float(getattr(pr, "peak_u", 0.0)), 4),
                   probe_viol=CR._jsonable(ep.track["viol"]), all_U=allv, probe_time_s=getattr(pr, "probe_time_s", None), wall_probe_s=wall_probe,
                   M_stop=None if cm is None else e + cm, guard_events=list(getattr(pr, "guard_events", [])), probe_summary=getattr(pr, "summary", None),
                   rungs=pr.rungs, probe_track=dict(pr.track))
        rec["verdict_theta"] = verdict(c_lo, c_hi)
        rec["verdict_true"] = verdict(c_true, c_true)
        rec["wall_s"] = round(time.time() - t0, 1)
        return CR._jsonable(rec)
    except Exception as ex:                                                 # noqa: BLE001 -- recorded, the shard goes on
        return dict(set=wset, task=task, i=i, method=method, error=repr(ex), tb=traceback.format_exc()[-1500:], wall_s=round(time.time() - t0, 1))


def run(a):
    import multiprocessing as mp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(",")
    for m in methods:
        assert m in METHODS, m
    calib_over = dict(x.split("=", 1) for x in a.calib.split(",")) if a.calib else {}
    worlds = world_list(a.sets.split(","))
    if a.only:
        keep = {tuple(x.split(":")) for x in a.only.split(",")}
        worlds = [w for w in worlds if (w[1], str(w[2])) in keep]
    jobs = [(s, t, i, m, calib_over, a.provisional_eps) for m in methods for (s, t, i) in worlds]
    k, K = (int(x) for x in a.shard.split("/"))
    jobs = jobs[k::K]
    f = out / f"trials_shard{k}of{K}.jsonl"
    done = set()
    if f.exists():
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["task"], r["i"], r["method"]))
    jobs = [j for j in jobs if (j[1], j[2], j[3]) not in done]
    (out / f"CONFIG_shard{k}of{K}.json").write_text(json.dumps(dict(methods=methods, sets=a.sets, calib_override=calib_over, provisional_eps=a.provisional_eps,
                                                                   n_jobs=len(jobs), skipped_done=len(done), procs=a.procs, host=os.uname().nodename,
                                                                   date=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    print(f"[probe_compare] shard {k}/{K}: {len(jobs)} jobs ({len(done)} done), procs {a.procs}", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.procs, maxtasksperchild=3) as pool, open(f, "a") as fh:
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(f"[probe_compare] {r.get('task')} {r.get('i')} {r.get('method')} -> {r.get('why', r.get('error'))} Theta {r.get('Theta')} "
                  f"U {list((r.get('all_U') or {}).keys())} wall {r.get('wall_s')}", flush=True)


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def summary(a):
    out = Path(a.out)
    recs = []
    for f in sorted(out.glob("trials_shard*.jsonl")):
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    last = {}
    for r in recs:                                                          # a resumed world: keep the last good record
        key = (r["task"], r["i"], r["method"])
        if "error" not in r or key not in last:
            last[key] = r
    ok = [r for r in last.values() if "error" not in r]
    errs = [r for r in last.values() if "error" in r]
    L = ["# probe-only comparison -- machine-written by probe_compare.summary", "",
         "Same worlds for every method (behind: smoke.behind_world i=0..19; calib: probe._seed('calib') i=0..19, IN-SAMPLE for the eps). Probe only, no carry.",
         "truth in Theta = sampled (n worlds), not a proof. 'refused (Theta)' = static row (+ base level for the behind tasks) refuses on the probe's Theta; "
         "'refused (true)' = refuses on the true contents point; 'unneeded' = refused on Theta but admitted on the true point.",
         "U = any U entry during the probe (all kinds, incl. the per-claw payload row on sim truth). ceiling = max last read L over the worlds that did not end free.", "",
         "| method | task | n | err | why (count) | truth in Theta | width kg p50/max | free | ceiling kg | u_read max | peak_u max | U worlds (kinds) | "
         "refused Theta / true / unneeded | ff rungs p50 | probe s p50 | eps source |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, t) in sorted({(r["method"], r["task"]) for r in last.values()}):
        rs = [r for r in ok if r["method"] == m and r["task"] == t]
        ne = sum(1 for r in errs if r["method"] == m and r["task"] == t)
        if not rs:
            L.append(f"| {m} | {t} | 0 | {ne} | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        why = Counter(r["why"] for r in rs)
        wid = [r["width"] for r in rs]
        ceil = [r["last_L"] for r in rs if r["why"] != "free" and r["last_L"] is not None]
        uk = Counter(k for r in rs for k in (r.get("all_U") or {}))
        nu = sum(1 for r in rs if r.get("all_U"))
        rt = sum(1 for r in rs if not r["verdict_theta"]["admitted"])
        rtrue = sum(1 for r in rs if not r["verdict_true"]["admitted"])
        un = sum(1 for r in rs if not r["verdict_theta"]["admitted"] and r["verdict_true"]["admitted"])
        L.append(f"| {m} | {t} | {len(rs)} | {ne} | {dict(why)} | {sum(r['theta_sound'] for r in rs)}/{len(rs)} | {np.median(wid):.2f}/{max(wid):.2f} | "
                 f"{why.get('free', 0)} | {_fmt(max(ceil) if ceil else None)} | {max(r['u_read_max'] for r in rs):.3f} | {max(r['peak_u'] for r in rs):.3f} | "
                 f"{nu} ({dict(uk)}) | {rt} / {rtrue} / {un} | {int(np.median([r['n_ff_rungs'] for r in rs]))} | "
                 f"{np.median([r['probe_time_s'] or 0 for r in rs]):.1f} | {sorted({r['eps_source'][:11] for r in rs})} |")
    src_rows = {}
    try:                                                                    # reproduction check: poslift on the behind worlds vs the source run (run_n20_v2b, ours)
        from world_model.v2.nero.contents import forced_admit as FA
        for line in open(FA.SRC / "trials.jsonl"):
            if line.strip():
                r = json.loads(line)
                if r.get("method") == "ours" and r.get("Theta"):
                    src_rows[(r["task"], r["i"])] = r["Theta"]
    except (OSError, ValueError):
        pass
    pl = [r for r in ok if r["method"] == "poslift" and (r["task"], r["i"]) in src_rows]
    if pl:
        same = [r for r in pl if r["Theta"] == src_rows[(r["task"], r["i"])]]
        close = [r for r in pl if np.allclose(r["Theta"], src_rows[(r["task"], r["i"])], atol=1e-3)]
        L += ["", f"poslift vs the source run (run_n20_v2b, ours, same worlds): Theta bit-identical {len(same)}/{len(pl)}, within 1e-3 kg {len(close)}/{len(pl)} "
                  "(the source records Theta also for worlds whose probe was skipped as no_probe_needed or ran under Rule 2 re-asks)"]
    L += ["", "## worlds where a method's truth falls outside Theta, or U occurred during the probe"]
    for r in sorted(ok, key=lambda r: (r["task"], r["i"], r["method"])):
        if not r["theta_sound"] or r.get("all_U"):
            L.append(f"- {r['method']} {r['task']} {r['i']}: M_true {r['M_true']:.3f} Theta {[round(x, 3) for x in r['Theta']]} why {r['why']} U {r.get('all_U')}")
    for r in errs:
        L.append(f"- ERROR {r['method']} {r['task']} {r['i']}: {r['error']}")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--sets", default="behind,calib")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default=None, help="task:i,task:i,...")
    ap.add_argument("--calib", default=None, help="method=CALIB.json,... (default: probe.default_calib(method))")
    ap.add_argument("--provisional-eps", action="store_true", help="a new method without its CALIB borrows poslift's eps (smoke only; recorded)")
    a = ap.parse_args(argv)
    run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
