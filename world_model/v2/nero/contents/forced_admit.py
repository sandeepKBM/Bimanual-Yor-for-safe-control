"""nero/contents/forced_admit.py -- carry the worlds our checker REFUSED anyway, to count how many actually fail.  SIMULATION ONLY.   (USER 11-12)

Question (USER 11-12): "run those refused worlds, let us see how many actually run into failures".  Source run: table_behind_same_2026-11-09/run_n20_v2b
(code b9b4d15): ours refused box_behind 13/20 and tray6_behind 20/20, all on the static rows after the weigh probe stopped (rung_not_admissible).

Per (world, variant) this script repeats the source run's pipeline up to the admission and then SKIPS the refusal:
  world     nero/base/smoke.behind_world(task, i)          (same function, same seed as the source run)
  episode   CarryEpisode(..., base="swerve", strict_claw=True via same_worlds._init_worker)   (as behind_job.v2_job)
  probe     probe.make_probe(..., rng=default_rng(seed), method, adaptive=True).run(M_stop = empty + c_admit_max(first plan))   (as skills_contents.Weigh;
            the record carries Theta so it can be compared with the source run)
  plan      the proposer's FIRST accepted plan from the source run's cache (offline: no network), the plan every carried box world used (0 re-asks there).
            No Rule 2 re-ask is run.
  verdicts  recorded, not obeyed: static cell on Theta (the refusal), static cell + base level on the TRUE contents point [c, c] (what the checker would do knowing
            the mass)
  carry     FORCED: compile the plan at the variant's base level, set_belief(<variant belief>, a_plan=A_PLAN), the same ramp / hold / run_segments as probe_select.run_world.
Variants (--variants):
  known:L1|L2|L3   belief = midpoint of the probe interval (what the robot knew), base level forced
  true:L1|L2|L3    belief = the true contents mass (oracle feedforward), base level forced
A_PLAN = 0.5: the a_admit the static cell gave every admitted world of the source run (7/7).  ASSUMED for the forced runs (the checker gave none on Theta).

Recorded besides the source run's fields: EVERY U violation kind (count, first t) -- CarryEpisode keeps only the first in track["viol"] -- by wrapping ep._viol on the
instance (no change to carry.py), the base monitor's flagged set, and the FL-MPC counters of each segment (already in the segment dicts).

python -m world_model.v2.nero.contents.forced_admit run --out DIR [--procs N] [--shard k/K] [--variants known:L1,known:L2] [--only box_behind:4,...] [--probe-only]
python -m world_model.v2.nero.contents.forced_admit summary --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path("/common/users/ss5772/bimanual_data/contents/table_behind_same_2026-11-09/run_n20_v2b")
PLAN_CACHE = Path("/common/users/ss5772/bimanual_data/contents/table_behind_same_2026-11-09/ours_plans_v2")
PLAN_SPEC = "codex:gpt-5.6-sol:high"
A_PLAN = 0.5                     # ASSUMED (see module doc): the a_admit of every admitted world in the source run
PROBE_METHOD = "poslift"         # source run CONFIG.json
ALL_VARIANTS = ("known:L1", "known:L2", "known:L3", "true:L1", "true:L2", "true:L3")


def refused_worlds(src=SRC):
    out = []
    for line in open(src / "trials.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method") == "ours" and r.get("outcome") == "refused_in_R":
            out.append(dict(task=r["task"], i=r["i"], seed=r.get("seed"), Theta=r.get("Theta"), M_true=r.get("M_true"), refused_by=r.get("refused_by"),
                            why=r.get("why"), decision=r.get("decision")))
    return sorted(out, key=lambda w: (w["task"], w["i"]))


_PLANS = {}


def first_plan(task):
    """the proposer's first accepted plan for `task` from the source run's cache (offline).  -> (plan_dict, compiled plan)"""
    if task not in _PLANS:
        from world_model.v2.stack.proposer import compile as PC, propose as PP
        chk = lambda pl: PC.compile_plan(pl, task, level="L1")
        plan, cp, rec, info = PP.propose_until_accepted(task, chk, backend_spec=PLAN_SPEC, cache_dir=str(PLAN_CACHE), offline=True)
        if cp is None:
            raise RuntimeError(f"{task}: no accepted plan in the cache ({plan.get('stage')}: {plan.get('reasons')})")
        _PLANS[task] = (plan, cp)
    return _PLANS[task]


def _calib():
    from world_model.v2.nero.contents import probe as P
    cfg = json.loads((SRC / "CONFIG.json").read_text())
    cal = P.load_calib(cfg["calib"], PROBE_METHOD)["eps"]
    return float(cal["EPS_LO"]), float(cal["EPS_FREE"])


def job(args):
    task, i, variant, probe_only = args
    t0 = time.time()
    os.environ["BIMANUAL_PROBE"] = PROBE_METHOD
    try:
        from world_model.v2.nero.contents import same_worlds as SW
        SW._init_worker()                                                   # strict_claw=True on every CarryEpisode (as the source run)
        from world_model.v2.adapters import base_envelope as BE, outcome_carry as OCC
        from world_model.v2.nero.base import smoke as SM
        from world_model.v2.nero.contents import behind_job as BJ, carry as CR, carry_cert as CC, probe as P
        from world_model.v2.stack import tasks as TK
        from world_model.v2.stack.proposer import compile as PC
        bt = BJ.base_task(task)
        belief_kind, level = variant.split(":")
        eps_lo, eps_free = _calib()
        plan, cp = first_plan(task)
        wp0 = cp.wp
        seed, w = SM.behind_world(task, i)
        ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp0, base="swerve", goal_support=TK.task(task).goal_support)
        allv = {}
        orig = ep._viol

        def _viol_all(kind, t, **kw):                                       # every U entry, not only the first (carry.py keeps the first)
            e = allv.setdefault(kind, dict(n=0, t_first=round(float(t), 3)))
            e["n"] += 1
            return orig(kind, t, **kw)
        ep._viol = _viol_all
        e = P.empty_mass(bt)
        M = P.true_mass(ep)
        c_true = M - e
        cm = BJ.c_admit_max(task, wp0)
        pr = P.make_probe(ep, bt, eps_lo, eps_free, rng=np.random.default_rng(seed), noise=True, method=PROBE_METHOD, adaptive=True)
        why = pr.run(M_stop=None if cm is None else e + cm)
        c_lo, c_hi = pr.M_lo - e, pr.M_hi - e
        probe_viol = CR._jsonable(ep.track["viol"])
        st_theta = BJ._static(task, c_lo, c_hi, w, wp0)
        st_true = BJ._static(task, c_true, c_true, w, wp0)
        lvl_true, rows_true = (None, None)
        if st_true["a_admit"] is not None:
            lvl_true, rows_true = BE.pick_level(task, P.interval_spec(bt, c_true, c_true), BJ.LEVELS, BJ.SOURCE)
        rec = dict(task=task, i=i, seed=seed, variant=variant, belief_kind=belief_kind, level=level, a_plan=A_PLAN, plan_id=plan.get("plan_id"),
                   M_true=M, c_true=c_true, Theta=[pr.M_lo, pr.M_hi], why=why, peak_u=round(getattr(pr, "peak_u", 0.0), 4), probe_viol=probe_viol,
                   verdict_theta=dict(action=st_theta["action"], a_admit=st_theta["a_admit"], u_nom=st_theta["u_nom"]),
                   verdict_true=dict(action=st_true["action"], a_admit=st_true["a_admit"], u_nom=st_true["u_nom"], base_level=lvl_true,
                                     base_rows=None if rows_true is None else BJ._rows_json(rows_true)))
        if probe_only:
            rec.update(outcome="probe_only", wall_s=round(time.time() - t0, 1))
            return rec
        belief = 0.5 * (c_lo + c_hi) if belief_kind == "known" else c_true
        ep.wp, ep.cc = PC.compile_plan(plan, task, build=False, level=level).wp, None
        ep.set_belief(belief, a_plan=A_PLAN)
        m_nom = ep.m_nom
        ep.ctl.mass = pr.m_ff
        pr._ramp(m_nom, 0.3)
        ep.run_phases([("hold", 0.4)])
        ep.ctl.mass = m_nom
        carry = CR.run_segments(ep, dict(segments=[]), branch_backups=True)
        any_U = bool(probe_viol or ep.track["viol"] or any(s["nominal"]["viol"] for s in carry["segments"]))
        outcome = "carried" if (carry["end_in_R"][0] and not any(s["nominal"]["viol"] for s in carry["segments"])) else "carry_failed"
        rec.update(belief=belief, carry=dict(segments=carry["segments"], end_in_R=carry["end_in_R"], pre_hold_viol=carry["pre_hold_viol"]), outcome=outcome, any_U=any_U)
        rec.update(OCC.measure(ep, task, True, any_U, carry["pre_hold_viol"], refused=False, reason=outcome, end_in_R=tuple(carry["end_in_R"])))
        bl = getattr(ep, "base_log", [])
        rec["base"] = dict(n_base_edges=len(bl), n_aborted=sum(1 for b in bl if b.get("aborted")), route_end_err_mm=[b.get("err_mm") for b in bl],
                           U_flags={k: (k in set(ep.bmon.flagged)) for k in BJ.BASE_FLAGS}, base_end_pose=[float(v) for v in ep.sb.pose()])
        rec["first_U"] = BJ._first_viol(ep, probe_viol)
        rec["all_U"] = allv
        rec["base_flagged"] = sorted(ep.bmon.flagged)
        fk = ("fl_steps", "fl_n_infeasible", "fl_n_sat_steps", "fl_n_tau_clipped", "fl_n_speed_binding")
        rec["fl"] = {k: int(sum(int((s.get("fl") or {}).get(k, 0) or 0) for s in carry["segments"])) for k in fk}
        rec["wall_s"] = round(time.time() - t0, 1)
        return CR._jsonable(rec)
    except Exception as ex:
        return dict(task=task, i=i, variant=variant, error=repr(ex), tb=traceback.format_exc()[-1500:], wall_s=round(time.time() - t0, 1))


def run(a):
    import multiprocessing as mp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    worlds = refused_worlds()
    if a.only:
        keep = {tuple(x.split(":")) for x in a.only.split(",")}
        worlds = [w for w in worlds if (w["task"], str(w["i"])) in keep]
    variants = a.variants.split(",")
    jobs = [(w["task"], w["i"], v, a.probe_only) for v in variants for w in worlds]
    k, K = (int(x) for x in a.shard.split("/"))
    jobs = jobs[k::K]
    done = set()
    f = out / f"trials_shard{k}of{K}.jsonl"
    if f.exists():
        for line in open(f):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["task"], r["i"], r["variant"]))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    (out / f"CONFIG_shard{k}of{K}.json").write_text(json.dumps(dict(src=str(SRC), plan_cache=str(PLAN_CACHE), plan_spec=PLAN_SPEC, a_plan=A_PLAN, probe_method=PROBE_METHOD,
                                                                   variants=variants, n_jobs=len(jobs), skipped_done=len(done), procs=a.procs, host=os.uname().nodename,
                                                                   probe_only=a.probe_only), indent=1))
    print(f"[forced] shard {k}/{K}: {len(jobs)} jobs ({len(done)} already done), procs {a.procs}", flush=True)
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=3) as pool, open(f, "a") as fh:
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            print(f"[forced] {r.get('task')} {r.get('i')} {r.get('variant')} -> {r.get('outcome', r.get('error'))} first_U={r.get('first_U')} "
                  f"all_U={list((r.get('all_U') or {}).keys())} wall {r.get('wall_s')}", flush=True)


def summary(a):
    out = Path(a.out)
    recs = []
    for f in sorted(out.glob("trials_shard*.jsonl")):
        recs += [json.loads(line) for line in open(f) if line.strip()]
    src = {(w["task"], w["i"]): w for w in refused_worlds()}
    L = ["# forced admission of the refused worlds -- machine-written by forced_admit.summary", "",
         f"source run: {SRC} (refused worlds only); plan = the proposer's first accepted plan; a_plan = {A_PLAN} (ASSUMED); belief known = probe-interval midpoint, true = true contents.",
         "success = finished AND goal (stack.outcome); unsafe = any U (probe + carry, incl. base flags and the per-claw payload row). all_U counts every U kind, not only the first.", ""]
    errs = [r for r in recs if "error" in r]
    ok = [r for r in recs if "error" not in r and r.get("outcome") != "probe_only"]
    po = [r for r in recs if r.get("outcome") == "probe_only"]
    if po:
        L.append("## probe-only check (Theta vs the source run)")
        for r in sorted(po, key=lambda r: (r["task"], r["i"])):
            s = src.get((r["task"], r["i"]), {})
            L.append(f"- {r['task']} {r['i']}: Theta {[round(x, 3) for x in r['Theta']]} vs source {[round(x, 3) for x in (s.get('Theta') or [])]}; why {r['why']}; "
                     f"verdict on Theta {r['verdict_theta']['action']}; verdict on TRUE mass {r['verdict_true']['action']} level {r['verdict_true']['base_level']}; wall {r['wall_s']} s")
        L.append("")
    L.append("| task | variant | n | errors | success | safe_success | unsafe | carried w/o U | U kinds (any time, worlds) | first U (worlds) | FL-MPC infeasible (worlds; steps) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (task, v) in sorted({(r["task"], r["variant"]) for r in ok + errs}):
        rs = [r for r in ok if r["task"] == task and r["variant"] == v]
        ne = sum(1 for r in errs if r["task"] == task and r["variant"] == v)
        res = [r.get("result") or {} for r in rs]
        kinds = Counter(k for r in rs for k in (r.get("all_U") or {}))
        kinds.update(k for r in rs for k in (r.get("base_flagged") or []) if k not in (r.get("all_U") or {}))
        first = Counter((r.get("first_U") or {}).get("kind") for r in rs if r.get("first_U"))
        inf = [r for r in rs if (r.get("fl") or {}).get("fl_n_infeasible", 0) > 0]
        L.append(f"| {task} | {v} | {len(rs)} | {ne} | {sum(bool(x.get('success')) for x in res)} | {sum(bool(x.get('safe_success')) for x in res)} | "
                 f"{sum(bool(x.get('unsafe')) for x in res)} | {sum(1 for r in rs if r.get('outcome') == 'carried' and not r.get('any_U'))} | {dict(kinds)} | {dict(first)} | "
                 f"{len(inf)}; {sum(r['fl']['fl_n_infeasible'] for r in inf)} |")
    L += ["", "## per world"]
    for r in sorted(ok, key=lambda r: (r["task"], r["i"], r["variant"])):
        x = r.get("result") or {}
        L.append(f"- {r['task']} {r['i']} {r['variant']}: M_true {r['M_true']:.2f} Theta {[round(t, 2) for t in r['Theta']]} | checker on true mass: {r['verdict_true']['action']} "
                 f"(level {r['verdict_true']['base_level']}) | {r['outcome']} success={x.get('success')} unsafe={x.get('unsafe')} | all_U {r.get('all_U')} base {r.get('base_flagged')} | "
                 f"FL infeasible {r['fl']['fl_n_infeasible']}/{r['fl']['fl_steps']} clipped {r['fl']['fl_n_tau_clipped']}")
    for r in errs:
        L.append(f"- ERROR {r['task']} {r['i']} {r['variant']}: {r['error']}")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--variants", default="known:L1,known:L2")
    ap.add_argument("--only", default=None, help="task:i,task:i,...")
    ap.add_argument("--probe-only", action="store_true")
    a = ap.parse_args(argv)
    run(a) if a.cmd == "run" else summary(a)


if __name__ == "__main__":
    main()
