"""nero/contents/same_worlds.py -- ours (probe -> admit -> act) vs AgentChord-style (act, no checks) on the SAME worlds.  SIMULATION ONLY.
USER 11-05: "count payload rating in U and run same worlds in both methods".

Worlds: probe._world(task, probe._seed("e2e", task, i)) -- uniform on the full prior (probe.PRIOR), the same worlds as the 11-03
probe e2e run (i < 20), so ours can be checked against that run.
U for BOTH methods = carry.py's U plus the per-claw payload row (strict_claw=True, T_cap=None: |Fz| > (3 kg - gripper) g, from the
weld constraint wrench, unfiltered, every control step; no twist row).  Set in each worker by wrapping CarryEpisode.
  theirs = baseline_ungated._trial: carry at a_nom (0.5 m/s^2, ASSUMED), pose monitors, no force/payload check, no backups.
           Payload feedforward mass = prior mid-contents (ASSUMED: it has no estimate of the contents).
  ours   = probe._e2e_job: probe ladder (eps MEASURED, CALIB.json) -> static admission on the probe's set -> carry with stored
           backups, or refuse and let go in place.
    python -m world_model.v2.nero.contents.same_worlds run --tasks box,tray6 --n 20 --procs 4 --out DIR
    python -m world_model.v2.nero.contents.same_worlds summary --out DIR

11-05 options (defaults = the behaviour above, unchanged):
  --plan-source codex   OURS runs the plan WRITTEN BY THE PROPOSER MODEL (stack/proposer: propose -> schema validation -> compile to closed-chain waypoints), not the fixed
                        plan.  One plan per task (the prompt holds the task + scene facts, no hidden truth); the plan is proposed once, compiled once, then every world runs
                        probe -> static admission on the probe's set over the COMPILED segments -> carry with the compiled recoveries, or the compiled fallback.
                        A reply that is invalid or unbuildable = outcome "plan_rejected" for every world of that task (counted, with the reason).
                        Rule 2: if the probe set is not inside the plan's belief B, ours refuses ("refused (probe set outside B)"; --outside-b carry = admit anyway) and
                        records it; the re-ask itself (stack/proposer/propose.reask) is NOT run inside the workers.
        --plan-spec     backend spec (default codex:gpt-5.6-sol:high; "mock:fixture" for dry runs), --plan-cache DIR (every request is stored there; REQUIRED),
        --offline       replay only: a cache miss raises, the network is never touched (without it a miss calls the live backend).
  --theirs agentchord   THEIRS = AgentChord's cached compiled graph via agentchord_probe.nero.execute.same_world_trial(task, i, plan_dir) (--agentchord-plan DIR);
                        default "ungated" = baseline_ungated._trial (the fixed plan, no checks).  THEIRS never runs the proposer plan.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import time

from world_model.v2.stack import outcome as OC

CALIB = None          # default: probe.default_calib() = the CALIB of the active probe method (poslift: repo copy; ladder: the 11-03 file)


def _init_worker():
    """payload row in U for both methods: every CarryEpisode built in this worker gets strict_claw=True (T_cap stays None)."""
    from world_model.v2.nero.contents import carry as CR
    base = CR.CarryEpisode
    if getattr(base, "_same_worlds_strict", False):
        return

    class _Strict(base):
        _same_worlds_strict = True

        def __init__(self, *a, **k):
            k["strict_claw"] = True
            super().__init__(*a, **k)
    CR.CarryEpisode = _Strict


def CR_resolve():
    from world_model.v2.nero.contents import carry as CR
    return CR.resolve_planner(None)


def _job(args):
    method, task, i, eps_lo, eps_free, *rest = args
    ex = rest[0] if rest else {}                      # 11-05 extras (plan source / theirs mode); absent = the original behaviour
    _init_worker()
    from world_model.v2.nero.contents import probe as P, baseline_ungated as BU, carry_cert as CC
    seed = P._seed("e2e", task, i)
    if method == "ours" and ex.get("rejected"):       # the proposer's plan is invalid / unbuildable: nothing is run, the event is counted
        r = dict(task=task, i=i, seed=seed, outcome="plan_rejected", any_U=False, plan_rejected=ex["rejected"], theta_sound=None)
    elif method == "ours":
        pl = ex.get("plan")
        r = P._e2e_job((task, i, eps_lo, eps_free) + ((pl,) if pl else ()), method=ex.get("probe_method"))
    elif ex.get("theirs") == "agentchord":
        from world_model.v2.agentchord_probe.nero import execute as AX
        r = AX.same_world_trial(task, i, ex["agentchord_plan"])
    else:
        BU.seed_of = lambda t, c, k: P._seed("e2e", t, k)          # same world as ours: CC.sample_world(task, PRIOR[task], rng(seed))
        lo, hi = P.PRIOR[task]
        r = BU._trial((task, (lo, hi), i, 0.5 * (lo + hi), True, CC.WELD_STIFF))
    r.update(method=method, seed_e2e=seed, planner=CR_resolve())
    if method == "ours" and ex.get("plan_source"):
        r.update(plan_source=ex["plan_source"], plan_id=ex.get("plan_id"), plan_record=ex.get("plan_record"))
        if ex.get("plan"):
            r["planner"] = "closed_chain"                 # a proposer plan is always planned by the closed-chain planner (CarryEpisode(wp=...))
    return r


def _first_U(r):
    if r.get("method") == "theirs":
        v = r.get("first_U")
        return v and f"{v.get('seg')}:{v.get('kind')}"
    if r.get("probe_viol"):
        return f"probe:{r['probe_viol'].get('kind')}"
    for s in r.get("carry", {}).get("segments", []):
        v = s.get("nominal", {}).get("viol")
        if v:
            return f"{s.get('seg')}:{v.get('kind')}"
    return None


def summary(out):
    out = pathlib.Path(out)
    rows = [json.loads(l) for l in open(out / "trials.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"], r["method"])].append(r)
    lines = ["# same worlds: ours (probe -> admit -> act) vs AgentChord-style (act)  -- machine-written by same_worlds.summary", "",
             "success = finished AND the task's goal terms (AgentChord's rule, USER 11-09, stack/tasks.py); safety is reported separately.  U includes the per-claw payload row (|Fz| > (3 kg - gripper) g, unfiltered). Worlds: probe e2e seeds, full prior.", "",
             "| task | method | n | errors | success (finished AND goal) | safe_success (success AND no U) | unsafe (any U) | at rest in R | refused | U kinds (first per run) | truth in probe set |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    n_rej = {k: sum(r.get("outcome") == "plan_rejected" for r in rs) for k, rs in by.items()}
    for (task, method), rs in sorted(by.items()):
        ok = [r for r in rs if "error" not in r]
        kinds = collections.Counter(_first_U(r) for r in ok if r.get("any_U"))
        t = OC.tally(ok)
        f = lambda k: f"{t[k]}" + (f" (+{t[k + '_unknown']} unknown)" if t.get(k + "_unknown") else "")
        sound = f"{sum(bool(r.get('theta_sound')) for r in ok)}/{len(ok)}" if method == "ours" else "-"
        lines.append(f"| {task} | {method} | {len(rs)} | {len(rs) - len(ok)} | {f('success')} | {f('safe_success')} | {f('unsafe')} | {f('in_R')} | {t['refused']} "
                     f"| {dict(kinds) or '-'} | {sound} |")
    if any(n_rej.values()):                              # proposer plans that could not be used (never silently dropped)
        lines += ["", "Ours, proposer plan REJECTED (counted in n, not in carried / refused): " + ", ".join(f"{k[0]} {v}/{len(by[k])}" for k, v in sorted(n_rej.items()) if v)]
    for (task, method), rs in sorted(by.items()):
        pb = [r for r in rs if r.get("probe_in_B")]
        if method == "ours" and pb:
            x0 = pb[0]["probe_in_B"]
            cov = sum(bool(r["probe_in_B"].get("covers_prior")) for r in pb)
            lines.append(f"{task}: probe set inside the proposer's belief B (B ∩ prior) in {sum(bool(r['probe_in_B']['inside']) for r in pb)}/{len(pb)} probed worlds "
                         f"(plan {pb[0].get('plan_id')}, model {((pb[0].get('plan_record') or {}).get('model'))}; B on {x0['param']} = {x0['B']}, prior {x0.get('prior')}).")
            if cov:
                lines.append(f"{task}: B covers the whole prior in {cov}/{len(pb)} of them: there Rule 2 cannot trigger, so 'inside' is no evidence.")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _proposer_plans(tasks, a, out):
    """--plan-source codex: one proposer plan per task (propose -> validate -> compile + build).  -> {task: extras dict for the 'ours' jobs}.  Written to OUT/plans/<task>.json
    (reply, record, compiled plan or the rejection).  A cache miss with --offline raises ProposerCacheMiss (no fallback to the network)."""
    from world_model.v2.stack.proposer import compile as PC, propose as PP, schema as PS
    (out / "plans").mkdir(exist_ok=True)
    res = {}
    for t in tasks:
        plan, rec = PP.propose(t, backend_spec=a.plan_spec, cache_dir=a.plan_cache, offline=a.offline)
        small = {k: rec.get(k) for k in ("kind", "spec", "model", "reasoning", "backend", "request_hash", "raw_reply_path", "status", "stage", "reasons", "plan_id", "cached")}
        item = dict(plan_source="codex", plan_record=small, plan_id=plan.get("plan_id"))             # full record (caveats, prompt hash) in OUT/plans/<task>.json
        doc = dict(plan=plan, record=rec)
        try:
            cp = PC.compile_plan(plan, t)
            item["plan"] = cp.job_plan(a.outside_b)
            doc["compiled"] = cp.to_json()
        except PS.PlanRejected as e:
            item["rejected"] = e.to_dict()
            doc["rejected"] = e.to_dict()
        if plan.get("status") != "ok":
            item["rejected"] = dict(status="plan_rejected", stage=plan.get("stage"), reasons=plan.get("reasons"))
            doc["rejected"] = item["rejected"]
        (out / "plans" / f"{t}.json").write_text(json.dumps(doc, indent=1, default=str))
        print(f"plan {t}: {item['plan_record']['status']}"
              + (f" REJECTED [{item['rejected']['stage']}] {item['rejected']['reasons']}" if item.get("rejected") else f" id {item['plan_id']} segments {item['plan']['wp'].names}"), flush=True)
        res[t] = item
    return res


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--tasks", default="box,tray6")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--calib", default=CALIB, help="default: the CALIB of the active probe method")
    ap.add_argument("--probe-method", choices=("poslift", "ladder"), default=None, help="default: env BIMANUAL_PROBE, else poslift")
    ap.add_argument("--planner", choices=("scripted", "closed_chain"), default=None,
                    help="plan BOTH methods execute and are admitted on (carry.resolve_planner; default = env BIMANUAL_PLANNER, else scripted = the legacy PLANS)")
    ap.add_argument("--plan-source", choices=("fixture", "codex"), default="fixture",
                    help="plan OURS executes: fixture = the fixed plan (default, today); codex = the plan the proposer model writes (stack/proposer)")
    ap.add_argument("--plan-spec", default="codex:gpt-5.6-sol:high", help="proposer backend spec (pan/llm.py), e.g. mock:fixture for a dry run")
    ap.add_argument("--plan-cache", help="proposer response cache dir (required with --plan-source codex)")
    ap.add_argument("--offline", action="store_true", help="proposer: replay from --plan-cache only; a miss fails loudly, the network is never called")
    ap.add_argument("--outside-b", choices=("refuse", "carry"), default="refuse", help="probe set not inside the plan's belief B: refuse (default) or carry anyway (recorded)")
    ap.add_argument("--theirs", choices=("ungated", "agentchord"), default="ungated")
    ap.add_argument("--agentchord-plan", help="dir (or plan.json) of AgentChord's cached compiled graph, for --theirs agentchord")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        return summary(a.out)
    if a.planner is not None:
        os.environ["BIMANUAL_PLANNER"] = a.planner               # workers are spawned after this: they inherit it (one plan for ours and theirs)
    from world_model.v2.nero.contents import carry as CR0, probe as PR0
    planner = CR0.resolve_planner(None)
    if a.probe_method is not None:
        os.environ["BIMANUAL_PROBE"] = a.probe_method            # workers inherit it
    probe_method = a.probe_method or PR0.PROBE_METHOD
    a.calib = a.calib or PR0.default_calib(probe_method)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = PR0.load_calib(a.calib, probe_method)["eps"]          # as probe.main e2e reads it (asserts CALIB method == probe method)
    eps_lo, eps_free = float(cal["EPS_LO"]), float(cal["EPS_FREE"])
    tasks = a.tasks.split(",")
    ex = {t: {m: {} for m in ("theirs", "ours")} for t in tasks}                     # per task / method extras (empty = the original job tuple)
    cfg = {}
    if a.theirs == "agentchord":
        try:
            from world_model.v2.agentchord_probe.nero import execute as AX
            AX.same_world_trial                                                        # noqa: B018 - existence check
        except Exception as e:                                                         # loud, before any job starts
            raise SystemExit(f"--theirs agentchord needs world_model.v2.agentchord_probe.nero.execute.same_world_trial ({e!r})")
        if not a.agentchord_plan:
            raise SystemExit("--theirs agentchord needs --agentchord-plan DIR")
        for t in tasks:
            ex[t]["theirs"] = dict(theirs="agentchord", agentchord_plan=a.agentchord_plan)
        cfg.update(theirs="agentchord", agentchord_plan=a.agentchord_plan)
    if a.plan_source == "codex":
        if not a.plan_cache:
            raise SystemExit("--plan-source codex needs --plan-cache DIR (every request and reply is stored there; with --offline it is replayed)")
        plans = _proposer_plans(tasks, a, out)
        for t in tasks:
            ex[t]["ours"] = plans[t]
        cfg.update(plan_source="codex", plan_spec=a.plan_spec, plan_cache=a.plan_cache, offline=a.offline, outside_b=a.outside_b,
                   plans={t: dict(status=plans[t]["plan_record"]["status"], plan_id=plans[t].get("plan_id"), rejected=plans[t].get("rejected")) for t in tasks})
    for t in tasks:
        ex[t]["ours"] = {**ex[t]["ours"], "probe_method": probe_method}      # explicit argument of probe._e2e_job (no module-global patching)
    jobs = [(m, t, i, eps_lo, eps_free) + ((ex[t][m],) if ex[t][m] else ()) for i in range(a.n) for t in tasks for m in ("theirs", "ours")]
    (out / "CONFIG.json").write_text(json.dumps(dict(tasks=a.tasks, n=a.n, procs=a.procs, calib=a.calib, probe_method=probe_method, probe_assumptions=PR0.method_assumptions(probe_method), eps_lo=eps_lo,
                                                     eps_free=eps_free, u_payload="strict_claw=True, T_cap=None", planner=planner, **cfg), indent=1, default=str))
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, initializer=_init_worker, maxtasksperchild=3) as pool, \
            open(out / "trials.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"{r.get('method')} {r.get('task')} i={r.get('i')} anyU={r.get('any_U')} {r.get('outcome', '')} "
                  f"{'ERR ' + r['error'] if 'error' in r else ''} t={time.time() - t0:.0f}s", flush=True)
    print(f"wall {time.time() - t0:.0f} s", flush=True)
    summary(out)


if __name__ == "__main__":
    main()
