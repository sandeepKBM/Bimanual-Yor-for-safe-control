"""nero/coffee_run.py -- (moved from nero/contents/: it drives AgentChord too, which nero/contents may not import) driver of the coffee-tray comparison (USER 11-09).  SIMULATION ONLY (the only network call = the Codex proposer / AgentChord plan, cached).

  plan-ours   --variant known|unknown --out DIR [--spec codex:gpt-5.6-sol:high] [--offline]     one proposer plan (adapters/coffee_proposer.py), cached in DIR/plan_cache
  plan-theirs --variant ... --out DIR [--spec ...]                                              one AgentChord plan (their Task/Recovery/Compile agents via the loopback gateway)
  run         --variant ... --out DIR --n 20 --p 0.0 --methods ours,theirs --procs K            n worlds (probe._seed("e2e","tray0",i)), both methods on the SAME worlds
  summary     --out DIR                                                                         tables
ONE model spec for both methods (--spec, default codex:gpt-5.6-sol:high); every record carries it.  The weigh probe is the stack's fixed skill (probe.make_probe), not chosen by the model.
Env: BIMANUAL_COFFEE_VARIANT is set from --variant before any worker starts.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

DEFAULT_SPEC = "codex:gpt-5.6-sol:high"
OUT_DEFAULT = "/common/users/ss5772/bimanual_data/contents/coffee_tray_nero_2026-11-09"


def _bind_theirs(ACC):
    """AgentChord's coffee module gets the sim modules from here (dependency rule: agentchord_probe/nero does not import nero/contents)."""
    from world_model.v2.nero.contents import carry as CR, coffee as CF, envelope as E, probe as P, same_worlds as SW
    ACC.bind(CF, CR, E, P, SW)
    return ACC


def vdir(out, variant):
    return pathlib.Path(out) / variant


def cmd_plan_ours(a):
    os.environ["BIMANUAL_COFFEE_VARIANT"] = a.variant
    from world_model.v2.adapters import coffee_proposer as PCF
    d = vdir(a.out, a.variant)
    (d / "ours_plan_cache").mkdir(parents=True, exist_ok=True)
    plan, cp, rec, info = PCF.propose_coffee_until_accepted(backend_spec=a.spec, cache_dir=str(d / "ours_plan_cache"), offline=a.offline)
    doc = dict(plan=plan, record=rec, reask=info)                  # re-asks after every rejection, no cap (USER 11-09); "rejected" only after a proposer error
    if cp is not None:
        doc["compiled"] = cp.to_json()
    else:
        doc["rejected"] = dict(status="plan_rejected", stage=plan.get("stage"), reasons=plan.get("reasons"))
    (d / "ours_plan.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(dict(status=plan.get("status"), plan_id=plan.get("plan_id"), rejected=doc.get("rejected"), n_reasks=info["n_reasks"], reask_attempts=info["attempts"], model=rec.get("model"), reasoning=rec.get("reasoning"),
                          steps=[(s["skill"], s["args"]) for s in plan.get("steps", [])]), default=str))


def cmd_plan_theirs(a):
    os.environ["BIMANUAL_COFFEE_VARIANT"] = a.variant
    from world_model.v2.agentchord_probe.nero import coffee as ACC
    _bind_theirs(ACC)
    d = vdir(a.out, a.variant) / "theirs_plan"
    meta = ACC.make_plan(d, spec=a.spec)
    print(json.dumps({k: meta[k] for k in ("task", "plan_id", "backend", "models", "proposer", "n_llm_calls", "n_errors", "plan_wall_s")}, default=str))


def _ours_job(args):
    """one ours episode: capsule phase -> (probe -> admission) -> carry of the proposer's compiled plan, via probe._e2e_job (stack rule: probe only when the prior is not admissible)."""
    import numpy as np
    from world_model.v2.nero.contents import coffee as CF, same_worlds as SW, probe as P
    CF.install_ours()
    SW._init_worker()
    i, p, d = args["i"], args["p"], pathlib.Path(args["dir"])
    doc = json.loads((d / "ours_plan.json").read_text())
    t0 = time.time()
    CF.JOB.update(p=p, inj_seed=int(np.random.SeedSequence([55, 1 if CF.variant() == "known" else 2, i, int(round(100 * p))]).generate_state(1)[0]), cap_viol=None, inj=None, ep=None)
    if "compiled" not in doc:                              # proposer plan rejected: counted for every world (as same_worlds)
        r = dict(task=CF.TASK, i=i, outcome="plan_rejected", any_U=False, plan_rejected=doc.get("rejected"), result=dict(success=False, unsafe=False, in_R=False, safe_success=False, refused=False, finished=False))
    else:
        from world_model.v2.adapters import coffee_proposer as PCF
        PCF.install()
        from world_model.v2.stack.proposer import compile as PC
        cp = PC.CompiledPlan.from_json(doc["compiled"])
        cal = P.load_calib(P.default_calib(), P.PROBE_METHOD)["eps"]
        r = P._e2e_job((CF.TASK, i, cal["EPS_LO"], cal["EPS_FREE"], cp.job_plan("refuse")))
        if "error" not in r:
            cv = CF.JOB.get("cap_viol")
            r["any_U"] = bool(r.get("any_U") or cv)
            r["capsule_U"] = cv
            res = r.get("result")
            if cv and res is not None:                         # the capsule phase has its own track (reset before the tray carry): fold its U into the outcome
                res["unsafe"], res["safe_success"] = True, False
            r["coffee"] = CF.record_extras()
            ep = CF.JOB.get("ep")
            if ep is not None and res is not None:             # second named goal set (AgentChord's tray rule), same stack.goal / stack.outcome path
                g = CF.general_result(ep, res.get("finished"), bool(r["any_U"]))
                r["goal_general"], r["result_general"] = g["goal"], g["result"]
    rec = r.get("record", {}) or {}
    w = CF.world_for(i)                                          # TRUTH for analysis only: the world's true hidden load; never passed to the planner / probe
    r.update(load_kg=w.load_kg, load_xy=list(map(float, w.load_xy)))
    r.update(method="ours", variant=CF.variant(), i=i, p=p, seed_e2e=P._seed("e2e", CF.TASK, i), plan_id=doc["plan"].get("plan_id"), n_reasks=(doc.get("reask") or {}).get("n_reasks"), plan_spec=doc["record"].get("spec"),
             plan_model=doc["record"].get("model"), plan_reasoning=doc["record"].get("reasoning"), job_wall_s=round(time.time() - t0, 1))
    if r.get("coffee"):
        r["coffee"] = dict(r["coffee"], cap_log=None)
    return r


def _theirs_job(args):
    from world_model.v2.agentchord_probe.nero import coffee as ACC
    _bind_theirs(ACC)
    return ACC.run_world(dict(plan=str(pathlib.Path(args["dir"]) / "theirs_plan" / "plan.json"), i=args["i"], p=args["p"],
                              inj_seed=int(__import__("numpy").random.SeedSequence([56, 1 if args["variant"] == "known" else 2, args["i"], int(round(100 * args["p"]))]).generate_state(1)[0])))


def _job(args):
    try:
        r = _ours_job(args) if args["method"] == "ours" else _theirs_job(args)
    except Exception as e:
        import traceback
        r = dict(method=args["method"], i=args["i"], error=repr(e), tb=traceback.format_exc()[-800:])
    r.update(variant=args["variant"], p=args["p"])
    return r


def cmd_run(a):
    import multiprocessing as mp
    os.environ["BIMANUAL_COFFEE_VARIANT"] = a.variant
    d = vdir(a.out, a.variant)
    tag = f"p{int(round(100 * a.p)):02d}"
    jobs = [dict(method=m, i=i, p=a.p, dir=str(d), variant=a.variant) for i in range(a.first, a.first + a.n) for m in a.methods.split(",")]
    (d / f"CONFIG_{tag}.json").write_text(json.dumps(dict(variant=a.variant, p=a.p, n=a.n, first=a.first, methods=a.methods, spec=a.spec, procs=a.procs, date=time.strftime("%Y-%m-%d"),
                                                          assumptions=__import__("world_model.v2.nero.contents.coffee", fromlist=["x"]).ASSUMPTIONS), indent=1, default=str))
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(d / f"trials_{tag}.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"{r.get('method')} v={r.get('variant')} p={r.get('p')} i={r.get('i')} outcome={r.get('outcome')} {(r.get('result') or {})} {'ERR ' + r['error'] if 'error' in r else ''} t={time.time() - t0:.0f}s", flush=True)


def cmd_summary(a):
    import collections
    out = pathlib.Path(a.out)
    rows = []
    for f in sorted(out.glob("*/trials_p*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r.get("variant"), r.get("p"), r.get("method"))].append(r)
    L = ["| variant | p | method | n | errors | success (strict) | success (general goal set) | safe_success | unsafe (any U) | refused | drops applied / atomic actions | U kinds | tray tilt max deg |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(by, key=lambda k: (str(k[0]), k[1] or 0, str(k[2]))):
        rs = by[k]
        ok = [r for r in rs if "error" not in r]
        g = lambda f: sum(bool(f(r)) for r in ok)
        res = lambda r: r.get("result") or {}
        ms = lambda r: r.get("measured") or {}
        kinds = collections.Counter()
        for r in ok:
            if r.get("any_U"):
                v = (r.get("first_U") or {}).get("kind") or (r.get("probe_viol") or {}).get("kind") or r.get("capsule_U") and r["capsule_U"][0].get("kind")
                if not v:
                    for sg in (r.get("carry") or {}).get("segments", []):
                        if sg["nominal"]["viol"]:
                            v = sg["nominal"]["viol"]["kind"]
                            break
                kinds[v] += 1
        inj = [((r.get("coffee") or {}).get("inj") or r.get("injections") or []) for r in ok]
        nd = sum(sum(1 for x in l if x.get("dropped")) for l in inj)
        na = sum(len(l) for l in inj)
        tl = [ms(r).get("tilt_deg") for r in ok if ms(r).get("tilt_deg") is not None]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {len(rs) - len(ok)} | {g(lambda r: res(r).get('success'))} | {g(lambda r: (r.get('result_general') or {}).get('success'))} | {g(lambda r: res(r).get('safe_success'))} | "
                 f"{g(lambda r: res(r).get('unsafe') or r.get('any_U'))} | {g(lambda r: res(r).get('refused') or r.get('outcome') in ('refused_in_R', 'refused_bad'))} | {nd}/{na} | {dict(kinds) or '-'} | {max(tl) if tl else '-'} |")
    L += ["", "ours (probe -> admission -> carry): probe stop reasons, final interval of the TOTAL carried mass (empty = tray + capsule 0.312 kg), refusals", "",
          "| variant | p | n | probe stop (why) | truth inside Theta | Theta width kg p50 / max | outcomes |", "|---|---|---|---|---|---|---|"]
    for k in sorted(by, key=lambda k: (str(k[0]), k[1] or 0, str(k[2]))):
        if k[2] != "ours":
            continue
        ok = [r for r in by[k] if "error" not in r and r.get("Theta")]
        if not ok:
            continue
        import numpy as _np
        w = [r["Theta"][1] - r["Theta"][0] for r in ok]
        L.append(f"| {k[0]} | {k[1]} | {len(ok)} | {dict(collections.Counter(r.get('why') for r in ok))} | {sum(bool(r.get('theta_sound')) for r in ok)}/{len(ok)} | {_np.median(w):.3f} / {max(w):.3f} | {dict(collections.Counter(r.get('outcome') for r in by[k]))} |")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan-ours", "plan-theirs", "run", "summary"):
        p = sp.add_parser(name)
        p.add_argument("--variant", choices=("known", "unknown"))
        p.add_argument("--out", default=OUT_DEFAULT)
        p.add_argument("--spec", default=DEFAULT_SPEC)
        p.add_argument("--offline", action="store_true")
        if name == "run":
            p.add_argument("--n", type=int, default=20)
            p.add_argument("--p", type=float, default=0.0)
            p.add_argument("--methods", default="ours,theirs")
            p.add_argument("--procs", type=int, default=4)
            p.add_argument("--first", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "plan-ours":
        return cmd_plan_ours(a)
    if a.cmd == "plan-theirs":
        return cmd_plan_theirs(a)
    if a.cmd == "run":
        return cmd_run(a)
    return cmd_summary(a)


if __name__ == "__main__":
    main()
