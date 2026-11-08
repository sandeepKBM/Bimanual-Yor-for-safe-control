"""nero/contents/segment_report.py -- per-SEGMENT view of a same_worlds run (USER 11-05: "the plan is a bunch of segments; we show our
things within that: the guarantees, recoveries and all").  Machine-written; reads trials.jsonl, writes SEGMENTS.md.  No simulation.

Per task and segment, for OURS (runs that were admitted and carried):
  guarantee  the admission verdict that let the run start (static rows, proof under MODEL over the probe's set; decision.u_nom,
             decision.a_admit) -- the same verdict covers every segment of the plan
  nominal    runs that reached the segment, runs that entered U in it (sampled), worst torque use |tau|/limit, worst object tilt [deg],
             min arm-arm and arm-static clearance [m] (FL-MPC residuals)
  recovery   stored backups tested from this segment (branch points x runs) and how many ended in U or out of R (sampled)
For THEIRS: runs that entered U, by first segment and kind.
    python -m world_model.v2.nero.contents.segment_report --run DIR
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib


def _fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def report(run):
    run = pathlib.Path(run)
    rows = [json.loads(l) for l in open(run / "trials.jsonl")]
    tasks = sorted({r["task"] for r in rows})
    L = [f"# Per-segment view -- machine-written by contents/segment_report.py from {run}/trials.jsonl", "",
         "Guarantee = admission before the run (static rows, proof under MODEL over the probe's set). Nominal and recovery columns are "
         "SAMPLED (n runs). Recovery = stored backups started from branch points in the segment.", ""]
    for task in tasks:
        ours = [r for r in rows if r["task"] == task and r.get("method") == "ours" and "error" not in r]
        theirs = [r for r in rows if r["task"] == task and r.get("method") == "theirs" and "error" not in r]
        car = [r for r in ours if r.get("outcome") == "carried" or r.get("carry")]
        outc = collections.Counter(str(r.get("outcome")) for r in ours)
        L += [f"## {task}", "",
              f"Ours: n {len(ours)}; outcomes {dict(outc)}; probe set contains truth {sum(bool(r.get('theta_sound')) for r in ours)}/{len(ours)}; "
              f"U during probe {sum(bool(r.get('probe_viol')) for r in ours)}.", ""]
        pl = [r for r in ours if r.get("plan_source")]
        if pl:                                            # 11-05: the plan was written by the proposer model; its segments (named after its steps) are the rows below
            rej = sum(r.get("outcome") == "plan_rejected" for r in pl)
            rc = pl[0].get("plan_record") or {}
            L.append(f"Plan source: proposer model {rc.get('model')} ({rc.get('reasoning')}), spec {rc.get('spec')}, plan id {pl[0].get('plan_id')}, "
                     f"reply {rc.get('raw_reply_path')}; plan_rejected in {rej}/{len(pl)} worlds"
                     + (f" ({pl[0]['plan_rejected'].get('stage')}: {pl[0]['plan_rejected'].get('reasons')})" if rej else "") + ".")
            pb = [r["probe_in_B"] for r in pl if r.get("probe_in_B")]
            if pb:
                L.append(f"Probe set inside the plan's belief B (B ∩ prior): {sum(bool(x['inside']) for x in pb)}/{len(pb)} probed worlds (B on {pb[0]['param']} = {pb[0]['B']}, "
                         f"prior {pb[0].get('prior')}; other B parameters are not probed).")
                cov = sum(bool(x.get("covers_prior")) for x in pb)
                if cov:
                    L.append(f"B covers the whole prior in {cov}/{len(pb)} of them: there Rule 2 cannot trigger, so 'inside' is no evidence.")
            L.append("")
        if car:
            u = [r["decision"].get("u_nom") for r in car if r.get("decision")]
            a = [r["decision"].get("a_admit") for r in car if r.get("decision")]
            L.append(f"Guarantee at admission (carried runs): u_nom max {_fmt(max(u) if u else None)} (admit if <= U_ADMIT), "
                     f"a_admit {sorted(set(a))} m/s^2.")
            L += ["", "| segment | runs | U in segment | tau use max | obj tilt max deg | min arm-arm m | min arm-static m | backups tested | backups failed |",
                  "|---|---|---|---|---|---|---|---|---|"]
            segs = collections.OrderedDict()
            for r in car:
                for s in r["carry"].get("segments", []):
                    d = segs.setdefault(s["seg"], dict(n=0, U=collections.Counter(), tau=[], tilt=[], aa=[], st=[], bt=0, bf=0))
                    d["n"] += 1
                    nom = s.get("nominal", {})
                    if nom.get("viol"):
                        d["U"][nom["viol"].get("kind")] += 1
                    d["tau"].append(nom.get("tau_ratio"))
                    d["tilt"].append(nom.get("obj_tilt_max"))
                    fl = s.get("fl", {})
                    d["aa"].append(fl.get("fl_res_armarm_min_m"))
                    d["st"].append(fl.get("fl_res_static_min_m"))
                    for fam in s.get("family", {}).values():
                        d["bt"] += int(fam.get("n", 0))
                        d["bf"] += int(fam.get("n_bad", 0))
            for name, d in segs.items():
                mx = lambda v: max([x for x in v if x is not None], default=None)
                mn = lambda v: min([x for x in v if x is not None], default=None)
                L.append(f"| {name} | {d['n']} | {sum(d['U'].values())} {dict(d['U']) or ''} | {_fmt(mx(d['tau']))} | {_fmt(mx(d['tilt']), 2)} | "
                         f"{_fmt(mn(d['aa']))} | {_fmt(mn(d['st']))} | {d['bt']} | {d['bf']} |")
        else:
            L.append("No carried runs (nothing admitted).")
        tu = collections.Counter((r.get("first_U") or {}).get("seg", "-") + ":" + str((r.get("first_U") or {}).get("kind", "-"))
                                 for r in theirs if r.get("any_U"))
        L += ["", f"Theirs (AgentChord-style, no checks): n {len(theirs)}; entered U {sum(bool(r.get('any_U')) for r in theirs)}; "
                  f"first U by segment:kind {dict(tu) or '-'}.", ""]
    (run / "SEGMENTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    report(ap.parse_args(argv).run)


if __name__ == "__main__":
    main()
