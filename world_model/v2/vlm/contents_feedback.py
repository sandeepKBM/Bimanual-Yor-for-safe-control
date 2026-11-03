"""vlm/contents_feedback.py -- does a VLM do better once a probe proves its mass belief wrong? (11-03)  SIMULATION ONLY.

Inputs: contents_eval records (image + facts + truth), one model's first answers (contents_eval ask), and probe traces for the same
worlds (nero/contents/probe_trace.py: the soft-to-strong ladder from the FULL prior, VLM-independent; every rung's reading and the
sound probe bound P_k = [lo_k, hi_k] on the total carried mass).

Replay per record: walk the rungs; the current belief V is FALSIFIED at rung k when P_k and V do not overlap.  Then the same model is
asked again with the image, the facts, its previous interval and the measurements so far (plus the sound bound they prove), up to
--max-reask times.  The ladder itself never uses V (it always starts soft from the prior), so the probe stays safe whatever the
model says; only the belief and the decision change.

Arms scored on the same worlds (guarantee type: none of the VLM intervals is sound; only P is):
  vlm0   : the first answer alone (no probe)
  probe  : P at the end of the ladder (free -> interval; else lower bound only)
  reask  : the model's last answer, and its intersection with P (at the last ask, and at the end of the ladder)
Decision: carry iff belief contents hi <= the static carry ceiling (probe.c_admit_max, MODEL).  Truth admissible iff true contents
<= ceiling.  false_safe = carry decided but truth not admissible.

    python -m world_model.v2.vlm.contents_feedback run --records R --answers A --traces T --ceilings C --endpoint URL --model M --out O
    python -m world_model.v2.vlm.contents_feedback score --runs O1 [O2 ...] --md out.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from world_model.v2.vlm.contents_eval import clean, parse_last_json

REASK_VERSION = "reask-v1-2026-11-03"
REASK = """A two-armed robot will lift the object in the image and must NOT underestimate the load.
Known facts: {facts}
Earlier you estimated the total mass (everything lifted: {cdef} plus the empty object) as [{v_lo:.2f}, {v_hi:.2f}] kg.
The robot then tested this with a gentle force probe while the object still rested on the counter: both arms held it and pushed up
with a slowly increasing lift force, reading the load they carried from their joint torques (read error at most about +/-{e:.2f} kg).
{lines}
These measurements prove: {proof}.
Your earlier estimate is inconsistent with these measurements. Give a revised 90% interval using the image, the facts and the
measurements. Too narrow an interval that misses the truth is worse than a wide one.
Reply with ONE JSON object and nothing after it:
{{"total_mass_kg": [lo, hi], "contents_kg": [lo, hi], "p_cover": <probability the true total mass is inside>, "evidence": "<=25 words"}}"""


def overlap(a, b, tol=1e-6):
    return a[0] <= b[1] + tol and b[0] <= a[1] + tol


def inter(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return [lo, hi] if lo <= hi else None


def evidence(rungs, k, e):
    lines = []
    for j, r in enumerate(rungs[:k + 1]):
        if r["free"]:
            lines.append(f"  step {j + 1}: lift force = weight of {r['m_ff']:.2f} kg -> the object lifted clear of the counter; "
                         f"measured mass {r['L']:.2f} kg.")
        else:
            lines.append(f"  step {j + 1}: lift force = weight of {r['m_ff']:.2f} kg -> arms carried {r['L']:.2f} kg; "
                         f"the object stayed on the counter.")
    lo, hi = rungs[k]["Theta"]
    proof = (f"the total mass is between {lo:.2f} and {hi:.2f} kg" if rungs[k]["free"]
             else f"the total mass is at least {lo:.2f} kg (no upper bound yet beyond the facts: at most {hi:.2f} kg)")
    return "\n".join(lines), proof


def replay(rec, ans, trace, ceil, empty, ask, max_reask, e_read):
    """one record: walk the probe trace against the model's belief, re-ask on falsification."""
    task, truth = rec["task"], rec["truth"]
    M, C = truth["total_mass_kg"], truth["contents_kg"]
    rungs = trace["rungs"]
    V0 = ans["parsed"]["total_mass_kg"] if ans.get("parsed") else None
    V, events = V0, []
    for k, r in enumerate(rungs):
        P = r["Theta"]
        tries = 0
        while V is not None and not overlap(V, P) and len([x for x in events if x["kind"] == "ask"]) < max_reask:
            lines, proof = evidence(rungs, k, e_read)
            text = REASK.format(facts=rec["facts"], cdef=rec["contents_def"], v_lo=V[0], v_hi=V[1], e=e_read, lines=lines, proof=proof)
            raw, fin, sec, err = ask(rec, text)
            o = clean(parse_last_json(raw, fin))
            Vn = o["total_mass_kg"] if o else None
            events.append(dict(kind="ask", rung=k, P=P, V_old=V, V_new=Vn, consistent=(Vn is not None and overlap(Vn, P)),
                               covers=(Vn is not None and Vn[0] <= M <= Vn[1]), raw=raw[-1500:], finish=fin, latency_s=round(sec, 2), error=err))
            tries += 1
            if Vn is None:
                break                                           # no answer: keep the falsified belief, stop asking at this rung
            V = Vn
        if V is not None and not overlap(V, P):
            events.append(dict(kind="falsified_unresolved", rung=k, P=P, V=V))
    P_end = rungs[-1]["Theta"] if rungs else None
    asks = [x for x in events if x["kind"] == "ask"]
    P_ask = asks[-1]["P"] if asks else P_end                       # never re-asked: the ladder ran to its end
    m_e = empty

    def decide(B):
        if B is None:
            return None
        return bool(B[1] - m_e <= ceil + 1e-9)

    admissible = bool(C <= ceil + 1e-9)
    beliefs = dict(vlm0=V0, probe=P_end, reask=V, reask_and_probe_end=(inter(V, P_end) if V and P_end else None),
                   reask_and_probe_at_ask=(inter(V, P_ask) if (V and P_ask) else None))
    arms = {}
    for name, B in beliefs.items():
        d = decide(B)
        arms[name] = dict(B=B, covers=(B is not None and B[0] - 1e-9 <= M <= B[1] + 1e-9), width=(None if B is None else B[1] - B[0]),
                          carry=d, false_safe=(d is True and not admissible), missed=(d is False and admissible))
    return dict(id=rec["id"], task=task, view=rec["view"], M_true=M, C_true=C, ceiling=ceil, admissible=admissible,
                n_rungs=len(rungs), trace_why=trace.get("why"), rung_falsified=(asks[0]["rung"] if asks else None),
                rungs_at_last_ask=(asks[-1]["rung"] + 1 if asks else None), n_asks=len(asks), events=events, arms=arms)


def cmd_run(a):
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(a.records)}
    answers = [json.loads(l) for l in open(a.answers)]
    traces = {}
    for l in open(a.traces):
        t = json.loads(l)
        if "error" not in t:
            traces[(t["task"], int(t["seed"]))] = t
    cj = json.loads(pathlib.Path(a.ceilings).read_text())
    ceil, empty = cj["c_admit_max"], cj["empty"]
    from world_model.v2.vlm.openai_backend import OpenAIChat
    cl = OpenAIChat(a.model, base_url=a.endpoint, timeout_s=a.timeout)

    def ask(r, text):
        t0 = time.time()
        try:
            res = cl.run_one(dict(id=r["id"], images=[r["image"]], text=text, thinking=(a.thinking == "on")), a.max_tokens)
            return res["raw_texts"][0], res["finish"][0], res["seconds"], None
        except Exception as e:
            return "", "error", time.time() - t0, repr(e)

    jobs = []
    for an in answers:
        r = recs[an["id"]]
        t = traces.get((r["task"], int(r["seed"])))
        if t is None:
            continue
        mt = abs(t["M_true"] - r["truth"]["total_mass_kg"])
        assert mt < 1e-6, f"{r['id']}: trace world != record world ({mt})"
        jobs.append((r, an, t))
    print(f"{len(jobs)} records with traces", flush=True)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(replay, r, an, t, ceil[r["task"]], empty[r["task"]], ask, a.max_reask, a.e_read) for r, an, t in jobs]
        for fu in futs:
            row = fu.result()
            row.update(model=a.model, reask_version=REASK_VERSION, first_prompt=answers[0].get("prompt_version"), thinking=a.thinking)
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
            print(row["id"], row["n_asks"], row["arms"]["reask"]["covers"], flush=True)


def _pct(x):
    return "-" if not x else f"{100 * np.mean(x):.0f}%"


def cmd_score(a):
    lines = [f"# VLM re-ask after probe falsification ({REASK_VERSION})", "",
             "Guarantee: only `probe` is sound (MEASURED read error, sampled n = 20/task). VLM intervals are not.",
             "Decision: carry iff belief contents hi <= static ceiling (MODEL).", ""]
    for p in a.runs:
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            continue
        lines += [f"## {rows[0]['model']}  first prompt {rows[0]['first_prompt']}  thinking {rows[0]['thinking']}  ({p})", ""]
        lines += ["| task | n | falsified | re-ask consistent w/ probe | V0 covers (falsified) | V1 covers | probe width @ask | V1 width | V1∩P width | V1∩P covers |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for task in sorted({r["task"] for r in rows}) + ["all"]:
            R = [r for r in rows if task == "all" or r["task"] == task]
            F = [r for r in R if r["n_asks"]]
            first = [r["events"][0] for r in F if r["events"] and r["events"][0]["kind"] == "ask"]
            v1 = [e for e in first if e["V_new"]]
            pw = [e["P"][1] - e["P"][0] for e in v1]
            vw = [e["V_new"][1] - e["V_new"][0] for e in v1]
            iv = [inter(e["V_new"], e["P"]) for e in v1]
            lines.append(f"| {task} | {len(R)} | {len(F)} | {_pct([e['consistent'] for e in first])} | "
                         f"{_pct([r['arms']['vlm0']['covers'] for r in F])} | {_pct([e['covers'] for e in first])} | "
                         f"{np.median(pw) if pw else float('nan'):.2f} | {np.median(vw) if vw else float('nan'):.2f} | "
                         f"{np.median([b[1]-b[0] for b in iv if b]) if any(iv) else float('nan'):.2f} | "
                         f"{_pct([b is not None and b[0] <= r['M_true'] <= b[1] for b, r in zip(iv, [r for r in F if r['events'] and r['events'][0]['kind']=='ask' and r['events'][0]['V_new']])])} |")
        lines += ["", "| task | arm | covers | width p50 | carry | false-safe | missed | rungs |", "|---|---|---|---|---|---|---|---|"]
        for task in sorted({r["task"] for r in rows}) + ["all"]:
            R = [r for r in rows if task == "all" or r["task"] == task]
            for arm in ("vlm0", "probe", "reask", "reask_and_probe_at_ask", "reask_and_probe_end"):
                A = [r["arms"]["reask_and_probe_end"] if (arm == "reask_and_probe_at_ask" and not r["n_asks"]) else r["arms"][arm] for r in R]
                w = [x["width"] for x in A if x["width"] is not None]
                rung = ("0" if arm == "vlm0" else
                        f"{np.median([r['rungs_at_last_ask'] or r['n_rungs'] for r in R]):.0f}" if arm == "reask_and_probe_at_ask"
                        else f"{np.median([r['n_rungs'] for r in R]):.0f}")
                lines.append(f"| {task} | {arm} | {sum(x['covers'] for x in A)}/{len(A)} | {np.median(w) if w else float('nan'):.2f} | "
                             f"{sum(x['carry'] is True for x in A)} | {sum(x['false_safe'] for x in A)} | {sum(x['missed'] for x in A)} | {rung} |")
        lines.append("")
    txt = "\n".join(lines)
    if a.md:
        pathlib.Path(a.md).write_text(txt)
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    for k in ("records", "answers", "traces", "ceilings", "endpoint", "model", "out"):
        r.add_argument(f"--{k}", required=True)
    r.add_argument("--max-reask", type=int, default=2)
    r.add_argument("--e-read", type=float, default=0.16)       # ~EPS_FREE + worst torque-noise bound (MEASURED + ASSUMED), told to the model
    r.add_argument("--thinking", default="on")
    r.add_argument("--max-tokens", type=int, default=8192)
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--workers", type=int, default=8)
    s = sp.add_parser("score")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--md")
    a = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
