"""Speed-envelope step 1 (11-16): is Rule 1 monotone in the plan acceleration a?  SIMULATION ONLY, offline.

For every box cell and a grid of a (<= the cell's a_admit), run adapters/rule1_contents.run_cell(..., a=a) on the closed-chain default plan
(fallback off, static rows only) and record, per step and recovery program, pass/fail and the failing row names.
A counterexample = a (cell, step, program) that passes at some a_hi but fails at a lower a_lo: then "check the top edge" does not cover that row.
Run:  python mono_rule1.py --out DIR [--procs N]   then   python mono_rule1.py --summarize DIR
"""
import argparse, json, os, pathlib, sys, tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

A_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]        # m/s^2; A_MIN 0.05 .. A_NOM 0.5 (checks/admit), every 0.05


def one(job):
    cell, a, out = job
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    from world_model.v2.adapters import rule1_contents as RC
    fn = pathlib.Path(out) / f"box_{cell[0]}-{cell[1]}_a{a:.2f}.json"
    if fn.exists():
        return str(fn)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        recs = RC.run_cell("box", cell, pathlib.Path(tmp), n_sampled=0, planner="closed_chain", fallback=False, a=a)
    rows = []
    for r in recs:
        rows.append(dict(seg=r["seg"], admitted=bool(r.get("admitted")), program=r.get("program"), refusal=r.get("refusal"), a_admit=r.get("plan_a_admit"),
                         per_program={n: dict(ok=p["ok"], failing=sorted({f["name"] for f in p["failing"]})) for n, p in (r.get("per_program") or {}).items()}))
    tmpfn = fn.with_suffix(".tmp")
    tmpfn.write_text(json.dumps(dict(cell=list(cell), a=a, steps=rows), indent=1))
    os.replace(tmpfn, fn)
    return str(fn)


def summarize(out):
    data = [json.loads(p.read_text()) for p in sorted(pathlib.Path(out).glob("box_*.json"))]
    table, cex = {}, []
    for d in data:
        for s in d["steps"]:
            if s.get("refusal") and "exceeds the static admission" in (s["refusal"] or ""):
                continue                                                  # a above a_admit: not a Rule 1 verdict
            for prog, p in s["per_program"].items():
                table.setdefault((tuple(d["cell"]), s["seg"], prog), {})[d["a"]] = (p["ok"], p["failing"])
    for k, byA in sorted(table.items()):
        As = sorted(byA)
        for i, alo in enumerate(As):
            for ahi in As[i + 1:]:
                if byA[ahi][0] and not byA[alo][0]:
                    cex.append(dict(cell=k[0], seg=k[1], program=k[2], a_fail=alo, a_pass=ahi, failing_at_low=byA[alo][1]))
    lines = ["| cell | step | program | " + " | ".join(f"a={a}" for a in A_GRID) + " |", "|" + "---|" * (3 + len(A_GRID))]
    for k, byA in sorted(table.items()):
        lines.append(f"| {k[0][0]}-{k[0][1]} | {k[1]} | {k[2]} | " + " | ".join(("pass" if byA[a][0] else "FAIL") if a in byA else "-" for a in A_GRID) + " |")
    res = dict(n_files=len(data), counterexamples=cex)
    (pathlib.Path(out) / "SUMMARY.json").write_text(json.dumps(res, indent=1, default=list))
    (pathlib.Path(out) / "SUMMARY.md").write_text("\n".join(lines) + f"\n\ncounterexamples (pass at a higher a, fail at a lower a): {len(cex)}\n"
                                                   + "".join(f"- {c}\n" for c in cex[:50]))
    print(f"{len(data)} files, {len(cex)} counterexamples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--one", nargs=3, default=None, metavar=("LO-HI", "A", "OUT"), help="internal: one (cell, a) job in this process")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize); sys.exit(0)
    if a.one:
        lo, hi = (float(x) for x in a.one[0].split("-"))
        print(one(((lo, hi), float(a.one[1]), a.one[2]))); sys.exit(0)
    from world_model.v2.nero.contents import carry_cert as CC
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, x, str(out)) for (c,) in CC.cells("box") for x in A_GRID]
    def run(job):                      # one subprocess per job (no multiprocessing semaphores in /dev/shm: the 11-16 sweep crash)
        c, x, o = job
        r = subprocess.run([sys.executable, __file__, "--one", f"{c[0]}-{c[1]}", str(x), o], capture_output=True, text=True, timeout=4 * 3600)
        return f"{c} a={x} rc={r.returncode} " + (r.stdout.strip().splitlines() or [""])[-1] + ("" if r.returncode == 0 else " ERR " + r.stderr.strip()[-400:])
    with ThreadPoolExecutor(a.procs) as ex:
        for line in ex.map(run, jobs):
            print(line, flush=True)
    summarize(out)
