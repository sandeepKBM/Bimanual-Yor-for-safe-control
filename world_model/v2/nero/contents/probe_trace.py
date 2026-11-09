"""nero/contents/probe_trace.py -- probe-ladder traces on the VLM test worlds (11-03).  SIMULATION ONLY.

For each world of a vlm/contents_eval records file (same task + seed -> same world), run the soft-to-strong ladder from the FULL
prior with no stop rule (until free or the next rung is not admissible), torque noise on (seeded), and store every rung's reading
and the sound probe bound [P_lo, P_hi] after it.  vlm/contents_feedback.py replays these traces against a VLM's belief: the
ladder does not depend on the VLM (it always starts from the full prior), so one trace serves every model.
Also writes the static carry ceiling per task (probe.c_admit_max, MODEL): carry admitted iff contents hi <= ceiling.

    python -m world_model.v2.nero.contents.probe_trace --records REC --calib CALIB.json --procs 3 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from world_model.v2.stack import tasks as TK

from . import carry_cert as CC
from . import probe as PR



def _job(args):
    task, seed, eps_lo, eps_free, method = args
    t0 = time.time()
    try:
        w = CC.sample_world(task, TK.prior_box(task), np.random.default_rng(seed))
        ep = PR._episode(w)
        M = PR.true_mass(ep)
        pr = PR.make_probe(ep, task, eps_lo, eps_free, rng=np.random.default_rng(seed + 1), noise=True, method=method)
        why = pr.run(M_stop=None)
        return dict(task=task, seed=seed, probe_method=method, M_true=M, empty=PR.empty_mass(task), why=why, rungs=pr.rungs,
                    final=[pr.M_lo, pr.M_hi], viol=ep.track["viol"], track=pr.track, wall_s=round(time.time() - t0, 1))
    except Exception as e:
        import traceback
        return dict(task=task, seed=seed, error=repr(e), tb=traceback.format_exc()[-600:])


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--calib", required=True, help="CALIB.json of the SAME method as --method (eps must match the method)")
    ap.add_argument("--method", choices=PR.METHODS, default=None, help="probe method (default: env BIMANUAL_PROBE, else poslift)")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ceilings-only", action="store_true")
    a = ap.parse_args(argv)
    if a.method is not None:
        os.environ["BIMANUAL_PROBE"] = a.method
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = PR.load_calib(a.calib, a.method or PR.PROBE_METHOD)["eps"]
    recs = [json.loads(l) for l in open(a.records)]
    worlds = sorted({(r["task"], int(r["seed"])) for r in recs})
    tasks = sorted({t for t, _ in worlds})
    if a.ceilings_only:
        ceil = {t: PR.c_admit_max(t) for t in tasks}
        (out / "CEILINGS.json").write_text(json.dumps(dict(c_admit_max=ceil, empty={t: PR.empty_mass(t) for t in tasks},
                                                           label="MODEL: largest contents mass c with [c, c] admitted (static rows)"), indent=1))
        print(ceil, flush=True)
        return
    method = a.method or PR.PROBE_METHOD
    jobs = [(t, s, cal["EPS_LO"], cal["EPS_FREE"], method) for t, s in worlds]
    print(f"{len(jobs)} worlds", flush=True)
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(out / "traces.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(r["task"], r["seed"], r.get("why"), r.get("error", "")[:100], flush=True)


if __name__ == "__main__":
    main()
