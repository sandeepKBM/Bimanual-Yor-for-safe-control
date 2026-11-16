"""Z3 planner prototype run (2026-11-16, owner): box, closed_chain, Z3 (stack/z3plan) + the checker oracle (adapters/z3_oracle_contents).
v2 (11-16): compile-rule prefilter (oracle.cheap_reject), Learned.free knobs, max_iters=2000, explicit status (found / unsat / max_iters).
Variant A: checker as-is.  Variant C: carry.PATH_MARGIN_MIN = 0.015 rad (ASSUMED: the break-even found by the sweep; any result is CONDITIONAL on it).
Each (variant, cell) runs in its own process; the returned plan is re-checked by a FRESH oracle (no cache).  SIMULATION ONLY."""
import json, os, sys, time, pathlib, multiprocessing as mp
OUT = pathlib.Path(__file__).parent
CELLS = [(1.0, 1.5), (2.5, 3.0), (4.5, 5.0)]
VARIANTS = {"A_asis": None, "C_margin0.015": 0.015}
MAX_ITERS = 2000


def job(arg):
    var, cell = arg
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["BIMANUAL_PLANNER"] = "closed_chain"
    from world_model.v2.nero.contents import carry
    if VARIANTS[var] is not None:
        carry.PATH_MARGIN_MIN = VARIANTS[var]
    from world_model.v2.adapters import z3_oracle_contents as ZO
    from world_model.v2.stack.z3plan import loop as L
    from world_model.v2.stack.z3plan.types import Theta
    th = Theta(*cell)
    for sub in ("", "_recheck"):   # owner fix 11-16: the oracle mkdtemp()s inside work_dir at stage d (Rule 1) and does not create it
        (OUT / "work_v2" / f"{var}_{cell[0]}-{cell[1]}{sub}").mkdir(parents=True, exist_ok=True)
    orc = ZO.make_oracle("box", work_dir=str(OUT / "work_v2" / f"{var}_{cell[0]}-{cell[1]}"))
    t0 = time.time()
    plan, log = L.solve(ZO.domains("box"), orc, th, L.LearnedStore(), cost=[("a", "max")], max_iters=MAX_ITERS, prefilter=orc.cheap_reject)
    status = log[-1].get("status")
    rec = dict(variant=var, margin_override=VARIANTS[var], cell=list(cell), plan=plan, status=status, oracle_calls=orc.n_calls, iters=len(log), wall_s=round(time.time() - t0, 1),
               log=[{k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)) for k, v in e.items()} for e in log])
    if plan is not None:
        fresh = ZO.make_oracle("box", work_dir=str(OUT / "work_v2" / f"{var}_{cell[0]}-{cell[1]}_recheck"))
        r = fresh(plan, th)
        rec["recheck"] = dict(ok=r.ok, rows=r.rows, tilt_missing=getattr(r, "tilt_missing", None))
    (OUT / f"v2_{var}_{cell[0]}-{cell[1]}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(var, cell, "plan" if plan else "NONE", status, rec["iters"], rec["wall_s"], rec.get("recheck", {}).get("ok"), flush=True)
    return var, cell


if __name__ == "__main__":
    args = [(v, c) for v in VARIANTS for c in CELLS]
    with mp.get_context("spawn").Pool(len(args), maxtasksperchild=1) as p:
        for _ in p.imap_unordered(job, args):
            pass
