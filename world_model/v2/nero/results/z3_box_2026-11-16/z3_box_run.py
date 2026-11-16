"""Z3 planner prototype run (2026-11-16, owner): box, closed_chain, Z3 (stack/z3plan) + the checker oracle (adapters/z3_oracle_contents).
Variant A: checker as-is.  Variant B: carry.PATH_MARGIN_MIN 0.05 -> 0.03 rad (ASSUMED override; the J7 hold finding), labelled conditional.
Each (variant, cell) runs in its own process; the returned plan is re-checked by a FRESH oracle (no cache).  SIMULATION ONLY."""
import json, os, sys, time, pathlib, multiprocessing as mp
OUT = pathlib.Path(__file__).parent
CELLS = [(1.0, 1.5), (2.5, 3.0), (4.5, 5.0)]
VARIANTS = {"A_asis": None, "B_margin0.03": 0.03}


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
    orc = ZO.make_oracle("box", work_dir=str(OUT / "work" / f"{var}_{cell[0]}-{cell[1]}"))
    t0 = time.time()
    plan, log = L.solve(ZO.domains("box"), orc, th, L.LearnedStore(), cost=[("a", "max")], max_iters=150)
    rec = dict(variant=var, margin_override=VARIANTS[var], cell=list(cell), plan=plan, iters=len(log), wall_s=round(time.time() - t0, 1),
               log=[{k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)) for k, v in e.items()} for e in log])
    if plan is not None:
        fresh = ZO.make_oracle("box", work_dir=str(OUT / "work" / f"{var}_{cell[0]}-{cell[1]}_recheck"))
        r = fresh(plan, th)
        rec["recheck"] = dict(ok=r.ok, rows=r.rows, tilt_missing=getattr(r, "tilt_missing", None))
    (OUT / f"{var}_{cell[0]}-{cell[1]}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(var, cell, "plan" if plan else "NONE", rec["iters"], rec["wall_s"], rec.get("recheck", {}).get("ok"), flush=True)
    return var, cell


if __name__ == "__main__":
    args = [(v, c) for v in VARIANTS for c in CELLS]
    with mp.get_context("spawn").Pool(len(args), maxtasksperchild=1) as p:
        for _ in p.imap_unordered(job, args):
            pass
