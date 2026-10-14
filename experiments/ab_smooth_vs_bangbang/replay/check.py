"""check.py -- replay every record through the CURRENT code and compare bitwise.

  python check.py --records /common/users/ss5772/bimanual_data/flmpc_replay [--atol 1e-9] [--coverage]
                  [--limit-per-class N] [--cls FLMPC,ValueFLMPC,RTFLMPC] [--json out.json]

Exit code 0 = every record is bit-identical (np.array_equal, NaN-aware) on: step() return value, the
sequence of solve_constrained_box_qp calls (inputs and outputs), and all controller side effects.
Exit code 1 = at least one record differs. The `tolerance report` column says how many failing records
would still pass an |diff| <= atol check (only informational; a bitwise FAIL is a FAIL).

SIMULATION ONLY.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from pathlib import Path

import replay_core as rc


def find_records(root):
    root = Path(root)
    return sorted(root.rglob("rec_*.pkl.gz"))


def run_check(paths, atol=1e-9, limit_per_class=None, classes=None, verbose=True, step_override_for=None):
    """Replay `paths`; return (summary dict, per-record results list)."""
    import sim_common as sc
    stores = {}
    results = []
    _t0 = time.perf_counter()
    per_cls = defaultdict(int)
    for p in paths:
        rec = rc.load_record(p)
        cls = rec["cls"]
        if classes and cls not in classes:
            continue
        if limit_per_class and per_cls[cls] >= limit_per_class:
            continue
        per_cls[cls] += 1
        store = stores.setdefault(p.parent, rc.ModelStore(p.parent / "models", search=[p.parent.parent / "models"]))
        try:
            out = rc.replay_one(rec, store, sc)
            ok, tol_ok, c = rc.check_one(rec, out, atol)
            r = dict(path=str(p), cls=cls, source=rec["source"], idx=rec["idx"], exact=ok, tol=tol_ok,
                     n_exact=c.n_exact, n_cmp=c.n, n_bytes=c.n_bytes, max_abs=c.max_abs,
                     fails=[(a, (b if isinstance(b, str) else float(b))) for a, b in c.fails][:8],
                     fallback=rec["tags"]["fallback"])
        except Exception as e:                       # a crash in replay is a regression too
            r = dict(path=str(p), cls=cls, source=rec["source"], idx=rec["idx"], exact=False, tol=False,
                     n_exact=0, n_cmp=0, n_bytes=0, max_abs=float("inf"),
                     fails=[("EXCEPTION", f"{type(e).__name__}: {e}")],
                     fallback=rec["tags"]["fallback"], tb=traceback.format_exc(limit=3))
        results.append(r)
        if verbose and len(results) % 50 == 0:
            print(f"[check] ... {len(results)} records replayed, {sum(not x['exact'] for x in results)} differing "
                  f"({time.perf_counter() - _t0:.0f}s)", flush=True)
    summ = OrderedDict()
    for cls in sorted({r["cls"] for r in results}):
        rs = [r for r in results if r["cls"] == cls]
        summ[cls] = dict(records=len(rs), exact=sum(r["exact"] for r in rs),
                         tol_only=sum((not r["exact"]) and r["tol"] for r in rs),
                         byte_identical_fields=f"{sum(r['n_bytes'] for r in rs)}/{sum(r['n_cmp'] for r in rs)}",
                         max_abs_diff=max((r["max_abs"] for r in rs), default=0.0))
    return summ, results


def tag_coverage(paths):
    """Coverage of option / row families / branches across the recorded set (from the tags, no replay)."""
    cnt = defaultdict(int)
    by_src = defaultdict(int)
    rows = set()
    n = 0
    for p in paths:
        rec = rc.load_record(p)
        t = rec["tags"]
        n += 1
        by_src[rec["source"].split(":")[0] + ":" + rec["cls"]] += 1
        for k in ("table_rows", "tilt_box", "cone", "leash", "speed_cap", "payload_M", "payload_bias", "delay",
                  "tau_max_vec", "adaptive_yield", "value_rows", "taumax_relaxed_inc"):
            cnt[k] += int(bool(t[k]))
        if t["cone"]:
            cnt[f"cone_mode={t['cone_mode']}"] += 1
        cnt["ego_hold=True"] += int(t["ego_hold"])
        cnt["ego_hold=False"] += int(not t["ego_hold"])
        cnt[f"fallback={t['fallback']}"] += 1
        cnt["hard_infeasible"] += int(t["hard_feasible"] is False)
        cnt[f"solver_calls={t['n_solver_calls']}"] += 1
        rows.add(t["hard_rows"])
    return n, dict(by_src), dict(sorted(cnt.items())), sorted(r for r in rows if r is not None)


def _ranges(lines):
    lines = sorted(lines)
    out, s = [], None
    for x in lines:
        if s is None:
            s = e = x
        elif x == e + 1:
            e = x
        else:
            out.append((s, e)); s = e = x
    if s is not None:
        out.append((s, e))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=rc.DEFAULT_DATA_DIR)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--coverage", action="store_true", help="line coverage of mpc_fl.FLMPC.step etc. during replay")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--cls", default=None, help="comma list: FLMPC,ValueFLMPC,RTFLMPC")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    paths = find_records(args.records)
    if not paths:
        print(f"[check] no records under {args.records}")
        return 2
    classes = set(args.cls.split(",")) if args.cls else None
    t0 = time.perf_counter()
    cov = None
    if args.coverage:
        import coverage
        import mpc_fl
        import flmpc_value
        cov = coverage.Coverage(include=[mpc_fl.__file__, flmpc_value.__file__], data_file=None)
        cov.start()
    summ, results = run_check(paths, args.atol, args.limit_per_class, classes)
    if cov is not None:
        cov.stop()
    n_all = len(results)
    n_ok = sum(r["exact"] for r in results)
    print(f"[check] {n_all} records from {args.records} replayed in {time.perf_counter() - t0:.0f}s")
    for cls, s in summ.items():
        print(f"  {cls:11s} records={s['records']:4d} bit-identical={s['exact']:4d} "
              f"tolerance-only(<= {args.atol:g})={s['tol_only']:3d} max|diff|={s['max_abs_diff']:.3g} "
              f"byte-identical-fields={s['byte_identical_fields']}")
    bad = [r for r in results if not r["exact"]]
    for r in bad[:10]:
        print(f"  FAIL {Path(r['path']).parent.name}/{Path(r['path']).name} {r['cls']} idx={r['idx']} "
              f"max|diff|={r['max_abs']:.3g} first={r['fails'][:2]}")
        if "tb" in r:
            print("    " + r["tb"].strip().replace("\n", "\n    "))
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more failing records")
    if cov is not None:
        import mpc_fl
        import inspect
        _, executable, _, missing, _ = cov.analysis2(mpc_fl.__file__)
        sl, el = inspect.getsourcelines(mpc_fl.FLMPC.step)
        lo, hi = el, el + len(sl) - 1
        ex = [l for l in executable if lo <= l <= hi]
        ms = [l for l in missing if lo <= l <= hi]
        print(f"[coverage] mpc_fl.FLMPC.step lines {lo}-{hi}: {len(ex) - len(ms)}/{len(ex)} executable lines hit "
              f"({100.0 * (len(ex) - len(ms)) / max(1, len(ex)):.1f}%)")
        print(f"[coverage] step lines NOT hit by any record: {_ranges(ms) or 'none'}")
        for fn in ("_relaxed_fallback", "_torque_safe_scale", "_predict_over_delay"):
            f = getattr(mpc_fl.FLMPC, fn)
            f = getattr(f, "__func__", f)
            sl, el = inspect.getsourcelines(f)
            ex = [l for l in executable if el <= l <= el + len(sl) - 1]
            ms = [l for l in missing if el <= l <= el + len(sl) - 1]
            print(f"[coverage] mpc_fl.FLMPC.{fn}: {len(ex) - len(ms)}/{len(ex)} hit; not hit: {_ranges(ms) or 'none'}")
        import flmpc_value
        _, executable, _, missing, _ = cov.analysis2(flmpc_value.__file__)
        sl, el = inspect.getsourcelines(flmpc_value._SolverProxy.solve_constrained_box_qp)
        ex = [l for l in executable if el <= l <= el + len(sl) - 1]
        ms = [l for l in missing if el <= l <= el + len(sl) - 1]
        print(f"[coverage] flmpc_value._SolverProxy.solve_constrained_box_qp: {len(ex) - len(ms)}/{len(ex)} hit; "
              f"not hit: {_ranges(ms) or 'none'}")
    if args.coverage:
        n_, by_src, cnt, rows = tag_coverage(paths)
        print(f"[coverage] records by source:class: {by_src}")
        print(f"[coverage] row/branch families present in recorded calls: {cnt}")
        print(f"[coverage] distinct hard-QP row counts: {rows}")
    verdict = "PASS" if n_ok == n_all else "FAIL"
    print(f"[check] RESULT: {verdict}  ({n_ok}/{n_all} records bit-identical)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(verdict=verdict, summary=summ, failing=bad[:50]), f, indent=1, default=str)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
