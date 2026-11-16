"""checks/tilt.py -- object-tilt row (checker design 11-05 §3, step K4).  PURE (no physics).  Guarantee: sampled, on a MEASURED residual table.

The table comes from `nero/contents/tilt_measure.py` (TABLE.json): per cell key ("task|lo-hi") and acceleration level a, the episodes of the nominal
carry in worlds drawn from the cell: n completed, n_errors, n_stopped (run cut short by a U violation: its tilt covers only the segments that ran),
n_over (peak tilt >= the U limit), n_fail (union of the three), p50 / p95 / max over the completed episodes (deg).  The row uses the nearest measured
level >= a_plan (conservative: the next HIGHER level).  ok iff n_stopped == 0 and n_errors == 0 and max < u_limit.  eps_upper = one-sided
Clopper-Pearson bound on the failure rate with n_fail failures in n_total.  An older table without n_stopped / n_errors gives `missing`.
"max over n" is a measurement, not a bound over the whole box; sim only (see the table's `noise_model` / `leaves_out`).
The recovery / dynamic-tilt rows stay NOT BUILT."""
from __future__ import annotations

from world_model.v2.stack.checks.dynamic import cp_upper
from world_model.v2.stack.checks.row import Row, missing


def _levels(cell):
    return sorted((float(a), v) for a, v in (cell.get("levels") or {}).items())


def tilt_row(residual_table=None, cell_key=None, a_plan=None, u_limit_deg=None, name="obj_tilt", conf=0.95) -> Row:
    if residual_table is None:
        return missing(name, "tilt / recovery rows not built: no measured residual table given (checker audit 11-06)")
    cell = (residual_table.get("cells") or {}).get(cell_key)
    if cell is None:
        return missing(name, f"cell {cell_key!r} is not in the measured tilt table")
    if a_plan is None or u_limit_deg is None:
        return missing(name, "tilt_row needs a_plan and u_limit_deg")
    up = [(a, v) for a, v in _levels(cell) if a >= float(a_plan) - 1e-9]
    if not up:
        top = max((a for a, _ in _levels(cell)), default=None)
        return missing(name, f"a_plan {a_plan} is above every measured level (max {top}) for {cell_key!r}")
    a_m, v = up[0]
    if "n_stopped" not in v or "n_errors" not in v:
        return missing(name, "tilt table lacks n_stopped / n_errors (older table): re-measure")
    n_ok, n_err, n_stop = int(v["n"]), int(v["n_errors"]), int(v["n_stopped"])
    n_tot = int(v.get("n_total", n_ok + n_err))
    mx = None if v.get("max") is None else float(v["max"])
    n_over = int(v["n_over"]) if "n_over" in v else (1 if mx is not None and mx >= float(u_limit_deg) else 0)   # exact when the table has it
    n_fail = int(v["n_fail"]) if "n_fail" in v else n_err + n_stop + n_over
    why = []
    if n_err:
        why.append(f"{n_err} episode(s) errored")
    if n_stop:
        why.append(f"{n_stop} episode(s) stopped early on a U violation (tilt covers only the segments that ran)")
    if mx is None:
        why.append("no completed episode")
    elif mx >= float(u_limit_deg):
        why.append(f"measured peak tilt {mx:.2f} deg >= U limit {u_limit_deg} deg ({n_over} episode(s) over)")
    ok = not why
    n_fail = max(n_fail, 1) if not ok else n_fail
    return Row(name, ok, margin=None if mx is None else float(u_limit_deg) - mx, unit="deg", guarantee="sampled" if ok else "refuted",
               worst=dict(a_measured=a_m, p50=v.get("p50"), p95=v.get("p95"), max=mx, n_errors=n_err, n_stopped=n_stop, n_over=n_over),
               n=n_tot, eps_upper=round(cp_upper(n_fail, n_tot, conf), 4) if n_tot else 1.0,
               assumptions=dict(conf=conf, bound="Clopper-Pearson on the failure rate (errored + stopped + over-U episodes), i.i.d. worlds from the cell",
                                tilt="MEASURED " + str(residual_table.get("file", "?")) + f" (n_total={n_tot}, max over completed n={n_ok}, level a={a_m} >= a_plan={a_plan})",
                                planner=residual_table.get("planner", cell.get("planner", "?")),
                                noise_model=residual_table.get("noise_model", "?"), leaves_out=residual_table.get("leaves_out", "?")),
               why="; ".join(why))
