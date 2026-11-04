"""nero/contents/tilt_sweep.py -- what tilt / spill the carry plan USES, at the corners of each belief cell, vs acceleration.
SIMULATION ONLY.

USER 11-02 pm: tilt and spill are the target; they come from imperfect tracking under the hidden load.  Do not guess a tracking
bound (friction, gains): start from the slow (safe) end, measure what the plan costs as it speeds up, and build the bound from that.

  budget (what the plan may spend), per segment -- the unsafe-set thresholds of carry.py:
      object tilt <= U_TILT_OBJ (10 deg, ASSUMED tolerance), mug tilt <= U_TILT_CUP (20 deg, ASSUMED),
      spill ratio <= 1 (MODEL: liquid surface tilted by the mug's own tilt + its acceleration reaches the lip; slosh NOT modelled)
  use (what it spends): nominal runs (no backup branches) of the full plan at a ladder of peak accelerations, at CORNER worlds:
      box   contents mass {lo, hi} x contents centre at the 4 xy corners of the reachable floor area      -> 8 worlds per cell
      tray  every mug at fill {lo, hi}, mugs packed into one corner (4 corners)                         -> 8 worlds
            + "lean": packed into a corner with the mugs on that side at hi and the rest at lo (4)       -> 4 worlds
  The corners are the extremes of the load the arms feel (total mass and first moment pi = (m, m c)).  Whether the corners also
  bound the INTERIOR is not assumed: compare.py-style check against the sampled interior worlds of the certificate run.
Guarantee type of the result: SAMPLED (corners + interior samples), never a proof.

    python -m world_model.v2.nero.contents.tilt_sweep --tasks box,tray6 --acc 0.1,0.25,0.5,1.0 --procs 6 --out DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from .. import gripper as GR
from . import carry as CR
from . import carry_cert as CC
from . import scenes as SC

ACC_LADDER = (0.1, 0.25, 0.5, 1.0)          # m/s^2 peak plan acceleration; 0.5 = a_nom (ASSUMED plan speed)
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def box_corners(cell):
    w0 = SC.BoxWorld()
    hx = w0.box_outer[0] / 2 - w0.wall_t - w0.contents_half[0]
    hy = w0.box_outer[1] / 2 - w0.wall_t - w0.contents_half[1]
    return [(f"m{m}_x{sx:+d}y{sy:+d}", SC.BoxWorld(contents_kg=float(m), contents_xy=(sx * hx, sy * hy)))
            for m in cell for sx, sy in SIGNS]


def _grid(n):
    return {2: (1, 2), 4: (2, 2), 6: (2, 3)}.get(n) or (1, n)


def tray_layout(n, sx, sy):
    """n mugs on a cols x rows grid (D + 2 mm pitch), the whole block pushed into corner (sx, sy) (edge 2 mm from the inner rim)."""
    w0 = SC.TrayWorld(n_cups=n)
    D, p = w0.cup_D, w0.cup_D + 0.002
    cols, rows = _grid(n)
    gx = (np.arange(cols) - (cols - 1) / 2) * p
    gy = (np.arange(rows) - (rows - 1) / 2) * p
    hx = w0.tray_lx / 2 - w0.rim_t - D / 2 - 0.002
    hy = w0.tray_ly / 2 - w0.rim_t - D / 2 - 0.002
    ox, oy = sx * (hx - gx.max()), sy * (hy - gy.max())
    assert hx - gx.max() >= -1e-9 and hy - gy.max() >= -1e-9, "mugs do not fit"
    return [(float(ox + x), float(oy + y)) for y in gy for x in gx]


def tray_corners(n, cell):
    lo, hi = cell
    out = []
    for sx, sy in SIGNS:
        xy = tray_layout(n, sx, sy)
        for f in (lo, hi):
            out.append((f"f{f}_c{sx:+d}{sy:+d}", SC.TrayWorld(n_cups=n, cup_xy=xy, fills_kg=[f] * n)))
        # lean: mugs on the (sx, sy) side of the block's centre full, the others at lo
        c = np.mean(xy, 0)
        fills = [hi if (sx * (x - c[0]) + sy * (y - c[1])) > 1e-9 or n == 1 else lo for x, y in xy]
        if not any(f == hi for f in fills):
            fills[0] = hi
        out.append((f"lean_c{sx:+d}{sy:+d}", SC.TrayWorld(n_cups=n, cup_xy=xy, fills_kg=fills)))
    return out


def corners(task, cell):
    return box_corners(cell) if task == "box" else tray_corners(int(task[4:]), cell)


def _job(args):
    task, cell, name, world, a, m_hat = args[:6]
    ctl = args[6] if len(args) > 6 else "lever"          # 11-03 opt-in: "coopimp" = contents/coop_imp.py
    t0 = time.time()
    try:
        r = CR.world_trial(world, f"{task}_{cell[0]}_{name}_a{a}", a_plan=a, branch_backups=False, m_hat=m_hat,
                           fric_comp=True, weld_solref=CC.WELD_STIFF, ctl=ctl)   # == config.carry_physics() unless BIMANUAL_LEGACY_CARRY_PHYS=1 (kept explicit: the 11-02 sweep physics)
        segs = {s["seg"]: {k: s["nominal"].get(k) for k in ("obj_tilt_max", "cup_tilt_max", "spill_ratio", "tau_ratio", "e_obj_deg",
                                                             "e_obj_mm", "viol", "end_in_R", "F_int_max", "twist_max") + FLK
                         + (("ci_F_int", "ci_T_int", "ci_e_p_mm", "ci_e_R_deg", "ci_tau_cmd") if ctl != "lever" else ())}
                for s in r["segments"]}
        for s_ in r["segments"]:                              # 11-03 item 10: the acceleration actually commanded (T_MIN clamps the move time)
            segs[s_["seg"]]["a_cmd_peak"] = (s_.get("accel") or {}).get("a_cmd_peak")
            segs[s_["seg"]]["t_min_clamped"] = (s_.get("accel") or {}).get("clamped")
        out = dict(task=task, cell=list(cell), corner=name, a=a, theta=r["theta"], segments=segs, physics=r.get("physics"),
                   end_in_R=r["end_in_R"][0], pre_hold_viol=r["pre_hold_viol"])
    except Exception as e:                                   # reported, never dropped
        out = dict(task=task, cell=list(cell), corner=name, a=a, error=repr(e))
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def jobs_for(tasks, acc, ctl="lever"):
    jobs = []
    for task in tasks:
        for (cell,) in CC.cells(task):
            sp = CC.cell_spec(task, cell)
            m_hat = 0.5 * (sp.contents.m_lo + sp.contents.m_hi)     # the robot's nominal contents = cell centre (as carry_cert)
            for name, w in corners(task, cell):
                for a in acc:
                    jobs.append((task, cell, name, w, float(a), m_hat) + ((ctl,) if ctl != "lever" else ()))
    return jobs


FLK = ("fl_steps", "fl_n_infeasible", "fl_n_relaxed", "fl_n_zero", "fl_n_scale_called", "fl_n_scale_shrunk", "fl_n_taumax_relaxed",
       "fl_res_armarm_min_m", "fl_res_static_min_m", "fl_res_n_armarm_pen", "fl_res_n_static_pen", "fl_res_tau_ratio_max", "fl_res_tau_joint",
       "fl_res_n_tau_over")     # 11-03 F2: FL-MPC fallback counters + true residuals per segment (nero/fl_log.py)


def summarise(recs):
    """per (task, cell, a): max over corners of each used quantity, as a fraction of its budget."""
    from collections import defaultdict
    agg = defaultdict(lambda: dict(n=0, err=0, U=0, tilt=0.0, cup=0.0, spill=0.0, tau=0.0, e_deg=0.0, worst=None))
    for r in recs:
        k = (r["task"], tuple(r["cell"]), r["a"])
        g = agg[k]
        if "error" in r:
            g["err"] += 1
            continue
        g["n"] += 1
        segs = r["segments"].values()
        g["U"] += int(any(s["viol"] for s in segs) or bool(r.get("pre_hold_viol")))
        t = max(s["obj_tilt_max"] for s in segs)
        if t > g["tilt"]:
            g["tilt"], g["worst"] = t, r["corner"]
        g["cup"] = max(g["cup"], max(s["cup_tilt_max"] for s in segs))
        g["spill"] = max(g["spill"], max(s["spill_ratio"] for s in segs))
        g["tau"] = max(g["tau"], max(s["tau_ratio"] for s in segs))
        g["e_deg"] = max(g["e_deg"], max(s["e_obj_deg"] for s in segs))
        from world_model.v2.nero import fl_log as FLL
        g.setdefault("fl_all", []).extend(segs)
        g["fl"] = FLL.aggregate(g["fl_all"])               # 11-03 F2: over all corners of this (cell, a)
    return agg


def summary_text(recs):
    agg = summarise(recs)
    L = [f"tilt sweep (corners, nominal runs, fc + stiff weld) -- max over corners; budget fractions: tilt/{CR.U_TILT_OBJ} deg, "
         f"mug/{CR.U_TILT_CUP} deg, spill/1, tau/16 Nm", ""]
    for (task, cell, a), g in sorted(agg.items()):
        L.append(f"{task:6s} {str(cell):12s} a={a:<5} n={g['n']:2d} U={g['U']:2d} err={g['err']}  tilt {g['tilt']:5.2f} deg "
                 f"({g['tilt'] / CR.U_TILT_OBJ:4.2f})  mug {g['cup']:5.2f} ({g['cup'] / CR.U_TILT_CUP:4.2f})  spill {g['spill']:4.2f}  "
                 f"tau {g['tau']:4.2f}  e_obj {g['e_deg']:4.2f} deg  a_cmd_peak {g.get('a_cmd', 0.0):5.3f}"
                 + ("  [T_MIN clamps a segment: commanded < a]" if g.get("clamped") else "")
                 + f"  worst={g['worst']}"
                 + (f"  FL: infeas {g['fl']['n_infeasible']}/{g['fl']['steps']} (relaxed {g['fl']['n_relaxed']}, zero {g['fl']['n_zero']}, "
                    f"scale-shrunk {g['fl']['n_scale_shrunk']}), true min arm-arm {g['fl']['armarm_min_m']:.3f} m, static {g['fl']['static_min_m']:.3f} m, "
                    f"tau/16 {g['fl']['tau_ratio_max']:.2f}" if g.get("fl", {}).get("n_with") else ""))
    return "\n".join(L)


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="box,tray2,tray4,tray6")
    ap.add_argument("--acc", default=",".join(map(str, ACC_LADDER)))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default="/data/local/ss5772/contents/tilt_sweep_2026-11-02")
    ap.add_argument("--ctl", choices=("lever", "coopimp"), default="lever", help="carry controller (coopimp = object-level impedance, opt-in)")
    ap.add_argument("--dry", action="store_true", help="list the jobs only")
    ap.add_argument("--cells", default="", help="keep only cells whose lower bound is in this list, e.g. 3.5,4.0,4.5 (default: all)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(a.tasks.split(","), [float(x) for x in a.acc.split(",")], a.ctl)
    if a.cells:
        keep = {float(x) for x in a.cells.split(",")}
        jobs = [j for j in jobs if float(j[1][0]) in keep]
    print(f"{len(jobs)} runs", flush=True)
    if a.dry:
        for j in jobs[:30]:
            print(j[0], j[1], j[2], j[4], CR._theta(j[3]))
        return
    meta = dict(date=time.strftime("%Y-%m-%d"), physics="fric_comp (carry.FRIC_COMP, ASSUMED gains) + weld solref " + str(CC.WELD_STIFF),
                grasp="ASSUMED rigid weld", **GR.provenance("rigid weld"), guarantee="SAMPLED: corner worlds of each cell, nominal runs, no backups",
                budget=dict(obj_tilt_deg=f"{CR.U_TILT_OBJ} ASSUMED", cup_tilt_deg=f"{CR.U_TILT_CUP} ASSUMED",
                            spill="ratio 1, MODEL quasi-static surface (mug tilt + acceleration); slosh NOT modelled"),
                m_hat="cell centre", plan="carry.PLANS (ASSUMED, user-confirmed)")
    (out / "META.json").write_text(json.dumps(meta, indent=1))
    recs = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(out / "runs.jsonl", "a") as f:
        for r in pool.imap_unordered(_job, jobs):
            recs.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    (out / "SUMMARY.txt").write_text(summary_text(recs))
    print(summary_text(recs))
    print(f"wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
