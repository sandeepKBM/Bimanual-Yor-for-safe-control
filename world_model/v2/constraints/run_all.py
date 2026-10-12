"""run_all.py -- the whole expressiveness study: 6 tasks (+ variants), margin equivalence, rows vs finite differences, cost, v0 probes.
python -m world_model.v2.constraints.run_all [--n 300] [--n-rows 100]   (CPU only; writes bimanual_data/constraints_v1/results.json)"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from world_model.v2.tray_rep.sets import ir
from world_model.v2.constraints import common as C, ir_ext
from world_model.v2.constraints.tasks import t1_tray, t2_cup, t3_stack, t4_bibox, t5_door, t6_pour

OUT = "/common/users/ss5772/bimanual_data/constraints_v1/results.json"


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def run_task(task, n, n_rows, seed=0, robust_rows=False, per_atom=False, norot=False):
    rng = np.random.default_rng(seed)
    S = task.sample(n, rng)
    out = dict(name=task.name, notes=task.notes)
    m = C.check_margin(task, S)
    out["margin"] = {k: v for k, v in m.items() if k not in ("err", "m_ir", "m_ref")}
    spec, _ = task.build()
    stat = [ir.status(spec, s) for s in S[:100]]
    out["status_counts"] = {k: stat.count(k) for k in ("true", "false", "unknown")}
    out["time"] = C.check_time(task, S, rng)
    rr = C.check_rows(task, S[:n_rows], rng, robust=robust_rows)
    out["rows"] = rr
    if per_atom:
        out["rows_per_atom"] = C.check_rows_per_atom(task, S[:n_rows], rng, robust=robust_rows)
    if norot and task.toy_norot is not None:
        out["rows_norot"] = C.check_rows(task, S[:n_rows], rng, robust=robust_rows, toy_fn=task.toy_norot)
        if per_atom:
            out["rows_per_atom_norot"] = C.check_rows_per_atom(task, S[:n_rows], rng, toy_fn=task.toy_norot, robust=robust_rows)
    return out, S


def probes(S2, S4, S5, S6, rng):
    """v0-only attempts and what they do."""
    res = {}
    # (2) cup volume with v0 shapes
    res["cup_v0_proxies"] = t2_cup.v0_proxies(S2)
    # (4) Not(Clear): margin works, rows -> NotConvex
    ents = t4_bibox.build()[1]
    nc = ir.Not(ir.Clear(ents["pad_L"], ents["box"], t4_bibox.D_C))
    ck = ir_ext.ContactKept(ents["pad_L"], ents["box"], t4_bibox.D_C)
    d = [abs(ir.margin(nc, s) - ir.margin(ck, s)) for s in S4[:100]]
    ref = [t4_bibox.D_C - t4_bibox.dists(s)[0] for s in S4[:100]]
    e_v0 = [abs(ir.margin(nc, s) - r) for s, r in zip(S4[:100], ref)]
    e_x = [abs(ir.margin(ck, s) - r) for s, r in zip(S4[:100], ref)]
    try:
        ir.rows(nc, S4[0], t4_bibox.toy(S4[0], rng))
        r = "rows ok"
    except ir.NotConvex:
        r = "NotConvex"
    res["contact_as_Not_Clear"] = dict(v0_rows=r, max_abs_err_v0_margin_vs_geom=float(max(e_v0)), max_abs_err_ContactKept_vs_geom=float(max(e_x)),
                                       median_err_v0=float(np.median(e_v0)), median_err_ContactKept=float(np.median(e_x)))
    # mj_geomDistance zero-return on separated pairs
    n0 = n = 0
    for s in S5[:150]:
        mj, g = t5_door.mj_dist(s), t5_door.geo_dist(s)
        for o in mj:
            n += 1
            n0 += (mj[o] == 0.0 and abs(g[o]) > 1e-6)
    res["mj_geomDistance_zero_but_not_touching"] = [n0, n]
    # (5) door: either side of an obstacle -> Or
    from world_model.v2.tray_rep.sets.ir import Region, Entity, BoxShape
    from world_model.v2.tray_rep.sets.zono import Polytope
    from world_model.v2.constraints.common import ent
    left = ent("left", Region(Polytope(np.array([[0, 1.0, 0]]), np.array([-0.1]))))
    right = ent("right", Region(Polytope(np.array([[0, -1.0, 0]]), np.array([-0.1]))))
    tray = t5_door.build(t5_door.SPEC_B)()[1]["tray"]
    tray_l = ent("tray", BoxShape(t5_door.TRAY_H), parent="left")
    spec_or = ir.Or(ir.Inside(ent("tray", BoxShape(t5_door.TRAY_H), parent="left"), left), ir.Inside(ent("tray", BoxShape(t5_door.TRAY_H), parent="right"), right))
    try:
        ir.rows(spec_or, S5[0], {"u0": np.zeros(6), "tray": {"p": np.zeros((3, 6))}})
        res["door_either_side_Or"] = "rows ok"
    except ir.NotConvex:
        res["door_either_side_Or"] = "NotConvex"
    # (6) Not(Upright) rows
    ents6 = t6_pour.build_with(t6_pour.SPEC)()[1]
    try:
        ir.rows(ir.Not(ir.Upright(ents6["cup"], t6_pour.TH_DONE)), S6[0], t6_pour.toy(S6[0], rng))
        res["pour_done_as_Not_Upright"] = "rows ok"
    except ir.NotConvex:
        res["pour_done_as_Not_Upright"] = "NotConvex"
    # Until: sweep over switch time
    spec6, _ = t6_pour.build_with(t6_pour.SPEC)()
    ms = []
    for s in S6[:50]:
        base = ir.margin(spec6, s)
        sw = spec6.switch_time(s)
        A, b, tags = ir.rows(spec6, s, t6_pour.toy(s, rng))
        ms.append((base, sw))
    res["pour_until_switch_time_median"] = float(np.median([x[1] for x in ms]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-rows", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    R = {}
    tasks = [("1_tray_ball", t1_tray.make(1), {}), ("1b_tray_ball_K6", t1_tray.make(6), {}),
             ("2_cup_beads", t2_cup.make(True), {}), ("2b_cup_beads_only", t2_cup.make(False), {}),
             ("3a_stack_centre", t3_stack.make(False), {}), ("3b_stack_substack", t3_stack.make(True), {}),
             ("4_bimanual_box", t4_bibox.make(), dict(per_atom=True, norot=True)),
             ("5a_door_clear_v0", t5_door.make("A"), dict(norot=True)), ("5b_door_clear_exact", t5_door.make("AX"), dict(norot=True)),
             ("5c_door_aperture", t5_door.make("B"), dict(norot=True)),
             ("6a_pour_until", t6_pour.make(t6_pour.SPEC), dict(robust_rows=True, per_atom=True)),
             ("6b_pour_over_only", t6_pour.make(t6_pour.SPEC_OVER_ONLY), {})]
    states = {}
    for key, task, kw in tasks:
        r, S = run_task(task, a.n, a.n_rows, **kw)
        R[key] = r
        states[key] = S
        print(f"[{time.time() - t0:6.1f}s] {key}: margin max err {r['margin']['max_abs_err']:.2e} sign {r['margin']['sign_agree']:.3f} | "
              f"rows {r['rows'].get('rows')} slack_max {r['rows'].get('slack_max_err', float('nan')):.2e} grad_med_rel {r['rows'].get('grad_med_rel_err', float('nan')):.2e} "
              f"sound_viol {r['rows'].get('sound_violations')}/{r['rows'].get('sound_n')} | ms {r['time']['ms_margin']:.2f}/{r['time']['ms_rows']:.2f}", flush=True)
    R["env_gaps"] = dict(cup=t2_cup.env_gap(states["2_cup_beads"]), stack_centre=t3_stack.env_gap(states["3a_stack_centre"]))
    R["probes"] = probes(states["2_cup_beads"], states["4_bimanual_box"], states["5a_door_clear_v0"], states["6a_pour_until"], np.random.default_rng(5))
    R["wall_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(jsonable(R), f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
