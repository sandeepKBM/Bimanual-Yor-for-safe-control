"""E1 -- equivalence: TruthProducer -> OCS -> consumers reproduce the shield's inputs exactly, and a teleop carry through the OCS proxy matches the
direct run (and data/molmospaces/teleop/carry_9seeds_summary.json) on every non-timing field.  SIMULATION ONLY.

    python -m world_model.v2.house.ocs.exp_e1 inputs                 # per-tick input equivalence on a live sim + the stereo-zonotope round trip   (~1 min)
    python -m world_model.v2.house.ocs.exp_e1 regress --seeds 0 1    # raw / shield x {direct run_episode, OCS-proxy run_episode}                   (~5 min, 4 procs)
    python -m world_model.v2.house.ocs.exp_e1 report                 # json + md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_TELEOP_FAST", "1")

import numpy as np

from world_model.v2.house.teleop.shield import Passthrough as _PT          # the ORIGINAL class (run_variant_fixed patches teleop.shield.Passthrough)

OUT = Path(os.environ.get("OCS_OUT", "data/molmospaces/ocs"))
SUMMARY = Path("data/molmospaces/teleop/carry_9seeds_summary.json")
TIMING_KEYS = {"wall_s", "ctl_ms_per_physics_step_median"}


# ====================================================================================== OCS shield wrapper
class OcsShield:
    """Shield that consumes the OCS: each filter() call builds a WorldState from ground truth (TruthProducer) and hands the Shield a ShieldView of the sim
    whose tray / cup / people / base reads come from that WorldState.  Everything else is delegated to the wrapped Shield."""

    def __init__(self, shield, chain=False):
        from world_model.v2.house.ocs.producers import TruthProducer, Pipeline
        object.__setattr__(self, "_sh", shield)
        object.__setattr__(self, "_pipe", Pipeline([TruthProducer(chain=chain, with_free=False)]))

    def filter(self, sim, cmd):
        from world_model.v2.house.ocs.views import LiveView
        from world_model.v2.house.ocs.consumers import ShieldView
        ws = self._pipe.step(LiveView(sim))
        return self._sh.filter(ShieldView(sim, ws), cmd)

    def __getattr__(self, n):
        return getattr(self._sh, n)

    def __setattr__(self, n, v):
        setattr(self._sh, n, v)


class OcsPassthrough:
    """shield OFF through the OCS path (the OCS is built and ignored, as the raw run ignores everything)."""
    n_fallback = 0
    n_veto = 0

    def __init__(self):
        from world_model.v2.house.ocs.producers import TruthProducer, Pipeline
        self._p = _PT()
        self._pipe = Pipeline([TruthProducer(with_free=False)])

    def filter(self, sim, cmd):
        from world_model.v2.house.ocs.views import LiveView
        self._pipe.step(LiveView(sim))
        return self._p.filter(sim, cmd)


def _cmp(a, b, path="", rtol=1e-9, atol=1e-12, out=None, skip=TIMING_KEYS):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip or k == "_wall":
                continue
            if k not in a or k not in b:
                out.append((path + "/" + k, "missing", a.get(k), b.get(k)))
            else:
                _cmp(a[k], b[k], path + "/" + k, rtol, atol, out, skip)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, "len", len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _cmp(x, y, f"{path}[{i}]", rtol, atol, out, skip)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or abs(a - b) <= atol + rtol * max(abs(a), abs(b)) or (math.isnan(a) and math.isnan(b))):
            out.append((path, "value", a, b))
    elif a != b:
        out.append((path, "value", a, b))
    return out


def worst(a, b, path=""):
    """max relative difference over all numeric leaves (excluding timing)."""
    d = _cmp(a, b, rtol=0.0, atol=0.0)
    w = 0.0
    for p, kind, x, y in d:
        if kind == "value" and isinstance(x, (int, float)) and isinstance(y, (int, float)):
            w = max(w, abs(x - y) / max(abs(x), abs(y), 1e-300))
    return w, len(d)


# ====================================================================================== part A: shield inputs
def inputs_equivalence(n_ticks=150, seed=0):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP, shield as SH
    from world_model.v2.house.ocs.producers import TruthProducer, Pipeline, ObstacleZonoProducer
    from world_model.v2.house.ocs.views import LiveView, Recording
    from world_model.v2.house.ocs import consumers as C
    from world_model.v2.tray_rep.sets import tray_spec as TS
    sim = TeleopSim("h5", grasped=True, seed=seed, with_people=True)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None, back_route=sc.back_route,
                           back_mode=sc.back_mode)
    sh = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw)
    sh.docks = [wp["cafe_counter_pick"][:2]]
    pe = Pipeline([TruthProducer(chain=False)])
    pc = Pipeline([TruthProducer(chain=True)])
    keys = ["env_tray_p", "env_tray_R", "env_cup_p", "people_xy", "people_v", "people_names", "view_tray_xmat", "view_pose", "view_base_v", "occ_edt",
            "chain_tray_p", "chain_cup_p", "chain_tray_R"]
    mx = {k: 0.0 for k in keys}
    names_ok = True
    for k in range(n_ticks):
        obs = sim.observe()
        cmd = op.step(obs)
        view = LiveView(sim)
        ws = pe.step(view)
        d = sim.d
        es = C.to_env_state(ws)
        ref = TS.EnvState(d.xpos[sim.tray], d.xquat[sim.tray], d.xpos[sim.cup])
        mx["env_tray_p"] = max(mx["env_tray_p"], float(np.abs(es.p_t - ref.p_t).max()))
        mx["env_tray_R"] = max(mx["env_tray_R"], float(np.abs(es.R_t - ref.R_t).max()))
        mx["env_cup_p"] = max(mx["env_cup_p"], float(np.abs(es.p_b - ref.p_b).max()))
        ps = C.to_people_state(ws)
        assert len(ps) == len(sim.people_state)
        for (xy, v), (xy0, v0) in zip(ps, sim.people_state):
            mx["people_xy"] = max(mx["people_xy"], float(np.abs(xy - xy0).max()))
            mx["people_v"] = max(mx["people_v"], float(np.abs(v - v0).max()))
        names_ok &= [i["name"] for i in C.person_info(ws)] == [p.name for p in sc.people]
        sv = C.ShieldView(sim, ws)
        mx["view_tray_xmat"] = max(mx["view_tray_xmat"], float(np.abs(sv.d.xmat[sim.tray] - d.xmat[sim.tray]).max()))
        mx["view_pose"] = max(mx["view_pose"], float(np.abs(sv.pose() - sim.pose()).max()))
        mx["view_base_v"] = max(mx["view_base_v"], float(np.abs(sv.base.v - sim.base.v).max()))
        if k == 0:
            g = C.to_occ(ws)
            mx["occ_edt"] = float(np.abs(g.edt - sim.occ.edt).max()) if ws.free is not None else float("nan")
        wc = pc.step(view)
        mx["chain_tray_p"] = max(mx["chain_tray_p"], float(np.abs(wc.world_pose("tray").p - d.xpos[sim.tray]).max()))
        mx["chain_cup_p"] = max(mx["chain_cup_p"], float(np.abs(wc.world_pose("cup").p - d.xpos[sim.cup]).max()))
        mx["chain_tray_R"] = max(mx["chain_tray_R"], float(np.abs(wc.world_pose("tray").R - d.xmat[sim.tray].reshape(3, 3)).max()))
        safe, _ = sh.filter(sim, cmd)
        sim.tick(safe)
    # the shield decisions on the proxy equal the direct ones (one-shot check on the last state): same filter call on identical shield state
    out = dict(n_ticks=n_ticks, max_abs_diff=mx, person_names_match=bool(names_ok))
    # stereo zonotopes: perception result -> OCS -> to_obstacle_zonos == the pipeline's inflated Zi
    rp = OUT / "rec" / "flagship_desk.pkl.gz"
    if rp.exists():
        rec = Recording.load(rp)
        res = rec.zono_obs_result
        pz = Pipeline([TruthProducer(chain=False, with_free=False), ObstacleZonoProducer()])
        ws = pz.step(rec.view(0))
        zz = dict(C.to_obstacle_zonos(ws, inflate=True))
        dc = dg = 0.0
        n = 0
        for z in res["zonos"]:
            oid = f"z{z['id']}"
            if oid in zz:
                n += 1
                dc = max(dc, float(np.abs(zz[oid].c - np.asarray(z["Zi"]["c"])).max()))
                a, b = zz[oid].G, np.asarray(z["Zi"]["G"])
                dg = max(dg, float(np.abs(a - b).max()) if a.shape == b.shape else float("inf"))
        out["stereo_zonos"] = dict(n=n, n_in_result=len(res["zonos"]), max_abs_diff_c=dc, max_abs_diff_G=dg)
        # truth obstacles: TruthProducer set == recorded truth set
        wt = Pipeline([TruthProducer(chain=False, with_free=False)]).step(rec.view(0))
        zt = dict(C.to_obstacle_zonos(wt, inflate=False))
        out["truth_obstacles"] = dict(n=len(zt), max_abs_diff=max((float(np.abs(zt[k].c - z.c).max()) + float(np.abs(zt[k].G - z.G).max())) for k, z in rec.obstacles))
    return out


# ====================================================================================== driver
def run_regress(seeds, procs=4):
    import multiprocessing as mp
    jobs = [(s, sh, v) for s in seeds for sh in (False, True) for v in ("direct", "ocs")]
    with mp.get_context("spawn").Pool(min(procs, len(jobs)), maxtasksperchild=1) as pool:
        res = pool.map(_run_job, jobs)
    return res


def _run_job(a):
    return run_variant_fixed(a)


def run_variant_fixed(args):
    seed, shielded, variant = args
    from world_model.v2.house.teleop import shield as SH, run as RUN
    if variant == "ocs":
        orig_S, orig_P = SH.Shield, SH.Passthrough
        SH.Shield = lambda *a, **k: OcsShield(orig_S(*a, **k))
        SH.Passthrough = lambda: OcsPassthrough()
    t0 = time.time()
    summ, rows = RUN.run_episode(shielded, seed=seed, task="carry", scene="h5", max_s=30.0, stream="closed", verbose=False)
    summ["_wall"] = time.time() - t0
    return dict(seed=seed, shielded=shielded, variant=variant, summary=json.loads(json.dumps(summ, default=str)))


def report(seeds, res, inputs):
    S = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    by = {(r["seed"], r["shielded"], r["variant"]): r["summary"] for r in res}
    rows = []
    for s in seeds:
        for sh in (False, True):
            key = f"{'shield' if sh else 'raw'}_s{s}"
            d, o = by.get((s, sh, "direct")), by.get((s, sh, "ocs"))
            ref = S.get(key)
            row = dict(run=key)
            if d and o:
                diffs = _cmp(d, o)
                w, n = worst(d, o)
                row.update(ocs_vs_direct_n_fields_differ=len(diffs), ocs_vs_direct_worst_rel=w, ocs_vs_direct_bit_identical=(n == 0),
                           ocs_vs_direct_diffs=[(p, a, b) for p, k, a, b in diffs[:10]])
            if o and ref:
                diffs = _cmp(ref, o)
                w, n = worst(ref, o)
                row.update(ocs_vs_summary_n_fields_differ=len(diffs), ocs_vs_summary_worst_rel=w, ocs_vs_summary_diffs=[(p, a, b) for p, k, a, b in diffs[:10]])
            if d and ref:
                diffs = _cmp(ref, d)
                w, n = worst(ref, d)
                row.update(direct_vs_summary_n_fields_differ=len(diffs), direct_vs_summary_worst_rel=w, direct_vs_summary_diffs=[(p, a, b) for p, k, a, b in diffs[:10]])
            rows.append(row)
    js = dict(inputs=inputs, regress=rows, n_seeds=len(seeds))
    (OUT / "e1_equivalence.json").write_text(json.dumps(js, indent=1, default=str))
    L = ["# E1 equivalence", "", "## Shield inputs (live sim, per tick, max abs difference vs the raw MuJoCo numbers)", "", "| input | max abs diff |", "|---|---|"]
    for k, v in (inputs or {}).get("max_abs_diff", {}).items():
        L.append(f"| {k} | {v:.3g} |")
    L.append(f"| person names match scene.people | {inputs.get('person_names_match')} |")
    if "stereo_zonos" in inputs:
        z = inputs["stereo_zonos"]
        L += [f"| stereo zonotopes (n={z['n']}/{z['n_in_result']}) centre | {z['max_abs_diff_c']:.3g} |", f"| stereo zonotopes generators | {z['max_abs_diff_G']:.3g} |"]
    if "truth_obstacles" in inputs:
        L.append(f"| truth obstacles (n={inputs['truth_obstacles']['n']}) | {inputs['truth_obstacles']['max_abs_diff']:.3g} |")
    L += ["", "## Teleop carry regression (all non-timing fields, rtol 1e-9)", "",
          "| run | OCS vs direct: fields differing / worst rel | OCS vs Oct-9 summary: fields differing / worst rel | direct vs Oct-9 summary |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {r.get('ocs_vs_direct_n_fields_differ', '-')} / {r.get('ocs_vs_direct_worst_rel', float('nan')):.2g} | "
                 f"{r.get('ocs_vs_summary_n_fields_differ', '-')} / {r.get('ocs_vs_summary_worst_rel', float('nan')):.2g} | "
                 f"{r.get('direct_vs_summary_n_fields_differ', '-')} / {r.get('direct_vs_summary_worst_rel', float('nan')):.2g} |")
    (OUT / "e1_equivalence.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["inputs", "regress", "report", "all"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what in ("inputs", "all"):
        inp = inputs_equivalence()
        (OUT / "e1_inputs.json").write_text(json.dumps(inp, indent=1, default=str))
        print(json.dumps(inp, indent=1, default=str), flush=True)
    if a.what in ("regress", "all"):
        res = run_regress(a.seeds, a.procs)
        (OUT / "e1_regress_raw.json").write_text(json.dumps(res, indent=1, default=str))
    if a.what in ("report", "all"):
        inp = json.loads((OUT / "e1_inputs.json").read_text())
        res = json.loads((OUT / "e1_regress_raw.json").read_text())
        report(a.seeds, res, inp)


if __name__ == "__main__":
    main()
