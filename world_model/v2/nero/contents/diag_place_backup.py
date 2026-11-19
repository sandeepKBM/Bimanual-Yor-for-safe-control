"""nero/contents/diag_place_backup.py -- diagnosis of the stored recovery of the place step (table-behind tasks).  SIMULATION ONLY, offline (proposer cache replay only).
11-13 fix F1.  Rebuilds a world of a same_worlds run (CONFIG.json: eps, plan spec / cache, probe method), runs ours exactly as the run did (behind_job.v2_job) and, at every
branch point of the chosen segment, logs the stored backup rollout (executor.run_segments -> ep.backup_runner).

One JSONL line per (world, branch point):
  jretreat      the knot the jretreat starts from: commanded q vs the retreat path's first q (rad, per side), nearest global knot index of the commanded q and its distance
                to knot 0 (rad), taken at the moment the jretreat phase starts (None if the phase list has none)
  claw          per claw: samples over F_cap, longest run over it [samples, s], max |Fz|, threshold; (a 5 ms single-sample rule = n_over >= 1; a 0.05 s dwell rule = longest_s >= 0.05)
                + a series of (t, |Fz|) from 0.25 s before the first flag to 0.5 s after it (every control step)
  obj           object z above the goal support top at weld-off and whether it touches a support geom then
  first_U       kind and time of the first U flag of the rollout; viol_all = every kind (count + first)
    python -m world_model.v2.nero.contents.diag_place_backup --run-dir DIR --worlds box_behind:2,3,11,15,18,19 --out OUT.jsonl [--seg s2_place] [--max-branches N]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import numpy as np

SIDES = ("left", "right")
DT_NOTE = "control step DT = carry.DT"


def _knot_info(ep, q_cmd):
    g = ep.cc.geo
    allq = np.hstack([g.q["left"], g.q["right"]])
    qc = np.concatenate([q_cmd["left"], q_cmd["right"]])
    d = np.linalg.norm(allq - qc, axis=1)
    return dict(nearest_knot=int(np.argmin(d)), nearest_dist_rad=float(d.min()), dist_to_knot0_rad=float(d[0]), n_knots=int(len(d)))


def _touch_support(ep):
    import mujoco
    m, d = ep.m, ep.d
    sup = set(getattr(ep, "support_geoms", {}).values())
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in sup and m.geom_bodyid[g2] == ep.obj) or (g2 in sup and m.geom_bodyid[g1] == ep.obj):
            if c.dist < 1e-3:
                return True
    return False


def make_runner(sink, world_tag):
    """-> runner(ep, seg, tr, phases): runs the backup phase by phase, logging; appends one dict to `sink`."""
    from world_model.v2.nero.contents import carry as CR
    from world_model.v2.nero.base import scene as BS

    def runner(ep, seg, tr, phases):
        rec = dict(world=world_tag, seg=seg, branch_t=tr, phases=[list(map(_j, p)) for p in phases], jretreat=None, obj=None)
        series = {s: [] for s in SIDES}
        orig = ep._check

        def chk(t, r):
            for s in SIDES:
                F, _ = ep.claw_wrench(s)
                series[s].append((float(t), abs(float(F[2]))))
            return orig(t, r)
        ep._check = chk
        try:
            for ph in phases:
                if ph[0] == "jretreat":
                    t = ep.k * CR.DT
                    qc = {s: np.asarray(ep._cmd(s, t), float) for s in SIDES}
                    rt = ep.cc.retreat(ph[1])
                    rec["jretreat"] = dict(where=ph[1], knot=_knot_info(ep, qc), cmd_vs_retreat_q0_rad={s: float(np.linalg.norm(qc[s] - rt["q"][s][0])) for s in SIDES},
                                           cmd_q={s: [round(float(x), 4) for x in qc[s]] for s in SIDES},
                                           retreat_q0={s: [round(float(x), 4) for x in rt["q"][s][0]] for s in SIDES})
                if ph[0] == "weld" and ph[1] is False and rec["obj"] is None:
                    top = BS.support_tops_behind(ep.world)[ep.goal_support or "counter2"]["top"]
                    rec["obj"] = dict(z_above_goal_support_m=float(ep.d.xpos[ep.obj][2] - top - 0.0), touches_support=_touch_support(ep), t=round(ep.k * CR.DT, 3),
                                      note="z = object body origin minus the goal support top (box: origin height at rest = z0 - counter top, see z_rest)",
                                      z_rest_m=float(ep.z0 - ep.world.counter_top) if hasattr(ep, "z0") else None)
                ep.run_phases([ph])
        finally:
            ep._check = orig
        thr = float(ep.F_cap)
        claw = {}
        for s in SIDES:
            arr = np.array(series[s]) if series[s] else np.zeros((0, 2))
            over = arr[:, 1] > thr if len(arr) else np.zeros(0, bool)
            longest, run = 0, 0
            for o in over:
                run = run + 1 if o else 0
                longest = max(longest, run)
            first = int(np.argmax(over)) if over.any() else None
            claw[s] = dict(thr_N=thr, n_samples=int(len(arr)), n_over=int(over.sum()), longest_run_samples=longest, longest_run_s=round(longest * CR.DT, 4),
                           max_Fz_N=float(arr[:, 1].max()) if len(arr) else None, first_over_t=None if first is None else float(arr[first, 0]),
                           series=None if first is None else [[round(a, 4), round(b, 2)] for a, b in arr[max(0, first - int(0.25 / CR.DT)):first + int(0.5 / CR.DT)]])
        rec["claw"] = claw
        v = ep.track["viol"]
        rec["first_U"] = None if not v else dict(kind=v.get("kind"), t=v.get("t"))
        rec["viol_all"] = json.loads(json.dumps(ep.track["viol_all"], default=float))
        sink.append(rec)
    return runner


def _j(x):
    return x if isinstance(x, (str, int, float, bool)) or x is None else (list(x) if isinstance(x, (tuple, list)) else str(x))


def run_world(run_dir, task, i, seg, out_f, max_branches=None, plans=None):
    from world_model.v2.nero.contents import behind_job as BJ, same_worlds as SW, carry as CR
    from world_model.v2.nero.base import smoke as SM
    cfg = json.loads((pathlib.Path(run_dir) / "CONFIG.json").read_text())
    SW._init_worker()                                                         # strict_claw=True for every episode, as in the run
    if plans is None:
        plans = _plans(cfg, [task], out_f)
    plan = plans[task]["plan"]
    seed, w = SM.behind_world(task, i)
    sink = []
    n_seen = [0]

    def hooks(ep):
        run = make_runner(sink, f"{task}:{i}")

        def limited(ep_, seg_, tr, ph):
            if seg_ != seg or (max_branches is not None and n_seen[0] >= max_branches):
                return ep_.run_phases(ph)
            n_seen[0] += 1
            return run(ep_, seg_, tr, ph)
        ep.backup_runner = limited
    # v2_job swallows nothing here: it returns a record (never raises); the runner fills `sink`
    r = BJ.v2_job(task, i, seed, w, float(cfg["eps_lo"]), float(cfg["eps_free"]), plan, cfg.get("probe_method"), True, hooks=hooks)
    summ = dict(world=f"{task}:{i}", outcome=r.get("outcome"), error=r.get("error"), tb=r.get("tb"), first_U_nominal=r.get("first_U"), n_branches_logged=len(sink))
    segs = (r.get("carry") or {}).get("segments", [])
    summ["family"] = {s["seg"]: s["family"] for s in segs}
    return summ, sink


def _plans(cfg, tasks, out):
    from world_model.v2.nero.contents import same_worlds as SW
    a = types.SimpleNamespace(plan_spec=cfg["plan_spec"], plan_cache=cfg["plan_cache"], offline=True, outside_b=cfg.get("outside_b", "refuse"), reask_limit=cfg.get("reask_limit"))
    d = pathlib.Path(str(out) + ".plans")
    d.mkdir(parents=True, exist_ok=True)
    return SW._proposer_plans(tasks, a, d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worlds", required=True, help="task:i,i,...[;task:i,...]  e.g. box_behind:2,3,11,15,18,19")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", default="s2_place")
    ap.add_argument("--max-branches", type=int, default=None)
    a = ap.parse_args(argv)
    todo = []
    for part in a.worlds.split(";"):
        t, ids = part.split(":")
        todo += [(t, int(x)) for x in ids.split(",")]
    cfg = json.loads((pathlib.Path(a.run_dir) / "CONFIG.json").read_text())
    plans = _plans(cfg, sorted({t for t, _ in todo}), a.out)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for t, i in todo:
            summ, sink = run_world(a.run_dir, t, i, a.seg, a.out, a.max_branches, plans)
            for rec in sink:
                f.write(json.dumps(dict(kind="branch", **rec), default=float) + "\n")
            f.write(json.dumps(dict(kind="world", **summ), default=float) + "\n")
            f.flush()
            print(f"{t}:{i} outcome={summ['outcome']} err={summ['error']} branches={len(sink)}", flush=True)


if __name__ == "__main__":
    main()
