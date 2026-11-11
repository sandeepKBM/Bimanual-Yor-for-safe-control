"""nero/contents/baseline_ungated.py -- AgentChord-style decision layer on the hidden-contents tasks.  SIMULATION ONLY.

NOT AgentChord itself (their planner needs GPT + their DexSim; not re-run here).  Their MECHANISMS, verified in their code (10-27/28,
memory agentchord-probe-setup): no force / payload / grasp check before acting; monitors are single-step POSE checks (object moved
> 1-2 cm since the previous step relative to the gripper; gripper opening below a threshold); recovery = regrasp.  Under the rigid weld
the object never moves relative to the claws and the claws stay as commanded, so neither monitor can fire: the policy is "execute the
plan".  Same plan, same controller (carry.LeverCtl, m_hat = cell centre), same speed (a_nom) as ours.
  * cells WE ADMIT: the executed plan is identical to ours, so their nominal outcome IS our nominal outcome on the same worlds
    (results/contents_carry_cert_2026-11-02); they have no stored backup.
  * cells WE REFUSE: run here, n worlds per cell, seeds fixed per (task, cell, i).
The single-step pose monitor is evaluated on the trace anyway (object displacement relative to the left claw > 1 cm between control
steps) and reported.

    python -m world_model.v2.nero.contents.baseline_ungated --n 20 --procs 10 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from . import carry as CR
from . import carry_cert as CC
from . import envelope as E
from . import scenes as SC

MON_STEP = 0.01          # m, their single-step "object moved" threshold (lower end of 1-2 cm, verified in their code 10-27)


def seed_of(task, cell, i):
    return int(hashlib.sha256(f"{task}|{cell[0]}-{cell[1]}|{i}".encode()).hexdigest()[:8], 16)


def _trial(args):
    task, cell, i, m_hat, *rest = args                       # 4-tuple jobs from runs launched before --fric-comp
    fc = rest[0] if rest else None
    ws = rest[1] if len(rest) > 1 else None                  # weld solref (None = MuJoCo default)
    seed = seed_of(task, cell, i)
    w = CC.sample_world(task, cell, np.random.default_rng(seed))
    t0 = time.time()
    try:
        ep = CR.CarryEpisode(w, a_plan=E.A_NOM, m_hat=m_hat, fric_comp=fc, weld_solref=ws)
        ep.run_phases([("hold", 0.4)])
        pre_hold_viol = ep.track["viol"]                     # a U entry during the load-transfer hold counts (R1 unsafe)
        segs, mon = [], dict(fired=False, max_step_m=0.0)
        rel_prev = None

        def step_with_monitor(_ep):
            nonlocal rel_prev
            d = ep.d
            rel = d.xmat[ep.gb["left"]].reshape(3, 3).T @ (d.xpos[ep.obj] - d.xpos[ep.gb["left"]])
            if rel_prev is not None:
                dd = float(np.linalg.norm(rel - rel_prev))
                mon["max_step_m"] = max(mon["max_step_m"], dd)
                mon["fired"] = mon["fired"] or dd > MON_STEP
            rel_prev = rel
        ep.step_hooks.append(step_with_monitor)
        for seg in CR.SEGS:
            ep.reset_track()
            ep.run_phases(ep.segment_phases(seg))
            okR, why = ep.in_R()
            segs.append(dict(seg=seg, viol=ep.track["viol"], end_in_R=okR, R_fail=why,
                             **{k: round(float(v), 4) for k, v in ep.track.items() if k != "viol"}))
        tr = SC.object_truth(ep.m, ep.d)
        dest = ep.d.xpos[ep.obj][2] - ep.z0
        r = dict(task=task, cell=list(cell), i=i, seed=seed, theta=CR._theta(w), segments=segs, monitor=mon, fric_comp=bool(fc),
                 weld_solref=ws,
                 end_in_R=ep.in_R(), dz_end=round(float(dest), 4), any_U=any(s["viol"] for s in segs),
                 first_U=next((dict(seg=s["seg"], **s["viol"]) for s in segs if s["viol"]), None),
                 truth_total_kg=round(float(tr["total_mass"]), 3), wall_s=round(time.time() - t0, 1))
        from world_model.v2.adapters import outcome_carry as OCC      # the ONE goal + success rule (same as ours and AgentChord)
        r.update(pre_hold_viol=pre_hold_viol, **OCC.measure(ep, task, True, bool(r["any_U"]) or pre_hold_viol is not None, pre_hold_viol, end_in_R=tuple(r["end_in_R"])))
        if ep.planner != "scripted":
            r["planner"] = ep.planner
    except Exception as e:
        r = dict(task=task, cell=list(cell), i=i, seed=seed, error=repr(e))
    return r


def main(argv=None):
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tray6,box")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--static", default="/data/local/ss5772/contents/carry_cert_2026-11-02/STATIC.json")
    ap.add_argument("--out", default="/data/local/ss5772/contents/baseline_ungated_2026-11-02")
    ap.add_argument("--fric-comp", action=argparse.BooleanOptionalAction, default=None, help="default = config.carry_physics()")
    ap.add_argument("--weld-stiff", action=argparse.BooleanOptionalAction, default=None,
                    help=f"weld solref {CC.WELD_STIFF} (as carry_cert --weld-stiff); default = config.carry_physics()")
    ap.add_argument("--planner", choices=CR.PLANNERS, default=None,
                    help="plan executed (default: env BIMANUAL_PLANNER, else scripted = legacy PLANS)")
    a = ap.parse_args(argv)
    if a.planner is not None:
        import os
        os.environ["BIMANUAL_PLANNER"] = a.planner                  # inherited by the spawned workers
    cp = CC.C.carry_physics()                                # 11-03 item 6: one source for the physics defaults
    a.fric_comp = bool(cp["fric_comp"]) if a.fric_comp is None else a.fric_comp
    ws = (cp["weld_solref"] if a.weld_stiff is None else (CC.WELD_STIFF if a.weld_stiff else None))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads(pathlib.Path(a.static).read_text())["cells"]
    jobs = []
    for key, c in st.items():
        task, cs = key.split("|")
        cell = tuple(float(x) for x in cs.split("-"))
        if task in a.tasks.split(",") and c["static"]["a_admit"] is None:          # the cells WE refuse
            m_hat = 0.5 * sum(c["static"]["contents_m"])
            jobs += [(task, cell, i, m_hat, a.fric_comp, ws) for i in range(a.n)]
    print(f"{len(jobs)} trials", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=3) as pool, open(out / "trials.jsonl", "a") as f:
        for r in pool.imap_unordered(_trial, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    print(f"wall {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
