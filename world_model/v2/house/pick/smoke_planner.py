"""pick/smoke_planner.py -- run the single-vs-bimanual planner on EVERY proposal of the pick scenes and tabulate VLM vs rule (MOCK backend by default).

    python -m world_model.v2.house.pick.smoke_planner --sets default,basket --planner vlm+veto --backend mock --out DIR
`--backend live` talks ONLY to the owner-launched file-queue daemon (world_model.v2.vlm.service); it is NOT the default and is never started by the tests.
Free house objects: the h5 house has none (see props.py), so the proposals are the props only.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from world_model.v2.house.teleop import sim as S
from . import session as SE
from .smoke import LABELS


def run(sets, planner, backend, out, seeds=(0,), mock_policy="default", wait_s=30.0, vlm_timeout_s=10.0):
    rows = []
    for st in sets:
        for seed in seeds:
            sim = S.TeleopSim("h5", with_people=False, task="pick", pick_props=st, seed=seed)
            for _ in range(25):
                sim.tick(S.Cmd())
            ps = SE.PickSession(planner=planner, backend=backend, mock_labels=LABELS, out_dir=out, seed=seed, mock_policy=mock_policy,
                                vlm_timeout_s=vlm_timeout_s)
            t0 = time.time()
            ps.propose(sim)
            for _ in range(int(wait_s / 0.05)):
                ps.refresh_decisions()
                if all(p["planner"]["status"] in ("done", "rule") for p in ps.proposals()):
                    break
                time.sleep(0.05)
            for p in ps.proposals():
                pl = p["planner"]
                rows.append(dict(set=st, seed=seed, id=p["id"], label=p["label"], extent=p["extent"], centroid=p["centroid"], rule=pl["rule"], vlm=pl["vlm"], final=dict(mode=p["mode"], arm=p["arm"]),
                                 vetoed=pl["vetoed"], reasons=pl["reasons"], fallback=pl["fallback"], latency_s=pl["latency_s"], source=pl["source"]))
            log = ps.vlm_log()
            ps.close()
    return rows, log


def table(rows):
    n = len(rows)
    vl = [r for r in rows if r["vlm"] is not None]
    agree = sum(1 for r in vl if ("single" if r["vlm"]["arms"] == 1 else "bimanual") == r["rule"]["mode"] and (r["rule"]["mode"] != "single" or r["vlm"]["arm"] == r["rule"]["arm"]))
    lat = [r["latency_s"] for r in vl if r["latency_s"] is not None]
    lines = [f"{'set':9s} {'label':8s} {'extent cm':18s} {'rule':14s} {'vlm':14s} {'final':14s} veto"]
    for r in rows:
        e = "x".join(f"{100*v:.0f}" for v in r["extent"])
        rl = f"{r['rule']['mode']}/{r['rule']['arm']}"
        vm = "-" if r["vlm"] is None else f"{'single' if r['vlm']['arms']==1 else 'bimanual'}/{r['vlm']['arm']}"
        lines.append(f"{r['set']:9s} {r['label']:8s} {e:18s} {rl:14s} {vm:14s} {r['final']['mode']+'/'+r['final']['arm']:14s} {'VETO: ' + '; '.join(r['reasons']) if r['vetoed'] else ''}")
    summ = dict(n=n, n_with_vlm=len(vl), agree=agree, vetoes=sum(1 for r in rows if r["vetoed"]), fallbacks=sum(1 for r in rows if r["fallback"]), latency_p50_s=float(np.median(lat)) if lat else None)
    return "\n".join(lines), summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="default,basket")
    ap.add_argument("--planner", default="vlm+veto")
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--out", default=SE.OUT_DIR)
    ap.add_argument("--mock-policy", default="default", help="default | naive_single | naive_two (mock backend only)")
    ap.add_argument("--wait-s", type=float, default=30.0, help="max wait per set for all VLM answers (reasoning models need more)")
    ap.add_argument("--vlm-timeout", type=float, default=10.0, help="planner latency budget; slower answers fall back to the rule")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows, log = run(a.sets.split(","), a.planner, a.backend, a.out, mock_policy=a.mock_policy, wait_s=a.wait_s, vlm_timeout_s=a.vlm_timeout)
    txt, summ = table(rows)
    print(txt)
    print(json.dumps(summ))
    with open(os.path.join(a.out, f"planner_table_{a.backend}_{a.mock_policy}.json"), "w") as f:
        json.dump(dict(rows=rows, summary=summ, vlm_log=log, backend=a.backend, planner=a.planner), f, indent=1, default=str)


if __name__ == "__main__":
    main()
