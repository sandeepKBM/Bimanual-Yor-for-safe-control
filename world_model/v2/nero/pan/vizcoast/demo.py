"""Demo / experiment driver for the pan constraint layer.  SIMULATION ONLY.

    python -m world_model.v2.nero.pan.vizcoast.demo --out DIR [--belief prereg|overload] [--proposer good|none|wrong|overprune|garbage|live]
           [--stageB --procs N]

--proposer llm --llm SPEC uses pan/llm.make(SPEC) (default codex:gpt-5.6-sol:high; later the open-weight model), cached in
DIR/llm_cache; every result records the model, reasoning effort and caveats (proposer_model).
--proposer live uses the open-weight Qwen daemon through house/lang/backend.LiveVLM wrapped in CachedVLM (DIR/vlm_cache);
it never starts the daemon and exits if it is down.  --stageB runs the falsification search in sim (CPU, N processes).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import domain as D
from . import evidence as E
from . import facts as F
from . import planner as P

BELIEFS = dict(prereg=D.PREREG, overload=D.OVERLOAD)


def backend(kind, out, llm_spec=None):
    if kind == "llm":
        from .. import llm as L
        return L.make(llm_spec, Path(out) / "llm_cache")
    if kind != "live":
        return F.mock_backend(kind)
    from world_model.v2.house.lang.backend import CachedVLM, LiveVLM
    live = LiveVLM()
    if not live.available():
        raise SystemExit("VLM daemon is not up; this script never starts it (world_model/v2/vlm/slurm/daemon.sbatch)")
    return CachedVLM(live, Path(out) / "vlm_cache")


def _model_info(proposer, llm_spec):
    from .. import llm as L
    if proposer == "llm":
        return L.info(llm_spec)
    if proposer == "live":
        return dict(backend="live", model="Qwen3.5-9B daemon", reasoning="daemon VLM_ENABLE_THINKING", caveats="")
    return L.info("mock:" + proposer)


def blocking_constraints(belief, cons, db):
    """for a no-plan result: which single constraint, when dropped, gives a Stage-A plan."""
    out = []
    for i, c in enumerate(cons):
        r = P.solve(belief, cons[:i] + cons[i + 1:], list(db))
        if r["plan"]:
            out.append(dict(constraint=list(c), plan_without_it=r["plan"]))
    return out


def run(belief_name, proposer, out, stage_b=False, procs=8, max_rounds=10, image=None, llm_spec=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b0 = BELIEFS[belief_name]
    t0 = time.time()
    pr = F.produce(backend(proposer, out, llm_spec), b0, image=image)
    ev_file = out / f"stageB_evidence_{belief_name}.jsonl"
    db = E.load(extra_jsonl=[ev_file])
    sim = None
    if stage_b:
        from . import simcheck as SC
        sim = SC.make_sim(ev_file, procs=procs, raw_dir=out / f"stageB_raw_{belief_name}")
    r = P.solve(pr["belief"], pr["constraints"], db, sim=sim, max_rounds=max_rounds)
    r.pop("plan_obj", None)
    rep = dict(observe_mode=D.OBS_MODE, obs_noise=D.OBS_NOISE if D.OBS_MODE == "built" else None, belief=belief_name, belief_box=b0.describe(), belief_after_proposer=pr["belief"].describe(), proposer=proposer,
               proposer_backend=pr["backend"], proposer_model=_model_info(proposer, llm_spec), proposer_constraints=[list(c) for c in pr["constraints"]], proposer_log=pr["log"],
               proposer_raw=pr["raw"][:2000], n_evidence=len(db), stageB=stage_b, **r, total_wall_s=round(time.time() - t0, 1))
    if r["plan"] is None and pr["constraints"]:
        rep["blocking_constraints"] = blocking_constraints(pr["belief"], pr["constraints"], E.load(extra_jsonl=[ev_file]))
    ptag = proposer if proposer != "llm" else "llm_" + llm_spec.split(":")[1]
    tag = f"{belief_name}_{ptag}{'_B' if stage_b else ''}{'_obsbuilt_' + D.OBS_NOISE if D.OBS_MODE == 'built' else ''}"
    (out / f"{tag}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("belief", "proposer", "plan", "stage", "verdict", "assumptions", "stats", "total_wall_s")}
                     | ({"blocking_constraints": rep["blocking_constraints"]} if "blocking_constraints" in rep else {}),
                     indent=1, default=float), flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--belief", default="prereg", choices=list(BELIEFS))
    ap.add_argument("--proposer", default="good", choices=list(F.CANNED) + ["live", "llm"])
    ap.add_argument("--llm", default=None, help="pan/llm spec for --proposer llm (default pan.llm.DEFAULT)")
    ap.add_argument("--stageB", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--image", default=None)
    ap.add_argument("--observe-mode", default="assumed", choices=["assumed", "built"],
                    help="assumed (10-28): observe_pose shrinks the box; built: the real observe_pose runs inside Stage B on the full box")
    ap.add_argument("--obs-noise", default="zed", help="observe_pose noise level in built mode")
    a = ap.parse_args(argv)
    D.OBS_MODE, D.OBS_NOISE = a.observe_mode, a.obs_noise
    if a.proposer == "llm" and a.llm is None:
        from .. import llm as L
        a.llm = L.DEFAULT
    run(a.belief, a.proposer, a.out, a.stageB, a.procs, a.max_rounds, a.image, a.llm)


if __name__ == "__main__":
    main()
