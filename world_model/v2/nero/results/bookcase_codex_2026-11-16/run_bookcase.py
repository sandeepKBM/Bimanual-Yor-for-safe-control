"""box_bookcase with a proposer-written plan (USER 11-16: "use this prompt now and run it").  SIMULATION ONLY.  OURS only (no AgentChord column: same_worlds
refuses a table-behind task without one).  Same path as same_worlds' "ours" job: _proposer_plans (propose_until_accepted -> compile at L1) -> behind_job per world
(probe -> Theta -> static admission -> base level -> Rule 1 / T4 when merged -> carry).

    PYTHONPATH=. MUJOCO_GL=osmesa python world_model/v2/nero/results/bookcase_codex_2026-11-16/run_bookcase.py \
        --spec codex:gpt-5.6-sol:high --plan-images zed+look --worlds 0 --out OUT --plan-cache OUT/cache

--spec mock:fixture = offline dry run (no network).  A live spec sends the prompt text and the two rendered images to that service (nothing else).
"""
import argparse
import json
import pathlib
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="mock:fixture")
    ap.add_argument("--plan-images", default="zed+look", choices=("none", "zed", "zed+look"))
    ap.add_argument("--worlds", default="0", help="comma list of world indices (nero/base/smoke.behind_world)")
    ap.add_argument("--plan-cache", required=True)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--propose-only", action="store_true", help="ask the proposer (and record / cache its plan), run no world")
    a = ap.parse_args()
    task = "box_bookcase"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from world_model.v2.nero.contents import probe as P, same_worlds as SW, behind_job as BJ
    method = P.PROBE_METHOD
    cal = P.load_calib(P.default_calib(method), method)["eps"]
    eps_lo, eps_free = float(cal["EPS_LO"]), float(cal["EPS_FREE"])
    worlds = [int(x) for x in a.worlds.split(",")]
    ns = argparse.Namespace(plan_spec=a.spec, plan_cache=a.plan_cache, offline=a.offline, outside_b="refuse", plan_images=a.plan_images,
                            plan_image_world=worlds[0], reask_limit=None)
    t0 = time.time()
    ex = SW._proposer_plans([task], ns, out)[task]
    (out / "proposer.json").write_text(json.dumps({k: v for k, v in ex.items() if k != "plan" or True}, default=str, indent=1))
    print("proposer:", ex.get("plan_record", {}).get("status"), ex.get("plan_id"), "rejected:", ex.get("rejected"), f"{time.time() - t0:.0f}s", flush=True)
    if ex.get("rejected") or a.propose_only:
        return
    for i in worlds:
        t1 = time.time()
        r = BJ.behind_job((task, i, eps_lo, eps_free, {**ex["plan"], "probe_method": method}), method=method)
        (out / f"world_{i}.json").write_text(json.dumps(r, default=str, indent=1))
        v2 = r.get("v2") or {}
        print(f"world {i}: outcome={r.get('outcome')} refused_by={r.get('refused_by')} any_U={r.get('any_U')} t_counts={v2.get('t_counts')} "
              f"reask_end={v2.get('reask_end')} level={r.get('base_level')} {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
