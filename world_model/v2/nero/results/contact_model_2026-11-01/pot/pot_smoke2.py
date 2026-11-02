"""v2 11-01: + yaw fold + v3s (stiff right wrist). pot (two side handles) smoke: chain start + pinch both, then lifts v1 / v3 from the same snapshot. SIMULATION ONLY, CPU."""
import json, os, sys, time
from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.cert import contact_falsify as F
from world_model.v2.nero.pan.segcert import worlds
tag, out = sys.argv[1], sys.argv[2]
w = dict(worlds("full") + worlds("iid") + worlds("overload"))[tag]
t0 = time.time()
ep = S.Episode(w, observe="zed", obs_seed=S.world_seed(tag), obj="pot", **{k: v for k, v in F.PLAN.items() if k != "push"})
r = dict(tag=tag, world=w, obs=ep.obs, refused=ep.refused)
if not ep.refused:
    ep.run_phases(ep.segment_phases("start"))
    ep.run_phases(ep.segment_phases("pinch"))
    r["pinch_contacts"] = F._mj_contacts(ep)[1]
    snap = ep.snapshot("lift_start")
    r["lift"] = {}
    for v in ("v1", "v3", "v3s"):
        ep.restore(snap); ep.reset_track()
        if v == "v3s": ep.ctl.fl.W.update(pi=3000.0, vi=15.0)   # in-memory: right arm as stiff as the left (mpc_fl.py not edited)
        lf = F._lift(ep, "v3" if v == "v3s" else v); lf["fail"] = F.failed(lf); r["lift"][v] = lf
r["wall_s"] = round(time.time() - t0, 1)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, f"{tag}.json"), "w").write(json.dumps(r, default=float))
print(tag, "done", flush=True)
