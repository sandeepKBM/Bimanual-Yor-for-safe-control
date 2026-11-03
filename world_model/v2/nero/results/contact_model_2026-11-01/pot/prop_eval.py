"""Evaluate one proposer plan on the pot in sim. SIMULATION ONLY, CPU. Usage: TAG KIND(best|alternative) REPLY.json OUTDIR"""
import json, os, sys, time
from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.cert import contact_falsify as F
from world_model.v2.nero.pan.segcert import worlds
tag, kind, rp, out = sys.argv[1:5]
pl = json.load(open(rp))["plans"][tag][kind]
lift = {"coupled": "v3", "independent": "v1"}[pl["lift_skill"]]
w = dict(worlds("full") + worlds("iid") + worlds("overload"))[tag]
t0 = time.time()
ep = S.Episode(w, observe="zed", obs_seed=S.world_seed(tag), obj="pot", grasp_frac=pl["grasp_frac"], grip_force=pl["grip_force"])
r = dict(tag=tag, kind=kind, plan=pl, lift=lift, world=w, obs=ep.obs, refused=ep.refused)
if not ep.refused:
    ep.run_phases(ep.segment_phases("start")); ep.run_phases(ep.segment_phases("pinch"))
    r["pinch_contacts"] = F._mj_contacts(ep)[1]
    ep.reset_track()
    lf = F._lift(ep, lift); lf["fail"] = F.failed(lf); r["result"] = lf
r["wall_s"] = round(time.time() - t0, 1)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, f"{tag}_{kind}.json"), "w").write(json.dumps(r, default=float)); print(tag, kind, "done", flush=True)
