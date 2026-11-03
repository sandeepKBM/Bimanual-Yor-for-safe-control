"""11-01 pot smoke at a claw force limit (USER: 40 N NERO variant approved). SIMULATION ONLY, CPU. Usage: TAG OUT CLAW_N GRIP_N"""
import json, os, sys, time
from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.cert import contact_falsify as F
from world_model.v2.nero.pan.segcert import worlds
tag, out, claw, grip = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
w = dict(worlds("full") + worlds("iid") + worlds("overload"))[tag]
t0 = time.time()
ep = S.Episode(w, observe="zed", obs_seed=S.world_seed(tag), obj="pot", grasp_frac=F.PLAN["grasp_frac"], grip_force=grip, claw_N=claw)
r = dict(tag=tag, claw_N=ep.claw_N, grip_N=grip, world=w, obs=ep.obs, refused=ep.refused)
if not ep.refused:
    ep.run_phases(ep.segment_phases("start")); ep.run_phases(ep.segment_phases("pinch"))
    r["pinch_contacts"] = F._mj_contacts(ep)[1]
    snap = ep.snapshot("lift_start"); r["lift"] = {}
    for v in ("v1", "v3"):
        ep.restore(snap); ep.reset_track()
        lf = F._lift(ep, v); lf["fail"] = F.failed(lf); r["lift"][v] = lf
r["wall_s"] = round(time.time() - t0, 1)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, f"{tag}_c{int(claw)}.json"), "w").write(json.dumps(r, default=float)); print(tag, "done", flush=True)
