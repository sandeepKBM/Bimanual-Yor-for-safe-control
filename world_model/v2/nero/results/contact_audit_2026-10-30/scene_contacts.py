"""contact points between each claw's finger meshes and the pan / lid in the real scene after the clamp (I03 and V10, v1 chain)."""
import json, os, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.cert import run_trials as RTR
out = {}
for tag in ("I03", "V10"):
    r = json.load(open(f"world_model/v2/nero/results/pan_cert_v2_2026-10-29/smokeB/raw/{tag}.json"))
    ep = S.Episode(r["world"], observe="zed", obs_seed=S.world_seed(tag), est=r["est"], **RTR.PLAN)
    ep.reset_track(); ep.run_phases(ep.segment_phases("start"))
    for seg in ("pinch", "push", "clamp"):
        ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy(); ep.run_phases(ep.segment_phases(seg))
    m, d = ep.m, ep.d
    rec = {}
    for side in S.SIDES:
        for f in ("finger1", "finger2"):
            g = m.geom(f"{side}_arm_{f}_col").id
            pts, fn = [], []
            for c in d.contact[:d.ncon]:
                if g in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == g else c.geom1
                    if m.body(m.geom_bodyid[other]).name in ("pan", "lid"):
                        f6 = np.zeros(6)
                        import mujoco
                        mujoco.mj_contactForce(m, d, list(d.contact).index(c) if False else 0, f6)
                        pts.append(c.pos.copy())
            P = np.array(pts) if pts else np.zeros((0, 3))
            spread = float(max(np.linalg.norm(a - b) for a in P for b in P)) if len(P) > 1 else 0.0
            rec[f"{side}_{f}"] = dict(n=len(P), spread_mm=round(1e3 * spread, 2))
    out[tag] = rec
    print(tag, rec, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
