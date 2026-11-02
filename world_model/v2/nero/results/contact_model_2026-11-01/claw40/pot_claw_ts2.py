"""11-01 diag: pot at 40 N, physics step 5 ms vs 1 ms (5 substeps, torque held), to tell a contact artefact from physics.
SIMULATION ONLY, CPU. Usage: TAG OUT NSUB [CLAW_N]"""
import json, os, sys, time
import mujoco
import numpy as np
tag, out, NSUB = sys.argv[1], sys.argv[2], int(sys.argv[3]); CLAW = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
from world_model.v2.nero.pan import segments as S, handle_grasp as HG, scene as SC, under_rim as UR
from world_model.v2.nero.pan.cert import contact_falsify as F
from world_model.v2.nero.pan.segcert import worlds


class _Sub:
    def __getattr__(self, n):
        return getattr(mujoco, n)

    @staticmethod
    def mj_step(m, d):
        if NSUB == 1:
            return mujoco.mj_step(m, d)
        dt = m.opt.timestep
        m.opt.timestep = dt / NSUB
        try:
            for _ in range(NSUB):
                mujoco.mj_step(m, d)
        finally:
            m.opt.timestep = dt


for mod in (S, HG, SC, UR):
    mod.mujoco = _Sub()
w = dict(worlds("full") + worlds("iid") + worlds("overload"))[tag]
t0 = time.time()
ep = S.Episode(w, observe="zed", obs_seed=S.world_seed(tag), obj="pot", grasp_frac=F.PLAN["grasp_frac"], grip_force=CLAW, claw_N=CLAW)
r = dict(tag=tag, nsub=NSUB, claw_N=CLAW, refused=ep.refused)
lid0 = ep.d.xpos[ep.lid].copy() - ep.d.xpos[ep.pan]
ep.run_phases(ep.segment_phases("start")); ep.run_phases(ep.segment_phases("pinch"))
r["lid_after_pinch_mm"] = 1e3 * float(np.linalg.norm(ep.d.xpos[ep.lid] - ep.d.xpos[ep.pan] - lid0))
r["pinch_contacts"] = F._mj_contacts(ep)[1]
snap = ep.snapshot("lift_start"); r["lift"] = {}
for v in ("v1", "v3"):
    ep.restore(snap); ep.reset_track()
    lf = F._lift(ep, v); lf["fail"] = F.failed(lf); r["lift"][v] = lf
r["wall_s"] = round(time.time() - t0, 1)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, f"{tag}_n{NSUB}_c{int(CLAW)}.json"), "w").write(json.dumps(r, default=float)); print(tag, NSUB, "done", flush=True)
