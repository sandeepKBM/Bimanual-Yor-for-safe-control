"""Physics A/B for the pan lift (USER 11-01: measure only, the MJCF is NOT edited). SIMULATION ONLY, CPU.
Settings applied to the loaded model in memory, for the WHOLE chain (start, pinch, push, clamp) and the lifts:
  base  : as the scene (finger condim 3, physics step 5 ms)
  c4    : finger geoms condim 4 (torsional friction, the fingers' existing friction[1] = 0.01 m, ASSUMED)
  ts1   : physics step 1 ms (5 substeps per 5 ms control step; FL-MPC torque held over the 5 substeps)
  c4ts1 : both
Usage: python physics_ab.py SETTING TAG OUTDIR"""
import json
import os
import sys
import time

import mujoco
import numpy as np

SETTING, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NSUB = 5 if "ts1" in SETTING else 1

from world_model.v2.nero.pan import segments as S          # noqa: E402
from world_model.v2.nero.pan import handle_grasp as HG     # noqa: E402
from world_model.v2.nero.pan import scene as SC            # noqa: E402
from world_model.v2.nero.pan import under_rim as UR        # noqa: E402
from world_model.v2.nero.pan.cert import contact_falsify as F   # noqa: E402
from world_model.v2.nero.pan.segcert import worlds         # noqa: E402


class _Sub:
    """mujoco proxy for the pan modules: mj_step = NSUB substeps of DT/NSUB with the applied torque held (zero-order hold)."""
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

_orig_init = S.Episode.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    if "c4" in SETTING:
        for s in S.SIDES:
            for f in (1, 2):
                self.m.geom_condim[self.m.geom(f"{s}_arm_finger{f}_col").id] = 4


S.Episode.__init__ = _init
# NOTE c4: condim is applied after the Episode is built (its settle runs with open fingers, no finger contacts) -> chain + lifts use it.

w = dict(worlds("full") + worlds("iid") + worlds("overload"))[TAG]
t0 = time.time()
ep = S.Episode(w, observe="zed", obs_seed=S.world_seed(TAG), **F.PLAN)
out = dict(setting=SETTING, tag=TAG, nsub=NSUB,
           finger_condim=int(ep.m.geom_condim[ep.m.geom("right_arm_finger1_col").id]))
if ep.refused:
    out["refused"] = ep.refused
else:
    ep.run_phases(ep.segment_phases("start"))
    for seg in S.SEGMENTS:
        if seg == "lift":
            break
        ep.run_phases(ep.segment_phases(seg))
    snap = ep.snapshot("lift_start")
    out["lift"] = {}
    for v in ("v1", "v2", "v3"):
        ep.restore(snap)
        ep.reset_track()
        lf = F._lift(ep, v)
        lf["fail"] = F.failed(lf)
        out["lift"][v] = lf
out["wall_s"] = round(time.time() - t0, 1)
os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, f"{SETTING}_{TAG}.json")
open(p + ".tmp", "w").write(json.dumps(out, default=float))
os.replace(p + ".tmp", p)
print(SETTING, TAG, "done", out["wall_s"], flush=True)
