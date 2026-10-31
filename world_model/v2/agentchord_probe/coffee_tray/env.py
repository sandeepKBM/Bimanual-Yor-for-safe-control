"""CoffeeTrayAgent-v3: 'setup coffee tray' (AgentChord paper, real-robot task, Table II 60%) as a task in THEIR sim.

Imported inside the container by ac_run.py (their code is not edited). Same base classes as their RearrangementAgentEnv.
Success = the paper's criterion made numeric (thresholds are ours, pre-registered in the README):
  capsule inside the tray's inner footprint (tray frame |x| <= 0.10, |y| <= 0.16) and resting on it (z within 0.04 m of the tray),
  tray within 0.03 m (xy) of its reset position + [0.10, 0], tray tilt < 10 deg.
"""
import json
import os

import numpy as np
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from embodichain.lab.gym.utils.registration import register_env

GEOM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry.json")))
INNER_X = GEOM["tray"]["lx"] / 2 - GEOM["tray"]["rim_t"]
INNER_Y = GEOM["tray"]["ly"] / 2 - GEOM["tray"]["rim_t"]


def _pose(env, uid):
    p = env.sim.get_rigid_object(uid).get_local_pose(to_matrix=True)
    p = p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
    return p.reshape(4, 4)


def coffee_tray_check(env, tray0, loose=False):
    tray, cap = _pose(env, "tray"), _pose(env, "capsule")
    rel = np.linalg.inv(tray) @ cap
    target = tray0[:2, 3] + np.array(GEOM["forward"])
    out = {
        "capsule_in_tray": bool(abs(rel[0, 3]) <= INNER_X and abs(rel[1, 3]) <= INNER_Y and abs(rel[2, 3]) <= 0.04),
        "tray_at_target": bool(np.linalg.norm(tray[:2, 3] - target) <= 0.03),
        "tray_level": bool(np.degrees(np.arccos(np.clip(tray[2, 2], -1, 1))) < 10),
        "capsule_rel_tray": rel[:3, 3].round(4).tolist(),
        "tray_xy_err": float(np.linalg.norm(tray[:2, 3] - target)),
        # minimal prompt gives no distance: 'moved forward' = >= 5 cm in +x, <= 5 cm sideways
        "tray_moved_forward": bool(tray[0, 3] - tray0[0, 3] >= 0.05 and abs(tray[1, 3] - tray0[1, 3]) <= 0.05),
        "tray_dxy": (tray[:2, 3] - tray0[:2, 3]).round(4).tolist(),
    }
    placed = out["tray_moved_forward"] if loose else out["tray_at_target"]
    out["success_strict"] = out["capsule_in_tray"] and out["tray_at_target"] and out["tray_level"]
    out["success_loose"] = out["capsule_in_tray"] and out["tray_moved_forward"] and out["tray_level"]
    out["success"] = out["capsule_in_tray"] and placed and out["tray_level"]
    return out


@register_env("CoffeeTrayAgent-v3", max_episode_steps=600)
class CoffeeTrayAgentEnv(BaseAgentEnv, EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        self.loose_success = kwargs.get("task_name") == "CoffeeTrayMin"
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed=None, options=None):
        self.tray0 = None  # their reset() calls is_task_success() before we can record the new start pose
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        self.tray0 = _pose(self, "tray")
        return obs, info

    def check_detail(self):
        return coffee_tray_check(self, self.tray0, loose=self.loose_success)

    def is_task_success(self):
        if getattr(self, "tray0", None) is None:
            return False
        return self.check_detail()["success"]
