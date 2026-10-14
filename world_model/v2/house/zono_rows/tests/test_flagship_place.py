"""tests for the flagship `--place-backend zono_rows` wiring (flagship_place.py + the perception / simulate hooks).  SIMULATION only, CPU, no GPU.
    PYTHONPATH=. python -m world_model.v2.house.zono_rows.tests.test_flagship_place       (or pytest)
The PlaceGuard test builds the flagship TeleopSim (desk start, tray held; ~1 min)."""
from __future__ import annotations

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from world_model.v2.house.zono_rows import flagship_place as FP, rows as R


def _result(wall_y=None, with_payload=True):
    """synthetic flagship 'desk' perception result: an optional wall zonotope across the naive line + a 0.3 x 0.2 x 0.1 m payload block ahead of the base."""
    zonos = []
    if wall_y is not None:
        zonos.append(dict(id=0, kind="obj", gt_key="wall", Zi=dict(c=np.array([0.0, wall_y, 0.8]), G=np.diag([1.0, 0.04, 0.6]))))
    zonos.append(dict(id=1, kind="support", gt_key="desk_top", Zi=dict(c=np.array([0.0, 1.2, 0.30]), G=np.diag([2.0, 1.0, 0.02]))))
    r = dict(kind="desk", zonos=zonos, tray0=[0.0, 0.45, 0.85])
    if with_payload:
        r["payload"] = [("tray", np.array([0.0, 0.45, 0.85]), np.diag([0.15, 0.10, 0.05])), ("chassis", np.array([0.0, 0.0, 0.3]), np.diag([0.15, 0.15, 0.3]))]
    return r


OBS = dict(pose=np.array([0.0, 0.0, math.pi / 2]), tray_pos=np.array([0.0, 0.45, 0.85]), t=0.0)


def test_backend_stops_short_of_a_wall_and_reaches_a_free_spot():
    free = FP.flagship_backend(_result(None), OBS, log=lambda s: None)
    assert free["ok"] and not free["stopped_short"] and np.linalg.norm(free["spot_t"] - FP.NAIVE_T) < 0.02, free["spot_t"]
    blk = FP.flagship_backend(_result(wall_y=1.0), OBS, log=lambda s: None)
    # tray front edge is at y = 0.45 + 0.10 = 0.55; the wall face at 0.96; margin 0.03 -> the base may travel ~0.38 m
    assert blk["ok"] and blk["stopped_short"] and 0.30 < blk["spot_t"][1] < 0.405 - 0.01, blk["spot_t"]
    assert abs(blk["spot_t"][0]) < 0.02
    pt = np.asarray(blk["path_t"])
    assert pt[0].tolist() == [0.0, 0.0] and np.all(np.diff(pt[:, 1]) >= -1e-9)
    assert blk["backend"] == "zono_rows" and len(blk["guard_zonos"]) == 2
    print(f"backend: free spot {np.round(free['spot_t'], 3).tolist()} (100% path), walled spot {np.round(blk['spot_t'], 3).tolist()} ({100 * blk['progress_frac']:.0f}%), stopped_short")


def test_backend_hook_and_fallbacks():
    from world_model.v2.house.flagship import perception as PE
    msgs = []
    plan = PE._zono_rows_plan(_result(wall_y=1.0), OBS, msgs.append)             # the hook the flagship already calls: must find the function, not a module
    assert plan is not None and plan["backend"] == "zono_rows"
    assert FP.flagship_backend(_result(wall_y=1.0, with_payload=False), OBS, log=msgs.append) is None and "no `payload`" in msgs[-1]
    # a wall in the base's face: no progress possible -> ok False (the operator then falls back to naive, as for an infeasible zed plan)
    r = _result(wall_y=0.62)
    pl = FP.flagship_backend(r, OBS, log=lambda s: None)
    assert pl is None or not pl["ok"] or np.linalg.norm(pl["spot_t"]) < 0.2
    print("hook: _zono_rows_plan -> plan dict; no payload -> None (zed planner kept)")


def test_place_guard_on_the_flagship_sim():
    import mujoco
    from world_model.v2.house.flagship import scene as FS
    from world_model.v2.house.office import scene as OS
    wd = OS.office_meta()["waypoints"]["lab_desk_place"]
    sim = FS.new_sim(0, cafe_walker=False, spawn=(wd["xy"][0], wd["xy"][1] - 0.15, wd["heading"]), grasped=True, counter=False)
    mujoco.mj_forward(sim.m, sim.d)
    yaw = float(sim.base.yaw)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    tray = sim.d.xpos[sim.tray].copy()
    from world_model.v2.house.zono_rows.flagship_place import PlaceGuard
    out = {}
    for name, dist in (("far", 2.0), ("near", 0.12)):
        # a wall zonotope across the heading, `dist` m beyond the tray centre (tray half-length 0.15 along the heading is ~inside)
        c = np.array([tray[0] + fwd[0] * (0.15 + dist), tray[1] + fwd[1] * (0.15 + dist), tray[2]])
        G = np.stack([np.array([-fwd[1], fwd[0], 0.0]) * 2.0, np.array([fwd[0], fwd[1], 0.0]) * 0.04, [0, 0, 0.6]], 1)
        g = PlaceGuard(sim, [(c, G)], ["wall"])
        g.bf.u_prev = np.array([fwd[0] * 0.3, fwd[1] * 0.3, 0.0])      # moving at the path speed already
        u_b = g.filter(sim, np.array([0.3, 0.0, 0.0]), 0.02)
        out[name] = u_b
    print(f"PlaceGuard on the flagship sim: far wall -> body twist {np.round(out['far'], 3)}, wall {0.12} m beyond the tray -> {np.round(out['near'], 3)}")
    assert out["far"][0] > 0.25                                    # far wall: nominal speed barely touched (slew-limited only)
    assert out["near"][0] <= out["far"][0] - 0.009                 # near wall: decelerating (as fast as the 0.5 m/s^2 slew allows: 0.01 per 20 ms tick)
    assert np.isfinite(out["near"]).all()


if __name__ == "__main__":
    test_backend_stops_short_of_a_wall_and_reaches_a_free_spot()
    test_backend_hook_and_fallbacks()
    test_place_guard_on_the_flagship_sim()
    print("ALL PASSED")
