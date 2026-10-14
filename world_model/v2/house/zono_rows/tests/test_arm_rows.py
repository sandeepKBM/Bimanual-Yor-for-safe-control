"""test for the flagship ARM zonotope rows (teleop/arm.py ArmCtl.set_zono + zono_rows/arm_rows.py).  SIMULATION only, CPU, no GPU.
    PYTHONPATH=. python -m world_model.v2.house.zono_rows.tests.test_arm_rows        (or pytest)
Builds the flagship TeleopSim (desk start, tray held; ~1 min).  Checks: default OFF is the old path (zono None, no table rows reach FL-MPC); a far zonotope gives
no rows and the SAME torque bit-for-bit; a wall zonotope in front of the tray gives rows -> `qp_stats()['n_calls_with_table_rows'] > 0` (the metric the flagship
JSON reports as flmpc_qp.n_calls_with_table_rows) and the returned torque satisfies them; ticks through TeleopSim.tick work."""
from __future__ import annotations

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_FLMPC_IMPL", "rt")

import numpy as np
import mujoco


def _wall(sim, dist):
    yaw = float(sim.base.yaw)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    tray = sim.d.xpos[sim.tray].copy()
    c = np.array([tray[0] + fwd[0] * (0.15 + dist), tray[1] + fwd[1] * (0.15 + dist), tray[2]])
    G = np.stack([np.array([-fwd[1], fwd[0], 0.0]) * 2.0, np.array([fwd[0], fwd[1], 0.0]) * 0.04, [0, 0, 0.6]], 1)
    return (c, G)


def test_arm_zono_rows_reach_flmpc():
    from world_model.v2.house.flagship import scene as FS
    from world_model.v2.house.office import scene as OS
    from world_model.v2.house.teleop.sim import Cmd
    wd = OS.office_meta()["waypoints"]["lab_desk_place"]
    sim = FS.new_sim(0, cafe_walker=False, spawn=(wd["xy"][0], wd["xy"][1] - 0.15, wd["heading"]), grasped=True, counter=False)
    ctl, d, fl = sim.ctl, sim.d, sim.ctl.fl
    mujoco.mj_forward(sim.m, d)
    assert ctl.zono is None
    snap = (fl.a_ego_prev.copy(), fl.a_int_prev.copy())

    def restore():
        fl.a_ego_prev[:], fl.a_int_prev[:] = snap

    te0, ti0 = ctl.tau(d)                                           # OFF: the original call (no table_rows)
    assert ctl.qp_stats()["n_calls_with_table_rows"] == 0

    restore()
    ctl.set_zono([_wall(sim, 3.0)], ["far"])                          # ON, nothing within d_act: empty rows -> identical torque
    te1, ti1 = ctl.tau(d)
    assert np.array_equal(te0, te1) and np.array_equal(ti0, ti1)
    assert ctl.zono.n_calls == 1 and ctl.zono.n_calls_with_rows == 0 and ctl.qp_stats()["n_calls_with_table_rows"] == 0

    restore()
    ctl.set_zono([_wall(sim, 0.12)], ["wall"])                        # ON, wall ~8 cm in front of the tray rim
    te2, ti2 = ctl.tau(d)
    q = ctl.qp_stats()
    assert ctl.zono.n_calls_with_rows == 1 and q["n_calls_with_table_rows"] >= 1, (ctl.zono.summary(), q)
    ae, be, ai, bi = ctl.zono.rows(d)                                 # same state again: rows the controller was given
    assert len(be) + len(bi) > 0
    viol = max(float(np.max(ae @ te2 - be, initial=-np.inf)), float(np.max(ai @ ti2 - bi, initial=-np.inf)))
    assert viol <= 1e-3 or not ctl.last_ok, viol
    print(f"ArmCtl.set_zono: far wall -> 0 rows + identical torque; near wall -> {len(be)}+{len(bi)} rows, qp_stats n_calls_with_table_rows={q['n_calls_with_table_rows']}, "
          f"max row violation of the returned torque {viol:.2e}, zono.h_min={ctl.zono.h_min:.3f}")

    restore()
    for _ in range(3):                                                # through TeleopSim.tick (substeps; the mocap-pose base twist estimate)
        o = sim.tick(Cmd())
    assert np.isfinite(o["pose"]).all() and ctl.zono.n_calls > 4
    ctl.clear_zono()
    assert ctl.zono is None


if __name__ == "__main__":
    test_arm_zono_rows_reach_flmpc()
    print("ALL PASSED")
