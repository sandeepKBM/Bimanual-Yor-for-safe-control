"""zono_rows -- perceived zonotopes as HARD HOCBF rows inside the (unmodified) FL-MPC.  SIMULATION ONLY.

    zono_hrep.py   zonotope (c, G) -> H-representation, conservative signed-distance lower bound, exact distance (validation)
    rows.py        zono_barrier_rows(...): per keypoint x zonotope HOCBF rows in the table_rows torque convention
    ctl.py         ZonoCtl: phase2.Ctl2 + the rows injected through FLMPC.step(table_rows=...) each step
    zono_fast.py   numba kernels (candidate search with a grid-hash cull, row ranking + assembly, base-filter rows); identical rows, ~3x faster
    arm_rows.py    ArmZonoRows: the row side of ZonoCtl for the flagship's ArmCtl (teleop/arm.py ArmCtl.set_zono; flagship --arm-zono-rows)

No hardware, no ZED SDK, no ROS, no network.
"""


def flagship_backend(result, obs, **kw):
    """flagship `--place-backend zono_rows` hook (lazy import; see flagship_place.py).  Returns dict(ok, spot_t, path_t, ...) or None."""
    from .flagship_place import flagship_backend as _fb
    return _fb(result, obs, **kw)
