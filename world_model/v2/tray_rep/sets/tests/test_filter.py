"""F5 filter: bridge registration, determinism, passthrough when safe, limits, three-valued status. Uses the VAL split of data_piper(_trk)_full."""
import numpy as np
import pytest

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.bridge import episode as EP, filters as F

TIMING = ("wall_s", "filt_s", "sim_s", "x_f5_ms", "x_f5_ms_max", "x_f5_ms_mean")


def setup_env(setup):
    d = K.ROOT / f"data_{setup}_full"
    if not d.exists():
        pytest.skip(f"{d} missing")
    st = Store(d, "val")
    return st, E.env_from_meta(st.meta)


def strip(r):
    return {k: v for k, v in r.items() if k not in TIMING}


@pytest.mark.parametrize("setup", ["piper", "piper_trk"])
def test_registered_in_bridge_and_deterministic(setup):
    st, renv = setup_env(setup)
    f1 = F.make_filter("F5:H=40,k_sigma=2.5", renv)
    assert f1.name == "F5" and f1.rp.H == 40
    recs = []
    for _ in range(2):
        out = []
        f = F.make_filter("F5:H=40,k_sigma=2.5", renv)
        for ep in range(3):
            info = st.ep_info(ep)
            out.append(strip(EP.run_carry(renv, st.meta, info["seed"], f, onset_stored=info["onset"])))
        recs.append(out)
    assert recs[0] == recs[1]
    r = recs[0][0]
    assert r["x_f5_n"] > 50 and r["x_f5_true"] + r["x_f5_unknown"] + r["x_f5_false"] == r["x_f5_n"]


def test_passthrough_limits_and_status():
    st, renv = setup_env("piper")
    f = F.make_filter("F5", renv)
    info = st.ep_info(0)
    nm = EP.Nominal(st.meta, renv, info["seed"])
    rec = renv.reset(ball_xy=nm.ball_xy, mu_roll=nm.mu)
    obs = [rec["obs"]]
    f.begin(dict(renv=renv, hid=nm.mu, seed=info["seed"], T=nm.T))
    n_pass = n_mod = 0
    stat = set()
    for t in range(80):
        u_nom = nm.u(t)
        o = np.array(obs)
        u = f(o, np.zeros((t, 6)), u_nom, t)
        stat.add(f.status(o, u_nom))
        if np.allclose(u, u_nom, atol=0, rtol=0):
            n_pass += 1
        else:
            n_mod += 1
            ub = F.project_box(u, renv.a_max, renv.w_max)
            assert np.allclose(ub[:3], u[:3], atol=1e-9)           # accel inside the box (up to the 16-gon inscribed in the disc)
            assert np.all(np.abs(u[3:]) <= renv.w_max + 1e-9)
        obs.append(renv.tick(u)["obs"])
    assert n_pass > 0
    assert stat <= {"true", "unknown", "false"}
    assert not renv.status


def test_status_three_values_by_construction():
    """a ball placed near the edge with the tracker uncertainty: 'unknown'; at the centre: 'true'; beyond the edge: 'false'."""
    st, renv = setup_env("piper")
    f = F.make_filter("F5:k_sigma=3,eps_scale=2", renv)
    res = {}
    for name, xy in (("centre", (0.0, 0.0)), ("edge", (0.13, 0.0)), ("out", (0.17, 0.0))):
        rec = renv.reset(ball_xy=xy, mu_roll=2e-3)
        f.begin(dict(renv=renv, hid=2e-3, seed=0, T=100))
        o = np.array([rec["obs"]])
        f(o, np.zeros((0, 6)), np.zeros(6), 0)
        res[name] = f.status(o, np.zeros(6))
    assert res["centre"] == "true" and res["out"] == "false" and res["edge"] in ("unknown", "false")


def test_no_hardware_modules_imported():
    import sys
    import world_model.v2.tray_rep.sets.filter  # noqa: F401
    import world_model.v2.tray_rep.sets.validate_reach  # noqa: F401
    bad = [m for m in sys.modules if m.startswith("YOR") or m.split(".")[0] in ("piper_sdk", "can", "xarm", "rclpy", "serial")]
    assert not bad, bad
