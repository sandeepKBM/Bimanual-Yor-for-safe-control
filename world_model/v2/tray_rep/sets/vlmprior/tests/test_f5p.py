"""F5P in the bridge harness: registration, arm=unif/online=0 == plain F5, arm sets, determinism 1 vs 2 workers, no hardware modules."""
import json
import sys

import numpy as np
import pytest

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.bridge import episode as EP, filters as F, run as R
from world_model.v2.tray_rep.sets.vlmprior import conformal as CF

TIMING = ("wall_s", "filt_s", "sim_s", "x_f5_ms", "x_f5_ms_max", "x_f5_ms_mean", "x_p_ms", "x_p_ms_mean")


def setup_env(setup):
    d = K.ROOT / f"data_{setup}_full"
    if not d.exists():
        pytest.skip(f"{d} missing")
    st = Store(d, "val")
    return st, E.env_from_meta(st.meta)


def strip(r, keep_p=True):
    return {k: v for k, v in r.items() if k not in TIMING and (keep_p or not k.startswith("x_p_"))}


def run_n(spec, st, renv, n=3):
    f = F.make_filter(spec, renv)
    out = []
    for ep in range(n):
        info = st.ep_info(ep)
        out.append(EP.run_carry(renv, st.meta, info["seed"], f, onset_stored=info["onset"]))
    return out


@pytest.mark.parametrize("setup", ["piper", "piper_trk"])
def test_unif_uninformative_is_plain_f5(setup):
    st, renv = setup_env(setup)
    a = run_n("F5", st, renv, 4)
    b = run_n("F5P:arm=unif,online=0", st, renv, 4)
    assert [strip(x, False) for x in a] == [strip(x, False) for x in b]
    assert all(x["x_p_lo_end"] == 5e-4 and x["x_p_hi_end"] == 6e-3 for x in b)


def test_arm_sets_on_a_carry():
    st, renv = setup_env("piper_trk")
    for spec in ("F5P:arm=point,rho=0.2", "F5P:arm=band,rho=0.2", "F5P:arm=conf,rho=0.2", "F5P:arm=unif,rho=0.2", "F5P:arm=oracle"):
        r = run_n(spec, st, renv, 2)[1]
        mu = r["mu"]
        for c in ("t0", "t1s", "t2s", "end"):
            lo, hi = r[f"x_p_lo_{c}"], r[f"x_p_hi_{c}"]
            assert 1e-4 - 1e-12 <= lo <= hi <= 3e-2 + 1e-12
        if "oracle" in spec:
            assert r["x_p_lo_end"] == r["x_p_hi_end"] == pytest.approx(mu, rel=1e-9) and r["x_p_cov_end"]
        if "point" in spec:
            assert r["x_p_lo_end"] == r["x_p_hi_end"]
        if "band" in spec:
            assert r["x_p_hi_end"] > r["x_p_lo_end"]
        if "conf" in spec or "unif" in spec:                         # inside the population range (online 95% interval is)
            assert r["x_p_lo_end"] >= 5e-4 * (1 - 1e-9) and r["x_p_hi_end"] <= 6e-3 * (1 + 1e-9)
        assert r["x_p_m"] is not None and r["x_f5_n"] > 50


def test_prior_stats_match_generator_and_conf_q_is_loaded():
    st, renv = setup_env("piper_trk")
    r = run_n("F5P:arm=conf,rho=0.4,pseed=3", st, renv, 3)
    from world_model.v2.tray_rep.sets.vlmprior import prior as PR
    for x in r:
        m, sg, cf = PR.draw_prior(np.log(x["mu"]), x["seed"], 0.4, 3)
        assert x["x_p_m"] == m and x["x_p_sigma"] == sg and x["x_p_conf"] == cf
        assert x["x_p_q"] == CF.load_q(0.4)


def test_determinism_1_vs_2_workers(tmp_path):
    outs = []
    for w in (1, 2):
        R.run("F5P:arm=conf,rho=0.2,pseed=1", "piper_trk", 0, "val", 6, w, "t", tmp_path / f"w{w}")
        _, pj, _ = R.paths(tmp_path / f"w{w}", "t", "val", "piper_trk", "F5P:arm=conf,rho=0.2,pseed=1", 0)
        outs.append([strip(r) for r in sorted(R.read_jsonl(pj), key=lambda r: r["ep"])])
    assert len(outs[0]) == 6 and outs[0] == outs[1]


def test_no_hardware_modules_imported():
    bad = [m for m in sys.modules if m.startswith("YOR") or m.split(".")[0] in ("piper_sdk", "can", "xarm", "rclpy", "serial")]
    assert not bad, bad
