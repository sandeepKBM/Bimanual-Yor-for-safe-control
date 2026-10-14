"""EQUIVALENCE CHECK (mandatory): with zero uncertainty, margin(inside(ball, shrink(tray_top, 0))) == the hand-coded env margin l(x).

How the env defines l:  d = sdf_rect(ball centre in the tray frame)   (inside > 0; outside = -hypot of the overshoots),
                        l = clip(d, M_LO, M_HI) / M_HI.
The IR mapping: the ball entity is the POINT at the ball centre (the env drops the ball when its CENTRE leaves the tray), the tray_top
entity is the prism over |x| <= HX, |y| <= HY, and the ball is expressed in the tray frame through the two WORLD poses. Then
margin(inside(ball, tray_top)) = d (exact Euclidean outside, because the prism is box-like), and  l = l_from_margin(d).
The user's spec shrinks the tray top by r_ball: that is the stricter rule "the ball BODY stays on the tray". It equals the same IR
atom with a sphere ball and no shrink:  d_body = d - r_ball. That mapping is checked too.
"""
import numpy as np
import pytest

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import Store
from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets import tray_spec as TS
from world_model.v2.testbeds.b_tray_ball.tray_env import sdf_rect, HX, HY

DATA = K.ROOT / "data_piper_full"


def spec_inside(r_shrink=0.0, ball_shape="centre"):
    ents = TS.env_entities(ball_shape)
    d = {"inside": {"a": "ball", "b": {"shrink": {"of": "tray_top", "by": "r_shrink"}}, "margin": 0.0}}
    return ir.from_dict(d, ents, dict(r_shrink=r_shrink))


def test_synthetic_poses_float64_exact():
    rng = np.random.default_rng(0)
    spec = spec_inside()
    worst = 0.0
    for _ in range(2000):
        axis = rng.normal(size=3) * np.array([0.1, 0.1, 0.0])
        c = np.array([axis[0], axis[1], 1.0])
        c /= np.linalg.norm(c)
        # quaternion of the minimal rotation z -> c
        w = 1 + c[2]
        q = np.array([w, -c[1], c[0], 0.0])
        q /= np.linalg.norm(q)
        R = TS.quat_to_R(q)
        pos = rng.normal(size=3)
        loc = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.25, 0.25), 0.03])
        st = TS.EnvState(pos, q, pos + R @ loc)
        d_ir = ir.margin(spec, st)
        d_ref = sdf_rect(loc[0], loc[1], False)
        worst = max(worst, abs(d_ir - d_ref))
    assert worst < 1e-9, worst


def test_recorded_states_match_env_margin():
    """MuJoCo states restored from the data_piper_full TEST snapshots: IR margin vs env.margin() (float64, same state)."""
    if not DATA.exists():
        pytest.skip("data_piper_full missing")
    st = Store(DATA, "test")
    renv = E.env_from_meta(st.meta)
    spec = spec_inside()
    worst_l, worst_d, n, n_out, n_edge = 0.0, 0.0, 0, 0, 0
    worst_logged = 0.0
    for ep in range(0, min(st.n_eps, 60)):
        for t in st.grid_ticks(ep)[::9]:
            snap, hid = st.snapshot_at(ep, int(t))
            renv.restore(snap, mu_roll=hid)
            e = renv.env
            d_env = e.dist()
            l_env = e.margin()
            s_ir = TS.env_state_from(renv)
            d_ir = ir.margin(spec, s_ir)
            l_ir = float(TS.l_from_margin(d_ir))
            worst_d = max(worst_d, abs(d_ir - d_env))
            worst_l = max(worst_l, abs(l_ir - l_env))
            lg = float(st.rows("l", ep, int(t), int(t) + 1)[0])
            worst_logged = max(worst_logged, abs(l_ir - lg))
            n += 1
            n_out += l_env < 0
            n_edge += 0 <= d_env < 0.02
    print(f"\n[equiv] {n} recorded states ({n_out} outside the tray, {n_edge} within 2 cm of the edge): max |d_ir - d_env| = {worst_d:.2e}, "
          f"max |l_ir - l_env| = {worst_l:.2e}, max |l_ir - l_logged(float32)| = {worst_logged:.2e}")
    assert n > 300
    assert worst_d < 1e-9 and worst_l < 1e-9
    assert worst_logged < 2e-6                      # the stored l is float32


def test_body_vs_centre_mapping():
    """user spec: inside(ball, shrink(tray_top, r_ball)) with a point ball == inside(sphere ball, tray_top) shifted by r_ball."""
    rng = np.random.default_rng(1)
    r = 0.03
    a = spec_inside(r_shrink=r, ball_shape="centre")
    b = spec_inside(r_shrink=0.0, ball_shape="body")
    for _ in range(300):
        pos = rng.normal(size=3)
        q = np.array([1.0, 0, 0, 0])
        loc = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15), 0.03])
        st = TS.EnvState(pos, q, pos + loc)
        assert abs(ir.margin(a, st) - ir.margin(b, st)) < 1e-12 or (ir.margin(a, st) < 0 and ir.margin(b, st) < 0)
        # inside both exactly when the signed distance to the unshrunk rectangle is >= r
        d = sdf_rect(loc[0], loc[1], False)
        assert (ir.margin(a, st) >= 0) == (d >= r - 1e-15)
