"""stack/checks/base_motion: analytic cases with a fake affine backend (no sim)."""
import numpy as np
import pytest

from world_model.v2.nero.contents.objects import Contents
from world_model.v2.stack.checks import base_motion as BM
from world_model.v2.stack.checks.base_tip import Accel


class Fake:
    """one arm, 7 joints; tau_j = t0_j + g_j . a_b + ga_j alpha + gw_j w2, fz = f0 + 0.1 a_bx."""
    arms = ("left",)
    F_cap = 25.0

    def __init__(self):
        rng = np.random.default_rng(3)
        self.tau_lim = np.full(7, 10.0)
        self.t0, self.g, self.ga, self.gw = rng.normal(size=7), rng.normal(size=(7, 2)), rng.normal(size=7), rng.normal(size=7)

    def loads(self, vertex, a_b, alpha, w2):
        m = vertex[0]
        tau = m * self.t0 + self.g @ np.asarray(a_b, float) + self.ga * alpha + self.gw * w2
        return {"left": (tau, 5.0 * m + 0.1 * a_b[0])}

    def spread(self, vertex, alpha, w2):
        return {"left": np.full(7, 0.01 * (alpha + w2))}


C = Contents(1.0, 2.0, np.zeros((1, 3)))


def _u(env, m, a_b, al, w2):
    tau, fz = env.loads((m, np.zeros(3)), a_b, al, w2)["left"]
    sp = env.spread((m, None), abs(al), w2)["left"]
    return max(float(np.max((np.abs(tau) + sp) / env.tau_lim)), abs(fz) / env.F_cap)


def test_line_disc_worst_is_exact():
    env = Fake()
    a = 0.8
    r = BM.torque_row(env, C, {"line": Accel(a_lin=a)})
    th = np.linspace(0, 2 * np.pi, 20001)
    brute = max(_u(env, m, (a * np.cos(t), a * np.sin(t)), 0.0, 0.0) for m in (1.0, 2.0) for t in th[::40])
    assert 1.0 - r.margin == pytest.approx(brute, rel=2e-3) and 1.0 - r.margin >= brute - 1e-9     # closed form >= any sampled direction, equal to within the grid
    assert r.guarantee == "proof"


def test_turn_corners_exact_and_guarantee_sources():
    env = Fake()
    acc = Accel(alpha=1.5, w=2.0, source="measured", n=7)
    r = BM.torque_row(env, C, {"turn": acc})
    rng = np.random.default_rng(0)
    brute = max(_u(env, m, (0.0, 0.0), rng.uniform(-1.5, 1.5), rng.uniform(0, 4.0)) for m in (1.0, 2.0) for _ in range(500))
    assert 1.0 - r.margin >= brute - 1e-9
    assert r.guarantee == "sampled" and r.n == 7 and r.worst["kind"] == "turn"


def test_binding_and_refusal():
    env = Fake()
    env.F_cap = 5.0                                    # fz = 5 m / 5 = 2 at m = 2 -> u > 1
    r = BM.torque_row(env, C, {"line": Accel(a_lin=0.0)})
    assert not r.ok and r.worst["binding"] == "claw_Fz" and r.margin == pytest.approx(1.0 - 2.0)


def test_open_content_limits_and_turn_corner():
    mug = BM.MugGeom(D=0.08, H=0.095, fill_h_max=0.085, mu=0.5, zc_max=0.0475)
    lim = BM.ratio_limits(mug)
    assert lim["spill"] == pytest.approx(2 * 0.01 / 0.08) and lim["slide"] == 0.5 and lim["tip"] == pytest.approx(0.04 / 0.0475)
    corners = np.array([[0.3, 0.4], [-0.3, 0.4]])                       # |r| = 0.5
    acc = Accel(alpha=1.0, w=1.0)
    r = BM.open_content_base_row(mug, corners, {"turn": acc})
    ratio = 0.5 * np.hypot(1.0, 1.0) / BM.G
    assert r.worst["term"] == "spill" and r.worst["ratio"] == pytest.approx(ratio, abs=1e-4)
    assert r.margin == pytest.approx((lim["spill"] - ratio) * BM.G) and r.ok and r.guarantee == "proof"
    r2 = BM.open_content_base_row(mug, corners, {"line": Accel(a_lin=0.3 * BM.G, source="measured", n=3)})
    assert not r2.ok and r2.worst["term"] == "spill" and r2.guarantee == "sampled" and r2.n == 3
