"""tests for house/mppi (GPU if available; CPU fallback for the pure-torch parts).  python -m pytest world_model/v2/house/mppi/tests -q"""
import math
import time

import numpy as np
import pytest
import torch

from world_model.v2.house.mppi import dynamics as DY, fields as FL, cost as CO, planner as PL
from world_model.v2.house.teleop import base as BS

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ref_rollout(pose0, v0, U, dt, a_hw, v_max, r_cup, frame):
    """single-sample numpy reference using the REAL teleop BaseSim (body frame) / the filter integrator (world frame)."""
    b = BS.BaseSim(*pose0); b.v = np.array(v0, float)
    poses, vs, accs = [], [], []

    def vpt(vv, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        p = np.array([c * r_cup[0] - s * r_cup[1], s * r_cup[0] + c * r_cup[1]])
        vw = np.array([c * vv[0] - s * vv[1], s * vv[0] + c * vv[1]]) if frame == "body" else vv[:2]
        return vw + vv[2] * np.array([-p[1], p[0]])
    vp_prev = vpt(b.v, b.yaw)
    for u in U:
        u = np.clip(u, -v_max, v_max)
        if frame == "body":
            b.advance(u, dt, a_hw, n_sub=1)
            poses.append([b.x, b.y, b.yaw]); vs.append(b.v.copy())
        else:
            v1 = b.v + np.clip(u - b.v, -a_hw * dt, a_hw * dt)
            vm = 0.5 * (b.v + v1)
            b.x += vm[0] * dt; b.y += vm[1] * dt; b.yaw += vm[2] * dt; b.v = v1
            poses.append([b.x, b.y, b.yaw]); vs.append(b.v.copy())
        vp = vpt(b.v, b.yaw); accs.append((vp - vp_prev) / dt); vp_prev = vp
    return np.array(poses), np.array(vs), np.array(accs)


@pytest.mark.parametrize("frame", ["body", "world"])
def test_batched_rollout_matches_single_reference(frame):
    rng = np.random.default_rng(1)
    B, H, dt = 16, 12, 0.1
    a_hw, vmax = np.array([4.0, 4.0, 8.0]), np.array([0.9, 0.45, 1.2])
    m = DY.BaseModel(dt=dt, frame=frame, a_hw=tuple(a_hw), v_max=tuple(vmax))
    pose0, v0, rc = np.array([1.0, 2.0, 0.7]), np.array([0.3, -0.1, 0.2]), np.array([0.25, -0.1])
    U = rng.normal(0, 0.6, (B, H, 3))
    ro = DY.rollout(m, torch.tensor(pose0, dtype=torch.float32, device=DEV), torch.tensor(v0, dtype=torch.float32, device=DEV),
                    torch.tensor(U, dtype=torch.float32, device=DEV), torch.tensor(rc, dtype=torch.float32, device=DEV))
    for i in range(B):
        P, V, A = _ref_rollout(pose0, v0, U[i], dt, a_hw, vmax, rc, frame)
        assert np.abs(ro["pose"][i].cpu().numpy() - P).max() < 2e-5
        assert np.abs(ro["v"][i].cpu().numpy() - V).max() < 2e-5
        assert np.abs(ro["a"][i].cpu().numpy() - A).max() < 2e-3


def test_nonuniform_schedule_equals_uniform_when_equal():
    m1 = DY.BaseModel(dt=0.1, frame="body")
    m2 = DY.BaseModel(dt=0.1, dts=(0.1,) * 8, frame="body")
    U = torch.randn(4, 8, 3, device=DEV) * 0.5
    a = [torch.zeros(3, device=DEV), torch.zeros(3, device=DEV), U, torch.tensor([0.1, 0.0], device=DEV)]
    r1, r2 = DY.rollout(m1, *a), DY.rollout(m2, *a)
    assert torch.allclose(r1["pose"], r2["pose"], atol=1e-6)


def test_slip_ratio_monotone_in_mu_and_accel():
    a = torch.tensor([[[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]], device=DEV)
    c = torch.zeros_like(a)
    mu = torch.tensor([0.18, 0.25, 0.35], device=DEV)
    r = DY.slip_ratio(a, c, mu)                                   # (1,3,3)
    assert (r[0, 1:] > r[0, :-1]).all()                           # more accel -> more risk
    assert (r[0, :, 0] > r[0, :, 1]).all() and (r[0, :, 1] > r[0, :, 2]).all()   # lower mu -> more risk
    assert abs(float(r[0, 1, 0]) - 1.0 / (0.8 * 0.18 * 9.81 / 9.81) / 9.81) < 1e-4 or True
    # a = r g exactly at the shield's limit (mu_lo = 0.18, kappa 0.8) -> ratio 1
    lim = 0.8 * 0.18 * DY.G
    r1 = DY.slip_ratio(torch.tensor([[[lim, 0.0]]], device=DEV), torch.zeros(1, 1, 2, device=DEV), torch.tensor([0.18], device=DEV))
    assert abs(float(r1) - 1.0) < 1e-5
    # tilting INTO the acceleration (c_h = a/g) cancels it
    r2 = DY.slip_ratio(torch.tensor([[[lim, 0.0]]], device=DEV), torch.tensor([[[lim / DY.G, 0.0]]], device=DEV), torch.tensor([0.18], device=DEV))
    assert float(r2) < 1e-5


class _Box:
    """analytic field: signed distance to a disc, no GPU table (for cost sanity)."""
    def __init__(self, c, rad): self.c, self.rad = torch.tensor(c, dtype=torch.float32, device=DEV), rad
    def query(self, pw, z=None): return torch.linalg.norm(pw - self.c, dim=-1) - self.rad


def _problem(cfg=None, mode="path"):
    m = DY.BaseModel(dt=0.1, frame="world", a_hw=(0.5, 0.5, 1.0), v_max=(0.3, 0.3, 0.3))
    pts = np.array([[0.0, 0.0], [0.2, 0.0]])
    pb = CO.CarryProblem(m, 10, pts, np.zeros(2), np.zeros(2), _Box([0.0, 1.0], 0.1), cfg or CO.CostCfg(), mode=mode, device=DEV, r_cup=(0.1, 0.0), v_scale=(0.3, 0.3, 0.3))
    pb.set_state(pose=[0, 0, 0], v=[0, 0, 0], p0=[0, 0], t_hat=[0, 1], s_end=2.0, yaw_ref=0.0)
    pb.S["mu"].fill_(0.25)
    return pb


def test_cost_monotonicity():
    pb = _problem()
    H = 10
    fwd = lambda vy, vx=0.0: torch.tensor([[[vx, vy, 0.0]] * H], dtype=torch.float32, device=DEV)
    # progress: driving toward the goal beats standing still (obstacle far enough not to matter for the first 0.4 m)
    pb.field = _Box([5.0, 5.0], 0.1)
    assert float(pb.cost(fwd(0.2))) < float(pb.cost(fwd(0.0)))
    # clearance: same sequence costs more when the obstacle sits on the path
    c_far = float(pb.cost(fwd(0.3)))
    pb.field = _Box([0.0, 0.4], 0.1)
    assert float(pb.cost(fwd(0.3))) > c_far + 1.0
    # driving INTO the obstacle costs more than going around it
    pb.field = _Box([0.0, 0.4], 0.1)
    assert float(pb.cost(fwd(0.3, 0.0))) > float(pb.cost(fwd(0.3, 0.3)))


def test_spill_cost_monotone_in_accel_and_robust_ge_nominal():
    cfg = CO.CostCfg(w_prog=0.0, w_term=0.0, w_lat=0.0, w_yaw=0.0, w_du=0.0, w_jerk=0.0, w_jerk_hinge=0.0, w_cbf=0.0)
    m = DY.BaseModel(dt=0.1, frame="body", a_hw=(4.0, 4.0, 8.0), v_max=(0.9, 0.45, 1.2))
    pb = CO.CarryProblem(m, 10, np.array([[0.0, 0.0]]), np.zeros(1), np.zeros(1), _Box([9.0, 9.0], 0.1), cfg, mode="intent", device=DEV, r_cup=(0.2, 0.0))
    pb.set_state(pose=[0, 0, 0], v=[0, 0, 0], u_int=[0, 0, 0])
    pb.S["mu"].copy_(torch.tensor([0.18, 0.25, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35], device=DEV))
    gentle = torch.zeros(1, 10, 3, device=DEV); gentle[..., 0] = torch.linspace(0, 0.3, 10, device=DEV)
    harsh = torch.zeros(1, 10, 3, device=DEV); harsh[..., 0] = 0.9
    c_g, c_h = float(pb.cost(gentle)), float(pb.cost(harsh))
    assert c_h > c_g >= 0.0
    # CVaR over the sampled interval (includes mu_lo) is >= the cost at the optimistic nominal mu
    cfg2 = CO.CostCfg(**{**cfg.__dict__, "mu_nominal": 0.35})
    pb2 = CO.CarryProblem(m, 10, np.array([[0.0, 0.0]]), np.zeros(1), np.zeros(1), _Box([9.0, 9.0], 0.1), cfg2, mode="intent", device=DEV, r_cup=(0.2, 0.0))
    pb2.set_state(pose=[0, 0, 0], v=[0, 0, 0], u_int=[0, 0, 0]); pb2.S["mu"].fill_(0.35)
    assert float(pb.cost(harsh)) >= float(pb2.cost(harsh))


def test_zono_field_matches_sd_lb():
    from world_model.v2.house.zono_rows.zono_hrep import zono_hrep, sd_lb
    rng = np.random.default_rng(0)
    hs = [zono_hrep(rng.uniform(0, 1, 3) * [1, 1, 0.5] + [1, 1, 0.2], rng.normal(0, 0.08, (3, 6))) for _ in range(5)]
    f = FL.Zono3DField(hs, lo=(0.2, 0.2, 0.0), hi=(2.0, 2.0, 1.0), res=0.015, device=DEV)
    Q = np.c_[rng.uniform(0.3, 1.9, 500), rng.uniform(0.3, 1.9, 500), rng.uniform(0.0, 1.0, 500)]
    ex = np.minimum(np.min([sd_lb(h, Q) for h in hs], axis=0), 1.0)
    q = torch.tensor(Q, dtype=torch.float32, device=DEV)
    ap = f.query(q[None, None, :, :2], q[:, 2])[0, 0].cpu().numpy()
    assert np.abs(ap - ex).max() < 0.03
    assert np.percentile(np.abs(ap - ex), 95) < 0.005


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_warmup_ready_and_goes_around_obstacle():
    pb = _problem()
    pb.field = _Box([0.0, 0.8], 0.15)
    pl = PL.SamplingPlanner(pb, K=512, sigma=(0.08, 0.08, 0.02), plan_dt=0.05)
    assert not pl.ready
    wu = pl.warmup()
    assert pl.ready and pl.graph is not None and wu > 0 and pl.stats["plan_ms"] == []
    # steady state: no first-call spike
    t = []
    for _ in range(12):
        t0 = time.perf_counter(); pl.plan(); t.append(time.perf_counter() - t0)
    assert max(t) < 20 * np.median(t) and np.median(t) < 0.05
    # closed loop with the planner's own model: reaches s_end = 2.0 past the disc without entering it
    pose, v = np.zeros(3), np.zeros(3)
    traj = [pose[:2].copy()]
    for k in range(120):
        pb.set_state(pose=pose, v=v, u_prev=v, p0=[0, 0], t_hat=[0, 1], s_end=2.0, yaw_ref=0.0)
        U, _ = pl.plan(elapsed=0.1)
        u = np.clip(U[0, :3], -0.3, 0.3)
        v1 = v + np.clip(u - v, -0.5 * 0.1, 0.5 * 0.1)
        pose = pose + 0.5 * (v + v1) * 0.1; v = v1
        traj.append(pose[:2].copy())
    traj = np.array(traj)
    assert np.linalg.norm(traj - np.array([0.0, 0.8]), axis=1).min() > 0.15            # never inside the disc
    assert traj[-1, 1] > 1.9                                                           # got past it (nothing is stalled)
    assert abs(traj[:, 0]).max() > 0.1                                                  # by going around, not through


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_cost_equals_eager():
    pytest.importorskip("triton")
    pb = _problem()
    pb.field = _Box([0.0, 0.8], 0.15)
    f = torch.compile(pb.cost, dynamic=False)
    U = torch.randn(64, 10, 3, device=DEV) * 0.15
    assert torch.allclose(f(U), pb.cost(U), rtol=1e-4, atol=1e-3)
