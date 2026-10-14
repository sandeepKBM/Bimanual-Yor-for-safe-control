"""online belief over ln(mu_roll): synthetic noise-free rollout, uninformative stream, cost, no hidden-mu access."""
import inspect
import time

import numpy as np

from world_model.v2.tray_rep.sets import reach as RC
from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM


def synth_stream(mu_true, T=150, amp=0.12, per=2.0, seed=0, ball0=(0.0, 0.0)):
    """obs rows (T+1, 13) of a noise-free carry on a sinusoidally tilted, non-translating tray; the ball follows reach.ball_step at mu_true."""
    s = np.array([ball0[0], ball0[1], 0.0, 0.0])
    me = RC.MU_EFF(mu_true)
    obs = np.zeros((T + 1, 13))
    a = np.zeros(3)

    def axis(t):
        th = amp * np.sin(2 * np.pi * t * 0.02 / per)
        ph = 0.6 * amp * np.sin(2 * np.pi * t * 0.02 / (1.3 * per) + 1.0)
        c = np.array([np.sin(th), np.sin(ph), 1.0])
        return c / np.linalg.norm(c)
    for t in range(T + 1):
        c = axis(t)
        obs[t, 0:3] = c
        obs[t, 9:11] = s[:2]
        if t < T:
            s, _, _ = RC.ball_step(s, c, axis(t + 1), a, me, RC.K_IN, RC.R_BALL)
    return obs


def test_concentrates_on_true_mu_noise_free():
    for mu_true in (7e-4, 1.5e-3, 2.5e-3):
        obs = synth_stream(mu_true)
        om = OM.OnlineMu(noise_free=True, lam=1.0)
        for row in obs:
            om.feed(row)
        m, sd = om.mean_sd_log()
        lt = np.log(mu_true)
        lo, hi = om.interval_log(0.9)
        assert abs(m - lt) < 0.25, (mu_true, m, lt)
        assert sd < 0.30                                   # prior sd is 0.717
        assert lo <= lt <= hi


def test_uninformative_stream_keeps_the_prior():
    obs = np.zeros((120, 13))
    obs[:, 2] = 1.0                                        # level tray, ball at rest: every node predicts "stays"
    om = OM.OnlineMu(noise_free=True, lam=1.0)
    for row in obs:
        om.feed(row)
    m, sd = om.mean_sd_log()
    assert abs(sd - 0.7173) < 0.01
    lo, hi = om.interval(0.9)
    assert abs(np.log(lo) - (np.log(5e-4) + 0.05 * np.log(12))) < 1e-9 and abs(np.log(hi) - (np.log(5e-4) + 0.95 * np.log(12))) < 1e-9


def test_tempering_widens_and_lam0_is_prior():
    obs = synth_stream(1.5e-3)
    sds = []
    for lam in (1.0, 0.3, 0.0):
        om = OM.OnlineMu(noise_free=True, lam=lam)
        for row in obs:
            om.feed(row)
        sds.append(om.mean_sd_log()[1])
    assert sds[0] < sds[1] < sds[2] and abs(sds[2] - 0.7173) < 0.01


def test_cost_per_tick_and_no_hidden_access():
    obs = synth_stream(1.5e-3, T=200)
    om = OM.OnlineMu(noise_free=True)
    om.feed(obs[0]); om.feed(obs[1])                       # compile
    om.reset()
    t0 = time.perf_counter()
    for row in obs:
        om.feed(row)
        om.mean_sd_log()
    ms = 1e3 * (time.perf_counter() - t0) / len(obs)
    assert ms < 1.0, ms
    import ast
    tree = ast.parse(inspect.getsource(OM))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {"hid", "mu_roll", "mu_true", "hidden"} & names                       # code only (the docstring mentions it)
    assert "ctx" not in inspect.signature(OM.OnlineMu.feed).parameters


def test_tracker_latency_and_hold_handled():
    """tracker-like stream: 3-tick delay, 30 Hz hold, noise 3 mm -> the belief still moves away from the prior toward the truth."""
    mu_true = 7e-4
    obs = synth_stream(mu_true, T=190)
    rng = np.random.default_rng(1)
    trk = obs.copy()
    held, acc = obs[0, 9:11].copy(), 0.0
    for t in range(len(obs)):
        acc += 30 / 50
        if acc >= 1 - 1e-9:
            acc -= 1
            held = obs[max(t - 3, 0), 9:11] + 0.003 * rng.normal(size=2)
        trk[t, 9:11] = held
    om = OM.OnlineMu(noise_free=False, lam=1.0)
    for row in trk:
        om.feed(row)
    m, sd = om.mean_sd_log()
    assert sd < 0.5 and abs(m - np.log(mu_true)) < 0.6
