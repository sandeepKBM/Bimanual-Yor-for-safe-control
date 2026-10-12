"""wm/selftest_1b.py -- fast checks of the phase-1b pieces (CPU, about 1-2 min). SIMULATION / SAVED DATA ONLY.
  python -m world_model.v2.tray_rep.wm.selftest_1b
Needs the smoke data <root>/smoke1b/data_piper_full (made by the smoke commands in the README; made here if missing).
Checks: clip carry == authority carry; slew env == authority env; realised library sequences are slew-idempotent; branch box
defaults; tracker (defaults bitwise, shapes, exact tray part, snapshot/restore); TrayDyn slew == env slew; policy
incremental filter == full re-filter; trainer checkpoint resume == straight run; G1 decision rule.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import gen_nominal as G
from world_model.v2.tray_rep import branches as BR
from world_model.v2.tray_rep import oracle as OR

FAILS = []
S1B = K.ROOT / "smoke1b"


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def ensure_data():
    d = S1B / "data_piper_full"
    if not (d / "test.npz").exists():
        base = [sys.executable, "-m", "world_model.v2.tray_rep.gen_nominal", "--config", "piper", "--obs-mode", "full",
                "--n-eps", "100", "--eps-per-shard", "5", "--out", str(d), "--workers", "4"]
        subprocess.run(base, check=True, cwd=str(K.REPO)); subprocess.run(base + ["--merge"], check=True, cwd=str(K.REPO))
    return d


def t_carry_and_slew():
    from world_model.v2.tray_rep.authority import scaled_probe as SP
    from world_model.v2.testbeds.b_tray_ball.carry import make_carry
    ok = True
    for seed in range(3):
        c1 = make_carry(np.random.default_rng(seed), 3.0); c2 = make_carry(np.random.default_rng(seed), 3.0)
        rng = np.random.default_rng(seed + 9)
        for t in range(150):
            c = rng.normal(size=3); c[2] = abs(c[2]) + 3; c /= np.linalg.norm(c)
            ok &= np.allclose(G.carry_u_clip(c1, t * 0.02, c, 0.3, 0.5), SP.carry_u_scaled(c2, t * 0.02, c, np.array([1.5, 1.5, 0.0]), 1.0))
    check("clip carry == authority carry_u_scaled", ok)
    cfg = dict(sx=.3, sy=.3, sw=.5, mode="clip", alpha=10.0)
    same = True
    for seed in (10000, 10001, 10002, 10003):
        info, _ = SP.run_carry_s((seed, 3.0, cfg, None))
        rng = np.random.default_rng(seed)
        r = E.TrayRepEnv("full", 0.3, False, 0.002, w_scale=0.5, alpha=10.0)
        r.reset(ball_xy=rng.normal(0, 0.005, 2)); cr = make_carry(rng, 3.0)
        for k in range(int(cr["T_end"] / 0.02)):
            r.tick(G.carry_u_clip(cr, k * 0.02, r.env.axis(), 0.3, 0.5))
            if r.env.status:
                break
        same &= (r.env.fail_tick // 4 if r.env.status == 1 else -1) == info["fail_tick"]
    check("env slew + clip carry: same failure tick as authority/scaled_probe (4 seeds)", bool(same))
    e = E.TrayRepEnv("full", 0.3, w_scale=0.5, alpha=10.0)
    check("piper env: U scaled (a x0.3, w x0.5), brake limits", np.allclose(e.U[:, :3], E.TrayRepEnv().U[:, :3] * 0.3)
          and np.allclose(e.U[:, 3:], E.TrayRepEnv().U[:, 3:] * 0.5) and e.a_max == 1.5 and e.w_max == 1.0)
    rng = np.random.default_rng(0)
    U = E.TrayRepEnv("full", 0.4).U
    A1 = BR.branch_actions(np.random.default_rng(5), 100, U, 0.4); A2 = BR.branch_actions(np.random.default_rng(5), 100, U, 0.4, w_scale=0.4)
    check("branch box: w_scale default == a_scale", np.array_equal(A1, A2))


def t_oracle_slew(d):
    from world_model.v2.tray_rep.store import Store
    from world_model.v2.testbeds.b_tray_ball.probe import run_seq
    st = Store(d, "test")
    renv = E.env_from_meta(st.meta)
    ep = int(np.flatnonzero(st.lens > 120)[0]); t = 80
    snap, hid = st.snapshot_at(ep, t)
    renv.set_mu_roll(hid)
    sn = snap.snap
    seqs = OR.best_sequence(sn.kin[:3].copy(), sn.kin[7:10].copy(), sn.kin[3:7].copy(), renv.U, 10, 100, renv.a_max, renv.w_max,
                            snap.w_slew, renv.alpha)
    ok = True
    for i in (0, 7, 30, 60, 85, 86):
        f1, *_ = run_seq(renv.env, sn, seqs[i])
        d1 = renv.env.d.qpos.copy()
        renv.restore(snap, hid)                        # the same sequence through the slewing env: idempotent
        for k in range(len(seqs[i])):
            renv.tick(seqs[i][k])
            if renv.env.status:
                break
        ok &= bool(np.allclose(d1, renv.env.d.qpos)) and (f1 == (renv.env.status == 1))
    check("realised library sequences give the same MuJoCo end state via the slewing env (idempotent)", ok)
    check("snapshot carries w_slew (nonzero somewhere on the grid)", any(np.abs(st.snapshot_at(ep, tt)[0].w_slew).max() > 0
                                                                         for tt in st.grid_ticks(ep, 30, 120)[:20]))


def t_tracker():
    def run(**kw):
        r = E.TrayRepEnv("full", 0.3, w_scale=0.5, alpha=10.0, **kw)
        r.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=1)
        out = []
        for k in range(60):
            out.append(r.tick(np.array([0.5 * np.sin(k / 8), 0.3, 0, 0.3 * np.cos(k / 9), 0.2, 0]))["obs"])
        return np.array(out)
    a, b = run(), run(obs_noise_m=0.0, obs_latency_ticks=0, obs_rate_hz=50.0, obs_dropout_p=0.0)
    check("tracker defaults reproduce obs bitwise", np.array_equal(a, b))
    c = run(obs_noise_m=0.003, obs_latency_ticks=3, obs_rate_hz=30.0)
    check("tracker: shape (13) and tray part exact", c.shape == a.shape and np.array_equal(c[:, :9], a[:, :9])
          and not np.allclose(c[:, 9:], a[:, 9:]))
    n0 = run(obs_noise_m=0.0, obs_latency_ticks=3, obs_rate_hz=50.0)
    check("tracker: latency 3 delays the ball position by 3 ticks (no noise, 50 Hz)", np.allclose(n0[3:, 9:11], a[:-3, 9:11], atol=1e-6))
    # snapshot / restore: deterministic tracker (no noise) must continue bitwise; noisy must repeat itself
    for kw, name in ((dict(obs_latency_ticks=3, obs_rate_hz=30.0), "deterministic"), (dict(obs_noise_m=0.003, obs_latency_ticks=3, obs_rate_hz=30.0, obs_dropout_p=0.1), "noisy")):
        r = E.TrayRepEnv("full", 0.3, w_scale=0.5, alpha=10.0, **kw)
        r.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=1)
        U = lambda k: np.array([0.5 * np.sin(k / 8), 0.3, 0, 0.3 * np.cos(k / 9), 0.2, 0])
        for k in range(30):
            r.tick(U(k))
        sn = r.snapshot()
        cont = [r.tick(U(30 + k))["obs"] for k in range(20)]
        runs = []
        for _ in range(2):
            r.restore(sn); runs.append([r.tick(U(30 + k))["obs"] for k in range(20)])
        ok = np.array_equal(runs[0], runs[1]) and (name != "deterministic" or np.array_equal(cont, runs[0]))
        check(f"tracker snapshot/restore ({name})", ok)


def t_dyn_policy_trainer(d):
    import torch
    from world_model.v2.tray_rep.wm import common as C, model as M, policy as PO, dyn as DY
    from world_model.v2.tray_rep.wm.trainer import TrayTrainer
    from world_model.v2.latent_hj import reach_value as rv
    torch.manual_seed(0)
    wm = M.build_wm(13); wm.eval()
    wm.o_mu.copy_(torch.randn(13) * 0.1); wm.o_sd.copy_(torch.rand(13) + 0.5)
    U = E.TrayRepEnv("full", 0.3, w_scale=0.5, alpha=10.0).U
    dyn = DY.TrayDyn(wm, None, "cpu", U, False, 10.0)      # projection off: random axes may trip it
    o_n = torch.randn(7, 13); u = torch.tensor(U[:7], dtype=torch.float32) * 2.0
    wp = o_n[:, 6:9] * wm.o_sd[6:9] + wm.o_mu[6:9]
    ue = dyn.effective_u(o_n, u)
    ref = wp + torch.clamp(u[:, 3:] - wp, -0.2, 0.2)
    check("TrayDyn slew == env slew rule (alpha*dt = 0.2 rad/s per tick)", torch.allclose(ue[:, 3:], ref, atol=1e-6) and torch.equal(ue[:, :3], u[:, :3]))
    net = rv.ValueNet(wm.d_h + 13, v_lo=-0.5).eval()
    pol = PO.WMGreedy(wm, net, "cpu", U, 10.0)
    obs = np.random.default_rng(1).normal(size=(40, 13)).astype(np.float32); act = np.random.default_rng(2).normal(size=(40, 6)).astype(np.float32)
    pol(obs[:30], act[:29]); u1 = pol(obs[:31], act[:30]); z_inc = pol.z.clone()
    pol2 = PO.WMGreedy(wm, net, "cpu", U, 10.0); u2 = pol2(obs[:31], act[:30])
    check("policy: incremental filter == full re-filter, same action", torch.allclose(z_inc, pol2.z, atol=1e-5) and np.array_equal(u1, u2))
    # trainer resume
    norm = dict(o_mu=[0.0] * 13, o_sd=[1.0] * 13, a_mu=[0.0] * 6, a_sd=[1.0] * 6)
    kw = dict(total_steps=8, dev="cpu", batch=6, n_os=16, K=3)
    A = TrayTrainer(3, norm, d, **kw); A.train(8, log_every=100)
    B = TrayTrainer(3, norm, d, **kw); B.train(4, log_every=100); B.save_ckpt(S1B / "tmp_ck.pt")
    Cc = TrayTrainer(3, norm, d, **kw); Cc.load_ckpt(S1B / "tmp_ck.pt"); Cc.train(4, log_every=100)
    same = all(torch.allclose(p, q, atol=1e-6) for p, q in zip(A.wm.parameters(), Cc.wm.parameters()))
    check("trainer: resume from checkpoint == straight run (8 steps, CPU)", same)
    (S1B / "tmp_ck.pt").unlink()
    check("trainer: aux head input = d_h + d_obs; norm width check", A.aux[0].in_features == A.wm.d_h + 13)


def t_g1():
    from world_model.v2.tray_rep.wm import g1
    check("G1 rule", g1.gate(400, 0.35, 300, 0.0)["PASS"] and not g1.gate(299, 0.35, 300, 0.0)["PASS"]
          and not g1.gate(400, 0.29, 300, 0.0)["PASS"] and not g1.gate(400, 0.35, 300, 0.03)["PASS"]
          and not g1.gate(400, None, 0, 0.0)["PASS"])


def main():
    d = ensure_data()
    t_carry_and_slew(); t_oracle_slew(d); t_tracker(); t_dyn_policy_trainer(d); t_g1()
    print("[selftest_1b] " + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
