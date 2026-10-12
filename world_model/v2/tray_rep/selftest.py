"""selftest.py -- runs every phase-1a smoke test quickly (CPU, <= 4 worker processes, a few minutes in total).
SIMULATION ONLY. Writes small outputs to /common/users/ss5772/bimanual_data/tray_rep/smoke/ and a summary JSON.

  python -m world_model.v2.tray_rep.selftest            (reuses finished shards; tests resume)
  python -m world_model.v2.tray_rep.selftest --fresh    (deletes the smoke dir first; the path is checked)

Checks (each prints PASS/FAIL and a number):
  env     bitwise determinism; bitwise snapshot/restore (same env and a fresh env); mu_roll hidden but effective;
          a_scale scales the set; obs[6:9] == act[3:6]; full[:9] == tray_only
  gen     40 episodes in both obs modes via the CLI: keys/shapes/dtypes, load_split compatibility (wm_audit.common),
          failure rate in 20-60%, resume skips finished shards, merged == shard concat, snapshot == logged state,
          regenerated episode == stored episode (bitwise)
  branch  50 branches under R and under Rc: shapes, logged prefix intact, root state == logged state, workers-independent,
          fraction of already-failed roots
  oracle  20 TEST states: label counts; sharded run + merge == single run
  cl_eval hold-still and oracle policy on 20 certified states: hold fails > 0, oracle fails ~ 0
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import gen_nominal as G
from world_model.v2.tray_rep import branches as BR
from world_model.v2.tray_rep import oracle as OR
from world_model.v2.tray_rep import cl_eval_mj as CL
from world_model.v2.tray_rep.store import Store
from world_model.v2.bead_rep import util as U

RES, FAILS = {}, []


def check(name, ok, info=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def gen_cli(out, obs_mode, n_eps, eps_per_shard, workers=4, extra=()):
    cmd = [sys.executable, "-m", "world_model.v2.tray_rep.gen_nominal", "--obs-mode", obs_mode, "--n-eps", str(n_eps),
           "--eps-per-shard", str(eps_per_shard), "--out", str(out), "--workers", str(workers), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(K.REPO))
    assert r.returncode == 0, r.stderr[-2000:]
    r2 = subprocess.run(cmd + ["--merge"], capture_output=True, text=True, cwd=str(K.REPO))
    assert r2.returncode == 0, r2.stderr[-2000:]
    return r.stdout, r2.stdout


# ====================================================================================================== env
def test_env():
    rng = np.random.default_rng(0)
    U85 = E.TrayRepEnv().U

    def run(renv, n, seed=0, snap_at=None):
        r = np.random.default_rng(seed)
        renv.reset(ball_xy=r.normal(0, 0.005, 2), mu_roll=0.002)
        traj, snap = [], None
        for k in range(n):
            if k == snap_at:
                snap = renv.snapshot()
            rec = renv.tick(U85[r.integers(85)] * r.uniform(0.2, 1.0))
            traj.append(np.concatenate([renv.env.d.qpos, renv.env.d.qvel, rec["obs"], rec["priv"], [rec["l"]]]))
        return np.array(traj), snap

    e1 = E.TrayRepEnv("full")
    a, snap = run(e1, 120, 1, snap_at=40)
    b, _ = run(E.TrayRepEnv("full"), 120, 1)
    check("env: determinism (two fresh envs, 120 ticks, bitwise)", np.array_equal(a, b))
    # restore into the same env, replay the same random actions from tick 40
    r = np.random.default_rng(1); r.normal(0, 0.005, 2)
    acts = [U85[r.integers(85)] * r.uniform(0.2, 1.0) for _ in range(120)]
    e1.restore(snap)
    c = []
    for k in range(40, 120):
        rec = e1.tick(acts[k])
        c.append(np.concatenate([e1.env.d.qpos, e1.env.d.qvel, rec["obs"], rec["priv"], [rec["l"]]]))
    check("env: snapshot/restore bitwise (same env, 80 ticks)", np.array_equal(np.array(c), a[40:]))
    e2 = E.TrayRepEnv("full"); e2.set_mu_roll(0.002)
    e2.restore(snap)
    c = []
    for k in range(40, 120):
        rec = e2.tick(acts[k])
        c.append(np.concatenate([e2.env.d.qpos, e2.env.d.qvel, rec["obs"], rec["priv"], [rec["l"]]]))
    check("env: snapshot/restore bitwise (fresh env, 80 ticks)", np.array_equal(np.array(c), a[40:]))
    # mu_roll: hidden but effective
    e3 = E.TrayRepEnv("full")
    for mu in (0.0005, 0.006):
        e3.reset(mu_roll=mu)
        for _ in range(30):
            e3.tick(np.array([0, 0, 0, 0, 0.5, 0.0]))        # tilt about y, ball starts to roll
        RES[f"loc_x_mu{mu}"] = float(e3.env.local()[0])
    check("env: mu_roll changes the ball trajectory", abs(RES["loc_x_mu0.0005"] - RES["loc_x_mu0.006"]) > 1e-3,
          f"x(0.0005)={RES['loc_x_mu0.0005']:.4f} x(0.006)={RES['loc_x_mu0.006']:.4f}")
    check("env: obs dims", E.TrayRepEnv("full").reset()["obs"].shape == (13,) and E.TrayRepEnv("tray_only").reset()["obs"].shape == (9,))
    es = E.TrayRepEnv("full", a_scale=0.5)
    check("env: a_scale scales the 85-set and the brake limits", np.allclose(es.U, 0.5 * U85) and es.a_max == 2.5 and es.w_max == 1.0)
    ef, et = E.TrayRepEnv("full"), E.TrayRepEnv("tray_only")
    ef.reset(); et.reset()
    ok = True
    for k in range(30):
        u = U85[k % 85]
        rf, rt = ef.tick(u), et.tick(u)
        ok &= np.array_equal(rf["obs"][:9], rt["obs"]) and np.allclose(rf["obs"][6:9], rf["act"][3:6])
    check("env: full[:9] == tray_only and obs[6:9] == act[3:6]", ok)
    RES["env_ms_per_tick"] = None
    t0 = time.perf_counter()
    for _ in range(300):
        ef.tick(U85[3])
    RES["env_ms_per_tick"] = 1e3 * (time.perf_counter() - t0) / 300


# ====================================================================================================== gen
def test_gen(smoke):
    from world_model.v2.wm_audit import common as C
    info = {}
    for mode in ("full", "tray_only"):
        out = smoke / f"data_{mode}"
        t0 = time.perf_counter()
        so, _ = gen_cli(out, mode, 40, 4, workers=4)
        wall = time.perf_counter() - t0
        info[mode] = dict(wall_s=wall)
        fails, n = 0, 0
        for sp in K.SPLITS:
            D = C.load_split(sp, data=out)                       # the pendulum pipeline's own loader
            od = E.OBS_DIMS[mode]
            N = int(D["lens"].sum())
            ok = (D["obs"].shape == (N, od) and D["act"].shape == (N, 6) and D["l"].shape == (N,) and D["n"].shape == (N, 3)
                  and D["nd"].shape == (N, 3) and len(D["start"]) == len(D["lens"]) == len(D["onset"]) == len(D["seed"])
                  and D["obs"].dtype == np.float32)
            check(f"gen[{mode}/{sp}]: wm_audit.common.load_split keys and shapes", ok, f"eps={len(D['lens'])} ticks={N}")
            fails += int((D["onset"] >= 0).sum()); n += len(D["lens"])
        info[mode]["fail_rate"] = fails / n
        info[mode]["episodes"] = n
        check(f"gen[{mode}]: nominal failure rate in 20-60%", 0.2 <= fails / n <= 0.6, f"{fails}/{n} = {fails / n:.2f}")
        f = np.load(out / "train.npz")
        check(f"gen[{mode}]: extra keys hid, priv", "hid" in f.files and f["priv"].shape[1] == E.PRIV_DIM
              and (E.MU_LO <= f["hid"]).all() and (f["hid"] <= E.MU_HI).all())
        check(f"gen[{mode}]: l in [-0.5, 1] and finite", f["l"].min() >= -0.5 and f["l"].max() <= 1.0 and np.isfinite(f["obs"]).all())
        # resume: rerunning must skip every shard
        r = subprocess.run([sys.executable, "-m", "world_model.v2.tray_rep.gen_nominal", "--obs-mode", mode, "--n-eps", "40",
                            "--eps-per-shard", "4", "--out", str(out), "--workers", "2"], capture_output=True, text=True, cwd=str(K.REPO))
        check(f"gen[{mode}]: resume skips finished shards", "0 new, 10 skipped" in r.stdout, r.stdout.strip().splitlines()[-1][:80])
    # merged == concat of shards; full[:, :9] == tray_only; same episodes
    a, b = np.load(smoke / "data_full" / "train.npz"), np.load(smoke / "data_tray_only" / "train.npz")
    check("gen: full[:, :9] == tray_only obs and identical act, l, hid, onset",
          np.array_equal(a["obs"][:, :9], b["obs"]) and np.array_equal(a["act"], b["act"]) and np.array_equal(a["l"], b["l"])
          and np.array_equal(a["hid"], b["hid"]) and np.array_equal(a["onset"], b["onset"]))
    # snapshot == logged state; regeneration == stored episode
    st = Store(smoke / "data_full", "train")
    renv = E.env_from_meta(st.meta)
    ok, worst = True, 0.0
    for ep in (0, 3, 7):
        L = st.lens[ep]
        for t in (20, 60, 100):
            if t >= L:
                continue
            snap, hid = st.snapshot_at(ep, t)
            renv.restore(snap, mu_roll=hid)
            rec = renv.current()
            lg, pv = st.rows("obs", ep, t, t + 1)[0], st.rows("priv", ep, t, t + 1)[0]
            err = max(np.abs(rec["obs"][:6] - lg[:6]).max(), np.abs(rec["priv"][:3] - pv[:3]).max())
            worst = max(worst, float(err))
    check("gen: restored snapshot == logged state (axis, v, ball position; float32)", worst < 1e-6, f"max err {worst:.1e}")
    info_ep = st.ep_info(2)
    e2 = G.run_episode(renv, info_ep["seed"], G.INTENSITY, with_snaps=False)
    same = (np.array_equal(e2["obs"], st.rows("obs", 2, 0, info_ep["lens"])) and np.array_equal(e2["act"], st.rows("act", 2, 0, info_ep["lens"])))
    check("gen: regenerated episode == stored episode (bitwise)", same)
    RES["gen"] = info


# ====================================================================================================== branches
def test_branches(smoke):
    D = smoke / "data_full"
    res = {}
    for law in ("R", "Rc"):
        out, meta = BR.make_branches_mj(D, "train", law, 50, seed=1, workers=4)
        U.atomic_savez(smoke / f"branches_{law}_smoke.npz", **out)
        U.jdump(meta, smoke / f"branches_{law}_smoke.json")
        N, od = 50, E.OBS_DIMS["full"]
        ok = (out["obs"].shape == (N, 160, od) and out["act"].shape == (N, 160, 6) and out["l"].shape == (N, 160)
              and out["feat"].shape == (N, 160, E.FEAT_DIM) and all(np.isfinite(out[k]).all() for k in ("obs", "act", "l", "feat")))
        check(f"branches[{law}]: shapes (50,160,13/6/-/10) and finite", ok)
        st = Store(D, "train")
        ok = True
        for i in range(0, N, 7):
            e, t0 = int(out["root_e"][i]), int(out["root_t"][i])
            ok &= np.array_equal(out["obs"][i, :60], st.rows("obs", e, t0 - 60, t0))
            ok &= np.array_equal(out["obs"][i, 60], st.rows("obs", e, t0, t0 + 1)[0])
            ok &= np.array_equal(out["l"][i, :60], st.rows("l", e, t0 - 60, t0)) and (t0 >= 60) and (t0 % 2 == 0)
        check(f"branches[{law}]: 60-tick logged prefix and root state equal the dataset", bool(ok))
        res[law] = {k: meta[k] for k in ("branch_fail_frac", "root_failed_frac", "tick_fail_frac", "fail_tick_median",
                                         "frac_safe_first25", "tunnel_frac", "wall_s")}
        print("   ", law, res[law], flush=True)
    check("branches: Rc roots are (almost) never already failed; R roots often are",
          res["Rc"]["root_failed_frac"] <= 0.1 and res["R"]["root_failed_frac"] > res["Rc"]["root_failed_frac"],
          f"R {res['R']['root_failed_frac']:.2f} Rc {res['Rc']['root_failed_frac']:.2f}")
    o1, _ = BR.make_branches_mj(D, "train", "Rc", 10, seed=3, workers=1)
    o2, _ = BR.make_branches_mj(D, "train", "Rc", 10, seed=3, workers=2)
    check("branches: result independent of the number of workers (bitwise)", all(np.array_equal(o1[k], o2[k]) for k in ("obs", "act", "l", "priv")))
    RES["branches"] = res


# ====================================================================================================== oracle + cl_eval
def test_oracle_cl(smoke):
    D = smoke / "data_full"
    out, meta = OR.run(D, "test", 20, seed=0, workers=4)
    U.atomic_savez(smoke / "oracle_smoke_20.npz", **out)
    U.jdump(meta, smoke / "oracle_smoke_20.json")
    check("oracle: 20 TEST states labelled", len(out["label"]) == 20 and set(np.unique(out["label"])) <= {0, 1, 2},
          f"hold_ok {meta['frac_hold_ok']:.2f} savable {meta['frac_savable']:.2f} uncertified {meta['frac_uncertified']:.2f} "
          f"{meta['mean_s_per_state']:.2f} s/state")
    p0, _ = OR.run(D, "test", 20, seed=0, workers=1, shard="0/2")
    p1, _ = OR.run(D, "test", 20, seed=0, workers=1, shard="1/2")
    lab = np.zeros(20, np.int8); lab[p0["idx"]] = p0["label"]; lab[p1["idx"]] = p1["label"]
    check("oracle: sharded runs (0/2 + 1/2) == single run", np.array_equal(lab, out["label"]))
    RES["oracle20"] = meta
    # a larger set for the closed-loop smoke: 100 episodes, full mode
    big = smoke / "data_full_100"
    gen_cli(big, "full", 100, 5, workers=4)
    ob, mb = OR.run(big, "test", 70, seed=0, workers=4)
    U.atomic_savez(smoke / "oracle_smoke_big.npz", **ob)
    sav = int((ob["label"] == OR.SAVABLE).sum())
    check("oracle: >= 20 certified-savable states in the 100-episode set", sav >= 20, f"{sav} of {len(ob['label'])}; "
          f"hold_ok {mb['frac_hold_ok']:.2f} savable {mb['frac_savable']:.2f} uncertified {mb['frac_uncertified']:.2f}")
    r_h, _ = CL.evaluate(CL.hold_still, big, ob, "test", T=50, max_states=20)
    r_o, _ = CL.evaluate(CL.OraclePolicy(), big, ob, "test", T=50, max_states=20)
    print("    hold:", r_h, "\n    oracle:", r_o, flush=True)
    check("cl_eval_mj: hold-still fails > 0 on certified states", r_h["fail_rate"] > 0, f"{r_h['fail_rate']:.2f} (n={r_h['n']})")
    check("cl_eval_mj: oracle policy fails ~ 0", r_o["fail_rate"] <= 0.05, f"{r_o['fail_rate']:.2f} (n={r_o['n']})")
    RES["cl_eval"] = dict(hold=r_h, oracle=r_o, big_oracle=mb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--only", default="env,gen,branch,oracle")
    a = ap.parse_args()
    smoke = K.SMOKE
    if a.fresh and smoke.exists():
        assert smoke.resolve() == Path("/common/users/ss5772/bimanual_data/tray_rep/smoke")
        shutil.rmtree(smoke)
    smoke.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    only = a.only.split(",")
    if "env" in only:
        test_env()
    if "gen" in only:
        test_gen(smoke)
    if "branch" in only:
        test_branches(smoke)
    if "oracle" in only:
        test_oracle_cl(smoke)
    RES["fails"] = FAILS
    RES["wall_s"] = time.perf_counter() - t0
    U.jdump(RES, smoke / "selftest_summary.json")
    print(f"[selftest] {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}  wall {RES['wall_s']:.0f}s", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
