"""selftest_pc.py -- smoke tests for the point-cloud perception layer. SIMULATION ONLY, CPU. Writes smoke/selftest_pc_summary.json.

  python -m world_model.v2.tray_rep.perception.selftest_pc                 (about 1-2 min)
  python -m world_model.v2.tray_rep.perception.selftest_pc --existing      (also runs tray_rep.selftest --fresh and wm.selftest_1b)

Checks
  ray_analytic     noise-free ray points lie on the sphere / tray plane (analytic distances) to < 1e-6 m
  noise_stat       z-depth noise of tray hits matches the stereo law (std ratio 0.9-1.1, outliers excluded)
  grazing_mask     no returned hit has incidence above the limit; the mask removes points when tightened
  est_noise_free   sphere fit recovers the centre to < 1 mm; the visible-point centroid is biased (>5 mm) and the fit is not
  est_robust       a few gross outliers do not move the fit more than a few mm
  determinism      same seed + same actions -> identical obs; a different seed changes the noisy part only
  snapshot         restore mid-episode (sensor queue + Kalman state included) reproduces the rest of the episode bitwise
  snapshot_cloud   same, with the compact cloud packed
  replay_equal     live 30 Hz / 3-tick env == offline replay of the 50 Hz payload log
  parent_unchanged PCRepEnv physics, act, l, priv and the first 9 obs columns == TrayRepEnv 'full'; OBS_DIMS untouched
  dataset_equal    (if data_piper_full exists) episodes equal the stored data_piper_full episodes (first 9 obs cols, act, l, priv)
  cloud            compact cloud shapes, mask, ball points present
  gen_smoke        gen_pc writes shards for all 3 splits (2 eps each), merge works, store.load_split reads them
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import gen_nominal as GN
from world_model.v2.tray_rep import store as ST
from world_model.v2.tray_rep.perception import depth_sim as D
from world_model.v2.tray_rep.perception import ball_estimator as BE
from world_model.v2.tray_rep.perception import pc_env as PC
from world_model.v2.tray_rep.perception import gen_pc as GP
from world_model.v2.testbeds.b_tray_ball import tray_env as TE

OUT = K.ROOT / "perception" / "smoke"
PY = sys.executable
RES = {}


def check(name, ok, **info):
    RES[name] = dict(ok=bool(ok), **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in info.items()})
    print(f"[selftest_pc] {'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    return ok


def clean_spec(name, **kw):
    return replace(D.CAMERAS[name], sigma_floor=0.0, sigma_disp_px=0.0, p_drop=0.0, p_out=0.0, quant_m=0.0, cos_min=0.0, **kw)


def toy_scene(ball_local=(0.02, -0.03), tilt=0.0):
    sc = D.Scene()
    q = np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0])
    R = D.quat_to_R(q)
    tp = np.array([0.1, 0.05, 1.0])
    bw = tp + R @ np.array([ball_local[0], ball_local[1], 0.03])
    sc.set_pose(tp, q, bw)
    return sc, tp, q, R, bw


def t_ray_analytic():
    errs_s, errs_p, n_s, n_p = [], [], 0, 0
    for tilt in (0.0, 0.2):
        sc, tp, q, R, bw = toy_scene(tilt=tilt)
        for name in ("head_aim", "wrist_l", "head_low"):
            cam = D.DepthCamera(clean_spec(name))
            fr = cam.capture(sc, tp, R, 0, 1, use_roi=False)
            if len(fr.pts_w) == 0:
                continue
            o = fr.origin
            u = fr.dir_w
            # analytic ray length to the sphere / to the tray top plane for each returned ray
            for i in range(len(fr.pts_w)):
                p = fr.pts_w[i]
                if fr.geom[i] == sc.gid_ball:
                    oc = o - bw
                    b = oc @ u[i]
                    disc = b * b - (oc @ oc - 0.03 ** 2)
                    t = -b - np.sqrt(max(disc, 0.0))
                    errs_s.append(abs(np.linalg.norm(p - o) - t)); n_s += 1
                elif fr.geom[i] == sc.gid_tray[0]:
                    nrm = R[:, 2]
                    den = nrm @ u[i]
                    if abs(den) > 1e-9:
                        t = (nrm @ (tp - o)) / den
                        pt_loc = (p - tp) @ R
                        if abs(pt_loc[2]) < 1e-6:                      # hit on the top face
                            errs_p.append(abs(np.linalg.norm(p - o) - t)); n_p += 1
    ok = n_s > 50 and n_p > 50 and max(errs_s) < 1e-5 and max(errs_p) < 1e-6
    return check("ray_analytic", ok, sphere_pts=n_s, plane_pts=n_p, max_err_sphere_m=max(errs_s), max_err_plane_m=max(errs_p))


def t_noise_stat():
    sc, tp, q, R, bw = toy_scene()
    sc.set_pose(tp, q, tp + [0, 0, 5.0])                                       # ball far above the tray: tray only
    spec = replace(D.CAMERAS["head_aim"], cos_min=0.0, p_drop=0.0, p_out=0.0, quant_m=0.0, W=320, H=180)
    cam = D.DepthCamera(spec)
    clean = D.DepthCamera(replace(spec, sigma_floor=0.0, sigma_disp_px=0.0))
    res = []
    for k in range(8):
        a = cam.capture(sc, tp, R, k, 11, use_roi=False)
        b = clean.capture(sc, tp, R, k, 11, use_roi=False)
        tray = a.geom == sc.gid_tray[0]
        if tray.sum() < 20:
            continue
        # same pixel sets (same mask, no dropout): compare z-depth directly
        m = min(len(a.z_depth), len(b.z_depth))
        if len(a.z_depth) == len(b.z_depth):
            dz = (a.z_depth - b.z_depth)
            sg = spec.sigma_z(b.z_depth)
            res.append(dz / sg)
    r = np.concatenate(res) if res else np.zeros(0)
    ok = len(r) > 500 and 0.9 < r.std() < 1.1 and abs(r.mean()) < 0.1
    return check("noise_stat", ok, n=len(r), z_std_ratio=r.std() if len(r) else -1, sigma_z_at_0p7m_mm=1e3 * float(D.CAMERAS["head"].sigma_z(0.7)),
                 sigma_z_at_1m_mm=1e3 * float(D.CAMERAS["head"].sigma_z(1.0)))


def t_grazing():
    sc, tp, q, R, bw = toy_scene()
    base = replace(D.CAMERAS["head_aim"], p_drop=0.0, p_out=0.0, sigma_floor=0.0, sigma_disp_px=0.0, quant_m=0.0, W=240, H=135)
    n = {}
    for lim in (0.0, float(np.cos(np.deg2rad(80))), float(np.cos(np.deg2rad(60)))):
        fr = D.DepthCamera(replace(base, cos_min=lim)).capture(sc, tp, R, 0, 1, use_roi=False)
        n[lim] = len(fr.pts_w)
        if lim > 0:                                                   # check incidence of every returned hit
            inc = []
            for i in range(len(fr.pts_w)):
                if fr.geom[i] == sc.gid_tray[0]:
                    inc.append(abs((R[:, 2]) @ fr.dir_w[i]) if abs(((fr.pts_w[i] - tp) @ R)[2]) < 1e-6 else 1.0)
                elif fr.geom[i] == sc.gid_ball:
                    nn = fr.pts_w[i] - bw; nn /= np.linalg.norm(nn)
                    inc.append(abs(nn @ fr.dir_w[i]))
            if len(inc) and min(inc) < lim - 1e-6:
                return check("grazing_mask", False, min_cos=min(inc), limit=lim)
    ks = sorted(n)
    ok = n[ks[0]] > n[ks[1]] > n[ks[2]]
    return check("grazing_mask", ok, hits_no_mask=n[ks[0]], hits_80deg=n[ks[1]], hits_60deg=n[ks[2]])


def t_est_noise_free():
    worst, cent_bias, n_tested, nmin = 0.0, [], 0, 10 ** 9
    cfg = BE.EstCfg()
    for name in ("head_aim", "head_low", "wrist_l"):
        spec = clean_spec(name)
        cam = D.DepthCamera(spec)
        for off in [(0, 0), (0.05, 0.02), (-0.08, 0.04), (0.1, -0.06), (-0.12, -0.07), (0.0, 0.08)]:
            sc, tp, q, R, bw = toy_scene(ball_local=off, tilt=0.05)
            sc.set_pose(tp, q, bw)
            fr = cam.capture(sc, tp, R, 0, 1, use_roi=False)
            pay = BE.measure([fr], cfg, [spec])
            nb = int((fr.geom == sc.gid_ball).sum())
            if pay[BE.I_VALID] < 0.5:
                continue
            c_true = np.array([off[0], off[1], 0.03])
            worst = max(worst, float(np.linalg.norm(pay[1:4] - c_true)))
            nmin = min(nmin, int(pay[BE.I_N]))
            n_tested += 1
            pts_t, org = fr.to_tray()
            bp = pts_t[fr.geom == sc.gid_ball]
            if len(bp) > 8:
                cent_bias.append(float(np.linalg.norm(bp.mean(0) - c_true)))
    ok = n_tested >= 12 and worst < 1e-3 and np.mean(cent_bias) > 5e-3
    return check("est_noise_free", ok, tested=n_tested, worst_centre_err_mm=1e3 * worst, mean_centroid_bias_mm=1e3 * float(np.mean(cent_bias)),
                 min_points=nmin)


def t_est_robust():
    spec = replace(D.CAMERAS["head_aim"], p_out=0.0, p_drop=0.0)
    cam = D.DepthCamera(spec)
    sc, tp, q, R, bw = toy_scene(ball_local=(0.03, 0.02))
    errs = []
    rng = np.random.default_rng(3)
    for k in range(30):
        fr = cam.capture(sc, tp, R, k, 5, use_roi=False)
        # add 3 gross outliers (above the tray, near the ball) to the cloud
        ball_pts = fr.pts_w[fr.geom == sc.gid_ball]
        if len(ball_pts) < 15:
            continue
        add = bw + rng.uniform(-0.05, 0.05, (3, 3)) + np.array([0, 0, 0.03])
        fr2 = D.Frame(np.vstack([fr.pts_w, add]), np.concatenate([fr.geom, -np.ones(3, np.int32)]), fr.origin, fr.cam_R, fr.tray_pos,
                      fr.tray_R, 0, fr.n_rays, fr.cam, np.concatenate([fr.z_depth, np.full(3, 0.7)]),
                      np.vstack([fr.dir_w, np.tile(fr.dir_w[0], (3, 1))]))
        pay = BE.measure([fr2], BE.EstCfg(), [spec])
        pay0 = BE.measure([fr], BE.EstCfg(), [spec])
        if pay[0] > 0.5 and pay0[0] > 0.5:
            errs.append(np.linalg.norm(pay[1:3] - pay0[1:3]))
    ok = len(errs) >= 15 and np.max(errs) < 0.006
    return check("est_robust", ok, n=len(errs), max_shift_mm=1e3 * float(np.max(errs)) if errs else -1)


def mk(cam="head_aim", **kw):
    cfg = GN.load_config("piper")
    return PC.PCRepEnv(cam, cfg["a_scale"], False, w_scale=cfg["w_scale"], alpha=cfg["alpha"], **kw)


def actions(n, seed=0):
    r = np.random.default_rng(seed)
    U = np.array(E.B.fallback_U()) * np.array([0.3] * 3 + [0.5] * 3)
    return U[r.integers(0, len(U), n)]


def run(renv, acts, seed, snap_at=None):
    rec = renv.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=seed)
    obs = [rec["obs"]]
    snap = None
    for t, u in enumerate(acts):
        if snap_at is not None and t == snap_at:
            snap = renv.snapshot()
        rec = renv.tick(u)
        obs.append(rec["obs"])
    return np.array(obs), snap


def t_determinism():
    acts = actions(60, 1)
    a, _ = run(mk(), acts, 5)
    b, _ = run(mk(), acts, 5)
    c, _ = run(mk(), acts, 6)
    same = np.array_equal(a, b)
    exact_same = np.array_equal(a[:, :9], c[:, :9])
    noisy_diff = not np.array_equal(a[:, 9:], c[:, 9:])
    return check("determinism", same and exact_same and noisy_diff, bitwise_same=same, first9_equal_other_seed=exact_same,
                 noisy_part_differs_other_seed=noisy_diff)


def t_snapshot(with_cloud):
    acts = actions(70, 2)
    kw = dict(want_cloud=True, pack_cloud=True) if with_cloud else {}
    e1 = mk(**kw)
    full, snap = run(e1, acts, 9, snap_at=33)
    e2 = mk(**kw)
    e2.restore(snap, mu_roll=0.002)
    obs2 = []
    clouds = []
    for t in range(33, len(acts)):
        rec = e2.tick(acts[t])
        obs2.append(rec["obs"])
        clouds.append(e2.cloud()[0].copy())
    # reference clouds
    e3 = mk(**kw)
    rec = e3.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=9)
    ref_c = []
    for t, u in enumerate(acts):
        rec = e3.tick(u)
        if t >= 33:
            ref_c.append(e3.cloud()[0].copy())
    ok = np.array_equal(full[34:], np.array(obs2))
    if with_cloud:
        ok = ok and np.array_equal(np.array(clouds), np.array(ref_c))
    # a snapshot through the arrays() / from_arrays() round trip (what the dataset stores)
    a = snap.arrays()
    s2 = E.RepSnap.from_arrays(a["state"], a["prev"], a["has_prev"], a["kin"], a["status"], a["k"], a["w_slew"], a["trk"])
    e4 = mk(**kw)
    e4.restore(s2, mu_roll=0.002)
    o4 = [e4.tick(acts[t])["obs"] for t in range(33, len(acts))]
    ok4 = np.array_equal(full[34:], np.array(o4))
    return check("snapshot_cloud" if with_cloud else "snapshot", ok and ok4, trk_len=len(snap.trk), roundtrip_equal=ok4)


def t_replay_equal():
    acts = actions(90, 3)
    live, _ = run(mk(rate_hz=30.0, latency_ticks=3), acts, 4)
    e = mk(rate_hz=50.0, latency_ticks=0, log_meas=True)
    run(e, acts, 4)
    T = len(live)
    pay = np.zeros((T, BE.PAYLOAD_LEN))
    for (t, p, nb, nr) in e.log:
        if t < T:
            pay[t] = p
    est, qual, since, nl = PC.replay(pay, 30.0, 3)
    d = np.abs(live[:, 9:13] - est).max()
    dq = np.abs(live[:, 13:] - qual).max()
    return check("replay_equal", d < 1e-5 and dq < 1e-5, max_est_diff=float(d), max_qual_diff=float(dq))


def t_parent_unchanged():
    acts = actions(80, 4)
    cfg = GN.load_config("piper")
    ref = E.TrayRepEnv("full", cfg["a_scale"], False, w_scale=cfg["w_scale"], alpha=cfg["alpha"])
    pc = mk()
    ra = ref.reset(ball_xy=(0.004, -0.003), mu_roll=0.002)
    rb = pc.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=1)
    ok, md = True, 0.0
    for u in acts:
        ra, rb = ref.tick(u), pc.tick(u)
        same = (np.array_equal(ra["obs"][:9], rb["obs"][:9]) and np.array_equal(ra["priv"], rb["priv"])
                and np.array_equal(ra["act"], rb["act"]) and ra["l"] == rb["l"])
        ok &= same
    dims_ok = dict(E.OBS_DIMS) == {"full": 13, "tray_only": 9}
    # the parent in 'full' mode is still privileged and bitwise: ball cols equal priv
    pr_ok = np.allclose(ra["obs"][9:13], ra["priv"][[0, 1, 3, 4]], atol=1e-6)
    return check("parent_unchanged", ok and dims_ok and pr_ok and rb["obs"].shape == (16,), first9_priv_act_l_equal=ok, obs_dims_untouched=dims_ok,
                 parent_full_still_privileged=pr_ok, pc_obs_dim=int(rb["obs"].shape[0]))


def t_dataset_equal():
    root = K.ROOT / "data_piper_full"
    f = root / "train" / "shard_0000.npz"
    if not f.exists():
        return check("dataset_equal", True, skipped="data_piper_full/train/shard_0000.npz not found")
    z = np.load(f)
    cfg = GN.load_config("piper")
    renv = mk()
    s = 0
    ok = True
    for j in range(2):
        L = int(z["lens"][j])
        ep = GN.run_episode(renv, GN.SEED0 + j, cfg["intensity"], with_snaps=False, carry_mode="clip")
        ok &= ep["obs"].shape[0] == L
        ok &= np.array_equal(ep["obs"][:, :9], z["obs"][s:s + L, :9])
        ok &= np.array_equal(ep["act"], z["act"][s:s + L]) and np.array_equal(ep["l"], z["l"][s:s + L])
        ok &= np.array_equal(ep["priv"], z["priv"][s:s + L]) and ep["onset"] == int(z["onset"][j]) and ep["hid"] == float(z["hid"][j])
        s += L
    return check("dataset_equal", ok, episodes_compared=2, source="data_piper_full/train/shard_0000.npz")


def t_cloud():
    acts = actions(60, 5)
    e = mk(want_cloud=True)
    rec = e.reset(ball_xy=(0.004, -0.003), mu_roll=0.002, track_seed=2)
    near, tot, nm = 0, 0, []
    for u in acts:
        rec = e.tick(u)
        pts, mask = e.cloud()
        nm.append(mask.sum())
        if mask.sum():
            loc = e.env.local()
            d = np.linalg.norm(pts[mask] - loc, axis=1)
            near += int((d < 0.045).sum()); tot += int(mask.sum())
    ok = pts.shape == (128, 3) and mask.dtype == bool and tot > 0 and near / tot > 0.8 and max(nm) <= 128
    return check("cloud", ok, K=128, mean_points=float(np.mean(nm)), frac_within_4p5cm_of_ball=near / max(tot, 1))


def t_gen_smoke():
    out = OUT / "gen_smoke"
    import shutil
    if out.exists():
        shutil.rmtree(out)
    cfg = GN.load_config("piper")
    sens = dict(rate_hz=None, latency_ticks=None, sigma_a=1.5, cam_overrides=None)
    n_eps, eps = 10, 2
    t0 = time.perf_counter()
    for g in range(5):
        r = GP.process_shard((g, str(out), "head_aim", cfg["a_scale"], False, cfg["intensity"], eps, GN.SEED0, 2, 20,
                              cfg["w_scale"], cfg["alpha"], "clip", sens, g == 0, True))
        assert not r["skipped"]
    wall = time.perf_counter() - t0
    for sp in K.SPLITS:
        GN.merge_split(out, sp, True, None)
    import json as _j
    (out / "meta.json").write_text(_j.dumps(dict(obs_mode="pc_est", obs_dim=16, snap_stride=2, snap_min_tick=20)))
    D3 = ST.load_split("train", out)
    ok = D3["obs"].shape[1] == 16 and len(D3["lens"]) == 6 and np.isfinite(D3["obs"]).all()
    cl = np.load(out / "train" / "cloud_0000.npz")
    ok &= cl["pts"].shape[0] == int(np.load(out / "train" / "shard_0000.npz")["lens"].sum()) and cl["pts"].dtype == np.float16
    # snapshot from the dataset restores and continues (obs from the stored state equals the live continuation)
    store = ST.Store(out, "train")
    ep, t = 0, int(store.grid_ticks(0)[3])
    snap, hid = store.snapshot_at(ep, t)
    e = mk()
    e.restore(snap, mu_roll=hid)
    cur = e.current()["obs"]
    logged = store.rows("obs", ep, t, t + 1)[0]
    ok &= np.allclose(cur[:6], logged[:6], atol=1e-6) and np.allclose(cur[9:], logged[9:], atol=1e-5)   # cols 6:9 (axis rate) need the previous tick
    return check("gen_smoke", ok, wall_s=wall, train_eps=len(D3["lens"]), obs_dim=int(D3["obs"].shape[1]), cloud_shape=list(cl["pts"].shape),
                 restore_matches_logged_row=bool(np.allclose(cur[9:], logged[9:], atol=1e-5)))


def run_existing():
    t0 = time.perf_counter()
    r1 = subprocess.run([PY, "-m", "world_model.v2.tray_rep.selftest", "--fresh"], capture_output=True, text=True, cwd=str(K.REPO))
    print(r1.stdout[-1500:], flush=True)
    check("existing_phase1a_selftest", r1.returncode == 0, rc=r1.returncode, wall_s=time.perf_counter() - t0)
    t0 = time.perf_counter()
    r2 = subprocess.run([PY, "-m", "world_model.v2.tray_rep.wm.selftest_1b"], capture_output=True, text=True, cwd=str(K.REPO))
    print(r2.stdout[-1500:], flush=True)
    check("existing_wm_selftest_1b", r2.returncode == 0, rc=r2.returncode, wall_s=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--existing", action="store_true", help="also run the phase-1a selftest and wm/selftest_1b")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tests = [t_ray_analytic, t_noise_stat, t_grazing, t_est_noise_free, t_est_robust, t_determinism, lambda: t_snapshot(False),
             lambda: t_snapshot(True), t_replay_equal, t_parent_unchanged, t_dataset_equal, t_cloud, t_gen_smoke]
    t0 = time.perf_counter()
    for t in tests:
        if a.only and a.only not in getattr(t, "__name__", "lambda"):
            continue
        try:
            t()
        except Exception as ex:                                  # a crash is a failure, with the message
            import traceback
            traceback.print_exc()
            check(getattr(t, "__name__", "test"), False, error=repr(ex))
    if a.existing:
        run_existing()
    RES["_all_pass"] = all(v["ok"] for k, v in RES.items() if k != "_all_pass")
    RES["_wall_s"] = time.perf_counter() - t0
    (OUT / "selftest_pc_summary.json").write_text(json.dumps(RES, indent=1))
    print(f"[selftest_pc] {'ALL PASS' if RES['_all_pass'] else 'SOME FAILED'} in {RES['_wall_s']:.0f}s", flush=True)
    sys.exit(0 if RES["_all_pass"] else 1)


if __name__ == "__main__":
    main()
