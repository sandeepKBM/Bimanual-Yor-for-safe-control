"""s2.py -- S2 desk scene (zed_demo, perceived zonotopes, tray carried over the desk, bottle on the path): base filter ALONE vs MPPI planner + base filter.
SIMULATION ONLY.

The loop is smoke_s2.run_one's (FL-MPC arms with the zonotope rows, kinematic mocap base driven by the FILTERED twist), with ONE change: the
nominal twist handed to zono_rows.base_filter.BaseFilter.step comes from either PathRef (baseline) or the sampling planner (this package).
The hard filter is unchanged; the planner only proposes.

    python -m world_model.v2.house.mppi.s2 --mode both --K 1024 --T 14 --tag s2_k1024
Needs s2_zonos.npz / s2_zonos_meta.json in ZONO_ROWS_OUT (smoke_s2 perceive; copied to the mppi out dir).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

OUT = Path(os.environ.get("MPPI_OUT", "/data/local/ss5772/molmospaces/mppi"))
os.environ.setdefault("ZONO_ROWS_OUT", str(OUT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BIMANUAL_FLMPC_IMPL", "rt")

import numpy as np
import mujoco
import torch

from . import dynamics as DY, fields as FL, cost as CO, planner as PL

T_NAIVE = np.array([0.0, 0.405])


def dedupe_points(P, r, tol=0.012):
    """merge points closer than tol (kept point's radius += the merge distance): conservative, ~no loss of coverage."""
    keep, rr = [], []
    for i in np.argsort(-r):
        if keep:
            dd = np.linalg.norm(P[keep] - P[i], axis=1)
            j = int(np.argmin(dd))
            if dd[j] < tol:
                rr[j] = max(rr[j], r[i] + dd[j]); continue
        keep.append(i); rr.append(r[i])
    return P[keep], np.array(rr)


def cover_points(P, r, n):
    """greedy farthest-point cover of robot points (P (K,3), r (K,)) by n centres; each dropped point's (distance + radius) is folded into its centre's
    radius, so the reduced set is a conservative (inflated) cover of the full one."""
    K = len(P)
    if K <= n:
        return P, r
    sel = [int(np.argmax(P[:, 2]))]
    dmin = np.linalg.norm(P - P[sel[0]], axis=1)
    while len(sel) < n:
        j = int(np.argmax(dmin)); sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
    C = P[sel]
    asg = np.argmin(np.linalg.norm(P[:, None] - C[None], axis=2), axis=1)
    rr = r[sel].copy()
    for i in range(K):
        rr[asg[i]] = max(rr[asg[i]], r[i] + np.linalg.norm(P[i] - C[asg[i]]))
    return C, rr


def build_scene(textures=False):
    from world_model.v2.house.zed_demo import rig as Z, execute as EX, perceive as PC
    from world_model.v2.house.zono_rows import smoke_s2 as S2, rows as R
    zonos, names, meta = S2.load_zonos()
    C, gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=textures)
    keys = PC.gt_keys(C.m)
    base0 = np.array(C.info["spawn"], float); yaw0 = float(C.info["yaw"])
    snap0 = EX.snapshot(C.d)
    path_t = np.stack([np.linspace(0, T_NAIVE[0], 60), np.linspace(0, T_NAIVE[1], 60)], 1)
    traj = EX.make_traj(path_t, base0, yaw0)
    zs = R.ZonoSet(zonos, names)
    return dict(C=C, keys=keys, base0=base0, yaw0=yaw0, snap0=snap0, traj=traj, zs=zs, names=names, EX=EX)


def build_problem(sc, ctl, bf, K_pts=None, H=24, dt=0.125, cfg=None, device="cuda", res=0.015):
    """CarryProblem for the S2 scene: world-twist base (the filter's slew 0.5 m/s^2 / 0.3 m/s box), robot points from the BaseFilter point set
    (arms + tray rim + ball + footprint) in the base frame, perceived-zonotope sd_lb volume on the GPU."""
    d = sc["C"].d
    traj = sc["traj"]
    o = np.array([traj.x[0], traj.y[0], 0.0]); yaw = float(traj.yaw[0])
    mujoco.mj_forward(sc["C"].m, d)
    p, r, v, nf = bf.points(d, (o[0], o[1]), yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    loc = (p - o) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]).T            # R^T (p - o)
    Pc, rc = dedupe_points(loc, r.copy()) if not K_pts else cover_points(loc, r.copy(), K_pts)
    hr = sc["zs"].h
    zmax = float(max(h.hi[2] for h in hr))
    field = FL.Zono3DField(hr, lo=(sc["base0"][0] - 0.9, sc["base0"][1] - 0.4, 0.0), hi=(sc["base0"][0] + 1.3, sc["base0"][1] + 1.4, min(zmax, 1.4)), res=res, device=device)
    model = DY.BaseModel(dt=dt, frame="world", a_hw=(0.5, 0.5, 1.0), v_max=(0.3, 0.3, 0.06))
    ball_loc = loc[[i for i, n in enumerate(bf.names) if n == "tracked:ball"][0], :2]
    prob = CO.CarryProblem(model, H, Pc[:, :2], rc, Pc[:, 2], field, cfg or CO.CostCfg(w_yaw=40.0), mode="path", device=device, r_cup=ball_loc, v_scale=(0.3, 0.3, 0.3))
    L = float(np.hypot(traj.x[-1] - traj.x[0], traj.y[-1] - traj.y[0]))
    t_hat = np.array([traj.x[-1] - traj.x[0], traj.y[-1] - traj.y[0]]) / L
    return prob, dict(p0=[traj.x[0], traj.y[0]], t_hat=t_hat, s_end=L, yaw_ref=yaw, n_pts=len(Pc), n_full=len(loc))


def run_episode(sc, args, use_planner, planner=None, prob=None, pinfo=None, mu_label=""):
    from world_model.v2.house import build as B
    from world_model.v2.house import phase2 as P2
    from world_model.v2.house.zono_rows.ctl import ZonoCtl
    from world_model.v2.house.zono_rows import base_filter as BF
    from world_model.v2.testbeds.b_tray_ball import tray_env as TE
    C, EX, traj, keys = sc["C"], sc["EX"], sc["traj"], sc["keys"]
    m, d = C.m, C.d
    EX.restore(m, d, sc["snap0"]); d.qfrc_applied[:] = 0
    rng = np.random.default_rng(0)
    ctl = ZonoCtl(C, zonos=sc["zs"], names=sc["names"], margin=0.02, alpha1=10.0, alpha2=10.0, d_act=0.15, max_rows=64, tilt=True, refine=0,
                  relax="uniform", tray_level=True, level_delta=0.015, track_bodies=["tray", "ball"])
    mon = EX.ClearMon(C, keys)
    DT = EX.DT
    tray, ball = m.body("tray").id, m.body("ball").id
    ld, rd = ctl.ld, ctl.rd
    dn = mujoco.MjData(m)
    n_set = int(EX.SETTLE_S / DT)
    N = n_set + int(args.T / DT)
    bcfg = BF.BaseCfg(gamma=1.5, a_brk=0.4, margin=0.03, d_act=0.40, max_rows=48, a_slew=(0.5, 0.5, 1.0), every=1)
    bf = BF.BaseFilter(ctl, bcfg, origin_xy=(traj.x[0], traj.y[0]), yaw=float(traj.yaw[0]), data=d)
    ref = BF.PathRef(traj, DT)
    pose = np.array([traj.x[0], traj.y[0], float(traj.yaw[0])])
    if use_planner and prob is None:
        prob, pinfo = build_problem(sc, ctl, bf)
    contacts, lost_at, tilt_max = {}, None, 0.0
    dmin, dmin_name, dmin_t = 9.0, "", 0.0
    bottle_g = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("bottle_a") and m.geom_contype[g]]
    bset = set(bottle_g)
    fpk, fimp, fbuf = 0.0, 0.0, np.zeros(6)
    tb = np.zeros(N); tray0 = d.xpos[tray].copy()
    interv, plan_log, traj_log = [], [], []
    u_nom = np.zeros(3); last_plan_k = -10 ** 9
    plan_every = int(round(args.plan_dt / DT))
    if use_planner:
        prob.set_state(pose=pose, v=np.zeros(3), p0=pinfo["p0"], t_hat=pinfo["t_hat"], s_end=pinfo["s_end"], yaw_ref=pinfo["yaw_ref"])
    for k in range(N):
        mujoco.mj_copyData(dn, m, d)
        dn.qpos[ctl.lq] += rng.normal(0, P2.ENC_Q, 6); dn.qpos[ctl.rq] += rng.normal(0, P2.ENC_Q, 6)
        dn.qvel[ctl.ld] += rng.normal(0, P2.ENC_V, 6); dn.qvel[ctl.rd] += rng.normal(0, P2.ENC_V, 6)
        tb0 = time.perf_counter()
        mujoco.mj_forward(m, dn)
        started = k >= n_set
        if use_planner:
            if started and k - last_plan_k >= plan_every:
                prob.set_state(pose=pose, v=bf.u_prev, u_prev=u_nom, p0=pinfo["p0"], t_hat=pinfo["t_hat"], s_end=pinfo["s_end"], yaw_ref=pinfo["yaw_ref"])
                U, inf = planner.plan(elapsed=(k - last_plan_k) * DT if last_plan_k > 0 else None)
                u_nom = U[0, :3].astype(float); last_plan_k = k
                plan_log.append((k * DT, inf["ms"], inf["cost"], inf["h_min"]))
            if not started:
                u_nom = np.zeros(3)
        else:
            u_nom, _ix = ref.nominal(pose[:2], started=started)
        up = bf.u_prev.copy()
        u, _bi = bf.step(dn, u_nom, pose[:2], pose[2], DT)
        pose = pose + u * DT
        d.mocap_pos[0] = (pose[0], pose[1], 0.0); d.mocap_quat[0] = B.yaw_quat(pose[2])
        dn.mocap_pos[0] = d.mocap_pos[0]; dn.mocap_quat[0] = d.mocap_quat[0]
        ctl.set_base_motion((u[0], u[1]), u[2], (pose[0], pose[1]))
        tb[k] = (time.perf_counter() - tb0) * 1e3
        te, ti = ctl.tau(dn)
        d.qfrc_applied[:] = 0.0; d.qfrc_applied[ld] = te; d.qfrc_applied[rd] = ti
        mujoco.mj_step(m, d)
        if started:   # hard-filter intervention = deviation of the filtered twist from the slew/box-clipped proposal
            lo_ = np.maximum(-np.array(bcfg.v_max), up - np.array(bcfg.a_slew) * DT); hi_ = np.minimum(np.array(bcfg.v_max), up + np.array(bcfg.a_slew) * DT)
            interv.append(float(np.linalg.norm((u - np.clip(u_nom, lo_, hi_)) / np.array(bcfg.w_scale))))
        for ci in range(d.ncon):
            cc_ = d.contact[ci]
            if cc_.geom1 in bset or cc_.geom2 in bset:
                mujoco.mj_contactForce(m, d, ci, fbuf); fpk = max(fpk, float(fbuf[0])); fimp += float(fbuf[0]) * DT
        for (x, y) in mon.contacts(d):
            contacts[f"{x}|{y}"] = contacts.get(f"{x}|{y}", 0) + 1
        rel, tilt = P2.tray_stats(m, d, tray, ball)
        tilt_max = max(tilt_max, tilt)
        if lost_at is None and (abs(rel[0]) > TE.HX or abs(rel[1]) > TE.HY or rel[2] < 0.0):
            lost_at = (k - n_set) * DT
        if k % 10 == 0:
            dd, nm = mon.measure(d)
            if dd < dmin:
                dmin, dmin_name, dmin_t = float(dd), nm, (k - n_set) * DT
            traj_log.append((k * DT, pose[0], pose[1], u[0], u[1], float(dd), u_nom[0], u_nom[1], interv[-1] if interv else 0.0))
    s_path = float((pose[:2] - np.array(pinfo["p0"] if pinfo else [traj.x[0], traj.y[0]])) @ (np.array([traj.x[-1] - traj.x[0], traj.y[-1] - traj.y[0]]) / np.hypot(traj.x[-1] - traj.x[0], traj.y[-1] - traj.y[0])))
    L = float(np.hypot(traj.x[-1] - traj.x[0], traj.y[-1] - traj.y[0]))
    iv = np.array(interv) if interv else np.zeros(1)
    st = np.array(bf.stats, float)
    res = dict(planner=bool(use_planner), ball_kept=lost_at is None, ball_lost_at_s=lost_at, min_true_dist_m=dmin, min_true_pair=dmin_name, min_true_dist_t_s=dmin_t,
               n_contact_steps=int(sum(contacts.values())), contacts=contacts, bottle_peak_normal_force_N=fpk, bottle_normal_impulse_Ns=fimp,
               tray_tilt_max_deg=float(tilt_max), path_progress_frac=s_path / L, base_final_xy=pose[:2].tolist(), base_start_xy=[float(traj.x[0]), float(traj.y[0])],
               lateral_offset_final_m=float(pose[0] - traj.x[0]), reached_path_end=bool(s_path >= L - 0.02), dist_to_path_end_m=float(np.hypot(pose[0] - traj.x[-1], pose[1] - traj.y[-1])),
               tray_final_xy=d.xpos[tray][:2].tolist(), per_object_min_m={k_: round(v, 4) for k_, v in sorted(mon.obj_min.items()) if v < 0.4},
               filter_intervention_frac_gt1e3=float((iv > 1e-3).mean()), filter_intervention_mean=float(iv.mean()), filter_intervention_max=float(iv.max()),
               filter_steps_braked=int(st[:, 4].sum()), filter_steps_peeled=int((st[:, 3] > 0).sum()), filter_steps_lifted=int((st[:, 2] > 0).sum()),
               filter_ms_median=float(np.median(tb[n_set:])), sim_s=float(N * DT), n_steps=N)
    if plan_log:
        pm = np.array(plan_log)
        res.update(plan_n=len(plan_log), plan_ms_median_in_loop=float(np.median(pm[:, 1])), plan_ms_p95_in_loop=float(np.percentile(pm[:, 1], 95)))
    return res, np.array(traj_log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "mppi", "both"], default="both")
    ap.add_argument("--K", type=int, nargs="+", default=[1024])
    ap.add_argument("--method", default="mppi")
    ap.add_argument("--T", type=float, default=14.0, help="run length after the 0.6 s settle [s]")
    ap.add_argument("--plan-dt", type=float, default=0.05)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--dt", type=float, default=0.125)
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile (inductor) the rollout+cost: ~30-90 s one-time startup, 2-3x faster plans")
    ap.add_argument("--no-cup", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sc = build_scene()
    out = {}
    if a.mode in ("base", "both"):
        r, tl = run_episode(sc, a, False)
        out["base_filter_alone"] = r; np.save(OUT / f"{a.tag}_base_traj.npy", tl)
        print("[base alone]", {k: v for k, v in r.items() if k not in ("contacts", "per_object_min_m")}, flush=True)
    if a.mode in ("mppi", "both"):
        from world_model.v2.house.zono_rows.ctl import ZonoCtl
        from world_model.v2.house.zono_rows import base_filter as BF
        for K in a.K:
            # build the problem once per K (needs a ctl/bf for the point set); then warm up (reported separately)
            C = sc["C"]; EX = sc["EX"]
            EX.restore(C.m, C.d, sc["snap0"])
            ctl = ZonoCtl(C, zonos=sc["zs"], names=sc["names"], margin=0.02, alpha1=10.0, alpha2=10.0, d_act=0.15, max_rows=64, tilt=True, refine=0,
                          relax="uniform", tray_level=True, level_delta=0.015, track_bodies=["tray", "ball"])
            bf = BF.BaseFilter(ctl, BF.BaseCfg(), origin_xy=(sc["traj"].x[0], sc["traj"].y[0]), yaw=float(sc["traj"].yaw[0]), data=C.d)
            t0 = time.perf_counter()
            prob, pinfo = build_problem(sc, ctl, bf, H=a.H, dt=a.dt)
            t_field = (time.perf_counter() - t0) * 1e3
            pl = PL.SamplingPlanner(prob, K=K, method=a.method, sigma=(0.08, 0.08, 0.02), use_graph=not a.no_graph, seed=a.seed, plan_dt=a.plan_dt, compile=a.compile)
            wu = pl.warmup()
            print(f"[K={K}] field build {t_field:.0f} ms, warm-up {wu:.0f} ms (graph={pl.stats['graph']}), n_pts={pinfo['n_pts']}/{pinfo['n_full']}", flush=True)
            r, tl = run_episode(sc, a, True, pl, prob, pinfo)
            r["planner_summary"] = pl.summary(); r["field_build_ms"] = t_field; r["warmup_ms"] = wu
            out[f"mppi_K{K}"] = r; np.save(OUT / f"{a.tag}_K{K}_traj.npy", tl)
            print(f"[mppi K={K}]", {k: v for k, v in r.items() if k not in ("contacts", "per_object_min_m")}, flush=True)
    (OUT / f"{a.tag}_results.json").write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


if __name__ == "__main__":
    main()
