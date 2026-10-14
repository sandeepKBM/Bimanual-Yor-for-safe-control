"""smoke_s2.py -- S2 payoff: the zed_demo desk scene, NAIVE straight-line tray placement (penetrates the bottle by 106 mm kinematically), executed by
FL-MPC (a) without and (b) with the PERCEIVED zonotopes as hard HOCBF rows.  SIMULATION ONLY.

    # 1) perceive once (stereo render -> SGBM -> cloud -> zonotopes); needs MUJOCO_GL=egl and a free non-zero GPU
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. python -m world_model.v2.house.zono_rows.smoke_s2 perceive
    # 2) closed-loop runs on the saved zonotopes (CPU only)
    PYTHONPATH=. python -m world_model.v2.house.zono_rows.smoke_s2 run [--margin 0.03 --alpha 10 --d-act 0.15 --max-rows 8]

Control rate: physics dt 5 ms, one FL-MPC tau() per mj_step (FLMPC DT = 5 ms), torque applied unscaled via qfrc_applied; the controller reads the
state with the zed_demo encoder noise (1e-4 rad, 2e-3 rad/s).  The base is the kinematic mocap path of zed_demo.execute.make_traj; its velocity
(known from the path) is given to the rows (ZonoCtl.set_base_motion); base acceleration is not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("ZONO_ROWS_OUT", "/data/local/ss5772/molmospaces/zono_rows"))
os.environ.setdefault("ZED_DEMO_OUT", str(OUT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import mujoco

T_NAIVE = np.array([0.0, 0.405])          # same as zed_demo.run: tray centre -> (4.2, 5.36), desk centre line


def perceive(a):
    from world_model.v2.house.zed_demo import rig as Z
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        g = Z.pick_gpu()
        if g is None:
            sys.exit("no free non-zero GPU for EGL")
        os.environ["CUDA_VISIBLE_DEVICES"] = g
    os.environ.setdefault("MUJOCO_GL", "egl")
    import cv2
    cv2.setNumThreads(1)
    from world_model.v2.house import phase1 as P1
    from world_model.v2.house.zed_demo import stereo as ST, perceive as PC, execute as EX
    OUT.mkdir(parents=True, exist_ok=True)
    C, gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT)
    m, d = C.m, C.d
    rg = Z.ZedRig(m, seed=0)
    cls = P1.geom_class(m)
    keys = PC.gt_keys(m)
    base0 = np.array(C.info["spawn"], float)
    sgbm = ST.make_sgbm()
    o = rg.capture(d)
    res = PC.run(m, d, rg, o, cls, keys, base0, float(C.info["yaw"]), seed=0, sgbm=sgbm)
    zs = res["zonos"]
    npz = {}
    meta = []
    for i, z in enumerate(zs):
        npz[f"c{i}"], npz[f"G{i}"] = z["Zi"].c, z["Zi"].G
        meta.append(dict(i=i, kind=z["kind"], gt_key=z["gt_key"], n_pts=z["n_pts"], n_gen=int(z["Zi"].G.shape[1])))
    np.savez(OUT / "s2_zonos.npz", n=len(zs), **npz)
    (OUT / "s2_zonos_meta.json").write_text(json.dumps(dict(zonos=meta, perceive_s=res["timing"]["perceive_s"], base0=base0.tolist(),
                                                           yaw0=float(C.info["yaw"])), indent=1))
    print(f"[perceive] {len(zs)} zonotopes saved ({sum(z['kind'] == 'obj' for z in zs)} obj + {sum(z['kind'] == 'support' for z in zs)} support), "
          f"perceive {res['timing']['perceive_s']:.1f}s", flush=True)
    os._exit(0)


def load_zonos():
    z = np.load(OUT / "s2_zonos.npz")
    meta = json.loads((OUT / "s2_zonos_meta.json").read_text())
    return [(z[f"c{i}"], z[f"G{i}"]) for i in range(int(z["n"]))], [f"Z{m['i']}:{m['gt_key']}" for m in meta["zonos"]], meta


def run_one(C, traj, keys, zonos, names, args, with_rows, seed=0, frames=None):
    from world_model.v2.house import build as B
    from world_model.v2.house import phase2 as P2
    from world_model.v2.house.zed_demo import execute as EX
    from world_model.v2.house.zono_rows.ctl import ZonoCtl
    from world_model.v2.testbeds.b_tray_ball import tray_env as TE
    m, d = C.m, C.d
    rng = np.random.default_rng(seed)
    ctl = ZonoCtl(C, zonos=zonos if with_rows else None, names=names, margin=args.margin, alpha1=args.alpha, alpha2=args.alpha,
                  d_act=args.d_act, max_rows=args.max_rows, tilt=True, refine=args.refine,
                  relax=None if args.relax == "none" else args.relax, relax_s0=args.relax_s0, relax_smax=args.relax_smax,
                  relax_bisect=args.relax_bisect, relax_max_solves=args.relax_max_solves, relax_frac=args.relax_frac,
                  tray_level=args.level, level_delta=args.level_delta, track_bodies=["tray", "ball"] if args.track else None,
                  level_relax=getattr(args, "level_relax", False), brake=getattr(args, "brake", False), ref_clip=getattr(args, "ref_clip", None),
                  proprio=True if getattr(args, "proprio", False) else None, proprio_ball=getattr(args, "proprio_ball", "prior"))
    mon = EX.ClearMon(C, keys)
    DT = EX.DT
    tray, ball = m.body("tray").id, m.body("ball").id
    ld, rd = ctl.ld, ctl.rd
    dn = mujoco.MjData(m)
    n_set, n_t, n_end = int(EX.SETTLE_S / DT), len(traj.t), int(EX.HOLD_END_S / DT)
    N = n_set + n_t + n_end
    tc = np.zeros(N)
    contacts, lost_at, tilt_max = {}, None, 0.0
    dmin, dmin_name, dmin_t = 9.0, "", 0.0
    dmin_nz, n_zero_d = 9.0, 0
    obj_at_dmin = {}
    tau_peak = np.zeros(12)
    trace = []
    tray0 = d.xpos[tray].copy()
    bottle_g = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("bottle_a") and m.geom_contype[g]]
    cls_ = mon.cls
    by_cls = {c: [g for g in mon.own if cls_[g] == c] for c in ("T", "P", "L", "R", "B")}
    bmin = {c: 9.0 for c in by_cls}
    bset = set(bottle_g)
    fpk, fimp, fbuf = 0.0, 0.0, np.zeros(6)
    bf = ref = None
    tb = np.zeros(N)
    if with_rows and getattr(args, "base_filter", False):
        from world_model.v2.house.zono_rows import base_filter as BF
        bf = BF.BaseFilter(ctl, BF.BaseCfg(gamma=args.bf_gamma, a_brk=args.bf_abrk, margin=args.bf_margin, d_act=args.bf_dact, max_rows=args.bf_max_rows,
                                           a_slew=(args.bf_aslew, args.bf_aslew, 2 * args.bf_aslew), every=args.bf_every), origin_xy=(traj.x[0], traj.y[0]), yaw=float(traj.yaw[0]), data=d)
        ref = BF.PathRef(traj, DT)
        pose = np.array([traj.x[0], traj.y[0], float(traj.yaw[0])])
    loc = None
    if bf is not None and getattr(args, "pose_source", "truth") == "est":       # house/loc: the base CBF + path reference read the ESTIMATED pose
        from world_model.v2.house.loc import estimator as ES, zono_est as ZE
        loc = ES.PoseEstimator(ES.scaled(ES.LocCfg(), args.loc_scale), seed=args.loc_seed, true_pose0=pose)
        ZE.attach(bf, lambda: pose)                                              # closure: always the latest TRUE pose
    bmin_every = getattr(args, "bottle_every", 0)
    for k in range(N):
        kk = min(max(k - n_set, 0), n_t - 1)
        if bf is None:
            d.mocap_pos[0] = (traj.x[kk], traj.y[kk], 0.0)
            d.mocap_quat[0] = B.yaw_quat(traj.yaw[kk])
            if 0 < k - n_set < n_t - 1:
                ctl.set_base_motion(((traj.x[kk + 1] - traj.x[kk]) / DT, (traj.y[kk + 1] - traj.y[kk]) / DT), 0.0, (traj.x[kk], traj.y[kk]))
            else:
                ctl.set_base_motion((0.0, 0.0), 0.0, (traj.x[kk], traj.y[kk]))
        mujoco.mj_copyData(dn, m, d)
        dn.qpos[ctl.lq] += rng.normal(0, P2.ENC_Q, 6); dn.qpos[ctl.rq] += rng.normal(0, P2.ENC_Q, 6)
        dn.qvel[ld] += rng.normal(0, P2.ENC_V, 6); dn.qvel[rd] += rng.normal(0, P2.ENC_V, 6)
        if bf is not None:                      # base twist through the zonotope clearance CBF; the mocap pose integrates the FILTERED twist
            tb0 = time.perf_counter()
            mujoco.mj_forward(m, dn)
            if loc is None:
                u_nom, _ix = ref.nominal(pose[:2], started=k >= n_set)
                u, _bi = bf.step(dn, u_nom, pose[:2], pose[2], DT)
                pose = pose + u * DT
            else:                                                                # u is a map-frame twist; the base executes it as a BODY twist
                pe = loc.pose.copy()
                u_nom, _ix = ref.nominal(pe[:2], started=k >= n_set)
                u, _bi = bf.step(dn, u_nom, pe[:2], pe[2], DT)
                ce, se, ct, st = math.cos(pe[2]), math.sin(pe[2]), math.cos(pose[2]), math.sin(pose[2])
                ub = (ce * u[0] + se * u[1], -se * u[0] + ce * u[1])
                u = np.array([ct * ub[0] - st * ub[1], st * ub[0] + ct * ub[1], u[2]])   # now the TRUE world twist
                pose = pose + u * DT
                loc.update(k * DT, pose, DT)
            d.mocap_pos[0] = (pose[0], pose[1], 0.0)
            d.mocap_quat[0] = B.yaw_quat(pose[2])
            dn.mocap_pos[0] = d.mocap_pos[0]; dn.mocap_quat[0] = d.mocap_quat[0]
            ctl.set_base_motion((u[0], u[1]), u[2], (pose[0], pose[1]))
            tb[k] = (time.perf_counter() - tb0) * 1e3
        t0 = time.perf_counter()
        te, ti = ctl.tau(dn)
        tc[k] = (time.perf_counter() - t0) * 1e3
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ld] = te
        d.qfrc_applied[rd] = ti
        mujoco.mj_step(m, d)
        tau_peak = np.maximum(tau_peak, np.abs(np.concatenate([te, ti])))
        for ci in range(d.ncon):                                   # normal force on the (static) bottle
            cc_ = d.contact[ci]
            if cc_.geom1 in bset or cc_.geom2 in bset:
                mujoco.mj_contactForce(m, d, ci, fbuf)
                fpk = max(fpk, float(fbuf[0])); fimp += float(fbuf[0]) * DT
        for (x, y) in mon.contacts(d):
            contacts[f"{x}|{y}"] = contacts.get(f"{x}|{y}", 0) + 1
        rel, tilt = P2.tray_stats(m, d, tray, ball)
        tilt_max = max(tilt_max, tilt)
        if lost_at is None and (abs(rel[0]) > TE.HX or abs(rel[1]) > TE.HY or rel[2] < 0.0):
            lost_at = (k - n_set) * DT
        if bmin_every and k % bmin_every == 0 and k % 10 != 0:               # denser true-distance sampling of the robot parts vs the bottle only
            for c_, gs in by_cls.items():
                for ga in gs:
                    for gb_ in bottle_g:
                        bmin[c_] = min(bmin[c_], mon._gd(d, ga, gb_, 0.4))
        if k % 10 == 0:
            dd, nm = mon.measure(d)
            for c_, gs in by_cls.items():
                for ga in gs:
                    for gb_ in bottle_g:
                        bmin[c_] = min(bmin[c_], mon._gd(d, ga, gb_, 0.4))
            trace.append(((k - n_set) * DT, float(dd), float(np.linalg.norm(d.xpos[tray][:2] - tray0[:2]))))
            if dd == 0.0:
                n_zero_d += 1                                           # exact 0.0 = mj_geomDistance degenerate return, not a penetration (see n_contact_steps)
            elif dd < dmin_nz:
                dmin_nz = float(dd)
            if dd < dmin:
                dmin, dmin_name, dmin_t = float(dd), nm, (k - n_set) * DT
        if frames is not None and k % EX.FRAME_EVERY == 0:
            frames.append(EX.snapshot(d))
    np.savez(OUT / f"s2_ms{args.tag}_{'rows' if with_rows else 'norows'}.npz", ms_total=tc, stats=np.array(ctl.stats, float))
    goal_xy = tray0[:2] + T_NAIVE
    fin = d.xpos[tray][:2]
    s = ctl.summary()
    bfres = {}
    if bf is not None:
        bfres = dict(**({"loc_pose_errors": loc.errors()} if loc is not None else {}), base_filter=bf.summary(), base_progress_frac=ref.progress(), base_path_remaining_m=float(ref.seg[-1] - ref.seg[ref.idx]),
                     base_final_xy=pose[:2].tolist(), base_goal_xy=[float(traj.x[-1]), float(traj.y[-1])],
                     base_final_dist_to_path_end_m=float(np.hypot(pose[0] - traj.x[-1], pose[1] - traj.y[-1])),
                     base_filter_ms_median=float(np.median(tb[n_set:])), base_filter_ms_p95=float(np.percentile(tb[n_set:], 95)),
                     base_filter_ms_max=float(tb[n_set:].max()))
    return dict(with_rows=with_rows, **bfres, min_true_dist_m=dmin, min_true_pair=dmin_name, min_true_dist_t_s=dmin_t, min_true_dist_nonzero_m=dmin_nz, n_zero_dist_samples=n_zero_d, n_contact_steps=int(sum(contacts.values())),
                contacts=contacts, tray_tilt_max_deg=float(tilt_max), ball_kept=lost_at is None, ball_lost_at_s=lost_at,
                final_tray_xy=fin.tolist(), goal_tray_xy=goal_xy.tolist(), final_dist_to_goal_m=float(np.linalg.norm(fin - goal_xy)),
                tray_xy_travel_m=float(np.linalg.norm(fin - tray0[:2])), per_object_min_m={k_: round(v, 4) for k_, v in sorted(mon.obj_min.items()) if v < 0.4},
                bottle_peak_normal_force_N=fpk, bottle_normal_impulse_Ns=fimp,
                bottle_min_dist_by_robot_part_m={c_: float(v) for c_, v in bmin.items()},
                tau_peak_max_Nm=float(tau_peak.max()), ctl_ms_median=float(np.median(tc[10:])), ctl_ms_p95=float(np.percentile(tc[10:], 95)),
                ctl_ms_max=float(tc[10:].max()), rows=s, n_steps=int(N), duration_s=float(N * DT), trace=trace[::3])


def run(a):
    from world_model.v2.house.zed_demo import rig as Z, execute as EX, perceive as PC
    zonos, names, meta = load_zonos()
    C, gt = Z.build_composed(pitch_deg=30.0, dt=EX.DT, textures=False)
    m, d = C.m, C.d
    keys = PC.gt_keys(m)
    base0 = np.array(C.info["spawn"], float)
    yaw0 = float(C.info["yaw"])
    snap0 = EX.snapshot(d)
    naive_path_t = np.stack([np.linspace(0, T_NAIVE[0], 60), np.linspace(0, T_NAIVE[1], 60)], 1)
    traj = EX.make_traj(naive_path_t, base0, yaw0)
    from world_model.v2.house.zono_rows import rows as R
    zs = R.ZonoSet(zonos, names)
    res = dict(setup=dict(n_zonotopes=len(zs), facets_total=int(sum(h.n_facets for h in zs.h)), facets_max=int(max(h.n_facets for h in zs.h)),
                          margin=a.margin, alpha1=a.alpha, alpha2=a.alpha, d_act=a.d_act, max_rows=a.max_rows, relax=a.relax, relax_frac=a.relax_frac, base_filter=a.base_filter, bf=dict(gamma=a.bf_gamma, a_brk=a.bf_abrk, margin=a.bf_margin, d_act=a.bf_dact, max_rows=a.bf_max_rows, a_slew=a.bf_aslew, every=a.bf_every), level=a.level, level_delta=a.level_delta, track=a.track, proprio=a.proprio, proprio_ball=a.proprio_ball, level_relax=a.level_relax, brake=a.brake, ref_clip=a.ref_clip, flmpc_impl=os.environ.get("BIMANUAL_FLMPC_IMPL", "py"), physics_dt=EX.DT,
                          control="one FLMPC.step per mj_step (DT=5 ms), torque applied unscaled", traj_s=float(traj.meta["duration_s"]),
                          base_xy0=base0.tolist(), tray0_xy=d.xpos[m.body("tray").id][:2].tolist(), naive_tray_target_t=T_NAIVE.tolist()))
    # kinematic replay of the naive path (no controller): the 106 mm penetration reference
    EX.restore(m, d, snap0)
    nv, _ = EX.replay_kinematic(C, traj, keys)
    res["naive_kinematic_replay"] = {k: v for k, v in nv.items() if k not in ("trace_t", "trace_d")}
    print("[naive kinematic]", res["naive_kinematic_replay"]["min_clearance_m"], res["naive_kinematic_replay"]["penetrating_pairs"], flush=True)
    frames = []
    for name, wr in (("with_rows", True),) + ((("without_rows", False),) if not a.only_rows else ()):
        EX.restore(m, d, snap0)
        d.qfrc_applied[:] = 0
        t0 = time.time()
        r = run_one(C, traj, keys, zs, names, a, wr, frames=frames if wr and a.frames else None)
        r["wall_s"] = time.time() - t0
        res[name] = r
        print(name, {k: v for k, v in r.items() if k not in ("trace", "contacts", "per_object_min_m", "rows")}, flush=True)
        print("   rows:", r["rows"], flush=True)
        print("   per_object_min:", r["per_object_min_m"], "contacts", r["contacts"], flush=True)
    (OUT / f"s2_results{a.tag}.json").write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    if a.frames:
        import pickle
        pickle.dump(frames, open(OUT / f"s2_frames{a.tag}.pkl", "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["perceive", "run"])
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--d-act", type=float, default=0.15)
    ap.add_argument("--max-rows", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--only-rows", action="store_true")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--relax", choices=["none", "uniform", "selective"], default="none", help="graceful relaxation of the zonotope rows: uniform scalar slack or per-row selective (needs BIMANUAL_FLMPC_IMPL=rt)")
    ap.add_argument("--relax-frac", type=float, default=0.9, help="selective: rows demanding more than this fraction of the torque-box best effort are lifted")
    ap.add_argument("--base-filter", action="store_true", help="route the base twist through the zonotope clearance CBF (base_filter.py) instead of the open-loop mocap path")
    ap.add_argument("--bf-gamma", type=float, default=1.5)
    ap.add_argument("--bf-abrk", type=float, default=0.4)
    ap.add_argument("--bf-margin", type=float, default=0.03)
    ap.add_argument("--bf-dact", type=float, default=0.40)
    ap.add_argument("--bf-max-rows", type=int, default=48)
    ap.add_argument("--bf-every", type=int, default=1, help="base filter solved every N-th control tick (twist held in between)")
    ap.add_argument("--bf-aslew", type=float, default=0.5, help="base linear slew limit [m/s^2] (yaw 2x)")
    ap.add_argument("--pose-source", choices=["truth", "est"], default="truth", help="est: base CBF / path reference read the house/loc pose estimate (needs --base-filter)")
    ap.add_argument("--loc-scale", type=float, default=1.0)
    ap.add_argument("--loc-seed", type=int, default=0)
    ap.add_argument("--bottle-every", type=int, default=0, help="sample the robot-part-to-bottle true distance every N steps (0 = only every 10th, as before)")
    ap.add_argument("--relax-s0", type=float, default=0.05)
    ap.add_argument("--relax-smax", type=float, default=50.0)
    ap.add_argument("--relax-bisect", type=int, default=3)
    ap.add_argument("--relax-max-solves", type=int, default=12)
    ap.add_argument("--level", action="store_true", help="keep the tray level: hard right-arm tilt row + grip-height band rows + tracked tray rim")
    ap.add_argument("--level-delta", type=float, default=0.015)
    ap.add_argument("--level-relax", action="store_true", help="last-resort tier: if a step is still on the a=0 fallback, slacken the tray_level rows together with the zonotope rows")
    ap.add_argument("--ref-clip", type=float, default=None, help="anti-windup: clip the joint reference to +-this [rad] around the measured q")
    ap.add_argument("--brake", action="store_true", help="a=0 fallback -> velocity brake (a = -qd/0.04 scaled into the torque box)")
    ap.add_argument("--track", action="store_true", help="tracked tray + ball keypoints (measured pose/velocity) instead of the fixed weld offsets")
    ap.add_argument("--proprio", action="store_true", help="PROPRIO-ONLY: tray rim from hand FK (no d.xpos tracked points); ball = fixed prior spheres (see --proprio-ball)")
    ap.add_argument("--proprio-ball", choices=["prior", "drop"], default="prior")
    a = ap.parse_args()
    perceive(a) if a.mode == "perceive" else run(a)


if __name__ == "__main__":
    main()
