"""gpusense/equiv.py -- GPU sensor / mapper / EDT equivalence vs the CPU ray-cast, numba mapper and scipy EDT on identical inputs.  SIMULATION ONLY.

  python -m world_model.v2.house.gpusense.equiv sensors   # needs fixture.py output
  python -m world_model.v2.house.gpusense.equiv mapping [n_frames]
Writes OUT/equiv_{sensors,mapping}.json and prints the headline numbers.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

OUT = Path("/data/local/ss5772/molmospaces/percept/gpu")


def _edge(z, g):
    e = np.zeros(z.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                e |= (np.roll(np.roll(g, dy, 0), dx, 1) != g) | (np.abs(np.roll(np.roll(z, dy, 0), dx, 1) - z) > 0.02)
    return e


def sensors():
    import mujoco
    from world_model.v2.house.gpusense import core as C
    from world_model.v2.house.people_track import camera as CAM
    from world_model.v2.house.grasp import wrist as WR
    F = pickle.load(open(OUT / "fix.pkl", "rb"))
    fix = F["fix"]
    m = mujoco.MjModel.from_binary_path(str(OUT / "h5.mjb"))
    pg = np.array(sorted(F["person_geoms"]), int)
    person_of = np.full(m.ngeom, -1, int)
    for g, p in F["person_geoms"].items():
        person_of[g] = p
    spec = CAM.head_spec()
    wc = {s: WR.WristCam(m, s) for s in ("left", "right")}
    cfgs = [C.GpuCamCfg("head", spec, CAM.G_CAST, ((pg, 5),), (), CAM.CUTOFF_MULT, F["self_geom"], person_of)]
    for s, w in wc.items():
        cfgs.append(C.GpuCamCfg("wrist_" + s, w.spec, WR.MASK_GROUPS, (), tuple(int(g) for g in np.flatnonzero(m.geom_bodyid == w.body_excl)), WR.CUTOFF_MULT,
                                w.is_robot_geom.copy(), None))
    core = C.SenseCore(str(OUT / "h5.mjb"), cfgs)
    d = mujoco.MjData(m)

    def setd(fr):
        d.qpos[:] = fr["qpos"]; d.mocap_pos[:] = fr["mpos"]; d.mocap_quat[:] = fr["mquat"]; d.eq_active[:] = fr["eq"]
        mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
        core.set_state(fr["qpos"], fr["mpos"], fr["mquat"], fr["eq"])

    rows = {"head": [], "wrist": []}

    def depth_row(zc, gc, zg, gg):
        both = (zc > 0) & (zg > 0)
        ed = _edge(zc, gc) | _edge(zg, gg)
        it = both & ~ed
        err = np.abs(zg - zc)
        return dict(hit_xor_frac=float(((zc > 0) ^ (zg > 0)).mean()), interior_px=int(it.sum()), silhouette_px=int((both & ed).sum()),
                    interior_med_mm=1e3 * float(np.median(err[it])), interior_p99_mm=1e3 * float(np.percentile(err[it], 99)), interior_max_mm=1e3 * float(err[it].max()),
                    interior_frac_lt1mm=float((err[it] < 1e-3).mean()), silhouette_frac_lt1mm=float((err[both & ed] < 1e-3).mean()),
                    all_frac_lt1mm=float((err[both] < 1e-3).mean()), robot_class_agree=None)

    mask = np.zeros(6, np.uint8); mask[list(CAM.G_CAST)] = 1
    cam = CAM.D.DepthCamera(spec)
    pgg = m.geom_group[pg].copy()
    for fr in fix:
        o, R = fr["pose"]; setd(fr)
        dirs = cam.dirs.reshape(-1, 3); cth = cam.cos_th.reshape(-1); n = len(dirs); dw = np.ascontiguousarray(dirs @ R.T)
        geom = np.full(n, -1, np.int32); dist = np.full(n, -1.0); nrm = np.zeros(3 * n)
        m.geom_group[pg] = 5
        mujoco.mj_multiRay(m, d, o, dw.reshape(-1), mask, True, -1, geom, dist, nrm, n, spec.z_max * CAM.CUTOFF_MULT)
        m.geom_group[pg] = pgg
        hit = (geom >= 0) & (dist > 0)
        zc = np.where(hit, dist * cth, 0.0).reshape(spec.H, spec.W); gc = np.where(hit, geom, -1).reshape(spec.H, spec.W)
        z, g = core.depth("head", o, R)
        zg, gg = z.cpu().numpy(), g.cpu().numpy()
        r = depth_row(zc, gc, zg, gg)
        cosc = np.abs(np.einsum("ij,ij->i", nrm.reshape(-1, 3), dw)).reshape(spec.H, spec.W)
        cg = C.depth_cos_incidence(core._cam("head"), z, (g >= 0) & (z > 0)).cpu().numpy()
        h = zc > 0
        r["grazing_mask_agree"] = float(((cosc >= spec.cos_min) == (cg >= spec.cos_min))[h].mean())
        r["geom_id_agree"] = float((gg == gc)[(zc > 0) & (zg > 0)].mean())
        # noise statistics: the GPU frame vs the CPU frame of the fixture (same state)
        job = dict(cam="head", kind="cloud", origin=o, R=R, tick=1, seed=0)
        nv = [len(core.run_job(dict(job, tick=t))["z"]) for t in range(10)]
        gr = core.run_job(job)
        zt = zg[gr["pix"][:, 0], gr["pix"][:, 1]]
        rr = (gr["z"] - zt) / spec.sigma_z(zt)
        cpu = fr["head"]
        zc_pix = zg[cpu["pix"][:, 0], cpu["pix"][:, 1]]
        rc = (cpu["z"] - zc_pix) / spec.sigma_z(zc_pix)
        r.update(n_valid_cpu=len(cpu["z"]), n_valid_gpu_mean=float(np.mean(nv)), resid_std_cpu=float(np.std(rc[np.abs(rc) < 4])), resid_std_gpu=float(np.std(rr[np.abs(rr) < 4])),
                 outlier_frac_gpu=float((np.abs(gr["z"] - zt) > 0.04).mean()), person_pts_cpu=int((cpu["person"] >= 0).sum()), person_pts_gpu=int((gr["person"] >= 0).sum()))
        a, b = core.run_job(job), core.run_job(job)
        r["deterministic"] = bool(np.array_equal(a["pts"], b["pts"]))
        rows["head"].append(r)
    for s, w in wc.items():
        for fr in fix[:6]:
            setd(fr)
            o, R = w.pose(d)
            dirs = w.cam.dirs.reshape(-1, 3); cth = w.cam.cos_th.reshape(-1); n = len(dirs); dw = np.ascontiguousarray(dirs @ R.T)
            geom = np.full(n, -1, np.int32); dist = np.full(n, -1.0)
            mujoco.mj_multiRay(m, d, o, dw.reshape(-1), w.mask, True, w.body_excl, geom, dist, None, n, w.spec.z_max * WR.CUTOFF_MULT)
            sp = w.spec; hit = (geom >= 0) & (dist > 0)
            zc = np.where(hit, dist * cth, 0.0).reshape(sp.H, sp.W); gc = np.where(hit, geom, -1).reshape(sp.H, sp.W)
            z, g = core.depth("wrist_" + s, o, R)
            zg, gg = z.cpu().numpy(), g.cpu().numpy()
            r = depth_row(zc, gc, zg, gg)
            both = (zc > 0) & (zg > 0)
            r["robot_class_agree"] = float((w.is_robot_geom[np.maximum(gc, 0)] == w.is_robot_geom[np.maximum(gg, 0)])[both].mean())
            r["cup_class_agree"] = float((w.is_cup_geom[np.maximum(gc, 0)] == w.is_cup_geom[np.maximum(gg, 0)])[both].mean())
            r["gt1cm_frac"] = float((np.abs(zg - zc)[both] > 0.01).mean())
            cpuw = fr["wrist_" + s]
            gr = core.run_job(dict(cam="wrist_" + s, kind="cloud", origin=o, R=R, tick=3, seed=0))
            r.update(n_valid_cpu=len(cpuw.z), n_valid_gpu=len(gr["z"]), n_self_cpu=cpuw.n_self, n_self_gpu=gr["n_self"], side=s)
            rows["wrist"].append(r)
    core.close()
    summ = {}
    for k, rs in rows.items():
        for key in rs[0]:
            if isinstance(rs[0][key], (int, float)) and not isinstance(rs[0][key], bool) and rs[0][key] is not None:
                summ[f"{k}.{key}"] = (float(np.min([r[key] for r in rs])), float(np.mean([r[key] for r in rs])), float(np.max([r[key] for r in rs])))
    json.dump(dict(rows=rows, summary_min_mean_max=summ), open(OUT / "equiv_sensors.json", "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:38s} min {v[0]:12.5g} mean {v[1]:12.5g} max {v[2]:12.5g}")


def mapping(nf_max=60):
    import mujoco
    import torch
    from scipy import ndimage
    from world_model.v2.house import build as B
    from world_model.v2.house.loc import mapper as MP
    from world_model.v2.house.loc.estimator import LocCfg, PoseEstimator, scaled
    from world_model.v2.house.teleop import scene as SC, geom as GM
    from world_model.v2.house.gpusense import mapping as GMAP, core as C
    from world_model.v2.tray_rep.perception import depth_sim as D
    scene = SC.get_scene("h5"); cfg = MP.MapCfg()
    m = B.load_house_spec(scene.house, True).compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    path = C.snapshot_model_bytes(m, str(OUT / "h5house.mjb"))
    true = SC.load_occupancy(scene)
    ny, nx = int(round(true.occ.shape[0] * true.res / cfg.res)), int(round(true.occ.shape[1] * true.res / cfg.res))
    poses = MP.exploration_path(np.asarray(scene.out_route, float), cfg)
    est = PoseEstimator(scaled(LocCfg(), cfg.map_loc_scale), seed=cfg.mapping_loc_seed, true_pose0=poses[0])
    cs = D.CamSpec(name="zed_map", mount="world", pos=(0, 0, 0), R=tuple(np.eye(3).ravel()), fovy=cfg.fovy, W=cfg.W, H=cfg.H, z_max=cfg.z_map_max)
    core = C.SenseCore(path, [C.GpuCamCfg("map", cs, (3, 4), kind="map")]); core.warmup()
    cam = MP.HeadDepth(m, d, cfg, seed=0)
    gm_e = GMAP.GpuMapper(core, cfg, true.x0, true.y0, ny, nx, "map")      # exact sequence semantics
    gm_p = GMAP.GpuMapper(core, cfg, true.x0, true.y0, ny, nx, "map")      # phased (cheaper) semantics
    L = np.zeros((int(math.ceil((cfg.obs_zhi - cfg.obs_zlo) / cfg.dz)), ny, nx), np.float32)
    every = int(round(1.0 / (cfg.frame_hz * 0.02)))
    okag, nfag, nf = [], [], 0
    orig = GMAP.integrate_gpu
    for k, p in enumerate(poses[1:], 1):
        est.update(k * 0.02, p, 0.02)
        if k % every:
            continue
        ok, pts, nofree, _ = cam.capture(p)
        o, ex, ey, ez, hit, fx, fy, fz = cam.place(ok, pts, nofree, est.pose)
        MP._integrate(L, cfg.res, true.x0, true.y0, float(o[0]), float(o[1]), float(o[2]), ex, ey, ez, hit, cfg.obs_zlo, cfg.obs_zhi, cfg.l_occ, cfg.l_free, cfg.l_clamp, fx, fy, fz, cfg.dz)
        okt, znt, nft = (torch.as_tensor(a, device="cuda") for a in (ok, pts[:, 0].copy(), nofree))
        gm_e.place_integrate(okt, znt, nft, est.pose)
        GMAP.integrate_gpu = lambda *a, **kw: orig(*a, exact=False, **kw)
        gm_p.place_integrate(okt, znt, nft, est.pose)
        GMAP.integrate_gpu = orig
        ok2, _, nf2 = gm_e.capture(p)                                       # sensor chain on the same true pose (independent noise draw)
        okag.append(float((ok2.cpu().numpy() == ok).mean())); nfag.append(float((nf2.cpu().numpy() == nofree).mean()))
        nf += 1
        if nf >= nf_max:
            break
    res = dict(frames=nf, grid=[ny, nx])
    for tag, gm in (("exact", gm_e), ("phased", gm_p)):
        Lg = gm.L.cpu().numpy()
        oc, og = (L > cfg.occ_thresh).any(0), (Lg > cfg.occ_thresh).any(0)
        res[f"{tag}_max_abs_dL"] = float(np.abs(Lg - L).max())
        res[f"{tag}_cells_dL_gt_1e-3"] = int((np.abs(Lg - L) > 1e-3).sum())
        res[f"{tag}_raw_occ_xor"] = int((oc ^ og).sum()); res["raw_occ_cells_cpu"] = int(oc.sum())
    pm = MP.PerceivedMap(cfg.res, true.x0, true.y0, L, cfg.asdict(), {})
    mc = pm.occ_mask()
    mg = GMAP.occ_mask_gpu(gm_e.L, cfg.occ_thresh, cfg.unknown, cfg.despeckle, cfg.close_iters).cpu().numpy()
    res["occ_mask_cells_cpu"] = int(mc.sum()); res["occ_mask_xor_exact"] = int((mc ^ mg).sum())
    for name, occ, r_ in (("perceived", mc, cfg.res), ("true_house", true.occ, true.res)):
        ref = ndimage.distance_transform_edt(~occ) * r_
        e = GMAP.edt_gpu(torch.as_tensor(occ, device="cuda"), r_).cpu().numpy()
        res[f"edt_{name}_max_abs_err_m"] = float(np.abs(e - ref).max())
        gref = GM.edt_grad(SC.OccGrid(r_, 0, 0, occ, ref))
        gx, gy = GMAP.grad_gpu(torch.as_tensor(ref, device="cuda", dtype=torch.float32), r_)
        res[f"grad_{name}_max_abs_err"] = float(max(np.abs(gx.cpu().numpy() - gref[0]).max(), np.abs(gy.cpu().numpy() - gref[1]).max()))
    # shield-facing API: GM.grid_lookup_grad on the host grid built from the GPU result == the same call on the CPU grid
    go = GMAP.GpuOccGrid(cfg.res, true.x0, true.y0, torch.as_tensor(mc, device="cuda"))
    gh = go.to_host(); gc_ = SC.OccGrid(cfg.res, true.x0, true.y0, mc, ndimage.distance_transform_edt(~mc) * cfg.res)
    rng = np.random.default_rng(0)
    qx = rng.uniform(true.x0, true.x0 + nx * cfg.res, 5000); qy = rng.uniform(true.y0, true.y0 + ny * cfg.res, 5000)
    a, b = GM.grid_lookup_grad(gh, qx, qy), GM.grid_lookup_grad(gc_, qx, qy)
    res["grid_lookup_grad_max_abs"] = [float(np.abs(x - y).max()) for x, y in zip(a, b)]
    tx, ty = torch.as_tensor(qx, device="cuda"), torch.as_tensor(qy, device="cuda")
    t = go.lookup_grad(tx, ty)
    res["gpu_lookup_grad_vs_host_max_abs"] = [float((x.cpu().numpy() - y).__abs__().max()) for x, y in zip(t, a)]
    res["sensor_ok_mask_agree_mean"] = float(np.mean(okag)); res["sensor_nofree_mask_agree_mean"] = float(np.mean(nfag))
    core.close()
    json.dump(res, open(OUT / "equiv_mapping.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "sensors":
        sensors()
    else:
        mapping(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
