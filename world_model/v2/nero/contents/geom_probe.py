"""contents/geom_probe.py -- `measure_box`, the first geometry probe (design: docs/status/geometry_probe_design_2026-11-19.md, sections 3, 5, 7).
SIMULATION ONLY, CPU rendering (osmesa), no GPU, no hardware.

Pipeline: the four lab cameras (USER 11-19; plan_images.room4_cameras, poses known exactly in sim) render depth + a segmentation pass;
depth gets observe_pose's noise model (NOISE table, i.i.d. + correlated field + dropout, then the nan-median filter; imported, not copied);
the robot's own pixels are dropped (self-filter); depth is back-projected with the pinhole model (fovy, image size), the camera position from the camera spec and the axes the
free camera renders with (scene.camera forward / up, OpenCV axes).  NOTE: scene.camera[0].pos is the left stereo eye, 34 mm (ipd/2) off the
image's mono pose; observe_pose.render uses it and so has that 34 mm lateral bias in its points.  measure_box fits an UPRIGHT oriented box to one entity's points.

SEGMENTATION = privileged sim segmentation (ASSUMED stand-in for a real segmenter, as in observe_pose and the FoundationPose bench).
The target's points are chosen by the segmentation pass's geom ids of the named entity (entity_geoms).  The hidden contents body is never
measured: entity_geoms("crate") is the geoms of body "obj" only.

Conventions of measure_box (all MODEL = our choices, not measured):
  * footprint (sx, sy, yaw): minimum-area rectangle (rotating calipers over the convex hull) of the xy points in the TOP BAND, i.e. points with
    z >= (TOP_PCT percentile of z) - BAND_M.  The top band is used (not all points) so that low protrusions (the crate's grasp tabs) and
    side faces seen at a slant do not enlarge the footprint.  yaw is folded into [-pi/4, pi/4) (rotation of the sx axis about +z from +x);
    sx / sy are swapped when the fold changes the axis.  (A box with a near-square footprint has an ill-defined yaw.)
  * top z = the TOP_PCT-th percentile of z (98); bottom z = support_top_z when given (the support is measured separately), else the
    BOTTOM_PCT-th percentile (2) of z over all the entity's points.  sz = top - bottom, cz = their mean, (cx, cy) = centre of the rectangle.
  * fit_ok False (with `why`) when fewer than MIN_POINTS points or fewer than MIN_CAMS cameras see the entity.
Not observable from these cameras: the bottom z of an entity resting on a support (hidden; it needs support_top_z, else the low percentile
is the lowest VISIBLE point), and the yaw of a near-square footprint.  Cameras are at 2.2 m looking down obliquely, so thin vertical entities
(bookcase sides) have their top band small: their footprint is mostly the 2 cm edge.
"""
from __future__ import annotations

import os

import numpy as np

MIN_POINTS, MIN_CAMS = 200, 2          # design (11-19 brief)
TOP_PCT, BOTTOM_PCT = 98.0, 2.0        # MODEL (our choice)
BAND_M = 0.02                          # MODEL (our choice): top band depth for the footprint
DENS_R, DENS_N = 0.015, 15            # MODEL (our choice): density outlier filter on the band xy: drop points with < DENS_N neighbours within DENS_R [m]
ERODE_PX = 3                           # MODEL (our choice): label-boundary erosion window [px] (was 11, which shrank silhouettes ~2.3 cm per side; flying pixels are now also removed by a depth-discontinuity test)
SEGMENTATION = "privileged sim segmentation (ASSUMED stand-in)"
ENTITY_NAMES = ("crate", "counter", "pan_table", "bookcase_l1_floor", "bookcase_l1_base", "bookcase_mid", "bookcase_top",
                "bookcase_side_left", "bookcase_side_right", "bookcase_back", "bookcase_plinth")
_NONROBOT_BODIES = ("obj", "contents", "pan", "lid", "hidden")   # free objects: not the robot, so not self-filtered


def _mj():
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco
    return mujoco


def entity_geoms(m, name):
    """-> sorted geom ids of the named entity.  'crate' = all geoms of body 'obj' (NOT 'contents')."""
    mj = _mj()
    if name == "crate":
        ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("obj").id]
    elif name == "pan_table":
        ids = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("pan_table_top")]
    elif name in ENTITY_NAMES:
        ids = [g for g in range(m.ngeom) if m.geom(g).name == name]
    else:
        raise ValueError(f"unknown entity {name!r}; known {ENTITY_NAMES}")
    if not ids:
        raise ValueError(f"entity {name!r} has no geoms in this model")
    return sorted(ids)


def true_box(m, d, name):
    """(cx, cy, cz, sx, sy, sz, yaw) from the model, for scoring only.  Geom boxes: exact (yaw folded as in measure_box; a single upright box).
    crate: union of the body's wall / floor / lid box geoms (tabs excluded: they are not part of the crate's box), upright, footprint from its walls."""
    mj = _mj()
    ids = entity_geoms(m, name)
    if name == "crate":
        ids = [g for g in ids if not (m.geom(g).name or "").startswith("tab")]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    yaw = 0.0
    for g in ids:
        assert m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX
        R = d.geom_xmat[g].reshape(3, 3)
        he = np.abs(R) @ m.geom_size[g]
        lo, hi = np.minimum(lo, d.geom_xpos[g] - he), np.maximum(hi, d.geom_xpos[g] + he)
    if name == "crate":
        yaw = float(np.arctan2(*d.geom_xmat[ids[0]].reshape(3, 3)[[1, 0], 0]))
        # footprint from the walls in the body frame, then back to world: take the body yaw and the oriented extents of all geoms
        R0 = d.geom_xmat[ids[0]].reshape(3, 3)
        pts = []
        for g in ids:
            for s in np.ndindex(2, 2, 2):
                pts.append(d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (m.geom_size[g] * (2 * np.array(s) - 1)))
        P = np.array(pts) @ R0           # coordinates in the (upright) crate frame
        l, h = P.min(0), P.max(0)
        c = R0 @ (0.5 * (l + h))
        ext = h - l
        sx, sy = ext[0], ext[1]
        return _fold(c[0], c[1], 0.5 * (lo[2] + hi[2]), sx, sy, hi[2] - lo[2], yaw)
    c, s = 0.5 * (lo + hi), hi - lo
    if len(ids) == 1:
        R = d.geom_xmat[ids[0]].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        s = 2 * np.array(m.geom_size[ids[0]])
    return _fold(c[0], c[1], c[2], s[0], s[1], s[2], yaw)


def _fold(cx, cy, cz, sx, sy, sz, yaw):
    k = np.round(yaw / (np.pi / 2))
    y = yaw - k * np.pi / 2
    if int(k) % 2:
        sx, sy = sy, sx
    y = (y + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return dict(cx=float(cx), cy=float(cy), cz=float(cz), sx=float(sx), sy=float(sy), sz=float(sz), yaw=float(y))



# ---------------------------------------------------------------- noise
NOISE_LEVELS = ("clean", "zed2i_lab", "low", "mid", "zed", "zed_calib")   # + the observe_pose levels
SIGMA_D_PX = 0.2          # ASSUMED (11-19, user approved): stereo disparity noise [px]
BASELINE_M = 0.12         # ZED 2i datasheet stereo baseline [m]
LAB_CORR_FRAC = 0.5       # ASSUMED (11-19, user approved): correlated-field sigma = this * sigma_z
LAB_CORR_PX_720 = 15      # ASSUMED (11-19, user approved): correlation length [px] at 720p, scaled with the image height
LAB_DROP = 0.05           # ASSUMED (11-19, user approved): pixel dropout
LAB_CALIB_M, LAB_CALIB_DEG = 0.003, 0.1   # ASSUMED (11-19, user approved): per-camera extrinsic error, drawn once per camera per draw
MED_M, MED_K = 0.001, 3.0     # MODEL (our choice): a pixel whose median-filtered depth differs from its own unfiltered depth by more than MED_M + MED_K * sigma_z keeps the unfiltered depth (at a convex edge or corner the 5x5 median returns a neighbouring surface's depth, up to ~2 mm off the surface at 4 m; on a plane the median equals the centre value)
DISC_M, DISC_K = 0.05, 3.0   # MODEL (our choice): flying-pixel test: a pixel is dropped when its 3x3 depth range > DISC_M + DISC_K * sigma_z


def sigma_z(z, f_px):
    """zed2i_lab per-pixel depth sigma at range z [m]: z^2 * SIGMA_D_PX / (f_px * BASELINE_M)."""
    return z ** 2 * SIGMA_D_PX / (f_px * BASELINE_M)


def _rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _apply_noise(depth, noise, rng, f_px):
    """-> (z with noise + nan-median filter, calib_m, calib_deg, sigma_img or None, z before the median filter).
    'zed2i_lab': sigma_z(z) i.i.d. + a Gaussian-smoothed correlated field of LAB_CORR_FRAC * sigma_z + LAB_DROP dropout (see the constants).
    other names / dicts: observe_pose.NOISE (sigma_1m, corr_1m z^2 laws)."""
    from scipy.ndimage import gaussian_filter
    from world_model.v2.nero.pan import observe_pose as OP
    z = depth.astype(float)
    h = z.shape[0]
    if noise == "zed2i_lab":
        sg = sigma_z(z, f_px)
        z = z + rng.normal(size=z.shape) * sg
        fld = gaussian_filter(rng.normal(size=z.shape), LAB_CORR_PX_720 * h / 720.0)
        z = z + fld / fld.std() * LAB_CORR_FRAC * sg
        z[rng.random(z.shape) < LAB_DROP] = np.nan
        cm, cd = LAB_CALIB_M, LAB_CALIB_DEG
    else:
        nz = OP.NOISE[noise] if isinstance(noise, str) else noise
        sg = nz["sigma_1m"] * z ** 2 if (nz["sigma_1m"] > 0 or nz["corr_1m"] > 0) else None
        if nz["sigma_1m"] > 0:
            z = z + rng.normal(size=z.shape) * nz["sigma_1m"] * z ** 2
        if nz["corr_1m"] > 0:
            fld = gaussian_filter(rng.normal(size=z.shape), nz["corr_px"])
            z = z + fld / fld.std() * nz["corr_1m"] * z ** 2
        if nz["drop"] > 0:
            z[rng.random(z.shape) < nz["drop"]] = np.nan
        cm, cd = nz.get("calib_m", 0.0), nz.get("calib_deg", 0.0)
    z_raw = z
    if OP.MEDIAN_K > 1:
        z = OP._nanmedian_filter(z, OP.MEDIAN_K)
    return z, cm, cd, sg, z_raw


def _robot_mask(m, seg):
    """pixels on the robot's own bodies (everything but the world body and the free objects obj / contents / pan / lid / hidden)."""
    mj = _mj()
    gid, typ = seg[..., 0], seg[..., 1]
    is_geom = (typ == int(mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
    body = np.where(is_geom, m.geom_bodyid[np.clip(gid, 0, m.ngeom - 1)], 0)
    free = {0} | {m.body(n).id for n in _NONROBOT_BODIES if mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) >= 0}
    return is_geom & ~np.isin(body, list(free))




OCC_PX = 4   # MODEL (our choice): occlusion-edge search radius [px]


def _occluder_gid(gimg, z, tol):
    """-> (H, W) int: per pixel, the geom id of the NEAREST pixel within OCC_PX px that has a different geom id (>= 0) and a depth smaller than this pixel's by more than tol
    (an occluder next to this pixel); -1 when none.  A pixel with such a neighbour lies on an OCCLUSION edge, not on the entity's own silhouette.
    (OCC_PX covers the pixels removed by the label erosion and the median filter, so the visible edge row is still adjacent to its occluder.)"""
    from scipy.ndimage import minimum_filter, maximum_filter
    h, w = gimg.shape
    r = OCC_PX
    out = np.full((h, w), -1, np.int32)
    cand = (minimum_filter(gimg, size=2 * r + 1, mode="nearest") != maximum_filter(gimg, size=2 * r + 1, mode="nearest")) & (gimg >= 0)
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return out
    zc = np.where(np.isfinite(z), z, np.inf)
    zp = np.pad(zc, r, constant_values=np.inf)
    gp = np.pad(gimg, r, constant_values=-1)
    g0 = gimg[ys, xs]
    lim = zc[ys, xs] - (tol[ys, xs] if np.ndim(tol) else tol)
    best = np.full(len(ys), np.inf)
    og = np.full(len(ys), -1, np.int32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            zn = zp[ys + r + dy, xs + r + dx]
            gn = gp[ys + r + dy, xs + r + dx]
            mk = (gn >= 0) & (gn != g0) & (zn < lim) & (zn < best)
            best = np.where(mk, zn, best)
            og = np.where(mk, gn, og)
    out[ys, xs] = og
    return out


def lab_clouds(m, d, world, noise="clean", rng=None, size=None):
    """-> [dict(label, points (N,3) world, geom_ids (N,) int, cam_pos (3,), sigma_m)], one per room4 camera.
    Depth noise = 'zed2i_lab' (this module) or observe_pose.NOISE[noise] (a name or a dict); NOISE_LEVELS.  The back-projection uses the camera pose the software believes:
    the true pose perturbed by the level's calibration error (once per camera per call).  Robot pixels dropped.  Pixels with no geom (sky), depth outside (0.05, 6) m
    or a depth discontinuity (flying pixel) dropped.  sigma_m = median per-pixel depth sigma of the kept points (0 for clean).
    m.vis.global_.fovy is set to ROOM_FOVY_DEG (as plan_images.render_world does for room4).  size default = plan_images.room_size()."""
    mj = _mj()
    from scipy.ndimage import minimum_filter, maximum_filter
    from world_model.v2.nero.contents import plan_images as PI
    rng = np.random.default_rng() if rng is None else rng
    w, h = PI.room_size() if size is None else size
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, m.vis.global_.offwidth), max(h, m.vis.global_.offheight)
    m.vis.global_.fovy = PI.ROOM_FOVY_DEG
    f = 0.5 * h / np.tan(0.5 * np.radians(PI.ROOM_FOVY_DEG))
    v, u = np.mgrid[0:h, 0:w]
    out = []
    m.vis.quality.offsamples = 0     # no multisampling for the depth / segmentation passes: on EGL the model default (4) resolves MSAA depth by averaging: +0.9 mm bias on every surface and up to -10 mm at edges (crate room_back 3.8 m: 2.6-3.9 mm outside the surface); osmesa has no MSAA; with 0 both renderers match the ray-cast depth to < 0.2 mm
    r = mj.Renderer(m, h, w)
    try:
        for c in PI.room4_cameras(m, d, world):
            cam = PI._look_free_camera(c["pos"], c["look_at"])
            r.enable_depth_rendering()
            r.update_scene(d, camera=cam)
            depth = r.render().copy()
            gl = r.scene.camera[0]
            fwd, up = (np.array(x, float) for x in (gl.forward, gl.up))
            pos = np.array(c["pos"], float)    # the camera pose from the spec (scene.camera[0].pos is the LEFT stereo eye, 34 mm off the mono image pose)
            r.disable_depth_rendering()
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=cam)
            seg = r.render().copy()
            r.disable_segmentation_rendering()
            fwd /= np.linalg.norm(fwd)
            up /= np.linalg.norm(up)
            R_wc = np.stack([np.cross(fwd, up), -up, fwd], 1)        # OpenCV camera axes (x right, y down, z forward) in world
            z, cm, cd, sg, z_raw = _apply_noise(depth, noise, rng, f)
            bad = np.abs(z - z_raw) > MED_M + MED_K * (0.0 if sg is None else np.nan_to_num(sg))   # the median took this pixel's depth from a neighbouring surface (convex edge / corner); NaN raw (dropout) keeps the median
            z = np.where(bad, z_raw, z)                              # such a pixel keeps its own (unfiltered) depth
            if cm > 0 or cd > 0:                                     # the software back-projects with a slightly wrong camera pose
                R_wc = _rot(rng.normal(size=3), cd) @ R_wc
                dirn = rng.normal(size=3)
                pos = pos + cm * dirn / np.linalg.norm(dirn)
            pc = np.stack([(u - 0.5 * w + 0.5) * z / f, (v - 0.5 * h + 0.5) * z / f, z], -1).reshape(-1, 3)
            gimg = np.where(seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
            k = ERODE_PX
            uni = np.ones(gimg.shape, bool)
            if k > 1:                                               # label-boundary erosion (small): a silhouette pixel may carry another surface's depth
                uni = (minimum_filter(gimg, size=k, mode="nearest") == maximum_filter(gimg, size=k, mode="nearest"))
            zf = np.where(np.isfinite(z), z, np.inf)
            zg = np.where(np.isfinite(z), z, -np.inf)
            rng3 = maximum_filter(zg, size=3, mode="nearest") - minimum_filter(zf, size=3, mode="nearest")   # inf where a neighbourhood has no valid pixel
            lim = DISC_M + DISC_K * (0.0 if sg is None else np.nan_to_num(sg))
            smooth = (rng3 <= lim)                                    # depth-discontinuity test (flying pixels between surfaces)
            gid = gimg.ravel()
            occ = _occluder_gid(gimg, z, TOL_BASE_M + TOL_SIGMA_K * (0.0 if sg is None else np.nan_to_num(sg))).ravel()
            ok = (np.isfinite(pc).all(1) & (pc[:, 2] > 0.05) & (pc[:, 2] < 6.0) & (gid >= 0) & uni.ravel() & smooth.ravel() & ~_robot_mask(m, seg).ravel())
            smed = 0.0 if sg is None else float(np.median(np.broadcast_to(sg, z.shape).ravel()[ok])) if ok.any() else 0.0
            out.append(dict(label=c["label"], points=pc[ok] @ R_wc.T + pos, geom_ids=gid[ok].astype(int), occ_gid=occ[ok].astype(int), cam_pos=pos, sigma_m=smed))
    finally:
        r.close()
    return out


# ---------------------------------------------------------------- box fit from faces
MIN_FACE_PTS = 60         # MODEL (our choice): a face needs this many inlier points (per camera)
TOL_BASE_M, TOL_SIGMA_K = 0.004, 1.5   # MODEL (our choice): plane inlier tolerance = TOL_BASE_M + TOL_SIGMA_K * sigma_m
REL_PEAK = 0.10           # MODEL (our choice): a plane peak counts when its count >= REL_PEAK * the camera's best peak of that kind
LINE_ITERS, SUBSAMPLE = 300, 20000     # MODEL: RANSAC for vertical-face lines (xy) on a random subsample, then refit on all
YAW_GATE_DEG = 8.0        # MODEL: a per-camera face direction further than this (mod 90 deg) from the heaviest one is dropped
OCC_FRAC = 0.5   # MODEL (our choice, from world 2 clean: occluded top faces 0.58-1.0, genuine unoccluded ones <= 0.35): a face peak in a camera is OCCLUDED when this fraction of its points has an occluder within OCC_PX px
OCC_FRAC_SIDE = 0.9   # MODEL (our choice): the same for a vertical face.  Higher than OCC_FRAC: a plane offset fitted to a PARTLY occluded vertical face is unbiased (a cutoff cannot make a vertical-plane peak: the width test rejects the strip), counter y-ends have 0.51-0.57 flagged and are measured to 0.7 mm; only a face that lies almost entirely on an occlusion edge is a bound
W_MIN_SIGMA_K_SIDE = 7.0  # MODEL (our choice): the same for a VERTICAL face.  The tangent p5-p95 spread of a perpendicular wide face's points that fall in one plane window is pure noise: measured zed2i_lab 2.6-5.2 sigma_m (n=1000s of points, world 21 / world 10 cameras), so 4 sigma accepted noise strips as faces (counter sy 0.57 m, back sy 1.4 m); a real face is >= 0.2 m wide (>= 8 sigma at 3.8 m, sigma 2.4 cm)
W_MIN_BASE_M, W_MIN_SIGMA_K = 0.008, 4.0   # MODEL (our choice): a face / top plane needs an in-plane width >= W_MIN_BASE_M + W_MIN_SIGMA_K * sigma_m (rejects strips of a perpendicular plane; thin edges narrower than this are not faces)
INTERIOR_MIN = 0.1    # MODEL (our choice): a horizontal plane peak needs this fraction of its points > tol inside its convex hull (see _interior_frac); fake row = 0.0, real tops 0.3-0.65 (worlds 0-9 clean, noisy zed2i_lab examples)
TOP_AGREE_M = 0.012   # MODEL: top_z is observed only when >= 2 cameras measured it and agree within this [m]: two cameras' LAB_CALIB error (3 mm + 0.1 deg at 4 m = 7 mm each, rss 10 mm) + 2 mm.  A top that is hidden (an occluder's edge row looks like a plane: w0 plinth, 15.5 mm low) is a different height in each camera
CENTRE_FREE_M = 0.001     # MODEL (our choice): a world centre coordinate may omit the other family's pair only when the omitted term is below this [m] in the worst case (see measure_box)


def _interior_frac(xy, tol):
    """fraction of the xy points farther than tol INSIDE their convex hull.  A real horizontal face has an interior (0.3-0.65 for a 0.2 x 0.28 crate top at 3.8 m / a 0.4 x 0.3 counter top);
    the "plane" cut out of two vertical faces by one occlusion-edge row lies along two edges of its own hull (0.0)."""
    from scipy.spatial import ConvexHull
    try:
        eq = ConvexHull(xy).equations
    except Exception:
        return 0.0
    dist = -(xy @ eq[:, :2].T + eq[:, 2]).max(1)
    return float((dist > tol).mean())


def _peaks(x, aux, tol, minpts, rel, wmin, flag=None):
    """1-D plane offsets: -> [(offset, n_inliers)] by descending count; offset = mean of the inliers within tol of a window peak (three LS refinements); a peak is dropped
    when its count < max(minpts, rel * best count) or its in-plane WIDTH < wmin.  aux = per-point in-plane coordinates ((N,) tangent for a vertical face, (N,2) xy for a
    horizontal one); width = p5-p95 range of the tangent / of the minor PCA axis of the inliers.  The width test rejects a thin strip cut out of a perpendicular plane (not a face).
    Rejected peaks are removed like accepted ones.  flag (bool per point) -> each peak also returns the number of flagged inliers (occlusion-edge pixels)."""
    x = np.asarray(x, float)
    out = []
    rem = x
    remaux = aux
    remflag = np.zeros(len(x), bool) if flag is None else np.asarray(flag, bool)
    best = None
    for _ in range(12):
        if len(rem) < minpts:
            break
        lo, hi = rem.min(), rem.max()
        nb = max(int(np.ceil((hi - lo) / (0.5 * tol))), 1)
        h, e = np.histogram(rem, bins=nb, range=(lo, lo + nb * 0.5 * tol))
        k = max(int(round(2 * tol / (0.5 * tol))), 1)               # window +-tol
        cs = np.convolve(h, np.ones(k + 1), mode="same")
        j = int(np.argmax(cs))
        wlo, whi = e[max(j - k // 2, 0)], e[min(j + k // 2 + 1, len(e) - 1)]
        c = float(np.mean(rem[(rem >= wlo) & (rem <= whi)]))         # centre of the points in the winning window (NOT the bin centre: a plane at the histogram edge, within 2 bins of the minimum, was 5 mm off the bin centre and lost all its points)
        for _ in range(3):
            sel = np.abs(rem - c) < tol
            if sel.sum() == 0:
                break
            c = float(np.average(rem[sel]))
        sel = np.abs(rem - c) < tol
        n = int(sel.sum())
        if n < minpts:
            break
        a = remaux[sel]
        if a.ndim == 1:
            wd = float(np.percentile(a, 95) - np.percentile(a, 5))
        else:
            ev = np.linalg.eigvalsh(np.cov((a - a.mean(0)).T))
            wd = float(3.29 * np.sqrt(max(ev[0], 0.0)))      # p5-p95 span of a Gaussian
        if wd >= wmin:
            if best is None:
                best = n
            if n < rel * best:
                break
            out.append((c, n, int(remflag[sel].sum())))
        rem, remaux, remflag = rem[~sel], remaux[~sel], remflag[~sel]
    return out


def _ransac_line(xy, tol, rng):
    """-> (theta of the line direction mod pi, normal offset, inlier mask) by RANSAC (subsample) + PCA refit; None if no line."""
    n = len(xy)
    sub = xy if n <= SUBSAMPLE else xy[rng.choice(n, SUBSAMPLE, replace=False)]
    best, bi = 0, None
    for _ in range(LINE_ITERS):
        i, j = rng.integers(0, len(sub), 2)
        dv = sub[j] - sub[i]
        L = np.hypot(*dv)
        if L < 0.05:
            continue
        nrm = np.array([-dv[1], dv[0]]) / L
        c = int((np.abs((sub - sub[i]) @ nrm) < tol).sum())
        if c > best:
            best, bi = c, (sub[i].copy(), nrm)
    if bi is None:
        return None
    p0, nrm = bi
    for _ in range(3):
        sel = np.abs((xy - p0) @ nrm) < tol
        if sel.sum() < 3:
            return None
        q = xy[sel]
        mu = q.mean(0)
        w, V = np.linalg.eigh(np.cov((q - mu).T))
        nrm = V[:, 0]
        p0 = mu
    sel = np.abs((xy - p0) @ nrm) < tol
    return float(np.arctan2(nrm[1], nrm[0])), sel


def _circ_mean_90(ang, wt):
    """weighted mean of angles modulo pi/2 -> value in [-pi/4, pi/4)."""
    z = np.sum(wt * np.exp(4j * np.asarray(ang)))
    return float((np.angle(z) / 4 + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def _wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _fit_faces(per, support_top_z, rng):
    """per: [(points (N,3), cam_pos, sigma_m)] of one entity.  -> (faces list, top (z, n, cams, rng) or None, yaw or None)."""
    tops, lines = [], []
    sides = []
    for ci, (P, cp, sg, occ) in enumerate(per):
        if len(P) < 20:
            sides.append(None)
            continue
        tol = TOL_BASE_M + TOL_SIGMA_K * sg
        rg = float(np.median(np.linalg.norm(P - cp, axis=1)))
        pk = _peaks(P[:, 2], P[:, :2], tol, MIN_FACE_PTS, 0.0, W_MIN_BASE_M + W_MIN_SIGMA_K * sg, occ)   # horizontal planes (2-D footprint wider than a strip)
        top = max(pk) if pk else None                                   # the highest
        if top is not None and _interior_frac(P[np.abs(P[:, 2] - top[0]) < tol, :2], tol) >= INTERIOR_MIN:
            tops.append((top[0], top[1], ci, rg, top[2]))        # credible top face; a plane band WITHOUT an interior (a thin strip or an occlusion-edge row) is still excluded from `side` below but is not reported as a top
        side = P if top is None else P[np.abs(P[:, 2] - top[0]) >= tol]
        sides.append((side, tol, rg, occ if top is None else occ[np.abs(P[:, 2] - top[0]) >= tol]))
        xy = side[:, :2]
        rem = np.ones(len(xy), bool)
        found = []
        for _ in range(3):
            if rem.sum() < MIN_FACE_PTS:
                break
            res = _ransac_line(xy[rem], tol, rng)
            if res is None:
                break
            th, sel = res
            n = int(sel.sum())
            if n < MIN_FACE_PTS or (found and n < 0.1 * found[0][1]):
                break
            found.append((th, n))
            idx = np.flatnonzero(rem)
            rem[idx[sel]] = False
        if found:
            th0 = found[0][0]
            for th, n in found:
                if abs(_wrap90(th - th0)) <= np.radians(YAW_GATE_DEG):
                    lines.append((th, n, ci, rg))
    if not lines:
        return [], tops, None, sides
    ang = np.array([l[0] for l in lines])
    wt = np.array([l[1] / l[3] ** 2 for l in lines])
    yaw = _circ_mean_90(ang, wt)
    keep = np.array([abs(_wrap90(a - yaw)) <= np.radians(YAW_GATE_DEG) for a in ang])
    if keep.any() and not keep.all():
        yaw = _circ_mean_90(ang[keep], wt[keep])
    return lines, tops, yaw, sides


def measure_box(clouds, target_geom_ids, support_top_z=None, rng=None):
    """fit an upright oriented box to one entity from PLANES (design 11-19 phase 2).  Per camera: the top face (highest horizontal plane), the vertical side faces
    (RANSAC lines in xy); yaw = the side-face direction (merged over cameras, weight n_points * range^-2); with the yaw fixed, per camera plane offsets of the 4 signed
    side normals (outermost per signed normal; the camera side gives the outward sign); cameras merged by the same weight.
    -> dict(fields={f: dict(value, observed, faces_used)}, faces=[dict(kind, normal, offset, n_points, cams)], n_points, n_cams, fit_ok, why, segmentation, conventions)
    + flat keys cx cy cz sx sy sz yaw (NaN when not observed).  fields: cx cy cz sx sy sz yaw top_z.
    A size or centre coordinate needs BOTH opposite faces of its family; a world cx / cy also needs the other family's pair unless |sin yaw| * (largest |coordinate| of the points) < CENTRE_FREE_M (cx = cu cos - cv sin,
    cy = cu sin + cv cos); sz / cz need the top face and support_top_z (the bottom is hidden: UNOBSERVED without it).  A single face is a one-sided bound: see `faces`
    (kind side: normal is the outward unit vector in xy, offset = normal . p; kind top: z).  yaw in [-pi/4, pi/4); the local x axis is (cos yaw, sin yaw)."""
    rng = np.random.default_rng(0) if rng is None else rng
    ids = np.asarray(sorted(set(int(g) for g in target_geom_ids)))
    per, nraw = [], []
    for c in clouds:
        P = c["points"][np.isin(c["geom_ids"], ids)]
        og = np.asarray(c.get("occ_gid", np.full(len(c["geom_ids"]), -1)))[np.isin(c["geom_ids"], ids)]
        per.append((P, np.asarray(c["cam_pos"], float), float(c.get("sigma_m", 0.0)), (og >= 0) & ~np.isin(og, ids)))
        nraw.append(len(P))
    n_cams = sum(n >= 20 for n in nraw)
    npts = int(sum(nraw))
    nan = float("nan")
    fields = {f: dict(value=nan, observed=False, faces_used=[]) for f in ("cx", "cy", "cz", "sx", "sy", "sz", "yaw", "top_z")}
    res = dict(fields=fields, faces=[], n_points=npts, n_cams=int(n_cams), fit_ok=False, why="", segmentation=SEGMENTATION,
               conventions=("box from planes: top = highest horizontal plane; side faces = vertical planes (RANSAC lines), yaw = side-face direction; "
                            "size / centre need both opposite faces; bottom = support_top_z else UNOBSERVED; yaw in [-pi/4, pi/4)"),
               **{k: nan for k in ("cx", "cy", "cz", "sx", "sy", "sz", "yaw")})
    if npts < MIN_POINTS:
        res["why"] = f"{npts} points < {MIN_POINTS}"
        return res
    if n_cams < MIN_CAMS:
        res["why"] = f"seen by {n_cams} camera(s) < {MIN_CAMS}"
        return res
    lines, tops, yaw, sides = _fit_faces(per, support_top_z, rng)
    faces, why = [], []
    top_z = None
    tops_ok = [t for t in tops if t[4] < OCC_FRAC * t[1]]
    tops_occ = [t for t in tops if t[4] >= OCC_FRAC * t[1]]
    if tops_ok:                                   # the largest set of cameras whose top heights agree; a single camera's top is not observed (cannot be told from an occlusion-edge row)
        best = max(([u for u in tops_ok if abs(u[0] - t[0]) <= TOP_AGREE_M] for t in tops_ok), key=lambda g: (len({u[2] for u in g}), sum(u[1] for u in g)))
        if len({u[2] for u in best}) >= 2:
            tops_ok = best
        else:
            tops_occ = tops_occ + tops_ok
            tops_ok = []
            why.append("top height seen by one camera only: not observed")
    if tops_ok:
        wt = np.array([t[1] / t[3] ** 2 for t in tops_ok])
        top_z = float(np.average([t[0] for t in tops_ok], weights=wt))
        fields["top_z"].update(value=top_z, observed=True, faces_used=["top"])
        faces.append(dict(kind="top", normal=[0.0, 0.0, 1.0], offset=top_z, n_points=int(sum(t[1] for t in tops_ok)), cams=sorted({int(t[2]) for t in tops_ok})))
    elif tops_occ:
        wt = np.array([t[1] / t[3] ** 2 for t in tops_occ])
        bz = float(np.average([t[0] for t in tops_occ], weights=wt))
        faces.append(dict(kind="bound", normal=[0.0, 0.0, 1.0], offset=bz, n_points=int(sum(t[1] for t in tops_occ)), cams=sorted({int(t[2]) for t in tops_occ}),
                          what="top_z >= offset: the visible top region ends at an occlusion edge"))
        why.append("top face ends at an occlusion edge: top_z is a one-sided bound")
    else:
        why.append("no top face")
    fam = {}       # (k, sign) -> (offset, n, cams)
    if yaw is None:
        why.append("no vertical face")
    else:
        fields["yaw"].update(value=yaw, observed=True, faces_used=["side_direction"])
        nrm = [np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])]
        cand = {(k, s): [] for k in (0, 1) for s in (1, -1)}
        sg_of = [p[2] for p in per]
        for ci, sd in enumerate(sides):
            if sd is None:
                continue
            side, tol, rg, occs = sd
            cp = per[ci][1]
            for k in (0, 1):
                u = side[:, :2] @ nrm[k]
                tg = side[:, :2] @ nrm[1 - k]
                for off, n, nf in _peaks(u, tg, tol, MIN_FACE_PTS, REL_PEAK, W_MIN_BASE_M + W_MIN_SIGMA_K_SIDE * sg_of[ci], occs):
                    s = 1 if cp[:2] @ nrm[k] > off else -1               # the camera is on the outward side of the face it sees
                    cand[(k, s)].append((off, n, ci, rg, tol, nf))
        for key, cs0 in cand.items():
            cs = [c for c in cs0 if c[5] < OCC_FRAC_SIDE * c[1]]
            cso = [c for c in cs0 if c[5] >= OCC_FRAC_SIDE * c[1]]
            if cso:                                                        # peaks that end at an occlusion edge: one-sided bounds only
                ob = max(cso, key=lambda c: key[1] * c[0])
                n2 = nrm[key[0]] * key[1]
                faces.append(dict(kind="bound", normal=[float(n2[0]), float(n2[1])], offset=float(key[1] * ob[0]), n_points=int(ob[1]), cams=[int(ob[2])], family="xy"[key[0]],
                                  what="the box extends at least to this plane: the visible face ends at an occlusion edge"))
            if not cs:
                continue
            s = key[1]
            ext = max(cs, key=lambda c: s * c[0])                          # outermost peak
            grp = [c for c in cs if abs(c[0] - ext[0]) <= 3 * ext[4]]
            wt = np.array([c[1] / c[3] ** 2 for c in grp])
            off = float(np.average([c[0] for c in grp], weights=wt))
            fam[key] = (off, int(sum(c[1] for c in grp)), sorted({int(c[2]) for c in grp}))
            n2 = nrm[key[0]] * s
            faces.append(dict(kind="side", normal=[float(n2[0]), float(n2[1])], offset=float(s * off), n_points=fam[key][1], cams=fam[key][2], family="xy"[key[0]]))
    pair = {}
    for k, nm in ((0, "x"), (1, "y")):
        if (k, 1) in fam and (k, -1) in fam:
            hi, lo = fam[(k, 1)][0], fam[(k, -1)][0]
            pair[k] = (hi - lo, 0.5 * (hi + lo))
            fields["s" + nm].update(value=hi - lo, observed=True, faces_used=[f"{nm}+", f"{nm}-"])
        else:
            why.append(f"{nm}-pair: " + ("none" if not any((k, s) in fam for s in (1, -1)) else "one face only"))
    if yaw is not None:
        c, s_ = np.cos(yaw), np.sin(yaw)
        allP = np.concatenate([p[0][:, :2] for p in per if len(p[0])])
        mu_, mv_ = (float(np.abs(allP @ nrm[k]).max()) for k in (0, 1))     # largest |coordinate| of the entity's points along the two face normals
        cu = pair[0][1] if 0 in pair else None
        cv = pair[1][1] if 1 in pair else None
        # a world centre coordinate is cu cos - cv sin (cx) / cu sin + cv cos (cy): the missing family's term is dropped only when its worst case |sin yaw| * (largest |coordinate|)
        # is below CENTRE_FREE_M (it is NOT free for a board far from the origin: |sin yaw| 0.017 at 2.2 m is 39 mm)
        if cu is not None and (cv is not None or abs(s_) * mv_ < CENTRE_FREE_M):
            fields["cx"].update(value=cu * c - (cv or 0.0) * s_, observed=True, faces_used=["x+", "x-"] + (["y+", "y-"] if cv is not None else []))
        if cv is not None and (cu is not None or abs(s_) * mu_ < CENTRE_FREE_M):
            fields["cy"].update(value=(cu or 0.0) * s_ + cv * c, observed=True, faces_used=["y+", "y-"] + (["x+", "x-"] if cu is not None else []))
    if top_z is not None and support_top_z is not None:
        fields["sz"].update(value=top_z - float(support_top_z), observed=True, faces_used=["top", "support_top_z"])
        fields["cz"].update(value=0.5 * (top_z + float(support_top_z)), observed=True, faces_used=["top", "support_top_z"])
    elif top_z is not None:
        why.append("bottom hidden: sz / cz need support_top_z")
    res["faces"] = faces
    for f, fd in fields.items():
        if f != "top_z" and fd["observed"]:
            res[f] = float(fd["value"])
    res["fit_ok"] = any(fd["observed"] for fd in fields.values())
    res["why"] = "ok" if all(fd["observed"] for f, fd in fields.items() if f != "top_z" and (support_top_z is not None or f not in ("sz", "cz"))) else "; ".join(why)
    return res
