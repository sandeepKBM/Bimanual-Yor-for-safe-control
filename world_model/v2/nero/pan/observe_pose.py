"""pan/observe_pose.py -- the observe_pose skill, built: estimate the covered pan's planar pose (dx, dy, yaw about its nominal centre)
from one depth image, so the claws can be re-aimed at the estimate.  SIMULATION ONLY, CPU rendering (osmesa), no GPU.

1. Render depth (and a segmentation pass) from the virtual head camera used by pan/graspnet_sim.py (azimuth 0, elevation -45,
   0.75 m; the robot's real camera mount is not modelled -- AGENTS.md: the ZED torso mount is a proposal).
2. Depth noise (NOISE levels): i.i.d. Gaussian sigma_1m * z^2 plus a spatially correlated field corr_1m * z^2 (stereo-like
   growth with range), random pixel dropout, and optionally a camera-extrinsic calibration error (the cloud is unprojected with
   a perturbed camera pose).  Then a 5x5 median pre-filter (MEDIAN_K).
3. Keep points above the stove top within REACH of the believed centre; drop the robot's own pixels (self-filter: the robot knows
   its own geometry; this is the only use of the segmentation pass).  The stove/world/counter are removed by the height crop.
4. Shape fit with the known pan geometry: centre = least-squares circle (template radius) through the visible wall points below
   the handle; yaw = principal axis of the handle points (beyond the rim, at handle height).  Both are referenced to the same fit
   on the noiseless nominal scene (the CAD-model render at the known pose), which cancels the fit's own bias.
   (A planar ICP was tried first: its trimming dropped the handle points, so yaw was unobservable -- 10-29 smoke.)

Output: dict(dx, dy, yaw_deg, rms_wall_mm, n_wall, n_handle, n_points, fit_ok).  The pose convention matches under_rim._perturb_pose:
p_true = Rz(yaw) (p_nom - c) + c + (dx, dy, 0), c = the pan's nominal centre.
"""
from __future__ import annotations

import os

import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter

from .graspnet_sim import CAM

NOISE = {   # ASSUMED synthetic noise model (not a real camera; 'zed' = ~1 % of range, the ZED-class datasheet figure). Per level: i.i.d. sigma at 1 m range [m] and a spatially CORRELATED sigma at 1 m (Gaussian-smoothed field, corr_px
            # correlation length; stereo block matching gives correlated error that a median filter cannot remove); both scale with
            # z^2; pixel dropout fraction; camera-extrinsic calibration error (translation m, rotation deg).
    "clean": dict(sigma_1m=0.0, corr_1m=0.0, corr_px=0, drop=0.0, calib_m=0.0, calib_deg=0.0),
    "low": dict(sigma_1m=0.002, corr_1m=0.0, corr_px=0, drop=0.02, calib_m=0.0, calib_deg=0.0),
    "mid": dict(sigma_1m=0.005, corr_1m=0.002, corr_px=15, drop=0.05, calib_m=0.0, calib_deg=0.0),
    "zed": dict(sigma_1m=0.010, corr_1m=0.005, corr_px=15, drop=0.10, calib_m=0.0, calib_deg=0.0),   # ~1 % of range (ZED-class)
    "zed_calib": dict(sigma_1m=0.010, corr_1m=0.005, corr_px=15, drop=0.10, calib_m=0.003, calib_deg=0.3),
}
REACH = 0.30          # m, keep points within this xy distance of the believed pan centre (pan r 0.096 + handle 0.13 + margin)
Z_ABOVE = 0.004       # m, keep points this far above the pan base (removes the stove top)
MIN_WALL, MIN_HANDLE = 300, 100   # design: fewer visible wall / handle points -> no fit (look again)
MEDIAN_K = 5          # k x k median pre-filter on every depth image (template included)
MAX_RMSE = 0.006      # design: fit_ok requires the wall points' RMS distance to the fitted circle below this [m]


def _rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def render(m, d, lookat, noise="clean", rng=None, renderer=None):
    """-> world-frame points (N,3) of the non-robot scene inside the camera view, with the noise model applied."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    nz = NOISE[noise] if isinstance(noise, str) else noise
    rng = np.random.default_rng() if rng is None else rng
    own = renderer is None
    r = mujoco.Renderer(m, CAM["height"], CAM["width"]) if own else renderer
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    depth = r.render().copy()
    gl = r.scene.camera[0]
    pos, fwd, up = (np.array(v, float) for v in (gl.pos, gl.forward, gl.up))
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    seg = r.render().copy()
    r.disable_segmentation_rendering()
    if own:
        r.close()
    fwd /= np.linalg.norm(fwd)
    up /= np.linalg.norm(up)
    R_wc = np.stack([np.cross(fwd, up), -up, fwd], 1)            # OpenCV camera axes in world (as graspnet_sim.render_cloud)
    fovy = np.radians(m.vis.global_.fovy)
    f = 0.5 * CAM["height"] / np.tan(0.5 * fovy)
    z = depth.astype(float)
    if nz["sigma_1m"] > 0:
        z = z + rng.normal(size=z.shape) * nz["sigma_1m"] * z ** 2
    if nz["corr_1m"] > 0:
        fld = gaussian_filter(rng.normal(size=z.shape), nz["corr_px"])
        z = z + fld / fld.std() * nz["corr_1m"] * z ** 2
    if nz["drop"] > 0:
        z[rng.random(z.shape) < nz["drop"]] = np.nan
    if nz["calib_m"] > 0 or nz["calib_deg"] > 0:                  # the robot unprojects with a slightly wrong camera pose
        R_wc = _rot(rng.normal(size=3), nz["calib_deg"]) @ R_wc
        dirn = rng.normal(size=3)
        pos = pos + nz["calib_m"] * dirn / np.linalg.norm(dirn)
    if MEDIAN_K > 1:
        z = _nanmedian_filter(z, MEDIAN_K)
    v, u = np.mgrid[0:CAM["height"], 0:CAM["width"]]
    pc = np.stack([(u - 0.5 * CAM["width"] + 0.5) * z / f, (v - 0.5 * CAM["height"] + 0.5) * z / f, z], -1).reshape(-1, 3)
    robot = _robot_mask(m, seg).ravel()
    ok = np.isfinite(pc).all(1) & (pc[:, 2] > 0.05) & (pc[:, 2] < 3.0) & ~robot
    return pc[ok] @ R_wc.T + pos


def _nanmedian_filter(z, k):
    """k x k median of the valid pixels (dropouts are NaN); a pixel with no valid neighbours stays NaN."""
    h = k // 2
    zp = np.pad(z, h, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(zp, (k, k))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win.reshape(z.shape[0], z.shape[1], -1), axis=-1)


def _robot_mask(m, seg):
    """pixels on the robot's own bodies (everything that is not the world body or a free object: pan, lid, hidden)."""
    gid, typ = seg[..., 0], seg[..., 1]
    is_geom = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
    body = np.where(is_geom, m.geom_bodyid[np.clip(gid, 0, m.ngeom - 1)], 0)
    free = {0} | {m.body(n).id for n in ("pan", "lid", "hidden") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
    return is_geom & ~np.isin(body, list(free))


def crop(pw, centre_xy, base_z):
    keep = (np.linalg.norm(pw[:, :2] - centre_xy, axis=1) < REACH) & (pw[:, 2] > base_z + Z_ABOVE) & (pw[:, 2] < base_z + 0.20)
    return pw[keep]


def _bands(p):
    base = p.pan_base_z()
    rim = p.pan_bottom_t + p.pan_wall_h                                  # rim height above the pan base
    return dict(wall=(base + 0.006, base + rim - p.handle_t - 0.0025),   # wall only: below the handle (handle top is flush with the rim)
                handle=(base + rim - p.handle_t - 0.001, base + rim + 0.002),
                r_out=p.pan_radius + p.pan_wall_t)


def _tukey(res):
    """Tukey biweight weights, scale from the MAD (c = 4.685); the robust fits use them in IRLS."""
    sc = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
    u = res / (4.685 * sc)
    return np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)


def _circle(xy, c0, r=None, iters=40):
    """robust circle centre (radius fixed if r is given, else fitted): Gauss-Newton with Tukey IRLS weights, from c0.
    Returns (centre, radius, RMS of the inlier residuals)."""
    c = np.asarray(c0, float).copy()
    rr = float(np.linalg.norm(xy - c, axis=1).mean()) if r is None else r
    w = np.ones(len(xy))
    for k in range(iters):
        v = xy - c
        dist = np.linalg.norm(v, axis=1) + 1e-12
        res = dist - rr
        if k >= 3:                                   # plain LS for a few steps, then down-weight outliers
            w = _tukey(res)
        J = -v / dist[:, None]
        if r is None:
            J = np.c_[J, -np.ones(len(xy))]
        sw = np.sqrt(w)[:, None]
        step = np.linalg.lstsq(J * sw, -res * sw[:, 0], rcond=None)[0]
        c += step[:2]
        if r is None:
            rr += step[2]
        if np.linalg.norm(step[:2]) < 1e-8 and k >= 3:
            break
    res = np.linalg.norm(xy - c, axis=1) - rr
    inl = w > 0
    return c, float(rr), float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else np.inf


def _fit(pw, p, c_guess, r_wall=None):
    """shape fit: centre = circle fit to the visible wall below the handle; yaw = principal axis of the handle points."""
    bd = _bands(p)
    w = pw[(pw[:, 2] > bd["wall"][0]) & (pw[:, 2] < bd["wall"][1])]
    w = w[np.linalg.norm(w[:, :2] - c_guess, axis=1) < bd["r_out"] + 0.03]
    if len(w) < MIN_WALL:
        return None
    c, r, rms_w = _circle(w[:, :2], c_guess)                   # free radius first (robust start), then the template radius
    if r_wall is not None:
        c, r, rms_w = _circle(w[:, :2], c, r_wall)
    h = pw[(pw[:, 2] > bd["handle"][0]) & (pw[:, 2] < bd["handle"][1])]
    rad = np.linalg.norm(h[:, :2] - c, axis=1)
    h = h[(rad > bd["r_out"] + 0.012) & (rad < bd["r_out"] + p.handle_len + 0.01)]
    if len(h) < MIN_HANDLE:
        return None
    wh = np.ones(len(h))
    for k in range(10):                                         # weighted PCA line fit with Tukey IRLS on the perpendicular residual
        mu = (wh[:, None] * h[:, :2]).sum(0) / wh.sum()
        hx = h[:, :2] - mu
        evec = np.linalg.eigh((wh[:, None] * hx).T @ hx)[1][:, -1]
        wh = _tukey(hx @ np.array([-evec[1], evec[0]]))
        if wh.sum() < MIN_HANDLE / 2:
            return None
    if evec @ (mu - c) < 0:
        evec = -evec
    return dict(c=c, r=r, ang=float(np.arctan2(evec[1], evec[0])), rms_wall=rms_w, n_wall=int(len(w)), n_handle=int(len(h)))


def template(m, d, p, renderer=None):
    """the same fit on the noiseless NOMINAL settled scene (call before perturbing) = the CAD reference: wall radius as rendered
    (24-sided wall) and the handle angle; cancels the polygon/rendering bias of the shape fit."""
    c = np.asarray(p.pan_centre_xy, float)
    pw = crop(render(m, d, _lookat(p), "clean", renderer=renderer), c, p.pan_base_z())
    ref = _fit(pw, p, c)
    if ref is None:
        raise RuntimeError("observe_pose: template fit failed on the nominal scene")
    return ref


def _lookat(p):
    return np.array([p.pan_centre_xy[0], p.pan_centre_xy[1], p.pan_base_z() + 0.05])


def estimate(m, d, p, tmpl, noise="clean", rng=None, renderer=None):
    """observe_pose: one depth image -> planar pose estimate of the pan relative to its nominal pose (fit_ok=False -> look again)."""
    c = np.asarray(p.pan_centre_xy, float)
    obs = crop(render(m, d, _lookat(p), noise, rng, renderer), c, p.pan_base_z())
    f = _fit(obs, p, c, tmpl["r"])
    if f is None:
        return dict(dx=0.0, dy=0.0, yaw_deg=0.0, fit_ok=False, n_points=int(len(obs)), noise=noise)
    yaw = float(np.degrees((f["ang"] - tmpl["ang"] + np.pi) % (2 * np.pi) - np.pi))
    dxy = f["c"] - tmpl["c"]
    return dict(dx=float(dxy[0]), dy=float(dxy[1]), yaw_deg=yaw, rms_wall_mm=1e3 * f["rms_wall"], n_wall=f["n_wall"],
                n_handle=f["n_handle"], n_points=int(len(obs)), fit_ok=bool(f["rms_wall"] < MAX_RMSE), noise=noise)


def true_pose(m, d, p, pan_pos0, pan_R0):
    """measured TRUE planar pose of the pan body vs its nominal settled pose (pos0, R0), same convention as the estimate."""
    c = np.r_[np.asarray(p.pan_centre_xy, float), 0.0]
    R = d.xmat[m.body("pan").id].reshape(3, 3) @ pan_R0.T
    yaw = np.arctan2(R[1, 0], R[0, 0])
    Rz = _rot([0, 0, 1], np.degrees(yaw))
    t = d.xpos[m.body("pan").id] - c - Rz @ (pan_pos0 - c)
    return dict(dx=float(t[0]), dy=float(t[1]), yaw_deg=float(np.degrees(yaw)))


def aim_transform(est, centre_xy):
    """-> f(point) and Rz for re-aiming targets/directions planned on the nominal pose at the ESTIMATED pose."""
    c = np.r_[np.asarray(centre_xy, float), 0.0]
    Rz = _rot([0, 0, 1], est["yaw_deg"])
    off = np.array([est["dx"], est["dy"], 0.0])
    return (lambda q: Rz @ (np.asarray(q, float) - c) + c + off), Rz


def scene_image(path, placement="none"):
    """RGB image of the settled NOMINAL covered-pan scene from the same head camera (the image both proposers see)."""
    import mujoco as mj
    from PIL import Image
    from . import covered as CV, scene as S
    cp = CV.CoveredParams(placement=placement)
    m = CV.compose(cp)
    d = mj.MjData(m)
    S.settle(m, d, 1.0)
    r = mj.Renderer(m, CAM["height"], CAM["width"])
    cam = mj.MjvCamera()
    cam.lookat[:] = _lookat(cp.pan)
    cam.distance, cam.azimuth, cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(path)
    r.close()
    return path
