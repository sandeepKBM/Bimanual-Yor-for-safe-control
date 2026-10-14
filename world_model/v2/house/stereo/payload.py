"""stereo/payload.py -- head-stereo PAYLOAD estimator: tray plane + red cup on the tray from the head ZED depth + left RGB.  SIMULATION ONLY.

Works entirely in the BASE frame (the head extrinsic = mount + lift encoder + pitch is a calibration constant of the robot; no base localisation enters):
  1. ROI = a box around the FK-predicted tray (the hands hold it: ocs.TrayFKProducer / ArmFK.tray); back-project valid depth pixels into the tray-prior frame.
  2. TRAY PLANE: near-plane band (|z_prior| < band) minus the cup mask -> iteratively re-weighted plane fit (Huber); gives the normal n (tray tilt from
     vertical, 2 DOF), the plane offset along n, and a covariance from the LSQ normal equations.
  3. CUP: red pixels (colour threshold on the left RGB: the paper cup is red) with valid depth, inside the ROI, above the plane -> fixed-radius cylinder fit
     (|dist to axis| - R)^2, axis = 4 parameters (base-centre position in the tray plane, tilt about the tray x / y axes), robust (soft-L1), initialised
     upright at the centroid pushed back along the view ray by (pi/4) R.  Output: cup base position IN THE TRAY FRAME + cup tilt relative to the tray, with the
     Gauss-Newton covariance (x floor).
  The cup radius / height are the shared engineering prior of ocs.util (CUP_R, CUP_H); nothing here reads a geom id or a simulator pose.

StereoPayloadProducer writes ocs ObjectStates (Source.STEREO): "tray" (Kind.PAYLOAD, parent "robot") and "cup" (Kind.CONTENT, parent "tray",
dynamics 'slip_on_support'), t_meas = the exposure time of the frame, visible = enough cup pixels in THIS frame (else the last estimate is held with
visible=False and a growing position set).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from world_model.v2.tray_rep.sets.ir import Pose, Unc, BoxShape, ZonoShape
from world_model.v2.tray_rep.sets.zono import Zonotope
from world_model.v2.house.ocs.state import Source, Kind, Belief, ObjectState, Relation, Producer
from world_model.v2.house.ocs import util as U
from .cfg import StereoCfg, head_axes


@dataclass
class PayloadCfg:
    roi_xy: tuple = (U.HX + 0.07, U.HY + 0.07)     # half sizes of the ROI box around the FK tray [m]
    roi_z: tuple = (-0.04, 0.22)                   # tray-frame z range of the ROI
    plane_band: float = 0.012                      # |z_prior| for the tray-plane points (before IRLS)
    extent_band: float = 0.0045                    # |d| from the fitted plane for the in-plane (plate extent) points
    slide_step: float = 0.002
    hw_margin: float = 0.004                       # added to the tray in-plane half-width in the position sets
    plateau_frac: float = 0.97
    cup_dilate: int = 6                            # px: cup mask dilation removed from the plane points
    red_ratio: float = 1.4
    red_min: int = 60
    min_cup_px: int = 120
    min_plane_pts: int = 300
    cup_r: float = 0.032                           # teleop-sim cup (data/contents.py: 12-gon, wall apothem 0.030 + 1.5 mm half-thickness, h 0.08).  NOTE ocs.util.CUP_R/H = 0.045/0.115 is the flagship paper cup
    cup_h: float = 0.080
    sigma_floor_pos: float = 0.003                 # m   position set floors (model mismatch: polygonal cup, translucency)
    sigma_floor_rot: float = 0.02                  # rad
    k_sigma: float = 3.0                           # set = k * sigma (re-fit by the study to cover >= 95 %)
    mu_prior: tuple = (0.18, 0.35)


def _huber_plane(P, iters=6, delta=0.004):
    """weighted plane z = n . p + d fit (normal estimated by SVD with Huber weights).  Returns centroid, unit normal (z-up), residual std, n_pts, cov of the normal (2x2 tilt about x,y)."""
    w = np.ones(len(P))
    for _ in range(iters):
        c = (P * w[:, None]).sum(0) / w.sum()
        Q = (P - c) * np.sqrt(w)[:, None]
        _, S, Vt = np.linalg.svd(Q, full_matrices=False)
        n = Vt[-1]
        if n[2] < 0:
            n = -n
        r = (P - c) @ n
        w = 1.0 / np.maximum(1.0, np.abs(r) / delta)
    sig = float(np.sqrt((w * r ** 2).sum() / max(w.sum() - 3, 1)))
    # tilt covariance: the normal is rotated by angle e about an in-plane axis; var = sig^2 / (sum w * (in-plane coordinate)^2)
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(n, e1)
    s1 = float((w * ((P - c) @ e1) ** 2).sum()); s2 = float((w * ((P - c) @ e2) ** 2).sum())
    cov = np.diag([sig ** 2 / max(s1, 1e-9), sig ** 2 / max(s2, 1e-9)])
    return c, n, sig, len(P), cov, (e1, e2)


def _slide(a, h, step, frac):
    """centre + half-width of the placement of a 2h-wide window (the plate) over 1-D points: the middle of the plateau where the count >= frac * max."""
    if len(a) < 10:
        return 0.0, h
    lo, hi = np.percentile(a, 0.5) - h, np.percentile(a, 99.5) + h
    cs = np.arange(lo, hi + step, step)
    srt = np.sort(a)
    cnt = np.searchsorted(srt, cs + h) - np.searchsorted(srt, cs - h)
    ok = cs[cnt >= frac * cnt.max()]
    return float(0.5 * (ok.min() + ok.max())), float(0.5 * (ok.max() - ok.min()))


def _rot_from_z(n, yaw_ref):
    """rotation whose z axis is n and whose x axis is as close as possible to yaw_ref (the prior x axis)."""
    z = n / np.linalg.norm(n)
    x = yaw_ref - (yaw_ref @ z) * z
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def red_mask(rgb, ratio, vmin):
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return (r > ratio * g + 8) & (r > ratio * b + 8) & (r > vmin)


def estimate(fr, cfg: StereoCfg, tray_fk, lift, pc: PayloadCfg | None = None, rgb=None):
    """-> dict(ok, tray=dict(R, p, tilt, tilt_cov, sig, n), cup=dict(ok, p (tray frame, z=0), tilt (rad, rel tray), cov (3,3 over x,y,tilt-mag proxy), n_px), ms)
    fr: stereo Frame (depth, + left unless rgb given); tray_fk = (p (3,), R (3,3)) FK prior in the BASE frame; lift = lift encoder [m]."""
    pc = pc or PayloadCfg()
    rgb = fr["left"] if rgb is None else rgb
    depth = fr["depth"]
    K = cfg.K()
    Rc = head_axes(cfg)
    o_c = np.asarray(cfg.mount, float) + np.array([0.0, 0.0, lift if cfg.lift_mounted else 0.0])
    oL = o_c - 0.5 * cfg.baseline * Rc[:, 0]
    pt, Rt = tray_fk
    ok = np.isfinite(depth)
    vv, uu = np.nonzero(ok)
    z = depth[vv, uu].astype(np.float64)
    pcam = np.stack([(uu - K["cx"]) / K["f"] * z, -(vv - K["cy"]) / K["f"] * z, -z], 1)
    Pb = pcam @ Rc.T + oL                                           # base frame
    Pt = (Pb - pt) @ Rt                                             # tray-prior frame
    inroi = (np.abs(Pt[:, 0]) < pc.roi_xy[0]) & (np.abs(Pt[:, 1]) < pc.roi_xy[1]) & (Pt[:, 2] > pc.roi_z[0]) & (Pt[:, 2] < pc.roi_z[1])
    red = red_mask(rgb, pc.red_ratio, pc.red_min)
    # cup-mask dilation (px) for excluding cup edge pixels from the plane
    import cv2
    redd = cv2.dilate(red.astype(np.uint8), np.ones((2 * pc.cup_dilate + 1,) * 2, np.uint8)).astype(bool)
    isred = red[vv, uu]
    nearred = redd[vv, uu]
    out = dict(ok=False, tray=None, cup=dict(ok=False, n_px=0))
    # ---- tray plane
    sel = inroi & ~nearred & (np.abs(Pt[:, 2]) < pc.plane_band)
    if sel.sum() >= pc.min_plane_pts:
        c, n_t, sig, n, cov, (e1, e2) = _huber_plane(Pt[sel])
        # second pass: re-select with the fitted plane (handles a tilted tray whose points leave the prior band)
        d_ = (Pt - c) @ n_t
        sel2 = inroi & ~nearred & (np.abs(d_) < pc.plane_band)
        if sel2.sum() >= pc.min_plane_pts:
            c, n_t, sig, n, cov, (e1, e2) = _huber_plane(Pt[sel2])
        n_b = Rt @ n_t                                              # fitted normal in the base frame
        c_b = Rt @ c + pt
        R_est = _rot_from_z(n_b, Rt[:, 0])
        tilt = math.acos(float(np.clip(n_b[2], -1, 1)))             # tray tilt from vertical (base frame is level)
        p_est = pt + float((c_b - pt) @ n_b) * n_b                  # plane offset along the normal is measured
        # IN-PLANE position from the plate points: every plate point lies inside the known 2hx x 2hy rectangle.  Per axis slide a window of the plate
        # width over the (tight-band) points and take the middle of the max-count plateau; half-width = half the plateau width (set-membership: the
        # centre is somewhere in the plateau) + floor.  Occluded / contaminated edges widen the plateau instead of biasing a bare extent.  Yaw = FK prior.
        ex = R_est[:, 0]
        ey = R_est[:, 1]
        selt = inroi & ~nearred & (np.abs((Pt - c) @ n_t) < pc.extent_band)
        Pin = (Pt[selt] @ Rt.T + pt) - p_est
        u_, v_ = Pin @ ex, Pin @ ey
        cu, wu = _slide(u_, U.HX, pc.slide_step, pc.plateau_frac)
        cv, wv = _slide(v_, U.HY, pc.slide_step, pc.plateau_frac)
        p_est = p_est + cu * ex + cv * ey
        out["tray"] = dict(R=R_est, p=p_est, tilt=tilt, tilt_cov=cov, sig=sig, n=int(n), n_prior_dev=math.acos(float(np.clip(n_t[2], -1, 1))),
                           inplane=(float(cu), float(cv)), inplane_hw=(float(wu), float(wv)), n_ext=int(selt.sum()))
        out["ok"] = True
        Rtray, ptray = R_est, p_est
    else:
        Rtray, ptray = Rt, pt
    # ---- cup
    selc = inroi & isred & (Pt[:, 2] > -0.005)
    ncup = int(selc.sum())
    out["cup"]["n_px"] = ncup
    if ncup >= pc.min_cup_px:
        Pc = (Pb[selc] - ptray) @ Rtray                             # tray-estimate frame
        view_dir = (oL - ptray) @ Rtray
        vxy = view_dir[:2] / (np.linalg.norm(view_dir[:2]) + 1e-9)
        c0 = Pc[:, :2].mean(0) - (math.pi / 4) * pc.cup_r * vxy     # centroid of the visible surface -> axis, pushed AWAY from the camera
        x0 = np.array([c0[0], c0[1], 0.0, 0.0])

        def axis_of(x):
            tx, ty = x[2], x[3]
            a = np.array([math.sin(ty) * math.cos(tx), -math.sin(tx), math.cos(ty) * math.cos(tx)])
            return a / np.linalg.norm(a)

        def res(x):
            a = axis_of(x)
            b = np.array([x[0], x[1], 0.0])
            q = Pc - b
            t = q @ a
            rad = np.linalg.norm(q - t[:, None] * a, axis=1)
            r = rad - pc.cup_r
            # only points on the cup body: height along the axis in [-0.01, h + 0.01]
            return np.where((t > -0.01) & (t < pc.cup_h + 0.01), r, 0.0)

        sol = least_squares(res, x0, loss="soft_l1", f_scale=0.004, max_nfev=60)
        x = sol.x
        J = sol.jac
        n_eff = max(ncup / 40.0, 4.0)                               # neighbouring pixels share their stereo error
        s2 = max(float(np.mean(sol.fun[np.abs(sol.fun) > 0] ** 2)) if np.any(sol.fun) else 1e-6, 1e-6)
        try:
            cov = np.linalg.inv(J.T @ J + 1e-9 * np.eye(4)) * s2 * (ncup / n_eff)
        except np.linalg.LinAlgError:
            cov = np.eye(4) * 1e-2
        a = axis_of(x)
        tiltc = math.acos(float(np.clip(a[2], -1, 1)))
        out["cup"] = dict(ok=True, n_px=ncup, p=np.array([x[0], x[1], 0.0]), tilt=tiltc, axis=a, cov=cov, res_rms=float(math.sqrt(s2)), tx=x[2], ty=x[3])
    return out


class StereoPayloadProducer(Producer):
    """ocs Producer: head-stereo tray + cup.  `view` must provide stereo_frame() (the newest DELIVERED frame or None), encoders(), fk (ArmFK), t, k."""
    name = "payload_stereo"

    def __init__(self, cfg: StereoCfg, pcfg: PayloadCfg | None = None, hold_s: float = 0.35, k_sigma: float | None = None):
        self.cfg, self.pc = cfg, pcfg or PayloadCfg()
        self.k = self.pc.k_sigma if k_sigma is None else k_sigma
        self.hold_s = hold_s
        self.reset()

    def reset(self, view=None):
        self.last_cup = None
        self.last_tray = None
        self.first_t = None
        self.n_frames = 0
        self.ms = []

    def update(self, ws, view, t):
        import time
        if self.first_t is None:
            self.first_t = t
        enc = view.encoders()
        fr = view.stereo_frame()
        if fr is not None and fr.get("_seen") is None:
            fr["_seen"] = True
            enc = fr.get("enc") or enc               # the FK prior / lift of the EXPOSURE time, not of the delivery time
            tp = view.fk.tray(enc["qL"], enc["qR"], enc["lift"])
            t0 = time.perf_counter()
            e = estimate(fr, self.cfg, tp, float(enc["lift"]), self.pc)
            self.ms.append(1e3 * (time.perf_counter() - t0))
            self.n_frames += 1
            if e["tray"] is not None:
                self.last_tray = (fr["t_sim"], e["tray"], fr.get("t_deliver", t))
            if e["cup"]["ok"]:
                self.last_cup = (fr["t_sim"], e["cup"], e["tray"], fr.get("t_deliver", t))
        pc = self.pc
        if self.last_tray is not None:
            tt, tr, td = self.last_tray
            sx, sy = (math.sqrt(max(tr["tilt_cov"][i, i], 0.0)) for i in (0, 1))
            rot = self.k * max(sx, sy) + pc.sigma_floor_rot
            ws.objects["tray"] = ObjectState("tray", Kind.PAYLOAD, BoxShape(np.array([U.HX, U.HY, 0.003])), Pose(tr["R"], tr["p"]),
                                             unc=Unc(U.ball_zono(max(self.k * tr["sig"], 0.004, max(tr["inplane_hw"]) + pc.hw_margin)), float(rot)), parent="robot", name="tray", name_source=Source.PRIOR,
                                             dynamics="attached", source=Source.STEREO, t_meas=tt, visible=(t - td) <= self.hold_s, age=t - self.first_t,
                                             extra=dict(tilt=tr["tilt"], plane_sig=tr["sig"], n_pts=tr["n"], note="in-plane position = FK prior, normal + offset = stereo plane"))
        if self.last_cup is not None:
            tc, cu, tr, td = self.last_cup
            vis = (t - td) <= self.hold_s
            cov = cu["cov"]
            sxy = math.sqrt(max(cov[0, 0], cov[1, 1], 0.0))
            r = self.k * sxy + pc.sigma_floor_pos + max(tr["inplane_hw"]) + pc.hw_margin     # rel. to the ESTIMATED tray frame: its in-plane set is part of the cup's
            if not vis:
                r += 0.01 * (t - td)                                  # the cup can slide while unseen: set grows with the unseen time (placeholder rate, conformal-fit later)
            stilt = math.sqrt(max(cov[2, 2], cov[3, 3], 0.0))
            rot = self.k * stilt + pc.sigma_floor_rot
            ax = cu["axis"]
            Rc_t = _rot_from_z(ax, np.array([1.0, 0, 0]))
            mu = Belief(pc.mu_prior[0], pc.mu_prior[1], None, None, Source.PRIOR, "ShieldCfg prior interval")
            ws.objects["cup"] = ObjectState("cup", Kind.CONTENT, ZonoShape(U.cup_shape()), Pose(Rc_t, cu["p"]),
                                            unc=Unc(Zonotope(np.zeros(3), np.diag([r, r, 0.003])), float(rot)), parent="tray", name="cup", name_source=Source.PRIOR,
                                            params={"mu": mu}, dynamics="slip_on_support", source=Source.STEREO, t_meas=tc, visible=vis, age=t - self.first_t,
                                            extra=dict(tilt_rel_tray=cu["tilt"], n_px=cu["n_px"], res_rms=cu["res_rms"], sigma_xy=sxy))
            ws.relations = [r_ for r_ in ws.relations if not (r_.a == "cup" and r_.rel == "on")] + [Relation("cup", "on", "tray", 1.0 if vis else 0.5, Source.STEREO)]
        ws.meta["producers"] = ws.meta.get("producers", []) + ["payload_stereo"]
