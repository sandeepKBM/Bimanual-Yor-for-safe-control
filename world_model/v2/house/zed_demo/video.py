"""video.py -- compose the 2x2 demo video (1920x1080, 25 fps) from the perception result, the plan and the recorded executions.  SIMULATION ONLY.

panels:  a = left ZED image (+ cluster overlay, zonotope wireframes + labels)    b = disparity (SGBM) with fill %
         c = point cloud (numpy orbit viewer) with planes / clusters / zonotopes / plan    d = third-person MuJoCo render with translucent zonotope geoms
"""
from __future__ import annotations

import math
import time

import numpy as np
import cv2
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from . import stereo as ST
from . import perceive as PC
from . import viz as VZ
from . import execute as EX

W, H, FPS = 1920, 1080, 25
PW, PH = 960, 540
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(sz, bold=True):
    try:
        return ImageFont.truetype(FONT if bold else FONT_R, sz)
    except Exception:
        return ImageFont.load_default()


F_BANNER, F_CAP, F_HUD, F_SMALL = font(30), font(19), font(21), font(16, False)
F_TITLE, F_H2, F_TXT = font(54), font(30), font(25, False)


class Ctx:
    """everything the renderer needs (filled by run.py)."""
    pass


# ------------------------------------------------------------------------------------------------ panel renderers
def fill_metrics(disp, seg, cls, num_disp=192):
    ok = np.isfinite(disp)
    rm = PC.robot_mask(seg, cls, dilate_px=9)
    act = np.zeros_like(ok)
    act[:, num_disp:] = True
    sc = act & ~rm
    return float(ok[act].mean()), float(ok[sc].mean())


def disp_panel(disp, fill, wipe=1.0, title=None, dmax=None):
    img = ST.colormap_disp(disp, dmax)
    img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, "no match", (8, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1, cv2.LINE_AA)
    cv2.putText(img, "(<192 px)", (8, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1, cv2.LINE_AA)
    if wipe < 1.0:
        k = int(PW * wipe)
        img[:, k:] = (14, 16, 22)
        cv2.line(img, (k, 0), (k, PH), (255, 255, 255), 2)
    return img


def depth_legend(img, d0, d1, K):
    """small colour bar: disparity px and the corresponding depth Z = f B / d."""
    bar = np.linspace(1, 0, 150)[:, None] * np.ones((1, 14))
    bar = cv2.applyColorMap((bar * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
    x0, y0 = PW - 60, 330
    img[y0:y0 + 150, x0:x0 + 14] = bar
    for frac, d in ((0, d1), (0.5, 0.5 * (d0 + d1)), (1, d0)):
        z = K["f"] * K["B"] / max(d, 1e-3)
        yy = y0 + int((1 - frac) * 149)
        cv2.putText(img, f"{d:.0f}px/{z:.2f}m", (x0 - 98, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def left_panel(ctx, left_full, origin, R, stage, zonos=None, cluster_overlay=None, scale=PW / 1280.0):
    img = cv2.resize(left_full, (PW, PH), interpolation=cv2.INTER_AREA).copy()
    K = ctx.K
    if cluster_overlay is not None:
        ov = cluster_overlay
        img = np.where(ov.any(2, keepdims=True), (0.45 * img + 0.55 * ov).astype(np.uint8), img)
    if stage >= 6 and zonos:
        for zd in zonos:
            VZ.draw_zono_image(img, zd["z"]["Zi"], K, origin, R, zd["color"], scale, 1)
        for zd in zonos:
            VZ.label_image(img, zd["label"], zd["z"]["Zi"], K, origin, R, zd["color"], scale)
    return img


def scene_panel(ctx, dv, stage, zonos, path_world=None, naive_world=None, spot=None, extra=None, cam=None):
    r3 = ctx.r3
    r3.update_scene(dv, camera=cam or ctx.cam3)
    scn = r3.scene
    if stage >= 6 and zonos:
        for zd in zonos:
            VZ.add_zono_scene(scn, zd["z"]["Zi"], zd["color"])
        VZ.add_zono_scene(scn, ctx.support_zi, (230, 210, 120), fill=0.10, width=0.003)
    if path_world is not None:
        VZ.add_polyline(scn, path_world, [0.1, 1.0, 0.35, 0.95], 0.007)
    if naive_world is not None:
        VZ.add_polyline(scn, naive_world, [1.0, 0.15, 0.1, 0.95], 0.007)
    if spot is not None:
        VZ.add_tray_ghost(scn, spot[:2], spot[2], ctx.yaw0 + math.pi / 2, [0.1, 1.0, 0.35, 0.35])
    if extra:
        extra(scn)
    return r3.render().copy()


# ------------------------------------------------------------------------------------------------ cloud panel
def cloud_panel(ctx, az, mode, live=None, trail=None, plan_view=False, naive=False, hit=False):
    cv = ctx.cv
    img = cv.canvas()
    z0 = ctx.z_desk
    cv.axes_floor(img, az, z0, 3.2, 5.2, 4.7, 6.0)
    if mode == "raw":
        cv.points(img, ctx.raw_P, ctx.raw_C, az, 2)
    elif mode == "filtered":
        cv.points(img, ctx.vox_P, ctx.vox_plane_C, az, 2)
    elif mode == "clusters":
        cv.points(img, ctx.vox_P, ctx.vox_cluster_C, az, 2)
    elif mode == "dim":
        cv.points(img, ctx.vox_P, ctx.vox_dim_C, az, 2)
    elif mode == "zono":
        cv.points(img, ctx.vox_P, ctx.vox_cluster_C, az, 2)
    if mode in ("zono", "dim") or plan_view:
        for zd in ctx.zonos:
            cv.zono(img, zd["z"]["Zi"], az, zd["color"], 1)
        cv.zono(img, ctx.support_zi, az, (230, 210, 120), 1)
    if plan_view:
        for p, c in ctx.heat:
            cv.points(img, p, c, az, 4)
        cv.marker(img, ctx.spot_xyz, az, (60, 255, 120), 9, "spot")
        cv.polyline(img, ctx.path_world, az, (60, 255, 120), 2)
    if naive:
        cv.polyline(img, ctx.naive_world, az, (255, 70, 60), 2)
    if trail is not None and len(trail) > 1:
        cv.polyline(img, trail, az, (255, 70, 60) if hit else (60, 255, 120), 2)
    if live is not None:
        cv.marker(img, live, az, (255, 70, 60) if hit else (255, 255, 255), 8)
    return img


# ------------------------------------------------------------------------------------------------ composition
def fit_font(dr, text, maxw, sizes, bold=True):
    for sz in sizes:
        f = font(sz, bold)
        if dr.textlength(text, font=f) <= maxw:
            return f
    return font(sizes[-1], bold)


def draw_hud(canvas, banner, stage_i, n_stages, caps, bottom=None, note=None, alert=None):
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, W, 52], fill=(0, 0, 0, 175))
    dr.text((16, 10), banner, font=fit_font(dr, banner, W - 16 - 260, (30, 28, 26, 24, 22, 20)), fill=(255, 255, 255))
    for i in range(n_stages):                                      # progress dots
        x = W - 30 - (n_stages - 1 - i) * 26
        dr.ellipse([x - 8, 18, x + 8, 34], fill=(70, 220, 120, 255) if i <= stage_i else (90, 90, 100, 200))
    for (x, y), t in zip(((0, 52), (PW, 52), (0, 52 + PH - 52 + 0), (PW, 52 + PH - 52 + 0)), caps):
        pass
    pos = ((10, 58), (PW + 10, 58), (10, PH + 8), (PW + 10, PH + 8))
    for (x, y), t in zip(pos, caps):
        tw = dr.textlength(t, font=F_CAP)
        dr.rectangle([x - 4, y - 2, x + tw + 8, y + 26], fill=(0, 0, 0, 150))
        dr.text((x + 2, y), t, font=F_CAP, fill=(235, 235, 235))
    dr.line([(PW, 0), (PW, H)], fill=(255, 255, 255, 120), width=2)
    dr.line([(0, PH), (W, PH)], fill=(255, 255, 255, 120), width=2)
    if bottom:
        dr.rectangle([0, H - 40, W, H], fill=(0, 0, 0, 185))
        aw = (dr.textlength(alert, font=F_HUD) + 40) if alert else 0
        dr.text((14, H - 34), bottom, font=fit_font(dr, bottom, W - 28 - aw, (21, 20, 19, 18, 17, 16, 15)), fill=(255, 255, 255))
    if note:
        dr.rectangle([0, H - 66, W, H - 40], fill=(0, 0, 0, 140))
        dr.text((14, H - 63), note, font=F_SMALL, fill=(215, 215, 215))
    if alert:
        tw = dr.textlength(alert, font=F_HUD)
        dr.rectangle([W - tw - 40, H - 40, W, H], fill=(200, 30, 30, 235))
        dr.text((W - tw - 20, H - 34), alert, font=F_HUD, fill=(255, 255, 255))
    return np.asarray(im)


def compose(a, b, c, d):
    out = np.empty((H, W, 3), np.uint8)
    out[:PH, :PW], out[:PH, PW:], out[PH:, :PW], out[PH:, PW:] = a, b, c, d
    return out


def card(lines, title, sub=None, thumbs=None, foot=None, bg=(10, 13, 22), txt_font=None):
    img = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(img, "RGBA")
    for y in range(H):                                                       # subtle vertical gradient
        t = y / H
        dr.line([(0, y), (W, y)], fill=(int(bg[0] + 18 * t), int(bg[1] + 22 * t), int(bg[2] + 40 * t)))
    dr.text((90, 70), title, font=F_TITLE, fill=(255, 255, 255))
    if sub:
        dr.text((92, 140), sub, font=F_H2, fill=(150, 200, 255))
    y = 200
    for ln in lines:
        if isinstance(ln, tuple):
            txt, colr = ln
        else:
            txt, colr = ln, (225, 228, 235)
        dr.text((100, y), txt, font=fit_font(dr, txt, W - 140, (25, 24, 23, 22, 21, 20, 19, 18, 17), False), fill=colr)
        y += 36
    if thumbs:
        for (x, yy, im_) in thumbs:
            img.paste(Image.fromarray(im_), (x, yy))
    if foot:
        dr.text((92, 178), foot, font=F_SMALL, fill=(170, 175, 190))
    return np.asarray(img)


def pipeline_card(ctx, frame_i, n):
    """intro: pipeline boxes revealed left to right."""
    img = Image.new("RGB", (W, H), (9, 12, 22))
    dr = ImageDraw.Draw(img, "RGBA")
    for y in range(H):
        t = y / H
        dr.line([(0, y), (W, y)], fill=(int(9 + 16 * t), int(12 + 20 * t), int(22 + 40 * t)))
    dr.text((90, 60), "Stereo camera -> zonotopes -> action", font=F_TITLE, fill=(255, 255, 255))
    dr.text((92, 135), "SIMULATION: YOR bimanual robot, simulated ZED 2 stereo head camera, MuJoCo office scene (nothing here touches hardware)",
            font=F_SMALL, fill=(150, 200, 255))
    steps = ["ZED 2 stereo pair\n(rendered, noisy)", "SGBM matching\ndisparity", "depth Z = f B / d\npoint cloud", "robot self-filter\nRANSAC planes",
             "DBSCAN\nsegments", "zonotope / object\n(order 3) + inflation", "free-spot search\n+ path on zonotopes", "FL-MPC carries the tray\n(arms + welds)"]
    bw, bh, gap = 206, 150, 20
    x0 = (W - (len(steps) * bw + (len(steps) - 1) * gap)) // 2
    reveal = frame_i / max(n - 1, 1) * (len(steps) + 0.5)
    for i, s in enumerate(steps):
        x = x0 + i * (bw + gap)
        y = 330
        on = reveal > i
        col = (60, 200, 120, 255) if on else (70, 75, 90, 255)
        dr.rounded_rectangle([x, y, x + bw, y + bh], radius=14, fill=(24, 34, 52, 255), outline=col, width=3)
        dr.text((x + 10, y + 8), str(i + 1), font=F_H2, fill=col)
        dr.multiline_text((x + 8, y + 58), s, font=font(16), fill=(240, 240, 245) if on else (130, 135, 150), spacing=4)
        if i < len(steps) - 1:
            dr.polygon([(x + bw + 6, y + bh // 2 - 10), (x + bw + gap - 6, y + bh // 2), (x + bw + 6, y + bh // 2 + 10)], fill=col)
    lines = ["Task: place the tray on the cluttered desk at a free spot found ONLY from the stereo-derived zonotopes.",
             "Ground truth (MuJoCo depth / segmentation / geom names) is used for evaluation and label text only, never as an input.",
             "Assumptions: baseline 0.12 m (ZED 2 nominal), HD720, fovy 70 deg, head mount = proposed torso/mast mount (confirm on hardware);",
             "scene textures are procedural speckle (the repo's office scene is texture-free, which would make block matching fail)."]
    y = 560
    for ln in lines:
        dr.text((100, y), ln, font=F_TXT, fill=(215, 220, 232))
        y += 44
    return np.asarray(img)


# ------------------------------------------------------------------------------------------------ timeline
def _noisy_left(ctx, dv, cam_id, rng_seed=None):
    left = ctx.rig._render(dv, cam_id)
    return ZR_noise(left, ctx.rig.rng)


def ZR_noise(img, rng):
    from .rig import add_sensor_noise
    return add_sensor_noise(img, rng)


def restore_state(ctx, snap):
    EX.restore(ctx.m, ctx.dv, snap)


def cluster_overlay(ctx, mode):
    """(PH, PW, 3) overlay of the segmentation result in the left panel."""
    res = ctx.res
    ov = np.zeros((PH, PW, 3), np.uint8)
    sc = PW / ctx.K["W"]
    Wd = ctx.K["W"]

    def put(pix, colors):
        v = (pix // Wd * sc).astype(int)
        u = (pix % Wd * sc).astype(int)
        ok = (v >= 0) & (v < PH - 2) & (u >= 0) & (u < PW - 2)
        ov[v[ok], u[ok]] = np.asarray(colors)[ok]

    if mode == "filter":
        pl = res["plane_lab"]
        pix = res["pix_voxel"]
        for i in range(len(res["planes"])):
            put(pix[pl == i], np.tile((90, 110, 160) if res["planes"][i]["kind"] == "horizontal" else (110, 110, 120), (int((pl == i).sum()), 1)))
        ov = cv2.dilate(ov, np.ones((3, 3), np.uint8))
        rm = cv2.resize(res["rmask"].astype(np.uint8), (PW, PH), interpolation=cv2.INTER_NEAREST).astype(bool)
        ov[rm] = (230, 60, 60)
    else:
        pix = res["pix_voxel"][res["keep"]]
        lab = res["lab_obj"]
        sel = lab >= 0
        cols = np.array([ctx.cluster_color[int(l)] for l in lab[sel]]) if sel.any() else np.zeros((0, 3))
        put(pix[sel], cols)
        ov = cv2.dilate(ov, np.ones((4, 4), np.uint8))
    return ov


def render_video(ctx, out_path, log=print):
    t_start = time.time()
    ctx.keep = {}
    wr = imageio.get_writer(str(out_path), fps=FPS, codec="libx264", quality=None, macro_block_size=None,
                            ffmpeg_params=["-crf", "19", "-preset", "medium", "-pix_fmt", "yuv420p"])
    K = ctx.K
    nst = 9
    nfr = [0]

    def emit(frame, tag=None):
        wr.append_data(frame)
        nfr[0] += 1
        if tag:
            imageio.imwrite(str(ctx.frames_dir / f"{tag}.png"), frame)
            ctx.keep[tag] = frame

    # ---- intro
    n = int(3.2 * FPS) if not getattr(ctx, 'quick', False) else 8
    for i in range(n):
        emit(pipeline_card(ctx, i, n))
    # ---- perception stages on the standing robot
    restore_state(ctx, ctx.snap0)
    origin0, R0 = ctx.origin0, ctx.R0
    left0 = ctx.left0
    right0 = ctx.right0
    ev = ctx.res["ev_scene"]
    T = ctx.res["timing"]

    def az_of(t):
        return -28.0 + 56.0 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 18.0))

    t_glob = [0.0]
    b_img_cache = {}

    def static_frame(stage, n_fr, banner, caps, make_a, make_b, make_c, make_d_kw, bottom, note=None):
        if getattr(ctx, 'quick', False):
            n_fr = min(n_fr, 6)
        for i in range(n_fr):
            fr = i / max(n_fr - 1, 1)
            az = az_of(t_glob[0])
            ctx.cam3.azimuth = 62.0 + 0.0 * az + 0.25 * az
            a = make_a(fr)
            b = make_b(fr)
            c = make_c(fr, az)
            d_ = scene_panel(ctx, ctx.dv, stage, ctx.zonos if stage >= 6 else None, **make_d_kw(fr))
            emit(draw_hud(compose(a, b, c, d_), banner, stage - 1, nst, caps, bottom, note), f"stage{stage}" if i == n_fr - 2 else None)
            t_glob[0] += 1.0 / FPS

    nz = len(ctx.zonos)
    cap_a = "a  LEFT ZED IMAGE (rectified, parallel axes)"
    cap_b = "b  DISPARITY (SGBM, left-right check)"
    cap_c = "c  POINT CLOUD (world frame) - orbiting"
    cap_d = "d  THIRD PERSON (MuJoCo)"
    empty_c = lambda: (ctx.cv.canvas())

    def txt_img(base, msg):
        im = base.copy()
        cv2.putText(im, msg, (300, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 150, 170), 2, cv2.LINE_AA)
        return im

    left_p = lambda stage, ov=None, zon=None: left_panel(ctx, left0, origin0, R0, stage, zon, ov)
    right_p = cv2.resize(right0, (PW, PH), interpolation=cv2.INTER_AREA)
    b_disp = lambda wipe: disp_panel(ctx.res["disp"], ctx.fill_active, wipe)

    def b_with_text(wipe):
        im = b_disp(wipe)
        cv2.putText(im, f"valid disparity: {100 * ctx.fill_scene:.0f}% of scene px (excl. robot body)", (14, PH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        depth_legend(im, ctx.dmin_show, ctx.dmax_show, K)
        return im

    no_d = lambda fr: {}
    # stage 1
    static_frame(1, int(2.0 * FPS), "1   STEREO PAIR rendered from the head mount (+ mild sensor noise)", (cap_a, "b  RIGHT ZED IMAGE (baseline 0.12 m)", cap_c, cap_d),
                 lambda fr: left_p(1), lambda fr: right_p, lambda fr, az: txt_img(empty_c(), "(no depth yet)"), no_d,
                 f"{ctx.K['W']}x{ctx.K['H']}  fovy 70 deg  f = {K['f']:.0f} px  B = {K['B']:.2f} m  -> disparity d = f B / Z  (0.5 m: {K['f'] * K['B'] / 0.5:.0f} px, 1.5 m: {K['f'] * K['B'] / 1.5:.0f} px)",
                 "Scene textures: procedural speckle added in the demo's own copy of the spec (the office scene is flat-coloured); robot links stay untextured")
    # stage 2
    static_frame(2, int(2.4 * FPS), "2   cv2.StereoSGBM  ->  disparity  ->  depth  Z = f B / d", (cap_a, cap_b, cap_c, cap_d),
                 lambda fr: left_p(2), lambda fr: b_with_text(min(1.0, fr * 2.0)), lambda fr, az: txt_img(empty_c(), "(no cloud yet)"), no_d,
                 f"SGBM {T['sgbm_s']:.2f} s (1 thread)   fill {100 * ctx.fill_scene:.0f}% of scene pixels   depth error vs MuJoCo GT: median |e| {1000 * ev['median_abs']:.1f} mm, p95 {1000 * ev['p95_abs']:.1f} mm, RMSE {1000 * ev['rmse']:.0f} mm (outliers)")
    # stage 3
    static_frame(3, int(2.4 * FPS), "3   BACK-PROJECT TO A WORLD-FRAME POINT CLOUD (pose = odometry + assumed mount)", (cap_a, cap_b, cap_c, cap_d),
                 lambda fr: left_p(3), lambda fr: b_with_text(1.0), lambda fr, az: cloud_panel(ctx, az, "raw"), no_d,
                 f"{ctx.res['n_raw_cloud']:,} stereo points in the full view (robot body masked out); colour = left image; ROI crop follows in step 4")
    # stage 4
    ov4 = cluster_overlay(ctx, "filter")
    static_frame(4, int(2.4 * FPS), "4   SELF-FILTER (robot body, red) + RANSAC PLANES removed (blue/grey)", (cap_a, cap_b, cap_c, cap_d),
                 lambda fr: left_p(4, ov4), lambda fr: b_with_text(1.0), lambda fr, az: cloud_panel(ctx, az, "filtered"), no_d,
                 f"ROI crop + 4 mm voxels + radius outlier filter: {ctx.res['timing']['n_after_filters']:,} pts;  desk-top plane z = {ctx.z_desk:.3f} m;  non-plane pts {int(ctx.res['keep'].sum()):,}")
    # stage 5
    ov5 = cluster_overlay(ctx, "clusters")
    static_frame(5, int(2.4 * FPS), "5   DBSCAN SEGMENTATION OF THE REMAINING POINTS (eps 18 mm)", (cap_a, cap_b, cap_c, cap_d),
                 lambda fr: left_p(5, ov5), lambda fr: b_with_text(1.0), lambda fr, az: cloud_panel(ctx, az, "clusters"), no_d,
                 f"{ctx.n_clusters} clusters;  GT eval (never an input): {ctx.rec_n}/{ctx.rec_total} GT objects recovered (IoU>=0.5)")
    # stage 6
    def a6(fr):
        k = max(1, int(round(fr * 1.6 * nz)))
        return left_p(6, None, ctx.zonos[:k])

    static_frame(6, int(3.0 * FPS), "6   ZONOTOPE PER OBJECT (order 3) + INFLATION by the measured stereo error", (cap_a + " + zonotopes", cap_b, cap_c + " + zonotopes", cap_d + " + zonotope geoms"),
                 a6, lambda fr: b_with_text(1.0), lambda fr, az: cloud_panel(ctx, az, "zono"), no_d,
                 f"{nz} zonotopes drawn (inflation {1000 * ctx.zonos[0]['z']['delta']:.0f} mm at 1 m); containment of GT collision geoms in the inflated sets: {100 * ctx.contain_task:.0f}% (task slab z<1.05 m).  Labels: Zn + GT name (display only)")
    # stage 7
    def d7(fr):
        return dict(path_world=ctx.path_world, spot=ctx.spot_xyz_yaw)

    static_frame(7, int(3.2 * FPS), "7   PLAN ON THE ZONOTOPES: max-clearance free spot + path outside the inflated sets", (cap_a + " + zonotopes", cap_b, "c  CLEARANCE OF THE PAYLOAD AT EACH TRAY SPOT (dots)", cap_d + " + plan"),
                 lambda fr: left_p(7, None, ctx.zonos), lambda fr: b_with_text(1.0), lambda fr, az: cloud_panel(ctx, az, "dim", plan_view=True), d7,
                 f"chosen spot: tray centre ({ctx.spot_xyz[0]:.2f}, {ctx.spot_xyz[1]:.2f}) m, zonotope clearance {1000 * ctx.plan_c_spot:.0f} mm (incl. inflation);  path {ctx.path_len:.2f} m;  naive desk-centre target would have clearance {1000 * ctx.naive_c:.0f} mm")
    log(f"[video] perception stages done, {nfr[0]} frames, {time.time() - t_start:.0f}s")

    # ---- live segments (execution + counterfactual)
    def live_segment(frames, stage, banner, path_world, naive_world, naive=False, spot=None):
        # trim the initial rest
        tray0 = frames[0]["tray"]
        k0 = 0
        for i, f in enumerate(frames):
            if np.linalg.norm(f["tray"][:2] - tray0[:2]) > 0.003:
                k0 = max(i - 6, 0)
                break
        fr_ = frames[k0:]
        if getattr(ctx, 'quick', False):
            fr_ = fr_[::12]
        trail = []
        disp_live = [ctx.res["disp"]]
        fill_live = [ctx.fill_scene]
        hit_ever = False
        hit_prev = [False]
        for i, f in enumerate(fr_):
            restore_state(ctx, f["snap"])
            origin, R = ctx.rig.left_pose(ctx.dv)
            if i % 5 == 0:
                L = _noisy_left(ctx, ctx.dv, ctx.rig.cam_l)
                Rr = ZR_noise(ctx.rig._render(ctx.dv, ctx.rig.cam_r), ctx.rig.rng)
                dsp, _ = ST.compute_disparity(L, Rr, ctx.sgbm)
                seg = ctx.rig._render(ctx.dv, ctx.rig.cam_l, seg=True)
                disp_live[0] = dsp
                fill_live[0] = fill_metrics(dsp, seg, ctx.cls)[1]
                left_cur = L
            else:
                left_cur = _noisy_left(ctx, ctx.dv, ctx.rig.cam_l)
            a = left_panel(ctx, left_cur, origin, R, 7, ctx.zonos)
            b = disp_panel(disp_live[0], 0, 1.0)
            cv2.putText(b, f"live disparity (5 Hz)  valid {100 * fill_live[0]:.0f}% of scene px", (14, PH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            hit = f["d"] < 0.0
            hit_ever = hit_ever or hit
            trail.append(np.array([f["tray"][0], f["tray"][1], 0.86]))
            az = az_of(t_glob[0])
            c = cloud_panel(ctx, az, "dim", live=trail[-1], trail=np.array(trail), naive=naive, hit=hit_ever)
            if naive:
                c = cloud_panel(ctx, az, "dim", live=trail[-1], trail=np.array(trail), naive=True, hit=hit_ever)
            ctx.cam3.azimuth = 62.0 + 0.25 * az
            d_ = scene_panel(ctx, ctx.dv, 7, ctx.zonos, path_world=None if naive else path_world, naive_world=naive_world if naive else None, spot=spot)
            dmm = 1000 * f["d"]
            bottom = (f"t = {f['t'] - fr_[0]['t']:4.1f} s    true min clearance (MuJoCo geom distance) = {dmm:6.0f} mm   contacts so far: {f['ncon']}    tray tilt {f['tilt']:.1f} deg"
                      + ("" if naive else f"    ball {'kept' if f['ball_ok'] else 'LOST'}"))
            alert = "COLLISION (naive plan)" if (naive and hit) else None
            caps = (cap_a + " + zonotopes", "b  LIVE DISPARITY (stereo refreshed at 5 Hz)", "c  CLOUD (planning snapshot) + live tray", cap_d + (" (red = naive path)" if naive else " (green = planned path)"))
            tag = None
            if not naive and (i == len(fr_) // 2 or i == len(fr_) - 1):
                tag = f"exec_{i}"
            if naive and ((hit and not hit_prev[0]) or i == len(fr_) - 1):
                tag = f"naive_{i}"
            hit_prev[0] = hit_ever
            emit(draw_hud(compose(a, b, c, d_), banner, stage - 1, nst, caps, bottom, None, alert), tag)
            t_glob[0] += 1.0 / FPS
            if (i + 1) % 50 == 0:
                log(f"[video] {banner[:10]} frame {i + 1}/{len(fr_)}  total {nfr[0]}  {time.time() - t_start:.0f}s")
        return fr_

    live_segment(ctx.frames_exec, 8, "8   EXECUTE: FL-MPC carry along the planned path (stereo keeps running)", ctx.path_world, None, False, ctx.spot_xyz_yaw)
    live_segment(ctx.frames_naive, 9, "9   COUNTERFACTUAL: naive straight line to the desk centre (kinematic replay)", None, ctx.naive_world, True,
                 ctx.naive_spot_yaw)
    # ---- end card
    ek = [k for k in ctx.keep if k.startswith("exec_")][-1]
    nk = [k for k in ctx.keep if k.startswith("naive_")]
    th = []
    for x, key in ((90, ek), (1030, nk[0] if nk else ek)):
        th.append((x, 618, cv2.resize(ctx.keep[key], (800, 450), interpolation=cv2.INTER_AREA)))
    end = ctx.end_card_fn(th)
    for i in range(int(6.0 * FPS) if not getattr(ctx, 'quick', False) else 3):
        emit(end, "end_card" if i == 0 else None)
    wr.close()
    log(f"[video] wrote {out_path}  {nfr[0]} frames = {nfr[0] / FPS:.1f}s  render {time.time() - t_start:.0f}s")
    return nfr[0]
