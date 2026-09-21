"""scripts/make_safety_demo.py

Renders artifacts/safety_demo.mp4: a 4-segment demo video with an on-screen
safety HUD (title / risk-or-margin meter / ALERT banner / outcome label),
built ONLY from real signals already produced by this repo's existing eval
harness and controllers. This script does not modify any physics/control
code in eval/, world_model/, or experiments/ -- it only imports those
modules (exactly like eval/run_eval.py and world_model/run_wm_eval_harness.py
already do) and adds a render + HUD-compositing loop around them.

Segments (each captured every CAPTURE_STRIDE physics steps -> video fps =
1/(DT*CAPTURE_STRIDE), i.e. real-time playback):

  1. SPILL baseline       -- eval config "none" (no semantic safety) driving
     the LEFT arm's fast cup tilt-transfer (eval/scene_setup.py). A SHADOW
     RealWMSpillMonitor (the SAME real trained model as segment 2) runs in
     parallel purely to draw the P(spill) meter -- its output is NEVER wired
     to control here, exactly matching what "no semantic safety" means.
     Expected outcome (verified by eval/RESULTS_DOES_WM_HELP.md, config
     "none": spill%=100/8): the cup spills.

  2. SPILL WM-protected   -- eval config "semantic_only": the REAL trained WM
     monitor (world_model/wm_semantic_monitor.RealWMSpillMonitor,
     wm_cup_full/best.pt + monitor_head_cup_full.pt, threshold 0.5) IS wired
     to the transfer-hold veto, via the same monkeypatch technique
     world_model/run_wm_eval_harness.py uses (safety_module.SEMANTIC_REGISTRY
     ["oracle"] -> RealWMSpillMonitor). Expected outcome (semantic_only:
     spill%=0/8): no spill, task completes.

  3. COLLISION adversarial -- experiments/ab_smooth_vs_bangbang/run_ab4.py
     condition "A" (open-loop, non-cooperative intruder; ego already runs its
     own HOCBF-QP in every condition, matching run_ab4's own docstring).
     Verified outcome (logs/v4_summary.json): collision=True, box_dropped=True.

  4. COLLISION cooperative -- run_ab4.py condition "C" (cooperative,
     ego-priority intruder). Verified outcome (logs/v4_summary.json):
     collision=False, box_dropped=False, min_h=+0.016 (never crosses zero).

HUD signals are all real telemetry already computed by the imported code:
  - P(spill) meter (segments 1-2): RealWMSpillMonitor.assess(...).risk
    (sigmoid output of the frozen MonitorHead on the frozen WM's latent).
  - veto flag (segment 2 alert banner): RealWMSpillMonitor.assess(...).veto
    (risk >= 0.5), same boolean that drives the transfer-hold in
    eval/run_eval.py's run_episode.
  - arm-arm collision margin `h` meter (segments 3-4): run_ab4.run_condition's
    own per-step log["h"] (sc.all_pair_barriers' worst-pair value), the exact
    quantity the HOCBF-QP safety filter constrains.
  - spilled / collision / box_dropped outcome flags: scripts/cup_task_utils.
    count_spilled, eval/scene_setup.left_right_contact, run_ab4's own
    collision/box_drift_trans logging.

The ONLY non-measured, chosen-for-visualization constant is ENGAGE_H_M below
(the collision-margin value below which segments 3-4's HUD shows the
"engaged" banner) -- picked by inspecting the real logged h(t) traces (see
this file's inline comment at ENGAGE_H_M) so the banner's on-window roughly
tracks when the intruder is actually close, not to manufacture a result.

Run (FOREGROUND, no nohup/background -- this is rendering, not training):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \\
        /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
        /common/users/ss5772/bimanual/scripts/make_safety_demo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
AB4_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(REPO_ROOT), str(AB4_DIR), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import)
from scripts import cup_task_utils as ct  # noqa: E402  (read-only import)
import scene_setup  # noqa: E402  (eval/, read-only import)
import safety_module as sm  # noqa: E402  (eval/, read-only import)
import run_ab4  # noqa: E402  (experiments/, read-only import)
from world_model.wm_semantic_monitor import RealWMSpillMonitor  # noqa: E402

ART_DIR = REPO_ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

WM_CHECKPOINT = str(REPO_ROOT / "world_model" / "logdir" / "wm_cup_full" / "best.pt")
HEAD_CHECKPOINT = str(REPO_ROOT / "world_model" / "logdir" / "wm_cup_full" / "monitor_head_cup_full.pt")
RISK_THRESHOLD = 0.5

CAPTURE_STRIDE = 5           # every 5 physics steps -> fps = 1/(0.005*5) = 40 -> real-time playback
VIDEO_FPS = 1.0 / (sc.DT * CAPTURE_STRIDE)
W, H = 1280, 720
INSET = 96                    # on-screen size of the 64x64 ZED inset (upscaled)

# Chosen-for-visualization threshold (NOT a certified safety value) for when
# segments 3-4's HUD shows the "engaged" banner -- picked by inspecting the
# real logged h(t) for run_ab4 condition C (logs/v4_C.npz): h ranges
# [0.016, 0.135] and spends ~76% of the episode below 0.05 while the
# intruder is actually near (matches the scripted ramp-in/hold/ramp-out
# window T_IN0=0.4s..T_OUT_END=5.8s almost exactly) -- i.e. this threshold
# tracks real physical proximity, not an arbitrary on-switch.
ENGAGE_H_M = 0.05

CUP_CAM = dict(lookat=np.array([0.40, -0.20, 0.72]), distance=1.35, azimuth=150, elevation=-12)
CANON_CAM = dict(lookat=np.array([0.42, -0.21, 0.65]), distance=0.9, azimuth=160, elevation=-15)


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


F_TITLE = _font(30)
F_LABEL = _font(22)
F_BANNER = _font(28)
F_OUTCOME = _font(34)

GREEN = (60, 200, 90)
YELLOW = (230, 200, 40)
RED = (220, 55, 55)
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)


def make_camera(spec):
    cam = mujoco.MjvCamera()
    cam.lookat = spec["lookat"]
    cam.distance = spec["distance"]
    cam.azimuth = spec["azimuth"]
    cam.elevation = spec["elevation"]
    return cam


def meter_color(frac):
    if frac < 0.4:
        return GREEN
    if frac < RISK_THRESHOLD:
        return YELLOW
    return RED


def draw_hud(rgb, *, title, meter_label, meter_frac, meter_text, banner_text,
             outcome_text, outcome_color, inset_img=None, low_is_danger=False,
             outcome_pos="top-right"):
    """Composite the HUD onto one RGB frame (H,W,3 uint8) with PIL, in place
    of a heavier per-frame matplotlib re-render (this needs to run ~800+
    times). `low_is_danger`: for margin/distance meters (segments 3-4) a LOW
    fraction is the dangerous end (unlike a P(spill) risk meter, where HIGH
    is dangerous) -- flips the color ramp accordingly."""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")

    # translucent top bar for the title
    draw.rectangle([0, 0, W, 46], fill=(0, 0, 0, 150))
    draw.text((14, 8), title, font=F_TITLE, fill=WHITE)

    # risk / margin meter (bottom-left)
    bar_x0, bar_y0, bar_w, bar_h = 14, H - 74, 340, 26
    draw.rectangle([bar_x0 - 6, bar_y0 - 30, bar_x0 + bar_w + 6, bar_y0 + bar_h + 8],
                   fill=(0, 0, 0, 150))
    draw.text((bar_x0, bar_y0 - 26), meter_label, font=F_LABEL, fill=WHITE)
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=WHITE, width=2)
    frac = float(np.clip(meter_frac, 0.0, 1.0))
    color = meter_color(1.0 - frac) if low_is_danger else meter_color(frac)
    if frac > 0.0:
        draw.rectangle([bar_x0 + 2, bar_y0 + 2, bar_x0 + 2 + frac * (bar_w - 4), bar_y0 + bar_h - 2],
                       fill=color)
    draw.text((bar_x0 + bar_w + 12, bar_y0 + 2), meter_text, font=F_LABEL, fill=WHITE)

    # ALERT banner (top-center, only when active)
    if banner_text:
        tw = draw.textlength(banner_text, font=F_BANNER)
        bx0, by0 = W / 2 - tw / 2 - 18, 60
        draw.rectangle([bx0, by0, bx0 + tw + 36, by0 + 46], fill=(200, 30, 30, 210))
        draw.text((bx0 + 18, by0 + 8), banner_text, font=F_BANNER, fill=WHITE)

    # outcome label: top-right (spill segments; short text, clear of the
    # centered banner) or bottom-right (collision segments; longer text such
    # as "SAFE, GRASP HELD" would otherwise overlap the centered banner --
    # bottom-right is free there since only the spill segments use the ZED inset)
    if outcome_text:
        tw = draw.textlength(outcome_text, font=F_OUTCOME)
        if outcome_pos == "bottom-right":
            ox0, oy0 = W - tw - 40, H - 60
        else:
            ox0, oy0 = W - tw - 40, 56
        draw.rectangle([ox0 - 12, oy0 - 6, ox0 + tw + 12, oy0 + 40], fill=outcome_color + (215,))
        draw.text((ox0, oy0), outcome_text, font=F_OUTCOME, fill=WHITE)

    # ZED inset (bottom-right) -- the WM's ACTUAL 64x64 input, upscaled for visibility
    if inset_img is not None:
        ins = Image.fromarray(inset_img).resize((INSET, INSET), Image.NEAREST)
        ix0, iy0 = W - INSET - 14, H - INSET - 14
        draw.rectangle([ix0 - 4, iy0 - 22, ix0 + INSET + 4, iy0 + INSET + 4], fill=(0, 0, 0, 150))
        draw.text((ix0, iy0 - 20), "WM input (64x64 ZED)", font=ImageFont.load_default(), fill=WHITE)
        img.paste(ins, (ix0, iy0))

    return np.asarray(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Segment 1/2: SPILL (cup task) -- own loop mirroring eval/run_eval.py's
# run_episode, with per-frame rendering + HUD data added. Reuses
# scene_setup/safety_module/sim_common/cup_task_utils exactly as run_eval.py
# does; does not edit any of them.
# ---------------------------------------------------------------------------
def run_spill_segment(*, use_real_wm_control: bool, seed: int, duration_s: float,
                       title: str) -> list[np.ndarray]:
    geometric = sm.NoGeometricSafety()
    geometric.reset()
    monitor = RealWMSpillMonitor(wm_checkpoint=WM_CHECKPOINT, head_checkpoint=HEAD_CHECKPOINT,
                                  risk_threshold=RISK_THRESHOLD)
    monitor.reset()

    model = scene_setup.load_model()
    data = scene_setup.init_cup_grasped_state(model, seed=seed, cup_jitter_m=0.0)
    _, left_qposadr, left_dofadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, right_dofadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    q0_left = data.qpos[left_qposadr].copy()
    sc.disable_left_arm_position_servos(model)

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    ee_body = model.body(scene_setup.EE_BODY_NAME).id
    cup_body = model.body(ct.CUP_BODY_NAME).id

    Jr_pinv = scene_setup.left_ee_jac_rot_pinv(model, data, ee_body, left_dofadr)
    q_transfer_dir = Jr_pinv @ scene_setup.TRANSFER_ROT_AXIS
    cup_rel_pos0, cup_rel_quat0 = scene_setup.cup_rel_pose(model, data, ee_body, cup_body)

    n_steps = int(round(duration_s / sc.DT))
    rng = np.random.default_rng(seed + 10_000)
    t_offset = float(rng.uniform(-0.05, 0.05))

    renderer3p = mujoco.Renderer(model, height=H, width=W)
    cam3p = make_camera(CUP_CAM)

    frames = []
    transfer_t_elapsed = 0.0
    ever_spilled_shown = False
    for k in range(n_steps):
        t = k * sc.DT
        data.ctrl[left_j7] = 0.0
        data.ctrl[left_j8] = 0.0
        mujoco.mj_forward(model, data)
        scene_setup.reattach_cup_rigidly(model, data, ee_body, cup_body, cup_rel_pos0, cup_rel_quat0)
        mujoco.mj_forward(model, data)

        verdict = monitor.assess(sm.SemanticObservation(model=model, data=data, t=t))
        veto = bool(verdict.veto) if use_real_wm_control else False
        if not veto:
            transfer_t_elapsed += sc.DT
        transfer_off = scene_setup.transfer_tilt_offset(max(0.0, transfer_t_elapsed + t_offset))
        q_target = q0_left + q_transfer_dir * transfer_off
        tau_task, M_ll, bias_l = scene_setup.task_torque(
            model, data, q_target, left_dofadr, left_qposadr, ee_body, cup_body, freeze=veto)
        geo_inputs = sm.GeometricInputs(model=model, data=data, left_dofadr=left_dofadr,
                                         right_dofadr=right_dofadr, tau_task=tau_task,
                                         M_ll=M_ll, bias_l=bias_l)
        geo_result = geometric.correct(geo_inputs)
        tau_cmd = np.clip(geo_result.tau, -sc.TAU_MAX, sc.TAU_MAX)
        data.qfrc_applied[left_dofadr] = tau_cmd

        spilled_now = ct.count_spilled(model, data) > 0
        ever_spilled_shown = ever_spilled_shown or spilled_now

        if k % CAPTURE_STRIDE == 0:
            renderer3p.update_scene(data, camera=cam3p)
            rgb = renderer3p.render().copy()
            zed = monitor._renderer.render().copy() if monitor._renderer is not None else None
            if use_real_wm_control:
                meter_label = "P(spill) -- REAL WM monitor (drives HOLD)"
                banner = "⚠ SPILL RISK → HOLDING" if verdict.veto else ""
            else:
                meter_label = "P(spill) -- REAL WM monitor (SHADOW, not acted on)"
                banner = ""
            outcome_text, outcome_color = None, BLACK
            if ever_spilled_shown:
                outcome_text, outcome_color = "SPILLED", RED
            elif t > duration_s - 1.0:
                outcome_text = "SAFE" if use_real_wm_control else None
                outcome_color = GREEN
            frame = draw_hud(
                rgb, title=title, meter_label=meter_label, meter_frac=verdict.risk,
                meter_text=f"{verdict.risk:.2f}", banner_text=banner,
                outcome_text=outcome_text, outcome_color=outcome_color, inset_img=zed)
            frames.append(frame)

        mujoco.mj_step(model, data)

    renderer3p.close()
    final_spilled = int(ct.count_spilled(model, data))
    print(f"  [spill:{'wm' if use_real_wm_control else 'baseline'}] "
          f"ever_spilled={ever_spilled_shown} final_spilled_count={final_spilled}")
    return frames


# ---------------------------------------------------------------------------
# Segment 3/4: COLLISION (v4 cooperative intruder) -- reuses
# run_ab4.run_condition's own per-step logging + capture callback (no changes
# to run_ab4.py needed: it already supports capture_steps/capture_cb).
# ---------------------------------------------------------------------------
def run_collision_segment(condition: str, title: str) -> list[np.ndarray]:
    n_steps = sc.N_STEPS
    capture_steps = set(range(0, n_steps, CAPTURE_STRIDE))
    state = {"renderer": None, "cam": make_camera(CANON_CAM)}
    captured = {}

    def cb(cond, k, model, data):
        if state["renderer"] is None:
            state["renderer"] = mujoco.Renderer(model, height=H, width=W)
        state["renderer"].update_scene(data, camera=state["cam"])
        captured[k] = state["renderer"].render().copy()

    # run_ab4.run_condition unconditionally np.savez's to its module-level
    # LOG_DIR -- redirect that to a scratch tempdir for the duration of this
    # call so this demo script never writes under experiments/ (this task's
    # write scope is artifacts/ + this file only). run_ab4.py itself is not
    # modified: LOG_DIR is a plain module global, re-read at call time.
    _orig_log_dir = run_ab4.LOG_DIR
    run_ab4.LOG_DIR = tempfile.mkdtemp(prefix="safety_demo_ab4_")
    try:
        log = run_ab4.run_condition(condition, capture_steps=capture_steps, capture_cb=cb)
    finally:
        run_ab4.LOG_DIR = _orig_log_dir
    if state["renderer"] is not None:
        state["renderer"].close()

    table_top_z = 0.55
    frames = []
    collision_so_far = False
    dropped_so_far = False
    for k in sorted(captured.keys()):
        t = float(log["t"][k])
        h_val = float(log["h"][k])
        collision_so_far = collision_so_far or bool(log["collision"][k] > 0.5)
        dropped_so_far = dropped_so_far or (log["box_z"][k] < table_top_z - 0.01) or \
            (log["box_drift_trans"][k] > 0.05)

        engaged = (h_val < ENGAGE_H_M) and not collision_so_far
        if condition == "C":
            banner = "⚠ COLLISION → COOPERATIVE AVOID" if engaged else ""
            margin_label = "arm-arm margin h -- COOPERATIVE (v4-C)"
        else:
            banner = ""
            margin_label = "arm-arm margin h -- NO COOPERATION"

        # map h in [-0.05, +0.15] -> [0,1] fraction (0=touching/negative, 1=safe)
        frac = float(np.clip((h_val + 0.05) / 0.20, 0.0, 1.0))

        outcome_text, outcome_color = None, BLACK
        if collision_so_far:
            outcome_text, outcome_color = "COLLISION", RED
        elif dropped_so_far:
            outcome_text, outcome_color = "BOX DROPPED", RED
        elif t > (sc.DURATION_S - 1.0):
            outcome_text, outcome_color = "SAFE, GRASP HELD", GREEN

        frame = draw_hud(
            captured[k], title=title, meter_label=margin_label, meter_frac=frac,
            meter_text=f"h={h_val:+.3f} m", banner_text=banner,
            outcome_text=outcome_text, outcome_color=outcome_color, inset_img=None,
            low_is_danger=True, outcome_pos="bottom-right")
        frames.append(frame)

    print(f"  [collision:{condition}] collision={collision_so_far} box_dropped={dropped_so_far} "
          f"min_h={float(np.min(log['h'])):.4f}")
    return frames


def encode_mp4(all_frames: list[np.ndarray], out_path: Path, fps: float):
    import imageio.v2 as imageio
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                 quality=8, macro_block_size=None)
    for f in all_frames:
        writer.append_data(f)
    writer.close()


def main():
    t0 = time.perf_counter()
    print("=== [1/4] SPILL baseline (config 'none', shadow WM telemetry only) ===")
    seg1 = run_spill_segment(use_real_wm_control=False, seed=0, duration_s=4.5,
                              title="BASELINE -- NO SEMANTIC SAFETY (fast cup tilt)")

    print("=== [2/4] SPILL WM-protected (config 'semantic_only', REAL WM veto) ===")
    seg2 = run_spill_segment(use_real_wm_control=True, seed=0, duration_s=4.5,
                              title="WM SAFETY -- REAL trained spill monitor")

    print("=== [3/4] COLLISION adversarial (run_ab4 condition A) ===")
    seg3 = run_collision_segment("A", "ADVERSARIAL INTRUDER -- NO COOPERATION")

    print("=== [4/4] COLLISION cooperative (run_ab4 condition C) ===")
    seg4 = run_collision_segment("C", "COOPERATIVE SAFETY -- v4-C (geometric)")

    all_frames = seg1 + seg2 + seg3 + seg4
    out_path = ART_DIR / "safety_demo.mp4"
    print(f"encoding {len(all_frames)} frames @ {VIDEO_FPS:.1f} fps -> {out_path}")
    encode_mp4(all_frames, out_path, VIDEO_FPS)

    meta = dict(
        n_frames=len(all_frames), fps=VIDEO_FPS, resolution=[W, H],
        segments=[
            dict(name="spill_baseline", n_frames=len(seg1)),
            dict(name="spill_wm", n_frames=len(seg2)),
            dict(name="collision_adversarial", n_frames=len(seg3)),
            dict(name="collision_cooperative", n_frames=len(seg4)),
        ],
        wall_time_s=time.perf_counter() - t0,
    )
    with open(ART_DIR / "safety_demo_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_path} ({meta['wall_time_s']:.1f}s wall)")


if __name__ == "__main__":
    main()
