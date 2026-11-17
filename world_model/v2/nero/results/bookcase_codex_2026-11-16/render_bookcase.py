"""Bookcase demo video from the Codex run 2 plan (USER 11-16: "go with b and run it again"; video asked with the bookcase task).  SIMULATION ONLY, offline, CPU
render (MUJOCO_GL=osmesa), no hardware.  The proposer is NOT called: the run-2 cache is replayed with --offline semantics (a cache miss raises).

Same path as run_bookcase.py (same_worlds._proposer_plans -> behind_job, same world index and seed), so the episode is the run-2 episode; the script
checks that M_true and the outcome equal the recorded run-2 world json and writes both into the README.  A step hook on every CarryEpisode built in this
process renders two views (wide: counter -> bookcase; close-up: the lower compartment) every K_FRAME control steps, plus an annotation panel.

    PYTHONPATH=. MUJOCO_GL=osmesa python world_model/v2/nero/results/bookcase_codex_2026-11-16/render_bookcase.py --run run2 --world 2 --out OUT
"""
import argparse
import json
import os
import pathlib
import textwrap
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from world_model.v2.nero.contents import carry as CR

VW, VH, PW = 640, 400, 660
K_FRAME = 7                          # control steps per frame (DT 5 ms -> 28.6 fps = real time)
SIDES = ("left", "right")
HERE = pathlib.Path(__file__).resolve().parent


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_T, F_B, F_S = font(17), font(14), font(13)


class Recorder:
    def __init__(self, ep, static_lines, goal_xy):
        self.ep, self.static_lines, self.goal_xy = ep, static_lines, np.asarray(goal_xy, float)
        m = ep.m
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.6
        self.r = mujoco.Renderer(m, VH, VW)
        self.frames, self.log, self.u_events = [], [], []
        self.seg = "probe (weigh)"
        self.s0 = ep.d.xpos[ep.obj].copy()
        mid = 0.5 * (self.s0[:2] + self.goal_xy)
        self.cams = []
        for look, dist, az, el in (((mid[0], mid[1], 0.7), 4.6, 90.0, -32.0), ((self.goal_xy[0], self.goal_xy[1], 0.95), 1.6, 135.0, -18.0)):
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = look
            c.distance, c.azimuth, c.elevation = dist, az, el
            self.cams.append(c)
        orig_store, orig_viol = ep.store, ep._viol

        def store(seg):
            self.seg = seg
            return orig_store(seg)

        def viol(kind, t, **kw):
            self.u_events.append(dict(kind=kind, t=round(float(t), 3), seg=self.seg))
            return orig_viol(kind, t, **kw)

        ep.store, ep._viol = store, viol
        ep.step_hooks.append(self)

    def stats(self):
        ep, d = self.ep, self.ep.d
        t = ep.k * CR.DT
        R = d.xmat[ep.obj].reshape(3, 3)
        p = d.xpos[ep.obj]
        claw = {s: float(np.linalg.norm(ep.claw_wrench(s)[0])) for s in SIDES}
        try:
            idx = {"left": ep.ctl.ld, "right": ep.ctl.rd}
            tau = {s: float(np.max(np.abs(d.qfrc_applied[idx[s]]) / ep.tau_lim)) for s in SIDES}
        except Exception:
            tau = None
        clr = ep.track.get("obj_world_clear_min", float("nan"))
        return dict(t=round(t, 3), seg=self.seg, obj_xyz=[round(float(x), 3) for x in p], to_goal_m=round(float(np.hypot(*(p[:2] - self.goal_xy))), 3),
                    tilt_deg=round(CR._tilt_deg(R), 2), claw_N=claw, tau_util=tau,
                    obj_world_min_mm=(None if not np.isfinite(clr) else round(clr * 1e3, 2)), U=sorted({e["kind"] for e in self.u_events}))

    def __call__(self, ep):
        if ep.k % K_FRAME:
            return
        st = self.stats()
        self.log.append(st)
        views = []
        for c in self.cams:
            self.r.update_scene(ep.d, c)
            views.append(self.r.render())
        self.frames.append(self.compose(np.vstack(views), st))

    def compose(self, img, st):
        H = img.shape[0]
        pan = Image.new("RGB", (PW, H), (18, 20, 26))
        dr = ImageDraw.Draw(pan)
        y = 8
        for kind, txt in self.static_lines:
            f, col = {"h": (F_T, (255, 214, 102)), "b": (F_B, (230, 230, 230)), "w": (F_B, (255, 140, 120)), "s": (F_S, (170, 190, 210))}[kind]
            for ln in textwrap.wrap(txt, 74 if kind != "h" else 60) or [""]:
                dr.text((10, y), ln, font=f, fill=col)
                y += f.size + 3
            y += 2
        y += 6
        dr.line([(8, y), (PW - 8, y)], fill=(90, 90, 110), width=1)
        y += 8
        live = [("h", f"LIVE  t = {st['t']:.2f} s   step: {st['seg']}"),
                ("b", f"crate centre (world) [m]: x {st['obj_xyz'][0]:+.3f}  y {st['obj_xyz'][1]:+.3f}  z {st['obj_xyz'][2]:.3f}   distance to goal xy {st['to_goal_m']:.3f} m"),
                ("b", f"crate tilt: {st['tilt_deg']:.2f} deg"),
                ("b", f"claw force |F| [N]: left {st['claw_N']['left']:.1f}   right {st['claw_N']['right']:.1f}")]
        if st["tau_util"]:
            live.append(("b", f"joint torque / limit (100 Nm): left {st['tau_util']['left']:.2f}   right {st['tau_util']['right']:.2f}"))
        live += [("b", "crate-world clearance, min so far [mm]: " + ("-" if st["obj_world_min_mm"] is None else f"{st['obj_world_min_mm']:.1f}")
                  + "   (resting/approach pairs excluded)"),
                 ("w" if st["U"] else "b", "unsafe (U) events: " + (", ".join(st["U"]) if st["U"] else "none"))]
        for kind, txt in live:
            f, col = {"h": (F_T, (120, 220, 255)), "b": (F_B, (235, 235, 235)), "w": (F_B, (255, 120, 100))}[kind]
            for ln in textwrap.wrap(txt, 74 if kind != "h" else 60):
                dr.text((10, y), ln, font=f, fill=col)
                y += f.size + 3
        out = Image.new("RGB", (img.shape[1] + PW, H))
        out.paste(Image.fromarray(img), (0, 0))
        out.paste(pan, (img.shape[1], 0))
        d2 = ImageDraw.Draw(out)
        d2.text((8, 6), "wide view: counter 1 (start) -> bookcase (goal)", font=F_S, fill=(255, 255, 255))
        d2.text((8, VH + 6), "close-up: lower compartment of the bookcase", font=F_S, fill=(255, 255, 255))
        return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run2")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["BIMANUAL_NO_NETWORK"] = "1"                   # replay only
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = HERE / a.run
    rec_world = json.loads((run / f"world_{a.world}.json").read_text())
    task = "box_bookcase"
    from world_model.v2.nero.contents import probe as P, same_worlds as SW, behind_job as BJ
    from world_model.v2.nero.base import smoke as SM
    from world_model.v2.stack import goal as G
    method = P.PROBE_METHOD
    cal = P.load_calib(P.default_calib(method), method)["eps"]
    eps_lo, eps_free = float(cal["EPS_LO"]), float(cal["EPS_FREE"])
    ns = argparse.Namespace(plan_spec="codex:gpt-5.6-sol:high", plan_cache=str(run / "cache"), offline=True, outside_b="refuse", plan_images="zed+look",
                            plan_image_world=a.world, reask_limit=None)
    (out / "replay").mkdir(parents=True, exist_ok=True)
    ex = SW._proposer_plans([task], ns, out / "replay")[task]
    assert not ex.get("rejected"), ex.get("rejected")
    plan = ex["plan"]["plan_dict"]
    _, w = SM.behind_world(task, a.world)
    goal_xy = G.goal_xy(task, w)

    d = rec_world["decision"]
    steps = "; ".join(f"{s['skill']} {json.dumps(s['args'])}" for s in plan["steps"])
    static = [("h", "box_bookcase -- Codex-written plan, checked, then run (SIMULATION)"),
              ("s", "Proposer: codex gpt-5.6-sol, reasoning high. Run 2 (11-16), reply 3 of 3; cap was 8. Input: goal sentence + scene facts + 2 torso-camera images."),
              ("w", "Reply 1 refused (schema): route used line_m/turn_deg fields."),
              ("w", "Reply 2 refused (planner): turning in place sweeps the right arm into counter 1 (-101 mm)."),
              ("b", "Accepted plan: " + steps),
              ("b", "Recoveries per step: " + json.dumps({k: [x["skill"] for x in v] for k, v in plan.get("recoveries", {}).items()})),
              ("b", f"Probe (weigh): Theta total mass [{rec_world['Theta'][0]:.2f}, {rec_world['Theta'][1]:.2f}] kg; true {rec_world['M_true']:.2f} kg (hidden from the checker)."),
              ("b", f"Admission: {d['action']} at {d['a_admit']} m/s^2 (u_nom {d['u_nom']}); base speed level {d['base_level']} (highest level whose tip-over row passes)."),
              ("s", "Not in this run: Rule 1 per step (box_bookcase not yet in RULE1_TASKS), T4 (branch not merged)."),
              ("s", "ASSUMED: bookcase sizes BC_*, look-around image LOOK_BACK_M 0.5 m, goal tolerances (xy 5 cm, z 2 cm, tilt 45 deg, AgentChord's rule).")]

    recs = []
    orig_init = CR.CarryEpisode.__init__

    def init(self, *aa, **kw):
        orig_init(self, *aa, **kw)
        recs.append(Recorder(self, static, goal_xy))

    CR.CarryEpisode.__init__ = init
    t0 = time.time()
    r = BJ.behind_job((task, a.world, eps_lo, eps_free, {**ex["plan"], "probe_method": method}), method=method)
    CR.CarryEpisode.__init__ = orig_init
    (out / f"world_{a.world}_replay.json").write_text(json.dumps(r, default=str, indent=1))
    same = dict(M_true=(abs(r["M_true"] - rec_world["M_true"]) < 1e-9), outcome=(r["outcome"] == rec_world["outcome"]), any_U=(r["any_U"] == rec_world["any_U"]))
    import imageio
    rec = max(recs, key=lambda x: len(x.frames))
    imageio.mimwrite(out / "01_bookcase_codex_run2.mp4", rec.frames, fps=round(1.0 / (K_FRAME * CR.DT), 1), macro_block_size=1)
    (out / "02_log.json").write_text(json.dumps(dict(static=static, live=rec.log, u_events=rec.u_events, same_as_recorded=same, outcome=r["outcome"],
                                                     goal_row=r.get("goal")), default=str, indent=1))
    print("frames", len(rec.frames), "same_as_recorded", same, "outcome", r["outcome"], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
