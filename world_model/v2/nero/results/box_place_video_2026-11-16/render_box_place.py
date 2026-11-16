"""Box place demo video (2026-11-16, USER request).  SIMULATION ONLY, offline, CPU render (MUJOCO_GL=osmesa), no VLM, no hardware.

USER 11-16: "remove shelf clearance that bound and run it ... give me that video and annotate it with all details, plans, decisions, controls".
What this run does:
  1. sets carry.U_OBJ_CLEAR_M (the D9 object-world clearance, 5 mm ASSUMED) to --clear (default 0.0) IN THIS PROCESS ONLY: the episode monitor and the Rule 1
     collision rows then count only real contact / penetration, not a 5 mm margin.  No code default is changed.
  2. checker decisions on the cell: static admission (K2) and Rule 1 (K6, with the 11-16 stop-finish rule) with the bound removed; the with-bound Rule 1 records
     (same code, same cell, computed earlier in this session) are read from --with-bound-dir for the side-by-side panel.
  3. one nominal box episode (closed-chain default plan, FL-MPC tracker, a = the admission's a_admit), no backups run, frames every 7 control steps (28.6 fps
     = real time), two views + an annotation panel.
Not shown: the probe (the plan is the closed-chain DEFAULT plan, not a VLM plan), stops / backups (no stop is triggered in a nominal run).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import textwrap
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["BIMANUAL_PLANNER"] = "closed_chain"

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from world_model.v2.adapters import rule1_contents as RC
from world_model.v2.nero import config as CFG
from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.stack import types as TY
from world_model.v2.stack.planners import closed_chain as CCP

VW, VH, PW = 640, 400, 660          # view size, panel width
K_FRAME = 7                          # control steps per frame (DT 5 ms -> 28.6 fps = real time)
SIDES = ("left", "right")


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


F_T, F_B, F_S = font(17), font(14), font(13)


def rule1_table(recs):
    out = []
    for r in recs:
        if r.get("admitted"):
            out.append(dict(step=r["seg"], verdict="ADMITTED", program=r["program"], guarantee=r.get("guarantee"), why=""))
        else:
            whys = sorted({(f.get("why") or f.get("guarantee") or "")[:120] for p in (r.get("per_program") or {}).values() for f in p["failing"]})
            out.append(dict(step=r["seg"], verdict="REFUSED", program=None, guarantee=r.get("guarantee"), why=(whys[0] if whys else r.get("refusal", ""))))
    return out


class Recorder:
    def __init__(self, ep, cams, static_lines):
        self.ep, self.cams, self.static_lines = ep, cams, static_lines
        m = ep.m
        m.vis.headlight.ambient[:] = 0.45
        m.vis.headlight.diffuse[:] = 0.6
        self.r = mujoco.Renderer(m, VH, VW)
        self.frames, self.log, self.u_events = [], [], []
        self.seg = "load transfer (hold 0.4 s)"
        self.tau_lim = CFG.tau_limit()
        self.s0 = ep.d.xpos[ep.obj].copy()
        orig_store, orig_viol = ep.store, ep._viol

        def store(seg):
            self.seg = seg
            return orig_store(seg)

        def viol(kind, t, **kw):
            self.u_events.append(dict(kind=kind, t=round(float(t), 3), seg=self.seg, **{k: (round(float(v), 4) if isinstance(v, (int, float, np.number)) else v) for k, v in kw.items()}))
            return orig_viol(kind, t, **kw)

        ep.store, ep._viol = store, viol
        ep.step_hooks.append(self)

    def stats(self):
        ep, d = self.ep, self.ep.d
        t = ep.k * CR.DT
        R = d.xmat[ep.obj].reshape(3, 3)
        idx = {"left": ep.ctl.ld, "right": ep.ctl.rd}           # the FL-MPC torque is applied as qfrc_applied at these dofs (carry.py _step_core)
        tau = {s: float(np.max(np.abs(d.qfrc_applied[idx[s]]) / ep.tau_lim)) for s in SIDES}
        claw = {s: float(np.linalg.norm(ep.claw_wrench(s)[0])) for s in SIDES}
        err = {s: float(np.max(np.abs(ep._cmd(s, t) - d.qpos[ep.addr[s]]))) for s in SIDES}
        p = d.xpos[ep.obj] - self.s0
        clr = ep.track.get("obj_world_clear_min", float("nan"))
        return dict(t=round(t, 3), seg=self.seg, obj_dxyz_mm=[round(float(x) * 1e3, 1) for x in p], tilt_deg=round(CR._tilt_deg(R), 2),
                    tau_util=tau, claw_N=claw, track_err_rad=err, obj_world_min_mm=(None if not np.isfinite(clr) else round(clr * 1e3, 2)),
                    U=sorted({e["kind"] for e in self.u_events}))

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
                ("b", f"object displacement from start [mm]: dx {st['obj_dxyz_mm'][0]:+.0f}  dy {st['obj_dxyz_mm'][1]:+.0f}  dz {st['obj_dxyz_mm'][2]:+.0f}"),
                ("b", f"object tilt: {st['tilt_deg']:.2f} deg"),
                ("b", f"joint torque / limit (100 Nm): left {st['tau_util']['left']:.2f}   right {st['tau_util']['right']:.2f}"),
                ("b", f"claw force |F| [N]: left {st['claw_N']['left']:.1f}   right {st['claw_N']['right']:.1f}"),
                ("b", f"tracking error max |q_cmd - q| [rad]: left {st['track_err_rad']['left']:.4f}   right {st['track_err_rad']['right']:.4f}"),
                ("b", "object-world clearance, min this step [mm]: " + ("-" if st["obj_world_min_mm"] is None else f"{st['obj_world_min_mm']:.1f}")
                 + "   (resting/approach pairs excluded)"),
                ("w" if st["U"] else "b", "U events: " + (", ".join(st["U"]) if st["U"] else "none"))]
        for kind, txt in live:
            f, col = {"h": (F_T, (120, 220, 255)), "b": (F_B, (235, 235, 235)), "w": (F_B, (255, 120, 100))}[kind]
            for ln in textwrap.wrap(txt, 74 if kind != "h" else 60):
                dr.text((10, y), ln, font=f, fill=col)
                y += f.size + 3
        out = Image.new("RGB", (img.shape[1] + PW, H))
        out.paste(Image.fromarray(img), (0, 0))
        out.paste(pan, (img.shape[1], 0))
        d2 = ImageDraw.Draw(out)
        d2.text((8, 6), "side view", font=F_S, fill=(255, 255, 255))
        d2.text((8, VH + 6), "close-up: place onto the shelf", font=F_S, fill=(255, 255, 255))
        return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="1.0-1.5")
    ap.add_argument("--contents", type=float, default=1.25, help="true contents mass of the world shown [kg] (the cell centre)")
    ap.add_argument("--clear", type=float, default=0.0, help="D9 object-world clearance used in THIS run [m] (code default 0.005, ASSUMED)")
    ap.add_argument("--with-bound-dir", required=True, help="Rule 1 records of the same cell with the 5 mm bound (run_cell json files)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cell = tuple(float(x) for x in a.cell.split("-"))
    clear0 = CR.U_OBJ_CLEAR_M
    CR.U_OBJ_CLEAR_M = float(a.clear)
    t0 = time.time()

    w = dataclasses.replace(CC.world_default("box"), contents_kg=a.contents)
    st = CC.static_cell("box", cell, CC.world_default("box"), planner="closed_chain")
    (out / "rule1_bound_removed").mkdir(exist_ok=True)
    recs_nb = RC.run_cell("box", cell, out / "rule1_bound_removed", n_sampled=0, planner="closed_chain", fallback=False)
    recs_wb = [json.loads(p.read_text()) for p in sorted(pathlib.Path(a.with_bound_dir).glob("box_*_step*.json"))]
    a_admit = float(recs_nb[0].get("plan_a_admit") or st.get("a_admit") or 0.5)
    tab_nb, tab_wb = rule1_table(recs_nb), rule1_table(recs_wb)
    plan = CCP.plan_for(CCP._nominal_inputs("box"), a_admit)
    durs = {s: round(plan.pieces[s].n * CCP.DT, 3) for s in plan.pieces}
    offs = plan.geo.offs
    nodes = {s: (np.round(offs[i1] * 1e3).astype(int).tolist()) for s, (i0, i1) in plan.geo.bounds.items()}
    backups = {s: [p[0] for p in (CCP.backup_phases_for(s) or [])] for s in plan.pieces}

    fmt = lambda tb: "; ".join(f"{r['step']}: {r['verdict']}" + (f" ({r['program']})" if r["program"] else "") for r in tb)
    why_wb = next((r["why"] for r in tab_wb if r["verdict"] == "REFUSED"), "")
    static_lines = [
        ("h", "BOX CARRY + PLACE ON SHELF  (simulation, NERO 7-DoF x2, NYU grippers)"),
        ("s", f"world: contents {a.contents:.2f} kg (true, hidden from the robot); belief cell {cell[0]}-{cell[1]} kg; box 0.25 kg; claws welded (rigid grasp)"),
        ("h", "PLAN (closed-chain planner, default waypoints; not a VLM plan)"),
        ("b", "object path (offsets from start, mm): " + "; ".join(f"{s} -> {nodes[s]}" for s in nodes)),
        ("b", "timing: TOPP-RA rest-to-rest per move, object accel <= a per axis, joint speed <= datasheet; durations " + ", ".join(f"{s} {d} s" for s, d in durs.items())),
        ("b", "stored backups: " + "; ".join(f"{s}: {'+'.join(v)}" for s, v in backups.items())),
        ("h", "DECISIONS (checkers)"),
        ("b", f"static admission (K2, torque/payload/support over the cell): {st.get('action')}; a_admit = {a_admit:.2f} m/s^2 -> run at a = {a_admit:.2f}"),
        ("b", f"Rule 1 with the D9 bound (5 mm): {fmt(tab_wb)}"),
        ("s", f"   why place refused: {why_wb}"),
        ("b", f"Rule 1 with the bound REMOVED (this run, d_min = {a.clear * 1e3:.1f} mm): {fmt(tab_nb)}"),
        ("s", f"stop rule (11-16): a stop in the last T = {TY.STOP_T_S} s of a move lets the move finish; else Hermite stop over T (decel v/T)"),
        ("h", "CONTROL"),
        ("b", "tracker: FL-MPC (tau = M a + c, QP with torque box 100 Nm/joint, clearance rows), 5 ms control step; monitors every step (U set)"),
        ("w", f"OVERRIDE FOR THIS VIDEO ONLY: D9 object-world clearance {clear0 * 1e3:.0f} mm (ASSUMED) -> {a.clear * 1e3:.1f} mm; code default unchanged"),
    ]

    ep = CR.CarryEpisode(w, a_plan=a_admit, planner="closed_chain")
    mujoco.mj_forward(ep.m, ep.d)
    s0 = ep.d.xpos[ep.obj].copy()
    goal = s0 + offs[-1]
    cams = []
    for az, el, dist, la in [(90, -15, 1.55, (s0 + goal) / 2 + [0, 0, 0.10]), (55, -8, 0.75, goal + [0, 0, 0.05])]:
        c = mujoco.MjvCamera()
        c.lookat[:] = la
        c.azimuth, c.elevation, c.distance = az, el, dist
        cams.append(c)
    rec_ = Recorder(ep, cams, static_lines)
    ep.run_phases([("hold", 0.4)])
    rec = dict(tag="demo_box_place", task=ep.task, theta=CR._theta(w), truth=CR._jsonable(ep.truth0), a_plan=a_admit, m_nom=ep.m_nom, ctl="lever",
               strict_claw=False, fric_comp=ep.fc, coop=ep.coop, weld_solref=ep.weld_solref, physics=ep.physics_flags(), segments=[])
    res = CR.run_segments(ep, rec, branch_backups=False)
    rec_.seg = "end (hold)"
    ep.run_phases([("hold", 0.6)])
    CR.U_OBJ_CLEAR_M = clear0

    fps = 1.0 / (K_FRAME * CR.DT)
    frames = rec_.frames + [rec_.frames[-1]] * int(fps * 1.5)
    vid = out / "box_place_annotated.mp4"
    imageio.mimwrite(vid, frames, fps=fps, macro_block_size=1, quality=8)
    segs = [dict(seg=s.get("seg"), end_in_R=s.get("nominal", {}).get("end_in_R"), viol=s.get("nominal", {}).get("viol"),
                 viol_all=list((s.get("nominal", {}).get("viol_all") or {}).keys())) for s in res.get("segments", [])]
    summary = dict(video=str(vid), n_frames=len(frames), fps=fps, wall_s=round(time.time() - t0, 1), clear_used_m=a.clear, clear_code_default_m=clear0,
                   contents_kg=a.contents, cell=cell, a_plan=a_admit, static_admission=dict(action=st.get("action"), a_admit=st.get("a_admit")),
                   rule1_with_bound=tab_wb, rule1_bound_removed=tab_nb, plan=dict(durations_s=durs, nodes_mm=nodes, backups=backups),
                   segments=segs, u_events=rec_.u_events, final=rec_.log[-1] if rec_.log else None)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "frame_log.json").write_text(json.dumps(rec_.log, default=str))
    print(json.dumps({k: summary[k] for k in ("video", "n_frames", "wall_s", "rule1_bound_removed", "segments", "u_events")}, indent=1, default=str))


if __name__ == "__main__":
    main()
