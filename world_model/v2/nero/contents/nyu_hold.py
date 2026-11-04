"""nyu_hold.py -- hold poses of the contents scenes (tray / box with side tabs) for the NYU gripper (11-03 grasp-geometry fix).

The AgileX capture (results/.../capture.json) came from the pan Episode's pre-grasp IK with the AgileX TCP (ee_grasp_z 0.120): the NYU jaws overlapped
the object / world at those poses.  Here the hold pose is re-solved for the NYU TCP (nyu_gripper.EE_GRASP_Z, 40 % of pivot-to-tip):
  * ee_grasp site on obj_grasp_left / obj_grasp_right, approach from the SIDE (left -y, right +y, as before), jaw closing axis = world z;
  * jaws set so the inner gap at the TCP = tab thickness + 2 x GRASP_CLEARANCE_M (gripper.opening_for_width);
  * among converged IK solutions: clearance >= CLEARANCE_MIN (5 mm, ASSUMED) of every robot geom against the world, against the object (arm links and the
    OTHER gripper; each welded gripper vs the object is exempt, see weldcontact) and left vs right arm, at the grasp pose AND the 5 cm lift pose; then the
    largest joint-limit margin.
Result file: contents/capture_nyu.json (same layout as capture.json["pot"], cached separately from the earlier capture.json, which nothing loads any more).
    python -m world_model.v2.nero.contents.nyu_hold --write
"""
from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np

from .. import config as C
from .. import gripper as G
from .. import clearance as CL
from ..ik import ArmIK
from ..democ import kin as K
from ..pan import scene as PS
from . import scenes as SC
from . import envelope as E

CAP_NYU = pathlib.Path(__file__).resolve().parent / "capture_nyu.json"
APPROACH = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([0.0, 1.0, 0.0])}
JAW = np.array([0.0, 0.0, 1.0])
LIFT = 0.05
# 11-03 jaw/body fix (contents tabs only; the pot keeps gripper.TCP_FRAC and its opening):
SHIFT_M = 0.055         # ASSUMED: ee_grasp site sits this far OUTSIDE the tab centre, along -approach, so the tab lies in the distal part of the pads and the closed jaw
                        # tips stop short of the object body (grasp depth = EE_GRASP_Z + SHIFT_M; tray needs >= 0.053: 0.045 left the tips 2.8 mm inside the rim).  Old value 0.0 (site on the tab centre).
BODY_CLEAR_MIN = 0.005  # ASSUMED (= CL.CLEARANCE_MIN): every jaw / palm geom of a welded gripper vs the object BODY geoms (tab_* excluded)
JAW_Z_FAR = 0.2110      # MEASURED nyu_gripper.JAW_TIP_Z_OPEN_M: deepest point of the pad region the tab can reach (gap is checked up to here)


def _model(world):
    m = PS.apply_sim_defaults(SC.build_spec(world).compile())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def hold_opening(w=SC.TAB_Z, shift=None, z_far=JAW_Z_FAR, clearance=None, tab_outer=0.21):
    """jaw opening for the tab grasp with the grasp point `shift` outside the tab: the most closed jaw angle whose pad-to-pad gap is >= w + 2 clearance at
    EVERY depth from the tab's outer end to z_far (the jaws are a V: closing on the tab at the TCP only (gripper.opening_for_width) makes the gap shrink
    below the tab thickness deeper along the pad, 14.5 mm at 0.184 m for a 20 mm tab).  -> dict(frac, nut, theta, gap_min_over_tab, z_lo, z_hi, ...).
    The pot path does not call this."""
    ng = G._nyu()
    cl = ng.GRASP_CLEARANCE_M if clearance is None else float(clearance)   # ASSUMED 2 mm per side, as gripper.opening_for_width
    shift = SHIFT_M if shift is None else float(shift)
    target = float(w) + 2.0 * cl
    z_lo = G.tcp_offset() + (SC.GRASP_Y + shift - tab_outer)     # MODEL: depth along the pads of the tab's outer end
    zs = np.linspace(z_lo, z_far, 12)
    best = None
    for th in np.arange(0.0, ng.THETA_MAX_RAD, 0.002):
        if min(G.jaw_gap_m(th, z) for z in zs) >= target:
            best = th
    if best is None:
        raise RuntimeError("no jaw angle keeps the gap >= tab thickness over the pad")
    u = G.theta_to_nut(best)
    return dict(frac=1.0 - u / ng.NUT_TRAVEL_M, nut=u, theta=float(best), target=target, z_lo=float(z_lo), z_hi=float(z_far),
                gap_min_over_tab=float(min(G.jaw_gap_m(best, z) for z in zs)))


def hold_finger_frac(w=SC.TAB_Z):
    """hold_opening() as the opening fraction that carry.HOLD_FINGER / gripper.set_opening take (gripper.opening_for_width(TAB_Z)["frac"] would put
    the jaws 9.4 mm inside the tab, this one leaves them 1.6 mm clear)."""
    return hold_opening(w)["frac"]


def set_grip(m, d, w=SC.TAB_Z, shift=None):
    o = hold_opening(w, shift)
    for s in C.SIDES:
        G.set_opening(m, d, s, o["frac"])
    return o


def jaw_body_min(m, d, obj_body="obj"):
    """-> (min distance [m], (gripper geom, body geom)) of every jaw / palm geom of both grippers vs the object BODY geoms (tab_* geoms excluded: the tab is
    between the pads).  Negative = penetration.  Own GJK (clearance._dist)."""
    robot, _, obj = CL.classify(m, obj_body)
    nm = lambda g: m.geom(g).name or f"g{g}"
    body = [g for g in obj if not nm(g).startswith("tab")]
    best = (CL._DMAX, None)
    for s in C.SIDES:
        for g in robot[s]:
            if CL._is_grip(m, g):
                for o in body:
                    v = CL._dist(m, d, g, o)
                    if v < best[0]:
                        best = (v, (nm(g), nm(o)))
    return best


def _pose_clear(m, d, ik, q, d_obj=0.0):
    for s in C.SIDES:
        d.qpos[ik[s].qa] = q[s]
    mujoco.mj_forward(m, d)
    rep = CL.report(m, d, "obj", welded=C.SIDES, detail=True)
    jb, pair = jaw_body_min(m, d)
    rep["jaw_body"] = jb
    rep["pairs"]["jaw_body"] = pair
    rep["min"] = min(rep["min"], jb)          # BODY_CLEAR_MIN == CL.CLEARANCE_MIN (5 mm, ASSUMED), so one threshold covers all
    return rep


def _tips_toward_object(ik, sols, approach):
    """ArmIK's approach error is a cross product, so it also accepts the anti-parallel gripper (jaw tips pointing AWAY from the object, palm over the body).
    Keep only the solutions whose gripper z axis points along `approach` (11-03 fix: at an outward-shifted grasp the right arm only had anti-parallel ones)."""
    keep = []
    for q, e in sols:
        ik.set_q(q)
        if float(ik.d.xmat[ik.gb].reshape(3, 3)[:, 2] @ approach) > 0.9:
            keep.append((q, e))
    return keep


def solve(world=None, n_sol=12, seeds=80, verbose=False, shift=None):
    """-> dict(q={side: q7}, clear=report, opening=..., cands=(nL, nR))"""
    world = world or SC.TrayWorld()
    m, d = _model(world)
    shift = SHIFT_M if shift is None else float(shift)
    op = set_grip(m, d, shift=shift)
    ik = {s: ArmIK(m, s) for s in C.SIDES}
    # ee_grasp site goes `shift` outside the tab centre (along -approach): the tab then lies `shift` deeper along the pads
    tgt = {s: d.site_xpos[m.site(f"obj_grasp_{s}").id].copy() - shift * APPROACH[s] for s in C.SIDES}
    sols = {s: _tips_toward_object(ik[s], ik[s].collect(tgt[s], APPROACH[s], JAW, n_sol=n_sol, seeds=seeds, seed=i), APPROACH[s]) for i, s in enumerate(C.SIDES)}
    best = None
    for qa, _ in sols["left"]:
        for qb, _ in sols["right"]:
            q = {"left": qa, "right": qb}
            rep = _pose_clear(m, d, ik, q)
            if rep["min"] < CL.CLEARANCE_MIN:
                continue
            ql = {}
            for s in C.SIDES:
                pik = K.PoseIK(m, s)
                p0, R0 = pik.ik.pose(q[s])
                qq, err = pik.solve(p0 + [0, 0, LIFT], R0, q[s], q_post=q[s], k_post=0.1, iters=200, tol=1e-9)
                ql[s] = qq
            rl = _pose_clear(m, d, ik, ql)
            if rl["min"] < CL.CLEARANCE_MIN:
                continue
            score = min(ik[s].margin(q[s]) for s in C.SIDES)
            if best is None or score > best[0]:
                best = (score, q, rep, rl)
    if best is None:
        raise RuntimeError(f"no hold pose with >= {CL.CLEARANCE_MIN*1e3:.0f} mm clearance ({len(sols['left'])}/{len(sols['right'])} IK solutions)")
    score, q, rep, rl = best
    return dict(q=q, clear=rep, clear_lift=rl, margin=float(score), opening=op, shift=shift, cands=(len(sols["left"]), len(sols["right"])), world=world,
                tcp=tgt)


def capture(world=None, **kw):
    """capture_nyu.json content: capture.json['pot'] layout, q / tcp per waypoint (continuation IK, orientation held, as envelope.capture)."""
    r = solve(world, **kw)
    world = r["world"]
    m, d = _model(world)
    set_grip(m, d, shift=r['shift'])
    q, tcp, clear = {}, {}, {}
    ik = {s: ArmIK(m, s) for s in C.SIDES}
    for s in C.SIDES:
        pik = K.PoseIK(m, s)
        q0 = np.asarray(r["q"][s])
        p0, R0 = pik.ik.pose(q0)
        for wp, dl in E.WAYPOINTS.items():
            qq = q0.copy()
            for i in range(1, 61):
                qq, err = pik.solve(p0 + np.asarray(dl, float) * i / 60, R0, qq, q_post=q0, k_post=0.1, iters=60, tol=1e-9)
            q.setdefault(wp, {})[s] = qq.tolist()
            tcp.setdefault(wp, {})[s] = pik.ik.pose(qq)[0].tolist()
            if err > 1e-4:
                q[wp][s + "_ik_err"] = float(err)
    for wp in q:
        for s in C.SIDES:
            d.qpos[ik[s].qa] = q[wp][s]
        clear[wp] = {k: v for k, v in CL.report(m, d, "obj", welded=C.SIDES).items()}
        clear[wp]["jaw_body"] = jaw_body_min(m, d)[0]
    old = json.loads((E.RESULTS / "capture.json").read_text())["pot"]      # object-side numbers do not depend on the gripper (pot: o0, R0, empty mass / COM)
    grip = float(m.body_subtreemass[m.body("left_arm_gripper_base").id])
    cap = dict(obj="pot", grasp_frac=old["grasp_frac"], sides=["left", "right"], q=q, tcp=tcp, qpos_rest=[], o0=old["o0"], R0=old["R0"],
               empty_mass=old["empty_mass"], empty_com_world=old["empty_com_world"], gripper_mass=grip,
               provenance=dict(gripper="nyu", tcp_z=G.tcp_offset(), grasp_shift_m=r['shift'], grasp_shift_label='ASSUMED (11-03; old 0.0)', body_clear_min_m=BODY_CLEAR_MIN, tab_thickness=SC.TAB_Z, opening=r["opening"], clearance_min_required_m=CL.CLEARANCE_MIN,
                               clearance_at_hold=r["clear"], clearance_at_lift=r["clear_lift"], clearance_by_waypoint=clear, n_ik_solutions=list(r["cands"]),
                               joint_margin=r["margin"],
                               note="solved by contents/nyu_hold.py; o0/R0/empty_* copied from results/contents_envelope_2026-11-01/capture.json (object side, gripper independent)"))
    return cap


def write(path=CAP_NYU, **kw):
    cap = capture(**kw)
    pathlib.Path(path).write_text(json.dumps({"pot": cap}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return cap


if __name__ == "__main__":
    import sys
    c = write()
    print(json.dumps(c["provenance"], indent=1, default=str))
