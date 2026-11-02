"""nero/contents/static_cert.py -- the STATIC rows of segment admission as exact corner checks (proof under MODEL).  SIM ONLY.

USER 11-01 (step 3): joint torque, tilt and static stability over the belief set move from sampled sim checks to exact corner
checks labelled "proof under MODEL"; sampling stays only for dynamic effects (contact chaos, lid / content motion, slosh, impacts).

Rows for one segment, before it runs, from the robot's CURRENT joint state and the segment's PLANNED moves (no simulation):
  torque_payload  every planned path sample: max over contents vertices x acceleration box of the LP utilisation u* <= 1
                  (envelope.Envelope; u* convex in (m, m c) and a -> exact at corners).  Covers the segment's backup too when the
                  backup retraces the same path (lift <-> lower).  NOT covered: between path samples, placement error.
  support_rest    for segments that START or END with the object resting on the stove: the total COM lies inside the contact area
                  for every contents vertex (cert/model_check.support_at_rest; linear in (m, s) -> exact).  With a flat support and
                  a rigid object this also gives static tilt = 0 (no rest tilt unless the COM leaves the area).
  open_content    planned peak acceleration (smoothstep 6 d / T^2) <= the quasi-static limit of each open content (lid slide
                  mu_min g; cup spill g 2 (H - h) / D).  MODEL quasi-static: slosh / impacts are NOT covered (sampled).
Every row carries guarantee, the worst vertex (counterexample candidate) and its assumptions.

Belief contents: pi_b = (m, s_u, s_w) about the pan centre, u = handle axis (cert/belief.py).  Mapped to the envelope's object frame:
m c_xy = s_u u + s_w w; m c_z in [m z_lo, m z_hi] (vertices of the lifted set are the base vertices at both heights: exact).
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..democ import kin as K
from . import envelope as E
from . import objects as OB

SIDES = ("left", "right")
N_PER_MOVE = 6          # design: path samples per planned move (plus the start)
GUAR_TORQUE = ("proof under MODEL (rigid weld ASSUMED, quasi-static, NERO MJCF arms, 16 Nm MIT command range, 3 kg payload) over the "
               "contents polytope x acceleration box at {n} planned path samples; between samples and placement error not covered")


def pi_from_belief(verts_b, u_obj_xy, z_lo, z_hi):
    """belief vertices (m, s_u, s_w) -> envelope pi rows [m, m cx, m cy, m cz] at both heights."""
    u = np.asarray(u_obj_xy, float) / np.linalg.norm(u_obj_xy)
    w = np.array([-u[1], u[0]])
    out = []
    for m, su, sw in np.asarray(verts_b, float):
        xy = su * u + sw * w
        for z in (z_lo, z_hi):
            out.append([m, xy[0], xy[1], m * z])
    return np.unique(np.round(np.array(out), 12), axis=0)


def planned_path(ep, phases, sides, n_per_move=N_PER_MOVE):
    """joint samples along the planned TCP moves ("move" phases only) from the robot's current q, by the same continuation IK as
    Episode._add_path.  -> (cap dict for envelope.Arms, offsets {key: object offset}).  The object offset is the common delta of
    the moving claws (rigid carry); a move where the claws move differently is not a carry and is rejected."""
    q = {s: ep.d.qpos[ep.addr[s]].copy() for s in sides}
    cap = dict(sides=list(sides), qpos_rest=ep.d.qpos.tolist(), q={"p0": {s: q[s].tolist() for s in sides}})
    offsets, off, k = {"p0": np.zeros(3)}, np.zeros(3), 0
    for ph in phases:
        if ph[0] != "move":
            continue
        deltas = {s: np.asarray(v, float) for s, v in ph[1].items() if s in sides}
        dl = next(iter(deltas.values()))
        if any(np.linalg.norm(v - dl) > 1e-9 for v in deltas.values()) or set(deltas) != set(sides):
            raise ValueError("static_cert: a carry move must move every holding claw by the same delta")
        pik = {s: K.PoseIK(ep.m, s) for s in sides}
        p0 = {s: pik[s].ik.pose(q[s]) for s in sides}
        qq = {s: q[s].copy() for s in sides}
        for i in range(1, 61):
            a = i / 60
            for s in sides:
                qq[s], err = pik[s].solve(p0[s][0] + dl * a * a * (3 - 2 * a), p0[s][1], qq[s], q_post=q[s], k_post=0.1, iters=60,
                                          tol=1e-9)
            if i % (60 // n_per_move) == 0:
                k += 1
                key = f"p{k}"
                cap["q"][key] = {s: qq[s].tolist() for s in sides}
                for s in sides:
                    e = float(np.linalg.norm(pik[s].ik.pose(qq[s])[0] - (p0[s][0] + dl * a * a * (3 - 2 * a))))
                    if e > E.IK_TOL:
                        cap["q"][key][s + "_ik_err"] = e
                offsets[key] = off + dl * a * a * (3 - 2 * a)
        q, off = qq, off + dl
    return cap, offsets


def episode_object(ep):
    """ObjectSpec + frame of the Episode's pan/pot (empty = pan + lid from the MJCF; contents set from the belief, not here)."""
    m, d = ep.m, ep.d
    pan, lid = m.body("pan").id, m.body("lid").id
    ms = np.array([m.body_mass[pan], m.body_mass[lid]])
    R0, o0 = d.xmat[pan].reshape(3, 3).copy(), d.xpos[pan].copy()
    com = (ms[:, None] * np.array([d.xipos[pan], d.xipos[lid]])).sum(0) / ms.sum()
    spec = OB.pan_like(ep.obj, float(ms.sum()), R0.T @ (com - o0))
    return spec, o0, R0


def static_rows(ep, seg, belief, a_cap=None, tau_preset="mit_fw_ge_1.11"):
    """exact static rows for segment `seg` of Episode `ep` under `belief` (cert/belief.Belief).  -> dict(ok, rows, assumptions)."""
    from ..pan.cert import model_check as MC
    phases = ep.segment_phases(seg)
    holding = [s for s in SIDES if ep.gmode.get(s) == "force"] if seg != "pinch" else []
    spec, o0, R0 = episode_object(ep)
    vb = belief.contents_vertices()
    u_obj = np.asarray(ep.p.handle_axis(), float)[:2]
    V = pi_from_belief(vb, u_obj, OB.PAN_BOTTOM_T, OB.PAN_BOTTOM_T + OB.PAN_WALL_H)
    a_plan, _ = MC.planned_accel(phases)
    rows = {}
    if holding and any(ph[0] == "move" for ph in phases):
        cap, offsets = planned_path(ep, phases, holding)
        arms = E.Arms(cap, model=ep.m)
        env = E.Envelope(spec, arms, o0, R0, holding, tau_lim=C.tau_limits(tau_preset), grip_mass=float(
            ep.m.body_subtreemass[ep.m.body("left_arm_gripper_base").id]), offsets=offsets)
        mode = "two_share" if len(holding) == 2 else "one"          # what our controller commands (primary row)
        a_use = a_plan if a_cap is None else a_cap
        u, worst = env.worst(V, list(offsets), a_use, mode)
        rows["torque_payload"] = dict(ok=bool(u <= 1.0), u_max=round(float(u), 4), worst=worst, a=round(float(a_use), 4),
                                      n_samples=len(offsets), mode=mode, guarantee=GUAR_TORQUE.format(n=len(offsets)))
        if len(holding) == 2:                                       # capability with the best split (info: a better split exists?)
            u2, _ = env.worst(V, list(offsets), a_use, "two")
            rows["torque_payload"]["u_max_best_split"] = round(float(u2), 4)
    if seg in ("start", "pinch", "push", "clamp"):        # the object rests on the stove during / at the end of these
        centre = ep.d.site_xpos[ep.m.site("pan_centre").id][:2]
        P, planes = MC.support_polygon(centre, ep.r_out, ep.stove[0], ep.stove[1])
        M0 = float(spec.empty_mass)
        c0 = (o0 + R0 @ spec.empty_com)[:2]
        r = MC.support_at_rest(vb, M0, c0, centre, (R0 @ np.r_[u_obj, 0.0])[:2], planes)
        r["note"] = "static tilt = 0 at rest follows (flat support, rigid object) when ok"
        rows["support_rest"] = r
    # lid on the rim: mu = MJCF 1.0 (MODEL) x belief friction scale (lower end) x lid scale box lower end (OB.LID_MU, ASSUMED);
    # the world's true lid friction is NOT used (it is part of theta)
    a_open = float(min(belief.mu)) * OB.LID_MU[0] * OB.G
    rows["open_content"] = dict(ok=bool(a_plan <= a_open), a_plan=round(float(a_plan), 4), a_open=round(float(a_open), 4),
                                guarantee="proof under MODEL (Coulomb lid / quasi-static liquid; slosh and impacts not covered)")
    ok = all(r["ok"] for r in rows.values())
    return dict(ok=ok, seg=seg, rows=rows, assumptions=dict(
        grasp="ASSUMED rigid weld once closed (user 11-01)", torque=f"{tau_preset} (MEASURED-spec command range)",
        payload=f"{E.PAYLOAD_KG} kg per arm (MEASURED-spec) minus gripper", contents="belief contents polytope (cert/belief.py)",
        z_range=f"contents COM height {OB.PAN_BOTTOM_T}..{OB.PAN_BOTTOM_T + OB.PAN_WALL_H} m (MODEL)"))
