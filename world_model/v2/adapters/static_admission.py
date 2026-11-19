"""adapters/static_admission.py -- runs stack/checks/admit.admit and returns carry_cert.static_cell's dict (BIMANUAL_CHECKER=unified).  SIMULATION ONLY.

The caller (nero/contents/carry_cert.static_cell) builds the PathSamples (scripted: payload_envelope.scripted_path; closed_chain: the planner's static_path;
a PlanRefused never gets here), so this module imports no planner.  Keys, rounding, row names and the guarantee texts are the legacy dict's; the decision
(action, a_admit, rows' ok / u / margin) is admit's.  T_cap is always None here (the old claw twist rule has no unified counterpart).
"""
from __future__ import annotations

import os

import numpy as np

from world_model.v2.adapters import payload_envelope as PE
from world_model.v2.nero.contents import envelope as E
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import admit as AD

ROW_GUAR = ("proof under MODEL (rigid weld ASSUMED, quasi-static, NERO MJCF arms, config.tau_limit() torque, 3 kg payload, no claw rows (user 11-02), "
            f"admit u <= {E.U_ADMIT} MODEL, USER 11-09)")      # carry_cert.static_cell's row_guar for T_cap=None
SUPPORT_GUAR = ("proof under MODEL (rigid object on a flat support: at rest iff the total COM is over the contact "
                "rectangle (pot: bottom disc); placement at the nominal destination, placement error NOT covered)")   # carry_cert.support_row's text


def support_top(task: str, world) -> dict:
    """the destination support top {top, x, y} of the task's world (legacy: shelf for the box, counter for trays/pot, counter2 of the swerve scene for table-behind)."""
    t = TK.task(task)
    if t.base:
        from world_model.v2.stack.planners import base_route as BR
        return BR.default_backend().supports(world)[t.goal_support]
    return SC.support_tops(world)[t.goal_support]


def plan_end_xy(task: str, world, wp):
    """MODEL: world xy of the carried object at the END of the compiled plan of a table-behind task: the run world's object start xy + the planned object offset (waypoint nodes, arm/base
    frame), mapped through the base route's end pose from the geometry backend's start pose (the transform closed_chain._x_obj(after_base) applies to the carry).  None when there is no
    waypoint plan or it has no base move."""
    if wp is None or not getattr(wp, "base", None):
        return None
    from world_model.v2.stack.planners import base_route as BR
    be = BR.default_backend()
    p0 = be.start_pose(world)
    after, route, _lv = wp.base[0]
    pe = BR.route_end(p0, BR.normalize_route(route))
    dl = float(pe[2]) - float(p0[2])
    x_end = np.asarray(world.obj_xy, float)[:2] + np.asarray(wp.nodes()[-1], float)[:2]
    return np.asarray(pe[:2], float) + BR.rot(dl) @ (x_end - np.asarray(p0[:2], float))


def admit_static(task: str, cell, world, spec, path: AD.PathSamples, planner: str, wp=None) -> dict:
    t = TK.task(task)
    be = PE.EnvelopeBackend(world)
    sup = support_top(task, world)
    if t.base:
        d = np.asarray(path.disp, float).copy()
        end_xy = plan_end_xy(task, world, wp)                               # USER 11-17: the support row at the PLAN's end pose
        if end_xy is None:                                                  # no plan end pose = MISSING = refused (never an admission)
            from world_model.v2.nero.contents import carry_cert as CCT
            return CCT.refused_static(task, cell, spec, "support_dest: the plan's end pose is not available (MISSING): " + ("no waypoint plan" if wp is None else "no base move in the plan"))
        d[:2] = end_xy - be.o_ref()[:2]
        path = AD.PathSamples(path.offs, path.per_seg, path.cap, d)
    geo = None
    if t.base:
        from world_model.v2.stack.planners import base_route as BR
        geo = BR.default_backend().support_geo(world, t.goal_support)
    adm = AD.admit(t, spec, path, be, (sup["x"], sup["y"], geo) if geo else (sup["x"], sup["y"]))
    rows = {}
    by_name = {r.name[len("torque_payload["):-1]: r for r in adm.rows if r.name.startswith("torque_payload[")}
    for seg, rr in adm.seg_rows.items():
        if seg == "release":
            continue
        rows[seg] = {}
        for name, r in rr.items():
            if name == "torque_payload":
                n = r["n_samples"]
                w = dict(r["worst"])
                if "a_vertex" in by_name[seg].worst:     # legacy worst["a"] = the binding acceleration-box vertex (payload.torque_row keeps it as "a_vertex")
                    w["a"] = by_name[seg].worst["a_vertex"]
                rows[seg][name] = dict(r, worst=w, guarantee=ROW_GUAR + f" over the cell x acceleration box at {n} path samples")
            elif name == "open_content":
                rows[seg][name] = dict(r, a_open=round(float(r["a_open"]), 4), guarantee="proof under MODEL (quasi-static liquid surface; slosh NOT covered)")
    sup_row = next(r for r in adm.rows if r.name == "support_dest")
    sd = dict(ok=sup_row.ok, margin_m=round(float(sup_row.margin), 4), worst=(dict(m=sup_row.worst["m"], c=sup_row.worst["c"]) if "m" in sup_row.worst else None),
              support=sup, region=sup_row.worst["region"], guarantee=SUPPORT_GUAR)
    if sup_row.worst["shape"]["kind"] == "Disc":
        sd["disc"] = dict(centre=[round(float(v), 4) for v in sup_row.assumptions["dest_xy"]], radius=round(float(sup_row.worst["shape"]["r"]), 4))
    rows["release"] = dict(support_dest=sd)
    last = list(path.per_seg)[-1] if wp is not None else "place"       # as static_cell
    rows.setdefault(last, {})["support_dest"] = sd
    out = dict(task=task, cell=list(cell), contents_m=list(adm.contents_m), action=adm.action, a_admit=adm.a_admit, a_open=adm.a_open,
               u_nom=round(adm.u_nom, 4), worst_nom=adm.worst_nom, u_one_nom=round(adm.u_one_nom, 4), worst_one=adm.worst_one, T_cap=None, rows=rows)
    if planner != "scripted":
        out["planner"] = planner
    if wp is not None:
        out["wp"] = wp.key_data()
    return out
