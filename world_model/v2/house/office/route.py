"""route.py -- multi-door route planning for the office on top of the existing nav.py primitives.  SIMULATION only.

nav.plan_route handles ONE door between a start and a goal.  The coffee run crosses two (lab -> door1 -> corridor -> door2 -> cafe
and back), so `plan_leg` generalises plan_route's logic (door_passage per door, A* between run-in/run-out points, corner
fillets, time_parameterize, validate_traj) to a door sequence, with optional end headings (straight run-in/out along the
requested heading) -- nav.py itself is not edited.  `plan_mission` chains the full lab_start -> cafe_counter_pick -> lab_desk_place
run with the holonomic hand-offs at the counter (slide back along the heading, then rotate in place) and validates the
concatenated trajectory on the fine grid.
"""
from __future__ import annotations

import math

import numpy as np

from .. import nav as N
from . import scene as S
from . import nav_grid as G

FP = N.DEFAULT_FP


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------------------------------------- one leg through k doors
def plan_leg(F, C, start, goal, doors, fp=FP, vmax=0.3, amax=0.25, jmax=None, dt=0.005, clear_min=0.05, lead=0.6,
             approach=1.2, fillet_radii=(0.6, 0.45, 0.3, 0.2), min_radius=0.2, turn_pen=1.5,
             extras=(0.8, 0.6, 0.4, 0.2, 0.0), pads=(0.0, 0.05, 0.10), search="first", verbose=False):
    """start/goal = (x, y, yaw|None).  doors = door dicts (nav.door_list-like) in crossing order.  Returns a dict like
    nav.plan_route (search="first": stop at the first (pad, run-in) combination that yields a valid trajectory; "all": nav's exhaustive
    search and pick the best by (turns, clearance, length)): out['route'] is None (with out['reason']) when no route; else out['route']['traj'] is a nav.Traj."""
    start = tuple(start); goal = tuple(goal)
    out = dict(start=list(map(float, start[:2])), goal=list(map(float, goal[:2])), fp=fp, route=None, doors=[d["id"] for d in doors],
               res_coarse=C.res, res_fine=F.res)
    # --- door passages (sign from the side the previous point is on)
    dps, prev = [], np.array(start[:2], float)
    for dr in doors:
        c, n = dr["centre"], dr["normal"]
        sign = -1 if float((prev - c) @ n) > 0 else +1
        dp = N.door_passage(F, fp, dr, sign)
        out.setdefault("door_passages", []).append({k: v for k, v in dp.items() if k != "curve"} | dict(door=dr["id"], travel_sign=sign))
        if not dp["ok"]:
            out["reason"] = f"no route: footprint does not fit through {dr['id']} on the 1 cm grid"
            return out
        if dp["clearance"] < clear_min:
            out["reason"] = f"no route: best {dr['id']} clearance {dp['clearance']:.3f} m < {clear_min} m"
            return out
        dps.append(dp)
        prev = c + np.array(dp["normal"]) * 1.0
    # end points with straight run-in / run-out along the requested heading
    s_pt = np.array(start[:2], float)
    g_pt = np.array(goal[:2], float)
    Ps = s_pt + lead * np.array([math.cos(start[2]), math.sin(start[2])]) if start[2] is not None else None
    Pg = g_pt - lead * np.array([math.cos(goal[2]), math.sin(goal[2])]) if goal[2] is not None else None
    fm, tried, cands = {}, [], []
    for pad, extra in [(pp, ee) for pp in pads for ee in extras]:
        if pad not in fm:
            fm[pad] = N.free_maps(C, fp, pad)
        free = fm[pad]
        # run-in / run-out points per door
        pts = []                                     # list of (P1, P3, h_door, n, wd, c, u)
        prev = Ps if Ps is not None else s_pt
        for k, (dr, dp) in enumerate(zip(doors, dps)):
            c, n, wd, u = np.array(dp["centre"]), np.array(dp["normal"]), np.array(dp["wall_dir"]), dp["u"]
            nxt = (np.array(doors[k + 1]["centre"]) if k + 1 < len(doors) else (Pg if Pg is not None else g_pt))
            gs = abs(float((prev - c) @ n)); ge = abs(float((nxt - c) @ n))
            lat_s = abs(float((prev - c) @ wd) - u); lat_g = abs(float((nxt - c) @ wd) - u)
            di = min(approach + extra, max(gs - 0.5, 0.6)) if lat_s < 0.5 else approach + extra
            de = min(approach + extra, max(ge - 0.5, 0.6)) if lat_g < 0.5 else approach + extra
            P1, P3 = c + wd * u - n * di, c + wd * u + n * de
            h = int(round(dp["heading"] / (2 * math.pi / N.NTH))) % N.NTH
            pts.append((P1, P3, h, n, wd, c, u))
            prev = P3
        # A* legs
        seq = [("s", (Ps if Ps is not None else s_pt), None)]
        legs, ok = [], True
        cur = (Ps if Ps is not None else s_pt, None)
        for (P1, P3, h, n, wd, c, u) in pts:
            nodes = N.astar(C, free, (cur[0][0], cur[0][1], cur[1]), (P1[0], P1[1], h), turn_pen)
            if nodes is None:
                ok = False
                break
            legs.append((nodes, P1, n))
            cur = (P3, h)
        if ok:
            gp = Pg if Pg is not None else g_pt
            nodes = N.astar(C, free, (cur[0][0], cur[0][1], cur[1]), (gp[0], gp[1], None), turn_pen)
            if nodes is None:
                ok = False
            else:
                legs.append((nodes, None, None))
        if not ok:
            tried.append(dict(pad=pad, extra=extra, astar=False))
            continue
        # polyline
        verts, keep = [tuple(s_pt)], {0}
        if Ps is not None:
            verts.append(tuple(Ps)); keep.add(1)
        for li, (nodes, P1, n) in enumerate(legs):
            vv = N.corners(nodes)
            inner = vv[1:-1]
            if P1 is not None and len(inner) >= 1:
                inner[-1] = N._snap_to_line(inner[-1], P1, n)
            if li > 0 and len(inner) >= 1:                                   # first corner after the previous P3: snap onto the door line
                Pp3, n_prev = pts[li - 1][1], pts[li - 1][3]
                inner[0] = N._snap_to_line(inner[0], Pp3, n_prev)
            verts += [tuple(v) for v in inner]
            if P1 is not None:
                verts.append(tuple(P1)); keep.add(len(verts) - 1)
                P3 = pts[li][1]
                verts.append(tuple(P3)); keep.add(len(verts) - 1)
        if Pg is not None:
            verts.append(tuple(Pg)); keep.add(len(verts) - 1)
        verts.append(tuple(g_pt)); keep.add(len(verts) - 1)
        verts = N.clean_polyline(verts, keep=keep)
        # the endpoints must stay exactly where requested (clean_polyline keeps indices of `keep`, positions unchanged)
        for R in fillet_radii:
            Q, s_, info = N.fillet(verts, R)
            traj = N.time_parameterize(Q, s_, vmax, amax, jmax, dt=dt)
            col, cl, ix = N.validate_traj(F, fp, traj)
            rmin = min([i["radius"] for i in info] or [9.0])
            ok2 = (not col.any()) and rmin >= 0.99 * min_radius and float(cl.min()) >= clear_min
            tried.append(dict(pad=pad, extra=extra, R=R, colliding=int(col.sum()), min_clear=float(cl.min()), min_corner_r=rmin,
                              length=traj.meta["length_m"], ok=ok2))
            if verbose:
                print(f"  pad={pad} extra={extra} R={R}: colliding {int(col.sum())}/{len(col)} min clear {cl.min():.3f} "
                      f"min corner r {rmin:.2f} len {traj.meta['length_m']:.2f} ok={ok2}", flush=True)
            if ok2:
                cands.append(dict(pad=pad, extra=extra, R=R, info=info, traj=traj, col=col, cl=cl, ix=ix, verts=verts,
                                  turns=len([i for i in info if abs(i["turn_deg"]) > 10])))
        if cands and search == "first":
            break
    out["tried"] = tried
    if not cands:
        out["reason"] = "no route: no collision-free trajectory with >= %.2f m footprint clearance and corner radius >= %.2f m" % (clear_min, min_radius)
        return out
    best = min(cands, key=lambda q: (q["turns"], -min(float(q["cl"].min()), 0.15), q["traj"].meta["length_m"]))
    traj, cl, ix = best["traj"], best["cl"], best["ix"]
    dv = traj.derivs()
    door_zone = {}
    for dr in doors:
        near = np.hypot(traj.x[ix] - dr["centre"][0], traj.y[ix] - dr["centre"][1]) < 1.3
        door_zone[dr["id"]] = float(cl[near].min()) if near.any() else None
    out["polyline"] = [list(map(float, v)) for v in best["verts"]]
    out["chosen"] = dict(pad=best["pad"], extra=best["extra"], R=best["R"])
    out["route"] = dict(traj=traj, fillet_radius=best["R"], corners=best["info"], n_turns=best["turns"],
                        min_clearance_all_m=float(cl.min()), min_clearance_door_zone_m=door_zone,
                        length_m=traj.meta["length_m"], duration_s=traj.meta["duration_s"], peak_speed=float(dv["speed"].max()),
                        peak_acc=float(dv["acc"].max()), peak_jerk=float(dv["jerk"].max()), peak_wz=float(np.abs(dv["wz"]).max()),
                        end_err_m=float(math.hypot(traj.x[-1] - g_pt[0], traj.y[-1] - g_pt[1])),
                        start_yaw=float(traj.yaw[0]), end_yaw=float(traj.yaw[-1]), caps=traj.caps)
    out["reason"] = "ok"
    return out


# ------------------------------------------------------------------------------------------------- holonomic hand-offs
def slide_traj(p0, p1, yaw, dt=0.005, vmax=0.2, amax=0.15):
    """Straight holonomic translation p0 -> p1 at constant yaw (zero velocity at both ends)."""
    L = float(np.hypot(*(np.asarray(p1) - np.asarray(p0))))
    s = np.append(np.arange(0, L, 0.01), L)
    Q = np.stack([np.interp(s, [0, L], [p0[0], p1[0]]), np.interp(s, [0, L], [p0[1], p1[1]])], 1)
    tr = N.time_parameterize(Q, s, vmax, amax, None, dt=dt)
    tr.yaw = np.full_like(tr.x, yaw)
    return tr


def rotate_traj(x, y, yaw0, yaw1, dt=0.005, wmax=0.4):
    """In-place yaw change (shortest way), min-jerk profile, peak rate ~ wmax."""
    dth = _wrap(yaw1 - yaw0)
    T = max(1.875 * abs(dth) / wmax, 0.5)
    t = np.arange(0, T + 0.3, dt)
    tau = np.clip((t - 0.15) / T, 0, 1)
    sm = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return N.Traj(t, np.full_like(t, x), np.full_like(t, y), yaw0 + dth * sm, dict(wmax=wmax), meta=dict(dt=dt, duration_s=float(t[-1])))


def concat(trajs):
    """Concatenate Trajs (same dt); returns (Traj, [(name, i0, i1), ...])."""
    t_off, T, X, Y, Yw, marks, n = 0.0, [], [], [], [], [], 0
    for name, tr in trajs:
        dt = float(tr.t[1] - tr.t[0])
        T.append(tr.t + t_off); X.append(tr.x); Y.append(tr.y); Yw.append(np.unwrap(tr.yaw))
        marks.append((name, n, n + len(tr.t)))
        n += len(tr.t)
        t_off += float(tr.t[-1]) + dt
    yaw = np.concatenate(Yw)
    # unwrap across segment joins, tolerant of 2pi jumps
    yaw = np.unwrap(yaw)
    big = trajs[0][1]
    return N.Traj(np.concatenate(T), np.concatenate(X), np.concatenate(Y), yaw, big.caps, meta=dict(dt=float(big.t[1] - big.t[0]),
                  duration_s=float(t_off), segments=marks)), marks


def rotation_ok(F, fp, x, y, yaw0, yaw1, clear_min=0.05, n=60):
    """In-place sweep yaw0 -> yaw1 (shortest) is collision-free with `clear_min` clearance on the fine grid."""
    d = _wrap(yaw1 - yaw0)
    P = np.array([[x, y, yaw0 + d * k / n] for k in range(n + 1)])
    col, cl = N.poses_clearance(F, fp, P)
    return (not col.any()) and float(cl.min()) >= clear_min, float(cl.min())


# ------------------------------------------------------------------------------------------------- the whole coffee run
def plan_mission(F=None, C=None, meta=None, fp=FP, vmax=0.3, amax=0.25, dt=0.005, backoff=0.7, clear_min=0.05, verbose=True):
    """lab_start -> door1 -> door2 -> cafe_counter_pick  |  slide back, rotate  |  -> door2 -> door1 -> lab_desk_place.
    Returns dict(legs=[...plan_leg outputs], traj=<concatenated nav.Traj>, marks=[(name,i0,i1)], validation=dict(...))."""
    meta = meta or S.office_meta()
    F = F or G.build_office_grid()
    C = C or N.coarsen(F, 5)
    doors = {d["id"]: d for d in G.office_doors()}
    w = meta["waypoints"]
    ls, pk, dk = w["lab_start"], w["cafe_counter_pick"], w["lab_desk_place"]
    legA = plan_leg(F, C, (*ls["xy"], ls["heading"]), (*pk["xy"], pk["heading"]), [doors["door1"], doors["door2"]], fp, vmax, amax,
                    dt=dt, clear_min=clear_min, verbose=verbose)
    res = dict(legs=dict(A=legA), ok=False)
    if legA["route"] is None:
        res["reason"] = "leg A: " + legA["reason"]
        return res
    # back off from the counter along -heading (holonomic slide), then rotate towards the route's first tangent
    h = pk["heading"]
    pb = np.array(pk["xy"]) - backoff * np.array([math.cos(h), math.sin(h)])
    legB = plan_leg(F, C, (pb[0], pb[1], None), (*dk["xy"], dk["heading"]), [doors["door2"], doors["door1"]], fp, vmax, amax,
                    dt=dt, clear_min=clear_min, verbose=verbose)
    res["legs"]["B"] = legB
    if legB["route"] is None:
        res["reason"] = "leg B: " + legB["reason"]
        return res
    trA, trB = legA["route"]["traj"], legB["route"]["traj"]
    ok_rot, rot_clear = rotation_ok(F, fp, pb[0], pb[1], h, float(trB.yaw[0]), clear_min)
    # the slide back must be clear too
    s_back = slide_traj(pk["xy"], pb, h, dt)
    segs = [("A_drive_to_counter", trA), ("slide_back", s_back), ("rotate", rotate_traj(pb[0], pb[1], h, float(trB.yaw[0]), dt)),
            ("B_drive_to_desk", trB)]
    full, marks = concat(segs)
    col, cl, ix = N.validate_traj(F, fp, full, every=4)
    res.update(ok=bool((not col.any()) and float(cl.min()) >= clear_min and ok_rot), traj=full, marks=marks, backoff_pose=pb.tolist(),
               validation=dict(n_poses=int(len(ix)), colliding=int(col.sum()), min_clearance_m=float(cl.min()),
                               rotation_ok=bool(ok_rot), rotation_min_clearance_m=rot_clear,
                               door_zone_min_clearance_m={"A": legA["route"]["min_clearance_door_zone_m"],
                                                          "B": legB["route"]["min_clearance_door_zone_m"]},
                               duration_s=float(full.t[-1]), length_A_m=legA["route"]["length_m"], length_B_m=legB["route"]["length_m"]))
    return res
