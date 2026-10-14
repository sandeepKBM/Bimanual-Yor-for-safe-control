"""ocs/evalcore.py -- replay a Recording through the perception producers + the truth producer and extract the E2 / E3 / E4 records.  SIMULATION ONLY.

    process(rec_path, calib_json | None, do=("e2", "e3", "e4"), seed_tag) -> dict of raw records (lists of tuples / dicts), picklable
Heavy part (producers: tracker + depth detector on stored frames) runs once per (recording, calibration); the experiments aggregate the records.
"""
from __future__ import annotations

import math
import time

import numpy as np

from world_model.v2.tray_rep.sets import ir as IR
from world_model.v2.tray_rep.sets.zono import Zonotope

from .state import Kind, Source, WorldState
from . import util as U, consumers as C, dynamics as D, producers as P
from .views import Recording

TICK = 0.02
TAUS = (0.2, 0.5, 1.0, 2.0)
GATE = 1.0
FRAME = "robot"          # E4 evaluates the spec on robot-frame entities (relative statements); consumers.to_ir_entities(frame="world") is the alternative


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def build_states(rec: Recording, calib: P.Calib | None):
    flag = rec.meta["scene"] == "flagship"
    ck = U.cup_kind_for_scene(rec.meta["scene"])          # the cup shape follows the recording's scene (h5: the sim cup; flagship: the paper-cup model)
    pipe = P.perception_pipeline(calib, people=not flag, cup=True, obstacles=flag, names=True, cup_kind=ck)
    tpipe = P.Pipeline([P.TruthProducer(chain=True, with_free=False, cup=ck)])
    pipe.reset()
    tpipe.reset()
    wsp, wst = [], []
    for k in range(rec.n):
        v = rec.view(k)
        wsp.append(pipe.step(v))
        wst.append(tpipe.step(v))
    return wsp, wst


def _in_set(ws: WorldState, oid: str, x, dims=(0, 1, 2)) -> bool:
    Pw = ws.world_pose(oid)
    u = C.world_unc(ws, oid)
    c = np.asarray(Pw.p, float)
    if u.pos is None:
        return bool(np.linalg.norm((np.asarray(x) - c)[list(dims)]) < 1e-9)
    return U.in_zonotope(c + u.pos.c, u.pos.G, x, dims=dims)


def _tilt_err(Ra, Rb):
    a, b = np.asarray(Ra)[:, 2], np.asarray(Rb)[:, 2]
    return float(math.acos(max(-1.0, min(1.0, float(a @ b)))))


# ====================================================================================== E2
def e2_records(rec: Recording, wsp, wst):
    A = rec.A
    out = dict(base=[], tray=[], cup=[], person=[], obstacle=[])
    for k in range(rec.n):
        ws = wsp[k]
        tr = rec.view(k).truth
        bt = np.asarray(tr.base)
        rws = C.robot_frame(ws)
        Rbt = U.rz(bt[2])
        pbt = np.array([bt[0], bt[1], 0.0])
        rel = lambda p: Rbt.T @ (np.asarray(p, float) - pbt)          # true position in the TRUE robot frame
        # ---- base
        bu = ws.robot.base_unc
        be = np.array([ws.robot.base.p[0], ws.robot.base.p[1], ws.robot.yaw])
        e_xy = float(np.hypot(*(be[:2] - bt[:2])))
        e_yaw = float(wrap(be[2] - bt[2]))
        cov_p = U.in_zonotope(np.r_[be[:2], 0.0] + (bu.pos.c if bu.pos is not None else 0.0), bu.pos.G if bu.pos is not None else np.zeros((3, 0)), np.r_[bt[:2], 0.0], dims=(0, 1))
        out["base"].append(dict(k=k, e_xy=e_xy, e_yaw=abs(e_yaw), cov_pos=bool(cov_p), cov_rot=bool(abs(e_yaw) <= bu.rot), sigma=float(rec.A["loc_sigma"][k])))
        # ---- tray
        if "tray" in ws.objects:
            Pw = ws.world_pose("tray")
            Rb_t = U.rz(bt[2])
            p_rel_true = Rb_t.T @ (np.asarray(tr.tray_p) - np.array([bt[0], bt[1], 0.0]))
            e_rel = float(np.linalg.norm(np.asarray(ws.objects["tray"].pose.p) - p_rel_true))
            tilt = _tilt_err(Pw.R, tr.tray_R)
            cp = _in_set(ws, "tray", tr.tray_p)
            u = C.world_unc(ws, "tray")
            out["tray"].append(dict(k=k, e_world=float(np.linalg.norm(Pw.p - tr.tray_p)), e_rel=e_rel, e_tilt=tilt, cov_pos=bool(cp), cov_rot=bool(tilt <= u.rot),
                                    visible=True, cov_pos_rel=bool(_in_set(rws, "tray", rel(tr.tray_p)))))
        # ---- cup
        if "cup" in ws.objects:
            o = ws.objects["cup"]
            Pw = ws.world_pose("cup")
            p_true_tray = np.asarray(tr.tray_R).T @ (np.asarray(tr.cup_p) - np.asarray(tr.tray_p))
            e_tray = float(np.linalg.norm((np.asarray(o.pose.p) - p_true_tray)[:2]))
            tilt_rel = _tilt_err(np.asarray(tr.tray_R).T @ np.asarray(tr.cup_R), np.eye(3))        # cup axis vs tray axis (truth), i.e. error of the 'upright on tray' prior
            tilt = _tilt_err(Pw.R, tr.cup_R)
            cp = _in_set(ws, "cup", tr.cup_p)
            u = C.world_unc(ws, "cup")
            raw = o.extra.get("raw", {})
            out["cup"].append(dict(k=k, e_world=float(np.linalg.norm(Pw.p - tr.cup_p)), e_tray=e_tray, e_tilt=tilt, e_tilt_rel=tilt_rel, cov_pos=bool(cp),
                                   cov_rot=bool(tilt <= u.rot), visible=bool(o.visible), truth_tray=p_true_tray.tolist(), cov_pos_rel=bool(_in_set(rws, "cup", rel(tr.cup_p))),
                                   raw={s: v.tolist() for s, v in raw.items()}, uv={s: list(v) for s, v in o.extra.get('uv', {}).items()}))
        # ---- people (per ground-truth person; nearest track within the gate)
        vis = rec.view(k).visibility()
        trks = [o for o in ws.objects.values() if o.kind == Kind.PERSON and o.source == Source.TRACKER]
        for j, (name, xy, v) in enumerate(tr.people):
            best, bd = None, GATE
            for o in trks:
                d = float(np.linalg.norm(np.asarray(ws.world_pose(o.oid).p)[:2] - xy))
                if d < bd:
                    best, bd = o, d
            prel_true = rel(np.r_[xy, 0.0])
            rec_ = dict(k=k, name=name, speed=float(np.linalg.norm(v)), in_fov=bool(vis[j][0]) if vis is not None else None,
                        los=bool(vis[j][1]) if vis is not None else None, rng_true=float(vis[j][2]) if vis is not None else None, matched=best is not None)
            if best is not None:
                u = C.world_unc(ws, best.oid)
                cov = _in_set(ws, best.oid, np.r_[xy, 0.0], dims=(0, 1))
                infl = max(U.unc_support(u, (math.cos(a), math.sin(a), 0.0)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)) if u.pos is not None else 0.0
                prel_est = np.asarray(best.pose.p, float)                       # robot-frame estimate (parent "robot")
                d_rel = (prel_est - prel_true)[:2]
                Rb_e = np.asarray(ws.robot.base.R)[:2, :2]
                Prel = Rb_e.T @ best.extra["P4"][:2, :2] @ Rb_e
                try:
                    mah = float(math.sqrt(max(d_rel @ np.linalg.solve(Prel + 1e-12 * np.eye(2), d_rel), 0.0)))
                except np.linalg.LinAlgError:
                    mah = float("nan")
                rec_.update(e=bd, cover=bool(cov), infl=float(infl), since=best.extra["since_upd_s"], age=best.extra["age_s"], rng=best.extra["range_m"],
                            track_visible=bool(best.visible), oid=best.oid, e_robot=float(np.linalg.norm(d_rel)), mahal=mah,
                            cover_robot=bool(_in_set(rws, best.oid, prel_true, dims=(0, 1))))
            out["person"].append(rec_)
    # ---- obstacles (flagship): fraction of the true object's boundary samples inside the union of the perceived clusters of that object
    if rec.obstacles and rec.meta["scene"] == "flagship":
        ws = wsp[-1]
        zon = {oid: Zt for oid, Zt in C.to_obstacle_zonos(ws, inflate=True)}
        for key, Zt in rec.obstacles:
            cl = [oid for oid, o in ws.objects.items() if o.kind in (Kind.OBSTACLE, Kind.SURFACE) and o.name == key]
            if not cl:
                out["obstacle"].append(dict(key=key, n_clusters=0, frac=0.0, n_pts=0))
                continue
            # 3x3x3 grid on the true OBB's surface (+ corners): points of Zt
            S = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if max(abs(a), abs(b), abs(c)) == 1], float)
            pts = Zt.c + S @ Zt.G.T
            inside = np.zeros(len(pts), bool)
            inside_nom = np.zeros(len(pts), bool)
            for i, x in enumerate(pts):
                inside[i] = any(U.in_zonotope(zon[o].c, zon[o].G, x) for o in cl)
                inside_nom[i] = any(U.in_zonotope(*_nominal(ws, o), x) for o in cl)
            # front-facing samples: on a face of the true box whose outward normal points toward the camera (the part the stereo pair can see)
            org = np.asarray(rec.zono_obs_result["origin"], float)
            Ginv = np.linalg.inv(Zt.G)
            front = np.zeros(len(pts), bool)
            for i, sgn in enumerate(S):
                for ax in range(3):
                    if sgn[ax] != 0:
                        nrm = Zt.G[:, ax] / np.linalg.norm(Zt.G[:, ax]) * sgn[ax]
                        if (org - pts[i]) @ nrm > 0:
                            front[i] = True
            fr_frac = float(inside[front].mean()) if front.any() else float("nan")
            fr_nom = float(inside_nom[front].mean()) if front.any() else float("nan")
            cen_err = float(np.linalg.norm(np.mean([zon[o].c for o in cl], 0) - Zt.c))
            out["obstacle"].append(dict(key=key, n_clusters=len(cl), frac=float(inside.mean()), frac_nominal=float(inside_nom.mean()), n_pts=len(pts), cen_err=cen_err,
                                        frac_front=fr_frac, frac_front_nominal=fr_nom, n_front=int(front.sum())))
    return out


def _nominal(ws, oid):
    o = ws.objects[oid]
    Pw = ws.world_pose(oid)
    Z = o.shape.zonotope().linear(np.asarray(Pw.R), np.asarray(Pw.p))
    return Z.c, Z.G


# ====================================================================================== E3
def _plan_from_truth(A, k, K, dt=0.1):
    n = len(A["t"])
    tw = []
    for j in range(K):
        kk = min(n - 1, k + int(round(j * dt / TICK)) + int(round(0.5 * dt / TICK)))
        tw.append(A["base_v"][kk])
    return D.RobotPlan(np.array(tw))


def e3_records(rec: Recording, wsp, wst, stride=5, taus=TAUS, dt=0.1):
    A = rec.A
    n = rec.n
    K = int(round(max(taus) / dt))
    res = []
    for k in range(0, n, stride):
        for planned in (False, True):
            plan = _plan_from_truth(A, k, K, dt) if planned else None
            for src, ws in (("perc", wsp[k]), ("truth", wst[k])):
                traj = D.rollout(ws, max(taus), dt, robot_plan=plan)
                for tau in taus:
                    j = int(round(tau / dt))
                    kk = k + int(round(tau / TICK))
                    if kk > n - 1:
                        continue
                    tr = rec.view(kk).truth
                    w = traj[j]
                    # base
                    pb = np.asarray(w.robot.base.p)[:2]
                    bu = w.robot.base_unc
                    cov = U.in_zonotope(np.r_[pb, 0.0] + (bu.pos.c if bu.pos is not None else 0.0), bu.pos.G if bu.pos is not None else np.zeros((3, 0)),
                                        np.r_[tr.base[:2], 0.0], dims=(0, 1))
                    res.append(dict(kind="base", src=src, planned=planned, tau=tau, k=k, e=float(np.hypot(*(pb - tr.base[:2]))), cov_set=bool(cov), vis=True))
                    # tray / cup (world position)
                    for kind, oid, ptrue in (("tray", "tray", tr.tray_p), ("cup", "cup", tr.cup_p)):
                        if oid not in w.objects:
                            continue
                        pw = np.asarray(w.world_pose(oid).p)
                        e = float(np.linalg.norm(pw - ptrue))
                        res.append(dict(kind=kind, src=src, planned=planned, tau=tau, k=k, e=e, cov_set=bool(_in_set(w, oid, ptrue)), vis=bool(ws.objects[oid].visible)))
                    # cup in the tray frame (slip model only)
                    if "cup" in w.objects:
                        ptt = np.asarray(tr.tray_R).T @ (np.asarray(tr.cup_p) - np.asarray(tr.tray_p))
                        Pt, Pc = w.world_pose("tray"), w.world_pose("cup")
                        pp = np.asarray(Pt.R).T @ (np.asarray(Pc.p) - np.asarray(Pt.p))
                        res.append(dict(kind="cup_tray_frame", src=src, planned=planned, tau=tau, k=k, e=float(np.linalg.norm((pp - ptt)[:2])), cov_set=None, vis=True))
                    # people: matched track (perception) or the same-named truth object (truth)
                    tru_by_name = {nm: (xy, v) for nm, xy, v in tr.people}
                    tru0 = {nm: (xy, v) for nm, xy, v in rec.view(k).truth.people}
                    for o in ws.objects.values():
                        if o.kind != Kind.PERSON:
                            continue
                        nm = o.name if src == "perc" else o.oid
                        if nm not in tru_by_name:
                            continue
                        xy_t, _ = tru_by_name[nm]
                        if src == "perc":
                            # gate at time k so a wrong association is not scored as a prediction error
                            if nm not in tru0 or np.linalg.norm(np.asarray(ws.world_pose(o.oid).p)[:2] - tru0[nm][0]) > GATE:
                                continue
                        pw = np.asarray(w.world_pose(o.oid).p)
                        e = float(np.linalg.norm(pw[:2] - xy_t))
                        sp = float(np.linalg.norm(tru0[nm][1])) if nm in tru0 else 0.0
                        res.append(dict(kind="person_moving" if sp > 0.1 else "person_static", src=src, planned=planned, tau=tau, k=k, e=e,
                                        cov_set=bool(_in_set(w, o.oid, np.r_[xy_t, 0.0], dims=(0, 1))), vis=bool(o.visible), name=nm))
    return res


# ====================================================================================== E4
SPEC_MAP = {"h5": dict(laptop=("p_stand",), person=("p_walker",)),
            "flagship": dict(laptop=("monitor_1",), person=("bottle_a",))}
SPEC_ATOMS = ("inside", "upright", "clear_laptop", "clear_person")


def _keys_for(ws: WorldState, names):
    return [o.oid for o in ws.objects.values() if o.name in names or o.oid in names]


def build_e4_spec(ents, ws, mapping, H=None, d_lap=0.1, d_per=0.5, theta=0.1, m_in=0.02):
    """G(inside(cup, tray_top, .02) AND upright(cup, .1) AND clear(cup, laptop, .1) AND clear(cup, person, .5)); the mapped entities may be several
    perceived clusters (all must be cleared).  Returns (spec, atoms dict name -> Node, missing list)."""
    atoms, missing = {}, []
    atoms["inside"] = IR.Inside(ents["cup_rel"], ents["tray_top"], m_in)
    atoms["upright"] = IR.Upright(ents["cup"], theta)
    for lname, d, key in (("laptop", d_lap, "clear_laptop"), ("person", d_per, "clear_person")):
        ks = _keys_for(ws, mapping[lname])
        if not ks:
            missing.append(lname)
            continue
        cl = [IR.Clear(ents["cup"], ents[k], d) for k in ks]
        atoms[key] = cl[0] if len(cl) == 1 else IR.And(*cl)
    body = IR.And(*atoms.values())
    return IR.G(body, 0, 0 if H is None else H), atoms, missing


def _status_of(spec, state):
    nom = IR.margin(spec, state, False)
    rob = IR.margin(spec, state, True)
    st = "false" if nom < 0 else ("true" if rob >= 0 else "unknown")
    return st, float(nom), float(rob)


def e4_records(rec: Recording, wsp, wst, stride=1, pred_stride=5, H=10, dt=0.1):
    mapping = SPEC_MAP[rec.meta["scene"]]
    rows = []
    for k in range(0, rec.n, stride):
        row = dict(k=k)
        for src, ws in (("perc", wsp[k]), ("truth", wst[k])):
            if "cup" not in ws.objects:
                row[src], row[src + "_missing"], row[src + "_atoms"] = ("unknown", float("nan"), float("nan")), ["cup"], {}
                continue
            ents = C.to_ir_entities(ws, frame=FRAME)
            spec, atoms, missing = build_e4_spec(ents, ws, mapping)
            row[src + "_missing"] = missing
            if missing:
                row[src] = ("unknown", float("nan"), float("nan"))
            else:
                row[src] = _status_of(spec, ws)
            # per-atom nominal margins and 3-valued statuses (diagnosis + the 'without inside' conjunction)
            row[src + "_atoms"] = {nm: float(IR.margin(IR.G(a, 0, 0), ws, False)) for nm, a in atoms.items()}
            row[src + "_astat"] = {nm: _status_of(IR.G(a, 0, 0), ws)[0] for nm, a in atoms.items()}
        rows.append(row)
    pred = []
    for k in range(0, rec.n - int(round(H * dt / TICK)), pred_stride):
        row = dict(k=k)
        traj_p = D.rollout(wsp[k], H * dt, dt, robot_plan=None)
        traj_t = [wst[k + int(round(j * dt / TICK))] for j in range(H + 1)]
        for src, tr_ in (("perc", traj_p), ("truth", traj_t)):
            if "cup" not in tr_[0].objects:
                row[src] = ("unknown", float("nan"), float("nan"))
                continue
            ents = C.to_ir_entities(tr_, frame=FRAME)
            spec, atoms, missing = build_e4_spec(ents, tr_[0], mapping, H=H)
            row[src] = ("unknown", float("nan"), float("nan")) if missing else _status_of(spec, tr_)
            row[src + "_astat"] = {nm: _status_of(IR.G(a, 0, H), tr_)[0] for nm, a in atoms.items()}
        pred.append(row)
    return dict(inst=rows, pred=pred, mapping=mapping)


# ====================================================================================== driver
def process(rec_path, calib_json=None, do=("e2", "e3", "e4"), e3_stride=5, tag=""):
    t0 = time.time()
    rec = Recording.load(rec_path)
    calib = P.Calib.from_json(calib_json) if calib_json else None
    wsp, wst = build_states(rec, calib)
    out = dict(path=str(rec_path), seed=rec.meta["seed"], scene=rec.meta["scene"], n=rec.n, tag=tag, t_states=time.time() - t0,
               privileged_perc=sorted({f.split(".")[0] + "." + f.split(".")[-1] for f in wsp[-1].privileged()}))
    if "e2" in do:
        out["e2"] = e2_records(rec, wsp, wst)
    if "e3" in do:
        out["e3"] = e3_records(rec, wsp, wst, stride=e3_stride)
    if "e4" in do:
        out["e4"] = e4_records(rec, wsp, wst)
    out["t_total"] = time.time() - t0
    return out


def process_job(a):
    return process(*a)
