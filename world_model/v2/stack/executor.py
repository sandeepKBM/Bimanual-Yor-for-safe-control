"""stack/executor.py -- LegacyCarryExecutor: runs a BimanualPlan from planners/scripted.py on a nero/contents/carry.CarryEpisode.
SIMULATION ONLY.  Extracted 2026-11-03 from CarryEpisode.run_phases and carry.run_segments (step 1 of the stack plan); the loops below are the
ONLY copy of that logic; CarryEpisode.run_phases / carry.run_segments delegate here.  Stepping, controller, monitors, snapshot/restore stay
on the episode (CarryEpisode._step / _check / snapshot / restore / _add_path).  No behaviour change.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack.types import SIDES, BimanualPlan


def run_phases(ep, phases, branch=None):
    """phases: ("move", delta, T) both claws | ("to", key, T) both claws back to stored pose `key` | ("hold", T) | ("stop", T)
    | ("weld", on) | ("retreat", r, T).  branch(ep, t_rel) -> called every 0.25 s inside a move (snapshot hooks)."""
    DT = CR.DT
    for ph in phases:
        t = ep.k * DT
        if ph[0] == "weld":
            for s in SIDES:
                ep.weld(s, ph[1])
            ep.ctl.on = bool(ph[1])                        # payload model off once the object is let go
            continue
        if ph[0] == "move":
            dur = ph[2]
            for s in SIDES:
                ep._add_path(s, t, t + dur, np.asarray(ph[1], float))
        elif ph[0] == "to":
            tgt = ep.stored[ph[1]]
            dist = max(float(np.linalg.norm(tgt[s] - ep.tcp(s))) for s in SIDES)
            dur = CR.move_time(dist, CR.A_BACKUP) if ph[2] is None else ph[2]
            for s in SIDES:
                ep._trim(s, t)
                ep._add_path(s, t, t + dur, to=tgt[s])
        elif ph[0] == "base_move":                              # table-behind (11-09): a swerve base route; the episode method owns it (stack core imports no nero)
            ep.base_move(ph[1], branch)
            continue
        elif ph[0] == "base_stop":                              # 11-13 F2: abort the base route, step until the base is at rest (ph[1] = max seconds); the episode method owns it
            ep.base_stop(ph[1])
            continue
        elif ph[0] == "retreat":
            dur = ph[2]
            for s in SIDES:
                d_ = np.array([0, ph[1] if s == "left" else -ph[1], 0])
                if getattr(ep, "base", "") == "swerve":         # retreat is outward in the ROBOT frame (a turned base: not world y); fixed base unchanged
                    d_ = ep.body_delta(d_)
                ep._add_path(s, t, t + dur, d_)
        elif ph[0] == "track":                                  # closed_chain plan: the planned joint trajectory of segment ph[1] (no continuation IK)
            pc = ep.cc.segment(ph[1])
            dur = pc.n * DT
            for s in SIDES:
                ep._add_joint_path(s, t, t + dur, pc.q[s], margin=ep.cc.seg_margin_min(ph[1]), err=ep.cc.seg_fk_max(ph[1]))
        elif ph[0] == "retrace":                                # closed_chain backup: planned joint path in reverse, rest to rest per segment
            q_now = {s: ep._cmd(s, t) for s in SIDES}
            for pc in ep.cc.retrace(q_now, ph[1], CR.A_BACKUP):
                t = ep.k * DT
                dur = pc.n * DT
                for s in SIDES:
                    ep._trim(s, t)
                    ep._add_joint_path(s, t, t + dur, pc.q[s])
                for i in range(pc.n):
                    ep._step()
            continue
        elif ph[0] == "advance":                                # closed_chain backup (11-13 F1): the planned path FORWARD from the commanded q to the end of ph[1], rest to rest per segment
            q_now = {s: ep._cmd(s, t) for s in SIDES}
            for pc in ep.cc.advance(q_now, ph[1], CR.A_BACKUP):
                t = ep.k * DT
                dur = pc.n * DT
                for s in SIDES:
                    ep._trim(s, t)
                    ep._add_joint_path(s, t, t + dur, pc.q[s])
                for i in range(pc.n):
                    ep._step()
            continue
        elif ph[0] == "jretreat":                               # closed_chain release retreat: planned claw translation (weld already off)
            rt = ep.cc.retreat(ph[1])
            dur = rt["T"]
            for s in SIDES:
                ep._add_joint_path(s, t, t + dur, rt["q"][s])
        elif ph[0] == "stop":
            dur = ph[1]
            for s in SIDES:                               # decelerate the COMMANDED reference (11-02: starting from the measured
                q = ep._cmd(s, t)                         # q baked the wrist sag into the reference; repeated -> 10 deg box tilt)
                v = (ep._cmd(s, t + 1e-3) - q) / 1e-3 * dur
                ep._trim(s, t)
                q1 = q + 0.5 * v
                wp = [(2 * a ** 3 - 3 * a ** 2 + 1) * q + (a ** 3 - 2 * a ** 2 + a) * v + (-2 * a ** 3 + 3 * a ** 2) * q1
                      for a in np.linspace(0, 1, 31)]
                ep.paths[s].append((t, t + dur, wp))
        else:
            dur = ph[1]
        n = int(round(dur / DT))
        for i in range(n):
            if branch is not None and ph[0] in ("move", "track") and i > 0 and i % int(0.25 / DT) == 0:
                branch(ep, i * DT)
            ep._step()


def run_segments(ep, plan: BimanualPlan, rec, branch_backups=True, t0=None, interrupt=None):
    """the segment loop of world_trial on an existing episode (11-02: probe.py runs it after the probe, same episode).
    interrupt (11-13 F2, default None = bit-identical): dict(seg=<index into plan.segments | mode name>, t=<s from that segment's start>, resume=False,
    level=None).  After that segment's nominal run + counterfactual family, the stored backup of the branch point nearest to t is run FOR REAL from the
    branch snapshot; rec["interrupt"] records the landing.  resume=False stops there; resume=True (base segment + base_stop landing at rest only) drives
    a continuation route from the landed base pose to the nominal end pose and runs the remaining plan segments."""
    t0 = time.time() if t0 is None else t0
    hold_viol = ep.track["viol"]
    queue = list(plan.segments)
    ix = -1
    while queue:
        sg = queue.pop(0)
        ix += 1
        seg = sg.mode
        resumed = bool(sg.meta.get("_resumed"))
        ep.store(seg)
        ep.reset_track()
        sn0 = ep.snapshot()
        branches = [(0.0, sn0)]
        bfun = (lambda e, tr: branches.append((tr, e.snapshot()))) if branch_backups and seg != "release" else None
        if "cc" in sg.meta:
            ep.cc = sg.meta["cc"]                              # 11-05: closed_chain plan (planned joint references), opt-in
        if seg != "release":
            ep.last_accel = sg.meta["accel"]                   # 11-03 item 10 (was set inside CarryEpisode.segment_phases)
        ep.run_phases(sg.meta["phases"], branch=bfun)
        nominal = dict(viol=ep.track["viol"], **{k: round(float(v), 4) for k, v in ep.track.items() if k not in ("viol", "viol_all")})
        nominal["viol_all"] = copy.deepcopy(ep.track["viol_all"])      # 11-12 B1 (additive)
        fl_nom = ep.ctl.rec.summary()          # 11-03 F2: nominal run of this segment (incl. the per-joint torque ratios); backups: nominal-run keys only
        okR, whyR = ep.in_R()
        nominal["end_in_R"], nominal["R_fail"] = okR, whyR
        fam = dict(n_bad=0, first_bad=None, n=0)
        bk = sg.meta["backup_phases"]
        if bk is not None and branch_backups:
            end = ep.snapshot()
            for tr, sn in branches[1:] + [(None, end)]:
                ep.restore(sn)
                ep.reset_track()
                runner = getattr(ep, "backup_runner", None)     # diag_place_backup.py: a logging runner(ep, seg, tr, phases); default = run the phases
                if runner is None:
                    ep.run_phases(bk)
                else:
                    runner(ep, seg, tr, bk)
                ok, why = ep.in_R()
                fam["n"] += 1
                if not ok:
                    fam["n_bad"] += 1
                    if fam["first_bad"] is None:
                        fam["first_bad"] = dict(t=tr, viol=ep.track["viol"], R_fail=why)
            ep.restore(end)
        rec["segments"].append(dict(seg=seg, nominal=nominal, family={0.0: fam}, fl=fl_nom,
                                    accel=(dict(ep.last_accel) if seg != "release" and getattr(ep, "last_accel", None) else None),
                                    **(dict(resumed=True) if resumed else {})))
        if interrupt is not None and not resumed and "interrupt" not in rec and (interrupt["seg"] == ix or interrupt["seg"] == seg):
            cont = _interrupt(ep, plan, sg, ix, branches, bk, interrupt, rec)
            if cont is None:
                break
            queue = cont + queue
            continue
        if nominal["viol"]:
            break
    rec["pre_hold_viol"] = hold_viol
    rec["truth_end"] = CR._jsonable(SC.object_truth(ep.m, ep.d))
    rec["end_in_R"] = ep.in_R()
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _landing(ep, k0):
    ok, why = ep.in_R()
    return dict(in_R=ok, R_fail=why, base_pose=(ep.sb.pose().tolist() if getattr(ep, "base", "") == "swerve" else None),
                obj_truth=CR._jsonable(SC.object_truth(ep.m, ep.d)), viol=ep.track["viol"], viol_all=copy.deepcopy(ep.track["viol_all"]),
                recovery_s=round((ep.k - k0) * CR.DT, 3), t_abs=round(ep.k * CR.DT, 3))


def _interrupt(ep, plan, sg, ix, branches, bk, spec, rec):
    """execute-and-continue: restore the branch snapshot nearest spec["t"], run the stored recovery for real, record the landing.
    -> None (stop the run) | list of continuation Segments (resume)."""
    out = rec["interrupt"] = dict(seg_index=ix, seg=sg.mode, t_req=float(spec["t"]), resume_req=bool(spec.get("resume", False)))
    if bk is None or len(branches) == 0:
        out["error"] = "no stored backup (or no branch points) for this segment"
        return None
    tr, sn = min(branches, key=lambda b: abs(b[0] - float(spec["t"])))
    out["t_branch"] = float(tr)
    ep.restore(sn)
    ep.reset_track()
    k0 = ep.k
    ep.base_stop_log = []
    ep.run_phases(bk)
    out["landing"] = _landing(ep, k0)
    out["backup"] = [ph[0] for ph in bk]
    if not out["resume_req"]:
        return None
    if "base_ref" not in sg.meta or sg.meta.get("end_pose") is None:
        out["resume"] = dict(done=False, why="resume is built for a base segment only (this segment has no base_ref / nominal end pose)")
        return None
    if not any(b["at_rest"] for b in getattr(ep, "base_stop_log", [])[-1:]):
        out["resume"] = dict(done=False, why="the landing is not a base_stop at rest")
        return None
    from world_model.v2.stack.planners import base_route as BR
    p_now, p_end = ep.sb.pose(), sg.meta["end_pose"]
    try:
        route = BR.continuation_route(p_now, p_end)
        rinfo = dict(route=[list(r) for r in route], from_pose=[float(a) for a in p_now], to_pose=[float(a) for a in p_end])
        if not route:
            raise ValueError("already at the nominal end pose")
        cc = sg.meta.get("cc")
        if cc is not None and getattr(ep, "world", None) is not None and ix > 0:
            rinfo["sweep"] = BR.check_continuation(ep.world, cc.geo, plan.segments[ix - 1].mode, p_now, route)
            if not rinfo["sweep"]["ok"]:
                raise ValueError(f"continuation sweep collides ({rinfo['sweep']['min_dist_m'] * 1e3:.1f} mm)")
        else:
            rinfo["sweep"] = "not run (no cc / world)"
    except ValueError as e:
        out["resume"] = dict(done=False, why=str(e))
        return None
    out["resume"] = dict(done=True, **rinfo)
    ref = dict(route=[[k, float(v)] for k, v in route], level=(sg.meta["base_ref"].get("level", "L1") if spec.get("level") is None else spec["level"]))
    csg = copy.copy(sg)
    csg.meta = dict(sg.meta, phases=[("base_move", ref)], base_ref=ref, backup_phases=None, _resumed=True)
    return [csg]


class LegacyCarryExecutor:
    """types.Executor.  sim.episode = a CarryEpisode; sim.handles: rec (record dict, default {"segments": []}), branch_backups (True), t0."""

    def run(self, plan, sim, monitors=()):
        h = sim.handles
        rec = h.get("rec")
        if rec is None:
            rec = dict(segments=[])
        return run_segments(sim.episode, plan, rec, h.get("branch_backups", True), h.get("t0"), interrupt=h.get("interrupt"))
