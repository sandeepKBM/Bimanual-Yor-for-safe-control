"""stack/executor.py -- LegacyCarryExecutor: runs a BimanualPlan from planners/scripted.py on a nero/contents/carry.CarryEpisode.
SIMULATION ONLY.  Extracted 2026-11-03 from CarryEpisode.run_phases and carry.run_segments (step 1 of the stack plan); the loops below are the
ONLY copy of that logic; CarryEpisode.run_phases / carry.run_segments delegate here.  Stepping, controller, monitors, snapshot/restore stay
on the episode (CarryEpisode._step / _check / snapshot / restore / _add_path).  No behaviour change.
"""
from __future__ import annotations

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


def run_segments(ep, plan: BimanualPlan, rec, branch_backups=True, t0=None):
    """the segment loop of world_trial on an existing episode (11-02: probe.py runs it after the probe, same episode)."""
    t0 = time.time() if t0 is None else t0
    hold_viol = ep.track["viol"]
    for sg in plan.segments:
        seg = sg.mode
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
        nominal = dict(viol=ep.track["viol"], **{k: round(float(v), 4) for k, v in ep.track.items() if k not in ("viol",)})
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
                ep.run_phases(bk)
                ok, why = ep.in_R()
                fam["n"] += 1
                if not ok:
                    fam["n_bad"] += 1
                    if fam["first_bad"] is None:
                        fam["first_bad"] = dict(t=tr, viol=ep.track["viol"], R_fail=why)
            ep.restore(end)
        rec["segments"].append(dict(seg=seg, nominal=nominal, family={0.0: fam}, fl=fl_nom,
                                    accel=(dict(ep.last_accel) if seg != "release" and getattr(ep, "last_accel", None) else None)))
        if nominal["viol"]:
            break
    rec["pre_hold_viol"] = hold_viol
    rec["truth_end"] = CR._jsonable(SC.object_truth(ep.m, ep.d))
    rec["end_in_R"] = ep.in_R()
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


class LegacyCarryExecutor:
    """types.Executor.  sim.episode = a CarryEpisode; sim.handles: rec (record dict, default {"segments": []}), branch_backups (True), t0."""

    def run(self, plan, sim, monitors=()):
        h = sim.handles
        rec = h.get("rec")
        if rec is None:
            rec = dict(segments=[])
        return run_segments(sim.episode, plan, rec, h.get("branch_backups", True), h.get("t0"))
