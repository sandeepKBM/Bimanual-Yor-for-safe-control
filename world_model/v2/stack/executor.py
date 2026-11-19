"""stack/executor.py -- LegacyCarryExecutor: runs a BimanualPlan from planners/scripted.py on a nero/contents/carry.CarryEpisode.
SIMULATION ONLY.  Extracted 2026-11-03 from CarryEpisode.run_phases and carry.run_segments (step 1 of the stack plan); the loops below are the
ONLY copy of that logic; CarryEpisode.run_phases / carry.run_segments delegate here.  Stepping, controller, monitors, snapshot/restore stay
on the episode (CarryEpisode._step / _check / snapshot / restore / _add_path).

11-17 (USER rulings, executor phase):
  * ("program", spec)   a Rule 1 recovery program as plain JSON (adapters/rule1_contents.Ctx.program_spec): fixed (the stored backup) or read_then_retrace (stop, hold + read the claw
                        load, pick the verified bin, retrace / advance at the bin's acceleration, the stored backup's tail).
  * ("transfer", rec)   a Rule 4 transfer (stop, hold, retrace, set-down tail) from an accepted rule4 admission record.
  * Rule 3 at run time  cover(theta_new, admission, t_branch): continue / switch (re-arm) / fallback (run now); a contradiction with no certified recovery goes to Rule 4 (injected
                        handles["rule4"], offline mock proposer) and then to the UNCERTIFIED cascade (retrace at A_MIN, the stored backup at A_BACKUP, hold) -- every such run is
                        certified False and counted (rec["rule3"]["n_uncertified"]).
  * certificates        run_segments(certs={step index: Rule 1 record}) binds each step's program to the plan (plan_key) and arms it as the family / interrupt recovery.
  * path stop           a stop during a closed_chain piece follows the planned path (arc speed falls linearly to 0 over T), not the Hermite curve in joint space.
Layering: this module imports no adapters and no new nero.contents module; physics reads go through episode methods (read_claw_load, landed_safe), Rule 4 through handles["rule4"].
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np

from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.stack import theta as TH
from world_model.v2.stack.types import SIDES, BimanualPlan


def _tup(ph):
    """one JSON phase (lists) -> the tuple form run_phases dispatches on (nested lists -> tuples)."""
    return tuple(tuple(x) if isinstance(x, list) else x for x in ph)


def _tups(phs):
    return [_tup(p) for p in phs]


# ------------------------------------------------------------------------------------------------------------------------------ path stop (11-17, R8)
def _path_stop(ep, t, dur):
    """a stop while a closed_chain piece (track / retrace / advance) is being tracked follows the PLANNED PATH: the object's arc speed falls linearly from v0 to 0 over dur
    (s = s0 + v0 tau - v0 tau^2 / 2 dur), the joint reference is the piece's own q at that arc (linear interpolation).  It ends at s0 + 0.5 v0 dur, which is inside Rule 1's checked span.
    v0 = Rule 1's formula (norm of np.gradient(pc.p, DT) at round(t / DT)).  -> True if handled (no active piece, a joint-only piece, or rest: False)."""
    ap = getattr(ep, "_active_piece", None)
    if ap is None:
        return False
    pc, t0 = ap
    n = pc.n
    kk = int(round((t - t0) / CR.DT))
    if kk <= 0 or kk >= n:
        return False
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pc.p, axis=0), axis=1))]
    if arc[-1] < 1e-9:
        return False                                              # joint-only piece (the object does not move): the Hermite stop
    v0 = float(np.linalg.norm(np.gradient(pc.p, CR.DT, axis=0), axis=1)[kk])
    if v0 <= 1e-9:
        return False
    s0 = float(arc[kk])
    tau = np.linspace(0.0, dur, 31)
    s = np.minimum(s0 + v0 * tau - v0 * tau * tau / (2.0 * dur), arc[-1])
    idx = np.interp(s, arc + np.arange(n + 1) * 1e-12, np.arange(n + 1, dtype=float))
    i0 = np.minimum(np.floor(idx).astype(int), n - 1)
    f = (idx - i0)[:, None]
    for sd in SIDES:
        q = (1.0 - f) * pc.q[sd][i0] + f * pc.q[sd][i0 + 1]
        ep._trim(sd, t)
        ep.paths[sd].append((t, t + dur, [q[i] for i in range(len(tau))]))
    ep._active_piece = None
    return True


# ------------------------------------------------------------------------------------------------------------------------------ phases
def run_phases(ep, phases, branch=None):
    """phases: ("move", delta, T) both claws | ("to", key, T) both claws back to stored pose `key` | ("hold", T) | ("stop", T)
    | ("weld", on) | ("retreat", r, T) | ("track", seg) | ("retrace", segs[, a]) | ("advance", segs[, a]) | ("jretreat", where) | ("base_move", ref) | ("base_stop", max_s)
    | ("program", spec) | ("transfer", rec).  branch(ep, t_rel) -> called every 0.25 s inside a move (snapshot hooks)."""
    DT = CR.DT
    for ph in phases:
        t = ep.k * DT
        if ph[0] in ("move", "to", "retreat", "jretreat", "base_move", "base_stop"):
            ep._active_piece = None                                # a stop during these is the Hermite stop
        if ph[0] == "weld":
            for s in SIDES:
                ep.weld(s, ph[1])
            ep.ctl.on = bool(ph[1])                        # payload model off once the object is let go
            continue
        if ph[0] == "program":
            _run_program(ep, ph[1], branch)
            continue
        if ph[0] == "transfer":
            _run_transfer(ep, ph[1], branch)
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
            ep._active_piece = (pc, t)
        elif ph[0] in ("retrace", "advance"):                   # closed_chain backup: the planned joint path reversed (retrace) / forward (advance), rest to rest per segment
            a_ = float(ph[2]) if len(ph) > 2 and ph[2] is not None else CR.A_BACKUP      # 11-17: optional 3rd element = the piece acceleration (default A_BACKUP: identical to before)
            q_now = {s: ep._cmd(s, t) for s in SIDES}
            pieces = (ep.cc.retrace if ph[0] == "retrace" else ep.cc.advance)(q_now, ph[1], a_)
            for pc in pieces:
                t = ep.k * DT
                dur = pc.n * DT
                for s in SIDES:
                    ep._trim(s, t)
                    ep._add_joint_path(s, t, t + dur, pc.q[s])
                ep._active_piece = (pc, t)
                for i in range(pc.n):
                    ep._step()
            continue
        elif ph[0] == "jretreat":                               # closed_chain release retreat: planned claw translation (weld already off)
            rt_ = ep.cc.retreat(ph[1])
            dur = rt_["T"]
            for s in SIDES:
                ep._add_joint_path(s, t, t + dur, rt_["q"][s])
        elif ph[0] == "stop":
            dur = ph[1]
            # 11-16: a stop in the LAST T seconds of a move lets the move finish.  ASSUMPTION (rest-to-rest): every reference entry in ep.paths ends at rest (TOPP-RA pieces,
            # smoothstep moves, retreats, Hermite stops), so the reference itself decelerates to rest at its end knot (_cmd holds the last waypoint after t1).  The fixed-T Hermite
            # stop below ran PAST the end of every move by up to a * T^2 / 8 when started within T of the end.  Rule 1 / Rule 4 model the identical rule.
            rem = {}
            for s in SIDES:
                ent = None
                for e in ep.paths[s]:                     # the entry _cmd uses at t: the LAST (t0, t1, wp) with t0 <= t (later entries override earlier ones)
                    if e[0] <= t:
                        ent = e
                    else:
                        break
                rem[s] = None if ent is None else float(ent[1] - t)
            finish = all(r is not None and 1e-9 < r <= dur + 1e-9 for r in rem.values())
            if getattr(ep, "stop_log", None) is None:
                ep.stop_log = []
            mode = "finish" if finish else "hermite"
            if not finish and _path_stop(ep, t, dur):
                mode = "path"
            ep.stop_log.append(dict(t=t, mode=mode, t_rem=rem, dur=dur))
            if mode == "hermite":
                for s in SIDES:                           # decelerate the COMMANDED reference (11-02: starting from the measured
                    q = ep._cmd(s, t)                     # q baked the wrist sag into the reference; repeated -> 10 deg box tilt)
                    v = (ep._cmd(s, t + 1e-3) - q) / 1e-3 * dur
                    ep._trim(s, t)
                    q1 = q + 0.5 * v
                    wp = [(2 * a ** 3 - 3 * a ** 2 + 1) * q + (a ** 3 - 2 * a ** 2 + a) * v + (-2 * a ** 3 + 3 * a ** 2) * q1
                          for a in np.linspace(0, 1, 31)]
                    ep.paths[s].append((t, t + dur, wp))
        else:
            dur = ph[1]
        n = int(round(dur / DT))
        br = getattr(ep, "_bread", None)
        if ph[0] == "hold" and br is not None and n >= 1:      # 11-17 boundary read: the claw load over the last steps of the segment-end hold (armed for the nominal run only)
            br["reads"].append(ep.read_claw_load(dur, br["n_read"]))
            continue
        for i in range(n):
            if branch is not None and ph[0] in ("move", "track") and i > 0 and i % int(0.25 / DT) == 0:
                branch(ep, i * DT)
            ep._step()


# ------------------------------------------------------------------------------------------------------------------------------ runtime context + Rule 3
def _rt(ep):
    return getattr(ep, "_rt", None) or {}


def _log(ep):
    if getattr(ep, "program_log", None) is None:
        ep.program_log = []
    return ep.program_log


def _event(rt, **kw):
    ev = dict(where=rt.get("where"), t_branch=rt.get("t_branch"), seg=rt.get("seg"), **kw)
    if rt.get("events") is not None:
        rt["events"].append(ev)
    return ev


def _rest_index(ep, rt):
    """geometry sample index (closed_chain) nearest the arms' commanded pose: where a stop + read hold ended (rest).  None if there is no plan to ask."""
    sg = rt.get("sg")
    cc = None if sg is None else sg.meta.get("cc")
    if cc is None:
        return None
    t = ep.k * CR.DT
    u, _d = cc._nearest(rt["seg"], {s: ep._cmd(s, t) for s in SIDES})
    return int(cc.geo.bounds[rt["seg"]][0] + int(round(u)))


def _spec_of(phases):
    """[("program", spec)] -> spec; anything else (a plain phase list) -> None."""
    if phases and len(phases) == 1 and phases[0][0] == "program":
        return phases[0][1]
    return None


def _suffix_at_rest(phases):
    """a recovery started from REST (after the stop + read hold): drop its leading stop.  [("program", fixed spec)] -> the same spec without it; a plain phase list likewise;
    a reading program cannot be run from rest (it starts with a stop and a read) -> None."""
    sp = _spec_of(phases)
    if sp is not None:
        if sp.get("kind") != "fixed":
            return None
        ph = list(sp.get("phases") or [])
        if ph and ph[0][0] == "stop":
            ph = ph[1:]
        return [("program", dict(sp, phases=ph))]
    ph = list(phases or [])
    if ph and ph[0][0] == "stop":
        ph = ph[1:]
    return ph


def _cascade_phases(step, fb_spec, at_rest):
    """USER Q2 cascade: (1) retrace_min = the stored backup's retrace / advance at A_MIN; (2) stored = the stored backup at A_BACKUP; (3) hold in place.  None = cannot be built."""
    if step == "hold":
        return [("hold", 1.0)]
    if fb_spec is None or fb_spec.get("kind") != "fixed" or not fb_spec.get("phases"):
        return None
    ph = [list(p) for p in fb_spec["phases"]]
    if at_rest and ph and ph[0][0] == "stop":
        ph = ph[1:]
    if step == "stored":
        return _tups(ph)
    if step == "retrace_min":
        i = next((i for i, p in enumerate(ph) if p[0] in ("retrace", "advance")), None)
        if i is None or fb_spec.get("a_min") is None:
            return None
        ph[i] = [ph[i][0], ph[i][1], float(fb_spec["a_min"])]
        return _tups(ph)
    raise ValueError(step)


def _safe_ok(ep, sg=None):
    """the SAFE-set test of a recovery landing.  Default: ep.landed_safe() (in_R AND both welds off); a BASE segment's family checks only 'paused' (the base at rest and no U; the full
    safe-set check happens at the end of an interrupt resume, USER Q8)."""
    if not hasattr(ep, "landed_safe"):
        return ep.in_R()
    if sg is not None and "base_ref" in sg.meta:
        bl = getattr(ep, "base_stop_log", None) or []
        why = []
        if not (bl and bl[-1].get("at_rest")):
            why.append("base not at rest")
        if ep.track["viol"]:
            why.append("U:" + ep.track["viol"]["kind"])
        return not why, why
    return ep.landed_safe()


def _decide(ep, rt, theta_new, t_b, rest_at=None):
    """Rule 3 + Rule 4 decision for a belief theta_new that arrived at branch time t_b.  No stepping.  -> dict(action, recovery, why, lookup_s, kind, certified, ...):
    kind 'continue' | 'switch' (spec) | 'fallback' (spec) | 'rule4' (record, r4) | 'uncert'.  Fail closed: anything that cannot be evaluated is 'uncert' (never a pass)."""
    from world_model.v2.stack.checks import cover as COV
    sg = rt.get("sg")
    meta = {} if sg is None else sg.meta
    adm = rt.get("admission")
    out = dict(kind="uncert", certified=False, action="fallback", recovery=None, why="", lookup_s=None)
    if adm is None or theta_new is None:
        out["why"] = "no admission or no theta_new: nothing certified"
    else:
        try:
            dec = COV.cover(theta_new, adm, float(t_b))
            out.update(action=dec.action, recovery=dec.recovery, why=dec.why, lookup_s=dec.lookup_s, certified=bool(dec.certified))
        except Exception as e:
            out["why"] = f"cover raised {e!r}: nothing certified"
        if out["action"] == "continue":
            out.update(kind="continue", certified=True)
            return out
        if out["action"] == "switch":
            ph = (meta.get("recoveries") or {}).get(out["recovery"])
            if ph is not None:
                out.update(kind="switch", phases=ph, certified=True)
                return out
            out["why"] += f"; recovery {out['recovery']!r} has no phases in this segment's meta"
        elif out["action"] == "fallback" and out["recovery"] is not None:
            ph = meta.get("fallback_phases")
            if ph is not None:
                out.update(kind="fallback", phases=ph, certified=True)
                return out
            out["why"] += "; the fallback has no phases in this segment's meta"
        out["certified"] = False
    out.update(kind="uncert", certified=False)
    # --- Rule 4 (injected; offline mock proposer): a certified transfer on theta_new, else the uncertified cascade ---
    r4 = (rt.get("handles") or {}).get("rule4")
    if r4 is not None and theta_new is not None:
        t_call = float(t_b) if np.isfinite(t_b) else rt.get("t_end")
        out["rule4"] = dict(called=t_call is not None)
        if t_call is not None:
            try:
                res = r4(rt["seg"], t_call, theta_new, rest_at=rest_at)
                rec4 = res.get("record") if isinstance(res, dict) else None
                out["rule4"].update(accepted=bool(res.get("accepted")), target=res.get("target"), wall_s=res.get("wall_s"), n_rounds=res.get("n_rounds"), why=res.get("why"))
                if res.get("accepted") and rec4 and rec4.get("transfer_phases"):
                    out.update(kind="rule4", record=rec4, certified=True)
            except Exception as e:
                out["rule4"].update(accepted=False, error=repr(e))
    else:
        out["rule4"] = dict(called=False, why="no rule4 handle" if r4 is None else "disabled / no theta_new")
    return out


def _execute(ep, rt, plan, at_rest, ev):
    """runs the recovery of a _decide plan FOR REAL from the current state; the uncertified cascade when it is not certified or cannot be run.  Fills ev; -> ev."""
    sg = rt.get("sg")
    fb_spec = _spec_of((sg.meta.get("fallback_phases") if sg is not None else None) or [])
    k0 = ep.k
    ran, ok_run = None, False
    if plan["kind"] in ("switch", "fallback"):
        ph = _suffix_at_rest(plan["phases"]) if at_rest else plan["phases"]
        if ph is None:
            ev["note"] = "the recovery cannot be run from rest (a reading program): treated as not runnable"
        else:
            try:
                run_phases(ep, ph)
                ran, ok_run = plan["recovery"] if plan["kind"] == "switch" else "fallback", True
            except Exception as e:
                ev["run_error"] = repr(e)
    elif plan["kind"] == "rule4":
        try:
            run_phases(ep, [("transfer", plan["record"])])
            ran, ok_run = f"rule4:{plan['record'].get('target')}", True
        except Exception as e:
            ev["run_error"] = repr(e)
            plan["certified"] = False
    if not ok_run:
        plan["certified"] = False
        order = ["retrace_min", "stored", "hold"]
        ev["cascade_tried"] = []
        for step in order:
            try:
                ph = _cascade_phases(step, fb_spec, at_rest)
                if ph is None:
                    ev["cascade_tried"].append(dict(step=step, built=False))
                    continue
                run_phases(ep, ph)
                ran = f"uncertified:{step}"
                ev["cascade_tried"].append(dict(step=step, built=True, ran=True))
                break
            except Exception as e:
                ev["cascade_tried"].append(dict(step=step, error=repr(e)))
    ev.update(ran=ran, certified=bool(plan["certified"]), recovery_s=round((ep.k - k0) * CR.DT, 3))
    ev["landed_safe"], ev["safe_why"] = (lambda r: (bool(r[0]), list(r[1])))(_safe_ok(ep, sg))
    return ev


def _decision_json(plan):
    return dict(action=plan.get("action"), recovery=plan.get("recovery"), why=plan.get("why"), lookup_s=plan.get("lookup_s"), plan_kind=plan.get("kind"),
                rule4=plan.get("rule4"))


def _run_marked(ep, phases, log, branch=None):
    """run_phases one phase at a time; the step range [k0, k1) of the retrace / advance phase goes to log['k_motion'] (the window in which the object's acceleration is the bin's a)."""
    for ph in phases:
        k0 = ep.k
        run_phases(ep, [ph], branch)
        if ph[0] in ("retrace", "advance") and "k_motion" not in log:
            log["k_motion"] = (k0, ep.k)


def _run_transfer(ep, rec, branch=None):
    """("transfer", rec): a Rule 4 transfer from an accepted rule4 admission record (adapters/rule4_contents.make_check: transfer_phases, plan_key, state).  Fail closed: a plan key, segment,
    branch time or rest-state mismatch raises ValueError (the caller falls to the uncertified cascade); nothing is run."""
    rt = _rt(ep)
    sg = rt.get("sg")
    cc = (None if sg is None else sg.meta.get("cc")) or getattr(ep, "cc", None)
    st = rec.get("state") or {}
    bad = []
    if not rec.get("transfer_phases"):
        bad.append("no transfer_phases")
    if cc is None or rec.get("plan_key") != dict(geo_key=cc.inp.key(), a_plan=round(float(cc.a_plan), 6)):
        bad.append("plan_key does not match the episode's plan")
    if st.get("seg") != rt.get("seg"):
        bad.append(f"state seg {st.get('seg')!r} != running segment {rt.get('seg')!r}")
    tb = rt.get("t_branch")
    t_want = rt.get("t_end") if (tb is None or not np.isfinite(tb)) else tb
    if t_want is None or abs(float(st.get("t", -1.0)) - float(t_want)) > 1e-6:
        bad.append(f"state t {st.get('t')} != branch time {t_want}")
    if st.get("rest_at") != rt.get("rest_at"):
        bad.append(f"state rest_at {st.get('rest_at')} != {rt.get('rest_at')}")
    if bad:
        raise ValueError("transfer refused: " + "; ".join(bad))
    lg = dict(program="transfer", target=rec.get("target"), stop_T=float(rec["transfer"]["stop_T"]) if rec.get("transfer") else None,
              a=(rec.get("transfer") or {}).get("a"), phases=[p[0] for p in rec["transfer_phases"]], k0=ep.k)
    _log(ep).append(lg)
    _run_marked(ep, _tups(rec["transfer_phases"]), lg, branch)


def _run_program(ep, spec, branch=None):
    """("program", spec).  fixed: its phases.  read: stop (path stop) -> hold + read the claw load -> the first VERIFIED bin containing y (a tie at an edge: the lowest index) -> that bin's
    retrace / advance at its acceleration a_j -> the stored backup's tail.  An invalid read, a y outside every verified bin or a reader exception is a mid-step CONTRADICTION: cover
    (Rule 3) on Theta_0 ∩ slab(y), then Rule 4, then the uncertified cascade."""
    log = dict(program=spec["name"], kind=spec["kind"], k0=ep.k, stop_T=spec.get("stop_T"))
    _log(ep).append(log)
    if spec["kind"] == "fixed":
        log["phases"] = [p[0] for p in spec["phases"]]
        run_phases(ep, _tups(spec["phases"]), branch)
        return
    run_phases(ep, [("stop", float(spec["stop_T"]))], branch)
    rd_spec = spec["read"]
    try:
        rd = ep.read_claw_load(rd_spec["hold_s"], rd_spec["n_read"])
    except Exception as e:
        rd = dict(y=float("nan"), nb=float("nan"), n=0, v_max=float("nan"), valid=False, why=f"reader exception: {e!r}")
    log.update(y=rd["y"], nb=rd["nb"], v_max=rd["v_max"], valid=bool(rd["valid"]), read_why=rd["why"])
    j = None
    if rd["valid"]:
        for b in spec["bins"]:                      # edges: lo <= y <= hi; ties -> the lowest index (first in the list)
            if b["lo"] <= rd["y"] <= b["hi"] and b["verified"] and b["motion"] in ("retrace", "advance") and b["segs"] is not None and b["a"] is not None:
                j = b["j"]
                break
    if j is not None:
        b = spec["bins"][j]
        log.update(bin=j, lo=b["lo"], hi=b["hi"], motion=b["motion"], segs=list(b["segs"]), a=b["a"], tail=[p[0] for p in spec.get("tail_phases", [])])
        _run_marked(ep, [(b["motion"], tuple(b["segs"]), float(b["a"]))] + _tups(spec.get("tail_phases", [])), log, branch)
        return
    _contradiction(ep, spec, rd, log)


def _contradiction(ep, spec, rd, log):
    rt = _rt(ep)
    sg = rt.get("sg")
    log.update(bin=None, contradiction=True)
    theta_new, terr = None, None
    th0 = rt.get("theta_0")
    try:
        if th0 is not None:
            theta_new = TH.theta_from_reading(TH.ReadingSpec.from_json(spec["read"]["spec"]), th0, rd["y"], theta_t=(rt.get("admission").theta_cert if rt.get("admission") is not None else None)) if rd["valid"] else th0     # invalid read: no information, Theta_0
    except Exception as e:
        terr = repr(e)
    ev = _event(rt, kind="read_contradiction", y=rd["y"], valid=bool(rd["valid"]), read_why=rd["why"], theta_new_cells=(None if theta_new is None else len(theta_new.cells)),
                **({"theta_error": terr} if terr else {}))
    rest_at = _rest_index(ep, rt) if sg is not None else None
    rt["rest_at"] = rest_at
    t_b = rt.get("t_branch")
    plan = _decide(ep, dict(rt), theta_new, float("inf") if t_b is None else float(t_b), rest_at=rest_at)
    ev.update(_decision_json(plan))
    if plan["kind"] == "continue":                  # fail closed: a contradiction that cover calls 'continue' means no verified bin held it -> not a pass
        plan = dict(plan, kind="uncert", certified=False, why=plan["why"] + "; no verified bin contains the reading")
    _execute(ep, rt, plan, True, ev)
    log.update(cover_action=ev.get("action"), ran=ev.get("ran"), certified=ev.get("certified"))
    rt["rest_at"] = None


# ------------------------------------------------------------------------------------------------------------------------------ certificates
def _plan_key_of(cc):
    return None if cc is None else dict(geo_key=cc.inp.key(), a_plan=round(float(cc.a_plan), 6))


def _check_certs(plan, certs):
    """-> [problems].  Every plan step with a stored recovery needs a certificate of its step index that is admitted, executable and bound (plan_key) to the plan being run."""
    bad = []
    for pix, sg in enumerate(plan.segments):
        if sg.mode == "release" or sg.meta.get("backup_phases") is None:
            continue
        rc = certs.get(pix)
        if rc is None:
            bad.append(f"step {pix} ({sg.mode}): no certificate")
            continue
        if not rc.get("admitted"):
            bad.append(f"step {pix} ({sg.mode}): the certificate is a refusal")
            continue
        if "base_ref" in sg.meta:
            continue                                                  # the base step runs its own base_stop program (no program_spec)
        spec = rc.get("program_spec")
        if spec is None:
            bad.append(f"step {pix} ({sg.mode}): the certificate has no program_spec")
            continue
        if not spec.get("executable", True) or not rc.get("executable", True):
            bad.append(f"step {pix} ({sg.mode}): program {spec.get('name')!r} is not executable")
            continue
        cc = sg.meta.get("cc")
        pk = rc.get("plan_key")
        if cc is None:
            if spec["kind"] != "fixed" or pk is not None:
                bad.append(f"step {pix} ({sg.mode}): a reading program / a plan_key needs a closed_chain plan")
        elif pk != _plan_key_of(cc):
            bad.append(f"step {pix} ({sg.mode}): plan_key {pk} != the episode's plan {_plan_key_of(cc)}")
    return bad


def _attach(sg, rc):
    """a copy of the segment whose meta carries the certificate (rule1 / program_phases / recoveries / fallback_phases / admission / theta_0 / read_spec)."""
    if rc is None or not rc.get("admitted") or rc.get("program_spec") is None:
        return sg
    from world_model.v2.stack.theta import Admission
    meta = dict(sg.meta)
    meta["rule1"] = {k: rc.get(k) for k in ("step", "seg", "program", "guarantee", "executable", "plan_key", "cell")}
    meta["program_phases"] = [("program", rc["program_spec"])]
    meta["recoveries"] = {n: [("program", sp)] for n, sp in (rc.get("programs_spec") or {}).items()}
    meta["fallback_phases"] = None if rc.get("fallback_spec") is None else [("program", rc["fallback_spec"])]
    meta["admission"] = Admission.from_json(rc["admission"])
    meta["theta_0"] = None if rc.get("theta_0") is None else TH.ThetaSet.from_json(rc["theta_0"])
    meta["read_spec"] = None if rc.get("read_spec") is None else TH.ReadingSpec.from_json(rc["read_spec"])
    sg2 = copy.copy(sg)
    sg2.meta = meta
    return sg2


# ------------------------------------------------------------------------------------------------------------------------------ the segment loop
def _run_family(ep, sg, rt, branch_list, bk, fam, label_end=True):
    """the counterfactual family of one segment: restore each snapshot, run the armed recovery, test the landing (SAFE set).  An exception counts as a bad branch (never a pass)."""
    runs = []
    for tr, sn in branch_list:
        ep.restore(sn)
        ep.reset_track()
        ep.program_log = []
        rt["where"], rt["t_branch"] = "family", (float("inf") if tr is None else float(tr))
        err = None
        try:
            runner = getattr(ep, "backup_runner", None)     # diag_place_backup.py: a logging runner(ep, seg, tr, phases); default = run the phases
            if runner is None:
                ep.run_phases(bk)
            else:
                runner(ep, sg.mode, tr, bk)
        except Exception as e:                              # 11-17: a family run that raises is a BAD branch, never a pass
            err = repr(e)
        ok, why = _safe_ok(ep, sg)
        if err is not None:
            ok, why = False, list(why) + [f"error:{err}"]
        fam["n"] += 1
        if not ok:
            fam["n_bad"] += 1
            if fam["first_bad"] is None:
                fam["first_bad"] = dict(t=tr, viol=ep.track["viol"], R_fail=why)
        if ep.program_log:
            runs.append(dict(t=tr, ok=bool(ok), **{"runs": [dict(x) for x in ep.program_log]}))
    if runs:
        fam["runs"] = runs
    return fam


def run_segments(ep, plan: BimanualPlan, rec, branch_backups=True, t0=None, interrupt=None, certs=None, handles=None):
    """the segment loop of world_trial on an existing episode (11-02: probe.py runs it after the probe, same episode).
    interrupt (11-13 F2, default None): dict(seg=<index into plan.segments | mode name>, t=<s from that segment's start>, resume=False, level=None, [theta_new, admission, stop]).
    After that segment's nominal run + counterfactual family, the stored backup of the latest branch point <= t (11-17; was the nearest) is run FOR REAL from the branch snapshot;
    rec["interrupt"] records the landing.  resume=False stops there; resume=True (base segment + base_stop landing at rest only) drives a continuation route from the landed
    base pose to the nominal end pose and runs the remaining plan segments.
    certs (11-17, default None = the planner's stored backups, no Rule 3 / 4): {step index: the Rule 1 step record of adapters/rule1_contents.run_cell}.  Each step's admitted program
    (stored backup or read_then_retrace) is armed as the family / interrupt recovery; a missing / mismatched certificate refuses the run before it starts (rec['refused']).
    handles: dict(rule4=callable(seg, t_branch, theta_new, rest_at=None) -> dict(accepted, record, target, wall_s)) for Rule 4 at run time (None = Rule 4 missing)."""
    t0 = time.time() if t0 is None else t0
    handles = handles or {}
    hold_viol = ep.track["viol"]
    certs_on = certs is not None
    events = []
    ep._rt = None
    ep._bread = None
    queue = list(plan.segments)
    if certs_on:
        bad = _check_certs(plan, certs)
        rec["rule3"] = dict(certs=True, n_uncertified=0, n_uncertified_family=0, events=[])
        if bad:
            rec["refused"] = "refuse (cert mismatch): " + "; ".join(bad)
            rec["cert_problems"] = bad
            queue = []
    ix, pix = -1, -1
    boundary = None                                                  # the previous segment's boundary read (11-17, USER Q7)
    while queue:
        sg = queue.pop(0)
        ix += 1
        resumed = bool(sg.meta.get("_resumed"))
        if not resumed:
            pix += 1
            if certs_on:
                sg = _attach(sg, certs.get(pix))
        seg = sg.mode
        ep.store(seg)
        ep.reset_track()
        has_cert = certs_on and not resumed and "admission" in sg.meta
        rt = dict(sg=sg, seg=seg, ix=pix, admission=sg.meta.get("admission"), theta_0=sg.meta.get("theta_0"), handles=handles, events=events, where="nominal", t_branch=0.0,
                  t_end=(float(sg.meta["piece"].n * CR.DT) if "piece" in sg.meta else None), rest_at=None) if has_cert else None
        ep._rt = rt
        bk = sg.meta["backup_phases"]
        if has_cert and sg.meta.get("program_phases") is not None:
            bk = sg.meta["program_phases"]
        # ---- boundary read -> Rule 3 at this step's t = 0 (11-17, USER Q7) ----
        if has_cert and boundary is not None and boundary.get("reads") and sg.meta.get("read_spec") is not None and sg.meta.get("theta_0") is not None:
            rd = boundary["reads"][-1]
            bev = dict(step=pix, seg=seg, y=rd["y"], nb=rd["nb"], valid=bool(rd["valid"]), v_max=rd["v_max"])
            rec.setdefault("boundary_reads", []).append(bev)
            if rd["valid"]:
                th_new = TH.theta_from_reading(sg.meta["read_spec"], sg.meta["theta_0"], rd["y"], theta_t=sg.meta["admission"].theta_cert)
                rt["where"], rt["t_branch"] = "boundary", 0.0
                plan_b = _decide(ep, rt, th_new, 0.0)
                bev.update(_decision_json(plan_b))
                if plan_b["kind"] == "switch":
                    bk = plan_b["phases"]
                    bev["armed"] = plan_b["recovery"]
                elif plan_b["kind"] != "continue":                    # fallback / Rule 4 / uncertified: the recovery runs FOR REAL here and the run ends
                    ev = _event(rt, kind="boundary_contradiction", step=pix, y=rd["y"])
                    ep.reset_track()
                    ep.program_log = []
                    kb = ep.k
                    _execute(ep, rt, plan_b, True, ev)
                    bev["ran"], bev["certified"] = ev.get("ran"), ev.get("certified")
                    rec["boundary"] = dict(bev, landing=_landing(ep, kb), event=ev)
                    rt["where"] = "nominal"
                    break
                rt["where"] = "nominal"
        boundary = None
        sn0 = ep.snapshot()
        branches = [(0.0, sn0)]
        bfun = (lambda e, tr: branches.append((tr, e.snapshot()))) if branch_backups and seg != "release" else None
        if "cc" in sg.meta:
            ep.cc = sg.meta["cc"]                              # 11-05: closed_chain plan (planned joint references), opt-in
        if seg != "release":
            ep.last_accel = sg.meta["accel"]                   # 11-03 item 10 (was set inside CarryEpisode.segment_phases)
        if has_cert and seg != "release" and "cc" in sg.meta and pix + 1 < len(plan.segments):
            nxt = certs.get(pix + 1)
            if nxt is not None and nxt.get("admitted") and nxt.get("read_spec") is not None:
                ep._bread = dict(n_read=40, reads=[])           # the hold at the end of this move is read (n_read = rule1_contents.N_READ_RT, ASSUMED)
        ep.run_phases(sg.meta["phases"], branch=bfun)
        boundary, ep._bread = ep._bread, None
        nominal = dict(viol=ep.track["viol"], **{k: round(float(v), 4) for k, v in ep.track.items() if k not in ("viol", "viol_all")})
        nominal["viol_all"] = copy.deepcopy(ep.track["viol_all"])      # 11-12 B1 (additive)
        fl_nom = ep.ctl.rec.summary()          # 11-03 F2: nominal run of this segment (incl. the per-joint torque ratios); backups: nominal-run keys only
        okR, whyR = ep.in_R()
        nominal["end_in_R"], nominal["R_fail"] = okR, whyR
        fam = dict(n_bad=0, first_bad=None, n=0)
        end_sn = None
        if bk is not None and branch_backups:
            end_sn = ep.snapshot()
            fbr = list(branches) if (pix > 0 and not resumed) else list(branches[1:])      # 11-17: t = 0 too after the first step (Rule 1 checks it)
            _run_family(ep, sg, rt if rt is not None else {}, fbr + [(None, end_sn)], bk, fam)
            ep.restore(end_sn)
            ep.program_log = []
            if rt is not None:
                rt["where"], rt["t_branch"] = "nominal", 0.0
        rec["segments"].append(dict(seg=seg, nominal=nominal, family={0.0: fam}, fl=fl_nom,
                                    accel=(dict(ep.last_accel) if seg != "release" and getattr(ep, "last_accel", None) else None),
                                    **(dict(resumed=True) if resumed else {})))
        if interrupt is not None and not resumed and "interrupt" not in rec and (interrupt["seg"] == pix or interrupt["seg"] == seg):
            cont = _interrupt(ep, plan, sg, pix, branches, bk, interrupt, rec, rt, end_sn, dict(handles=handles, events=events))
            if cont is None:
                break
            queue = cont + queue
            continue
        if nominal["viol"]:
            break
    ep._rt = None
    ep._bread = None
    if events and "rule3" not in rec:
        rec["rule3"] = dict(certs=False, n_uncertified=0, n_uncertified_family=0, events=[])
    if "rule3" in rec:
        rec["rule3"]["events"] = [dict(e) for e in events]
        rec["rule3"]["n_uncertified"] = sum(1 for e in events if not e.get("certified", True) and e.get("where") != "family")
        rec["rule3"]["n_uncertified_family"] = sum(1 for e in events if not e.get("certified", True) and e.get("where") == "family")
    rec["pre_hold_viol"] = hold_viol
    rec["truth_end"] = CR._jsonable(SC.object_truth(ep.m, ep.d))
    rec["end_in_R"] = ep.in_R()
    if hasattr(ep, "landed_safe"):
        rec["end_landed_safe"] = ep.landed_safe()
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _landing(ep, k0):
    ok, why = ep.in_R()
    d = dict(in_R=ok, R_fail=why, base_pose=(ep.sb.pose().tolist() if getattr(ep, "base", "") == "swerve" else None),
             obj_truth=CR._jsonable(SC.object_truth(ep.m, ep.d)), viol=ep.track["viol"], viol_all=copy.deepcopy(ep.track["viol_all"]),
             recovery_s=round((ep.k - k0) * CR.DT, 3), t_abs=round(ep.k * CR.DT, 3))
    if hasattr(ep, "landed_safe"):
        ok_s, why_s = ep.landed_safe()
        d.update(landed_safe=bool(ok_s), safe_fail=list(why_s))
    return d


def _interrupt(ep, plan, sg, ix, branches, bk, spec, rec, rt=None, end_sn=None, ctx=None):
    """execute-and-continue: restore the latest branch snapshot <= spec["t"] (11-17), run the armed recovery for real, record the landing.
    spec may carry a belief that arrived at t: theta_new (+ admission; default the segment's certificate) -> Rule 3:
      continue  keep the armed recovery (nothing runs unless spec["stop"]);   switch  re-arm that recovery (the family branches >= t_branch re-run with it: family_after_switch);
      fallback certified on theta_new  run the fallback now (certified);   anything else  Rule 4 (handles["rule4"]) then the uncertified cascade (certified False, counted).
    spec["stop"] (default: True without theta_new, False with it) runs the armed recovery from t_branch.  -> None (stop the run) | list of continuation Segments (resume)."""
    out = rec["interrupt"] = dict(seg_index=ix, seg=sg.mode, t_req=float(spec["t"]), resume_req=bool(spec.get("resume", False)))
    if bk is None or len(branches) == 0:
        out["error"] = "no stored backup (or no branch points) for this segment"
        return None
    t_req = float(spec["t"])
    le = [b for b in branches if b[0] <= t_req + 1e-9]                    # the latest snapshot <= t: a certified state
    tr, sn = max(le, key=lambda b: b[0]) if le else branches[0]
    out["t_branch"] = float(tr)
    theta_new = spec.get("theta_new")
    stop = bool(spec.get("stop", theta_new is None))
    rt_i = dict(rt or {})
    ctx = ctx or {}
    rt_i.update(sg=sg, seg=sg.mode, where="interrupt", t_branch=float(tr), events=ctx.get("events", (rt or {}).get("events", [])), handles=ctx.get("handles", (rt or {}).get("handles", {})),
                t_end=(float(sg.meta["piece"].n * CR.DT) if "piece" in sg.meta else None), rest_at=None)
    if spec.get("admission") is not None:
        rt_i["admission"] = spec["admission"]
    if theta_new is not None:
        ep._rt = rt_i
        plan_i = _decide(ep, rt_i, theta_new, float(tr))
        cv = out["cover"] = dict(t=t_req, t_b=float(tr), **_decision_json(plan_i), certified=bool(plan_i["certified"]))
        if plan_i["kind"] == "continue":
            cv["ran"] = "nothing (admitted recovery kept; nominal run stands)" if not stop else "the armed recovery (stop requested)"
            if not stop:
                ep._rt = rt
                return []
        elif plan_i["kind"] == "switch":
            bk = plan_i["phases"]                                              # re-arm: the later stops use it
            cv["armed"] = plan_i["recovery"]
            fam = dict(n_bad=0, first_bad=None, n=0)
            if end_sn is not None:
                _run_family(ep, sg, rt_i, [b for b in branches if b[0] >= tr - 1e-9] + [(None, end_sn)], bk, fam)
            out["family_after_switch"] = fam
            cv["ran"] = "nothing yet (re-armed)" if not stop else f"re-armed {plan_i['recovery']!r}, then run (stop requested)"
            if not stop:
                if end_sn is not None:
                    ep.restore(end_sn)
                ep._rt = rt
                return []
        else:                                                                   # fallback (certified) / Rule 4 / uncertified: runs now
            ep.restore(sn)
            ep.reset_track()
            ep.program_log = []
            k0 = ep.k
            ep.base_stop_log = []
            rt_i["where"] = "interrupt"
            ev = _event(rt_i, kind="interrupt_cover", t_req=t_req)
            _execute(ep, rt_i, plan_i, False, ev)
            cv.update(ran=ev.get("ran"), certified=ev.get("certified"), cascade_tried=ev.get("cascade_tried"))
            out["landing"] = _landing(ep, k0)
            out["backup"] = [ev.get("ran")]
            ep._rt = rt
            return _after_landing(ep, plan, sg, ix, tr, spec, out)
    ep.restore(sn)
    ep.reset_track()
    k0 = ep.k
    ep.base_stop_log = []
    ep.program_log = []
    rt_i["where"] = "interrupt"
    ep._rt = rt_i if (rt is not None or rt_i.get("admission") is not None) else None
    ep.run_phases(bk)
    ep._rt = rt
    out["landing"] = _landing(ep, k0)
    out["backup"] = [ph[0] for ph in bk]
    if ep.program_log:
        out["program"] = [dict(x) for x in ep.program_log]
    return _after_landing(ep, plan, sg, ix, tr, spec, out)


def _after_landing(ep, plan, sg, ix, tr, spec, out):
    """the resume part of an interrupt (unchanged 11-13 F2 logic): a base segment landed at rest drives the PLANNED remaining route and runs the later plan segments."""
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
        # USER 11-17: resume = finish the PLANNED route from the landed pose
        from world_model.v2.stack.checks import base_stop as BS
        planned = BR.normalize_route(sg.meta["base_ref"]["route"])
        lv = sg.meta["base_ref"].get("level", "L1")
        segs = BS.segment_table(planned, BR.default_backend().levels()[lv], BR.GAP_S)
        i_seg = BS.interrupted_index(segs, float(tr))
        route = BR.remaining_route(BR.route_start(p_end, planned), planned, i_seg, p_now)
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
    """types.Executor.  sim.episode = a CarryEpisode; sim.handles: rec (record dict, default {"segments": []}), branch_backups (True), t0, interrupt, certs, rule4 (11-17)."""

    def run(self, plan, sim, monitors=()):
        h = sim.handles
        rec = h.get("rec")
        if rec is None:
            rec = dict(segments=[])
        return run_segments(sim.episode, plan, rec, h.get("branch_backups", True), h.get("t0"), interrupt=h.get("interrupt"), certs=h.get("certs"), handles=h)
