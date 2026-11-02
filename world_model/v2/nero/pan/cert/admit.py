"""pan/cert/admit.py -- admission of a segment with a stored backup (exists b, for all theta).  SIMULATION ONLY.

segment_trials(ep, seg): in ONE world (one theta), run the segment, snapshot branch points (every BRANCH_DT in motion,
BRANCH_DT_HOLD in holds, every contact event, the end), and from every branch point run every backup of the family.
A family member b passes the world if every branch ends safe (no U) and in R_V2.

certify(records, seg): over many worlds (the certification samples of a belief cell) pick the longest b that passes ALL of
them -> the stored backup, or a counterexample (theta, t, why).  Guarantee type: sampled (n stated).
"""
from __future__ import annotations

import numpy as np

from .. import segments as S
from . import backups as BK

BRANCH_DT, BRANCH_DT_HOLD = 0.2, 0.5   # design: branch spacing [s] (v1 used 0.1; backups are now short, events still every 0.1 s)
R_VERSION = "v2"


def segment_trials(ep, seg, family=BK.FAMILY, branch_dt=BRANCH_DT, monitor=None, perturb_lid=0.0, rng=None):
    """-> record dict for this segment in this world; leaves ep at the segment's end (or abort point).
    perturb_lid: if > 0, each branch is also re-run with the lid shifted by +-perturb_lid [m] (chaos check)."""
    pre_k = ep.k
    ep.reset_track()
    ep.sense["pan_at_push_start"] = ep.d.xpos[ep.pan].copy()
    rule = S._branch_rule(branch_dt, max(branch_dt, BRANCH_DT_HOLD))
    status, info = ep.run_phases(ep.segment_phases(seg), branch=rule, monitor=monitor)
    nominal = dict(status=status, viol=ep.track["viol"], margin=round(ep.track["margin"], 4),
                   abort=info.get("t") and dict(t=info["t"], why=info["why"]))
    if status == "done":
        nominal["success"] = ep.success(seg)
        nominal["observed"] = ep.observe_boundary(seg)
    end = ep.snapshot("end")
    snaps = info["snaps"] + [end]
    per = {float(d): [] for d in family}
    la = ep.m.jnt_qposadr[ep.m.body(ep.lid).jntadr[0]]
    for sn in snaps:
        for d in family:
            variants = [0.0] + ([perturb_lid, -perturb_lid] if perturb_lid else [])
            for dl in variants:
                ep.restore(sn)
                if dl:
                    import mujoco
                    ep.d.qpos[la] += dl
                    mujoco.mj_forward(ep.m, ep.d)
                ep.reset_track()
                bk = BK.make(ep, seg, d)
                ep.run_phases(list(bk.phases))
                ok_R, mg, st = ep.in_R(reach=True, version=R_VERSION)
                per[float(d)].append(dict(t=round(sn["t"], 3), label=sn["label"], lid_dl=dl, safe=ep.track["viol"] is None,
                                          viol=ep.track["viol"], inR=ok_R, R_fail=[k for k, v in mg.items() if v <= 0],
                                          lid_mm=st["lid_mm"]))
    ep.restore(end)
    fam = {}
    for d, rs in per.items():
        bad = [r for r in rs if not (r["safe"] and r["inR"])]
        fam[d] = dict(n=len(rs), n_bad=len(bad), first_bad=bad[0] if bad else None)
    return dict(seg=seg, t0=round(pre_k * S.DT, 3), t1=round(ep.k * S.DT, 3), nominal=nominal, n_branch=len(snaps), family=fam,
                trials={str(d): rs for d, rs in per.items()})


def world_trials(ep, segments=S.SEGMENTS, **kw):
    """start + every segment in order (continues after failures so later segments are also sampled); one theta."""
    out = dict(refused=ep.refused, segments=[])
    if ep.refused:
        return out
    ep.run_phases(ep.segment_phases("start"))
    for seg in segments:
        rec = segment_trials(ep, seg, **kw)
        out["segments"].append(rec)
        if rec["nominal"]["status"] != "done" or rec["nominal"]["viol"]:
            break
    return out


def certify(records, seg, family=BK.FAMILY, static=None, conf=0.95):
    """11-01 (user step 3): STATIC rows (torque/payload, rest support/tilt, open-content limits) come from exact corner checks
    (contents/static_cert.static_rows: proof under MODEL); the sampled worlds cover only the DYNAMIC effects.  The result reports
    both, plus a one-sided Clopper-Pearson bound on the dynamic failure rate (i.i.d. worlds from the cell, conf).
    static=None keeps the pre-11-01 behaviour (sampled only)."""
    from ..segcert import cp_upper
    dyn = _certify_sampled(records, seg, family)
    n = dyn["n_worlds"]
    k = 0 if dyn["ok"] else len({c.get("tag") for c in dyn.get("counterexamples", [])})
    dyn["eps_upper"] = round(cp_upper(k, n, conf), 4) if n else 1.0
    dyn["eps_conf"] = conf
    if static is None:
        return dyn
    out = dict(dyn, ok=bool(dyn["ok"] and static["ok"]), static=static,
               guarantee=f"static rows: proof under MODEL ({', '.join(static['rows'])}); dynamic rows: {dyn['guarantee']}, "
                         f"failure rate <= {dyn['eps_upper']} at {conf:.0%} (Clopper-Pearson, i.i.d. worlds)")
    if not static["ok"]:
        out.setdefault("counterexamples", [])
        for name, r in static["rows"].items():
            if not r["ok"]:
                out["counterexamples"].append(dict(where="static", row=name, worst=r.get("worst") or r.get("worst_contents"),
                                                   guarantee=r["guarantee"]))
    return out


def _certify_sampled(records, seg, family=BK.FAMILY):
    """records: world_trials outputs (each with 'theta' and 'tag').  -> dict(ok, backup_retreat, n, counterexample).
    A world where the robot refuses at B0 is not a sample of this segment; a world whose earlier segment failed is not either."""
    worlds = [(r, next((s for s in r["segments"] if s["seg"] == seg), None)) for r in records if not r.get("refused")]
    worlds = [(r, s) for r, s in worlds if s is not None]
    nominal_bad = [(r, s) for r, s in worlds if s["nominal"]["viol"]]
    for d in family:
        bad = [(r, s) for r, s in worlds if s["family"][_key(s["family"], d)]["n_bad"]]
        if not bad and not nominal_bad:
            return dict(ok=True, seg=seg, backup_retreat=float(d), n_worlds=len(worlds), guarantee=f"sampled (n = {len(worlds)} worlds)")
    # no family member passes every world: counterexample from the best member
    best = min(family, key=lambda d: sum(bool(s["family"][_key(s["family"], d)]["n_bad"]) for _, s in worlds))
    cx = []
    for r, s in nominal_bad:
        cx.append(dict(tag=r.get("tag"), theta=r.get("theta"), where="segment", t=s["nominal"]["viol"]["t"], why=s["nominal"]["viol"]["kind"]))
    for r, s in worlds:
        fb = s["family"][_key(s["family"], best)]["first_bad"]
        if fb:
            cx.append(dict(tag=r.get("tag"), theta=r.get("theta"), where="backup", t=fb["t"],
                           why=(fb["viol"] or {}).get("kind") or ",".join(fb["R_fail"])))
    return dict(ok=False, seg=seg, best_retreat=float(best), n_worlds=len(worlds), counterexamples=cx,
                guarantee=f"refuted (counterexample among n = {len(worlds)} worlds)")


def _key(fam, d):
    """family dicts are keyed by float in memory and by str after a JSON round trip."""
    if d in fam:
        return d
    return str(float(d))
