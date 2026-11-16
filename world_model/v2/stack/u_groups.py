"""u_groups.py -- the U kinds of a run, grouped for the summary tables (11-12 Stage 0a; recording only, no existing number depends on it).

GROUPS: contents = what happened to the load / scene; base = the swerve base flags; rating = the arm / claw rating rows.
`nominal_kinds(rec)` = the U kinds of the NOMINAL path of a record: probe viol, every segment's nominal viol_all (B1) or, for records without it,
the first viol; base U_flags_ever (new) or U_flags (old).  Returns None when the record carries none of the new keys (-> "-" in the summary).
`recovery_of(segments)` = {seg: {n, n_bad}} from each segment's stored backup family (B4 step 1; recorded, not counted as U or in the outcome).
"""
from __future__ import annotations

GROUPS = dict(
    contents=("obj_tilt", "obj_drop", "spill", "cup_tipped", "cup_off_tray", "lid_off", "lid_slide", "load_world_contact", "collision"),
    base=("wheel_liftoff", "base_world_contact", "route_unfinished"),
    rating=("claw_payload", "tau_sat", "claw_twist"),
)
_GROUP_OF = {k: g for g, ks in GROUPS.items() for k in ks}


def group_of(kind):
    return _GROUP_OF.get(kind, "other")


def recovery_of(segments):
    """{seg: {n, n_bad}} over the segments of a carry record (each has family {t: {n, n_bad, ...}}); a segment without a stored backup -> n = 0."""
    out = {}
    for s in segments or []:
        fam = (s.get("family") or {})
        out[s["seg"]] = dict(n=int(sum(f.get("n", 0) for f in fam.values())), n_bad=int(sum(f.get("n_bad", 0) for f in fam.values())))
    return out


def recovery_bad(rec_recovery):
    return any(v["n_bad"] > 0 for v in (rec_recovery or {}).values())


def nominal_kinds(rec):
    """set of U kinds on the nominal path (probe, ramp / hold, segments, base), or None if the record has no viol_all / U_flags_ever
    (old record: the first-U-only data cannot answer 'any kind')."""
    segs = (rec.get("carry") or {}).get("segments") or []
    base = rec.get("base") or {}
    has_new = any("viol_all" in (s.get("nominal") or {}) for s in segs) or "U_flags_ever" in base
    if not has_new:
        return None
    ks = set()
    for v in (rec.get("probe_viol"), (rec.get("carry") or {}).get("pre_hold_viol")):
        if v:
            ks.add(v["kind"])
    for s in segs:
        nom = s.get("nominal") or {}
        ks.update((nom.get("viol_all") or {}).keys())
        if nom.get("viol"):
            ks.add(nom["viol"]["kind"])
    ks.update(base.get("U_flags_ever") or [])
    return ks


def by_group(kinds):
    out = {}
    for k in sorted(kinds):
        out.setdefault(group_of(k), []).append(k)
    return out
