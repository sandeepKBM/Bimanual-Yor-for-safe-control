"""stack/u_groups.py: the U-kind groups, the nominal-kind reader and the stored-recovery counters (11-12 Stage 0a; recording only)."""
from world_model.v2.stack import u_groups as UG


def test_group_mapping_and_other():
    assert UG.group_of("obj_tilt") == "contents" and UG.group_of("collision") == "contents" and UG.group_of("load_world_contact") == "contents"
    assert UG.group_of("wheel_liftoff") == "base" and UG.group_of("route_unfinished") == "base"
    assert UG.group_of("claw_payload") == "rating" and UG.group_of("tau_sat") == "rating" and UG.group_of("claw_twist") == "rating"
    assert UG.group_of("something_new") == "other"
    assert UG.by_group({"spill", "tau_sat", "zzz"}) == dict(contents=["spill"], rating=["tau_sat"], other=["zzz"])
    assert len({k for ks in UG.GROUPS.values() for k in ks}) == sum(len(ks) for ks in UG.GROUPS.values())      # no kind in two groups


def _seg(seg, viol=None, viol_all=None, n=3, n_bad=0):
    nom = dict(viol=viol)
    if viol_all is not None:
        nom["viol_all"] = viol_all
    return dict(seg=seg, nominal=nom, family={0.0: dict(n=n, n_bad=n_bad)})


def test_nominal_kinds_new_and_old_records():
    old = dict(task="box_behind", carry=dict(segments=[_seg("lift", viol=dict(kind="spill", t=1.0))], pre_hold_viol=None), base=dict(U_flags={}))
    assert UG.nominal_kinds(old) is None                                           # first-U only: cannot answer 'any kind'
    new = dict(task="box_behind", probe_viol=dict(kind="claw_payload"), carry=dict(segments=[_seg("lift", viol=dict(kind="obj_tilt", t=1.0), viol_all=dict(obj_tilt=dict(n=2), spill=dict(n=1)))],
                                                                                   pre_hold_viol=dict(kind="tau_sat")), base=dict(U_flags_ever=["wheel_liftoff"]))
    assert UG.nominal_kinds(new) == {"claw_payload", "obj_tilt", "spill", "tau_sat", "wheel_liftoff"}
    assert UG.nominal_kinds(dict(carry=dict(segments=[_seg("lift", viol_all={})]), base={})) == set()           # new record, no U: empty set, not None


def test_recovery_counts():
    r = UG.recovery_of([_seg("lift", n=4, n_bad=0), _seg("carry", n=5, n_bad=2)])
    assert r == dict(lift=dict(n=4, n_bad=0), carry=dict(n=5, n_bad=2)) and UG.recovery_bad(r) is True
    assert UG.recovery_of(None) == {} and UG.recovery_bad({}) is False and UG.recovery_bad(UG.recovery_of([_seg("lift")])) is False
