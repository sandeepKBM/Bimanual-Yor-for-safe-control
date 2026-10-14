"""offline tests of the language layer (MockVLM only; the saved flagship perception pickles are optional data)."""
import json
import os
import pickle

import numpy as np
import pytest

from world_model.v2.house.lang import backend as BE, grounding as GR, authoring as AU, prior as PR, session as SE, hooks as HK
from world_model.v2.tray_rep.sets import ir as IR, tray_spec as TS

PKL = "/data/local/ss5772/molmospaces/flagship/perception/perception_desk.pkl"
needs_pkl = pytest.mark.skipif(not os.path.exists(PKL), reason="saved flagship perception result not available")


def _prop(i, c, half=(0.1, 0.1, 0.1), box=(10, 10, 60, 60)):
    G = np.diag(half)
    return GR.Proposal(i, np.asarray(c, float), G, np.asarray(c, float), G, box, 500, 1.0)


def _grounding(spec):
    """spec = {name: [(id, centre), ...]} -> Grounding with exact names (no VLM)."""
    props, names, groups = [], {}, {}
    for nm, items in spec.items():
        for i, c in items:
            props.append(_prop(i, c))
            names[i] = (nm, 0.9)
            groups.setdefault(nm, []).append(i)
    return GR.Grounding(names=names, raw_names={i: n for i, (n, _) in names.items()}, props=props, groups=groups)


def _cat(spec=None):
    spec = spec or {"monitor": [(0, (3.0, 0.0, 1.0))], "lamp": [(1, (5.0, 1.0, 1.0)), (2, (5.0, -1.0, 1.0))]}
    return AU.build_catalog(_grounding(spec))


def J(spec, unresolved=()):
    return json.dumps({"spec": spec, "unresolved": list(unresolved)})


def base_atoms(extra=(), theta=0.1):
    return {"G": {"lo": 0, "hi": "H", "of": {"and": [
        {"inside": {"a": "cup", "b": {"shrink": {"of": "tray_top", "by": 0.03}}, "margin": 0.0}},
        {"upright": {"a": "tray", "theta_max": theta}}, *extra]}}}


class Script(BE.Backend):
    """returns the queued texts in order, records prompts."""
    name, model = "script", "script"

    def __init__(self, *texts):
        self.q, self.calls = list(texts), []

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        self.calls.append(text)
        return dict(texts=[self.q.pop(0)] * n, seconds=0.1, wall_s=0.1, cached=False)


# ------------------------------------------------------------------------------------------------ grounding
def test_parse_marks_formats():
    assert GR.parse_marks("1 | bottle | 0.9\n2 | desk part | 0.6\n3 | x", 3) == {1: ("bottle", 0.9), 2: ("desk part", 0.6), 3: ("x", 0.5)}
    assert GR.parse_marks('{"marks": [[1, "mug", 0.8], [2, "lamp", 0.4]]}', 2) == {1: ("mug", 0.8), 2: ("lamp", 0.4)}
    assert GR.parse_marks('```json\n{"marks":[{"id":1,"name":"a","confidence":0.7}]}\n```', 1) == {1: ("a", 0.7)}
    assert GR.parse_marks("I cannot do that", 3) == {}
    assert GR.parse_marks("7 | far | 0.9", 3) == {}                       # out-of-range marks ignored


def test_norm_name():
    assert GR.norm_name("The Monitors") == "monitor"
    assert GR.norm_name("desk part") == "desk_part"
    assert GR.norm_name("glass") == "glass"


def test_ground_unknown_and_missing_and_reserved():
    be = Script("1 | tray | 0.9\n2 | unknown | 0.9\n")                       # mark 3 missing
    props = [_prop(10, (0, 0, 1)), _prop(11, (1, 0, 1)), _prop(12, (2, 0, 1))]
    img = np.zeros((100, 100, 3), np.uint8)
    g = GR.ground(be, img, props, None)
    assert g.names[10][0] == "tray_obj"                                       # reserved entity names are never reused
    assert g.names[11][0] == "unknown" and g.names[11][1] <= 0.3
    assert g.names[12] == ("unknown", 0.0) and g.groups == {"tray_obj": [10]}
    assert "Marks to name: [1, 2, 3]" in be.calls[0]


def test_score_grounding():
    g = _grounding({"monitor": [(0, (0, 0, 0))], "cup": [(1, (1, 0, 0))], "unknown": [(2, (2, 0, 0))]})
    s = GR.score_grounding(g, {0: "monitor_1", 1: "pencup_a", 2: "lamp_1"})
    assert s["acc_lenient"] == pytest.approx(2 / 3) and s["n_unknown"] == 1
    assert GR.gt_class("mug_desk1") == "mug" and GR.gt_class("desk_1") == "desk" and GR.gt_class("milk_jug") == "milk_jug"


@needs_pkl
def test_proposals_from_saved_result():
    r = pickle.load(open(PKL, "rb"))
    props = GR.proposals_from_result(r)
    assert 5 <= len(props) <= GR.MAX_MARKS and all(p.box[2] > p.box[0] for p in props)
    im = GR.render_marks(r["left"], props)
    assert im.shape == r["left"].shape and (im != r["left"]).any()


# ------------------------------------------------------------------------------------------------ authoring
def test_author_good_spec():
    cat = _cat()
    be = Script(J(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.05}}])))
    r = AU.author(be, "bring the coffee and keep it 5 cm away from the monitor", cat, log=lambda s: None)
    assert r.status == "vlm" and r.clearances == {"monitor": 0.05} and r.check["sem"]
    assert "monitor" in r.pretty() and len(be.calls) == 1


def test_author_group_expansion_and_probes():
    cat = _cat()                                                              # 'lamp' has two fragments
    be = Script(J(base_atoms([{"clear": {"a": "tray", "b": "lamp", "d": 0.10}}])))
    r = AU.author(be, "avoid the lamp", cat, log=lambda s: None)
    assert r.status == "vlm" and r.clearances == {"lamp": 0.10}
    flat = json.dumps(r.spec_expanded)
    assert "lamp#0" in flat and "lamp#1" in flat
    c = AU.check_spec(r.spec, cat, "avoid the lamp")
    assert c["probes_caught"]["overlap_lamp#0"] and c["probes_caught"]["overlap_lamp#1"] and c["margins"]["nominal"] > 0


def test_author_repair_unknown_entity():
    cat = _cat()
    bad = J(base_atoms([{"clear": {"a": "tray", "b": "laptop", "d": 0.05}}]))
    good = J(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.05}}]))
    be = Script(bad, good)
    r = AU.author(be, "stay 5 cm away from the monitor", cat, log=lambda s: None)
    assert r.status == "repaired" and len(r.attempts) == 2
    assert "unknown entity" in be.calls[1] and "laptop" in be.calls[1]       # the compile error is sent back


def test_author_fallback_logs_why():
    cat = _cat()
    be = Script("not json at all", J({"F": {"lo": 0, "hi": "H", "of": {"inside": {"a": "cup", "b": "tray_top", "margin": 0.0}}}}))
    r = AU.author(be, "bring it", cat, log=lambda s: None)
    assert r.status == "fallback" and r.spec == TS.TRAY_SPEC_FULL and r.clearances == {}
    assert any("FALLBACK" in x for x in r.reasons) and len(r.attempts) == 2


def test_check_rejects_goal_encoding_weak_and_insane_specs():
    cat = _cat()
    # a spec without the cup containment cannot catch ball_off
    c = AU.check_spec({"G": {"lo": 0, "hi": "H", "of": {"upright": {"a": "tray", "theta_max": 0.1}}}}, cat)
    assert not c["ok"] and any("cup leaving" in e for e in c["errors"])
    # theta_max = 3.0 switches the tilt constraint off
    c = AU.check_spec(base_atoms(theta=3.0), cat)
    assert not c["ok"] and any("theta_max" in e for e in c["errors"])
    # numeric clearance out of bounds
    c = AU.check_spec(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 5.0}}]), cat)
    assert not c["ok"]
    # the valid tray-only spec is fine
    assert AU.check_spec(base_atoms(), cat)["ok"]


def test_coverage_and_distance_checks():
    cat = _cat()
    c = AU.check_spec(base_atoms(), cat, "bring it to my desk but keep it away from the monitor")
    assert not c["ok"] and c["uncovered"] == ["monitor"]
    # the destination ('desk' is not an object here) is not a keep-away target even though a keep-away phrase exists
    assert AU.check_spec(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.05}}]), cat, "bring it to the monitor desk, then stay 5 cm away from the monitor")["ok"]
    c = AU.check_spec(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.30}}]), cat, "keep 5 cm away from the monitor")
    assert not c["ok"] and any("distance" in e for e in c["errors"])


def test_unresolved_object_reported_not_invented():
    cat = _cat()
    be = Script(J(base_atoms(), unresolved=["laptop"]))
    r = AU.author(be, "bring the coffee, keep it away from the laptop", cat, log=lambda s: None)
    assert r.status == "vlm" and r.unresolved == ["laptop"] and r.clearances == {}


def test_augment_mode_keeps_hand_spec():
    cat = _cat()
    r = AU.author(Script(J(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.05}}]))), "avoid the monitor", cat, mode="augment", log=lambda s: None)
    assert r.status == "vlm" and "and" in r.spec_expanded and r.spec_expanded["and"][0] == TS.TRAY_SPEC_FULL


# ------------------------------------------------------------------------------------------------ prior
def test_prior_fusion_numbers():
    f = PR.fuse_samples([(0.30, 0.10), (0.40, 0.10)], k=1.0)
    assert f["center"] == pytest.approx(0.35) and f["sigma_indep"] == pytest.approx(0.1 / np.sqrt(2))
    assert f["sigma"] == pytest.approx(np.sqrt(0.01 + 0.005)) and f["lo"] == pytest.approx(0.35 - f["sigma"])
    # inverse-variance weighting: the confident sample dominates
    f = PR.fuse_samples([(0.2, 0.01), (0.8, 1.0)])
    assert f["center"] == pytest.approx(0.2, abs=0.01)
    # clipped to the sane range
    f = PR.fuse_samples([(0.05, 0.02)], k=2.0)
    assert f["lo"] >= PR.SANE_MU[0] and f["hi"] > f["lo"]


def test_estimate_mu_parse_and_default():
    be = Script(*[json.dumps({"cup_tray_friction": {"estimate": 0.3, "sigma": 0.1}})] * 2)
    m = PR.estimate_mu(be, ["a.png", "b.png"], n_queries=3)                 # Script ignores images
    assert m.source == "vlm" and m.n_parsed == 6 and m.n_queries == 6 and m.lo < 0.3 < m.hi
    m = PR.estimate_mu(Script("garbage"), ["a.png"], n_queries=2)
    assert m.source == "default" and (m.lo, m.hi) == PR.DEFAULT_MU


# ------------------------------------------------------------------------------------------------ cache
def test_cached_backend_hit_and_readonly(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"png-bytes")
    inner = Script("A", "B")
    c = BE.CachedVLM(inner, tmp_path / "c")
    o1 = c.ask("hi", images=[str(img)], system="s")
    o2 = c.ask("hi", images=[str(img)], system="s")
    assert o1["texts"] == o2["texts"] == ["A"] and not o1["cached"] and o2["cached"] and len(inner.calls) == 1 and (c.n_hit, c.n_miss) == (1, 1)
    img.write_bytes(b"other")                                               # a different image is a different key
    assert c.ask("hi", images=[str(img)], system="s")["texts"] == ["B"]
    ro = BE.CachedVLM(Script(), tmp_path / "c", mode="ro")
    img.write_bytes(b"png-bytes")
    assert ro.ask("hi", images=[str(img)], system="s")["cached"]
    with pytest.raises(KeyError):
        ro.ask("new", images=[str(img)], system="s")


def test_mock_default_rules_flow():
    cat = _cat()
    r = AU.author(BE.MockVLM(), "bring it, keep it away from the monitor", cat, log=lambda s: None)
    assert r.status == "vlm" and r.clearances.get("monitor") == 0.05


# ------------------------------------------------------------------------------------------------ hooks
def _lr(cat=None, spec=None, mu=None):
    cat = cat or _cat()
    sr = AU.author(Script(J(spec or base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.20}}]))), "keep 20 cm away from the monitor", cat, log=lambda s: None)
    mu = mu or PR.MuInterval(0.30, 0.55, 0.4, 0.1, 1.0, "vlm")
    g = _grounding({"monitor": [(0, (3.0, 0.0, 1.0))]})
    cz = {pid: d for nm, d in sr.clearances.items() for pid in cat.zono_ids(nm)}
    return SE.LangResult("x", g, cat, sr, mu, cz, {}, [], "script", "t")


def test_clearance_result_inflates_only_named_and_does_not_mutate():
    zono = lambda i, c: dict(id=i, kind="obj", gt_key="k", n_pts=100, delta=0.0, range_m=1.0, Z=dict(c=np.array(c, float), G=np.eye(3) * 0.1), Zi=dict(c=np.array(c, float), G=np.eye(3) * 0.1))
    res = dict(zonos=[zono(0, (0, 0, 0)), zono(1, (1, 0, 0))])
    out, added = HK.clearance_result(res, {0: 0.10, 1: 0.01})
    assert added == {0: pytest.approx(0.08)}                                 # d=0.01 is below the rows' own margin: unchanged
    assert out["zonos"][0]["Zi"]["G"].shape == (3, 6) and out["zonos"][1] is res["zonos"][1] and res["zonos"][0]["Zi"]["G"].shape == (3, 3)
    from world_model.v2.tray_rep.sets.zono import Zonotope
    Z = Zonotope(out["zonos"][0]["Zi"]["c"], out["zonos"][0]["Zi"]["G"])
    lo, hi = Z.interval_hull()
    assert hi[0] == pytest.approx(0.1 + 0.08)


def test_shield_defaults_and_set_lang_roundtrip():
    from world_model.v2.house.teleop import shield as SH
    s0 = SH.Shield(None)
    assert (s0.cfg.mu_lo, s0.cfg.mu_hi) == (0.18, 0.35) and s0.ir_mon.lang is None and s0.lang is None
    spec0 = s0.ir_mon.spec
    lr = _lr()                                                    # VLM interval [0.30, 0.55]
    s0.set_lang(lr)                                               # default 'conservative': the VLM may only widen the engineering prior
    assert (s0.cfg.mu_lo, s0.cfg.mu_hi) == (0.18, 0.55) and s0.ir_mon.lang is not None
    assert s0.cfg.ratio == pytest.approx(min(0.8 * 0.18, np.tan(0.30)))
    lr.mu_policy = "vlm"                                          # research ablation: raw VLM interval
    s0.set_lang(lr)
    assert (s0.cfg.mu_lo, s0.cfg.mu_hi) == (0.30, 0.55)
    assert s0.cfg.ratio == pytest.approx(min(0.8 * 0.30, np.tan(0.30)))
    s0.set_lang(None)
    assert (s0.cfg.mu_lo, s0.cfg.mu_hi) == (0.18, 0.35) and s0.ir_mon.lang is None
    assert type(s0.ir_mon.spec) is type(spec0)


def test_shield_ir_monitor_uses_the_language_spec():
    """the monitor's margin with the language spec goes negative when the tray is within d of the named object, and equals the hand spec far away."""
    from world_model.v2.house.teleop import shield as SH
    lr = _lr()
    cfg = SH.ShieldCfg()
    node_l, node_h = lr.monitor_spec(cfg), SH.IRMonitor(cfg).spec
    I = np.eye(3)
    far = TS.EnvState([-10, 0, 1.0], [1, 0, 0, 0], [-10.045, 0, 1.06])
    assert IR.margin(node_l, far) == pytest.approx(IR.margin(node_h, far), abs=1e-9)        # no object nearby: same margin as the hand spec
    near = TS.EnvState([2.75, 0, 1.0], [1, 0, 0, 0], [2.705, 0, 1.06])                      # tray edge ~ 5 cm from the monitor box (x 2.9) -> inside the 20 cm clearance
    assert IR.margin(node_l, near) < 0 < IR.margin(node_h, near)


# ------------------------------------------------------------------------------------------------ integration on the saved flagship capture
def _gt_named_backend(res):
    """MockVLM whose grounding answers use the gt class of each mark (a perfect VLM): mark k -> class of proposals_from_result(res)[k-1]."""
    props = GR.proposals_from_result(res)
    gt = {z["id"]: z["gt_key"] for z in res["zonos"]}
    lines = "\n".join(f"{k} | {GR.gt_class(gt[p.id])} | 0.9" for k, p in enumerate(props, 1))
    auth = J(base_atoms([{"clear": {"a": "tray", "b": "monitor", "d": 0.10}}]))
    return BE.MockVLM(rules=[(r"Marks to name", lines), (r"constraint spec", auth)]), props, gt


@needs_pkl
def test_lang_perception_service_inflates_named_zonotopes_and_calls_shield():
    from world_model.v2.house.flagship.perception import PerceptionService
    from world_model.v2.house.teleop import shield as SH
    res = pickle.load(open(PKL, "rb"))
    be, props, gt = _gt_named_backend(res)
    sess = SE.LangSession(be, out_dir=None, log=lambda s: None)
    inner = PerceptionService("cached", cache_dir=os.path.dirname(PKL), log=lambda s: None)
    got = []
    svc = HK.LangPerceptionService(inner, sess, "bring it to the desk, keep it 10 cm away from the monitor", "/tmp", on_result=lambda lr, k: got.append((lr, k)), log=lambda s: None)
    plan = svc.sense_desk(dict(t=0.0, pose=res["pose"]))
    assert plan is not None and len(got) == 1 and got[0][1] == "desk"
    lr = got[0][0]
    assert lr.spec.status == "vlm" and lr.spec.clearances == {"monitor": 0.10}, lr.spec.reasons
    mon_ids = [z["id"] for z in res["zonos"] if z["gt_key"] == "monitor_1" and z["id"] in {p.id for p in props}]
    assert set(lr.clearance_by_zono) == set(mon_ids) and len(mon_ids) >= 2
    zs = {z["id"]: z for z in svc.results["desk"]["zonos"]}
    for i in mon_ids:
        assert zs[i]["Zi"]["G"].shape[1] == res["zonos"][i]["Zi"]["G"].shape[1] + 3 and zs[i]["lang_clearance_m"] == 0.10
    assert all(zs[i]["Zi"]["G"].shape[1] == res["zonos"][i]["Zi"]["G"].shape[1] for i in zs if i not in mon_ids)
    sh = SH.Shield(None)
    sh.set_lang(lr)
    assert sh.ir_mon.lang is lr


@needs_pkl
def test_web_lang_thread_mock_snapshot(tmp_path):
    import time
    from world_model.v2.house.lang import web as LW
    w = LW.WebLang("mock", cache_dir=str(tmp_path / "c"), out_dir=str(tmp_path / "o"), source="snapshot", log=lambda s: None)
    assert w.state()["status"] == "idle" and not w.submit("   ")
    assert w.submit("bring it to my desk and keep it away from the monitor\x00\x1b")
    t0 = time.time()
    while w.busy() and time.time() - t0 < 60:
        time.sleep(0.05)
    s = w.state()
    assert s["status"] == "done" and s["hud"]["spec_status"] in ("vlm", "repaired", "fallback") and s["marks_jpg"]
    p = w.take_pending()
    assert p is w.last and w.take_pending() is None
    w.clear()
    assert w.take_pending() == LW.CLEAR and w.state()["status"] == "idle"
    assert LW.clean_instruction("a\x00b\n" + "x" * 1000).startswith("a b") and len(LW.clean_instruction("x" * 1000)) == LW.MAX_INSTRUCTION


def test_no_substitution_of_missing_object():
    cat = _cat()
    sub = J(base_atoms([{"clear": {"a": "tray", "b": "lamp", "d": 0.10}}]), unresolved=["laptop"])          # asked for the laptop, constrained the lamp
    ok = J(base_atoms(), unresolved=["laptop"])
    be = Script(sub, ok)
    r = AU.author(be, "bring the coffee, keep it 10 cm away from the laptop", cat, log=lambda s: None)
    assert r.status == "repaired" and r.clearances == {} and r.unresolved == ["laptop"]
    assert "substitute" in be.calls[1]


def test_parse_answer_lifts_misplaced_unresolved():
    txt = json.dumps({"spec": {"G": {"lo": 0, "hi": "H", "of": {"upright": {"a": "tray", "theta_max": 0.1}}}, "unresolved": ["vase"]}, "unresolved": ["lamp"]})
    spec, un = AU.parse_answer(txt)
    assert list(spec) == ["G"] and un == ["lamp", "vase"]
    spec, un = AU.parse_answer(json.dumps({"G": {"lo": 0, "hi": "H", "of": {"upright": {"a": "tray", "theta_max": 0.1}}}, "unresolved": ["x"]}))
    assert list(spec) == ["G"] and un == ["x"]
    assert AU.parse_answer("nothing")[0] is None


@needs_pkl
def test_clearance_changes_the_zonotope_base_filter_route():
    """the per-object clearance from the spec reaches the zono_rows base filter: padding the bottle by 15 cm shortens the filtered naive route."""
    from world_model.v2.house.zono_rows import flagship_place as FP
    r = pickle.load(open(PKL, "rb"))
    obs = dict(pose=r["pose"], tray_pos=r["tray0"])
    base = FP.flagship_backend(r, obs, log=lambda s: None)
    ids = [z["id"] for z in r["zonos"] if z["gt_key"] == "bottle_a"]
    r2, added = HK.clearance_result(r, {i: 0.15 for i in ids})
    pad = FP.flagship_backend(r2, obs, log=lambda s: None)
    assert added and pad["progress_frac"] < base["progress_frac"] - 0.1
