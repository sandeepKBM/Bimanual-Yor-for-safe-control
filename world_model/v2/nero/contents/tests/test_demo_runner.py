"""demo_runner (Stage 0d): world specs, arm parsing, phase tagging (synthetic), refusal split (synthetic), summary (synthetic), resume / freeze, additive COM point.  SIMULATION ONLY, no network.
PYTHONPATH=. python -m pytest -q world_model/v2/nero/contents/tests/test_demo_runner.py       (one test builds one episode: seconds)
"""
import argparse
import json

import numpy as np
import pytest

from world_model.v2.nero.contents import demo_runner as DR


# ------------------------------------------------------------------------------------------------------------------------------ worlds
def test_explicit_world_equals_existing_world():
    from world_model.v2.nero.base import smoke as SM
    seed, w = SM.behind_world("box_behind", 2)
    spec = dict(id="x", task="box_behind", contents_kg=w.contents_kg, contents_xy=list(w.contents_xy))
    seed2, w2 = DR.build_world(spec)
    assert w2 == w and w2.shelf is False                        # identical scene except nothing: same contents -> same world
    assert seed2 == DR.seed_of("x") and seed2 != seed
    assert DR.build_world(dict(id="x", task="box_behind", i=2)) == (seed, w)


def test_explicit_world_validation():
    with pytest.raises(NotImplementedError):
        DR.validate_spec(dict(id="t", task="tray6_behind", contents_kg=1.0, contents_xy=[0, 0]))
    with pytest.raises(ValueError):
        DR.build_world(dict(id="a", task="box_behind", contents_kg=9.0, contents_xy=[0, 0]))         # outside the prior box
    with pytest.raises(ValueError):
        DR.build_world(dict(id="a", task="box_behind", contents_kg=1.0, contents_xy=[0.0, 0.2]))      # outside the interior
    with pytest.raises(ValueError):
        DR.validate_spec(dict(id="a", task="box_behind", contents_kg=1.0))
    assert DR.seed_of("a") == DR.seed_of("a") != DR.seed_of("b")


def test_d1_grid():
    g = DR.d1_grid()
    assert len(g) == 24 and len({w["id"] for w in g}) == 24
    assert sorted({w["total_kg"] for w in g}) == sorted(DR.D1_TOTAL_KG)
    for w in g:
        DR.build_world(w)                                          # every grid world builds (inside the prior and the interior)
        assert abs(w["contents_kg"] + 0.25 - w["total_kg"]) < 1e-9 and w["contents_xy"][0] == 0.0
    off = {w["offset"]: w["contents_xy"][1] for w in g}
    assert off["centre"] == 0.0 and abs(off["half"] - 0.5 * off["wall"]) < 1e-4 and abs(off["wall"] - 0.085) < 1e-9


def test_world_list_hash_and_freeze(tmp_path):
    DR.write_worlds(tmp_path / "w")
    p = tmp_path / "w" / "worlds_d1_box44.json"
    d = DR.load_worlds(p)
    assert len(d["worlds"]) == 44 and "ASSUMED" in d["assumptions"]["N1"]
    out = tmp_path / "run"
    out.mkdir()
    h = DR.freeze_worlds(p, out)
    assert (out / "WORLDS.sha256").read_text().split()[0] == h == DR.file_sha256(p)
    assert DR.freeze_worlds(p, out) == h                           # same list again: fine
    with pytest.raises(SystemExit):
        DR.freeze_worlds(tmp_path / "w" / "worlds_existing40.json", out)
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps(dict(worlds=[dict(id="a", task="box_behind", i=0), dict(id="a", task="box_behind", i=1)])))
    with pytest.raises(ValueError):
        DR.load_worlds(dup)


def test_claw_axis_is_y_and_true_com_matches_world():
    """the D1 offset is along the claw-to-claw axis: the claw sites differ in y (not x) in the object frame; the true COM point equals the world's contents_xy."""
    import mujoco
    from world_model.v2.nero.contents import behind_job as BJ, carry as CR, carry_cert as CC
    from world_model.v2.stack import tasks as TK
    spec = dict(id="t", task="box_behind", contents_kg=1.5, contents_xy=[0.0, 0.0425])
    _, w = DR.build_world(spec)
    ep = CR.CarryEpisode(w, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, base="swerve", goal_support=TK.task("box_behind").goal_support)
    mujoco.mj_forward(ep.m, ep.d)
    R = ep.d.xmat[ep.m.body("obj").id].reshape(3, 3)
    o = ep.d.site_xpos[ep.m.site("obj_ref").id]
    cl = {s: R.T @ (ep.d.site_xpos[ep.site[s]] - o) for s in CR.SIDES}
    d = np.abs(cl["left"] - cl["right"])
    assert d[1] > 0.2 and d[0] < 0.05, cl                         # claws apart along y
    com = BJ.true_com(ep)
    assert abs(com[0]) < 5e-3 and abs(com[1] - 0.0425) < 5e-3


# ------------------------------------------------------------------------------------------------------------------------------ additive COM point
def test_interval_spec_com_is_additive():
    from world_model.v2.nero.contents import objects as OB, probe as P
    a, b = P.interval_spec("box", 1.0, 2.0), OB.boxed(1.0, 2.0)
    assert np.array_equal(a.contents.P, b.contents.P) and a.contents.P.shape == (8, 3)
    s = P.interval_spec("box", 1.5, 1.5, com=(0.0, 0.04, 0.055))
    assert s.contents.P.shape == (1, 3) and np.allclose(s.contents.P[0], [0.0, 0.04, 0.055]) and "ORACLE" in s.contents.labels["P"]
    assert len(s.contents.pi_vertices()) == 1
    with pytest.raises(NotImplementedError):
        P.interval_spec("tray6", 1.0, 1.0, com=(0, 0, 0))


# ------------------------------------------------------------------------------------------------------------------------------ arms
def test_parse_arm():
    a = DR.parse_arm("ours")
    assert (a.kind, a.probe, a.reask, a.theta_mode) == ("ours", None, True, None)
    a = DR.parse_arm("ours:probe=poslift_ff:reask=off")
    assert (a.probe, a.reask) == ("poslift_ff", False)
    assert DR.parse_arm("oracle_theta").theta_mode == "oracle_theta" and DR.parse_arm("no_probe").theta_mode == "no_probe" and DR.parse_arm("estimate:probe=claw_guard").probe == "claw_guard"
    a = DR.parse_arm("checker_off:L2")
    assert (a.kind, a.level) == ("checker_off", "L2")
    a = DR.parse_arm("agentchord:L3:plan=/x/y")
    assert (a.kind, a.level, a.plan) == ("agentchord", "L3", "/x/y")
    for bad in ("nope", "checker_off", "checker_off:L4", "agentchord:L3", "ours:L1", "ours:plan=/x", "oracle_mass:probe=poslift", "ours:reask=maybe", "ours:probe=a:probe=b"):
        with pytest.raises(ValueError):
            DR.parse_arm(bad)
    arms = DR.parse_arms("ours,oracle_mass,checker_off:L1,checker_off:L2")
    assert [x.raw for x in arms] == ["ours", "oracle_mass", "checker_off:L1", "checker_off:L2"]
    with pytest.raises(ValueError):
        DR.parse_arms("ours,ours")


def test_unknown_probe_method_raises():
    DR.check_probe_method("poslift")
    DR.check_probe_method(None)
    with pytest.raises(ValueError):
        DR.check_probe_method("no_such_probe")
    from world_model.v2.nero.contents import probe as P
    with pytest.raises(ValueError):
        P.make_probe(None, "box", 0.1, 0.1, method="no_such_probe")


def test_ctx_defaults_off():
    """the additive Ctx fields default to the unchanged path."""
    from world_model.v2.nero.contents import probe_select as PS
    f = {x.name: x.default for x in __import__("dataclasses").fields(PS.Ctx)}
    assert f["theta_mode"] is None and f["on_phase"] is None and f["dry"] is False and f["coverage_mode"] == "enforce"


# ------------------------------------------------------------------------------------------------------------------------------ phase tagging (synthetic episode)
class FakeEp:
    """mimics what executor.run_segments calls: store -> reset_track -> run_phases (nominal, with U) -> [restore(branch) -> reset_track -> run_phases(backup)]* -> restore(end)."""

    def __init__(self):
        self.bmon = argparse.Namespace(flagged=set())
        self.log = []

    def _viol(self, kind, t, **kw):
        self.log.append(("viol", kind))

    def restore(self, sn):
        self.log.append(("restore", sn))

    def store(self, seg):
        self.log.append(("store", seg))

    def run_phases(self, phases, branch=None):
        self.log.append(("run", phases))
        for ph in phases:
            if ph == "nominal_U":
                self._viol("obj_tilt", 1.0)
                self._viol("obj_tilt", 1.1)
                self._viol("spill", 1.2)
            if ph == "backup_U":
                self._viol("tau_sat", 0.5)
                self.bmon.flagged.add("wheel_liftoff")
            if ph == "base_nom":
                self.bmon.flagged.add("route_unfinished")


def test_phase_tagging():
    ep = FakeEp()
    tr = DR.PhaseTracker()
    tr.attach(ep)
    ep._viol("claw_payload", 0.1)                                   # probe phase
    tr.set("ramp_hold")
    ep._viol("tau_sat", 0.2)
    tr.set("nominal")
    for seg in ("lift", "route"):                                   # executor.run_segments, two segments
        ep.store(seg)
        ep.run_phases(["nominal_U"] if seg == "lift" else ["base_nom"])
        for sn in ("b1", "b2"):
            ep.restore(sn)
            ep.run_phases(["backup_U"])
        ep.restore("end")
    ep.store("release")                                              # the closing restore(end) must NOT tag the next segment's run as backup
    ep.run_phases(["nominal_U"])
    ep.restore("end2")                                              # after the last segment: nothing follows
    U, flags = tr.result()
    assert U["probe"]["claw_payload"] == dict(n=1, t_first=0.1)
    assert U["ramp_hold"]["tau_sat"]["n"] == 1
    assert U["nominal"]["obj_tilt"] == dict(n=4, t_first=1.0) and U["nominal"]["spill"]["n"] == 2   # lift + release segments, 2 + 2 entries
    assert U["backup"]["tau_sat"]["n"] == 4 and "obj_tilt" not in U["backup"]
    assert "wheel_liftoff" in flags["backup"] and "route_unfinished" in flags["nominal"]
    assert tr.n_backup_calls == 4 and tr.phase == "nominal"
    assert ("viol", "obj_tilt") in ep.log                          # the original _viol still runs
    with pytest.raises(ValueError):
        tr.set("bogus")


# ------------------------------------------------------------------------------------------------------------------------------ refusal split + summary (synthetic records)
def rec(world, arm, kind=None, refused=False, nominal=(), backup=(), success=None, M=2.0, task="box_behind", theta_star_admit=None, spec=None, base_flags=None):
    r = dict(world_id=world, arm=arm, arm_kind=kind or arm.split(":")[0], task=task, M_true=M, spec=spec or dict(id=world, task=task, i=0),
             outcome="refused_in_R" if refused else "carried", any_U=bool(nominal),
             U_by_phase={p: {k: dict(n=1, t_first=1.0) for k in ks} for p, ks in (("nominal", nominal), ("backup", backup)) if ks},
             result=dict(refused=refused, success=(not refused and not nominal) if success is None else success, safe_success=(not refused and not nominal) if success is None else success,
                         unsafe=bool(nominal)), base_level=None if refused else "L1")
    if base_flags:
        r["base_flags_by_phase"] = base_flags
    if theta_star_admit is not None:
        r["verdicts"] = dict(theta_star=dict(admit=theta_star_admit))
    return r


def test_u_groups():
    assert DR.u_group("obj_tilt") == "contents" and DR.u_group("wheel_liftoff") == "base" and DR.u_group("tau_sat") == "rating" and DR.u_group("claw_payload") == "rating"
    assert DR.u_group("load_world_contact") == "contents" and DR.u_group("whatever") == "other"


def test_refusal_split():
    # necessary: oracle_theta refuses and checker_off:L1 shows a contents U
    by = {"oracle_theta": rec("w", "oracle_theta", refused=True), "oracle_mass": rec("w", "oracle_mass", refused=True), "checker_off:L1": rec("w", "checker_off:L1", nominal=("obj_tilt",))}
    assert DR.refusal_class(rec("w", "ours", refused=True), by) == "necessary"
    by["checker_off:L1"] = rec("w", "checker_off:L1")                                                 # model-only: no U although oracle_theta refuses
    assert DR.refusal_class(rec("w", "ours", refused=True), by) == "model_only"
    by["checker_off:L1"] = rec("w", "checker_off:L1", nominal=("claw_payload",))                      # a rating U alone is not a contents / base U -> model-only
    assert DR.refusal_class(rec("w", "ours", refused=True), by) == "model_only"
    by["checker_off:L1"] = rec("w", "checker_off:L1", base_flags={"nominal": ["wheel_liftoff"]}, nominal=())   # a base flag counts
    assert DR.refusal_class(rec("w", "ours", refused=True), by) == "necessary"
    del by["checker_off:L1"]                                                                          # no checker_off record -> stays necessary
    assert DR.refusal_class(rec("w", "ours", refused=True), by) == "necessary"
    by2 = {"oracle_theta": rec("w", "oracle_theta"), "oracle_mass": rec("w", "oracle_mass")}
    assert DR.refusal_class(rec("w", "ours", refused=True), by2) == "unnecessary_mass"
    by2["oracle_mass"] = rec("w", "oracle_mass", refused=True)
    assert DR.refusal_class(rec("w", "ours", refused=True), by2) == "unnecessary_position"
    assert DR.refusal_class(rec("w", "ours", refused=True), {"oracle_theta": rec("w", "oracle_theta")}) == "undetermined"
    assert DR.refusal_class(rec("w", "ours", refused=True), {}) == "undetermined"


def synthetic_records():
    out = []
    # w1: ours refuses, oracle_theta admits, oracle_mass refuses -> unnecessary_position;  w2: everything admits, ours has a contents U (wrong admit) and a backup-only U
    # w3: ours refuses, oracle_theta refuses, checker_off:L1 U in contents -> necessary
    for w, M in (("w1", 1.5), ("w2", 2.5), ("w3", 4.5)):
        out.append(rec(w, "oracle_theta", refused=(w == "w3"), M=M))
        out.append(rec(w, "oracle_mass", refused=(w != "w2"), M=M))
        out.append(rec(w, "checker_off:L1", nominal=("obj_tilt",) if w == "w3" else (), M=M))
    out.append(rec("w1", "ours", refused=True, M=1.5))
    out.append(rec("w2", "ours", nominal=("spill",), backup=("tau_sat", "wheel_liftoff"), M=2.5, theta_star_admit=False))
    out.append(rec("w3", "ours", refused=True, M=4.5))
    return out


def test_build_summary_counts():
    cells, by_world = DR.build_summary(synthetic_records())
    c = cells["box_behind"]["ours"]
    assert (c["n"], c["refused"], c["success"], c["safe_success"], c["unsafe"], c["wrong_admit"], c["admit_theta_star_refuses"]) == (3, 2, 0, 0, 1, 1, 1)
    assert dict(c["refusal"]) == {"unnecessary_position": 1, "necessary": 1} and c["wrong_admit_worlds"] == ["w2"]
    assert dict(c["groups"]) == {"contents": 1} and c["backup_only"] == 1 and dict(c["backup_kinds"]) == {"tau_sat": 1, "wheel_liftoff": 1}
    co = cells["box_behind"]["checker_off:L1"]
    assert (co["n"], co["unsafe"], co["safe_success"]) == (3, 1, 2)
    assert dict(cells["box_behind"]["oracle_mass"]["refusal"]) == {"unnecessary_position": 1, "necessary": 1}


def test_last_record_wins_and_errors_counted():
    r1 = rec("w1", "ours", refused=True)
    r2 = dict(rec("w1", "ours"), error="boom")
    cells, _ = DR.build_summary([r1, r2])
    assert cells["box_behind"]["ours"]["errors"] == 1 and cells["box_behind"]["ours"]["n"] == 1


def test_summary_files_and_figures(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    recs = synthetic_records()
    for i, r in enumerate(recs):
        r["spec"] = dict(id=r["world_id"], task="box_behind", contents_kg=1.0, contents_xy=[0.0, 0.04 * i % 0.085])
    (out / "trials_shard0of1.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    (out / "CONFIG_shard0of1.json").write_text(json.dumps(dict(worlds_sha256="abc", code_sha="deadbeef")))
    DR.summary(argparse.Namespace(out=str(out)))
    sj = json.loads((out / "summary.json").read_text())
    assert sj["cells"]["box_behind"]["ours"]["wrong_admit"] == 1 and sj["worlds_sha256"] == "abc"
    md = (out / "SUMMARY.md").read_text()
    assert "| ours | 3 |" in md and "unnecessary_position" in md
    for f in ("F4_outcome_matrix_box_behind.png", "F2_level_staircase_box_behind.png", "F1_map_box_behind.png", "F3_bound_vs_measured_stub.png"):
        assert (out / f).stat().st_size > 1000, f


def test_resume_skips_finished_and_retries_errors(tmp_path):
    f = tmp_path / "trials_shard0of2.jsonl"
    f.write_text(json.dumps(dict(world_id="a", arm="ours", outcome="carried")) + "\n" + json.dumps(dict(world_id="b", arm="ours", error="x")) + "\n{broken\n")
    (tmp_path / "trials_shard1of2.jsonl").write_text(json.dumps(dict(world_id="c", arm="oracle_mass")) + "\n")
    assert DR.read_done(tmp_path) == {("a", "ours"), ("c", "oracle_mass")}
