"""T4 gate (owner-written, 11-16): Rule 1 refusals go back to the proposer, tagged FIXABLE / NOT_FIXABLE (stack/checks/fixability.py).  SIMULATION ONLY,
CPU, no network, mock proposer only.  USER 11-16: N = 3 identical failure signatures stop the loop (ASSUMED); torque/payload on the path = FIXABLE; Rule 4 out.

Pure part: tags on a REAL Rule 1 row (box 1.0-1.5, place, a = 0.5: data/t4_real_place_row.json), one representative name per known family, the evidence +
prompt block.  Loop part (one simulated box world each, minutes): Rule 1 is a STUB injected through probe_select.Ctx.rule1 (fixed_job(..., rule1=)) so each
case is forced; the mock proposer answers the T4 re-ask (mock.t4_reply), wrapped here to capture the request text.
    (a) fixable -> one T4 re-ask whose prompt names the step, pair, distance vs limit and the knobs, then the carry runs on the re-asked plan
    (b) only NOT_FIXABLE -> refused_by "rule1", no re-ask
    (c) mixed -> fixable rows in the FIX block, the missing row only in FIXED CONSTRAINTS
    (d) the same signature 3 times -> "t4_repeat", refused, Rule 1 called exactly 3 times
    (e) no truth (true contents mass, seed) in any T4 prompt
    (f) every family name is tagged (no UNTAGGED); an unknown name is UNTAGGED
    (g) a task not in RULE1_TASKS -> refused as Rule 1 MISSING, Rule 1 never called
    (h) a Rule 1 crash fails CLOSED (owner 11-16): refused_by "rule1", reask_end "rule1_error"
"""
import copy
import json
import pathlib

import pytest

from world_model.v2.nero.contents import probe as P
from world_model.v2.nero.contents import probe_select as PS
from world_model.v2.nero.contents.tests.test_probe_select import EPS, plan_for
from world_model.v2.stack.checks import fixability as FX
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import prompt as PR

REAL = json.loads((pathlib.Path(__file__).parent / "data" / "t4_real_place_row.json").read_text())
WORLD_I = 2                      # a light box world (contents ~1.18 kg): the static admission admits after the weigh, so the loop reaches Rule 1
RECOV_ROW = dict(name="stored_backup:motion@t=0.25/stop_back[place]", ok=False, guarantee="refuted", margin=-0.01,
                 worst={"stop": dict(ok=False, guarantee="refuted", margin=-0.01, why="torque utilisation 1.01 > 1 during the stop")})   # FIXABLE by the recovery choice only
MISSING_ROW = dict(name="stored_backup:motion@t=0.49/stop_back[place]", ok=False, guarantee="missing", why="no model past the last planned sample")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("BIMANUAL_NO_NETWORK", "1")
    monkeypatch.setattr(P, "PROBE_METHOD", "poslift")


def recs(place_rows=None):
    """run_cell-shaped step records: lift + carry admitted; place refused with `place_rows` (None = admitted)."""
    ok = lambda i, s: dict(step=i, seg=s, admitted=True, program="stored_backup", per_program={}, rows=[])
    out = [ok(0, "lift"), ok(1, "carry")]
    if place_rows is None:
        out.append(ok(2, "place"))
    else:
        out.append(dict(step=2, seg="place", admitted=False, program=None, refusal="no recovery program viable", rows=[],
                        per_program={"stored_backup": dict(failing=copy.deepcopy(place_rows))}))
    return out


class Stub:
    """Rule 1 stand-in: answers[k] for the k-th call (the last one repeats); records the calls."""
    def __init__(self, *answers, raise_on_call=False):
        self.answers, self.calls, self.raise_on_call = answers, [], raise_on_call

    def __call__(self, task, cell, wp, a):
        self.calls.append((task, cell, a))
        if self.raise_on_call:
            raise RuntimeError("stub Rule 1 crash")
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


@pytest.fixture
def prompts(monkeypatch):
    seen = []
    orig = MK.t4_reply

    def wrap(task, text):
        seen.append(text)
        return orig(task, text)
    monkeypatch.setattr(MK, "t4_reply", wrap)
    return seen


def ok_path_rows(task, wp, a, cell):
    """11-17: the planned-path rows run before Rule 1 (probe_select.py ~384); this loop gate tests Rule 1 / T4, so the path rows are stubbed to pass.
    (Real rows on this box world refuse: right J7 1.5502 vs limit 1.5197 at the hold pose; see test_path_rows_refuse_before_rule1.)"""
    return [PS._row_dict("planned_path", True, 0.0, "sampled", "stubbed in the T4 gate")]


def run(tmp_path, stub, path_rows=ok_path_rows):
    plan = plan_for("box", tmp_path)
    r = PS.fixed_job(("box", WORLD_I, *EPS, plan), method="poslift", rule1=stub, path_rows=path_rows)
    assert "error" not in r, r.get("tb")
    return r


# ---------------------------------------------------------------------------------------------------------------------------------------------- pure
def test_real_place_row_is_fixable_geometry():
    lv = FX.leaves(REAL)
    assert [x["name"] for x in lv] == ["collision"]
    t = FX.tag_leaf(lv[0]["name"], lv[0]["guarantee"], lv[0]["why"])
    assert t.kind == FX.FIXABLE_GEOMETRY and t.key == "collision_swept:obj_world" and t.pair == "shelf|box_wall_e"
    assert t.distance_m == pytest.approx(0.0071) and t.swept_bound_m == pytest.approx(0.0021) and t.d_min_m == pytest.approx(0.005)
    assert any("placement" in k for k in t.knobs) and any("lift" in k for k in t.knobs)


@pytest.mark.parametrize("name,guarantee,why,kind", [
    ("collision", "refuted", "obj_world shelf|box_wall_e distance 0.0031 m, swept bound 0.0031 m < d_min 0.0050 m at t = 0.000 s (sample 3)", FX.FIXABLE),
    ("collision", "refuted", "arm_arm left_link6|right_link6 distance 0.0090 m, swept bound 0.0040 m < d_min 0.0050 m at t = 0.100 s (interval 2)", FX.FIXABLE_GEOMETRY),
    ("collision", "missing", "scripted plan: no collision model", FX.NOT_FIXABLE),
    ("support_dest", "refuted", "", FX.FIXABLE),
    ("torque_payload[place]", "refuted", "", FX.FIXABLE),
    ("rest_payload", "refuted", "", FX.FIXABLE),
    ("open_content[carry]", "refuted", "", FX.FIXABLE),
    ("stop", "refuted", "", FX.FIXABLE),
    ("hold_after_stop", "refuted", "", FX.FIXABLE),
    ("retrace", "refuted", "", FX.FIXABLE),
    ("open_content[stop]", "refuted", "", FX.FIXABLE),
    ("joint_pos", "refuted", "", FX.FIXABLE),
    ("stop_back[place]", "missing", "", FX.NOT_FIXABLE),
    ("dynamic[place]", "refuted", "", FX.NOT_FIXABLE),
    ("stored_backup:coverage", "refuted", "", FX.NOT_FIXABLE),
    ("obj_tilt", "refuted", "", FX.NOT_FIXABLE),
    ("base_tip", "refuted", "", FX.NOT_FIXABLE),
    ("recovery[base_move]", "missing", "", FX.NOT_FIXABLE),
])
def test_known_families_are_tagged(name, guarantee, why, kind):
    t = FX.tag_leaf(name, guarantee, why)
    assert t.kind == kind, (name, t)
    assert t.kind != FX.UNTAGGED and t.key
    assert (t.knobs == ()) == (kind == FX.NOT_FIXABLE), (name, t.knobs)   # a fixable row always names a knob; a fixed one never does


def test_unknown_name_is_untagged():
    assert FX.tag_leaf("some_new_row[x]", "refuted", "").kind == FX.UNTAGGED


def test_mixed_rows_split_and_prompt_blocks():
    fix, fixed, untag = PS.rule1_failures(recs([REAL, MISSING_ROW]))
    assert untag == [] and len(fix) == 1 and len(fixed) == 1
    assert fix[0]["pair"] == "shelf|box_wall_e" and fix[0]["step_index"] == 2 and fix[0]["recovery_program"] == "stored_backup" and fix[0]["branch_t"] == pytest.approx(0.25)
    assert fixed[0]["row"].endswith("stop_back[place]")
    ev = dict(trigger="T4", rule1_refusals=[{k: d[k] for k in ("step_index", "step", "recovery_program", "branch_t", "row", "pair", "distance_m", "d_min_m", "kind", "knobs")} for d in fix],
              fixed_constraints=[dict(step_index=d["step_index"], row=d["row"], why=d["why"]) for d in fixed])
    txt = PR.reask_evidence_block(ev)
    a, b = txt.index("RULE 1 REFUSALS YOU CAN FIX"), txt.index("FIXED CONSTRAINTS")
    assert "(T4)" in txt and "[2]" in txt
    assert "shelf|box_wall_e" in txt[a:b] and "stop_back" not in txt[a:b] and "stop_back[place]" in txt[b:]
    assert PS.t4_signature(fix) == ((2, "collision_swept:obj_world", "shelf|box_wall_e"),)


def test_recovery_row_is_fixable_by_the_recovery_only():
    fix, fixed, _ = PS.rule1_failures(recs([RECOV_ROW]))
    assert len(fix) == 1 and fixed == [] and fix[0]["key"].startswith("recovery_torque") and not any("placement" in k for k in fix[0]["knobs"])


def test_only_missing_rows_are_not_fixable():
    fix, fixed, _ = PS.rule1_failures(recs([MISSING_ROW]))
    assert fix == [] and len(fixed) == 1


# ---------------------------------------------------------------------------------------------------------------------------------------------- loop (sim)
def test_a_fixable_reasks_then_rule1_admits(tmp_path, prompts):
    stub = Stub(recs([REAL]), recs(None))
    r = run(tmp_path, stub)
    v = r["v2"]
    assert len(stub.calls) == 2 and v["t_counts"].get("T4") == 1 and v["n_reasks"] == 1 and v["reasks"][0]["trigger"] == "T4"
    assert v["first_plan_id"] != v["plan_id"]                                    # Rule 1 admitted the re-asked plan
    # USER 11-17 "Assert the cert refusal": the stub's admitted records carry no program_spec / plan_key, so the executor's certificate check
    # (stack/executor.py ~501) refuses the run, fail closed.  A real Rule 1 record carries them (adapters/rule1_contents.py).
    assert "carry" not in r and r["refused_by"] == "cert_mismatch" and r["outcome"] == "refused_cert_mismatch", (r.get("refused_by"), r.get("outcome"))
    assert any("no program_spec" in p_ for p_ in (r["decision"].get("cert_problems") or [])), r["decision"].get("cert_problems")
    assert len(prompts) == 1
    t = prompts[0]
    assert "RULE 1 REFUSALS YOU CAN FIX" in t and "shelf|box_wall_e" in t and "0.0071" in t and "0.005" in t and "placement" in t
    # (e) no truth in the prompt
    m_true = r.get("M_true") or v.get("M_true")
    assert m_true is not None
    for s in {f"{m_true:.3f}", f"{m_true:.4f}", repr(float(m_true))}:
        assert s not in t, s
    assert "seed" not in t.lower()


def test_b_only_not_fixable_refuses_without_reask(tmp_path, prompts):
    stub = Stub(recs([MISSING_ROW]))
    r = run(tmp_path, stub)
    v = r["v2"]
    assert len(stub.calls) == 1 and v["n_reasks"] == 0 and prompts == []
    assert r["refused_by"] == "rule1" and v["reask_end"] == "rule1_not_fixable" and "carry" not in r
    assert v["attempts"][-1]["decision"]["action"] == PS.T4_REFUSE_ACTION


def test_d_repeat_guard_stops_after_three(tmp_path, prompts):
    # the same refusal whatever the proposer does.  A recovery-only row: the mock then changes only the recovery skill (a placement row makes the mock shift the move
    # dx, and after two shifts the STATIC admission refuses the placement before Rule 1 runs a third time: seen 11-16)
    stub = Stub(recs([RECOV_ROW]))
    r = run(tmp_path, stub)
    v = r["v2"]
    assert PS.MAX_SAME_T4_SIGNATURE == 3
    assert len(stub.calls) == 3 and v["reask_end"] == "t4_repeat" and v["n_reasks"] == 2 and len(prompts) == 2
    assert r["refused_by"] == "rule1" and "carry" not in r
    assert v["attempts"][-1]["decision"]["action"] == PS.T4_NO_PLAN_ACTION


def test_g2_task_without_rule1_fails_closed(tmp_path, monkeypatch, prompts):
    """USER 11-17 fail closed: a task not in RULE1_TASKS refuses as Rule 1 MISSING (never a silent admission); Rule 1 is not called."""
    monkeypatch.setattr(PS, "RULE1_TASKS", ())
    stub = Stub(raise_on_call=True)
    r = run(tmp_path, stub)
    v = r["v2"]
    assert stub.calls == [] and "carry" not in r
    assert v["attempts"][-1]["rule1"]["guarantee"] == "missing", v["attempts"][-1].get("rule1")


def test_path_rows_refuse_before_rule1(tmp_path, prompts):
    """11-17: a failing planned-path row refuses the attempt before Rule 1 runs (fail closed)."""
    bad = lambda task, wp, a, cell: [PS._row_dict("joint_pos", False, -0.03, "refuted", "stub: joint margin")]
    stub = Stub(raise_on_call=True)
    r = run(tmp_path, stub, path_rows=bad)
    v = r["v2"]
    assert stub.calls == [] and "carry" not in r
    assert any(a.get("decision", {}).get("refused_by") == "path_rows" for a in v["attempts"]), [a.get("decision") for a in v["attempts"]]


def test_h_rule1_crash_fails_closed(tmp_path, prompts):
    stub = Stub(raise_on_call=True)
    r = run(tmp_path, stub)
    v = r["v2"]
    assert len(stub.calls) == 1 and r["refused_by"] == "rule1" and v["reask_end"] == "rule1_error" and "carry" not in r and prompts == []
