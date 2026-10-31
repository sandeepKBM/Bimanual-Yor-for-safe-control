"""Faithful VIZ-COAST baseline: PDDL grounding, Z3 SATPlan, VLM-program clauses, blocks, loop with a fake scene.  No sim."""
import pytest

from world_model.v2.nero.pan.vizcoast_faithful import coast as CO
from world_model.v2.nero.pan.vizcoast_faithful import pddl_lite as PL
from world_model.v2.nero.pan.vizcoast_faithful import satplan as SP
from world_model.v2.nero.pan.vizcoast_faithful import vrm as V


@pytest.fixture(scope="module")
def task():
    return PL.load(V.PDDL)


def test_grounding_statics_and_equality(task):
    names = {str(a) for a in task.actions}
    assert "(pick l handle)" in names and "(pick l rim)" not in names          # can-grasp is static
    assert not any(a.name == "lift-two" and a.args[0] == a.args[2] for a in task.actions)
    assert not any(a.name == "lift-one" and "lid" in a.args for a in task.actions)  # lid is not part of the pan


def test_plans_and_vlm_clauses(task):
    p, H, _ = SP.plan(task)
    assert H == 2 and p[-1].name == "lift-one"
    code = V.extract_code(V.P_TWO)
    p, H, n = SP.plan(task, code)
    assert [a.name for a in p] == ["pick", "push", "clamp", "lift-two"] and n > 0


def test_program_error_and_restricted_exec(task):
    enc = SP.Encoding(task, 2)
    with pytest.raises(Exception):
        SP.apply_program(V.extract_code(V.P_BUGGY), enc)
    with pytest.raises(Exception):
        SP.apply_program("import os\ndef constraints(api): pass", enc)


def test_block_forces_alternative(task):
    p, _, _ = SP.plan(task)
    st = SP.states_along(task, p)
    p2, _, _ = SP.plan(task, None, [(st[1], p[1])])
    assert [str(a) for a in p2] != [str(a) for a in p]


class FakeScene:
    """refinement fails for any plan containing a one-arm lift; counts calls."""
    n_stream_calls = 0

    def refine(self, plan):
        self.n_stream_calls += 1
        for i, a in enumerate(plan):
            if a.name == "lift-one":
                return False, [], i, "fake: one-arm lift unreachable"
        return True, [dict(action=str(a)) for a in plan], None, None


class _Geom:   # VRM needs a scene for the geometry text; the loop only needs refine()
    pass


def test_loop_blocks_then_revises(monkeypatch, task):
    monkeypatch.setattr(V, "geometry_text", lambda scene: "(fake geometry)")
    r = CO.run(V.mock_backend("none"), FakeScene(), K=5)        # 2 one-arm skeletons blocked -> a two-arm plan
    assert r["status"] == "ok" and r["refinement_failures"] == 2 and "lift-two" in r["plan"][-1]
    r = CO.run(V.mock_backend("none"), FakeScene(), K=2)        # blocks dropped after K=2, an unchanged VLM program -> cycles
    assert r["status"] == "fail_limit"
    r = CO.run(V.mock_backend("buggy"), FakeScene(), K=2)
    assert r["status"] == "ok" and any(e["event"] == "program_error" for e in r["events"])
