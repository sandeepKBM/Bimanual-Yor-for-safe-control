"""PotUniverse + PreGrasp on pan/pot handles.  SIMULATION ONLY, CPU.   PYTHONPATH=. python -m pytest -q world_model/v2/stack/tests/test_pot_universe.py"""
import numpy as np
import pytest

from world_model.v2.nero import gripper as G
from world_model.v2.nero.pan.handle_grasp import FINGER_OPEN
from world_model.v2.nero.pan.theta import THETA
from world_model.v2.stack import registry as R
from world_model.v2.stack.skills import pregrasp as PG
from world_model.v2.stack.types import SkillCall, Universe
from world_model.v2.stack.universes.pot import PotUniverse

# one search per handle, coarse: the best cells of the 11-04 smoke (pot nominal), refine 1 round
FAST = dict(opens=[FINGER_OPEN], lim=dict(inset=(0.0, 0.05)), grid=dict(inset=(0.0, 0.01, 0.02), tilt_deg=(0.0,)), refine_rounds=1, procs=3)


def test_registry_and_protocol():
    assert R.get("universes", "pot") is PotUniverse
    assert isinstance(PotUniverse(), Universe)


def test_theta_box_labels_are_the_pan_theta():
    b = PotUniverse().theta_box()
    assert set(b.fields) == set(THETA)
    for k, (lo, hi, lab) in THETA.items():
        f = b.fields[k]
        assert (f.lo, f.hi, f.src) == (lo, hi, lab)
        assert f.tag == ("MODEL" if lab.startswith("MODEL") else "ASSUMED")
    nom = PotUniverse().nominal_theta()
    assert all(b.fields[k].lo <= v <= b.fields[k].hi for k, v in nom.items())


def test_worlds_deterministic_per_seed():
    u = PotUniverse()
    a, b = u.worlds("all", n=3, seed=5), u.worlds("all", n=3, seed=5)
    assert [w.world_id for w in a] == [w.world_id for w in b] and [w.theta for w in a] == [w.theta for w in b]
    assert [w.native for w in a] == [w.native for w in b]
    c = u.worlds("random", n=3, seed=6)
    assert [w.theta for w in c] != [w.theta for w in a[-3:]]
    fl = u.theta_box().fields
    for w in a:
        assert all(fl[k].lo <= v <= fl[k].hi for k, v in w.theta.items())
    with pytest.raises(ValueError):
        u.worlds("nope")


@pytest.mark.parametrize("obj,handles", [("pot", ("pan_handle", "pan_handle2")), ("pan", ("pan_handle",))])
def test_build_pan_and_pot(obj, handles):
    u = PotUniverse(obj)
    sim = u.build(u.worlds("nominal")[0])
    names = {sim.m.geom(i).name for i in range(sim.m.ngeom)}
    assert ("pan_handle2" in names) == (obj == "pot") and "pan_handle" in names
    assert sim.handles["params"].handle2_len == (0.0 if obj == "pan" else sim.handles["params"].handle_len)
    ws = u.initial_state(sim)
    oid = u.object_id()
    assert oid in ws.objects and f"{oid}.pose" in ws.privileged()              # truth producer is flagged
    assert u.tasks(sim.world)[0].obj == oid
    assert not u.success(sim, u.tasks(sim.world)[0])["ok"]                       # arms at home: nothing straddled


def test_handle_features_geometry():
    sim = PotUniverse().build(PotUniverse().worlds("nominal")[0])
    f = PG.handle_features(sim.handles["params"])
    p = sim.handles["params"]
    assert f["handle"].side == "left" and f["handle2"].side == "right" and f["handle"].thickness == pytest.approx(p.handle_t)
    tip = p.pan_radius + p.pan_wall_t + p.handle_len
    assert np.allclose(f["handle"].pos[:2], [0.0, tip], atol=1e-9) and np.allclose(f["handle2"].pos[:2], [0.0, -tip], atol=1e-9)
    assert np.allclose(f["handle"].approach, [0, -1, 0]) and np.allclose(f["handle2"].approach, [0, 1, 0])
    assert G.opening_for_width(p.handle_t)["frac"] > 0


@pytest.mark.parametrize("handle", ["handle", "handle2"])
def test_pregrasp_finds_candidate_on_each_nominal_handle(handle):
    """nominal pot (handle_len = THETA midpoint): a passing candidate for the left hand on handle and for the right hand on handle2."""
    u = PotUniverse("pot")
    sim = u.build(u.worlds("nominal")[0])
    ws = u.initial_state(sim)
    side = {"handle": "left", "handle2": "right"}[handle]
    feats = {handle: u.grasp_features(sim)[handle]}
    call = SkillCall("pregrasp", dict(obj="pot", features=feats, constraints=FAST), (side,))
    sk = PG.PreGrasp()
    ok, why = sk.pre(ws, call)
    assert ok, why
    res = sk.search(ws, call, sim)
    print(f"\n{handle}: ok={res.ok} cells={res.n_cells} wall={res.wall_s:.0f}s best={res.ranked[0]['cell'] if res.ok else res.reason} "
          f"margin={res.best['margin'] if res.ok else None}")
    assert res.ok, res.reason
    assert res.best["margin"] > 0 and res.best["ik_res"][side] <= PG.IK_TOL
    assert set(res.best["q"]) == {side}


def test_pregrasp_both_handles_jointly_with_arm_arm_check():
    """one call, both hands (arm-arm h >= H_MIN checked inside the search) on the nominal pot."""
    u = PotUniverse("pot")
    sim = u.build(u.worlds("nominal")[0])
    ws = u.initial_state(sim)
    cons = dict(FAST, grid=dict(inset=(0.0, 0.01, 0.02, 0.03), tilt_deg=(0.0,), dz=(0.0, 0.005)), refine_rounds=0)
    res = PG.PreGrasp().search(ws, u.pregrasp_call(sim, cons), sim)
    print(f"\njoint: ok={res.ok} cells={res.n_cells} wall={res.wall_s:.0f}s reason={res.reason[:200]}")
    assert res.ok, res.reason
    assert set(res.best["q"]) == {"left", "right"} and res.best["h"] >= PG.H_MIN
