"""stack step 1 tests: contents universe worlds == carry_cert's, schema validator, registry, producer."""
import numpy as np
import pytest

from world_model.v2.stack import registry
from world_model.v2.stack.eval import schema
from world_model.v2.stack.universes.contents import ContentsUniverse
from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC


@pytest.mark.parametrize("task,k", [("tray2", 1), ("box", 2)])
def test_worlds_identical_to_cert(task, k):
    uni = ContentsUniverse([task])
    ws = uni.worlds("cert", n=2)
    cells = [c for (c,) in CC.cells(task)]
    assert len(ws) == 2 * len(cells)
    cell = cells[k]
    got = [w for w in ws if w.theta["cell"] == list(cell)]
    assert len(got) == 2
    for i, w in enumerate(got):
        s = CC.cell_seed(task, cell, i)
        ref = CC.sample_world(task, cell, np.random.default_rng(s))
        assert w.seed == s
        assert w.theta == dict(CR._theta(ref), task=task, cell=list(cell), i=i)


def test_corners_and_box():
    uni = ContentsUniverse(["tray2"])
    ws = uni.worlds("corners")
    assert len(ws) >= 3 * 8 and len({w.world_id for w in ws}) == len(ws)
    box = uni.theta_box("tray2").assumptions()
    assert all(v["tag"] in ("MEASURED", "MODEL", "ASSUMED") for v in box.values())
    assert uni.tasks(ws[0])[0].name == "carry"


def test_registry():
    assert registry.get("universes", "contents") is ContentsUniverse
    with pytest.raises(NotImplementedError):
        registry.get("universes", "house")
    with pytest.raises(KeyError):
        registry.get("universes", "nope")
    assert registry.get("producers", "truth").__name__ == "ContentsTruthProducer"


def test_schema_validator():
    rec = schema.build_record(universe="contents", world_id="w", seed=1, theta={}, task="tray2", planner="scripted", outcome="ok")
    assert set(schema.KEYS) <= set(rec)
    bad = dict(rec, outcome="maybe")
    with pytest.raises(ValueError):
        schema.validate(bad)
    with pytest.raises(ValueError):
        schema.validate({k: v for k, v in rec.items() if k != "code_rev"})


def test_producer_flags_truth_pose():
    uni = ContentsUniverse(["tray2"])
    w = uni.worlds("cert", n=1)[0]
    sim = uni.build(w)
    from world_model.v2.stack.types import WorldState
    ws = WorldState(0.0, None)
    uni.producers("truth")[0].update(ws, sim, 0.0)
    assert ws.privileged() == ["obj.pose"]
    assert set(ws.robot.q) == {"left", "right"} and ws.objects["obj"].params["fill_kg"].source.value == "prior"
    assert uni.success(sim, uni.tasks(w)[0])["ok"] in (True, False)
