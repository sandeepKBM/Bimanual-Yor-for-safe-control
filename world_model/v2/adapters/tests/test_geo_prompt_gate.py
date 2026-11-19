"""Owner gate (11-19, plan Phases 4-5): the geometry-mode first prompt carries NO scene numbers; the probe runner fails closed.
SIMULATION ONLY, CPU, no network, no proposer backend (mock only).
PYTHONPATH=. python -m pytest -q world_model/v2/adapters/tests/test_geo_prompt_gate.py

Checks:
  1. geo_first_prompt(box_bookcase): no number taken from the scene (support tops and extents, every bookcase board's world extents read from the
     compiled model, crate size, crate start, goal xy, base start pose) appears in it, in any of the printed forms (4, 3, 2 decimals, %g), for worlds 0-4;
  2. the prompt carries the schema, the probe cards and the photo note; the old prompts never carry the geometry marker;
  3. the mock geometry reply parses with geometry.parse_priors;
  4. run_probe writes a reading only for a field that is observed AND has an error-table entry (fail closed), and never for an unknown entity."""
import re

import numpy as np
import pytest

from world_model.v2.adapters import geom_probe_run as GR
from world_model.v2.stack import geometry as G
from world_model.v2.stack.proposer import mock as MK
from world_model.v2.stack.proposer import prompt as PR

TASK = "box_bookcase"


def _scene_numbers(i):
    import mujoco
    from world_model.v2.nero.base import scene as BS, smoke as SM
    from world_model.v2.nero.contents import scenes as SC
    _, w = SM.behind_world(TASK, i)
    m = BS.build_spec(w).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    vals = []
    for g in range(m.ngeom):
        n = m.geom(g).name
        if n.startswith(("bookcase", "counter", "pan_table")):
            c, s = d.geom_xpos[g], m.geom_size[g]
            vals += list(c) + list(c - s) + list(c + s) + list(2 * s)
    for k, v in SC.support_tops(w).items():
        vals += [v["top"], *v["x"], *v["y"]]
    vals += list(w.box_outer) + list(w.obj_xy[:2]) + [SC.GRASP_Z]
    from world_model.v2.stack.planners import base_route as BR
    vals += list(BR.default_backend().start_pose(w)[:2])
    try:
        from world_model.v2.stack import tasks as TK
        gx, gy = TK.task(TASK).goal_dxy
        vals += [w.obj_xy[0] + gx, w.obj_xy[1] + gy]
    except Exception:
        pass
    return [float(v) for v in vals if np.isfinite(v)]


def _forms(v):
    out = set()
    for f in ("%.4f", "%.3f", "%.2f", "%g"):
        s = f % v
        s = s.rstrip("0").rstrip(".") if "." in s else s
        if len(s.lstrip("-")) >= 4 and s not in ("0", "-0"):       # 0.82 and longer; short strings (0.8, 1.0) collide with robot facts
            out.add(s)
    return out


@pytest.mark.parametrize("i", range(5))
def test_no_scene_numbers(i):
    text = PR.geo_first_prompt(TASK)
    hits = []
    for v in _scene_numbers(i):
        for s in _forms(v):
            if re.search(r"(?<![\d.])" + re.escape(s) + r"(?!\d)", text):
                hits.append(s)
    assert not hits, sorted(set(hits))[:20]


def test_prompt_content_and_marker():
    text = PR.geo_first_prompt(TASK)
    assert MK.GEO_MARK in text and "measure_box" in text and "room_front" in text
    for f in G.FIELDS:
        assert f in text
    assert MK.GEO_MARK not in PR.user_prompt(TASK)                      # the old prompt never triggers geometry mode
    with pytest.raises(ValueError):
        PR.geo_first_prompt("box")


def test_mock_reply_parses():
    gs = G.parse_priors(MK.geo_reply(TASK))
    assert "crate" in gs.entities and gs.entities_with_role("carried")


class _FakeGP:
    def __init__(self, fields):
        self.fields = fields

    def lab_clouds(self, *a, **k):
        return []

    def entity_geoms(self, m, name):
        return [1]

    def measure_box(self, *a, **k):
        return dict(fit_ok=True, fields=self.fields)


def test_run_probe_fails_closed(monkeypatch):
    fields = {f: dict(value=0.5, observed=(f != "sz"), faces_used=[]) for f in G.FIELDS}
    fake = _FakeGP(fields)
    from world_model.v2.nero.contents import geom_probe as GPm             # run_probe imports it late: patch the module itself
    for name in ("lab_clouds", "entity_geoms", "measure_box"):
        monkeypatch.setattr(GPm, name, getattr(fake, name))
    gs = G.GeometrySet()
    gs.add_entity("crate", "object", roles=("carried",))
    err = {"crate": {"cx": 0.01, "cy": 0.01, "sz": 0.01}}               # sz has an entry but is unobserved; sx etc. have no entry
    rds = GR.run_probe(dict(probe="measure_box", entity="crate", target="crate"), None, None, None, gs, err, clouds=[])
    got = {r for r in ("cx", "cy", "cz", "sx", "sy", "sz", "yaw") if gs.entities["crate"].fields[r].readings}
    assert got == {"cx", "cy"}, got
    assert all(r.source == "MEASURED" and r.guarantee == "sampled" for r in rds)
    with pytest.raises(G.GeometryMissing):
        gs.require("crate", "sx")
    with pytest.raises(ValueError):
        GR.run_probe(dict(probe="measure_box", entity="ghost", target="crate"), None, None, None, gs, err, clouds=[])
