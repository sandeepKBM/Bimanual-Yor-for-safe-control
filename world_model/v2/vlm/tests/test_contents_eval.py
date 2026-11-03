import json, math, pytest
import numpy as np
from world_model.v2.vlm import contents_eval as CE


def test_last_json_and_truncation():
    assert CE.parse_last_json('a {"x":1} b {"x":2}')["x"] == 2
    assert CE.parse_last_json('<think>{"x":9}</think>{"x":3}')["x"] == 3
    assert CE.parse_last_json('{"x":1} {"x": [1,', None)["x"] == 1          # trailing fragment ignored
    assert CE.parse_last_json('{"x":1}', "length") is None                  # cut at the cap = no answer
    assert CE.parse_last_json("no json") is None


def test_clean():
    o = CE.clean({"total_mass_kg": [3, 1], "p_cover": "0.9"})
    assert o["total_mass_kg"] == [1.0, 3.0] and o["p_cover"] == 0.9
    assert CE.clean({"total_mass_kg": "x"}) is None


def test_winkler_and_conformal():
    assert CE.winkler(1, 2, 1.5) == 1
    assert CE.winkler(1, 2, 3) == pytest.approx(1 + 20)
    # intervals centred at 1 with halfwidth 1; truths at distances 0..9 -> need k = d; 90% of 10 -> 9th smallest = 8
    ivs = [(0.0, 2.0)] * 10
    ys = [1.0 + d for d in range(10)]
    assert CE.conformal_k(ivs, ys, 0.9) == 8.0


def test_score_and_mock_roundtrip(tmp_path, capsys):
    recs = []
    for i, task in enumerate(["tray2", "tray2", "tray2", "tray2", "box", "box"]):
        t = 1.0 + 0.3 * i
        recs.append(dict(id=f"r{i}", task=task, view="torso", image="", facts="f", contents_def="c",
                         truth=dict(total_mass_kg=t, contents_kg=0.5 * t, total_fill_ml=None if task == "box" else 100.0),
                         prior=dict(total_mass_kg=[0.5, 5.0], contents_kg=[0.1, 4.0], total_fill_ml=None if task == "box" else [0, 500])))
    rp = tmp_path / "rec.jsonl"; rp.write_text("\n".join(map(json.dumps, recs)))
    ap = tmp_path / "ans.jsonl"
    CE.main(["ask", "--mock", "--records", str(rp), "--out", str(ap)])
    rows = [json.loads(l) for l in open(ap)]
    assert len(rows) == 6 and all(r["parsed"] for r in rows)
    sp = tmp_path / "s.md"
    CE.main(["score", "--answers", str(ap), "--out", str(sp)])
    txt = sp.read_text()
    assert "point-centre" in txt and "prior (full box)" in txt and "mock" in txt


def test_render_tiny(tmp_path):
    pytest.importorskip("mujoco")
    CE.main(["render", "--n-per-task", "1", "--tasks", "box", "--out", str(tmp_path)])
    r = [json.loads(l) for l in open(tmp_path / "records.jsonl")]
    assert len(r) == 2 and (tmp_path / "img").exists()
    assert r[0]["truth"]["total_mass_kg"] > 0.75


def test_compose_v2_invisible_fill_is_full_range_and_box_product():
    from world_model.v2.vlm import contents_eval as CE
    o = CE.compose_v2("tray4", {"n_mugs": 4, "fill_visible": False, "fill_frac": [0.2, 0.3]})
    assert o["total_fill_ml"] == [0.0, 1000.0]                       # not visible -> 0..capacity, whatever fill_frac says
    assert abs(o["total_mass_kg"][0] - 1.5) < 1e-9 and abs(o["total_mass_kg"][1] - 2.5) < 1e-9
    o = CE.compose_v2("tray2", {"n_mugs": 2, "fill_visible": True, "fill_frac": [0.5, 0.25]})
    assert abs(o["total_fill_ml"][0] - 125.0) < 1e-9 and abs(o["total_fill_ml"][1] - 250.0) < 1e-9
    o = CE.compose_v2("box", {"n_items": [6, 12], "item_mass_kg": [0.4, 0.3]})
    assert abs(o["contents_kg"][0] - 1.8) < 1e-9 and abs(o["contents_kg"][1] - 4.8) < 1e-9
    assert CE.compose_v2("box", {"n_items": "many"}) is None
