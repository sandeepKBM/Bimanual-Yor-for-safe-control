"""fast unit tests for the pick-proposer fine-tune data (no GPU / GL; mujoco only imported for the box-drawing / projection checks)."""
import json
import math
import os
import re

import numpy as np
import pytest

from world_model.v2.house.pick import assign as A
from world_model.v2.house.pick import planner as PL
from world_model.v2.vlm.ftdata import labels as L
from world_model.v2.vlm.ftdata import priors as P
from world_model.v2.vlm.ftdata import split as S

INVENTORY = "/data/local/ss5772/ft_study/inventory.json"
# categories met by gen.enumerate_instances over all 21 local houses (OBB filter, freejoint, mesh visuals) -- 2026-10-16 scan, plus the earlier inventory's
SCANNED_CATEGORIES = """Apple AlarmClock BaseballBat BasketBall Book Bottle Bowl Box Bread ButterKnife Candle CellPhone Cloth CoffeeMachine Cup DeskLamp DishSponge Egg Fork GarbageCan
Kettle Knife Ladle Laptop Lettuce Mug Newspaper Pan PaperTowelRoll Pen Pencil PepperShaker Pillow Plate Potato Pot RemoteControl SaltShaker SoapBar SoapBottle Spatula Spoon
SprayBottle Statue TeddyBear TissueBox Toaster Tomato ToiletPaper Vase WineBottle""".split()


# ---------------------------------------------------------------------------------------------------------------- priors
def test_prior_table_covers_all_categories():
    t = P.load()
    missing = [c for c in SCANNED_CATEGORIES if c not in t]
    assert not missing, missing
    if os.path.exists(INVENTORY):
        inv = json.load(open(INVENTORY))
        cats = {r["cat"] for r in inv if r["free"] and r["mass"] < 5 and max(r["ext"]) < 0.5}
        assert cats <= set(t), sorted(cats - set(t))


def test_prior_entries_sane():
    for c, p in P.load().items():
        assert 0 < p.mass_lo < p.mass_hi <= 10, c
        assert isinstance(p.fragile, bool) and isinstance(p.has_handle, bool) and p.display and p.contents == "empty"
        if p.density:
            assert 0 < p.density[0] < p.density[1]
    assert "ASSUMPTION" in json.load(open(P.PRIOR_PATH))["_ASSUMPTION"]


def test_mass_range_density_and_bounds():
    t = P.load()
    lo, hi = P.mass_range("Book", 0.43 * 0.30 * 0.03)           # 3.9 L at 450-900 -> 1.7-3.5 clipped to <=2.5
    assert 0.1 <= lo < hi <= 2.5
    lo_s, hi_s = P.mass_range("Book", 0.15 * 0.10 * 0.02)       # small book 0.3 L -> 0.135-0.27 kg
    assert lo_s == pytest.approx(0.14, abs=0.015) and hi_s == pytest.approx(0.27, abs=0.01)
    lo_b, hi_b = P.mass_range("Book", 10.0)                     # absurdly large -> clipped to bounds
    assert hi_b == 2.5 and lo_b < hi_b
    lo_t, hi_t = P.mass_range("Book", 1e-6)                     # absurdly small -> lower bound
    assert lo_t == 0.1 and hi_t > lo_t
    assert P.mass_range("Mug", 1.0) == [0.2, 0.5]               # no density: bounds regardless of volume
    for c in t:
        for v in (1e-5, 1e-3, 5e-2):
            lo, hi = P.mass_range(c, v)
            assert 0 < lo < hi


# ---------------------------------------------------------------------------------------------------------------- labels
def _random_cases(n=400, seed=1):
    rng = np.random.default_rng(seed)
    cats = sorted(P.load())
    for _ in range(n):
        c = cats[int(rng.integers(len(cats)))]
        a, b, h = np.sort(rng.uniform(0.01, 0.48, 2))[::-1].tolist() + [float(rng.uniform(0.01, 0.4))]
        yield c, a, b, h, [float(rng.uniform(0.3, 1.5)), float(rng.uniform(-0.8, 0.4)), float(rng.uniform(0, 1))], float(rng.uniform(-1.5, 1.5))


@pytest.mark.parametrize("policy", ["least_bad", "drop"])
def test_answers_parse_roundtrip_and_equal_assign(policy):
    rng = np.random.default_rng(0)
    n_ok = n_inf = n_drop = 0
    for c, a, b, h, cen, yaw in _random_cases():
        prior = P.load()[c]
        mr = P.mass_range(c, a * b * h)
        ne = L.noisy_extent(rng, a, b, h)
        rl = L.rule_label(*ne, cen, yaw, mr[1], policy)
        plan = A.assign(ne, cen, yaw, mass_vlm=mr[1])               # the rule, called directly on the same inputs
        if rl is None:
            assert policy == "drop" and plan.mode == "infeasible"
            n_drop += 1
            continue
        assert rl["rule_mode"] == plan.mode
        ans = L.answer_obj(rl, mr, prior)
        text = json.dumps(ans)
        p = PL.parse_vlm_plan(text)
        assert p is not None
        assert (p["arms"], p["arm"]) == (ans["arms"], ans["arm"]) and p["grasp"] == ans["grasp"]
        assert p["mass_range"] == mr and p["mass_kg"] == mr[1]
        assert p["contents"] == "empty" and p["fragile"] == prior.fragile and p["handle"] == prior.has_handle
        assert p["why"] == ans["why"] and len(ans["why"].split()) <= 22
        if plan.mode == "single":
            assert (ans["arms"], ans["arm"], ans["grasp"]) == (1, plan.arm, "pinch")
        elif plan.mode == "bimanual":
            assert (ans["arms"], ans["arm"], ans["grasp"]) == (2, "both", "squeeze")
        else:
            n_inf += 1
            assert ans["why"].startswith("No grasp fits")
            assert (ans["arms"] == 1) == (ans["arm"] in ("left", "right"))
        n_ok += plan.mode != "infeasible"
    assert n_ok > 20 and (n_inf > 0 if policy == "least_bad" else n_drop > 0)


def test_noisy_extent_sigma_and_order():
    rng = np.random.default_rng(3)
    xs = np.array([L.noisy_extent(rng, 0.2, 0.1, 0.05) for _ in range(4000)])
    assert np.all(xs[:, 0] >= xs[:, 1])
    assert abs(xs[:, 2].std() - 0.005) < 0.0004 and abs(xs[:, 2].mean() - 0.05) < 0.0004


def test_prompts_are_the_planners_own():
    sys_ = L.system_prompt()
    assert sys_ == PL.SYSTEM.format(open_cm=A.SINGLE_MAX_WIDTH_M * 100, pay=A.SINGLE_MAX_KG, minsep=A.MIN_HAND_SEP_M * 100, maxspan=A.MAX_SPAN_M * 100)
    assert "6 cm" in sys_ and "0.75 kg" in sys_ and "25 and 45 cm" in sys_
    u = L.user_prompt("mug", 0.123, 0.081, 0.095, 0.55, 0.12)
    assert u == PL.USER.format(label="mug", a_cm=12.3, b_cm=8.1, h_cm=9.5, x=0.55, y=0.12, side="left")
    assert "(right of the robot centre line)" in L.user_prompt("mug", 0.1, 0.1, 0.1, 0.5, -0.4)
    assert "(left of the robot centre line)" in L.user_prompt("mug", 0.1, 0.1, 0.1, 0.5, -0.2)


def test_head_constants_match_fp_render():
    from world_model.v2.house.ocs import fp_render as FR
    assert (L.HEAD_X, L.HEAD_Y, L.HEAD_Z) == pytest.approx(tuple(FR.HEAD_POS))
    from world_model.v2.vlm.ftdata import gen
    assert gen.PITCH_DEG == FR.HEAD_PITCH_DEG and gen.FOVY == FR.HEAD_FOVY


def test_record_format_and_no_sim_mass_in_text():
    prior = P.load()["Mug"]
    rl = L.rule_label(0.081, 0.08, 0.095, [0.5, -0.1, 0.9], 0.0, 0.5)
    ans = L.answer_obj(rl, [0.2, 0.5], prior)
    meta = dict(task="arms", mass_sim_kg=0.0707, asset_id="Mug_1", house=3)
    r = L.arms_record("/x/y.png", "mug", (0.081, 0.08, 0.095), [0.5, -0.1, 0.9], ans, meta)
    assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
    u = r["messages"][1]["content"]
    assert u[0] == {"type": "image", "image": "/x/y.png"} and u[1]["type"] == "text"
    txt = json.dumps(r["messages"])
    assert "0.0707" not in txt and "Mug_1" not in txt and "mass_sim" not in txt
    assert list(json.loads(r["messages"][2]["content"])) == ["arms", "arm", "grasp", "mass_kg", "contents", "fragile", "handle", "why"]
    n = L.name_record("/x/y.png", "mug", meta)
    assert n["messages"][1]["content"][1]["text"] == "Name the boxed object in one or two words." and n["messages"][2]["content"] == "mug"


def test_label_noise_kinds():
    rng = np.random.default_rng(0)
    kinds = [L.pick_label(rng, "Mug") for _ in range(3000)]
    frac = lambda k: sum(1 for _, kk in kinds if kk == k) / len(kinds)
    assert 0.02 < frac("generic") < 0.07 and 0.04 < frac("wrong") < 0.09
    assert all(l == "mug" for l, k in kinds if k == "true") and all(l != "mug" for l, k in kinds if k == "wrong")


# ---------------------------------------------------------------------------------------------------------------- geometry / drawing
def test_camera_projection_conventions():
    from world_model.v2.vlm.ftdata import gen
    ctr = np.array([1.0, 2.0, 0.8])
    for az in (0, 37, 90, 200):
        cam = gen.make_cam(ctr, az, -36.3, 1.0, 0, 0)
        uv, z = gen.project(ctr[None], cam["pos"], cam["fwd"], cam["right"], cam["up"])
        assert uv[0] == pytest.approx([320, 180], abs=1e-6) and z[0] == pytest.approx(1.0)
        assert cam["up"][2] > 0 and cam["pos"][2] > ctr[2]                      # looking down from above
    cam = gen.make_cam(ctr, 0, -36.3, 1.0, 100, 40)                         # object right / below centre in the image
    uv, _ = gen.project(ctr[None], cam["pos"], cam["fwd"], cam["right"], cam["up"])
    assert uv[0] == pytest.approx([420, 220], abs=1e-6)
    # robot frame: az=0 camera looks along +x; a point to the camera's left (+y world) has +y robot
    cam = gen.make_cam(np.zeros(3), 0, -36.3, 1.0, 0, 0)
    rf = gen.robot_frame(np.array([0.5, 0.3, 0.0]), cam)
    rel = np.array([0.5, 0.3, 0.0]) - cam["pos"]
    assert rf[0] == pytest.approx(L.HEAD_X + rel[0]) and rf[1] == pytest.approx(L.HEAD_Y + 0.3)
    cam = gen.make_cam(np.zeros(3), 90, -36.3, 1.0, 0, 0)                     # facing +y: world -x is robot left
    assert gen.robot_frame(np.array([-0.3, 0.0, 0.0]), cam)[1] > L.HEAD_Y + 0.25


def test_sample_cam_hits_pixel_target():
    from world_model.v2.vlm.ftdata import gen
    rng = np.random.default_rng(5)
    Pw = np.random.default_rng(0).uniform(-0.5, 0.5, (500, 3)) * np.array([0.12, 0.08, 0.1]) + np.array([1, 1, 0.8])
    hs = []
    for _ in range(60):
        cam = gen.sample_cam(rng, Pw.mean(0), Pw, 0.12)
        if cam is not None:
            pb = cam["proj_bbox"]
            hs.append(pb[3] - pb[1])
            assert -gen.PITCH_DEG - gen.PITCH_JIT - 1e-6 <= cam["elevation"] <= -gen.PITCH_DEG + gen.PITCH_JIT + 1e-6
            assert gen.DIST_RANGE[0] <= cam["distance"] <= gen.DIST_RANGE[1]
    assert len(hs) > 30 and min(hs) >= gen.PX_ACCEPT[0]


def test_box_matches_session_crop_png(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from world_model.v2.house.pick import session as SE
    from world_model.v2.vlm.ftdata import gen
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    box = [120, 80, 260, 190]

    class Fake:
        out_dir = str(tmp_path)

    p = SE.PickSession._crop_png(Fake(), dict(rgb=rgb), dict(id=7, box2d=box, label="x"))
    ref = cv2.imread(p)
    mine_path = str(tmp_path / "mine.png")
    gen.write_png(mine_path, gen.draw_selected_box(rgb, box))
    mine = cv2.imread(mine_path)
    assert ref.shape == mine.shape == (360, 640, 3) and np.array_equal(ref, mine)
    # colour / thickness: the selection colour is yellow (RGB 255,220,0) 3 px wide, antialiased
    on = mine[134, 120 - 1:120 + 2][:, ::-1]
    assert (on[:, 0] > 200).any() and (on[:, 1] > 150).any() and (on[:, 2] < 80).any()


# ---------------------------------------------------------------------------------------------------------------- resume
def test_resume_scan_and_seed(tmp_path):
    from world_model.v2.vlm.ftdata import gen
    os.makedirs(tmp_path / "shards")
    with open(tmp_path / "shards" / "h1_w0.jsonl", "w") as f:
        f.write(json.dumps({"meta": {"sample_id": "h1_i0001_v0", "task": "arms"}}) + "\n")
        f.write(json.dumps({"meta": {"sample_id": "h1_i0001_v0|name", "task": "name"}}) + "\n")
        f.write('{"meta": {"sample_id": "h1_i0002_v0", "ta')           # a line cut by a kill
    assert gen.done_ids(str(tmp_path)) == {"h1_i0001_v0"}
    assert gen.seed_for(0, 1, 2, 3) == gen.seed_for(0, 1, 2, 3) != gen.seed_for(0, 1, 2, 4)


# ---------------------------------------------------------------------------------------------------------------- splits
def _fake_records(n_houses=8, seed=0):
    rng = np.random.default_rng(seed)
    cats = sorted(P.load())[:30]
    recs = []
    for h in range(n_houses):
        for i in range(60):
            c = cats[int(rng.integers(len(cats)))]
            asset = f"{c}_{int(rng.integers(3))}"
            ext = [0.05 + 0.01 * (hash(c) % 30), 0.05, 0.05]
            for v in range(2):
                for task in ("arms", "name"):
                    a = {"arms": 1, "arm": "left"}
                    recs.append({"messages": [{}, {}, {"role": "assistant", "content": json.dumps(a) if task == "arms" else c}],
                                 "meta": dict(sample_id=f"h{h}_i{i:04d}_v{v}", task=task, house=h, instance=i, category=c, asset_id=asset, true_extent_obb=ext, rule_mode="single")})
    return recs


def test_split_disjoint_no_leakage():
    recs = _fake_records()
    spec = S.make_spec([r["meta"] for r in recs], n_cats=9, n_houses=4, asset_frac=0.10, val_frac=0.05, seed=0)
    assert len(spec["held_cats"]) == 9 and len(spec["held_houses"]) == 4 and spec["held_assets"]
    out = S.split_records(recs, spec)
    assert sum(len(v) for v in out.values()) == len(recs)
    ids = {s: {r["meta"]["sample_id"] + r["meta"]["task"] for r in rs} for s, rs in out.items()}
    seen = set()
    for s in S.SPLITS:
        assert not (ids[s] & seen), s
        seen |= ids[s]
    tr = out["train"] + out["val"]
    assert tr and out["val"] and all(out[s] for s in ("test_cat", "test_house", "test_asset"))
    cats_tr, houses_tr, assets_tr = ({r["meta"][k] for r in tr} for k in ("category", "house", "asset_id"))
    assert not (cats_tr & set(spec["held_cats"])) and not (houses_tr & set(spec["held_houses"])) and not (assets_tr & set(spec["held_assets"]))
    assert {r["meta"]["category"] for r in out["test_cat"]} <= set(spec["held_cats"])
    assert {r["meta"]["house"] for r in out["test_house"]} <= set(spec["held_houses"])
    assert {r["meta"]["asset_id"] for r in out["test_asset"]} <= set(spec["held_assets"])
    # held-out assets belong to categories that stay seen
    assert all(a.rsplit("_", 1)[0] in cats_tr for a in spec["held_assets"])
    # an instance never straddles two files
    where = {}
    for s, rs in out.items():
        for r in rs:
            assert where.setdefault((r["meta"]["house"], r["meta"]["instance"]), s) == s
    man = S.manifest(out, spec)
    assert sum(v["n_records"] for v in man["splits"].values()) == len(recs)


def test_split_is_deterministic():
    recs = _fake_records()
    m = [r["meta"] for r in recs]
    assert S.make_spec(m, seed=2) == S.make_spec(m, seed=2) and S.make_spec(m, seed=2)["held_cats"] != S.make_spec(m, seed=3)["held_cats"]
