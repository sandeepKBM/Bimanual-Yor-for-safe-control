"""Owner gate (11-19, plan Phase 2): the depth box fit on the four lab cameras (nero/contents/geom_probe.py), box_bookcase, CLEAN sim depth.
SIMULATION ONLY, CPU (osmesa), no network.  Heavy (10 worlds x 4 cameras at 2K): run on ilab cores.
PYTHONPATH=. MUJOCO_GL=osmesa python -m pytest -q world_model/v2/nero/contents/tests/test_geom_probe_gate.py

Checks (truth from the model, computed here, independent of the fit):
  1. every field the fit marks observed is within TOL of the true box (5 mm / 0.5 deg, ASSUMED gate tolerance), worlds 0-9;
  2. the crate's footprint, top and yaw are observed in every world (the four cameras see it);
  3. the hidden contents never appear in any camera's cloud; robot pixels are dropped;
  4. back-projection: every crate point lies within 2 mm of the crate's true surface box (clean depth; 2 mm = pixel footprint at 4 m, ASSUMED).
The fit's own true_box is NOT used: the truth here is the geom pose and size read directly."""
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco

from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.contents import geom_probe as GP
from world_model.v2.nero.contents import plan_images as PI
from world_model.v2.nero.contents import scenes as SC

TOL_M, TOL_DEG = 0.005, 0.5           # ASSUMED (gate tolerance, 11-19)
WORLDS = range(10)
BOARDS = ("counter", "bookcase_l1_floor", "bookcase_l1_base", "bookcase_plinth", "bookcase_mid", "bookcase_top", "bookcase_side_left", "bookcase_side_right",
          "bookcase_back", "pan_table")
GEOM_OF = {"pan_table": "pan_table_top"}      # entity -> the geom that defines its truth box


def _scene(i):
    _, w = SM.behind_world("box_bookcase", i)
    m = BS.build_spec(w).compile()
    d = mujoco.MjData(m)
    SC.settle(m, d, PI.SETTLE_S)
    mujoco.mj_forward(m, d)
    return w, m, d


def _true_geom_box(m, d, name):
    g = m.geom(name).id
    R = d.geom_xmat[g].reshape(3, 3)
    s = 2 * m.geom_size[g]
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return dict(cx=d.geom_xpos[g][0], cy=d.geom_xpos[g][1], cz=d.geom_xpos[g][2], sx=s[0], sy=s[1], sz=s[2], yaw=yaw)


def _true_crate(m, d):
    """union of the crate's wall / floor / lid boxes (no tabs) in the obj body frame -> upright box."""
    b = m.body("obj").id
    R = d.xmat[b].reshape(3, 3)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom(g).name.startswith("tab"):
            continue
        Rg = d.geom_xmat[g].reshape(3, 3)
        c = R.T @ (d.geom_xpos[g] - d.xpos[b])
        e = np.abs(R.T @ Rg) @ m.geom_size[g]
        lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
    cw = d.xpos[b] + R @ ((lo + hi) / 2)
    s = hi - lo
    return dict(cx=cw[0], cy=cw[1], cz=cw[2], sx=s[0], sy=s[1], sz=s[2], yaw=float(np.arctan2(R[1, 0], R[0, 0])))


def _fold(t):
    """the fit folds yaw to [-pi/4, pi/4) and swaps sx/sy: do the same to the truth."""
    t = dict(t)
    y = (t["yaw"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    k = int(round((t["yaw"] - y) / (np.pi / 2))) % 2
    if k:
        t["sx"], t["sy"] = t["sy"], t["sx"]
    t["yaw"] = y
    t["top_z"] = t["cz"] + t["sz"] / 2           # the fit also reports the top face height
    return t


def _err(f, v, t):
    if f == "yaw":
        return abs(np.degrees((v - t + np.pi / 4) % (np.pi / 2) - np.pi / 4)), TOL_DEG
    return abs(v - t), TOL_M


@pytest.fixture(scope="module", params=list(WORLDS))
def fitted(request):
    w, m, d = _scene(request.param)
    clouds = GP.lab_clouds(m, d, w, noise="clean", rng=np.random.default_rng(0))
    sup = float(SC.support_tops(w)["counter"]["top"])
    out = {"crate": (GP.measure_box(clouds, GP.entity_geoms(m, "crate"), support_top_z=sup), _fold(_true_crate(m, d)))}
    for n in BOARDS:
        out[n] = (GP.measure_box(clouds, GP.entity_geoms(m, n)), _fold(_true_geom_box(m, d, GEOM_OF.get(n, n))))
    return request.param, m, d, clouds, out


def test_observed_fields_match_truth(fitted):
    i, m, d, clouds, out = fitted
    bad = []
    for name, (fit, truth) in out.items():
        for f, rec in fit.get("fields", {}).items():
            if rec.get("observed"):
                e, tol = _err(f, rec["value"], truth[f])
                if e > tol:
                    bad.append((name, f, round(float(e), 4)))
    assert not bad, f"world {i}: {bad}"


def test_crate_observed(fitted):
    i, m, d, clouds, out = fitted
    fit = out["crate"][0]
    assert fit["fit_ok"], (i, fit.get("why"))
    for f in ("cx", "cy", "sx", "sy", "yaw", "sz", "cz"):
        assert fit["fields"][f]["observed"], (i, f)


def test_no_hidden_or_robot_points(fitted):
    i, m, d, clouds, out = fitted
    hidden = {g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name == "contents"}
    robot = {g for g in range(m.ngeom) if any(m.body(int(m.geom_bodyid[g])).name.startswith(p) for p in ("left_arm", "right_arm", "Lift", "base", "Part_", "MK4"))}
    for c in clouds:
        ids = set(np.asarray(c["geom_ids"]).tolist())
        assert not (ids & hidden), (i, c["label"])
        assert not (ids & robot), (i, c["label"])


def test_crate_points_on_surface(fitted):
    i, m, d, clouds, out = fitted
    t = _true_crate(m, d)
    b = m.body("obj").id
    R = d.xmat[b].reshape(3, 3)
    crate = {g for g in GP.entity_geoms(m, "crate") if not m.geom(g).name.startswith("tab")}   # the grasp tabs stick out of the box: not part of the truth box
    half = np.array([t["sx"], t["sy"], t["sz"]]) / 2
    for c in clouds:
        sel = np.isin(np.asarray(c["geom_ids"]), list(crate))
        if not sel.any():
            continue
        q = (np.asarray(c["points"])[sel] - np.array([t["cx"], t["cy"], t["cz"]])) @ R
        outside = np.maximum(np.abs(q) - half, 0).max(1)                  # 0 inside or on the box
        assert outside.max() <= 0.002, (i, c["label"], float(outside.max()))
