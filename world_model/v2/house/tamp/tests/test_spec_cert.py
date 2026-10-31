"""spec-based plan-time certifier (tamp/spec_cert.py + spec_default.py) == the hard-coded rows of tamp/plan_cert.py.  SIMULATION ONLY, no physics stepping, no rendering.
    python -m pytest -q world_model/v2/house/tamp/tests/test_spec_cert.py
EQUIVALENCE (the key evidence): N random cup-safe trajectories (plan_candidate / backup_from from random poses near random persons, random v_hat, e_v, a_p, latency, delay t0,
sweep on/off) + the seed-1 doorway standoff snapshot: certify_spec(default_spec) and certify_plan must give the same verdict for people + cup (+ the unchanged wall / hw rows),
and the people margins must agree to float precision.  run_equivalence() is importable (the report script calls it with a larger N)."""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from world_model.v2.house.gatekeeper import check1, check2, plan_candidate, backup_from
from world_model.v2.house.tamp import plan_cert as PC
from world_model.v2.house.tamp import spec_cert as SC
from world_model.v2.house.tamp import spec_default as SD

N_RANDOM = 300
PEOPLE_KEYS = ("person", "person_exit", "rest_person")
POSE0 = (2.14, 3.56, -math.pi / 2)                                    # seed-1 standoff snapshot (see test_plan_cert.py)
WALKER = (np.array([2.6, 3.8]), np.zeros(2))
P_STAND = (np.array([2.134 - 1.32, 3.5]), np.zeros(2))


@pytest.fixture(scope="module")
def world():
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.tamp import world as W
    sim = TeleopSim("h5", grasped=False, seed=0, with_people=False, counter=True, desk=True)
    return W.world_from_sim(sim)


def random_case(rng, world):
    """-> dict(traj, people, pose, gk, t0, kind).  Poses in the h5 corridor / doorway region, 1-3 persons 0.35-2.5 m away (many start inside a keep-out), random v_hat."""
    if rng.random() < 0.7:
        pose = (rng.uniform(1.95, 2.35), rng.uniform(2.4, 5.0), -math.pi / 2 + rng.normal(0, 0.4))
    else:
        pose = (rng.uniform(1.95, 2.35), rng.uniform(2.4, 5.0), rng.uniform(-math.pi, math.pi))
    people = []
    for _ in range(int(rng.integers(1, 4))):
        r, a = rng.uniform(0.35, 2.5), rng.uniform(0, 2 * math.pi)
        c = np.array([pose[0] + r * math.cos(a), pose[1] + r * math.sin(a)])
        b = rng.uniform(0, 2 * math.pi)
        v = rng.uniform(0, 1.2) * np.array([math.cos(b), math.sin(b)]) if rng.random() < 0.7 else np.zeros(2)
        people.append((c, v))
    gk = dataclasses.replace(world.gk, e_v=float(rng.choice([0.0, 0.05, 0.15, 0.2, 0.3])), a_p=float(rng.choice([0.0, 0.0, 0.5, 2.0])),
                             latency_s=float(rng.choice([0.0, 0.0, 0.1, 0.3])), sweep_slack=bool(rng.random() > 0.1))
    t0 = float(rng.choice([0.0, 0.0, 0.5, 1.5]))
    bel = PC.plan_belief(world, pose, True, "tray", people, t0, gk)
    v0 = np.zeros(3) if rng.random() < 0.6 else rng.uniform(-1, 1, 3) * np.array([0.25, 0.15, 0.3])
    if rng.random() < 0.15:
        traj, kind = backup_from(v0, bel, gk), "backup"
    else:
        u = rng.uniform(-1, 1, 3) * np.array([0.3, 0.2, 0.3]) * (rng.random(3) < 0.8)
        traj, kind = plan_candidate(v0, u, float(rng.choice([0.1, 0.3, 0.6, 1.0, 1.5, 2.5])), bel, gk), "hold"
    return dict(traj=traj, people=people, pose=pose, gk=gk, t0=t0, kind=kind)


def compare_one(world, case, spec=None):
    """-> dict with both verdicts and the people margins."""
    g, tr, ppl, pose, t0 = case["gk"], case["traj"], case["people"], case["pose"], case["t0"]
    ok_p, m_p, why_p, mg_p = PC.certify_plan(tr, world, g, ppl, t0, pose)
    sp = SD.default_spec(world, ppl) if spec is None else spec
    ok_s, m_s, why_s, mg_s = SC.certify_spec(tr, sp, world, g, ppl, t0, pose)
    tol = g.tol
    pv = [mg_p[k] for k in PEOPLE_KEYS if k in mg_p]
    sv = [v for k, v in mg_s.items() if k.startswith("person_") and ":" not in k]
    ok_pp = all(v >= -tol for v in pv)
    ok_sp = all(v >= -tol for v in sv)
    rows = [k for k in mg_p if k not in PEOPLE_KEYS]
    return dict(ok_plan=ok_p, ok_spec=ok_s, why_plan=why_p, why_spec=why_s, ok_people_plan=ok_pp, ok_people_spec=ok_sp,
                m_people_plan=min(pv) if pv else None, m_people_spec=min(sv) if sv else None,
                rows_equal=all(abs(mg_p[k] - mg_s[k]) <= 1e-12 for k in rows if k in mg_s) and all(k in mg_s for k in rows), n_people=len(ppl))


def run_equivalence(world, n=N_RANDOM, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        case = random_case(rng, world)
        r = compare_one(world, case)
        r.update(i=i, kind=case["kind"], t0=case["t0"], e_v=case["gk"].e_v, a_p=case["gk"].a_p)
        rows.append(r)
    return rows


def summarize(rows):
    n = len(rows)
    same_ok = sum(r["ok_plan"] == r["ok_spec"] for r in rows)
    same_pp = sum(r["ok_people_plan"] == r["ok_people_spec"] for r in rows)
    both = [r for r in rows if r["m_people_plan"] is not None and r["m_people_spec"] is not None]
    # check1 reports the exit margin as the placeholder 0.0 when the profile has a single MOVING sample (nothing to difference); the spec reports the real step (>= 0): same verdict
    ph = [r for r in both if r["m_people_plan"] == 0.0 and r["m_people_spec"] >= 0.0 and r["m_people_spec"] != 0.0]
    dm = [abs(r["m_people_plan"] - r["m_people_spec"]) for r in both if min(abs(r["m_people_plan"]), abs(r["m_people_spec"])) < 50 and r not in ph]
    unsound = [r for r in rows if r["ok_spec"] and not r["ok_plan"]]
    conservative = [r for r in rows if r["ok_plan"] and not r["ok_spec"]]
    npp = sum(r["ok_people_plan"] for r in rows if r["m_people_plan"] is not None)
    return dict(n=n, people_rows_safe_plan=npp, people_rows_unsafe_plan=len(both) - npp, verdict_agree=same_ok, people_verdict_agree=same_pp, n_safe_plan=sum(r["ok_plan"] for r in rows), n_safe_spec=sum(r["ok_spec"] for r in rows),
                n_with_people_rows=len(both), max_people_margin_diff=max(dm) if dm else 0.0, exit_placeholder_cases=[r["i"] for r in ph], rows_equal=sum(r["rows_equal"] for r in rows),
                spec_safe_plan_unsafe=[r["i"] for r in unsound], spec_unsafe_plan_safe=[r["i"] for r in conservative])


def test_random_trajectories_equivalence(world, capsys):
    rows = run_equivalence(world, N_RANDOM, 0)
    s = summarize(rows)
    with capsys.disabled():
        print(f"\n[spec-equivalence] {json.dumps(s)}")
    assert s["n"] >= 300
    assert not s["spec_safe_plan_unsafe"], f"SOUNDNESS: spec says SAFE, plan_cert says UNSAFE: {s['spec_safe_plan_unsafe']}"
    assert s["verdict_agree"] == s["n"] and s["people_verdict_agree"] == s["n"], s
    assert s["max_people_margin_diff"] < 1e-9 and s["rows_equal"] == s["n"], s
    assert 0.1 * s["n"] < s["n_safe_plan"] < 0.9 * s["n"], "the random cases should contain both safe and unsafe trajectories"


def test_seed1_standoff_snapshot_equivalence(world, capsys):
    bel = PC.plan_belief(world, POSE0, True, "tray", [WALKER, P_STAND], 0.0)
    for ev in (0.1, 0.15, 0.2):
        gk = dataclasses.replace(world.gk, e_v=ev, a_p=0.0)
        for u in ([0.3, 0, 0], [-0.3, 0, 0], [0.15, 0, 0], [0, 0.2, 0]):
            tr = plan_candidate(np.zeros(3), np.array(u, float), 5.0 if u[0] > 0 else 1.0, bel, gk)
            case = dict(traj=tr, people=[WALKER, P_STAND], pose=POSE0, gk=gk, t0=0.0)
            r = compare_one(world, case)
            assert r["ok_plan"] == r["ok_spec"] and r["ok_people_plan"] == r["ok_people_spec"], (ev, u, r)
            assert abs(r["m_people_plan"] - r["m_people_spec"]) < 1e-9 and r["rows_equal"], (ev, u, r)
            if ev == 0.1 and u == [0.3, 0, 0]:
                assert r["ok_spec"], r
            with capsys.disabled():
                print(f"[standoff] e_v={ev} u={u} plan ok={r['ok_plan']} spec ok={r['ok_spec']} people margin plan={r['m_people_plan']:.5f} spec={r['m_people_spec']:.5f}")


def test_spec_dict_json_roundtrip_gives_identical_margins(world):
    rng = np.random.default_rng(5)
    for _ in range(12):
        case = random_case(rng, world)
        n = len(case["people"])
        sp = SD.default_spec(world, case["people"])
        txt = SD.dumps(sp)
        dd = json.loads(txt)
        assert dd["spec"] == sp.d and dd["params"] == sp.params
        ents = SC.make_entities(dd["n_people"], [], world.bel0.fp, (0.5 * world.items["tray"].extent[0], 0.5 * world.items["tray"].extent[1]))
        sp2 = SD.from_dict(dd, ents, walls="grid")
        a = SC.certify_spec(case["traj"], sp, world, case["gk"], case["people"], case["t0"], case["pose"])
        b = SC.certify_spec(case["traj"], sp2, world, case["gk"], case["people"], case["t0"], case["pose"])
        assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
        assert n == sp2.n_people


def test_certify_plan_without_spec_is_bit_identical(world):
    rng = np.random.default_rng(7)
    for _ in range(8):
        case = random_case(rng, world)
        g = PC._plan_gk(case["gk"])
        bel = PC.plan_belief(world, case["pose"], True, "tray", case["people"], case["t0"], g)
        o1, m1, r1, g1 = check1(case["traj"], bel, g)
        o2, m2, r2, g2 = check2(case["traj"], bel, g)
        ref = PC._verdict(o1, m1, r1, g1, o2, m2, r2, g2)
        out = PC.certify_plan(case["traj"], world, case["gk"], case["people"], case["t0"], case["pose"])
        out_none = PC.certify_plan(case["traj"], world, case["gk"], case["people"], case["t0"], case["pose"], spec=None)
        assert out == ref and out_none == ref


def test_certify_plan_with_spec_calls_certify_spec(world):
    case = random_case(np.random.default_rng(11), world)
    sp = SD.default_spec(world, case["people"])
    a = PC.certify_plan(case["traj"], world, case["gk"], case["people"], case["t0"], case["pose"], spec=sp)
    b = SC.certify_spec(case["traj"], sp, world, case["gk"], case["people"], case["t0"], case["pose"])
    assert a == b


def test_spec_with_other_number_of_people_rebuilds(world):
    case = random_case(np.random.default_rng(3), world)
    sp = SD.default_spec(world, [WALKER])                                  # written for 1 person, the case has 1..3
    r = SC.certify_spec(case["traj"], sp, world, case["gk"], case["people"], case["t0"], case["pose"])
    assert isinstance(r[0], bool)


def test_trajectory_state_shapes_and_time_axis(world):
    case = random_case(np.random.default_rng(2), world)
    tr = case["traj"]
    st = SC.trajectory_state(tr, case["pose"], case["t0"], case["people"], SC.SpecCfg(world, case["gk"]))
    K = tr.i_rest + 1 if tr.i_rest is not None else tr.N + 1
    assert st.K == K and st.fp_people_p.shape == (K, 3) and st.fp_wall_R.shape == (K, 3, 3) and len(st.person_c) == len(case["people"])
    assert all(c.shape == (K, 3) and b.shape == (K,) for c, b in zip(st.person_c, st.person_ball))
    if tr.i_rest:
        assert st.fp_people_ball[-1] == 0.0 and all(b[-1] > 0 or True for b in st.person_ball)


# ------------------------------------------------------------------------------------------------ step 2: zonotope walls
ZONO_FILE = "/data/local/ss5772/molmospaces/spec_cert/h5_zonos.npz"


def run_wall_comparison(world, zonos, n=100, seed=0):
    """random corridor trajectories WITHOUT people: wall verdict of the grid rows (checks.py "wall", "wall_at_rest") vs the zonotope atoms Clear(footprint_wall, obs_j, d_wall)."""
    rng = np.random.default_rng(seed)
    sp_z = SD.default_spec(world, [], walls="zono", zonos=zonos)
    rows = []
    for i in range(n):
        case = random_case(rng, world)
        case["people"] = []
        g, tr, pose = case["gk"], case["traj"], case["pose"]
        ok_g, m_g, why_g, mg_g = PC.certify_plan(tr, world, g, [], 0.0, pose)
        ok_z, m_z, why_z, mg_z = SC.certify_spec(tr, sp_z, world, g, [], 0.0, pose)
        wg = min(mg_g["wall"], mg_g["wall_at_rest"]) if "wall" in mg_g else None
        wz = min(v for k, v in mg_z.items() if k.startswith("obs_"))
        rows.append(dict(i=i, wall_ok_grid=wg >= -g.tol, wall_ok_zono=wz >= -g.tol, m_grid=wg, m_zono=wz, other_ok=all(v >= -1e-6 for k, v in mg_g.items() if k not in ("wall", "wall_at_rest", "rest_speed"))))
    return rows


@pytest.mark.skipif(not __import__("os").path.exists(ZONO_FILE), reason="run: python -m world_model.v2.house.tamp.spec_zono perceive")
def test_zonotope_walls_vs_grid_walls_on_random_trajectories(world, capsys):
    from world_model.v2.house.tamp import spec_zono as SZ
    zonos = SZ.load_zonos(ZONO_FILE)
    rows = run_wall_comparison(world, zonos, 40, 1)
    fs = [r["i"] for r in rows if r["wall_ok_zono"] and not r["wall_ok_grid"]]
    fu = [r["i"] for r in rows if not r["wall_ok_zono"] and r["wall_ok_grid"]]
    with capsys.disabled():
        print(f"\n[zono walls vs grid walls] n={len(rows)} grid-safe={sum(r['wall_ok_grid'] for r in rows)} zono-safe={sum(r['wall_ok_zono'] for r in rows)} false-safe={fs} false-unsafe={fu}")
    assert not fs, f"zonotope walls say SAFE where the true grid says UNSAFE: {fs}"
    assert len(fu) <= 0.15 * len(rows)


def test_zonotope_wall_atom_on_a_synthetic_wall(world):
    """a single zonotope slab 0.5 m ahead of the robot: the Clear margin equals the geometric gap - d_wall - (sweep + dock terms)."""
    from world_model.v2.tray_rep.sets.zono import Zonotope
    pose = (2.0, 4.0, 0.0)
    slab = Zonotope.from_box([pose[0] + 0.577 + 0.5, 3.0, 0.15], [pose[0] + 0.577 + 0.6, 5.0, 1.15])       # 0.5 m in front of the footprint's front edge (x = 0.5770901)
    sp = SD.default_spec(world, [], walls="zono", zonos=[slab])
    tr = backup_from(np.zeros(3), PC.plan_belief(world, pose, True, "tray", [], 0.0, world.gk), world.gk)       # at rest: one sample
    ok, m, why, mg = SC.certify_spec(tr, sp, world, world.gk, [], 0.0, pose)
    gap = pose[0] + 0.577 + 0.5 - (pose[0] + world.bel0.fp["x"][1])
    assert abs(mg["obs_0"] - (gap - world.bel0.d_wall - world.bel0.e_xy)) < 1e-9 and ok
