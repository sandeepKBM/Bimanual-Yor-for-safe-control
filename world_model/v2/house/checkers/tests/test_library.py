"""Certified viability library tests (CPU, synthetic beliefs, no MuJoCo).  nice -n 15 python -m pytest -q world_model/v2/house/checkers/tests/test_library.py"""
import math

import numpy as np
import pytest
from scipy import ndimage

from world_model.v2.house.gatekeeper import GKConfig, Belief, plan_candidate, check1, check2
from world_model.v2.house.teleop.scene import OccGrid
from world_model.v2.house.checkers.library import Library, RouteProgress, Ctx, ctx_from_belief, contained, path_info, map_key

GK = GKConfig()
R_CUP = np.array([0.35, -0.20])


def make_grid(wall_x=5.0, size=10.0, res=0.02):
    n = int(size / res)
    occ = np.zeros((n, n), bool)
    occ[:, int(wall_x / res):] = True
    occ[:3, :] = occ[-3:, :] = True
    occ[:, :3] = True
    return OccGrid(res, 0.0, 0.0, occ, ndimage.distance_transform_edt(~occ) * res)


GRID = make_grid()
GRID2 = make_grid(wall_x=4.9)


def bel(pose=(1.5, 5.0, 0.0), **kw):
    d = dict(r_cup=R_CUP.copy(), grid=GRID, pose_wall=np.array(pose, float), pose_people=np.array(pose, float), e_xy=0.03, e_th=0.005)
    d.update(kw)
    return Belief(**d)


def seg(b, u=(0.3, 0.0, 0.0), T_S=0.6, v0=(0, 0, 0)):
    return plan_candidate(np.array(v0, float), np.array(u, float), T_S, b, GK)


def admit(tr, b):
    return check1(tr, b, GK)[0] and check2(tr, b, GK)[0]


def rest_lib(xs=(1.0, 1.5, 2.0, 2.5), y=5.0, grow=None, **kw):
    lib = Library(GK, progress_fn=RouteProgress([[0.0, y], [4.0, y]]).remaining, grow=grow, **kw)
    for x in xs:
        b = bel(pose=(x, y, 0.0))
        e = lib.add(seg(b), b, GK)
        assert e is not None
    return lib


# ------------------------------------------------------------------------------------------------------------------ add / grow
def test_add_rejects_uncertified_and_grow_stays_admitted():
    lib = Library(GK)
    b_bad = bel(pose=(4.7, 5.0, 0.0))                                   # 0.3 m from the wall: a 0.3 m/s ramp+hold+stop cannot clear it
    tr = seg(b_bad, T_S=1.0)
    assert not admit(tr, b_bad)
    assert lib.add(tr, b_bad, GK) is None and lib.n_rejected == 1
    b = bel(pose=(1.5, 5.0, 0.0), people=(np.array([3.0, 7.5]),))
    tr = seg(b)
    e = lib.add(tr, b, GK, grow=dict(xy=0.3, p=0.5, mu=0.05))
    assert e is not None and e.X.e_xy > b.e_xy + 1e-3 and e.X.mu_lo < b.mu_lo
    # the grown region is itself admitted, and a bit more is not (bisection reached the boundary)
    bx = b.with_(e_xy=e.X.e_xy, e_th=e.X.e_th, e_p=e.X.e_p, mu_lo=e.X.mu_lo)
    assert admit(tr, bx)


# ------------------------------------------------------------------------------------------------------------------ containment soundness / monotonicity
def _rand_belief(rng, ppl):
    pose = (rng.uniform(0.7, 2.8), 5.0 + rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1))
    e = rng.uniform(0.0, 0.09)
    kw = dict(pose=pose, e_xy=e, e_th=0.15 * e, mu_lo=rng.uniform(0.12, 0.2), e_p=rng.uniform(0, 0.1), v_p_max=rng.uniform(0.8, 1.5))
    if ppl:
        kw["people"] = tuple(np.array([rng.uniform(0.5, 4.0), rng.uniform(2.0, 8.0)]) for _ in range(rng.integers(1, 3)))
    return bel(**kw)


@pytest.mark.parametrize("ppl", [False, True])
def test_containment_soundness_never_returns_inadmissible(ppl):
    rng = np.random.default_rng(1)
    base = bel(people=(np.array([2.0, 7.5]), np.array([3.5, 2.5]))) if ppl else bel()
    lib = Library(GK, grow=dict(xy=0.12, p=0.2, mu=0.10))
    for x in np.arange(0.9, 3.0, 0.15):
        b = base.with_(pose_wall=np.array([x, 5.0, 0.0]), pose_people=np.array([x, 5.0, 0.0]))
        for u, T in (((0.3, 0, 0), 0.6), ((0.3, 0, 0), 0.1), ((0.2, 0.1, 0.0), 0.3)):
            lib.add(plan_candidate(np.zeros(3), np.array(u), T, b, GK), b, GK)
    assert len(lib) > 10
    n_hit = 0
    for _ in range(400):
        b = _rand_belief(rng, ppl)
        for m in lib.lookup(b, gk=GK):
            n_hit += 1
            tr = m.entry.traj()
            assert check1(tr, b, GK)[0] and check2(tr, b, GK)[0], (m.entry.summary(), b)     # independent re-check
    assert n_hit > 30                                                                          # not vacuous


def test_containment_independent_of_gatekeeper_verify_flag():
    lib = rest_lib(grow=dict(xy=0.1, mu=0.1))
    rng = np.random.default_rng(3)
    for _ in range(100):
        b = _rand_belief(rng, False)
        a = [m.entry.id for m in lib.lookup(b)]
        c = [m.entry.id for m in lib.lookup(b, verify=True)]
        assert a == c


def test_monotone_shrinking_belief_only_adds_entries():
    rng = np.random.default_rng(2)
    lib = Library(GK, grow=dict(xy=0.12, p=0.2, mu=0.10))
    base = bel(people=(np.array([2.0, 7.5]),))
    for x in np.arange(0.9, 3.0, 0.2):
        b = base.with_(pose_wall=np.array([x, 5.0, 0.0]), pose_people=np.array([x, 5.0, 0.0]))
        lib.add(seg(b), b, GK)
    grew = 0
    for _ in range(200):
        b = _rand_belief(rng, True)
        s = rng.uniform(0.0, 1.0)
        b2 = b.with_(e_xy=b.e_xy * s, e_th=b.e_th * s, e_p=b.e_p * s, v_p_max=b.v_p_max * (0.5 + 0.5 * s), mu_lo=b.mu_lo + (1 - s) * 0.05)
        big = {m.entry.id for m in lib.lookup(b)}
        small = {m.entry.id for m in lib.lookup(b2)}
        assert big <= small
        grew += len(small) > len(big)
    assert grew > 0                                                                            # shrinking really does add entries sometimes


def test_param_map_velocity_and_people_conditions():
    b = bel(people=(np.array([2.0, 7.2]),))
    lib = Library(GK)
    e = lib.add(seg(b), b, GK)
    assert e is not None
    assert [m.entry.id for m in lib.lookup(b)] == [e.id]
    assert lib.lookup(b.with_(mu_lo=b.mu_lo - 0.02)) == []                      # mu set not inside the certified interval
    assert lib.lookup(b.with_(mu_hi=b.mu_hi + 0.05)) == []
    assert lib.lookup(b.with_(mu_lo=b.mu_lo + 0.02)) != []                      # learned (tighter) -> still applicable (T2b)
    assert lib.lookup(b.with_(grid=GRID2)) == []                                 # other map
    assert lib.lookup(b, v=np.array([0.1, 0, 0])) == []                         # not at rest
    assert lib.lookup(b.with_(v_p_max=2.0)) == []                               # person faster than certified
    assert lib.lookup(b.with_(people=(np.array([2.0, 7.1]),))) == []            # person moved: neither covered (e_p = 0) nor far
    assert lib.lookup(b.with_(people=())) != []                                 # fewer people = smaller set
    assert lib.lookup(b.with_(people=(np.array([2.0, 7.2]), np.array([9.0, 9.0])))) != []   # extra person far away
    assert lib.lookup(b.with_(people=(np.array([2.0, 7.2]), np.array([1.7, 5.4])))) == []   # extra person right next to the path
    # a velocity box gives a nonzero-velocity entry
    v0 = np.array([0.2, 0, 0])
    e2 = lib.add(plan_candidate(v0, np.array([0.2, 0, 0]), 0.3, b, GK), b, GK, v0=v0, v_tol=0.01)
    assert e2 is not None
    assert e2.id in [m.entry.id for m in lib.lookup(b, v=np.array([0.205, 0, 0]))]
    assert e2.id not in [m.entry.id for m in lib.lookup(b, v=np.array([0.25, 0, 0]))]


def test_pose_shift_and_state_set_radius():
    b = bel(pose=(1.5, 5.0, 0.0), e_xy=0.02, e_th=0.003)
    lib = Library(GK, grow=dict(xy=0.10))
    e = lib.add(seg(b), b, GK)
    r = e.X.e_xy
    assert r > 0.05
    near = b.with_(pose_wall=np.array([1.5 + 0.5 * (r - 0.02), 5.0, 0.0]), pose_people=np.array([1.5 + 0.5 * (r - 0.02), 5.0, 0.0]))
    assert lib.lookup(near)
    assert lib.lookup(b.with_(pose_wall=np.array([1.5 + 2 * r, 5.0, 0.0]), pose_people=np.array([1.5 + 2 * r, 5.0, 0.0]))) == []
    assert lib.lookup(b.with_(e_xy=r + 0.01)) == []                              # state set grew (T2c): same centre, no containment
    # heading offset counts through the lever arm of the path
    n_hit = 0
    for th in (0.0, 0.01, 0.02, 0.05, 0.1, 0.3):                                 # heading offsets: whatever is returned must really be admitted
        bt = b.with_(pose_wall=np.array([1.5, 5.0, th]), pose_people=np.array([1.5, 5.0, th]))
        for m in lib.lookup(bt):
            n_hit += 1
            assert admit(m.entry.traj(), bt)
    assert n_hit >= 1 and lib.lookup(b.with_(pose_wall=np.array([1.5, 5.0, 0.3]), pose_people=np.array([1.5, 5.0, 0.3]))) == []


# ------------------------------------------------------------------------------------------------------------------ ranking / dedupe / hook
def test_ranked_by_progress_and_dedupe_dominance():
    lib = Library(GK, progress_fn=RouteProgress([[0.0, 5.0], [4.0, 5.0]]).remaining)
    b = bel()
    slow = lib.add(seg(b, T_S=0.1), b, GK)
    fast = lib.add(seg(b, T_S=1.0), b, GK)
    assert fast.progress > slow.progress
    assert [m.entry.id for m in lib.lookup(b)] == [fast.id, slow.id]
    # identical segment again: dedupe returns the existing entry
    n = len(lib)
    again = lib.add(seg(b, T_S=1.0), b, GK)
    assert again.id == fast.id and len(lib) == n
    # a segment with a strictly larger certified region replaces the dominated one
    lib2 = Library(GK, progress_fn=None)
    small = lib2.add(seg(b), b, GK)
    big = lib2.add(seg(b), b.with_(e_xy=0.0, e_th=0.0), GK)                      # smaller belief -> X ⊆ existing? (dominated by `small`)
    assert big.id == small.id
    lib3 = Library(GK)
    first = lib3.add(seg(b), b.with_(e_xy=0.0, e_th=0.0), GK)
    second = lib3.add(seg(b), b, GK)                                              # bigger X -> the old entry is dropped
    assert len(lib3) == 1 and second.id != first.id and lib3.n_merged == 1


def test_hook_signature_adds_and_never_inserts_uncertified():
    lib = Library(GK)
    f = lib.hook()
    b = bel()
    tr = seg(b)
    e = f(tr, b, GK, t=1.2, step=12, T_S=0.6, dev=0.0)
    assert e is not None and e.prov["step"] == 12 and e.prov["t"] == 1.2
    bad = bel(pose=(4.8, 5.0, 0.0))
    assert f(seg(bad, T_S=1.0), bad, GK, t=2.0, step=13) is None and len(lib) == 1


# ------------------------------------------------------------------------------------------------------------------ invalidation
def test_invalidation_intersection_keeps_falsification_drops():
    lib = Library(GK, grow=dict(mu=0.12))
    b = bel()
    e = lib.add(seg(b), b, GK)
    lo, hi = e.X.mu_lo, e.X.mu_hi
    r = lib.learn(mu=(lo + 0.01, hi))                                              # learning: intersection, entries stay
    assert not r["falsified"] and len(lib) == 1
    r = lib.learn(mu=(lo + 0.02, hi - 0.02))
    assert not r["falsified"] and len(lib) == 1 and lib.lookup(b.with_(mu_lo=lo + 0.02, mu_hi=hi - 0.02))
    r = lib.learn(mu=(0.01, lo - 0.05))                                            # disjoint from the stored set: FALSIFIED -> entry certified for >= mu_lo is dropped
    assert r["falsified"] == ["mu"] and len(lib) == 0 and lib.invalidated
    lib = Library(GK)
    for x, vp in ((1.0, 1.0), (2.0, 1.5), (3.0, 2.0)):
        bb = bel(pose=(x, 5.0, 0.0), people=(np.array([x, 8.5]),), v_p_max=vp)
        assert lib.add(seg(bb), bb, GK) is not None
    assert len(lib) == 3
    assert lib.falsified("v_p", 1.8) == 2                                           # observed person speed 1.8 m/s: only the v_p_max=2.0 entry covers it
    assert [e.X.v_p_max for e in lib.entries.values()] == [2.0] and len(lib.invalidated) == 2
    lib2 = rest_lib()
    assert lib2.falsified("map", map_key(GRID)) == 0
    assert lib2.falsified("map", GRID2) == len(rest_lib())
    lib3 = rest_lib()
    assert lib3.monitor_failure(tags=("mu",)) == len(lib3.invalidated) > 0 and len(lib3) == 0


# ------------------------------------------------------------------------------------------------------------------ recover()
def test_recover_direct_move_look_none():
    lib = rest_lib(xs=(1.5, 2.5), grow=dict(xy=0.08))
    ex = lib.entries[0].X.e_xy
    assert ex > 0.05
    # direct
    b0 = bel(pose=(1.5, 5.0, 0.0), e_xy=0.02, e_th=0.003)
    rc = lib.recover(b0)
    assert rc.mode == "direct" and rc.entry is not None and rc.matches
    # move: 0.5 m off both boxes, small state set -> short certified connector into entry 0 or 1
    b1 = bel(pose=(1.95, 5.25, 0.15), e_xy=0.02, e_th=0.003)
    assert lib.lookup(b1) == []
    rc = lib.recover(b1)
    assert rc.mode == "move" and rc.legs and rc.t_connect > 0
    for leg in rc.legs:                                                           # the connector passes check1 AND check2 under its own belief
        assert check1(leg.traj, leg.bel, GK)[0] and check2(leg.traj, leg.bel, GK)[0], leg.margins
        assert leg.traj.i_rest is not None
    end = rc.legs[-1].pose_w1
    assert np.hypot(*(end[:2] - rc.entry.X.pose_w[:2])) < 5e-3 and abs(end[2] - rc.entry.X.pose_w[2]) < 5e-3
    arr = ctx_from_belief(b1, GK).at(pose_w=end, pose_p=rc.legs[-1].pose_p1, v=np.zeros(3), v_tol=0.0)
    assert contained(arr, rc.entry.X, rc.entry.P, extra_latency=rc.t_connect).ok
    # look: pose near an entry centre, state set too big for the box
    b2 = bel(pose=(1.5, 5.0, 0.0), e_xy=ex + 0.05, e_th=0.01)
    rc = lib.recover(b2)
    assert rc.mode == "look" and "e_xy" in rc.look["reduce"] and rc.look["reduce"]["e_xy"][1] <= ex + 1e-9
    assert rc.entry.id == rc.look["entry_id"]
    # shrinking the belief as the look suggests turns it into a direct hit
    b3 = b2.with_(e_xy=rc.look["reduce"]["e_xy"][1] * 0.999, e_th=0.003)
    assert lib.recover(b3).mode == "direct"
    # none: parameter set outside every entry (looking cannot fix it)
    rc = lib.recover(bel(pose=(1.5, 5.0, 0.0), mu_lo=0.05))
    assert rc.mode == "none"
    # none: the library is empty
    assert Library(GK).recover(b0).mode == "none"


def test_recover_priority_direct_beats_move_beats_look():
    lib = rest_lib(xs=(1.5, 2.5), grow=dict(xy=0.08))
    rc = lib.recover(bel(pose=(1.5, 5.0, 0.0), e_xy=0.02, e_th=0.003), allow=("direct", "move", "look"))
    assert rc.mode == "direct"
    assert lib.recover(bel(pose=(1.5, 5.0, 0.0), e_xy=0.02, e_th=0.003), allow=("move", "look")).mode == "move"       # already inside: zero-length connector or hop to a neighbour
    big = bel(pose=(1.5, 5.0, 0.0), e_xy=lib.entries[0].X.e_xy + 0.04, e_th=0.01)
    assert lib.recover(big, allow=("move", "look")).mode == "look"                                                     # move impossible (belief larger than every box)


def test_move_refuses_when_connector_not_certifiable():
    lib = rest_lib(xs=(1.5,), grow=dict(xy=0.05))
    b = bel(pose=(4.75, 5.0, 0.0), e_xy=0.02, e_th=0.003)                          # beside the wall; the straight leg to x=1.5 is long (3.2 m) and the belief starts very near the wall
    rc = lib.recover(b, max_connect=1.0)
    assert rc.mode in ("look", "none")


def test_people_aged_connector_needs_far_people():
    lib = rest_lib(xs=(1.5,), grow=dict(xy=0.08))
    b = bel(pose=(2.0, 5.2, 0.0), e_xy=0.02, e_th=0.003, people=(np.array([1.5, 6.0]),))
    # a person 1 m from the target that can walk 1.5 m/s: the entry holds no person context, the arrival is NOT certified -> no move
    assert lib.recover(b).mode in ("none", "look")
    assert lib.recover(b).mode != "move"
    far = bel(pose=(2.0, 5.2, 0.0), e_xy=0.02, e_th=0.003, people=(np.array([1.5, 7.8]),), v_p_max=0.4)      # slow person assumption, 2.8 m away
    rc = lib.recover(far)
    assert rc.mode == "move" and all(check1(l.traj, l.bel, GK)[0] and check2(l.traj, l.bel, GK)[0] for l in rc.legs)
    assert rc.legs[-1].bel.latency_s >= 0.0 and len(rc.legs) >= 1


def test_lookup_is_fast():
    lib = rest_lib(xs=np.arange(0.6, 3.5, 0.05), grow=dict(xy=0.05))
    b = bel(pose=(2.0, 5.0, 0.0))
    for _ in range(5):
        lib.lookup(b)
    assert len(lib) > 40 and lib.stats()["lookup_ms_p95"] < 50.0
