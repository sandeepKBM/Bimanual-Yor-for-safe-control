"""tests for ocs/anchored.py (general SE(3)-anchored tracker).  SIMULATION ONLY; the recording tests skip when the OCS recordings are absent."""
from pathlib import Path

import numpy as np
import pytest

from world_model.v2.house.ocs import anchored as AN, producers as P, util as U
from world_model.v2.house.ocs.state import Kind, Source, Belief, WorldState, RobotState
from world_model.v2.tray_rep.sets.ir import Pose

REC = Path("/data/local/ss5772/molmospaces/ocs/rec")
CAL = Path("/data/local/ss5772/molmospaces/ocs/v2_cupfix/calib_h5.json")


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_world_zonotope_is_exact_image():
    rng = np.random.default_rng(0)
    m = AN.ObjectModel("cup", Kind.CONTENT, U.cup_shape("sim"))
    for _ in range(20):
        R, p, n = _rot(rng), rng.normal(size=3), rng.normal(size=3)
        Zw = m.world_zonotope(R, p)
        assert np.isclose(Zw.support(n), m.shape.support(R.T @ n) + n @ p)


def test_from_points_contains_onboarding_cloud():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 400)
    pts = np.c_[0.03 * np.cos(a), 0.03 * np.sin(a), rng.uniform(0, 0.08, 400)]
    m = AN.ObjectModel.from_points("obj", Kind.MOVABLE, pts)
    assert all(U.in_zonotope(m.shape.c, m.shape.G, x, tol=1e-6) for x in pts)


def _ctx(a, R=np.eye(3)):
    return AN.SupportCtx(R, R, np.asarray(a, float), 0.02)


def test_slide_still_level_support_does_not_move():
    prop = AN.SupportSlideProp()
    m = AN.ObjectModel("cup", Kind.CONTENT, U.cup_shape("sim"), parent="tray", params={"mu": Belief(0.18, 0.35)})
    H = prop.init(AN.Measurement(0.0, np.array([0.01, -0.02, 0.0]), 0.01), m)
    for _ in range(100):
        H = prop.step(H, _ctx([0.0, 0.0, 0.0]))
    assert np.allclose(H.P, [0.01, -0.02, 0.0]) and np.allclose(H.V, 0.0)


def test_slide_strong_brake_lo_slides_farther_than_hi():
    prop = AN.SupportSlideProp(a_dist=0.0)
    m = AN.ObjectModel("cup", Kind.CONTENT, U.cup_shape("sim"), parent="tray", params={"mu": Belief(0.18, 0.35)})
    H = prop.init(AN.Measurement(0.0, np.zeros(3), 0.01), m)
    for _ in range(10):                                   # 0.2 s of 5 m/s^2 support acceleration along +x (above mu_hi * g = 3.4)
        H = prop.step(H, _ctx([5.0, 0.0, 0.0]))
    lo = H.P[H.mu == 0.18][0, 0]
    hi = H.P[H.mu == 0.35][0, 0]
    mid = H.P[0, 0]
    assert lo < mid < hi < 0.0                            # the body lags the support (slides toward -x), more for lower friction


def test_condition_reanchors_and_resets_inconsistent_velocities():
    m = AN.ObjectModel("cup", Kind.CONTENT, U.cup_shape("sim"), parent="tray", params={"mu": Belief(0.18, 0.35)})
    pr = AN.AnchoredProducer(m, [], AN.SupportSlideProp(), k_fals=2.0)
    pr.H = AN.SupportSlideProp().init(AN.Measurement(0.0, np.zeros(3), 0.01), m)
    pr.H.V[:] = 0.3
    pr.H.V[0] = 0.0
    pr.H.P[1:, 0] = 0.05                                  # every non-nominal hypothesis predicted 5 cm away
    pr._condition(AN.Measurement(0.1, np.zeros(3), 0.01))  # measured at the nominal: 5 cm > 2 * 1 cm
    assert np.allclose(pr.H.P, 0.0) and np.allclose(pr.H.V, 0.0) and pr.n_reset == len(pr.H.P) - 1


def test_default_pipeline_is_unchanged():
    ps = P.perception_pipeline(people=False, names=False).producers
    assert any(isinstance(p, P.CupWristProducer) for p in ps) and not any(isinstance(p, AN.AnchoredProducer) for p in ps)


needs_rec = pytest.mark.skipif(not ((REC / "carry_s4.pkl.gz").exists() and CAL.exists()), reason="OCS recordings / calibration not present")


@needs_rec
def test_blob_source_matches_cupwrist_while_visible():
    from world_model.v2.house.ocs.views import Recording
    rec = Recording.load(REC / "carry_s4.pkl.gz")
    cal = P.Calib.load(CAL)
    ck = U.cup_kind_for_scene(rec.meta["scene"])
    old = P.Pipeline([P.BaseProducer(cal), P.TrayFKProducer(cal), P.CupWristProducer(cal, cup=ck)])
    an = AN.cup_on_tray(cal, ck)
    an.prop, an.k_fals = AN.StaticProp(), None
    new = P.Pipeline([P.BaseProducer(cal), P.TrayFKProducer(cal), an])
    old.reset(); new.reset()
    n_cmp = 0
    for k in range(240):
        a, b = old.step(rec.view(k)), new.step(rec.view(k))
        assert ("cup" in a.objects) == ("cup" in b.objects)
        if "cup" in a.objects:
            oa, ob = a.objects["cup"], b.objects["cup"]
            assert np.allclose(oa.pose.p, ob.pose.p) and oa.visible == ob.visible and oa.t_meas == ob.t_meas
            if oa.visible:
                assert np.allclose(oa.unc.pos.G, ob.unc.pos.G) and oa.unc.rot == ob.unc.rot
                n_cmp += 1
    assert n_cmp > 100


@needs_rec
def test_seed4_hidden_slide_is_covered():
    """held-out seed 4: the cup slides out of the wrist view during a hard brake (t ~ 4.5 s) and stays hidden; the frozen estimate is 3.1 cm off."""
    from world_model.v2.house.ocs.views import Recording
    rec = Recording.load(REC / "carry_s4.pkl.gz")
    cal = P.Calib.load(CAL)
    pipe = P.perception_pipeline(cal, people=False, names=False, cup_kind=U.cup_kind_for_scene(rec.meta["scene"]), cup_tracker="anchored")
    pipe.reset()
    A = rec.A
    errs, cov = [], []
    for k in range(rec.n):
        ws = pipe.step(rec.view(k))
        o = ws.objects.get("cup")
        if o is None or o.visible or A["t"][k] < 5.0:
            continue
        tR, tp = A["tray_R"][k].reshape(3, 3), A["tray_p"][k]
        pt = tR.T @ (A["cup_p"][k] - tp)
        half = np.diag(o.unc.pos.G)[:2]
        d = np.abs(pt[:2] - np.asarray(o.pose.p)[:2])
        errs.append(float(np.linalg.norm(pt[:2] - np.asarray(o.pose.p)[:2])))
        cov.append(bool(np.all(d <= half + 1e-9)))
    assert len(errs) > 1000
    assert np.mean(cov) == 1.0 and np.median(errs) < 0.015


def test_recorded_fp_source_reads_and_fuses(tmp_path):
    import json
    from world_model.v2.house.ocs.fp_source import RecordedFPSource
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p = np.full((2, 2, 3), np.nan)
    R = np.full((2, 2, 3, 3), np.nan)
    p[0, 0], p[0, 1], p[1, 1] = [0.01, 0.02, 0.004], [0.03, 0.00, 0.004], [-0.02, 0.01, 0.004]
    R[0, 0], R[0, 1], R[1, 1] = np.eye(3), Rz, np.eye(3)
    f = tmp_path / "fp_s9.npz"
    np.savez(f, ticks=np.array([10, 20]), ok=np.array([[True, True], [False, True]]), p_tray=p, R_tray=R, mask_px=np.array([[100, 400], [0, 50]]),
             mode=np.array([[2, 2], [0, 1]], np.int8), ms=np.zeros((2, 2)), meta=json.dumps({"W": 320}))
    src = RecordedFPSource(f, None)
    m = AN.ObjectModel("cup", Kind.CONTENT, U.cup_shape("sim"), parent="tray", on_support=True)
    view = lambda k: type("V", (), {"k": k})()
    a = src.measure(view(10), None, m, 0.2)
    assert np.allclose(a.p, [0.02, 0.01, 0.0]) and np.allclose(a.R, Rz) and a.r == src.pos_prior and a.extra["fp_sides"] == ["left", "right"]
    b = src.measure(view(20), None, m, 0.4)
    assert np.allclose(b.p, [-0.02, 0.01, 0.0]) and b.extra["fp_sides"] == ["right"]
    assert src.measure(view(15), None, m, 0.3) is None
