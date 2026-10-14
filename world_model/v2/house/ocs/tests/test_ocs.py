"""ocs tests: schema, world_pose chain, privileged audit, consumer equivalence on a synthetic state, rollout monotonicity.  CPU, no sim."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from world_model.v2.tray_rep.sets import ir as IR, tray_spec as TS, reach as RE
from world_model.v2.tray_rep.sets.ir import Pose, Unc, BoxShape, ZonoShape, Point
from world_model.v2.tray_rep.sets.zono import Zonotope

from world_model.v2.house.ocs import (Source, Kind, Belief, ObjectState, RobotState, WorldState)
from world_model.v2.house.ocs import util as U, consumers as C, dynamics as D, producers as P

EYE = np.eye(3)


def _tray_ws(yaw=0.4, base=(1.0, 2.0)):
    """a hand-built state: robot at (base, yaw), tray in the robot frame, cup in the tray frame, person + wall obstacle in the world."""
    rs = RobotState(base=U.base_pose(base[0], base[1], yaw), base_unc=Unc(U.ball_zono(0.02), 0.01), base_vel=np.array([0.3, 0.0, 0.1]), yaw=yaw)
    ws = WorldState(t=1.0, robot=rs)
    ws.objects["tray"] = ObjectState("tray", Kind.PAYLOAD, BoxShape([U.HX, U.HY, 0.003]), Pose(EYE.copy(), np.array([0.45, -0.21, 0.85])),
                                     unc=Unc(U.ball_zono(0.005), 0.01), parent="robot", dynamics="attached", source=Source.FK)
    ws.objects["cup"] = ObjectState("cup", Kind.CONTENT, ZonoShape(U.cup_shape()), Pose(EYE.copy(), np.array([-0.045, 0.0, 0.0])),
                                    unc=Unc(Zonotope(np.zeros(3), np.diag([0.02, 0.02, 0.003])), 0.05), parent="tray", dynamics="slip_on_support",
                                    source=Source.WRIST, params={"mu": Belief(0.18, 0.35, source=Source.PRIOR)})
    ws.objects["trk0"] = ObjectState("trk0", Kind.PERSON, ZonoShape(U.person_shape()), Pose(EYE.copy(), np.array([3.0, 2.0, 0.0])),
                                     unc=Unc(U.disc_zono(0.1), 0.0), vel=np.array([-0.4, 0.0, 0, 0, 0, 0]), dynamics="const_vel", source=Source.TRACKER,
                                     extra={"sigma_v": 0.2})
    ws.objects["wall"] = ObjectState("wall", Kind.OBSTACLE, ZonoShape(Zonotope(np.zeros(3), np.diag([0.5, 0.05, 1.0]))), Pose(EYE.copy(), np.array([2.0, 4.0, 1.0])),
                                     dynamics="static", source=Source.STEREO)
    return ws


# ------------------------------------------------------------------------------------------------ schema / frames / audit
def test_schema_defaults():
    o = ObjectState("a", Kind.UNKNOWN, Point(), Pose(EYE, np.zeros(3)))
    assert o.unc.pos is None and o.unc.rot == 0.0 and o.visible and o.parent is None and o.dynamics == "static" and o.name_source is None
    ws = WorldState(t=0.0, robot=RobotState(base=U.base_pose(0, 0, 0)))
    assert ws.by_name("a") is None
    assert ws.privileged() == ["robot.base", "robot.q"] or set(ws.privileged()) <= {"robot.base", "robot.q"}      # default RobotState sources are TRUTH / ENCODER


def test_world_pose_chain():
    ws = _tray_ws(yaw=0.4, base=(1.0, 2.0))
    Pt = ws.world_pose("tray")
    Rb = U.rz(0.4)
    assert np.allclose(Pt.p, Rb @ np.array([0.45, -0.21, 0.85]) + np.array([1.0, 2.0, 0.0]))
    Pc = ws.world_pose("cup")
    assert np.allclose(Pc.p, Pt.R @ np.array([-0.045, 0.0, 0.0]) + Pt.p)
    assert np.allclose(Pc.R, Rb)


def test_world_unc_composes_and_grows_with_chain():
    ws = _tray_ws()
    ut, uc = C.world_unc(ws, "tray"), C.world_unc(ws, "cup")
    d = np.array([1.0, 0.0, 0.0])
    # parent's set contributes: cup set >= tray set >= base set along every sampled direction; rotation bounds add
    for a in np.linspace(0, 2 * math.pi, 9):
        d = np.array([math.cos(a), math.sin(a), 0.0])
        assert U.unc_support(uc, d) >= U.unc_support(ut, d) - 1e-12 >= U.unc_support(ws.robot.base_unc, d) - 1e-12
    assert math.isclose(uc.rot, 0.01 + 0.05, rel_tol=1e-12)          # tray tilt + cup tilt; the planar base's yaw bound (0.01) enters positions only


def test_privileged_audit():
    ws = _tray_ws()
    ws.robot.base_source = Source.ODOM
    assert ws.privileged() == []
    ws.assert_honest()
    ws.objects["trk0"].name, ws.objects["trk0"].name_source = "person", Source.TRUTH
    ws.objects["cup"].params["mu"] = Belief(0.25, 0.25, 0.25, source=Source.TRUTH)
    ws.objects["tray"].source = Source.TRUTH
    pv = ws.privileged()
    assert set(pv) == {"trk0.name", "cup.params.mu", "tray.pose"}
    try:
        ws.assert_honest()
        raise SystemExit("assert_honest should have raised")
    except AssertionError:
        pass
    ws.assert_honest(allow=("trk0", "cup", "tray"))


# ------------------------------------------------------------------------------------------------ synthetic truth view -> producers -> consumers
def _fake_truth(occ=None):
    yaw = 0.7
    Rt = U.rz(yaw + 0.01)
    q = U.R_to_quat(Rt)
    return SimpleNamespace(t=1.5, base=np.array([1.0, 2.0, yaw]), base_v=np.array([0.2, 0.1, 0.05]), lift=0.0, qL=np.zeros(6), qR=np.zeros(6), qdL=np.zeros(6),
                           qdR=np.zeros(6), tray_p=np.array([1.3, 2.1, 0.85]), tray_R=U.quat_to_R(q), tray_q=q, cup_p=np.array([1.28, 2.1, 0.8501]),
                           cup_R=EYE.copy(), cup_q=np.array([1.0, 0, 0, 0]), mu_true=0.25,
                           people=[("p_walker", np.array([3.2, 3.8]), np.array([-0.3, 0.0])), ("p_stand", np.array([1.1, 3.5]), np.zeros(2))],
                           obstacles=[("box_a", Zonotope(np.array([2.0, 2.0, 0.5]), np.diag([0.1, 0.2, 0.3])))], occ=occ)


def test_consumer_equivalence_truth():
    tr = _fake_truth()
    view = SimpleNamespace(t=tr.t, k=75, truth=tr)
    ws = P.Pipeline([P.TruthProducer(chain=False)]).step(view)
    es = C.to_env_state(ws)
    ref = TS.EnvState(tr.tray_p, tr.tray_q, tr.cup_p)
    assert np.array_equal(es.p_t, ref.p_t) and np.array_equal(es.R_t, ref.R_t) and np.array_equal(es.p_b, ref.p_b)
    ps = C.to_people_state(ws)
    assert len(ps) == 2
    for (xy, v), (_, xy0, v0) in zip(ps, tr.people):
        assert np.array_equal(xy, xy0) and np.array_equal(v, v0)
    assert [i["name"] for i in C.person_info(ws)] == ["p_walker", "p_stand"]
    # chain=True reproduces the same world poses to rounding
    ws2 = P.Pipeline([P.TruthProducer(chain=True)]).step(view)
    for oid in ("tray", "cup"):
        assert np.allclose(ws2.world_pose(oid).p, ws.world_pose(oid).p, atol=1e-12)
        assert np.allclose(ws2.world_pose(oid).R, ws.world_pose(oid).R, atol=1e-12)
    zs = dict(C.to_obstacle_zonos(ws, inflate=False))
    assert np.allclose(zs["box_a"].c, [2.0, 2.0, 0.5])
    assert "tray.pose" in ws.privileged() and "robot.base" in ws.privileged() and "p_walker.pose" in ws.privileged()


def test_occ_roundtrip():
    from scipy import ndimage
    from world_model.v2.house.teleop.scene import OccGrid
    occ = np.zeros((40, 50), bool)
    occ[10:15, 20:30] = True
    g = OccGrid(0.05, -1.0, -2.0, occ, ndimage.distance_transform_edt(~occ) * 0.05)
    ws = P.Pipeline([P.TruthProducer()]).step(SimpleNamespace(t=0.0, k=0, truth=_fake_truth(g)))
    g2 = C.to_occ(ws)
    assert np.array_equal(g2.occ, g.occ) and np.array_equal(g2.edt, g.edt) and (g2.res, g2.x0, g2.y0) == (g.res, g.x0, g.y0)
    assert C.to_occ(ws) is g2                       # cached


def test_obstacle_zono_roundtrip_stereo_result():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(3, 5))
    c = np.array([3.0, 5.0, 0.8])
    delta = 0.021
    Zi = Zonotope(c, np.concatenate([G, np.eye(3) * delta], 1))
    res = dict(t_sim=0.0, zonos=[dict(id=7, kind="obj", gt_key="monitor_1", n_pts=100, delta=delta, range_m=1.2, Z=dict(c=c, G=G), Zi=dict(c=c, G=Zi.G))])
    view = SimpleNamespace(t=0.5, k=25, zono_obs=lambda: res, truth=_fake_truth())
    ws = P.Pipeline([P.TruthProducer(), P.ObstacleZonoProducer()]).step(view)
    ws.objects.pop("box_a")
    out = dict(C.to_obstacle_zonos(ws, inflate=True))
    assert np.array_equal(out["z7"].c, Zi.c) and np.array_equal(out["z7"].G, Zi.G)
    NamesProducer = P.NamesProducer()
    NamesProducer.update(ws, view, 0.5)
    assert ws.objects["z7"].name == "monitor_1" and ws.objects["z7"].name_source == Source.TRUTH and "z7.name" in ws.privileged()


# ------------------------------------------------------------------------------------------------ IR entities
def test_ir_spec_on_ocs():
    ws = _tray_ws()
    ents = C.to_ir_entities(ws)
    ws.objects["wall"].name = "laptop"
    ents = C.to_ir_entities(ws)
    spec = TS.build_spec({"and": [{"inside": {"a": "cup", "b": "tray_top", "margin": 0.02}}, {"upright": {"a": "cup", "theta_max": 0.3}},
                                  {"clear": {"a": "cup", "b": "laptop", "d": 0.1}}, {"clear": {"a": "cup", "b": "trk0", "d": 0.5}}]},
                         ents, H=0, theta_max=0.3, lo=0)
    st = IR.status(spec, ws)
    assert st in ("true", "unknown", "false")
    # the cup centre is well inside the tray; the person is > 1 m away; the wall is > 1 m away -> nominal margin positive
    assert IR.margin(spec, ws, False) > 0
    # moving the person next to the cup makes it false
    ws.objects["trk0"].pose = Pose(EYE.copy(), np.asarray(ws.world_pose("cup").p) + np.array([0.2, 0.0, 0.0]))
    assert IR.status(spec, ws) == "false"


def test_ir_time_axis_rollout():
    ws = _tray_ws()
    ws.objects["trk0"].pose = Pose(EYE.copy(), np.array([3.0, 2.0, 0.0]))
    traj = D.rollout(ws, 2.0, 0.1, robot_plan=D.hold_plan([0.3, 0.0, 0.0], 20))
    ents = C.to_ir_entities(traj)
    spec = TS.build_spec({"G": {"lo": 0, "hi": 20, "of": {"clear": {"a": "cup", "b": "trk0", "d": 0.5}}}}, ents, H=20, lo=0)
    v = IR.margin(spec, traj, False)
    assert np.isfinite(v)


# ------------------------------------------------------------------------------------------------ dynamics
def _radii(traj, oid):
    return np.array([D.set_radius(w, oid) for w in traj])


def test_rollout_monotone_and_motion():
    ws = _tray_ws()
    traj = D.rollout(ws, 2.0, 0.1, robot_plan=None)
    assert len(traj) == 21 and math.isclose(traj[-1].t - ws.t, 2.0)
    for oid in ("tray", "cup", "trk0", "wall"):
        r = _radii(traj, oid)
        assert np.all(np.diff(r) >= -1e-12), (oid, r)
    assert D.set_radius(traj[-1], "trk0") > D.set_radius(traj[0], "trk0") + 0.1
    assert D.set_radius(traj[-1], "wall") == D.set_radius(traj[0], "wall") == 0.0           # static: no growth
    # const_vel person: p(2 s) = p0 + 2 v
    assert np.allclose(traj[-1].world_pose("trk0").p, np.array([3.0 - 0.8, 2.0, 0.0]))
    # base follows its twist (const twist -> arc), tray and cup ride along (attached chain)
    d0 = np.linalg.norm(traj[-1].robot.base.p - traj[0].robot.base.p)
    assert d0 > 0.3
    rel0 = np.linalg.inv(np.asarray(traj[0].robot.base.R)) @ (traj[0].world_pose("tray").p - traj[0].robot.base.p)
    rel1 = np.linalg.inv(np.asarray(traj[-1].robot.base.R)) @ (traj[-1].world_pose("tray").p - traj[-1].robot.base.p)
    assert np.allclose(rel0, rel1)


def test_rollout_occluded_object_keeps_growing():
    ws = _tray_ws()
    o = ws.objects["trk0"]
    o.visible, o.t_meas = False, ws.t - 1.0
    r1 = _radii(D.rollout(ws, 1.0, 0.1), "trk0")
    o.t_meas = ws.t
    r2 = _radii(D.rollout(ws, 1.0, 0.1), "trk0")
    assert np.all(np.diff(r1) >= -1e-12) and r1[-1] - r1[0] >= r2[-1] - r2[0] - 1e-12      # older measurement => faster-growing increments (convex h)


def test_slip_set_wider_under_hard_braking_and_matches_reach():
    ws = _tray_ws(yaw=0.0)
    ws.robot.base_vel = np.array([0.9, 0.0, 0.0])
    # brake hard: twist drops to 0 within one step -> |a| = 9 m/s^2 >> mu g -> the cup slides forward in the tray frame
    plan = D.RobotPlan(np.zeros((10, 3)))
    traj = D.rollout(ws, 1.0, 0.1, robot_plan=plan)
    p0, p1 = traj[0].objects["cup"].pose.p, traj[-1].objects["cup"].pose.p
    assert p1[0] > p0[0] + 0.05, (p0, p1)
    # gentle plan: no slide
    gentle = D.RobotPlan(np.tile([0.9, 0.0, 0.0], (10, 1)))
    traj2 = D.rollout(ws, 1.0, 0.1, robot_plan=gentle)
    assert np.allclose(traj2[-1].objects["cup"].pose.p, traj2[0].objects["cup"].pose.p, atol=1e-9)
    # level tray, yaw 0: the in-plane frame of ball_step equals the tray frame, so one 20 ms slide step equals reach.ball_step directly
    c = np.array([0.0, 0.0, 1.0])
    s0 = np.array([0.0, 0.0, 0.0, 0.0])
    a = np.array([-9.0, 0.0, 0.0])
    s_ref, _, _ = RE.ball_step(s0, c, c, a, 0.25, 0.0, 1.0)
    P_, V_ = D._slide_traj(np.zeros(2), np.zeros(2), [EYE, EYE], [a], 0.25, 0.0, 1.0, 1)
    assert np.allclose(P_[1], s_ref[:2]) and np.allclose(V_[1], s_ref[2:])


def test_robot_frame_and_robot_parented_person():
    ws = _tray_ws(yaw=0.4, base=(1.0, 2.0))
    # a person tracked relative to the robot (parent "robot"): world pose composes through the base
    prel = np.array([1.2, -0.3, 0.0])
    ws.objects["trk0"] = ObjectState("trk0", Kind.PERSON, ZonoShape(U.person_shape()), Pose(U.rz(0.4).T.copy(), prel), unc=Unc(U.disc_zono(0.1), 0.0),
                                     vel=np.array([-0.4, 0, 0, 0, 0, 0]), parent="robot", dynamics="const_vel", source=Source.TRACKER)
    pw = ws.world_pose("trk0").p
    assert np.allclose(pw, U.rz(0.4) @ prel + np.array([1.0, 2.0, 0.0]))
    rf = C.robot_frame(ws)
    assert np.allclose(rf.world_pose("trk0").p, prel) and C.world_unc(rf, "tray").pos is not None
    # robot-frame set has NO base term for robot-measured objects, world set does
    d = np.array([1.0, 0.0, 0.0])
    assert U.unc_support(C.world_unc(rf, "trk0"), d) < U.unc_support(C.world_unc(ws, "trk0"), d)
    # a map object (parent None) inherits the base set in the robot frame
    assert U.unc_support(C.world_unc(rf, "wall"), d) > 0.0
    # the person keeps its WORLD track while the robot drives: world position moves by v*t only
    traj = D.rollout(ws, 1.0, 0.1, robot_plan=D.hold_plan([0.5, 0.0, 0.0], 10))
    assert np.allclose(traj[-1].world_pose("trk0").p, pw + np.array([-0.4, 0.0, 0.0]), atol=1e-9)
    ents = C.to_ir_entities(ws, frame="robot")
    assert "cup_rel" in ents


def test_cup_dims_match_sim_geometry():
    """the h5 (teleop-sim) cup shape comes from the compiled MJCF cup, not a constant: measure the cup geoms of the model independently and compare."""
    import mujoco
    from world_model.v2.data import contents as DC
    r, h = U.cup_dims("sim")
    assert abs(r - 0.0325) < 5e-4 and abs(h - DC.CUP_HEIGHT) < 1e-6          # 12-gon walls: apothem 0.030 + 1.5 mm, circumscribed corner 0.0325; rim at 0.080
    # independent check: simulator geometry via a body-frame AABB of the compiled geoms
    cup = DC._extract_body(DC.SCENE.read_text(), "cup").replace('material="cup_mat"', 'rgba="0.8 0.1 0.1 1"')
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{cup}</worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    top = max(d.geom_xpos[g][2] - d.xpos[m.body("cup").id][2] + (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][1]) for g in range(m.ngeom))
    assert abs(top - h) < 1e-6
    wall0 = m.geom("cup_wall_0")
    assert abs(r - math.hypot(wall0.pos[0] + wall0.size[0], wall0.size[1])) < 1e-6          # outermost wall corner
    # the flagship paper cup is a different (larger) object, read from the flagship perception module
    from world_model.v2.house.flagship import perception as FP
    assert U.cup_dims("flagship") == (FP.CUP_R, FP.CUP_H) and U.cup_dims("flagship")[0] > r and U.cup_dims("flagship")[1] > h
    assert U.cup_kind_for_scene("h5") == "sim"
    # the zonotope follows: the sim cup set is strictly smaller
    zs, zf = U.cup_shape("sim"), U.cup_shape("flagship")
    assert zs.G.max() < zf.G.max() and abs(zs.c[2] - h / 2) < 1e-9


def test_cup_kind_plumbed_through_producers():
    """TruthProducer / perception_pipeline put the requested cup shape on the cup object (default stays the flagship cup)."""
    pr = P.CupWristProducer(None, cup="sim")
    assert abs(pr.plane_z - U.cup_dims("sim")[1] / 2) < 1e-12
    assert abs(P.CupWristProducer(None).plane_z - U.cup_dims("flagship")[1] / 2) < 1e-12
    assert P.TruthProducer(cup="sim").cup == "sim" and P.TruthProducer().cup == "flagship"
