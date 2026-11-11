"""from_carry on a real (short) box episode agrees with the episode's own in_R / track."""
from world_model.v2.adapters import contents_specs as CS
from world_model.v2.adapters.state_carry import from_carry
from world_model.v2.nero.contents import carry as CR
from world_model.v2.stack import goal as GL
from world_model.v2.stack import tasks as TK
from world_model.v2.stack.checks import monitors as MON


def test_from_carry_box_hold():
    w = CS.build_world("box")
    ep = CR.CarryEpisode(w, planner="scripted")
    ep.run_phases([("hold", 0.4)])
    s = from_carry(ep)
    t = TK.task("box")
    okR, whyR = ep.in_R()
    r = MON.recoverable(s, t, include_U=True)
    assert r.ok == okR, (r.why, whyR)
    assert s.now["obj_tilt_deg"] == CR._tilt_deg(ep.d.xmat[ep.obj].reshape(3, 3))
    assert (ep.track["viol"] is None) == MON.u_row(s, t).ok
    g = GL.check_goal(GL.goal_for("box", w), s)           # still on the counter: the goal is not reached
    assert not g.ok and "xy_err" in g.worst
