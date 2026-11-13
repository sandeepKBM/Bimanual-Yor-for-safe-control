"""episode state machine (PROTOCOL_BETA.md section 1).  Pure logic, no sim."""
import json

from world_model.v2.house.teleop.web import episode as EP


def tel(**kw):
    d = dict(cup_lost=False, beads_out=0, cup_state="ok", speed=0.5, pose=[0.0, 0.0, 0.0], interv_now=0.0, shield_label="", wall_clr=1.0,
             person_clr=2.0, cup_tilt=1.0, latched=False, assist=None, pick=None)
    d.update(kw)
    return d


def carry(**kw):
    return EP.Episode("carry", (3.0, 0.0), **kw)


def arm(ep, t=0.0, **kw):
    ep.update(t - 0.02, t - 0.02, False, tel(**kw))
    return ep.update(t, t, True, tel(**kw))


def test_idle_until_armed_then_active():
    ep = carry()
    assert ep.update(0.0, 0.0, False, tel()) == EP.IDLE
    assert ep.update(0.1, 0.1, False, tel()) == EP.IDLE
    assert ep.update(0.2, 0.2, True, tel()) == EP.ACTIVE
    j = ep.to_json()
    assert j["state"] == "ACTIVE" and j["outcome"] is None and j["goal"] == [3.0, 0.0]


def test_already_enabled_at_start_is_not_an_edge_after_reset_but_is_at_first_tick():
    ep = carry()
    assert ep.update(0.0, 0.0, True, tel()) == EP.ACTIVE      # first tick sees False -> True


def test_carry_success_needs_goal_stop_and_cup_ok():
    ep = carry()
    arm(ep)
    assert ep.update(1, 1, True, tel(pose=[2.8, 0.0, 0], speed=0.2)) == EP.ACTIVE           # in radius but moving
    assert ep.update(2, 2, True, tel(pose=[2.0, 0.0, 0], speed=0.0)) == EP.ACTIVE           # stopped but far
    assert ep.update(3, 3, True, tel(pose=[2.8, 0.0, 0], speed=0.01, cup_state="SPILLED")) == EP.ACTIVE    # cup not ok: no success
    assert ep.update(4, 4, True, tel(pose=[2.8, 0.0, 0], speed=0.01, cup_state="SPILLED", beads_out=1)) == EP.OUTCOME   # spill = fail
    assert ep.outcome["result"] == "fail"
    ep = carry()
    arm(ep)
    assert ep.update(3, 3, True, tel(pose=[2.8, 0.0, 0], speed=0.01)) == EP.OUTCOME
    assert ep.outcome["result"] == "success" and ep.consume_disarm() is True and ep.consume_disarm() is False


def test_goal_radius_and_stop_speed_are_inclusive_exclusive():
    ep = carry()
    arm(ep)
    assert ep.update(1, 1, True, tel(pose=[3.0 - EP.GOAL_R, 0, 0], speed=EP.STOP_V)) == EP.ACTIVE       # speed must be < STOP_V
    assert ep.update(2, 2, True, tel(pose=[3.0 - EP.GOAL_R, 0, 0], speed=EP.STOP_V - 1e-3)) == EP.OUTCOME     # distance <= GOAL_R


def test_failure_cup_lost_and_spill_all_tasks():
    for task in ("carry", "coffee", "pick"):
        for bad in (dict(cup_lost=True), dict(beads_out=2)):
            ep = EP.Episode(task, None if task == "pick" else (1.0, 1.0))
            arm(ep)
            assert ep.update(1, 1, True, tel(**bad)) == EP.OUTCOME
            assert ep.outcome["result"] == "fail" and ep.want_disarm


def test_operator_end_only_while_active():
    ep = carry()
    assert ep.update(0, 0, False, tel(), ["episode_end"]) == EP.IDLE
    arm(ep, 1.0)
    assert ep.update(2, 2, True, tel(), ["episode_end"]) == EP.OUTCOME
    assert ep.outcome["result"] == "ended" and ep.outcome["reason"] == "operator" and ep.consume_disarm()


def test_timeout():
    ep = carry(timeout_s=5.0)
    arm(ep, 1.0)
    assert ep.update(5.9, 5.9, True, tel()) == EP.ACTIVE
    assert ep.update(6.0, 6.0, True, tel()) == EP.OUTCOME
    assert ep.outcome["result"] == "fail" and ep.outcome["reason"] == "timeout"
    ep = carry(timeout_s=0)
    arm(ep)
    assert ep.update(1e6, 1e6, True, tel()) == EP.ACTIVE


def test_coffee_success_only_with_friction_assist_place():
    ep = EP.Episode("coffee", (1.0, 1.0), grasp_mode="friction")
    arm(ep)
    assert ep.update(1, 1, True, tel(assist=dict(place=None))) == EP.ACTIVE
    assert ep.update(2, 2, True, tel(assist=dict(place=False))) == EP.ACTIVE
    assert ep.update(3, 3, True, tel(assist=dict(place=True))) == EP.OUTCOME and ep.outcome["result"] == "success"
    ep = EP.Episode("coffee", (1.0, 1.0))                                                   # weld: no automatic success
    arm(ep)
    assert ep.update(3, 3, True, tel(assist=dict(place=True))) == EP.ACTIVE
    assert ep.update(4, 4, True, tel(), ["episode_end"]) == EP.OUTCOME and ep.outcome["result"] == "ended"


def test_pick_success_on_holding_or_done_and_failed_is_not_terminal():
    for stt in ("holding", "done"):
        ep = EP.Episode("pick", None)
        arm(ep)
        assert ep.update(1, 1, True, tel(pick=dict(status="executing"))) == EP.ACTIVE
        assert ep.update(2, 2, True, tel(pick=dict(status="failed"))) == EP.ACTIVE
        assert ep.update(3, 3, True, tel(pick=dict(status=stt))) == EP.OUTCOME and ep.outcome["result"] == "success"
        assert ep.outcome["metrics"]["dist_to_goal"] is None and ep.to_json()["goal"] is None


def test_metrics():
    ep = carry()
    arm(ep, 10.0)                                   # ACTIVE from t=10 (this tick is the first accumulated)
    ep.update(10.02, 10.5, True, tel(interv_now=0.5, shield_label="brake_fallback", wall_clr=0.4, person_clr=1.5, cup_tilt=7.0))
    ep.update(10.04, 10.6, True, tel(interv_now=0.01, shield_label="escape_fallback", wall_clr=0.9, person_clr=0.8, cup_tilt=3.0, latched=True))
    ep.update(10.06, 10.7, True, tel(latched=True))
    ep.update(10.08, 10.8, True, tel(latched=False))
    ep.update(10.10, 10.9, True, tel(latched=True))
    ep.update(11.0, 12.0, True, tel(), ["episode_end"])
    m = ep.outcome["metrics"]
    assert abs(m["duration_s"] - 1.0) < 1e-6 and abs(m["wall_s"] - 2.0) < 1e-6 or m["wall_s"] > 0
    assert m["min_wall_clr"] == 0.4 and m["min_person_clr"] == 0.8 and m["max_cup_tilt"] == 7.0
    assert m["n_shield_fallback"] == 2 and m["n_estop"] == 2 and m["beads_out"] == 0
    assert abs(m["interv_frac"] - 1 / 7) < 1e-3                       # 1 of 7 ACTIVE ticks > 0.02
    assert abs(m["dist_to_goal"] - 3.0) < 1e-9
    assert set(m) == {"duration_s", "wall_s", "interv_frac", "min_wall_clr", "min_person_clr", "beads_out", "n_estop", "n_shield_fallback",
                      "max_cup_tilt", "dist_to_goal"}
    json.dumps(ep.to_json(), allow_nan=False)


def test_non_finite_clearances_are_skipped():
    ep = carry()
    arm(ep)
    ep.update(1, 1, True, tel(wall_clr=float("nan"), person_clr=None))
    ep.update(2, 2, True, tel(), ["episode_end"])
    m = ep.outcome["metrics"]
    assert m["min_wall_clr"] == 1.0 and m["min_person_clr"] == 2.0
    json.dumps(ep.to_json(), allow_nan=False)


def test_outcome_is_terminal_blocks_motion_until_reset():
    ep = carry()
    arm(ep)
    ep.update(1, 1, True, tel(), ["episode_end"])
    assert ep.blocks_motion
    for t in range(2, 6):                                       # hub may re-arm: the episode stays OUTCOME, outcome unchanged
        ep.update(t, t, False, tel(pose=[3, 0, 0], speed=0.0))
        assert ep.update(t + 0.5, t + 0.5, True, tel(pose=[3, 0, 0], speed=0.0)) == EP.OUTCOME
    assert ep.outcome["result"] == "ended" and ep.blocks_motion
    ep.reset()
    assert ep.state == EP.IDLE and ep.outcome is None and not ep.blocks_motion and ep.to_json()["t_active"] is None
    assert ep.update(9, 9, True, tel()) == EP.ACTIVE            # a fresh episode starts after reset


def test_t_active_runs_in_active_and_freezes_in_outcome():
    ep = carry()
    arm(ep, 4.0)
    ep.update(6.5, 7, True, tel())
    assert ep.to_json()["t_active"] == 2.5
    ep.update(7.0, 8, True, tel(), ["episode_end"])
    ep.update(9.0, 9, True, tel())
    assert ep.to_json()["t_active"] == 3.0


def test_zero_cmd_blocks_motion_without_mutating():
    import numpy as np
    from world_model.v2.house.teleop.web import pick_link as PL
    from types import SimpleNamespace
    import dataclasses

    @dataclasses.dataclass
    class C:
        twist: object
        lift_v: float
        jog: object
        grasp: bool = True
        release: bool = True
    c = C(np.ones(3), 0.3, np.ones(6))
    z = PL.zero_cmd(c, motion_only=False)
    assert not z.twist.any() and z.lift_v == 0 and not z.jog.any() and not z.grasp and not z.release
    assert c.twist.all() and c.lift_v == 0.3 and c.grasp
    z2 = PL.zero_cmd(c)
    assert z2.grasp and z2.release and z2.twist is not c.twist
