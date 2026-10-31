"""tuck / carry presets (keys N / M, pad D-pad), server flag defaults.  No sim, no GPU.  python -m pytest -q world_model/v2/house/teleop/web/tests"""
import numpy as np

from world_model.v2.house.teleop.web import inputs as IN
from world_model.v2.house.teleop.web.tests.test_inputs import kbd, pad, run, armed_hub, TICK


def obs_at(dp, yaw):
    return dict(tray_offset=(np.asarray(dp, float), float(yaw)))


def test_keys_mapped():
    assert IN.KBD_EVENTS["KeyN"] == "tuck" and IN.KBD_EVENTS["KeyM"] == "carry"
    assert IN.PAD_EVENTS[14] == "tuck" and IN.PAD_EVENTS[15] == "carry"
    old = {"KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE", "KeyR", "KeyF", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "KeyT", "KeyY"}
    assert old <= set(IN.KBD_AXES) and not ({"KeyN", "KeyM"} & set(IN.KBD_AXES))        # no clash with an axis key
    assert len({*IN.KBD_EVENTS}) == len(IN.KBD_EVENTS)


def test_tuck_drives_jog_to_target_and_stops():
    h = armed_hub()
    kbd(h, ["KeyN"], 0.1)
    kbd(h, [], 0.11)
    dp, yaw = np.zeros(3), 0.0
    t = 0.12
    cmd = h.step(obs_at(dp, yaw), now=t, dt=TICK)
    assert h.tuck_goal == "tuck" and cmd.jog[0] > 0 and cmd.jog[1] > 0 and cmd.jog[2] == 0 and cmd.jog[5] > 0
    full = IN.CONFIG["jog_v_full"] * h.scale
    for _ in range(2000):
        kbd(h, [], t)                                              # heartbeat keeps the source fresh
        cmd = h.step(obs_at(dp, yaw), now=t, dt=TICK)
        assert np.linalg.norm(cmd.jog[:3]) <= full + 1e-12 and abs(cmd.jog[5]) <= IN.CONFIG["jog_w_full"] * h.scale + 1e-12
        dp = dp + cmd.jog[:3] * TICK
        yaw += cmd.jog[5] * TICK
        t += TICK
        if h.tuck_goal is None:
            break
    assert h.tuck_goal is None and h.tuck_state() == "tucked"
    assert np.allclose(dp, IN.CONFIG["tuck_dp"], atol=0.01) and abs(yaw - IN.CONFIG["tuck_yaw"]) < 0.02
    assert not np.any(h.step(obs_at(dp, yaw), now=t, dt=TICK).jog)                 # done: no more jog
    kbd(h, ["KeyM"], t); kbd(h, [], t + 0.01)
    assert h.tuck_goal == "carry"
    cmd = h.step(obs_at(dp, yaw), now=t + 0.02, dt=TICK)
    assert cmd.jog[0] < 0 and cmd.jog[1] < 0 and cmd.jog[5] < 0                    # back toward the normal carry pose


def test_tuck_cancel_and_gates():
    h = IN.InputHub()                                                              # disarmed: the key does nothing
    kbd(h, ["KeyN"], 0.0)
    assert h.tuck_goal is None
    h = armed_hub()
    kbd(h, ["KeyN"], 0.1); kbd(h, [], 0.11)
    assert h.tuck_goal == "tuck"
    kbd(h, ["ArrowUp"], 0.12)                                                      # manual tray jog overrides (cancels) the preset
    cmd = h.step(obs_at(np.zeros(3), 0.0), now=0.13, dt=TICK)
    assert h.tuck_goal is None and cmd.jog[0] > 0
    kbd(h, [], 0.14); kbd(h, ["KeyN"], 0.15); kbd(h, [], 0.16)
    assert h.tuck_goal == "tuck"
    kbd(h, ["Space"], 0.17)                                                        # Space: zero everything
    assert h.tuck_goal is None
    kbd(h, [], 0.18); kbd(h, ["KeyN"], 0.19); kbd(h, [], 0.2)
    kbd(h, ["Escape"], 0.21)                                                       # Esc: latched stop
    assert h.tuck_goal is None and not h.enabled
    h2 = armed_hub()
    kbd(h2, ["KeyN"], 0.1); kbd(h2, [], 0.11)
    h2.disconnect(None)
    assert h2.tuck_goal is None
    assert armed_hub().step(None, now=0.5).jog.tolist() == [0.0] * 6               # no obs["tray_offset"]: the preset machinery is inert (old callers unchanged)


def test_pad_tuck_needs_deadman():
    h = IN.InputHub()
    pad(h, 0.0, btn=[9]); pad(h, 0.01)
    h.step(None, now=0.02)
    assert h.enabled
    pad(h, 0.1, btn=[4, 14]); pad(h, 0.11, btn=[4])                                # LB + D-pad left
    assert h.tuck_goal == "tuck" and h.tuck_src == "pad"
    c = h.step(obs_at(np.zeros(3), 0.0), now=0.12, dt=TICK)
    assert c.jog[0] > 0
    pad(h, 0.13, btn=[])                                                           # LB released: the tuck pauses (zero jog), goal kept
    c = h.step(obs_at(np.zeros(3), 0.0), now=0.14, dt=TICK)
    assert not np.any(c.jog) and h.tuck_goal == "tuck"


def test_server_defaults():
    from world_model.v2.house.teleop.web import server as S
    a = S.parse_args([])
    assert a.view == "ego" and a.people == "on" and a.footprint == "fixed" and a.d_wall_door is None and a.fps == 15.0
    assert S.CAMERAS[0] == "ego" and S.CAMERAS.index("chase") == 1
    b = S.parse_args(["--view", "chase", "--people", "static", "--footprint", "live", "--d-wall-door", "0.05"])
    assert (b.view, b.people, b.footprint, b.d_wall_door) == ("chase", "static", "live", 0.05)
