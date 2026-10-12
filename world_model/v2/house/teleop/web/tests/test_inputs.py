"""input mapping / deadzone / watchdog / deadman unit tests (no sim, no GPU).  python -m pytest -q world_model/v2/house/teleop/web/tests"""
import numpy as np

from world_model.v2.house.teleop import base as BS
from world_model.v2.house.teleop.web import inputs as IN

TICK = 0.02


def kbd(hub, keys, t, focus=True, seq=0):
    hub.feed(dict(src="kbd", seq=seq, keys=list(keys), focus=focus), now=t)


def pad(hub, t, axes=(0, 0, 0, 0), btn=(), mapping="standard", vals=None, seq=0):
    b = [0.0] * 17
    for i in btn:
        b[i] = 1.0
    for i, v in (vals or {}).items():
        b[i] = v
    hub.feed(dict(src="pad", seq=seq, connected=True, mapping=mapping, axes=list(axes), buttons=b), now=t)


def run(hub, t0, dur, feed=None):
    """step the hub for dur seconds at 50 Hz, optionally re-feeding each tick; returns (last cmd, t_end)."""
    t, c = t0, None
    for _ in range(int(round(dur / TICK))):
        if feed:
            feed(t)
        c = hub.step(None, now=t, dt=TICK)
        t += TICK
    return c, t


def armed_hub():
    h = IN.InputHub()
    kbd(h, ["Enter"], 0.0)
    kbd(h, [], 0.01)
    assert h.step(None, now=0.02).twist.tolist() == [0, 0, 0] and h.enabled
    return h


def test_deadzone_rescale():
    assert IN.deadzone(0.04, 0.05) == 0.0 and IN.deadzone(-0.05, 0.05) == 0.0 + 0.0
    assert abs(IN.deadzone(1.0, 0.05) - 1.0) < 1e-12 and abs(IN.deadzone(-0.525, 0.05) + 0.5) < 1e-12


def test_disarmed_by_default_and_enter_arms():
    h = IN.InputHub()
    kbd(h, ["KeyW"], 0.0)
    c, _ = run(h, 0.0, 0.5, feed=lambda t: kbd(h, ["KeyW"], t))
    assert not h.enabled and not np.any(c.twist)
    kbd(h, ["KeyW", "Enter"], 0.6)
    c, _ = run(h, 0.6, 0.6, feed=lambda t: kbd(h, ["KeyW"], t))
    assert h.enabled and c.twist[0] > 0


def test_keyboard_signs_rep103_and_hold_semantics():
    h = armed_hub()
    t = 0.1
    for keys, idx, sign in ((["KeyW"], 0, 1), (["KeyS"], 0, -1), (["KeyA"], 1, 1), (["KeyD"], 1, -1), (["KeyQ"], 2, 1), (["KeyE"], 2, -1)):
        c, t = run(h, t, 0.8, feed=lambda tt, k=keys: kbd(h, k, tt))
        exp = np.zeros(3)
        exp[idx] = sign
        assert np.allclose(c.twist, exp * BS.V_STICK * 0.6, atol=1e-9), (keys, c.twist)
        kbd(h, [], t)
        c, t = run(h, t, 0.3, feed=lambda tt: kbd(h, [], tt))
        assert not np.any(c.twist)                                    # keyup -> zero (hold-to-move)
    kbd(h, ["KeyR"], t); c, t = run(h, t, 0.8, feed=lambda tt: kbd(h, ["KeyR"], tt))
    assert c.lift_v > 0
    kbd(h, ["ArrowUp", "ArrowLeft", "PageUp", "KeyT"], t)
    c, t = run(h, t, 0.8, feed=lambda tt: kbd(h, ["ArrowUp", "ArrowLeft", "PageUp", "KeyT"], tt))
    assert c.jog[0] > 0 and c.jog[1] > 0 and c.jog[2] > 0 and c.jog[5] > 0 and not np.any(c.twist)


def test_opposite_keys_cancel_and_ramp():
    h = armed_hub()
    c, _ = run(h, 0.1, 0.5, feed=lambda t: kbd(h, ["KeyW", "KeyS"], t))
    assert not np.any(c.twist)
    h = armed_hub()
    kbd(h, ["KeyW"], 0.1)
    c = h.step(None, now=0.1, dt=TICK)
    assert 0 < c.twist[0] < 0.2 * BS.V_STICK[0]                       # ramp: first tick is small


def test_watchdog_zeroes_and_auto_disarms():
    h = armed_hub()
    c, t = run(h, 0.1, 0.6, feed=lambda tt: kbd(h, ["KeyW"], tt))
    assert c.twist[0] > 0
    c, t = run(h, t, 0.2)                                             # silence < 250 ms: the last message is still fresh
    c, t = run(h, t, 0.1)                                             # now > 250 ms silent
    assert not np.any(c.twist) and h.watchdog_tripped and h.enabled
    c, t = run(h, t, 3.2)
    assert not h.enabled                                              # silent > auto_disable_s -> disarmed
    kbd(h, ["KeyW"], t)
    assert not np.any(h.step(None, now=t).twist)                      # stays zero until re-armed


def test_blur_and_space_and_esc():
    h = armed_hub()
    c, t = run(h, 0.1, 0.8, feed=lambda tt: kbd(h, ["KeyW"], tt))
    assert c.twist[0] > 0
    kbd(h, ["KeyW"], t, focus=False)                                  # tab blur -> zero immediately
    assert not np.any(h.step(None, now=t, dt=TICK).twist)
    kbd(h, ["KeyW"], t + 0.02)
    c, t = run(h, t + 0.02, 0.8, feed=lambda tt: kbd(h, ["KeyW"], tt))
    assert c.twist[0] > 0
    kbd(h, ["KeyW", "Space"], t)                                      # Space: zero, still armed, held W suppressed until released
    c, t = run(h, t, 0.5, feed=lambda tt: kbd(h, ["KeyW", "Space"], tt))
    assert h.enabled and not np.any(c.twist)
    kbd(h, [], t); kbd(h, ["KeyW"], t + 0.02)
    c, t = run(h, t + 0.02, 0.8, feed=lambda tt: kbd(h, ["KeyW"], tt))
    assert c.twist[0] > 0
    kbd(h, ["KeyW", "Escape"], t)                                     # Esc: latched stop
    c, t = run(h, t, 0.3, feed=lambda tt: kbd(h, ["KeyW"], tt))
    assert not h.enabled and h.latched and not np.any(c.twist)
    pad(h, t, btn=[9]); h.step(None, now=t)                           # pad Start does not clear the latch
    assert not h.enabled
    kbd(h, [], t + 0.05); kbd(h, ["Enter"], t + 0.06)                 # Enter clears and arms
    h.step(None, now=t + 0.06)
    assert h.enabled and not h.latched


def test_speed_presets_and_zx():
    h = armed_hub()
    kbd(h, ["Digit3"], 0.1); kbd(h, [], 0.11)
    assert h.scale == 1.0 and h.preset_name == "fast"
    kbd(h, ["KeyZ"], 0.12); kbd(h, [], 0.13)
    assert abs(h.scale - 1.0 / 1.1) < 1e-9 and h.preset_name == "custom"
    kbd(h, ["KeyX"], 0.14); kbd(h, [], 0.15); kbd(h, ["KeyX"], 0.16)
    assert h.scale == 1.0                                              # capped at full stick
    pad(h, 0.2, btn=[3]); pad(h, 0.21)
    assert h.preset == 0                                               # Y cycles 2 -> 0


def test_pad_deadman_and_signs():
    h = armed_hub()
    ax = (-0.5, 0.0, -0.9, -1.0)                                       # left-X left, right stick up-left, RY up
    c, t = run(h, 0.1, 0.8, feed=lambda tt: pad(h, tt, ax))            # LB NOT held -> zero
    assert not np.any(c.twist) and not np.any(c.jog)
    c, t = run(h, t, 0.8, feed=lambda tt: pad(h, tt, ax, btn=[4]))
    # REP-103: up = +vx, left = +vy, left X = +wz (CCW)
    assert c.twist[0] > 0 and c.twist[1] > 0 and c.twist[2] > 0
    pad(h, t, ax)                                                      # LB released -> immediate hard zero, no low-pass tail
    assert not np.any(h.step(None, now=t, dt=TICK).twist)


def test_pad_deadzone_dpad_lift_and_tray_mode():
    h = armed_hub()
    c, t = run(h, 0.1, 0.5, feed=lambda tt: pad(h, tt, (0.04, 0.03, -0.04, 0.04), btn=[4]))
    assert not np.any(c.twist)                                         # inside the 0.05 deadzone
    c, t = run(h, t, 0.8, feed=lambda tt: pad(h, tt, btn=[4, 12]))
    assert c.lift_v > 0
    c, t = run(h, t, 0.8, feed=lambda tt: pad(h, tt, (0, -1, 1, -1), btn=[4, 5], vals={7: 1.0}))
    assert not np.any(c.twist)                                         # tray mode: base is zero
    assert c.jog[0] > 0 and c.jog[2] > 0 and c.jog[5] < 0 and c.jog[3] > 0   # LY up = x+, RY up = z+, RX right = yaw-, RT = roll+
    c, t = run(h, t, 0.8, feed=lambda tt: pad(h, tt, btn=[4, 5], vals={6: 1.0}))
    assert c.jog[3] < 0


def test_pad_nonstandard_warns_and_disables_triggers():
    h = armed_hub()
    c, t = run(h, 0.1, 0.8, feed=lambda tt: pad(h, tt, btn=[4, 5], vals={7: 1.0}, mapping=""))
    assert h.by_name["pad"].warn and c.jog[3] == 0.0


def test_pad_buttons_events_and_disarm():
    h = armed_hub()
    pad(h, 0.1, btn=[0]); pad(h, 0.12); pad(h, 0.14, btn=[1])
    c = h.step(None, now=0.14)
    assert c.grasp and c.release                                       # A, B one-shots (armed)
    assert not h.step(None, now=0.16).grasp
    pad(h, 0.2, btn=[2]); pad(h, 0.22, btn=[11])
    assert h.pop_server_events() == ["shield_toggle", "camera_next"]
    pad(h, 0.3, btn=[8]); h.step(None, now=0.3)
    assert not h.enabled                                               # Back disarms
    pad(h, 0.4, btn=[9]); pad(h, 0.42); h.step(None, now=0.42)
    assert h.enabled                                                   # Start arms
    h.disconnect("pad")
    assert not h.enabled                                               # pad disconnect -> zero + disarm


def test_grasp_ignored_when_disarmed_and_arbitration():
    h = IN.InputHub()
    kbd(h, ["KeyG"], 0.0)
    assert not h.step(None, now=0.0).grasp
    h = armed_hub()
    t = 0.1
    c, t = run(h, t, 0.6, feed=lambda tt: kbd(h, ["KeyW"], tt))
    assert h.active == "kbd" and c.twist[0] > 0
    c, t = run(h, t, 0.8, feed=lambda tt: (kbd(h, ["KeyW"], tt), pad(h, tt, (0, 0, -1, 0), btn=[4])))
    assert h.active == "pad"                                           # newer activity wins; the held W is ignored meanwhile
    assert c.twist[0] == 0.0 and c.twist[1] > 0                        # pad RX left -> +vy; the held W (vx) is ignored
