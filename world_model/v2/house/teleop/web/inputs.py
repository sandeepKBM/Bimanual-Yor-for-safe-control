"""teleop/web/inputs.py -- InputSource abstraction for browser (and, later, ROS `joy`) teleop.  SIMULATION ONLY.

Every source exposes the same interface the other operators have:  step(obs) -> sim.Cmd.   Nothing here talks to a device or a robot;
browser messages (parsed JSON dicts) are the only input.  The pure mapping/safety logic has no I/O and takes an injectable clock `now`
(seconds, monotonic) so it is unit-testable.

Layers
  InputSource (base)   one physical device.  feed(msg, now) parses a device message into a held-state + one-shot events;
                       intent(now, dt) returns the watchdog-gated, deadzoned, ramped/low-passed Intent in [-1, 1].
    WebKeyboardSource  message {"src":"kbd","keys":[KeyboardEvent.code,...],"focus":bool}   HOLD-to-move
    WebGamepadSource   message {"src":"pad","connected":bool,"mapping":"standard","axes":[..],"buttons":[..]}
    (a ROS source would subclass InputSource: feed() takes a sensor_msgs/Joy-like dict {"axes":[..],"buttons":[..]} and reuses the
     gamepad parser with PAD_MAP_ROS -- see INTERFACE.md.)
  InputHub             owns the shared state (enabled, speed preset, active source, watchdog) and arbitrates between sources:
                       the source with the most recent *activity* (new non-zero input or a button edge) drives; the others are ignored.
                       hub.step(obs) -> Cmd,  hub.pop_server_events() -> events for the server (shield / camera / inset / reset).

All tunables live in CONFIG (one dict) and the two maps KBD_AXES/KBD_EVENTS and PAD_*; change them there only.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------------------------------------------------------ config
CONFIG = dict(
    watchdog_s=0.25,                 # a source with no message for this long contributes a ZERO command (moveit_servo 0.1 s < this < twist_mux 0.5 s)
    auto_disable_s=3.0,              # every source silent this long -> armed=False (operator must re-arm)
    presets=(0.30, 0.60, 1.00),      # speed presets (fraction of full stick); Y (pad) / keys 1-3
    preset_names=("slow", "medium", "fast"),
    preset_default=1,
    speed_step=1.1,                  # keyboard Z/X: scale x0.9 / x1.1 (teleop_twist_keyboard style)
    speed_min=0.05, speed_max=1.0,
    pad_deadzone=0.05,               # ros2 joy and YOR joystick.py both use 0.05; rescaled
    pad_lowpass_alpha=0.7,           # y += (1-alpha)*(x-y) per 20 ms tick  (YOR uses 0.9 at its own loop rate; the sim also slews)
    pad_trigger_threshold=0.5,
    kbd_ramp_up_per_s=4.0,           # keyboard stick 0 -> 1 in 0.25 s
    kbd_ramp_down_per_s=20.0,        # key release: 1 -> 0 in 50 ms
    jog_v_full=0.08,                 # tray jog full-scale [m/s]   (= ShieldCfg.jog_v_max)
    jog_w_full=0.15,                 # [rad/s]                     (= ShieldCfg.jog_w_max)
    lift_v_full=0.12,                # [m/s]                       (= sim.LIFT_V_MAX; shield clamps to 0.06)
)

# SIGN CONVENTION (ROS REP-103, what lands in sim.Cmd): +vx forward, +vy LEFT, +wz CCW, +z up.  Browser stick axes are +right/+down -> inverted below.
# keyboard: KeyboardEvent.code -> (channel, sign).  Channels: vx vy wz lift jx jy jz jroll jyaw  (all HELD: each key is its own deadman)
KBD_AXES = {
    "KeyW": ("vx", +1), "KeyS": ("vx", -1),
    "KeyA": ("vy", +1), "KeyD": ("vy", -1),
    "KeyQ": ("wz", +1), "KeyE": ("wz", -1),
    "KeyR": ("lift", +1), "KeyF": ("lift", -1),
    "ArrowUp": ("jx", +1), "ArrowDown": ("jx", -1), "ArrowLeft": ("jy", +1), "ArrowRight": ("jy", -1),
    "PageUp": ("jz", +1), "PageDown": ("jz", -1),
    "KeyT": ("jyaw", +1), "KeyY": ("jyaw", -1),
}
# keyboard: code -> one-shot event on key DOWN edge
KBD_EVENTS = {
    "KeyG": "grasp", "KeyH": "release", "Digit1": "preset:0", "Digit2": "preset:1", "Digit3": "preset:2",
    "KeyZ": "speed_down", "KeyX": "speed_up",
    "Space": "zero", "Escape": "estop", "Enter": "enable",
    "KeyV": "shield_toggle", "KeyC": "camera_next", "KeyB": "inset_toggle", "Backspace": "reset",
}
# gamepad ("standard" layout).  axes: 0 LX 1 LY 2 RX 3 RY (right/down = +).  buttons: 0 A 1 B 2 X 3 Y 4 LB 5 RB 6 LT 7 RT 8 View=Back 9 Start 10 LS 11 RS 12-15 dpad U/D/L/R
# NOTE button 8 is BOTH "View" and "Back" in the standard layout -> it is DISARM (YOR safety convention); camera cycle lives on RS click.
PAD_EVENTS = {0: "grasp", 1: "release", 2: "shield_toggle", 3: "preset_next", 9: "arm", 8: "disarm", 11: "camera_next", 10: "inset_toggle"}
PAD_DEADMAN_BUTTON = 4     # LB HELD = deadman (teleop_twist_joy enable_button); released -> one explicit zero, motion zero
PAD_TRAY_BUTTON = 5        # RB held (together with LB) = tray mode
PAD_LT, PAD_RT = 6, 7      # tray mode: roll -/+ (analog value 0..1).  Disabled when mapping != 'standard'
PAD_LIFT_UP, PAD_LIFT_DOWN = 12, 13
# per mode: output channel -> (axis index, sign).  sign -1 because stick up/left is negative and REP-103 wants forward/left positive.
PAD_BASE = {"vx": (3, -1), "vy": (2, -1), "wz": (0, -1)}                         # right stick = vx/vy (YOR joystick.py), left stick X = wz
PAD_TRAY = {"jx": (1, -1), "jy": (0, -1), "jz": (3, -1), "jyaw": (2, -1)}        # left stick tray x/y, right stick Y tray z, right stick X tray yaw
N_PAD_AXES, N_PAD_BUTTONS = 4, 17

CHANNELS = ("vx", "vy", "wz", "lift", "jx", "jy", "jz", "jroll", "jyaw")


# ------------------------------------------------------------------------------------------------------------------ helpers
def deadzone(x: float, dz: float) -> float:
    """dead-zone with rescale (YOR joystick.py): |x|<dz -> 0, else sign*(|x|-dz)/(1-dz)."""
    a = abs(x)
    if a < dz:
        return 0.0
    return float(np.sign(x) * min(1.0, (a - dz) / (1.0 - dz)))


def _slew(y: float, x: float, up: float, down: float, dt: float) -> float:
    """move y toward target x; |y| growing uses rate `up`, shrinking (or sign flip through 0) uses `down`."""
    if x == y:
        return y
    rate = up if abs(x) > abs(y) and (x * y >= 0) else down
    step = rate * dt
    return float(np.clip(x, y - step, y + step))


@dataclass
class Intent:
    """normalised operator intent in [-1, 1] per channel (vx, vy, wz, lift, jx, jy, jz, jyaw)."""
    ch: dict = field(default_factory=lambda: {c: 0.0 for c in CHANNELS})

    def nonzero(self) -> bool:
        return any(abs(v) > 1e-6 for v in self.ch.values())


# ------------------------------------------------------------------------------------------------------------------ sources
class InputSource:
    """base class for one device.  Subclasses implement _parse(msg, now) -> (target dict channel->[-1,1], held bool-ish, events list)."""
    name = "base"

    def __init__(self, cfg=None):
        self.cfg = dict(CONFIG, **(cfg or {}))
        self.lock = threading.Lock()
        self.last_msg_t = -1e9
        self.last_active_t = -1e9
        self.last_seq = -1
        self.target = {c: 0.0 for c in CHANNELS}
        self.out = {c: 0.0 for c in CHANNELS}
        self.events: list[str] = []
        self.connected = True
        self.warn = ""

    # ---- device message in
    def feed(self, msg: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self.lock:
            self.last_msg_t = now
            self.last_seq = int(msg.get("seq", self.last_seq))
            tgt, evs = self._parse(msg, now)
            prev = self.target
            self.target = tgt
            if evs or (tgt != prev and any(abs(v) > 1e-6 for v in tgt.values())):
                self.last_active_t = now
            if evs:
                self.events.extend(evs)

    def _parse(self, msg, now):  # pragma: no cover
        raise NotImplementedError

    def fresh(self, now: float) -> bool:
        return (now - self.last_msg_t) <= self.cfg["watchdog_s"] and self.connected

    def pop_events(self) -> list:
        with self.lock:
            ev, self.events = self.events, []
        return ev

    def zero(self) -> None:
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.target = {c: 0.0 for c in CHANNELS}
            self.out = {c: 0.0 for c in CHANNELS}
            self.events = []

    def _smooth(self, tgt: dict, dt: float) -> dict:
        raise NotImplementedError

    def intent(self, now: float, dt: float) -> Intent:
        """watchdog-gated, smoothed intent.  A stale source decays to zero immediately (no ramp: STOP is not smoothed)."""
        with self.lock:
            if not self.fresh(now):
                self.target = {c: 0.0 for c in CHANNELS}
                self.out = {c: 0.0 for c in CHANNELS}
            else:
                self.out = self._smooth(self.target, dt)
            return Intent(dict(self.out))


class WebKeyboardSource(InputSource):
    """HOLD-to-move keyboard.  The browser sends the set of currently held KeyboardEvent.code values; keydown sets, keyup clears.
    focus=false (tab blur / hidden) clears everything."""
    name = "kbd"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.held: set = set()
        self.suppress: set = set()          # motion keys still physically down after a Space "zero": ignored until released

    def zero(self):
        """Space: zero everything now; keys currently down stay ignored until they are released (no surprise resume)."""
        with self.lock:
            self.suppress = {k for k in self.held if k in KBD_AXES}
            self.target = {c: 0.0 for c in CHANNELS}
            self.out = {c: 0.0 for c in CHANNELS}

    def _parse(self, msg, now):
        if not msg.get("focus", True):
            self.held = set()
            self.suppress = set()
            return {c: 0.0 for c in CHANNELS}, []
        keys = {k for k in msg.get("keys", []) if isinstance(k, str)}
        evs = []
        for k in sorted(keys - self.held):                       # rising edges only
            e = KBD_EVENTS.get(k)
            if e:
                evs.append(e)
        self.held = keys
        self.suppress &= keys
        tgt = {c: 0.0 for c in CHANNELS}
        for k in keys - self.suppress:
            ax = KBD_AXES.get(k)
            if ax:
                tgt[ax[0]] += ax[1]
        return {c: float(np.clip(v, -1, 1)) for c, v in tgt.items()}, evs

    def _smooth(self, tgt, dt):
        c = self.cfg
        if not any(abs(v) > 1e-9 for v in tgt.values()):         # key-up to idle / blur / opposite keys: explicit hard zero, no ramp tail
            return {k: 0.0 for k in CHANNELS}
        return {k: _slew(self.out[k], tgt[k], c["kbd_ramp_up_per_s"], c["kbd_ramp_down_per_s"], dt) for k in CHANNELS}


class WebGamepadSource(InputSource):
    """Xbox pad via the browser Gamepad API ('standard' mapping).  LB HELD = deadman; deadzone 0.05 rescaled + low-pass (YOR joystick.py).
    Modes while LB is held: base (default) / tray (RB also held)."""
    name = "pad"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.prev_btn = np.zeros(N_PAD_BUTTONS, bool)
        self.connected = False
        self.mode = "idle"
        self.deadman = False
        self.hard_zero = True

    def _parse(self, msg, now):
        zero = {c: 0.0 for c in CHANNELS}
        if not msg.get("connected", True):
            self.connected = False
            self.prev_btn[:] = False
            self.deadman, self.hard_zero, self.mode = False, True, "idle"
            return zero, []
        std = msg.get("mapping", "standard") == "standard"
        self.warn = "" if std else f"pad mapping {msg.get('mapping')!r} is not 'standard': triggers disabled, check button layout"
        self.connected = True
        ax = list(msg.get("axes", []))[:N_PAD_AXES]
        ax += [0.0] * (N_PAD_AXES - len(ax))
        raw = list(msg.get("buttons", []))[:N_PAD_BUTTONS]
        raw += [0.0] * (N_PAD_BUTTONS - len(raw))
        ax = [float(a) if np.isfinite(a) else 0.0 for a in ax]
        bv = np.array([float(b) if np.isfinite(b) else 0.0 for b in raw])
        down = bv > self.cfg["pad_trigger_threshold"]
        evs = [PAD_EVENTS[i] for i in range(N_PAD_BUTTONS) if down[i] and not self.prev_btn[i] and i in PAD_EVENTS]
        self.prev_btn = down
        self.deadman = bool(down[PAD_DEADMAN_BUTTON])
        if not self.deadman:                                   # LB up: explicit zero, no smoothing
            self.hard_zero, self.mode = True, "idle"
            return zero, evs
        self.hard_zero = False
        dz = self.cfg["pad_deadzone"]
        tray = bool(down[PAD_TRAY_BUTTON])
        self.mode = "tray" if tray else "base"
        tgt = dict(zero)
        for ch, (i, sg) in (PAD_TRAY if tray else PAD_BASE).items():
            tgt[ch] = sg * deadzone(ax[i], dz)
        tgt["lift"] = float(down[PAD_LIFT_UP]) - float(down[PAD_LIFT_DOWN])
        if tray and std:
            tgt["jroll"] = deadzone(bv[PAD_RT], dz) - deadzone(bv[PAD_LT], dz)
        return tgt, evs

    def _smooth(self, tgt, dt):
        if self.hard_zero:
            return {c: 0.0 for c in CHANNELS}
        a = self.cfg["pad_lowpass_alpha"]
        al = a ** max(dt / 0.02, 1e-3)                          # filter is defined per 20 ms tick
        return {k: float(al * self.out[k] + (1 - al) * tgt[k]) if abs(tgt[k]) > 1e-9 or abs(self.out[k]) > 1e-3 else 0.0 for k in CHANNELS}


# ------------------------------------------------------------------------------------------------------------------ hub
class InputHub:
    """arbitrates sources, owns enable / preset / watchdog, builds the sim Cmd.  step(obs) -> Cmd (same interface as the operators)."""

    SERVER_EVENTS = ("shield_toggle", "camera_next", "inset_toggle", "reset")

    def __init__(self, sources=None, cfg=None):
        self.cfg = dict(CONFIG, **(cfg or {}))
        self.sources = sources or [WebKeyboardSource(self.cfg), WebGamepadSource(self.cfg)]
        self.by_name = {s.name: s for s in self.sources}
        self.lock = threading.Lock()
        self.enabled = False
        self.latched = False
        self.preset = int(self.cfg["preset_default"])
        self._scale = float(self.cfg["presets"][self.preset])
        self.active = "none"
        self.watchdog_tripped = False
        self.stopped_reason = "not enabled"
        self._pending_cmd = dict(grasp=False, release=False)
        self._server_events: list[str] = []
        self.last_step_t = None
        self.last_seq = {s.name: -1 for s in self.sources}
        self.phase = "WEB"
        self.done = False

    # ---- inbound
    def feed(self, msg: dict, now: float | None = None) -> None:
        src = self.by_name.get(msg.get("src"))
        if src is None:
            return
        src.feed(msg, now)
        self.last_seq[src.name] = src.last_seq
        self._drain_events(src)

    def _drain_events(self, src) -> None:
        with self.lock:
            for ev in src.pop_events():
                self._on_event(ev)

    def _on_event(self, ev: str) -> None:
        cf = self.cfg
        if ev == "enable":                                       # Enter: arm AND clear an Esc latch
            self.latched = False
            self.enabled = True
            self.stopped_reason = ""
        elif ev == "arm":                                        # pad Start: arm (does not clear an Esc latch)
            if not self.latched:
                self.enabled = True
                self.stopped_reason = ""
        elif ev == "disarm":                                     # pad Back
            self._stop("disarmed (Back)")
        elif ev == "estop":                                      # Esc: latched stop, only Enter clears
            self.latched = True
            self._stop("ESTOP latched (Esc) - press Enter to clear")
        elif ev == "zero":                                       # Space: zero everything, stay armed
            for s in self.sources:
                s.zero()
            self._pending_cmd = dict(grasp=False, release=False)
        elif ev == "preset_next":
            self.preset = (self.preset + 1) % len(cf["presets"])
            self._scale = float(cf["presets"][self.preset])
        elif ev.startswith("preset:"):
            self.preset = int(ev.split(":")[1]) % len(cf["presets"])
            self._scale = float(cf["presets"][self.preset])
        elif ev in ("speed_up", "speed_down"):
            k = cf["speed_step"] if ev == "speed_up" else 1.0 / cf["speed_step"]
            self._scale = float(np.clip(self._scale * k, cf["speed_min"], cf["speed_max"]))
        elif ev in ("grasp", "release"):
            if self.enabled:
                self._pending_cmd[ev] = True
        elif ev in self.SERVER_EVENTS:
            self._server_events.append(ev)

    def _stop(self, why: str) -> None:
        self.enabled = False
        self.stopped_reason = why
        self._pending_cmd = dict(grasp=False, release=False)
        for s in self.sources:
            s.zero()

    def disconnect(self, name: str | None = None) -> None:
        """gamepad unplugged / websocket closed: zero + disable."""
        with self.lock:
            for s in self.sources:
                if name in (None, s.name):
                    s.reset()
                    s.last_msg_t = -1e9
                    if s.name == "pad":
                        s.connected = False
            if name in (None, self.active) or name == "pad":
                self.enabled = False
                self.stopped_reason = f"{name or 'client'} disconnected"

    def pop_server_events(self) -> list:
        with self.lock:
            ev, self._server_events = self._server_events, []
        return ev

    # ---- outbound
    @property
    def scale(self) -> float:
        return self._scale

    @property
    def preset_name(self) -> str:
        return self.cfg["preset_names"][self.preset] if abs(self._scale - self.cfg["presets"][self.preset]) < 1e-9 else "custom"

    def step(self, obs=None, now: float | None = None, dt: float | None = None):
        from ..sim import Cmd, TICK
        now = time.monotonic() if now is None else now
        dt = TICK if dt is None else dt
        for s in self.sources:                                   # events from a source that is now silent are still delivered
            self._drain_events(s)
        ints = {s.name: s.intent(now, dt) for s in self.sources}
        fresh = {s.name: s.fresh(now) for s in self.sources}
        with self.lock:
            cand = [s for s in self.sources if fresh[s.name]]
            self.active = max(cand, key=lambda s: s.last_active_t).name if cand else "none"
            # auto-disable if every source has been silent for too long
            newest = max((s.last_msg_t for s in self.sources), default=-1e9)
            self.watchdog_tripped = (now - newest) > self.cfg["watchdog_s"]
            if self.enabled and (now - newest) > self.cfg["auto_disable_s"]:
                self.enabled = False
                self.stopped_reason = "watchdog: no input"
            if not self.enabled or self.active == "none":
                it = Intent()
            else:
                it = ints[self.active]
            grasp, release = self._pending_cmd["grasp"], self._pending_cmd["release"]
            self._pending_cmd = dict(grasp=False, release=False)
            sc = self.scale
        return self.to_cmd(it, sc, grasp and self.enabled, release and self.enabled)

    def to_cmd(self, it: Intent, scale: float, grasp=False, release=False):
        from .. import base as BS
        from ..sim import Cmd
        c, cf = it.ch, self.cfg
        twist = np.array([c["vx"], c["vy"], c["wz"]]) * BS.V_STICK * scale
        jog = np.zeros(6)
        jog[:3] = np.array([c["jx"], c["jy"], c["jz"]]) * cf["jog_v_full"] * scale
        jog[3] = c["jroll"] * cf["jog_w_full"] * scale
        jog[5] = c["jyaw"] * cf["jog_w_full"] * scale
        return Cmd(twist=twist, lift_v=float(c["lift"] * cf["lift_v_full"] * scale), jog=jog, grasp=bool(grasp), release=bool(release))

    def status(self, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        return dict(enabled=self.enabled, latched=self.latched, src=self.active, preset=self.preset, preset_name=self.preset_name, scale=self.scale,
                    watchdog=self.watchdog_tripped, stopped=self.stopped_reason,
                    pad=dict(connected=self.by_name["pad"].connected, mode=getattr(self.by_name["pad"], "mode", ""), deadman=getattr(self.by_name["pad"], "deadman", False),
                             warn=self.by_name["pad"].warn) if "pad" in self.by_name else None,
                    ack=dict(self.last_seq))
