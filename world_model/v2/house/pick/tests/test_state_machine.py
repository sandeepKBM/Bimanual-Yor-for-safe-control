"""PickSkill state machine against a MOCKED sim (no MuJoCo model, no GPU, no controller)."""
import math
import types

import numpy as np

from world_model.v2.house.pick import skill as SK, props as PP

TICK = SK.TICK


class FakeModel:
    def __init__(self):
        self.names = {}

    def _id(self, n):
        return self.names.setdefault(n, 20 + len(self.names))

    def joint(self, n):
        return types.SimpleNamespace(id=self._id("j" + n))

    def actuator(self, n):
        return types.SimpleNamespace(id=self._id("a" + n))

    @property
    def jnt_qposadr(self):
        return _Ident()


class _Ident:
    def __getitem__(self, i):
        return i


class FakeCtl:
    lq, rq = np.arange(0, 6), np.arange(6, 12)
    ld, rd = lq, rq

    def __init__(self):
        self.ref = [np.zeros(6), np.zeros(6)]

    def set_ref(self, qL, qR):
        self.ref = [qL.copy(), qR.copy()]


class FakeHS:
    """perfect-tracking Cartesian servo; `block` (arm -> plane value along the approach axis) makes a hand stop at contact like a real object would."""
    block = {}
    start = [(0.5, 0.1, 0.89), (0.5, -0.5, 0.89)]       # pre-grasp poses (single: 9 cm above the grasp pose)

    def __init__(self, sim):
        self.sim = sim
        self.tgt, self.goal = [None, None], [None, None]
        self.active = False
        self.ki = 1.2
        self.v_max = 0.05
        self.pos = [np.array(p, float) for p in FakeHS.start]

    def hand_pose(self, k):
        return self.pos[k].copy(), np.eye(3)

    def start_from_current(self):
        for k in (0, 1):
            self.tgt[k] = (self.pos[k].copy(), np.eye(3))
            self.goal[k] = (self.pos[k].copy(), np.eye(3))
        self.active = True

    def set_goal(self, k, p, R, jump=False):
        self.goal[k] = (np.array(p, float), R)
        if self.tgt[k] is None or jump:
            self.tgt[k] = (np.array(p, float), R)

    def step(self, dt, slow=1.0):
        for k in (0, 1):
            if self.goal[k] is None or self.tgt[k] is None:
                continue
            tp, R = self.tgt[k]
            gp, _ = self.goal[k]
            e = gp - tp
            n = np.linalg.norm(e)
            lim = self.v_max * dt
            tp = tp + (e if n <= lim else e * lim / n)
            self.tgt[k] = (tp, R)
            new = tp.copy()
            if k in FakeHS.block:
                ax, val, sgn = FakeHS.block[k]
                if sgn * (new[ax] - val) > 0:
                    new[ax] = val
            self.pos[k] = new

    def at_goal(self, k, tol_p=0.003, tol_r=0.02):
        return np.linalg.norm(self.goal[k][0] - self.pos[k]) < tol_p


class FakeSim:
    def __init__(self, obj_width=0.05):
        self.m = FakeModel()
        self.d = types.SimpleNamespace(qpos=np.zeros(64), ctrl=np.zeros(64), ncon=0, qvel=np.zeros(64))
        self.ctl = FakeCtl()
        self.of = types.SimpleNamespace(qL=np.zeros(6), qR=np.zeros(6))
        self.t = 0.0
        self.obj_width = obj_width
        self.house_geom, self.robot_geom = np.zeros(1, bool), np.zeros(1, bool)

    def phys(self):
        """one tick of 'physics': arm joints jump to the reference, fingers follow ctrl until they meet the object."""
        self.d.qpos[self.ctl.lq], self.d.qpos[self.ctl.rq] = self.ctl.ref
        for s in ("left", "right"):
            m = self.m
            a7, a8 = m.actuator(f"{s}_arm_joint7_pos").id, m.actuator(f"{s}_arm_joint8_pos").id
            j7, j8 = m.joint(f"{s}_arm_joint7").id, m.joint(f"{s}_arm_joint8").id
            q = max(self.d.ctrl[a7], self.obj_width / 2)           # stalls on the object (width 0 -> closes fully)
            self.d.qpos[j7], self.d.qpos[j8] = q, -q
        self.t += TICK


def make_wp(mode, arms):
    wp = SK.Waypoints(mode, arms)
    for s in arms:
        wp.q_pre[s] = np.full(6, 0.3)
        wp.q_grasp[s] = np.full(6, 0.35)
        wp.q_lift[s] = np.full(6, 0.4)
        wp.goal_pre[s] = (np.array([0.5, 0.1 if s == "left" else -0.5, 0.89]) if mode == "single" else np.array([0.5, 0.2 if s == "left" else -0.6, 0.8]), np.eye(3))
        wp.goal_grasp[s] = (np.array([0.5, 0.1 if s == "left" else -0.5, 0.8]), np.eye(3))
        wp.goal_lift[s] = (np.array([0.5, 0.1 if s == "left" else -0.5, 0.9]), np.eye(3))
        wp.app_dir[s] = np.array([0.0, -1.0 if s == "left" else 1.0, 0.0]) if mode == "bimanual" else np.array([0, 0, -1.0])
    wp.width = 0.05
    return wp


def run(sim, sk, n=3000, until=("HOLD", "FAILED", "DONE_RELEASED"), cancel_at=None):
    states = [sk.state]
    for i in range(n):
        if cancel_at is not None and i == cancel_at:
            sk.cancel()
        sk.step(sim)
        sim.phys()
        # the mocked hand follows the references of the mocked servo; nothing else to do
        if sk.state != states[-1]:
            states.append(sk.state)
        if sk.state in until and (sk.state != "HOLD" or sk.done):
            break
    return states


def test_single_arm_sequence(monkeypatch):
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    sim = FakeSim(obj_width=0.05)
    sk = SK.PickSkill(sim, make_wp("single", ("left",)))
    sim.d.qpos[sim.ctl.lq] = 0.3                         # the arm 'arrives' at the pre-grasp pose
    states = run(sim, sk)
    assert states == ["TRANSIT", "APPROACH", "CLOSE", "LIFT", "HOLD"], states
    assert sk.done and sk.fail is None
    assert 0.045 < sk.gap_closed < 0.056                  # stalled on the 5 cm object, not fully closed
    assert sk.state == "HOLD"


def test_single_arm_closes_on_nothing_fails(monkeypatch):
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    sim = FakeSim(obj_width=0.0)
    sk = SK.PickSkill(sim, make_wp("single", ("right",)))
    states = run(sim, sk)
    assert states[-1] == "FAILED" and sk.fail[0] == "NO_GRASP", (states, sk.fail)


def test_single_arm_slip_detected(monkeypatch):
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    sim = FakeSim(obj_width=0.05)
    sk = SK.PickSkill(sim, make_wp("single", ("left",)))
    for i in range(3000):
        sk.step(sim)
        sim.phys()
        if sk.state == "LIFT":
            sim.obj_width = 0.0                           # the object slips out: jaws close
        if sk.state in ("FAILED", "HOLD"):
            break
    assert sk.state == "FAILED" and sk.fail[0] == "SLIP"


def test_bimanual_stall_contact_squeeze_lift(monkeypatch):
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    # the object blocks the hands 1.5 cm short of the squeeze goal (they push, stall, and the skill declares contact from proprioception only)
    FakeHS.block = {0: (1, 0.115, -1), 1: (1, -0.515, 1)}
    FakeHS.start = [(0.5, 0.2, 0.8), (0.5, -0.6, 0.8)]
    try:
        sim = FakeSim()
        sk = SK.PickSkill(sim, make_wp("bimanual", SK.SIDES))
        states = run(sim, sk)
        assert states[:5] == ["TRANSIT", "APPROACH", "SQUEEZE", "LIFT", "HOLD"], states
        assert "contact_left_t" in sk.info and "contact_right_t" in sk.info
        assert 10 < sk.info["contact_left_residual_mm"] < 20
        # LIFT completes on z alone (the blocked hands never reach the commanded squeeze depth in y)
        assert sk.done
    finally:
        FakeHS.block = {}
        FakeHS.start = [(0.5, 0.1, 0.89), (0.5, -0.5, 0.89)]


def test_bimanual_blocked_far_from_goal_fails_as_no_grasp(monkeypatch):
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    FakeHS.block = {0: (1, 0.17, -1), 1: (1, -0.57, 1)}          # blocked 7 cm short of the goal: not a contact, an obstruction
    FakeHS.start = [(0.5, 0.2, 0.8), (0.5, -0.6, 0.8)]
    old = SK.T_MAX["APPROACH"]
    SK.T_MAX["APPROACH"] = 6.0
    try:
        sim = FakeSim()
        sk = SK.PickSkill(sim, make_wp("bimanual", SK.SIDES))
        states = run(sim, sk, n=1500)
        assert states[-1] == "FAILED" and sk.fail[0] == "NO_GRASP", (states, sk.fail)
    finally:
        SK.T_MAX["APPROACH"] = old
        FakeHS.block = {}
        FakeHS.start = [(0.5, 0.1, 0.89), (0.5, -0.5, 0.89)]


def test_cancel_releases_and_retreats(monkeypatch):
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    sim = FakeSim(obj_width=0.05)
    sk = SK.PickSkill(sim, make_wp("single", ("left",)))
    sim.d.qpos[sim.ctl.lq] = 0.3
    states = run(sim, sk, until=("DONE_RELEASED", "FAILED"), cancel_at=900)
    assert states[-3:] == ["RELEASE", "RETREAT", "DONE_RELEASED"], states
    assert np.allclose(sk.qref["left"], PP.READY_Q["left"], atol=1e-6)
    a7 = sim.m.actuator("left_arm_joint7_pos").id
    assert sim.d.ctrl[a7] > 0.03                          # fingers re-opened


def test_step_is_cheap_and_never_renders(monkeypatch):
    import time
    monkeypatch.setattr(SK, "HandServo", FakeHS)
    sim = FakeSim()
    sk = SK.PickSkill(sim, make_wp("single", ("left",)))
    t = time.time()
    for _ in range(500):
        sk.step(sim)
        sim.phys()
    assert (time.time() - t) / 500 < 2e-3                  # well inside the 20 ms tick (the real cost is HandServo.step, measured in the smoke)
