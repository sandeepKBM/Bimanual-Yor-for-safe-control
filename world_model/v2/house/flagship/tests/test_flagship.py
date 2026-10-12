"""Headless tests for the flagship pipeline (no GPU, no display, no robot).  python -m pytest -q world_model/v2/house/flagship/tests

  test_state_machine_scripted      CoffeeRunOperator drives a kinematic ghost robot through every mission phase in order (stub perception).
  test_operator_deterministic      same seed -> identical command stream; different seed differs.
  test_replay_determinism          two short real sim runs (same seed) are bit-identical, and replaying the saved qpos/mocap reproduces the recorded tray pose
                                   (so the full render can replay without re-simulating).
  test_schedule_compress_and_freeze  render.build_schedule: time-lapse only where nothing happens, a 2 s freeze per first event, one perception window each.
"""
import math
import os
import tempfile

import numpy as np
import pytest

from world_model.v2.house.flagship import scene as FS, mission as MS
from world_model.v2.house.teleop import base as BS
from world_model.v2.house.teleop.sim import TICK


class StubServices:
    name = "stub"

    def sense_counter(self, obs):
        return dict(kind="counter")

    def sense_desk(self, obs):
        return dict(ok=True, spot_t=np.array([0.0, 0.05]), path_t=np.stack([np.linspace(0, 0, 8), np.linspace(0, 0.05, 8)], 1))


class GhostWorld:
    """kinematic stand-in for TeleopSim: base integrates the twist, lift integrates lift_v, grasp errors derive from the lift height."""

    def __init__(self, scene):
        wp = scene.waypoints
        s = wp["lab_start"]
        self.b = BS.BaseSim(s[0], s[1], s[2])
        self.lift = 0.0
        self.grasped = False
        self.t = 0.0
        self.dock = np.array(wp["cafe_counter_pick"])

    def obs(self):
        pose = np.array([self.b.x, self.b.y, self.b.yaw])
        near = np.linalg.norm(pose[:2] - self.dock[:2]) < 0.03 and abs(BS.wrap(pose[2] - self.dock[2])) < 0.03
        ez = 0.0 if self.grasped else FS.H_PICK - self.lift
        can = bool(near and abs(ez) < 0.01 and not self.grasped)
        return dict(t=self.t, pose=pose, v=self.b.v.copy(), lift_h=self.lift, grasped=self.grasped, grasp_err_p=np.array([0.0, 0.0, ez]),
                    grasp_err_r=np.zeros(3), can_grasp=can, tray_pos=np.array([pose[0], pose[1] + 0.45, 0.85 + self.lift]), bump_new=0)

    def step(self, cmd):
        if cmd.grasp:
            self.grasped = True
        self.b.advance(cmd.twist, TICK, BS.A_HW)
        self.lift = float(np.clip(self.lift + cmd.lift_v * TICK, 0.0, 0.4))
        self.t += TICK


def test_state_machine_scripted():
    sc = FS.make_scene(cafe_walker=False)
    w = GhostWorld(sc)
    op = MS.CoffeeRunOperator(1, sc, StubServices())
    seen = ["OUT"]
    for k in range(int(400 / TICK)):
        cmd = op.step(w.obs())
        w.step(cmd)
        if op.phase != seen[-1]:
            seen.append(op.phase)
        if op.done:
            break
    assert seen == [p for p in MS.PHASES if p != "ABORT"], seen        # ABORT is the raw side's crash branch
    assert op.place_info["planned"] is True
    # the lift reached counter height before the grasp and the carry height is back at 0 when driving home
    ev = dict((p, t) for t, p in op.events)
    assert ev["RAISE"] < ev["CREEP"] < ev["ALIGN"] < ev["LIFT"] < ev["BACKOUT"] < ev["LOWER_C"] < ev["BACK"] < ev["SENSE_D"] < ev["PLACE"] < ev["RELEASE"]


def test_operator_deterministic():
    sc = FS.make_scene(cafe_walker=False)

    def stream(seed):
        w = GhostWorld(sc)
        op = MS.CoffeeRunOperator(seed, sc, MS.NullServices())
        out = []
        for _ in range(600):
            c = op.step(w.obs())
            w.step(c)
            out.append(c.twist.copy())
        return np.array(out)

    a, b, c = stream(3), stream(3), stream(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_replay_determinism():
    import mujoco
    from world_model.v2.house.flagship import simulate as SM
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        r1 = SM.run_side(True, 0, max_s=2.0, perception="off", out=td, tag="a", start="counter", verbose=False, tail_s=0.0)
        r2 = SM.run_side(True, 0, max_s=2.0, perception="off", out=td, tag="b", start="counter", verbose=False, tail_s=0.0)
        za, zb = (dict(np.load(os.path.join(td, f"run_{k}.npz"))) for k in "ab")      # dict(): loads + closes the files
        assert np.array_equal(za["qpos"], zb["qpos"]) and np.array_equal(za["mocap_pos"], zb["mocap_pos"])
        # replay: set qpos / mocap from the file, forward, compare the recorded tray world position
        sim = FS.new_sim(0, cafe_walker=False, spawn=None)
        m, d = sim.m, sim.d
        for i in (0, 40, 99):
            d.qpos[:] = za["qpos"][i]
            d.mocap_pos[:] = za["mocap_pos"][i]
            d.mocap_quat[:] = za["mocap_quat"][i]
            mujoco.mj_forward(m, d)
            assert np.allclose(d.xpos[m.body("tray").id], za["s_tray_pos"][i], atol=1e-4)


class FakeRun:
    """the attributes render.build_schedule needs, with scripted content."""

    def __init__(self, side, T, events=(), corridor=(10.0, 30.0)):
        n = int(T / TICK)
        self.side, self.n, self.T = side, n, T
        t = (np.arange(n) + 1) * TICK
        x = np.interp(t, [0, 5, T], [2.0, 6.0, 25.0])
        self.ph = np.zeros(n, np.int16)
        self.a = dict(s_t=t, s_pose=np.stack([x, np.full(n, 3.0), np.zeros(n)], 1), s_pdist=np.full(n, 9.0), s_wall=np.full(n, 0.5),
                      s_beads_out=np.zeros(n), s_bump_cum=np.zeros(n))
        self.wall = self.a["s_wall"]
        for te in events:
            self.a["s_beads_out"][int(te / TICK):] = 3
        self.meta = dict(events=[[0.0, "OUT"], [T - 5, "SENSE_C"]] if side == "shield" else [[0.0, "OUT"]])


    def idx(self, t):
        return int(np.clip(round(t / TICK) - 1, 0, self.n - 1))

    def done_at(self, t):
        return t >= self.T

    from world_model.v2.house.flagship.render import Run
    events = Run.events
    own = ref = staticmethod(lambda t: t)

    def rate(self, t, d=0.5):
        return 1.0


def test_schedule_compress_and_freeze():
    from world_model.v2.house.flagship import render as RD
    raw = FakeRun("raw", 60.0, events=(40.0,))
    shd = FakeRun("shield", 62.0)
    percs = {"counter": dict(kind="counter")}
    sched = RD.build_schedule([raw, shd], percs, dict())
    secs = {}
    for f in sched:
        secs[f["sec"]] = secs.get(f["sec"], 0) + 1
    assert secs["title"] == int(RD.CARD_S * RD.FPS) and secs["end"] == int(RD.END_S * RD.FPS)
    assert secs["event_raw_spill"] == int(RD.FREEZE_S * RD.FPS)
    assert secs["counter_perc"] == int(RD.PERC_S * RD.FPS)
    assert secs.get("drive_compressed", 0) > 50                   # something was time-lapsed
    sims = [f for f in sched if f["kind"] == "sim" and f["speed"] > 0]
    ts = np.array([f["t"] for f in sims])
    assert np.all(np.diff(ts) >= 0)                              # sim clock monotone, covers the whole run
    assert ts[-1] >= 61.0
    # the video is shorter than real time thanks to the time-lapse
    n_sim = len([f for f in sched if f["kind"] == "sim" and f["sec"] in ("sim", "drive_compressed")])
    assert n_sim / RD.FPS < 62.0
