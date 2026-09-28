"""experiments/baselines/mjpc/mjpc_planner.py

Python side of the REAL MuJoCo MPC (MJPC) baseline -- SIMULATION ONLY.

* MJPCServer: spawns the headless C++ planner (cpp/mjpc_server.cc, built
  against google-deepmind/mujoco_mpc; binary at MJPC_SERVER_BIN) and talks to
  it over stdin/stdout pipes (no sockets, no network, no hardware).
* MJPCController: drop-in for mpc_fl.FLMPC's interface (`.NP`, `.step(...)`
  -> (tau_left, tau_right, feasible)) so scripts/make_cup_collect_safety_demo.py
  `run_episode(controller="flmpc")` drives the SAME episode loop (same lift
  phase, same cup re-pin, same weld/release logic, same torque filter + clip,
  same logging) with MJPC planning the torques instead of FL-MPC. It is
  injected at runtime by run_mjpc_compare.py (no repo file is edited).

The controller receives exactly what FL-MPC receives -- the right arm's task
reference (track_r, sampled over the horizon), the left hold pose q_lift, the
cup keypoints, the plant state -- and returns 12 joint torques.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"),
           str(REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"), str(REPO_ROOT / "eval"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_planning_model as mpm  # noqa: E402

MJPC_SERVER_BIN = os.environ.get(
    "MJPC_SERVER_BIN", "/common/users/ss5772/tools/mujoco_mpc/build/bin/mjpc_bimanual_server")


class MJPCServer:
    def __init__(self, xml_path: str, threads: int = 8, log_path: str | None = None):
        self._errf = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self.p = subprocess.Popen([MJPC_SERVER_BIN, xml_path, str(int(threads))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errf,
                                  bufsize=0)
        hdr = self._read(24)
        self.nq, self.nv, self.na, self.nu, self.nud, self.nmocap = struct.unpack("<6i", hdr)

    def _read(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("mjpc server died (see its stderr log)")
            buf += chunk
        return buf

    def step(self, t: float, qpos, qvel, act, userdata, mocap_pos, mocap_quat, n_iter: int = 1):
        payload = np.concatenate([[t], qpos, qvel, act, userdata,
                                  np.ravel(mocap_pos), np.ravel(mocap_quat)]).astype("<f8")
        self.p.stdin.write(struct.pack("<2i", 1, int(n_iter)) + payload.tobytes())
        out = self._read(4 + 8 * (self.nu + 2))
        status = struct.unpack("<i", out[:4])[0]
        vals = np.frombuffer(out[4:], dtype="<f8")
        return status, vals[:self.nu].copy(), float(vals[self.nu]), float(vals[self.nu + 1])

    def _named(self, cmd: int, name: str, value: float) -> int:
        b = name.encode()
        self.p.stdin.write(struct.pack("<2i", cmd, len(b)) + b + struct.pack("<d", float(value)))
        return struct.unpack("<i", self._read(4))[0]

    def set_weight(self, name: str, value: float) -> int:
        return self._named(2, name, value)

    def set_param(self, name: str, value: float) -> int:
        return self._named(4, name, value)

    def reset_to(self, action):
        a = np.asarray(action, dtype="<f8")
        assert a.shape == (self.nu,)
        self.p.stdin.write(struct.pack("<i", 6) + a.tobytes())
        return struct.unpack("<i", self._read(4))[0]

    def reset(self):
        self.p.stdin.write(struct.pack("<i", 3))
        return struct.unpack("<i", self._read(4))[0]

    def close(self):
        try:
            self.p.stdin.write(struct.pack("<i", 0))
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
        if self._errf is not subprocess.DEVNULL:
            self._errf.close()


# Filled by run_mjpc_compare.py before run_episode: {"info": build_episode info, "cfg": dict}
CONTEXT: dict = {}


class MJPCController:
    """mpc_fl.FLMPC-compatible shim (see module docstring)."""

    def __init__(self, model, **_fl_kwargs):
        info, cfg = CONTEXT["info"], CONTEXT["cfg"]
        self.info, self.cfg = info, cfg
        self.NP = 32                      # only used by run_episode to sample q_*_ref (we resample ourselves)
        self.DT = float(model.opt.timestep)
        self.a_ego_prev = np.zeros(6)     # interface parity (real-grasp friction FF only)
        self.a_int_prev = np.zeros(6)
        tmp = Path(cfg["tmp_dir"])
        tmp.mkdir(parents=True, exist_ok=True)
        tag = f"s{cfg.get('seed', 0)}_{os.getpid()}"
        self.xml = str(tmp / f"mjpc_plan_{tag}.xml")
        self.act_mode = cfg.get("act_mode", "torque")
        mpm.build_planning_xml(self.xml, info["table_top_z"], weights=cfg.get("weights"),
                               params=cfg.get("params"), agent=cfg.get("agent"), act_mode=self.act_mode)
        self.m_plan = mujoco.MjModel.from_xml_path(self.xml)
        self.qmap, self.vmap = mpm.planning_joint_map(self.m_plan, model)
        self.qmap = np.array(self.qmap)
        self.vmap = np.array(self.vmap)
        # plant dof of each planning actuator (actuator order: left j1..6, right j1..6)
        self.act_dof = np.array([model.jnt_dofadr[model.joint(model_name).id]
                                 for model_name in [self.m_plan.actuator(i).name.replace("tau_", "")
                                                    for i in range(self.m_plan.nu)]])
        self.server = MJPCServer(self.xml, threads=cfg.get("threads", 8),
                                 log_path=str(tmp / f"mjpc_server_{tag}.log"))
        assert self.server.nud == mpm.N_USERDATA and self.server.nq == self.m_plan.nq
        self.left_ee = info["left_ee_body"]
        # cube mocap obstacles: plan mocap id -> plant cube body id (+ weld eq id if selected)
        eq_by_body = {c["body_id"]: c["eq_id"] for c in info["cube_plan"]}
        self.cubes = [(mid, model.body(name).id, eq_by_body.get(model.body(name).id))
                      for mid, name in mpm.planning_mocap_cubes(self.m_plan)]
        self.mocap_pos = np.zeros((self.m_plan.nmocap, 3))
        self.mocap_quat = np.tile([1.0, 0.0, 0.0, 0.0], (self.m_plan.nmocap, 1))
        self._seen_held, self._released = set(), set()
        self.ud = np.zeros(mpm.N_USERDATA)
        # leash anchor == FL-MPC's (EE FK at q_ego_ref[0] == q_lift)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = info["data"].qpos
        scratch.qpos[info["left_qposadr"]] = info["q_lift"]
        mujoco.mj_forward(model, scratch)
        self.ud[mpm.UD_HOLDPOS:mpm.UD_HOLDPOS + 3] = scratch.xpos[self.left_ee]
        self.ud[mpm.UD_QHOLD:mpm.UD_QHOLD + 6] = info["q_lift"]
        kps = info["cup_barrier_keypoints"]
        assert len(kps) <= mpm.MAX_CUP
        self.ud[mpm.UD_NCUP] = len(kps)
        for i, (bid, off, r) in enumerate(kps):
            assert bid == self.left_ee
            self.ud[mpm.UD_CUPKP + 4 * i:mpm.UD_CUPKP + 4 * i + 4] = [off[0], off[1], off[2], r]
        self.dt_ref = float(cfg.get("dt_ref", 0.01))
        horizon = float((cfg.get("agent") or {}).get("agent_horizon", mpm.DEFAULT_AGENT["agent_horizon"]))
        self.nk = min(mpm.MAX_KNOTS, int(np.ceil(horizon / self.dt_ref)) + 3)
        self.ud[mpm.UD_DTREF] = self.dt_ref
        self.ud[mpm.UD_NK] = self.nk
        self.n_iter = int(cfg.get("n_iter", 1))
        self.plan_every = int(cfg.get("plan_every", 1))
        self.cup_axis_captured = False
        self.k = 0
        self.plan_ms, self.best_cost, self.step_ms = [], [], []
        CONTEXT["instance"] = self

    def step(self, model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
             q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs_all, **kw):
        t0 = time.perf_counter()
        track_r = self.info["track_r"]
        t_r = self.k * self.DT            # first call is at t_r == 0 (RIGHT_START_T), one call per step
        if not np.allclose(track_r.sample(t_r), q_int_ref[0], atol=1e-9):
            raise RuntimeError("MJPCController: reference clock out of sync with run_episode")
        q_arm = data.qpos[np.concatenate([left_qposadr, right_qposadr])]
        qd_arm = data.qvel[np.concatenate([left_dofadr, right_dofadr])]
        if self.k == 0 and self.act_mode == "pd":   # nominal setpoint policy = hold current pose
            self.server.reset_to(q_arm)
        if not self.cup_axis_captured:    # same lazy capture as mpc_fl.FLMPC (first step)
            R = data.xmat[self.left_ee].reshape(3, 3)
            self.ud[mpm.UD_CUPAXIS:mpm.UD_CUPAXIS + 3] = R.T @ np.array([0.0, 0.0, 1.0])
            self.cup_axis_captured = True
        tt = t_r + np.arange(self.nk) * self.dt_ref
        ref = np.zeros((self.nk, 12))
        for j, tj in enumerate(tt):
            ref[j, :6] = track_r.sample(tj)
            ref[j, 6:] = track_r.sample_dot(tj)
        self.ud[mpm.UD_T0] = data.time
        self.ud[mpm.UD_REF:mpm.UD_REF + 12 * self.nk] = ref.ravel()
        qpos = np.zeros(self.m_plan.nq)
        qvel = np.zeros(self.m_plan.nv)
        qpos[self.qmap[:, 0]] = data.qpos[self.qmap[:, 1]]
        qvel[self.vmap[:, 0]] = data.qvel[self.vmap[:, 1]]
        act = data.qfrc_applied[self.act_dof].copy()    # plant torque-filter state (last applied torque)
        # every un-held cube is a static obstacle at its current plant pose (an 8 g cube
        # pressed onto the table is effectively immovable, and the gripper does hit the
        # target cube in the plant); the HELD cube is parked out of the planning scene
        # (its arm collisions are disabled in the plant while welded).
        for mid, bid, eq in self.cubes:
            if eq is not None and data.eq_active[eq]:
                self.mocap_pos[mid] = [0.0, 0.0, -10.0 - mid]
            else:
                self.mocap_pos[mid] = data.xpos[bid]
            self.mocap_quat[mid] = data.xquat[bid]
        n_iter = self.n_iter if (self.k % self.plan_every == 0) else 0
        status, ctrl, plan_ms, best = self.server.step(data.time, qpos, qvel, act, self.ud,
                                                       self.mocap_pos, self.mocap_quat, n_iter=n_iter)
        self.k += 1
        self.plan_ms.append(plan_ms)
        self.best_cost.append(best)
        self.step_ms.append(1e3 * (time.perf_counter() - t0))
        if self.act_mode == "pd":         # identical PD law to the planning model's actuator
            tau = np.clip(mpm.PD_KP * (ctrl - q_arm) - mpm.PD_KD * qd_arm, -100.0, 100.0)
        else:
            tau = ctrl
        return tau[:6].copy(), tau[6:].copy(), True

    def close(self):
        self.server.close()
