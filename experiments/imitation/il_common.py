"""experiments/imitation/il_common.py -- shared plumbing for
"demos -> bimanual BC policy -> deployed under the FL-MPC safety filter".

SIMULATION ONLY. Nothing here opens a connection to any robot/SDK/network.

The flagship demo (scripts/make_cup_collect_safety_demo.py, owned by another
agent) is imported READ-ONLY and driven through monkeypatches of its
module-level globals (resolved at call time, so patching the module
attribute works -- see AGENTS.md "Gotchas" re: def-time defaults):

  * demo.right_gripper_opening  -> per-step HOOK (called once per control
    step for t >= RIGHT_START_T, BEFORE the controller reads track_r).
    expert mode: record obs + the expert's gripper command.
    policy mode: query the policy at POLICY_STRIDE, refresh the reference
    chunk, return the policy's gripper command.
  * demo.cube_pin_active        -> weld ("magnet") events.
    expert mode: original position-triggered logic (recorded).
    policy mode: weld ON when the policy commands closed AND the grasp site is
    within PIN_GATE_M of an unreleased cube's grasp point; weld OFF when the
    policy commands open. (No attach-from-afar.)
  * info["track_r"]             -> PolicyTrack (policy mode only): the task
    reference FL-MPC consumes (q_int_ref horizon = track_r.sample(t_r+k*DT)).
  * scene_setup.init_cup_grasped_state -> + deterministic per-seed cube xy
    jitter (the stock scene never moves the cubes; only the cup jitters).
  * demo._CUBE_OVERRIDE         -> fixed canonical pick ORDER (removes the
    "which cube first" multimodality that the farthest-first sort creates
    under jitter).
  * mpc_fl.FLMPC.step + sim_common.solve_constrained_box_qp -> OVERRIDE METER:
    on FL-MPC's main QP we also compute the box-only (no safety rows)
    solution x_unc = clip(H^-1 (-f)) and log ||a_con - a_unc|| for the
    first-block accel of each arm = how much the safety constraints
    (arm-arm, cup-in-barrier, table floor, torque box, cup tilt, leash)
    bent the executed motion away from pure tracking of the reference.

ACTION SPACE (what the policy outputs; NOT torques -- FL-MPC stays the
low-level certified layer): a chunk of K+1 knots at spacing CHUNK_DT_S
    a[j] = [ q_ref_right(t + j*CHUNK_DT_S) - q_right_meas(t)  (6, rad),
             grip_cmd_right(t + j*CHUNK_DT_S)                  (1, 0=closed..1=open) ]
    j = 0..K  -> shape (K+1, 7)
The LEFT (cup) arm's task reference is a constant hold at q_lift in the
flagship (FL-MPC's ego ref is hard-wired to info["q_lift"]), so it is not
learned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"),
           str(REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"), str(REPO_ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_cup_collect_safety_demo as demo  # noqa: E402  (READ-ONLY import; patched in-process only)
import sim_common as sc  # noqa: E402
import mpc_fl  # noqa: E402
import scene_setup  # noqa: E402

cct = demo.cct
bt = demo.bt

# ---------------------------------------------------------------------------
# spec constants
# ---------------------------------------------------------------------------
DT = sc.DT                       # 0.005 s control step (200 Hz)
POLICY_STRIDE = 10               # policy queried every 10 steps -> 20 Hz
CHUNK_K = 10                     # knots j=1..K beyond "now"
CHUNK_STRIDE = 10                # knot spacing in control steps
CHUNK_DT_S = CHUNK_STRIDE * DT   # 0.05 s -> chunk covers 0.5 s (FL-MPC horizon = 0.16 s)
ACT_DIM = 7
PIN_GATE_M = 0.15                # policy-mode weld gate [m]: ||grasp_site - (cube + GRASP_Z_OFFSET z)|| < this. LOOSE ON PURPOSE:
                                 # the stock expert welds at a fixed clock time with NO gate and its site is
                                 # typically ~0.10-0.13 m from the grasp point then (gripper_base collision
                                 # geom hits the cube/table), so a 3 cm gate would fail a perfect imitator.
GRIP_CLOSE_TH = 0.3              # policy grip < this (and in gate) -> weld ON
GRIP_OPEN_TH = 0.15              # policy grip > this while welded -> weld OFF (release)
IMG_SIZE = 96
IMG_CAMS = ("zed_cam", "right_arm_egocentric", "left_arm_egocentric")
CUBE_NAMES_ALL = [f"cube_{i}" for i in range(6)]
# canonical order = the stock farthest-from-cup order at zero jitter
CANONICAL_ORDER = ["cube_0", "cube_2", "cube_1", "cube_5", "cube_3", "cube_4"]
CUBE_JITTER_M = 0.03

OBS_FIELDS = [  # (name, dim) -- SSOT for the flat observation vector
    ("right_q", 6), ("right_qd", 6), ("left_q", 6), ("left_qd", 6),
    ("grip_right", 1), ("grip_left", 1),
    ("right_site_pos", 3),
    ("cup_pos", 3), ("cup_up", 3),
    ("cube_pos", 18),            # cube_0..cube_5 world xyz
    ("cube_rel_site", 18),       # cube_i - right grasp site
    ("task_mask", 6),            # TASK SPEC (goal conditioning): 1 if cube_i is in this episode's pick
                                 # set (the stock IK filter can drop cube_0 under jitter -> 2-cube task)
]
OBS_DIM = sum(d for _, d in OBS_FIELDS)


# ---------------------------------------------------------------------------
# global per-process episode context (one episode at a time per process)
# ---------------------------------------------------------------------------
class _Ctx:
    mode = None          # None | "expert" | "policy"
    ep = None            # EpisodeState


CTX = _Ctx()

_ORIG = dict(
    right_gripper_opening=demo.right_gripper_opening,
    cube_pin_active=demo.cube_pin_active,
    init_cup_grasped_state=scene_setup.init_cup_grasped_state,
    flmpc_step=mpc_fl.FLMPC.step,
    solve_qp=sc.solve_constrained_box_qp,
)
_CUBE_JITTER = {"m": 0.0}


def _init_with_cube_jitter(model, seed=0, cup_jitter_m=0.0):
    data = _ORIG["init_cup_grasped_state"](model, seed=seed, cup_jitter_m=cup_jitter_m)
    j = _CUBE_JITTER["m"]
    if j > 0.0:
        rng = np.random.default_rng(10_000 + int(seed))
        for name in CUBE_NAMES_ALL:
            bid = model.body(name).id
            qa = int(model.jnt_qposadr[model.body_jntadr[bid]])
            data.qpos[qa:qa + 2] += rng.uniform(-j, j, size=2)
        mujoco.mj_forward(model, data)
    return data


class EpisodeState:
    def __init__(self, info, mode, record=True, images=False, policy=None):
        self.info = info
        self.model, self.data = info["model"], info["data"]
        m = self.model
        self.mode = mode
        self.policy = policy
        self.scratch = mujoco.MjData(m)
        self.site_id = m.site("right_arm_ee_grasp").id
        self.cube_bids = [m.body(n).id for n in CUBE_NAMES_ALL]
        self.cup_body = info["cup_body"]
        self.rj7 = int(m.jnt_qposadr[m.joint("right_arm_joint7").id])
        self.lj7 = int(m.jnt_qposadr[m.joint("left_arm_joint7").id])
        self.track_r_expert = info["track_r"]
        sel = {c["name"] for c in info["cube_plan"]}
        self.task_mask = np.array([1.0 if n in sel else 0.0 for n in CUBE_NAMES_ALL])
        self.record = record
        self.rec = {k: [] for k in ("t_r", "obs", "q_ref", "grip_cmd", "pinned", "pin_idx")}
        self.pin_events = []           # (t_r, cube_idx, gate_dist, "on"/"off")
        self.images = images
        self.img = {c: [] for c in IMG_CAMS}
        self.img_step = []
        self._renderer = None
        # policy-mode state
        self.track = None
        self.grip_now = 1.0
        self.pinned_idx = None
        # override meter
        self.ov = {k: [] for k in ("t_r", "d_int", "d_ego", "a_int_unc", "feasible")}
        self._ov_pending = False
        self._ov_rec = None

    # ---------------- observation ----------------
    def obs(self):
        m, d, s = self.model, self.data, self.scratch
        s.qpos[:] = d.qpos
        s.qvel[:] = d.qvel
        mujoco.mj_kinematics(m, s)
        rq = d.qpos[self.info["right_qposadr"]]
        rqd = d.qvel[self.info["right_dofadr"]]
        lq = d.qpos[self.info["left_qposadr"]]
        lqd = d.qvel[self.info["left_dofadr"]]
        site = s.site_xpos[self.site_id].copy()
        cup = s.xpos[self.cup_body].copy()
        cup_up = s.xmat[self.cup_body].reshape(3, 3)[:, 2].copy()
        cubes = np.stack([s.xpos[b] for b in self.cube_bids])
        parts = [rq, rqd, lq, lqd,
                 [d.qpos[self.rj7] / 0.035], [d.qpos[self.lj7] / 0.035],
                 site, cup, cup_up, cubes.ravel(), (cubes - site).ravel(), self.task_mask]
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(np.float32)

    def site_pos_live(self):
        return self.data.site_xpos[self.site_id].copy()

    def render_images(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=IMG_SIZE, width=IMG_SIZE)
        s = self.scratch
        s.qpos[:] = self.data.qpos
        s.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, s)
        for c in IMG_CAMS:
            self._renderer.update_scene(s, camera=c)
            self.img[c].append(self._renderer.render().copy())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------------
# Policy-driven task reference (duck-types JointWaypointTrack.sample/.sample_dot)
# ---------------------------------------------------------------------------
class PolicyTrack:
    def __init__(self, q0):
        self.t0 = 0.0
        self.q = np.asarray(q0, float)[None].repeat(2, 0)    # dense (n,6) at DT
        self.qd = np.zeros_like(self.q)
        self.g = np.ones(2)

    def set_chunk(self, t0, q_knots, g_knots):
        """q_knots: (K+1,6) absolute joint refs at t0 + j*CHUNK_DT_S; g_knots (K+1,)."""
        tk = np.arange(q_knots.shape[0]) * CHUNK_DT_S
        n = int(round(tk[-1] / DT)) + 1
        td = np.arange(n) * DT
        q = np.stack([np.interp(td, tk, q_knots[:, j]) for j in range(6)], axis=1)
        self.qd = np.gradient(q, DT, axis=0)
        self.q = q
        self.g = np.clip(np.interp(td, tk, g_knots), 0.0, 1.0)
        self.t0 = float(t0)

    def _idx(self, t):
        return int(np.clip(round((t - self.t0) / DT), 0, self.q.shape[0] - 1))

    def sample(self, t):
        return self.q[self._idx(t)].copy()

    def sample_dot(self, t, h=1e-3):
        i = self._idx(t)
        return (self.qd[i].copy() if i < self.q.shape[0] - 1 else np.zeros(6))

    def grip(self, t):
        return float(self.g[self._idx(t)])


# ---------------------------------------------------------------------------
# patched demo hooks
# ---------------------------------------------------------------------------
def _hook_gripper(t, bounds, n_cubes, release_t=None):
    ep = CTX.ep
    if ep is None:
        return _ORIG["right_gripper_opening"](t, bounds, n_cubes, release_t)
    t_r = t - demo.RIGHT_START_T
    k_r = int(round(t_r / DT))
    o = ep.obs()
    if ep.mode == "expert":
        g = _ORIG["right_gripper_opening"](t, bounds, n_cubes, release_t)
        q_ref = ep.track_r_expert.sample(t_r)
    else:
        if ep.track is None or k_r % POLICY_STRIDE == 0:
            chunk = ep.policy(o)                              # (K+1, 7)
            q_now = ep.data.qpos[ep.info["right_qposadr"]]
            q_knots = q_now[None] + chunk[:, :6]
            if ep.track is not None:                          # C0 continuity at the re-plan instant
                q_knots[0] = ep.track.sample(t_r)
            if ep.track is None:
                ep.track = PolicyTrack(q_now)
                ep.info["track_r"] = ep.track                 # (already bound in run_episode -- see install)
            ep.track.set_chunk(t_r, q_knots, chunk[:, 6])
        g = ep.track.grip(t_r)
        q_ref = ep.track.sample(t_r)
    ep.grip_now = g
    if ep.record:
        ep.rec["t_r"].append(t_r)
        ep.rec["obs"].append(o)
        ep.rec["q_ref"].append(np.asarray(q_ref, np.float32))
        ep.rec["grip_cmd"].append(g)
    if ep.images and k_r % POLICY_STRIDE == 0:
        ep.render_images()
        ep.img_step.append(len(ep.rec["t_r"]) - 1)
    return g


def _hook_pin(t_r, bounds, n_cubes, *, right_ee_xy=None, place_xy_by_idx=None,
              right_ee_speed=None, released=None, **kw):
    ep = CTX.ep
    if ep is None:
        return _ORIG["cube_pin_active"](t_r, bounds, n_cubes, right_ee_xy=right_ee_xy,
                                        place_xy_by_idx=place_xy_by_idx,
                                        right_ee_speed=right_ee_speed, released=released, **kw)
    plan = ep.info["cube_plan"]
    site = ep.site_pos_live()
    if ep.mode == "expert":
        on, idx = _ORIG["cube_pin_active"](t_r, bounds, n_cubes, right_ee_xy=right_ee_xy,
                                           place_xy_by_idx=place_xy_by_idx,
                                           right_ee_speed=right_ee_speed, released=released, **kw)
    else:
        on, idx = False, None
        if ep.pinned_idx is not None:
            if ep.grip_now > GRIP_OPEN_TH:
                on, idx = False, ep.pinned_idx
            else:
                on, idx = True, ep.pinned_idx
        elif ep.grip_now < GRIP_CLOSE_TH:
            best, bd = None, PIN_GATE_M
            for i, c in enumerate(plan):
                if released is not None and released[i]:
                    continue
                gp = ep.data.xpos[c["body_id"]] + np.array([0.0, 0.0, demo.GRASP_Z_OFFSET])
                dd = float(np.linalg.norm(site - gp))
                if dd < bd:
                    best, bd = i, dd
            if best is not None:
                on, idx = True, best
    # bookkeeping (both modes): edges
    prev = ep.pinned_idx
    if on and prev is None:
        c = plan[idx]
        gp = ep.data.xpos[c["body_id"]] + np.array([0.0, 0.0, demo.GRASP_Z_OFFSET])
        ep.pin_events.append((float(t_r), int(idx), float(np.linalg.norm(site - gp)), "on"))
        ep.pinned_idx = idx
    elif not on and prev is not None:
        ep.pin_events.append((float(t_r), int(prev), float("nan"), "off"))
        ep.pinned_idx = None
    if ep.record and len(ep.rec["pinned"]) < len(ep.rec["t_r"]):
        ep.rec["pinned"].append(bool(on))
        ep.rec["pin_idx"].append(-1 if idx is None or not on else int(idx))
    return on, idx


def _flmpc_step_wrapped(self, *a, **kw):
    ep = CTX.ep
    if ep is not None:
        ep._ov_pending = True
        ep._ov_rec = None
    tau_e, tau_i, feas = _ORIG["flmpc_step"](self, *a, **kw)
    if ep is not None and ep._ov_rec is not None:
        x_con, x_unc = ep._ov_rec
        NV, nb = self.NV, self.n_blk
        ep.ov["d_ego"].append(float(np.linalg.norm(x_con[0:NV] - x_unc[0:NV])))
        ep.ov["d_int"].append(float(np.linalg.norm(x_con[nb:nb + NV] - x_unc[nb:nb + NV])))
        ep.ov["a_int_unc"].append(float(np.linalg.norm(x_unc[nb:nb + NV])))
        ep.ov["feasible"].append(bool(feas))
        ep.ov["t_r"].append(len(ep.ov["t_r"]))
    if ep is not None:
        ep._ov_pending = False
    return tau_e, tau_i, feas


def _solve_qp_wrapped(H, f, lower, upper, a_ineq, b_ineq, *a, **kw):
    res = _ORIG["solve_qp"](H, f, lower, upper, a_ineq, b_ineq, *a, **kw)
    ep = CTX.ep
    if ep is not None and ep._ov_pending:
        ep._ov_pending = False      # first call inside FLMPC.step = the main QP
        try:
            x_unc = np.linalg.solve(H, -f)
        except np.linalg.LinAlgError:
            x_unc = np.linalg.lstsq(H, -f, rcond=None)[0]
        x_unc = np.clip(x_unc, lower, upper)
        ep._ov_rec = (np.asarray(res[0]).copy(), x_unc)
    return res


def install_patches(cube_jitter_m=CUBE_JITTER_M, n_cubes=3, override_meter=False):
    """Idempotent. In-process only -- no file on disk is modified."""
    _CUBE_JITTER["m"] = float(cube_jitter_m)
    demo._CUBE_OVERRIDE = CANONICAL_ORDER[:max(int(n_cubes), 2)]
    demo.right_gripper_opening = _hook_gripper
    demo.cube_pin_active = _hook_pin
    scene_setup.init_cup_grasped_state = _init_with_cube_jitter
    if override_meter:
        mpc_fl.FLMPC.step = _flmpc_step_wrapped
        sc.solve_constrained_box_qp = _solve_qp_wrapped
    else:
        mpc_fl.FLMPC.step = _ORIG["flmpc_step"]
        sc.solve_constrained_box_qp = _ORIG["solve_qp"]


# ---------------------------------------------------------------------------
# episode runner (expert or policy) + metrics
# ---------------------------------------------------------------------------
def placed_selected(model, data, info, init_pos, radius=demo.KEEP_ZONE_RADIUS_M, z_tol=0.05,
                    vel_tol=0.05, min_move_m=0.05):
    """count_placed_cubes restricted to the SELECTED cubes AND requiring the cube to
    have actually been moved (> min_move_m from its t=0 position). The stock metric
    counts every cube_* in the keep zone, incl. un-selected cube_3 (resting inside the
    zone at t=0) and, under jitter, a selected cube that STARTS inside the zone."""
    plan = info["cube_plan"]
    center = np.mean(np.stack([c["place_xyz"][:2] for c in plan]), axis=0)
    n = 0
    for c in plan:
        b = c["body_id"]
        p = data.xpos[b]
        if np.hypot(p[0] - center[0], p[1] - center[1]) > radius:
            continue
        if abs(p[2] - (info["table_top_z"] + cct.CUBE_HALF)) > z_tol:
            continue
        if np.linalg.norm(p[:2] - init_pos[b][:2]) < min_move_m:
            continue
        da = model.body_dofadr[b]
        if np.linalg.norm(data.qvel[da:da + 3]) > vel_tol:
            continue
        n += 1
    return n


def run(seed, *, mode="expert", policy=None, cup_jitter_m=0.02, n_cubes=3, images=False,
        time_scale=1.0, override_meter=False, cube_jitter_m=CUBE_JITTER_M, max_steps=None):
    install_patches(cube_jitter_m=cube_jitter_m, n_cubes=n_cubes, override_meter=override_meter)
    info = demo.build_episode(seed=seed, scenario="crossing", cup_jitter_m=cup_jitter_m, n_cubes=n_cubes)
    n_exp = info["n_steps_total"]
    init_pos = {c["body_id"]: info["data"].xpos[c["body_id"]].copy() for c in info["cube_plan"]}
    info["n_steps_total"] = int(round(n_exp * time_scale))
    ep = EpisodeState(info, mode, record=True, images=images, policy=policy)
    if mode == "policy":
        # run_episode binds track_r = info["track_r"] at entry, so the PolicyTrack
        # must exist BEFORE the call; it is re-chunked in place by the hook.
        ep.track = PolicyTrack(info["q_home"])
        info["track_r"] = ep.track
    CTX.mode, CTX.ep = mode, ep
    try:
        log, _ = demo.run_episode(info, controller="flmpc", max_steps=max_steps)
    finally:
        CTX.mode, CTX.ep = None, None
        ep.close()
    m, d = info["model"], info["data"]
    s = demo.summarize(log)
    tr = log["right_track_t"] >= 0
    s.update(
        seed=int(seed), mode=mode, n_cubes_sel=len(info["cube_plan"]),
        cube_names=[c["name"] for c in info["cube_plan"]],
        placed_sel_final=int(placed_selected(m, d, info, init_pos)),
        cup_tilt_max_deg=float(np.degrees(np.max(log["left_cup_tilt_rad"][tr]))) if tr.any() else 0.0,
        n_steps=int(len(log["t"])), n_steps_expert=int(n_exp),
        n_pin_on=sum(1 for e in ep.pin_events if e[3] == "on"),
        pin_gate_dists=[round(e[2], 4) for e in ep.pin_events if e[3] == "on"],
    )
    if ep.ov["d_int"]:
        di, de = np.array(ep.ov["d_int"]), np.array(ep.ov["d_ego"])
        s.update(ov_int_mean=float(di.mean()), ov_int_p95=float(np.percentile(di, 95)),
                 ov_int_frac_gt1=float(np.mean(di > 1.0)), ov_int_frac_gt5=float(np.mean(di > 5.0)),
                 ov_ego_mean=float(de.mean()), ov_ego_frac_gt1=float(np.mean(de > 1.0)),
                 ov_infeasible_frac=float(1.0 - np.mean(ep.ov["feasible"])))
    # reference-vs-executed right-EE deviation (FK of the consumed reference)
    if ep.rec["q_ref"]:
        qr = np.stack(ep.rec["q_ref"])
        sc_ = mujoco.MjData(m)
        sc_.qpos[:] = d.qpos
        ee_ref = np.zeros((len(qr), 3))
        for i, q in enumerate(qr):
            sc_.qpos[info["right_qposadr"]] = q
            mujoco.mj_kinematics(m, sc_)
            ee_ref[i] = sc_.site_xpos[ep.site_id]
        obs = np.stack(ep.rec["obs"])
        o_site = obs[:, 26:29]   # right_site_pos slice (see OBS_FIELDS)
        dev = np.linalg.norm(ee_ref - o_site, axis=1)
        s.update(ref_ee_dev_mean_m=float(dev.mean()), ref_ee_dev_p95_m=float(np.percentile(dev, 95)))
    return s, ep, log, info


def obs_slice(name):
    i = 0
    for n, dmn in OBS_FIELDS:
        if n == name:
            return slice(i, i + dmn)
        i += dmn
    raise KeyError(name)


assert obs_slice("right_site_pos") == slice(26, 29) and OBS_DIM == 77
