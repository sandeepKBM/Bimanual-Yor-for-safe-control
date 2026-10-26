"""tamp/rh/shadow.py -- STAGE 0 "shadow mode" of the stereo-only planner input.  SIMULATION ONLY (MuJoCo + EGL render + OpenCV SGBM; no robot, no network).

rh_episode(..., observe="shadow"): the robot is driven EXACTLY as with observe="truth" (planning on sim truth).  In parallel, a stereo pipeline runs every sim tick
(stereo.live.StereoOCS in "inproc" mode: head-ZED render pair -> SGBM -> self-mask -> DepthDetector -> MultiTracker KF; PoseEstimator = wheel odometry + ZED VIO), and at every
planning decision the SAME candidates are certified by api.admit() on an ESTIMATE state:
    pose   = estimated pose (the truth-predicted state is placed in the estimated frame by the body-frame relative motion, see place_est_state),
    people = confirmed tracks (c, v_hat) in the estimated frame,
    world  = the planner World whose occupancy grid is the STEREO-BUILT perceived map (loc/mapper, perceived_h5.npz) and whose pose box e_xy = k_pose * loc.sigma_xy.
false_safe   = the candidate the estimate path admits FAILS the truth certificate (true pose, true people, true grid);
false_unsafe = the candidate the truth path admitted is REFUSED by the estimate path.

Background subtraction of the people detector: stereo.live.StereoOCS(det_occ=<perceived OccGrid>) switches the render-based subtraction (people-free twin render = the TRUE house
model) OFF and makes DepthDetector reject returns near occupied cells of the PERCEIVED map (use_static_map=True, occ=perceived).  The static-depth fallback that would cast
rays against the true model is also disabled (zs = inf), so nothing of the true house model reaches the detector.

Remaining truth reads (listed in result()["shadow"]["truth_reads"]) are sensor-simulation inputs or unreplaced planner inputs, not estimator outputs.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

PERSON_R = 0.22
TORSO_Z = 0.9
GATE_M = 0.5
R_MAX = 4.5                         # DepthDetector.r_max default (m): farther people are out of the detector's design range

TRUTH_READS = [
    "people: none read by the estimate path (tracks only); truth people are used for metrics and for the false_safe re-certification",
    "pose: PoseEstimator.update(t, sim.pose(), dt) -- the simulated wheel-odometry / VIO sensor is driven by the true motion (it is the sensor model, as in loc/)",
    "pose of a predicted (drive) decision: the truth-predicted state (filt.predict_state on sim truth) is placed in the estimated frame by the relative body motion",
    "walls: perceived stereo-built map (loc/mapper); NOT the true grid",
    "people detector background: perceived map EDT (bg render off, zs=inf); NOT the true house model",
    "stereo render: the twin is built from the true sim model (textured copy) and loaded with the true state each frame -- the camera's world",
    "self-mask: the FK-known robot geometry from a segmentation render of the true model (perfect-FK assumption, stereo/README)",
    "encoders (tray FK prior): true joint state + noise (stereo.live)",
    "cup_rel / cup_world / tray mass / pose box template: World built by world_from_sim from the coffee-world sim (nominal payload geometry), UNCHANGED in Stage 0",
    "the false_safe/false_unsafe bookkeeping, people matching, visibility and map error read truth by design (evaluation only)",
]


# ===================================================================================== pure helpers (unit-tested)
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def se2_inv(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-(c * p[0] + s * p[1]), -(-s * p[0] + c * p[1]), wrap(-p[2]))


def se2_mul(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def place_est_state(pose_pred_true, pose_true_now, pose_est_now):
    """estimated map pose of a state predicted from the true state: est_now o (true_now^-1 o pred)  (the same relative BODY motion applied to the estimate)."""
    rel = se2_mul(se2_inv(tuple(pose_true_now)), tuple(pose_pred_true))
    return se2_mul(tuple(pose_est_now), rel)


def pose_error(est, true):
    """-> (xy error [m], |yaw error| [rad])"""
    return float(math.hypot(est[0] - true[0], est[1] - true[1])), float(abs(wrap(est[2] - true[2])))


def match_people(truth_xy, track_xy, gate: float = GATE_M):
    """Hungarian match (cost = Euclidean distance, pairs farther than `gate` are not matches).
    -> (pairs [(i_truth, j_track, dist)], missed [i_truth], false_tracks [j_track])"""
    from scipy.optimize import linear_sum_assignment
    T, K = np.asarray(truth_xy, float).reshape(-1, 2), np.asarray(track_xy, float).reshape(-1, 2)
    if len(T) == 0 or len(K) == 0:
        return [], list(range(len(T))), list(range(len(K)))
    D = np.linalg.norm(T[:, None, :] - K[None, :, :], axis=2)
    big = 1e6
    r, c = linear_sum_assignment(np.where(D <= gate, D, big))
    pairs = [(int(i), int(j), float(D[i, j])) for i, j in zip(r, c) if D[i, j] <= gate]
    mi, mj = {p[0] for p in pairs}, {p[1] for p in pairs}
    return pairs, [i for i in range(len(T)) if i not in mi], [j for j in range(len(K)) if j not in mj]


def classify_visibility(points, origin, R, K, W, H, ray_blocked, r_max: float = R_MAX):
    """per 3-D point: 'visible' (inside the frustum, range <= r_max, line of sight clear) | 'occluded' (inside the frustum / range, blocked) | 'out_of_view' (outside the
    frustum or beyond r_max).  Camera axes as stereo.adapters: camera coords = R^T (P - origin) = (right, up, back).  ray_blocked(origin, P) -> bool."""
    out = []
    o, R = np.asarray(origin, float), np.asarray(R, float)
    for P in np.asarray(points, float).reshape(-1, 3):
        c = R.T @ (P - o)
        z = -c[2]
        if z <= 0.1 or float(np.linalg.norm(P - o)) > r_max:
            out.append("out_of_view")
            continue
        u, v = K["cx"] + K["f"] * c[0] / z, K["cy"] - K["f"] * c[1] / z
        if not (0.0 <= u < W and 0.0 <= v < H):
            out.append("out_of_view")
            continue
        out.append("occluded" if ray_blocked(o, P) else "visible")
    return out


def pct(x, q):
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(a, q)) if len(a) else None


def summarize(rows, tick_ms, frame_ms, map_error=None):
    """summary block of the per-decision rows (see Shadow.on_decision for the row fields)."""
    n = len(rows)
    cnt = {k: sum(r["people"].get(k, 0) for r in rows) for k in ("n_truth", "n_match", "miss_visible", "miss_occluded", "miss_out_of_view", "miss_no_frame", "n_false_tracks")}
    n_vis = cnt["miss_visible"] + sum(r["people"].get("n_match_visible", 0) for r in rows)
    n_occ = cnt["miss_occluded"] + sum(r["people"].get("n_match_occluded", 0) for r in rows)
    pe = [e for r in rows for e in r["people"].get("pos_err", [])]
    ve = [e for r in rows for e in r["people"].get("vel_err", [])]
    return dict(
        n_decisions=n, n_false_safe=sum(1 for r in rows if r["false_safe"]), n_false_unsafe=sum(1 for r in rows if r["false_unsafe"]),
        n_est_admit=sum(1 for r in rows if r["est_adm"] is not None), n_truth_admit=sum(1 for r in rows if r["truth_adm"] is not None),
        n_same_candidate=sum(1 for r in rows if r["est_adm"] == r["truth_adm"]),
        people=dict(counts=cnt, n_visible_person_decisions=n_vis, n_occluded_person_decisions=n_occ,
                    miss_rate_visible=(cnt["miss_visible"] / n_vis) if n_vis else None, miss_rate_occluded=(cnt["miss_occluded"] / n_occ) if n_occ else None,
                    pos_err_p50_m=pct(pe, 50), pos_err_p95_m=pct(pe, 95), vel_err_p50_ms=pct(ve, 50), vel_err_p95_ms=pct(ve, 95), gate_m=GATE_M),
        pose=dict(xy_err_p50_m=pct([r["pose_err_xy"] for r in rows], 50), xy_err_p95_m=pct([r["pose_err_xy"] for r in rows], 95),
                  yaw_err_p50_rad=pct([r["pose_err_yaw"] for r in rows], 50), yaw_err_p95_rad=pct([r["pose_err_yaw"] for r in rows], 95)),
        stereo_ms_per_tick=dict(mean=float(np.mean(tick_ms)) if len(tick_ms) else None, p50=pct(tick_ms, 50), p95=pct(tick_ms, 95), n_ticks=len(tick_ms)),
        stereo_ms_per_frame=dict(mean=float(np.mean(frame_ms)) if len(frame_ms) else None, p50=pct(frame_ms, 50), p95=pct(frame_ms, 95), n_frames=len(frame_ms)),
        map_error=map_error)


# ===================================================================================== the shadow pipeline
class Shadow:
    def __init__(self, sim, world, gk_plan, cfg, seed: int = 0, map_path=None, stereo_cfg=None, mode: str = "inproc"):
        from ...stereo.cfg import StereoCfg
        from ...stereo import live as LV
        from ...loc import mapper as MP
        self.sim, self.world, self.gk, self.cfg = sim, world, gk_plan, cfg
        self.perc = MP.resolve_occgrid(sim.scene, map_path)                    # the stereo-BUILT map (default cache perceived_h5.npz)
        self.w_est = dataclasses.replace(world, bel0=world.bel0.with_(grid=self.perc))
        self.ocs = LV.attach(sim, stereo_cfg or StereoCfg(mode="3way"), mode=mode, seed=seed, people=True, payload=False, det_occ=self.perc)
        self.rows, self.tick_ms, self.frame_ms = [], [], []
        self._nf = 0
        self.mask = np.zeros(6, np.uint8)
        self.mask[[3, 4]] = 1
        self.gid = np.zeros(1, np.int32)

    # ------------------------------------------------------------------ per tick
    def step(self, sim):
        t0 = time.perf_counter()
        self.ocs.step(sim)
        self.tick_ms.append(1e3 * (time.perf_counter() - t0))
        s = self.ocs.sensors
        if s.n_frames != self._nf:
            self._nf = s.n_frames
            fr = s.frame
            if fr is not None and "ms" in fr:
                self.frame_ms.append(float(fr["ms"]["total"]))

    # ------------------------------------------------------------------ helpers
    def tracks(self):
        return [(np.asarray(tr.x[:2], float).copy(), np.asarray(tr.x[2:4], float).copy()) for tr in self.ocs._pp.trk.output(confirmed_only=True)]

    def _blocked(self, o, P):
        import mujoco
        m, d = self.sim.m, self.sim.d
        v = P - o
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            return False
        dist = mujoco.mj_ray(m, d, np.asarray(o, float), v / L, self.mask, 1, -1, self.gid)
        return bool(self.gid[0] >= 0 and dist >= 0 and dist < L - PERSON_R)

    def _vis(self, pts3):
        fr = self.ocs.sensors.frame                               # newest DELIVERED frame (stereo_frame() is only set on the delivery tick)
        if fr is None:
            return ["no_frame"] * len(pts3)
        return classify_visibility(pts3, fr["origin"], fr["R"], fr["K"], self.ocs.sensors.c.W, self.ocs.sensors.c.H, self._blocked)

    def people_row(self, truth_people, tracks):
        tx = [np.asarray(c, float) for c, _ in truth_people]
        kx = [c for c, _ in tracks]
        pairs, missed, false_t = match_people(tx, kx, GATE_M)
        vis = self._vis([np.array([c[0], c[1], TORSO_Z]) for c in tx])
        row = dict(n_truth=len(tx), n_tracks=len(kx), n_match=len(pairs), n_false_tracks=len(false_t), miss_visible=0, miss_occluded=0, miss_out_of_view=0, miss_no_frame=0,
                   n_match_visible=0, n_match_occluded=0, pos_err=[], vel_err=[], vis=vis)
        for i in missed:
            row["miss_" + vis[i]] += 1
        for i, j, dist in pairs:
            row["pos_err"].append(round(dist, 4))
            row["vel_err"].append(round(float(np.linalg.norm(np.asarray(truth_people[i][1], float) - tracks[j][1])), 4))
            if vis[i] in ("visible", "occluded"):
                row["n_match_" + vis[i]] += 1
        return row

    # ------------------------------------------------------------------ per planning decision
    def on_decision(self, kind, t, state, delay, cands, adm, true_now_pose):
        """state: the TRUTH state the loop planned from (pose may be predicted); adm: the candidate the truth path admitted (or None)."""
        from .api import State, admit
        est_now = tuple(float(x) for x in self.ocs.loc.pose)
        pose_est = place_est_state(state.pose, true_now_pose, est_now)
        trk = self.tracks()
        v_est = tuple(float(x) for x in self.ocs.loc.v_odo)
        st_est = State(pose_est, v_est, state.t, [(c, v) for c, v in trk])
        e_xy = float(self.gk.k_pose) * float(self.ocs.loc.sigma_xy)
        w = dataclasses.replace(self.w_est, bel0=self.w_est.bel0.with_(e_xy=e_xy, e_th=self.gk.th_over_xy * e_xy))
        c_est, log_est = admit(cands, st_est, w, self.gk, self.cfg, True, t0=float(delay))
        fs = fu = False
        fs_m = fs_r = fu_m = fu_r = None
        if c_est is not None:                                                     # re-certify the estimate-admitted candidate under TRUTH
            _, lg = admit([c_est], state, self.world, self.gk, self.cfg, True, t0=float(delay))
            if not lg[0][1]:
                fs, fs_m, fs_r = True, lg[0][2], lg[0][3]
        if adm is not None:                                                       # the truth-admitted candidate under the ESTIMATES
            c2, lg = admit([adm], st_est, w, self.gk, self.cfg, True, t0=float(delay))
            if not lg[0][1]:
                fu, fu_m, fu_r = True, lg[0][2], lg[0][3]
        pe_xy, pe_yaw = pose_error(est_now, true_now_pose)
        prow = self.people_row(state.people, trk)
        self.rows.append(dict(t=round(float(t), 2), kind=kind, delay_s=round(float(delay), 3), n_cands=len(cands), truth_adm=None if adm is None else adm.source,
                              est_adm=None if c_est is None else c_est.source, est_first_reason=log_est[0][3] if log_est else None,
                              false_safe=fs, false_safe_margin=fs_m, false_safe_reason=fs_r, false_unsafe=fu, false_unsafe_margin=fu_m, false_unsafe_reason=fu_r,
                              pose_err_xy=pe_xy, pose_err_yaw=pe_yaw, e_xy_used=e_xy, sigma_xy=float(self.ocs.loc.sigma_xy),
                              frames=int(self.ocs.sensors.n_frames), people=prow))

    # ------------------------------------------------------------------ result
    def map_error(self):
        try:
            from ...loc import mapper as MP
            route = np.asarray(self.sim.scene.out_route, float)
            tz = MP.true_grid_zband(self.sim.scene)
            return dict(vs_zband=MP.map_error(self.perc, tz, route), vs_planner_true_grid=MP.map_error(self.perc, self.world.bel0.grid, route))
        except Exception as e:                                                    # pragma: no cover  (reported, never fatal to the episode)
            return dict(error=repr(e))

    def result(self):
        out = dict(summary=summarize(self.rows, self.tick_ms, self.frame_ms, self.map_error()), rows=self.rows, truth_reads=TRUTH_READS,
                   stereo=dict(mode="inproc", cfg=self.ocs.sensors.c.asdict(), latency_ms=self.ocs.sensors.latency_ms, ready_ms=self.ocs.ready_ms),
                   detector=dict(bg=bool(self.ocs._pp.det.bg), use_static_map=bool(self.ocs._pp.det.use_static_map), occ="perceived", min_pts=int(self.ocs._pp.det.min_pts)))
        return out

    def close(self):
        self.ocs.close()
