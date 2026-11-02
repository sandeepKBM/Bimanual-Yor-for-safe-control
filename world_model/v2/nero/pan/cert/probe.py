"""pan/cert/probe.py -- pivot probe readings -> torque rows on pi = (m, s_u, s_w).  SIMULATION ONLY.

Proposal, "Probing": the left claw raises the handle a few mm while the far rim stays on the stove; the pan is a beam on two
supports.  Moment balance of the pan about two lines through the far-rim pivot (stove force has no moment about them):

    pitch, axis w (horizontal, perpendicular to the handle):  M_w = g (M0 a0 + m r_out + s_u)
    roll,  axis u (along the handle):                         M_u = -g (M0 b0 + s_w)

M_w, M_u: moments the pan exerts on the left claw about those lines, from joint torques (claw 6-wrench at the gripper body).
a0, b0: pan + lid COM offsets from the pivot along u, w (MODEL: MJCF geometry, a constant in the pan frame).
The frame (u, w, pivot) is what the ROBOT knows after the pinch: handle axis = aimed yaw + left-claw rotation (arm FK), pivot =
claw position - (lever from grasp to the far rim) u.  Errors in that frame, the small tilt and a rim patch instead of a point
are model error: eps = MEASURED max residual over calibration worlds + the propagated sensor-noise bound.

    python -m world_model.v2.nero.pan.cert.probe calibrate --out DIR --n 40 --procs P     # measures eps (noise-free)
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

G = 9.81
PROBE_DZ = 0.015       # design: handle raise [m] (as in baseline_scripted)
PROBE_T = 0.6          # design: raise duration [s]; peak commanded acceleration 6 dz / T^2 = 0.25 m/s^2
HOLD_T = 0.6           # design: hold before reading [s]
N_READ = 60            # design: readings averaged over the last N_READ steps of the hold (0.3 s)
TAU_NOISE = 0.05       # ASSUMED: joint-torque sensor noise, uniform within +-TAU_NOISE per joint [Nm] (bounded noise model)


def _wrench6(ep, s, rng=None):
    """claw 6-wrench (force, moment about the gripper-body origin) the environment exerts, from contact joint torques.
    Same ASSUMED perfect joint-friction compensation as segments.Episode.claw_wrench.  Returns (w6, P) with w6 = P tau."""
    import mujoco
    m, d = ep.m, ep.d
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jp, jr, ep.gb[s])
    J = np.vstack([jp, jr])[:, ep.dofs[s]]
    tau = ep.contact_torque()[ep.dofs[s]]
    if rng is not None:
        tau = tau + rng.uniform(-TAU_NOISE, TAU_NOISE, tau.shape)
    P = np.linalg.pinv(J.T)
    return P @ tau, P


def pan_frame_model(ep):
    """MODEL constants of the empty pan + lid in the pan frame (origin pan centre, axes u, w): mass M0 and COM (c0_u, c0_w)."""
    m, d = ep.m, ep.d
    ids = [ep.pan, ep.lid]
    M0 = float(sum(m.body_mass[i] for i in ids))
    com = sum(m.body_mass[i] * d.xipos[i] for i in ids) / M0
    centre = d.site_xpos[m.site("pan_centre").id]
    Rp = d.xmat[ep.pan].reshape(3, 3) @ ep.R0.T
    u = Rp @ ep.p.handle_axis()
    w = np.cross([0, 0, 1.0], u)
    return M0, float((com - centre) @ u), float((com - centre) @ w)


def jaw_centre(ep, s):
    """the claw's grasp point by robot FK: the `<side>_arm_ee_grasp` site that the pre-grasp IK placed on the handle target
    (Episode.tcp is the gripper-base origin; the finger geom centres sit ~32 mm further along the approach)."""
    return ep.d.site_xpos[ep.m.site(f"{s}_arm_ee_grasp").id].copy()


def robot_frame(ep):
    """(u, w, pivot_xy) as the robot knows it after the pinch: arm FK yaw and claw position, MODEL handle geometry."""
    o = ep.observe_boundary("pinch")
    yaw = np.radians(o["yaw_arm_deg"])
    u0 = ep.p.handle_axis()
    u = np.array([np.cos(yaw) * u0[0] - np.sin(yaw) * u0[1], np.sin(yaw) * u0[0] + np.cos(yaw) * u0[1], 0.0])
    w = np.cross([0, 0, 1.0], u)
    p = ep.p
    lever = p.pan_radius + p.pan_wall_t + ep.plan["grasp_frac"] * p.handle_len + ep.r_out   # grasp -> far rim, along u
    piv = jaw_centre(ep, "left")
    piv[:2] -= lever * u[:2]
    piv[2] = p.pan_base_z()
    return u, w, piv


def readings(ep, rng=None, dz=PROBE_DZ):
    """run the pivot probe from the pinch boundary (raise, hold, read, lower, hold) and return the averaged moments.
    Leaves the pan back on the stove.  rng: sensor noise (None = noise-free, for calibration)."""
    u, w, piv = robot_frame(ep)
    Mw, Mu, nb_w, nb_u = [], [], [], []

    def br(e, kind, i, n):
        if kind == "hold" and i >= n - N_READ and len(Mw) < N_READ:
            w6, P = _wrench6(e, "left", rng)
            r = e.d.xpos[e.gb["left"]] - piv
            Mw.append(float(np.cross(r, w6[:3]) @ w + w6[3:] @ w))
            Mu.append(float(np.cross(r, w6[:3]) @ u + w6[3:] @ u))
            lw = np.r_[np.cross(w, r), w]              # M_w = l_w . w6, since (r x F) . w = F . (w x r)
            lu = np.r_[np.cross(u, r), u]
            nb_w.append(float(np.abs(lw @ P).sum() * TAU_NOISE))
            nb_u.append(float(np.abs(lu @ P).sum() * TAU_NOISE))
        return None

    tilt = []
    ep.run_phases([("move", {"left": np.array([0, 0, dz])}, PROBE_T), ("hold", HOLD_T)], branch=br)
    st = ep.pan_state()
    tilt.append(st["tilt"])
    lid_mm = st["lid_mm"]
    ep.run_phases([("move", {"left": np.array([0, 0, -dz])}, PROBE_T), ("hold", 0.3)])
    return dict(M_w=float(np.mean(Mw)), M_u=float(np.mean(Mu)), noise_w=float(np.max(nb_w)), noise_u=float(np.max(nb_u)),
                tilt_deg=float(tilt[0]), lid_mm_at_top=float(lid_mm), frame=dict(u=u.tolist(), pivot=piv.tolist()),
                lid_mm_after=float(ep.pan_state()["lid_mm"]))


def rows(rd, M0, c0_u, c0_w, r_out):
    """readings -> (a, y): a . pi = y for pi = (m, s_u, s_w).  a0 = r_out + c0_u, b0 = c0_w (pivot is r_out behind the centre)."""
    a0, b0 = r_out + c0_u, c0_w
    a = np.array([[G * r_out, G, 0.0], [0.0, 0.0, G]])
    y = np.array([rd["M_w"] - G * M0 * a0, -rd["M_u"] - G * M0 * b0])
    return a, y


# ---------------- calibration (MEASURED eps) ----------------
def _calib_world(rng):
    """a calibration theta: random pose in the prior box, friction, contents anywhere in the disc (or none)."""
    from . import belief as B
    m = float(rng.uniform(0.0, B.PRIOR_M[1])) if rng.uniform() > 0.15 else 0.0
    r, a = B.C_MAX * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    pi = (m, m * r * np.cos(a), m * r * np.sin(a))
    w = B.contents_world(pi)
    return dict(placement=w["placement"], hidden_mass=w["hidden_mass"],
                perturb=dict(dx=float(rng.uniform(-B.PRIOR_DXY, B.PRIOR_DXY)), dy=float(rng.uniform(-B.PRIOR_DXY, B.PRIOR_DXY)),
                             yaw_deg=float(rng.uniform(-B.PRIOR_YAW, B.PRIOR_YAW)), mu_scale=float(rng.uniform(*B.PRIOR_MU)))), pi


def _job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, world, pi = job
    t0 = time.time()
    try:
        from .. import segments as S
        ep = S.Episode(world, observe="zed", obs_seed=S.world_seed(tag), **dict(push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2"))
        if ep.refused:
            return dict(tag=tag, world=world, pi=pi, refused=ep.refused, wall_s=round(time.time() - t0, 1))
        ep.reset_track()
        ep.run_phases(ep.segment_phases("start"))
        ep.run_phases(ep.segment_phases("pinch"))
        M0, c0u, c0w = pan_frame_model(ep)
        rd = readings(ep, rng=None)
        a, y = rows(rd, M0, c0u, c0w, ep.r_out)
        res = (a @ np.asarray(pi, float) - y).tolist()
        return dict(tag=tag, world=world, pi=pi, readings=rd, model=dict(M0=M0, c0_u=c0u, c0_w=c0w), residual=res,
                    U=ep.track["viol"], wall_s=round(time.time() - t0, 1))
    except Exception:
        return dict(tag=tag, world=world, pi=pi, error=traceback.format_exc()[-1500:], wall_s=round(time.time() - t0, 1))


def calibrate(a):
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    jobs = []
    for i in range(a.n):
        w, pi = _calib_world(rng)
        jobs.append((f"C{i:02d}", w, pi))
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [j for j in jobs if j[0] not in done]
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_job, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            print(r["tag"], r.get("residual"), r.get("refused", "")[:60] if r.get("refused") else "", r.get("error", "")[-300:], flush=True)
    rs = [json.loads(p.read_text()) for p in sorted((out / "raw").glob("*.json"))]
    ok = [r for r in rs if r.get("residual")]
    R = np.abs(np.array([r["residual"] for r in ok])) if ok else np.zeros((0, 2))
    summ = dict(n=len(rs), n_used=len(ok), refused=sum(bool(r.get("refused")) for r in rs), errors=sum("error" in r for r in rs),
                eps_model_max=R.max(0).tolist() if len(R) else None, eps_model_p95=np.percentile(R, 95, axis=0).tolist() if len(R) else None,
                label="MEASURED max over n (noise-free, model error only); not a bound", rows=["pitch M_w", "roll M_u"],
                lid_mm_at_top_max=max((r["readings"]["lid_mm_at_top"] for r in ok), default=None),
                tilt_deg_max=max((r["readings"]["tilt_deg"] for r in ok), default=None),
                U_entered=[r["tag"] for r in ok if r.get("U")])
    (out / "calibration.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("calibrate")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=101)
    s.add_argument("--procs", type=int, required=True)
    a = ap.parse_args(argv)
    dict(calibrate=calibrate)[a.cmd](a)


if __name__ == "__main__":
    main()
