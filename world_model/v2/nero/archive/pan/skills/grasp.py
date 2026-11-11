"""pan/skills/grasp.py -- grasp-candidate sets for the approach skill (RoboTwin pattern: annotated contact points x a rotation
sweep about the contact point, here filtered by IK + exact clearance along the whole planned path).  SIMULATION ONLY.

v1 used ONE grasp per claw (left: handle at grasp_frac, approach along the handle from its end, jaw vertical; right: one point
at the stove edge, horizontal approach from the far side).  v2 sweeps candidates and keeps the feasible one with the best
clearance (not RoboTwin's "first reachable").

    python -m world_model.v2.nero.pan.skills.grasp coverage --out DIR --set full|iid|overload --procs P [--tags ...]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import itertools
import json
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

# candidate grids (design; ASSUMED ranges, raised 10-30)
LEFT_FRAC = (0.65, 0.55, 0.75, 0.45, 0.85)          # handle contact points (fraction of the handle length from the wall)
LEFT_YAW = (0.0, -15.0, 15.0, -30.0, 30.0)           # approach rotation about the jaw axis (vertical) [deg]
LEFT_ROLL = (0.0, -20.0, 20.0)                       # jaw tilt about the approach axis [deg]
RIGHT_DX = (0.0, -0.02, 0.02, -0.04, 0.04)           # clamp point shift along the stove edge [m]
RIGHT_YAW = (0.0, -15.0, 15.0, -30.0, 30.0)          # approach rotation about the vertical [deg]
RIGHT_PITCH = (0.0, 20.0)                            # approach tilted down [deg] (fingers stay above / below the stack)
N_SOL, SEEDS = 8, 48                                 # IK solutions / seeds per VARIANT; the v1 grasp keeps v1's 16 / 120


def _rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def left_candidates(m, d, p, aim, Rz):
    from world_model.v2.nero.pan.handle_grasp import LEFT_APPROACH, LEFT_JAW, handle_point
    pan = m.body("pan").id
    for frac, yaw, roll in itertools.product(LEFT_FRAC, LEFT_YAW, LEFT_ROLL):
        # handle point in the NOMINAL pan pose, then moved by the robot's aim (estimate), like v1's aim(handle_nom)
        tgt = aim(_handle_nominal(m, p, frac))
        app0, jaw0 = Rz @ LEFT_APPROACH, Rz @ LEFT_JAW
        Ry = _rot(jaw0, yaw)
        app, jaw = Ry @ app0, jaw0
        Rr = _rot(app, roll)
        yield dict(frac=frac, yaw=yaw, roll=roll), tgt, app, Rr @ jaw


_HN = {}


def _handle_nominal(m, p, frac):
    """handle centre-line point at `frac` in the nominal settled pan pose (scene formula, MODEL)."""
    ax = p.handle_axis()
    loc = ax * (p.pan_radius + p.pan_wall_t + frac * p.handle_len) + np.array([0, 0, p.pan_bottom_t + p.pan_wall_h - p.handle_t / 2])
    key = (id(m), frac)
    if key not in _HN:
        _HN[key] = loc
    return _NOM_POSE[0] + _NOM_POSE[1] @ loc


_NOM_POSE = [None, None]


def right_candidates(target_R0):
    from world_model.v2.nero.pan.under_rim import RIGHT_APPROACH, RIGHT_JAW
    for dx, yaw, pitch in itertools.product(RIGHT_DX, RIGHT_YAW, RIGHT_PITCH):
        tgt = target_R0 + np.array([dx, 0.0, 0.0])
        Rv = _rot([0, 0, 1], yaw)
        app = Rv @ RIGHT_APPROACH
        side = np.cross(app, [0, 0, 1.0])
        Rp = _rot(side, pitch)
        yield dict(dx=dx, yaw=yaw, pitch=pitch), tgt, Rp @ app, Rp @ RIGHT_JAW


NOMINAL = dict(left=dict(frac=0.65, yaw=0.0, roll=0.0), right=dict(dx=0.0, yaw=0.0, pitch=0.0))


def _ik_budget(c, side):
    return dict(n_sol=16, seeds=120) if c == NOMINAL[side] else dict(n_sol=N_SOL, seeds=SEEDS)


def search(ep_like, which="both"):
    """-> dict with the v2 grasp per claw (or the reason none exists).  ep_like: an Episode built with est = the aim."""
    import mujoco
    from world_model.v2.nero.pan import segments as S
    from world_model.v2.nero.pan.handle_grasp import FINGER_OPEN, LIFT, _set_fingers
    from world_model.v2.nero.pan.under_rim import APPROACH_IN, EDGE_CLEAR
    from world_model.v2.nero.pan import observe_pose as O
    m, p = ep_like.m, ep_like.p
    cl = S.CLAMP["v2"]
    d = mujoco.MjData(m)
    d.qpos[:] = ep_like.d.qpos
    for s_ in ("left", "right"):                        # both arms at the model home pose (v1 may have left them at its pre-grasps)
        d.qpos[ep_like.addr[s_]] = m.qpos0[ep_like.addr[s_]]
    mujoco.mj_forward(m, d)
    est = ep_like.obs["est"]
    aim, Rz = O.aim_transform(est, p.pan_centre_xy)
    _NOM_POSE[0], _NOM_POSE[1] = ep_like.pos0, ep_like.R0
    push, u = ep_like.plan["push"], Rz @ p.handle_axis()
    out = dict(left=None, right=None, n_tried=dict(left=0, right=0))
    best = None
    for c, tgt, app, jaw in left_candidates(m, d, p, aim, Rz):
        out["n_tried"]["left"] += 1
        try:
            r = S.find_pregrasp_clear(m, d, "left", tgt, app, jaw,
                                      [(-push * u, p.handle_t / 2), (-push * u + np.array([0, 0, LIFT]), p.handle_t / 2)],
                                      cl["clear"], FINGER_OPEN, jm=cl["jm"]["left"], **_ik_budget(c, "left"))
        except RuntimeError as e:
            continue
        sc = min(r["clear_mm"] / (1e3 * cl["clear"]), r["path_joint_margin"] / cl["jm"]["left"])
        if best is None or sc > best[0]:
            best = (sc, c, r)
        if c == NOMINAL["left"]:
            break                                        # the v1 grasp is feasible: keep it (same behaviour as v1 there)
    if best is None:
        out["why"] = "no left grasp candidate"
        return out
    out["left"] = dict(cand=best[1], clear_mm=best[2]["clear_mm"], jm=best[2]["path_joint_margin"], score=best[0])
    d.qpos[ep_like.addr["left"]] = best[2]["q"]
    _set_fingers(m, d, "left", FINGER_OPEN)
    mujoco.mj_forward(m, d)
    stove_edge = p.pan_centre_xy[1] - S.S.ASSUMED["stove_half"][1]
    c_after = aim(np.array([p.pan_centre_xy[0], p.pan_centre_xy[1], 0.0])) - push * u
    z_mid = p.pan_base_z() + 0.5 * (p.pan_bottom_t + p.pan_wall_h + S.CV.ASSUMED_COVERED["lid_t"])
    tR0 = np.array([c_after[0], stove_edge - EDGE_CLEAR - cl["tip"], z_mid + cl["dz"]])
    bestR = None
    for c, tgt, app, jaw in right_candidates(tR0):
        out["n_tried"]["right"] += 1
        try:
            r = S.find_pregrasp_clear(m, d, "right", tgt - APPROACH_IN * app, app, jaw,
                                      [(APPROACH_IN * app, cl["open_R"]), (APPROACH_IN * app + np.array([0, 0, LIFT]), 0.5 * S.STACK_H)],
                                      cl["clear"], cl["open_R"], jm=cl["jm"]["right"], **_ik_budget(c, "right"))
        except RuntimeError:
            continue
        sc = min(r["clear_mm"] / (1e3 * cl["clear"]), r["path_joint_margin"] / cl["jm"]["right"])
        if bestR is None or sc > bestR[0]:
            bestR = (sc, c, r)
        if c == NOMINAL["right"]:
            break
    if bestR is None:
        out["why"] = "no right grasp candidate"
        return out
    out["right"] = dict(cand=bestR[1], clear_mm=bestR[2]["clear_mm"], jm=bestR[2]["path_joint_margin"], score=bestR[0])
    return out


def _cov_job(job):
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    tag, w = job
    t0 = time.time()
    try:
        from world_model.v2.nero.pan import segments as S
        pt = w["perturb"]
        est = dict(dx=pt.get("dx", 0.0), dy=pt.get("dy", 0.0), yaw_deg=pt.get("yaw_deg", 0.0))   # aim = TRUE pose (kinematic coverage)
        ep = S.Episode(w, observe="zed", obs_seed=S.world_seed(tag), est=est, push=0.07, grasp_frac=0.65, grip_force=10.0, clamp="v2")
        r = dict(tag=tag, world=w, v1_refused=ep.refused)
        r["v2"] = search(ep)
    except Exception:
        r = dict(tag=tag, world=w, error=traceback.format_exc()[-1500:])
    r["wall_s"] = round(time.time() - t0, 1)
    return r


def main(argv=None):
    from world_model.v2.nero.pan.segcert import worlds
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("coverage")
    s.add_argument("--out", required=True)
    s.add_argument("--set", default="full", choices=["full", "iid", "overload"])
    s.add_argument("--tags", default=None)
    s.add_argument("--procs", type=int, default=1)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    ws = worlds(a.set)
    if a.tags:
        keep = a.tags.split(",")
        ws = [(t, w) for t, w in ws if t in keep]
    done = {p.stem for p in (out / "raw").glob("*.json")}
    jobs = [(t, w) for t, w in ws if t not in done]
    with get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_cov_job, jobs):
            (out / "raw" / f"{r['tag']}.json").write_text(json.dumps(r, default=float))
            v2 = r.get("v2") or {}
            print(r["tag"], "v1:", (r.get("v1_refused") or "ok")[:40], "| v2:", v2.get("why") or "ok",
                  (v2.get("left") or {}).get("cand"), (v2.get("right") or {}).get("cand"), v2.get("n_tried"),
                  r.get("error", "")[-300:], r["wall_s"], flush=True)
    (out / f"_DONE_{a.set}").write_text("ok")


if __name__ == "__main__":
    main()
