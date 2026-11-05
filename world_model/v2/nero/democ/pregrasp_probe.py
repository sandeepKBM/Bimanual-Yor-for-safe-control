"""pregrasp_probe.py -- score Demo C pre-grasp candidates for the NYU gripper (SIMULATION ONLY, CPU).

A candidate = symmetric per-hand parameters (both hands mirror each other across the box's y ends):
  inset    [m]   gripper site depth inside the box end face along the approach (kin.INSET = 0.02 is today's value; < 0 = outside the face)
  open     [-]   jaw opening fraction (1 = fully open, gap 82 mm at the TCP; gripper.opening_for_width)
  roll_deg       rotation of the jaw axis about the approach axis, from box z
  tilt_deg       approach pitched about world x toward -z (0 = horizontal approach along -/+y; > 0 = coming down onto the end edge)
  dz       [m]   site height offset from the box centre
Checks (all MODEL, from the compiled MJCF):
  ik      both hands reach the pose (collect_pose residual <= 1e-6), any jaw sign
  free    with the jaws at `open`: no robot contact (kin.bad_contacts) and arm-arm proxy h >= 0.03 m (find_pregrasp's rule)
  grips   with the jaws fully closed (opening 0): BOTH jaws of EACH hand touch the box  -> the box is between the jaws
Score = min joint-limit margin [rad] over the best passing (L, R) pair.
Usage: python -m world_model.v2.nero.democ.pregrasp_probe --cands cands.json --out out.json [--procs 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import mujoco
import numpy as np

from world_model.v2.nero import adapters as A, config as C, gripper as G
from world_model.v2.nero.democ import kin as K, scene as S

IK_TOL = 1e-6     # MODEL (collect_pose ok threshold)
H_MIN = 0.03      # as find_pregrasp
N_SOL, SEEDS = 6, 25   # reduced from find_pregrasp's 12 / 60 for speed (fewer IK branches tried -> can only miss solutions, not admit bad ones)

_M = {}


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def targets(p, cand, jaw_sign):
    c = np.array(p.box_centre, float)
    h = p.box_half[1] - cand["inset"]
    tilt, roll = np.radians(cand.get("tilt_deg", 0.0)), np.radians(cand.get("roll_deg", 0.0))
    out = {}
    for side, sy, js in (("left", +1.0, jaw_sign[0]), ("right", -1.0, jaw_sign[1])):
        ap = np.array([0.0, -sy, 0.0])
        jaw = np.array([0.0, 0.0, js])
        # tilt: rotate approach + jaw about world x so the approach points down toward the box end (mirror for the two ends)
        Rt = _rx(sy * tilt)
        ap, jaw = Rt @ ap, Rt @ jaw
        # roll: rotate the jaw axis about the approach
        jaw = np.cos(roll) * jaw + np.sin(roll) * np.cross(ap, jaw)
        R = K.grasp_rot(ap, jaw)
        pos = c + np.array([0.0, sy * h, cand.get("dz", 0.0)])
        out[side] = (pos, R)
    return out


def _model():
    if "m" not in _M:
        p = S.DemoCParams()
        m = S.compose(p)
        _M.update(p=p, m=m, d=mujoco.MjData(m), pb=A.PairBarriers(m), pik={s: K.PoseIK(m, s) for s in C.SIDES},
                  box=m.geom("democ_box_geom").id,
                  jaws={s: (set(G.col_geoms(m, s, "finger1")), set(G.col_geoms(m, s, "finger2"))) for s in C.SIDES})
    return _M


def _grips(M, ql, qr):
    m, d, p = M["m"], M["d"], M["p"]
    K.set_state(m, d, p, ql, qr, 0.0)
    mujoco.mj_forward(m, d)
    touch = {s: [False, False] for s in C.SIDES}
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if M["box"] not in (g1, g2):
            continue
        g = g2 if g1 == M["box"] else g1
        for s in C.SIDES:
            for k in (0, 1):
                if g in M["jaws"][s][k]:
                    touch[s][k] = True
    return all(all(v) for v in touch.values()), touch


def evaluate(cand):
    M = _model()
    m, d, p = M["m"], M["d"], M["p"]
    res = dict(cand=cand, ik=False, n_pairs=0, n_free=0, n_pass=0, best=None, fail_contacts={})
    sols = {}
    for js in (+1.0, -1.0):
        tg = targets(p, cand, (js, js))
        for s in C.SIDES:
            sols[(s, js)] = [(q, e) for q, e in K.collect_pose(M["pik"][s], tg[s][0], tg[s][1], N_SOL, SEEDS,
                                                                 seed=0 if s == "left" else 1, ok=IK_TOL)]
    res["n_ik"] = {f"{s}{int(j):+d}": len(v) for (s, j), v in sols.items()}
    res["ik"] = any(sols[("left", j)] for j in (1.0, -1.0)) and any(sols[("right", j)] for j in (1.0, -1.0))
    for jl in (1.0, -1.0):
        for jr in (1.0, -1.0):
            for ql, _ in sols[("left", jl)]:
                for qr, _ in sols[("right", jr)]:
                    res["n_pairs"] += 1
                    K.set_state(m, d, p, ql, qr, cand["open"])
                    bad = K.bad_contacts(m, d)
                    if bad:
                        for pr in set(bad):
                            k = "|".join(pr)
                            res["fail_contacts"][k] = res["fail_contacts"].get(k, 0) + 1
                        continue
                    h = M["pb"].min_h(d)
                    if h < H_MIN:
                        continue
                    res["n_free"] += 1
                    ok, touch = _grips(M, ql, qr)
                    if not ok:
                        continue
                    res["n_pass"] += 1
                    score = float(min(M["pik"]["left"].margin(ql)[0], M["pik"]["right"].margin(qr)[0]))
                    if res["best"] is None or score > res["best"]["margin"]:
                        res["best"] = dict(margin=score, h=float(h), jaw_sign=(jl, jr), ql=ql.tolist(), qr=qr.tolist())
    res["fail_contacts"] = dict(sorted(res["fail_contacts"].items(), key=lambda kv: -kv[1])[:5])
    res["passed"] = res["n_pass"] > 0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    cands = json.load(open(a.cands))
    if a.procs > 1:
        with mp.Pool(a.procs, maxtasksperchild=3) as pool:
            out = pool.map(evaluate, cands, chunksize=1)
    else:
        out = [evaluate(c) for c in cands]
    json.dump(dict(rule=dict(ik_tol=IK_TOL, h_min=H_MIN, n_sol=N_SOL, seeds=SEEDS, grips="both jaws of each hand touch the box when fully closed"),
                   results=out), open(a.out, "w"), indent=1)
    for r in out:
        b = r["best"]
        print(json.dumps(r["cand"]), "ik", r["ik"], "pairs", r["n_pairs"], "free", r["n_free"], "pass", r["n_pass"],
              "margin", None if b is None else round(b["margin"], 3), flush=True)


if __name__ == "__main__":
    main()
