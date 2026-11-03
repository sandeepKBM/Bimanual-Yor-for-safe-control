"""democ/grasp_cert_calib.py -- identify the contact constants of the compliant grasp certificate from short MuJoCo probes.  SIM ONLY, CPU.

Scene facts (read from the compiled model): opt.cone = elliptic (1), impratio 10, dt 5 ms, Newton solver.  Box geom condim 4 solref (0.005,1)
solimp (.95,.99,.001,.5,2); pedestal condim 3 solref (0.02,1) solimp (.9,.95,.001,.5,2); flat finger pads condim 3 solref (0.005,1) solimp (.9,.95,.001,.5,2);
solmix default => MuJoCo averages the two geoms' solref/solimp.  Finger joint2 = -joint1 is a SOFT equality (solref 0.005 1, solimp .99 .999 1e-5).
Probes (arms kinematic as in the cross-check, a known external force on the box via xfrc_applied, 200 steps per level):
  support_probe : box on the pedestal, hands open.  Fz sweep -> normal stiffness per corner (slope of F vs dz / 4 corners);  Fx sweep -> x(t) after the
                  step: if the tangential contact were a spring x would saturate; it ramps linearly (creep) -> damping per corner ct = F / (4 v).
  pad_statistics: from a cross-check json (rows carry the MuJoCo contact list at release+0.3 s): median |f_n| / penetration per class (pedestal, top pad,
                  bottom pad).  NOT a clean spring constant (pad classes are bimodal, see report) -- it documents why pad kn / kappa are NOT identified.
    python -m world_model.v2.nero.democ.grasp_cert_calib [--xcheck rows.json] [--out ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from . import kin as K
from . import grasp_cert_xcheck as X


def support_probe(mu=0.4):
    cache = {}
    p, m, pre = X.build(mu, 0.0, cache, "flat")
    d = mujoco.MjData(m)
    K.set_state(m, d, p, pre["ql"], pre["qr"], K.FINGER_OPEN)
    qa = {s: A.arm_addrs(m, s) for s in C.SIDES}
    hold = {"left": pre["ql"].copy(), "right": pre["qr"].copy()}
    act = {s: m.actuator(f"{s}_arm_gripper_pos").id for s in C.SIDES}
    box = m.body("democ_box").id

    def step(n, F=None):
        for _ in range(n):
            for s in C.SIDES:
                d.ctrl[act[s]] = K.FINGER_OPEN
            d.xfrc_applied[box] = 0 if F is None else F
            mujoco.mj_step(m, d)
            for s in C.SIDES:
                d.qpos[qa[s][0]] = hold[s]; d.qvel[qa[s][1]] = 0.0
            mujoco.mj_forward(m, d)
    step(200)
    z0 = d.xpos[box].copy(); out = dict(mass=float(m.body_mass[box]), n_corners=4)
    pts = []
    for Fz in (-2.0, -5.0, -10.0, -20.0):
        step(200, np.array([0, 0, Fz, 0, 0, 0.]))
        pts.append((Fz, float(d.xpos[box][2] - z0[2])))
    step(200)
    out["fz_dz"] = pts
    out["kn_per_corner_N_per_m"] = float(np.polyfit([z for _, z in pts], [f for f, _ in pts], 1)[0] / 4)
    x0 = d.xpos[box].copy(); rows = []
    for Fx in (0.5, 1.0, 1.5):
        step(1); xs = []
        for _ in range(10):
            step(40, np.array([Fx, 0, 0, 0, 0, 0.])); xs.append(float(1e3 * (d.xpos[box][0] - x0[0])))
        v = (xs[-1] - xs[4]) / (5 * 40 * X.DT) * 1e-3                       # steady velocity [m/s] over the second half
        rows.append(dict(F=Fx, x_mm_per_0p2s=xs, v_m_s=v, ct_per_corner=Fx / v / 4))
        step(200); x0 = d.xpos[box].copy()
    out["creep"] = rows
    out["ct_per_corner_Ns_per_m"] = float(np.mean([r["ct_per_corner"] for r in rows]))
    out["tangential_is_spring"] = False                                      # x(t) is a ramp (constant velocity), not a saturating curve
    return out


def pad_statistics(path):
    d = json.loads(Path(path).read_text())
    ks = {}
    for r in d["rows"]:
        if r["scenario"] != "B":
            continue
        for f in r["sim_forces_t0p3"] or []:
            fn, pen = abs(f["f_box"][2]), -f["dist"]
            if pen > 1e-6 and fn > 0.3:
                key = "pedestal" if "pedestal" in f["geom"] else ("pad_top" if f["pos"][2] > 0 else "pad_bottom")
                ks.setdefault(key, []).append(fn / pen)
    return {k: dict(n=len(v), median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75))) for k, v in ks.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xcheck", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "democ_grasp_cert_calib_2026-10-26.json"))
    a = ap.parse_args(argv)
    res = dict(support=support_probe())
    if a.xcheck:
        res["pad_force_over_penetration"] = pad_statistics(a.xcheck)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
