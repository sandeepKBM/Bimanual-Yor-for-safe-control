"""validate_hj.py -- grid-resolution agreement (n=201 vs n=301) and closed-loop sign check.

(1) Resolution: for a theta subset, V_201 vs V_301 evaluated at random in-grid states:
    max/mean |dV|, and sign disagreement rate (overall and for |V| > 0.02).
(2) Simulation: from random states, run the greedy fallback argmax_u V(f(x,u)) (n=201
    table, same theta) for 3 s in the continuous dynamics; compare sign(V) with the
    realized min margin. V > 0 must imply no failure (soundness of the safe side);
    V < 0 should imply failure under the greedy fallback (tightness, heuristic).
    python -m world_model.v2.g3_killtest.validate_hj
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.g3_killtest import core as C  # noqa: E402
from world_model.v2.g3_killtest import config as K  # noqa: E402


@njit(cache=True)
def greedy_sim(cid, s0, s1, th0, th1, V, d0, d1, U, T):
    lm = 1e9
    for t in range(T):
        best = -1e9
        bu = 0.0
        for k in range(U.shape[0]):
            y0, y1 = C.step(cid, s0, s1, U[k], th0, th1, C.DT, C.SUB)
            v = C.interp2(V, d0, d1, y0, y1)
            if v > best:
                best = v
                bu = U[k]
        s0, s1 = C.step(cid, s0, s1, bu, th0, th1, C.DT, C.SUB)
        lv = C.margin(cid, s0, s1, th0, th1)
        if lv < lm:
            lm = lv
    return lm


def main():
    rng = np.random.default_rng(7)
    out = {}
    for cls in ("T", "S", "P"):
        cid = K.CLASSES[cls]["cid"]
        sub = 8
        from world_model.v2.g3_killtest.build_values import build
        r1 = build(cls, 201, subset=sub)
        r2 = build(cls, 301, subset=sub)
        res = dict(theta=r1["TH"].tolist(), per_theta=[])
        tot = dict(n=0, sign_dis=0, sign_dis_far=0, n_far=0, maxd=0.0, sumd=0.0,
                   sim_n_pos=0, sim_pos_fail=0, sim_n_neg=0, sim_neg_fail=0, sim_n_band=0)
        for k in range(len(r1["TH"])):
            th0, th1 = r1["TH"][k]
            h0, h1 = C.grid_halfwidths(cid, th0, th1)
            # (1) resolution agreement on random states (inner 90% of the grid)
            X0 = rng.uniform(-0.9 * h0, 0.9 * h0, 4000)
            X1 = rng.uniform(-0.9 * h1, 0.9 * h1, 4000)
            # include the rest line (psi or v exactly 0) which matters for T/S
            X1[:1000] = 0.0
            v1 = np.array([C.interp2(r1["V"][k], r1["AX"][k, 0], r1["AX"][k, 1], a, b) for a, b in zip(X0, X1)])
            v2 = np.array([C.interp2(r2["V"][k], r2["AX"][k, 0], r2["AX"][k, 1], a, b) for a, b in zip(X0, X1)])
            d = np.abs(v1 - v2)
            dis = np.sign(v1) != np.sign(v2)
            far = (np.abs(v1) > 0.02) | (np.abs(v2) > 0.02)
            # (2) closed-loop sign check with the n=201 table
            sidx = np.r_[np.arange(0, 100), np.arange(1000, 1700)]     # 100 rest-line + 700 general states
            lm = np.array([greedy_sim(cid, X0[i], X1[i], th0, th1, r1["V"][k], r1["AX"][k, 0], r1["AX"][k, 1],
                                      C.U_FB, 300) for i in sidx])
            vv = v1[sidx]
            pos = vv > 0.02
            neg = vv < -0.02
            pt = dict(theta=[float(th0), float(th1)], max_dV=float(d.max()), mean_dV=float(d.mean()),
                      sign_dis=float(dis.mean()), sign_dis_far=float(dis[far].mean()) if far.any() else 0.0,
                      sim_pos_fail=float((lm[pos] <= 0).mean()) if pos.any() else None,
                      sim_neg_fail=float((lm[neg] <= 0).mean()) if neg.any() else None,
                      n_pos=int(pos.sum()), n_neg=int(neg.sum()))
            res["per_theta"].append(pt)
            tot["n"] += len(d); tot["sign_dis"] += int(dis.sum()); tot["sign_dis_far"] += int(dis[far].sum())
            tot["n_far"] += int(far.sum()); tot["maxd"] = max(tot["maxd"], float(d.max())); tot["sumd"] += float(d.sum())
            tot["sim_n_pos"] += int(pos.sum()); tot["sim_pos_fail"] += int((lm[pos] <= 0).sum())
            tot["sim_n_neg"] += int(neg.sum()); tot["sim_neg_fail"] += int((lm[neg] <= 0).sum())
        res["summary"] = dict(
            sign_disagree=tot["sign_dis"] / tot["n"], sign_disagree_far=tot["sign_dis_far"] / max(tot["n_far"], 1),
            max_abs_dV=tot["maxd"], mean_abs_dV=tot["sumd"] / tot["n"],
            sim_fail_rate_given_V_pos=tot["sim_pos_fail"] / max(tot["sim_n_pos"], 1),
            sim_fail_rate_given_V_neg=tot["sim_neg_fail"] / max(tot["sim_n_neg"], 1),
            n_pos=tot["sim_n_pos"], n_neg=tot["sim_n_neg"])
        out[cls] = res
        print(cls, json.dumps(res["summary"]), flush=True)
    p = K.OUT / "hj_validation.json"
    p.write_text(json.dumps(out, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
