"""authority.py -- fine-lead and actuator-authority sweep with the FAST quasi-static (QS) model, states taken from MuJoCo
carries. QS agreed with MuJoCo on 97.4% of (state, action-sequence) labels in the main probe, so this is a cheap stand-in for
the MuJoCo oracle (DERIVED-from-QS, not MuJoCo-verified). Scales the fallback set (accel x sa, tilt rate x sw) and the brake limits.
  python -m world_model.v2.testbeds.b_tray_ball.authority
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.b_tray_ball import probe as P
from world_model.v2.testbeds.b_tray_ball import analytic as AN
from world_model.v2.testbeds.b_tray_ball.tray_env import TrayEnv, DT, HX, HY

LEADS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
SCALES = [(1.0, 1.0), (0.4, 0.5), (0.2, 0.25), (0.1, 0.15)]


def sdf_vec(x, y, lip):
    ex = (HX - x) if lip else (HX - np.abs(x)); ey = HY - np.abs(y)
    inside = (ex >= 0) & (ey >= 0)
    ox = np.maximum((x - HX) if lip else (np.abs(x) - HX), 0); oy = np.maximum(np.abs(y) - HY, 0)
    return np.where(inside, np.minimum(ex, ey), -np.hypot(ox, oy))


def main():
    t0 = time.perf_counter()
    infos = [P.run_carry((s, 1.7, 0, None))[0] for s in range(10_000, 10_200)]
    fails = [(i["seed"], i["fail_tick"]) for i in infos if i["fail_tick"] > 30][:48]
    env = TrayEnv(); p = env.p
    states = []
    for seed, ft in fails:
        _, snaps = P.run_carry((seed, 1.7, 0, dict(onset=ft, leads=set(LEADS))))
        for lead, (sn, wl) in snaps.items():
            states.append((seed, lead, sn, wl))
    print("states", len(states), flush=True)
    out = {}
    for sa, sw in SCALES:
        U = P.U85.copy(); U[:, :3] *= sa; U[:, 3:] *= sw
        cls = []
        for seed, lead, sn, wl in states:
            env.restore(sn); pb, vb, wb = P.ball_world(env)
            R = env._R(env.quat); loc0 = (pb - env.pos) @ R; vrel = R.T @ (vb - env.v - np.cross(wl, pb - env.pos))
            seqs = P.lib_sequences(env.pos.copy(), env.v.copy(), env.quat.copy(), U, P.D_HOLD, P.K_H, 5.0 * sa, 2.0 * sw)
            fail = np.zeros(len(seqs), bool)
            for i in range(len(seqs)):
                xs, ys = AN.qs_rollout(loc0[0], loc0[1], vrel[0], vrel[1], env.pos.copy(), env.v.copy(), env.quat.copy(), seqs[i],
                                       p.radius, p.inertia_k, p.mu_roll, 4, 1.5, DT)
                fail[i] = (sdf_vec(xs, ys, False) < 0).any()
            n = len(U); h = fail[n]
            cls.append("hold_ok" if not h else ("savable" if ((~fail[:n]).any() or not fail[n + 1]) else "not_certified"))
        cls = np.array(cls); leads = np.array([s[1] for s in states]); seeds = np.array([s[0] for s in states])
        by = {int(L): float(np.mean(cls[leads == L] == "savable")) for L in LEADS}
        byn = {int(L): float(np.mean(cls[leads == L] == "not_certified")) for L in LEADS}
        # per-episode last lead where the state is NOT savable-or-hold_ok  -> first-unsavable lead = smallest lead whose state is certified
        ok_leads = []
        for s in sorted(set(seeds)):
            m = [(l, c) for (l, c, sd) in zip(leads, cls, seeds) if sd == s]
            good = [l for l, c in m if c in ("savable", "hold_ok")]
            ok_leads.append(min(good) if good else 999)
        out[f"sa{sa}_sw{sw}"] = dict(frac={k: float(np.mean(cls == k)) for k in ("hold_ok", "savable", "not_certified")},
                                      savable_by_lead=by, not_certified_by_lead=byn,
                                      last_certified_lead_median=float(np.median(ok_leads)),
                                      frac_eps_never_certified=float(np.mean(np.array(ok_leads) == 999)))
        print(f"scale a x{sa} w x{sw}:", out[f"sa{sa}_sw{sw}"], flush=True)
    P.merge_json("authority_sweep_QS", out)
    print("wall", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
