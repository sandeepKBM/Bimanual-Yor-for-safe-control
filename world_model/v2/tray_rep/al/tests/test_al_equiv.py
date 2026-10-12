"""al/tests/test_al_equiv.py -- the AL reference arms reproduce the rebaseline F5P arms record-for-record (same VAL carries): AL_oracle = F5P arm=oracle, AL_unif = arm=unif,
AL_point = arm=point, AL_static = arm=conf,online=0, AL_online0:ivmode=central = arm=conf (lam 0.1, sa 0.04: pass lam/sa explicitly to both). SIMULATION ONLY. CPU.
  python -m world_model.v2.tray_rep.al.tests.test_al_equiv [--n 3] [--regime normal|stressed]
"""
from __future__ import annotations

import argparse

from world_model.v2.tray_rep.al import run_al as RA
from world_model.v2.tray_rep.boed import plan_cl as PC
from world_model.v2.tray_rep.boed import run_cl as RCL
from world_model.v2.tray_rep.wrongpoint import run_cl_wp as WP

KEYS = ("dropped", "n_ticks", "goal_tick", "n_interv", "t_complete_s", "cl_t_arrive_s", "mu", "x_f5_backup", "x_f5_true", "x_f5_false", "x_f5_unknown", "x_p_lo_end", "x_p_hi_end", "x_p_cov_end",
        "x_p_lo_t1s", "x_p_hi_t1s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--regime", default="normal")
    a = ap.parse_args()
    R = RA.REGIMES[a.regime]
    F5P = "calib_file=v2,eps_a=0,lam=0.1,sa=0.04"
    pairs = [("AL_oracle", "F5P:arm=oracle," + F5P), ("AL_unif:lam=0.1,sa=0.04", "F5P:arm=unif," + F5P), ("AL_point:rho={rho},lam=0.1,sa=0.04", "F5P:arm=point,rho={rho},pseed=0," + F5P),
             ("AL_static:rho={rho}", "F5P:arm=conf,rho={rho},pseed=0,online=0," + F5P),
             ("AL_online0:rho={rho},ivmode=central,lam=0.1,sa=0.04", "F5P:arm=conf,rho={rho},pseed=0," + F5P)]
    ok = True
    for al, f5p in pairs:
        al, f5p = al.format(rho=R["rho"]), f5p.format(rho=R["rho"])
        cfgw = dict(filt=f5p, I=0.99, setup="piper_trk", seed=0, split="val", omega=PC.OMEGA, e_lo=PC.E_LO, e_hi=PC.E_HI, data=RA.DATA, cell="x", arm_key="x", rho=None, pseed=0,
                    mu_range=R["mu_range"], auth=R["auth"], wrong=(1.0, 2.0))
        WP.G.clear(); WP._init(cfgw)
        ref = [WP._work(ep) for ep in range(a.n)]
        cfg = dict(policy=al, regime=a.regime, auth=R["auth"], mu_range=R["mu_range"], rho=R["rho"], setup="piper_trk", seed=0, split="val", data=RA.DATA, value="placeholder", voi="",
                   log_ll=False, omega=PC.OMEGA, e_lo=PC.E_LO, e_hi=PC.E_HI, wrong=(1.0, 2.0))
        RA.G.clear(); RA._init(cfg)
        mine = [RA._work(ep) for ep in range(a.n)]
        bad = []
        for r, m in zip(ref, mine):
            for k in KEYS:
                if r.get(k) != m.get(k) and not (r.get(k) != r.get(k) and m.get(k) != m.get(k)):
                    bad.append((r["ep"], k, r.get(k), m.get(k)))
        print(f"{al:55s} vs {f5p[:34]:34s}: {'IDENTICAL' if not bad else 'DIFF ' + str(bad[:4])}")
        ok &= not bad
    print("EQUIVALENCE", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
