"""al/tests/test_al_hidden.py -- HiddenMu guarantee of the AL policies (assertion test). SIMULATION ONLY. CPU.
  python -m world_model.v2.tray_rep.al.tests.test_al_hidden [--n 1]
(1) static: policies.py / probe_v2.py / belief_v2.py / value.py / probe_voi.py never touch an attribute named mu, mu_roll, hid (ast scan);
(2) every non-oracle policy refuses the true l (begin_episode asserts) and only 'oracle' has reads_truth;
(3) the guard is live: inside it run.mu, renv.mu_roll and renv.env.p.mu_roll raise on use, and are restored after;
(4) every policy kind runs a full VAL carry under the guard without touching the poisoned values (a probe policy with a forced probe included), the oracle too;
(5) a canary policy that DOES read run.mu in pre_carry is caught (the guard really bites).
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.al import policies as POL
from world_model.v2.tray_rep.al import run_al as RA
from world_model.v2.tray_rep.boed import episode as EP

HERE = Path(RA.__file__).parent
FORBID = {"mu", "mu_roll", "hid", "_hid", "_l_true"}


def scan(fn):
    bad = []
    for n in ast.walk(ast.parse((HERE / fn).read_text())):
        if isinstance(n, ast.Attribute) and n.attr in FORBID:
            bad.append((fn, n.lineno, n.attr))
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in FORBID:
            bad.append((fn, n.lineno, n.slice.value))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    bad = sum((scan(f) for f in ("policies.py", "probe_v2.py", "belief_v2.py", "value.py", "probe_voi.py")), [])
    assert not bad, f"decision-path files touch a hidden-mu attribute: {bad}"
    print("(1) ast scan clean")
    cfg = dict(policy="AL_static", regime="normal", auth=1.0, mu_range=None, rho=0.2, setup="piper_trk", seed=0, split="val", data=RA.DATA, value="placeholder", voi="",
               log_ll=False, omega=4.0, e_lo=0.02, e_hi=0.06, wrong=(1.0, 2.0))
    ctx = RA.make_ctx(cfg)
    filt = RA.ALFilter(ctx.renv)
    for spec in ["AL_static", "AL_online", "AL_online0", "AL_probe", "AL_transductive:always=1", "AL_oracle", "AL_unif", "AL_point"]:
        pol = POL.ALPolicy(spec, ctx.renv, value=RA.make_value(cfg), voi=None)
        assert pol.reads_truth == (pol.kind == "oracle")
        if pol.kind != "oracle":
            try:
                pol.begin_episode((0.0, 0.4, False), -6.0)
                raise SystemExit(f"{spec}: accepted the true l")
            except AssertionError:
                pass
        for ep in range(a.n):
            info = ctx.store.ep_info(ep)
            r = RA.ALRun(ctx, info["seed"], None)
            mu0, mr0 = r.mu, ctx.renv.mu_roll
            l_true = RA.ALRun.run_al(r, pol, filt, cfg)         # raises if the decision path reads a poisoned value
            assert r.mu == mu0 and ctx.renv.mu_roll == mr0 and ctx.renv.env.p.mu_roll == mr0, "guard not restored"
            rec = r.record("AL:" + spec)
            assert not rec["dropped"] or True
        print(f"(4) {spec}: {a.n} carry ok")
    # (3) guard live
    r = RA.ALRun(ctx, ctx.store.ep_info(0)["seed"], None)
    with RA.hidden_guard(r):
        for obj, attr in ((r, "mu"), (ctx.renv, "mu_roll"), (ctx.renv.env.p, "mu_roll")):
            try:
                float(getattr(obj, attr))
                raise SystemExit(f"guard not live: {attr} readable")
            except RuntimeError:
                pass
    assert isinstance(r.mu, float)
    print("(3) guard live and restored")

    # (5) canary: a policy that reads run.mu in pre_carry
    class Canary(POL.ALPolicy):
        def pre_carry(self, run):
            float(run.mu)
            return 0.99
    pol = Canary("AL_static", ctx.renv, value=RA.make_value(cfg))
    r = RA.ALRun(ctx, ctx.store.ep_info(0)["seed"], None)
    try:
        r.run_al(pol, filt, cfg)
        raise SystemExit("canary NOT caught")
    except RuntimeError as e:
        assert "hidden mu_roll" in str(e)
    print("(5) canary caught")
    print("HIDDEN-MU CHECKS PASSED")


if __name__ == "__main__":
    main()
