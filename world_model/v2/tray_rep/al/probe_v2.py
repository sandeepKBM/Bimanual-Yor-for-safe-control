"""al/probe_v2.py -- the pickup probe for the AL policies: set certificate (boed.p7.SetCertificate, robot-computable) + the fixed tilt probe (boed.probe.ProbeController)
+ the probe likelihood on the v2 mu_eff map (belief_v2.probe_ll_v2). SIMULATION ONLY. CPU. Nothing here reads the hidden mu: it acts on a boed.episode.Run through
obs / renv state, like Run.probe, except that Run.probe updates a belief with the OLD map (boed.model.MU_EFF) and this does not.

    prepare(run)                         probe ball filter from obs[0] (as P3/P7)
    certify(run, theta, mu_w)  -> dict   SetCertificate.check at the worst-case mu mu_w (the lower end of the CURRENT mu set), tilt theta [rad]
    probe(run, theta)          -> dict   runs the probe (run.tick), returns ll (40,) on the v2 map, ticks, dropped, reason, n_hold
The probe tilts about the tray y axis (the ball rolls along x, toward the centre): ProbeController / SetCertificate hard-code that; a y-direction variant needs both
generalised (not done: the ball model is isotropic, so the information is the same by symmetry; only the geometry of the tray edge differs).
"""
from __future__ import annotations

from dataclasses import replace

from world_model.v2.tray_rep.al import belief_v2 as BV
from world_model.v2.tray_rep.boed import probe as PB
from world_model.v2.tray_rep.boed import p7 as P7

_CERTS = {}


def spec_for(ctx, theta):
    return replace(ctx.spec, theta=float(theta))


def prepare(run):
    c = run.c
    run.kf = PB.make_kf(c.trk, c.noisefree)
    run.kf.reset_prior(sx=0.006, sv=0.01)
    run.kf.feed(run.obs[0], 0)


def certify(run, theta, mu_w):
    c = run.c
    key = (id(c.renv), round(float(theta), 6))
    if key not in _CERTS:
        _CERTS[key] = P7.SetCertificate(c.renv, spec_for(c, theta), c.p7)
    res = _CERTS[key].check(run.kf, run.obs[0], float(mu_w))
    return dict(ok=bool(res["ok"]), dmin=float(res["dmin"]), stop_worst=int(res["stop_worst"]), ms=float(res["ms"]), mu_w=float(mu_w))


def probe(run, theta):
    c = run.c
    spec = spec_for(c, theta)
    ctrl = PB.ProbeController(spec, run.kf, c.renv.w_max, False)
    n0 = run.n
    guard = spec.n_settle_max + spec.n_pre + spec.n_hold_max + spec.n_bal_max + 5
    for _ in range(guard):
        run.tick(ctrl.step(c.renv.env.axis(), run.n))
        if run.dropped or ctrl.done:
            break
    run.x["n_probe"] += run.n - n0
    run.info["probe_run"] = True
    run.info["probe_drop"] = bool(run.dropped)
    out = dict(ll=None, ticks=int(run.n - n0), dropped=bool(run.dropped), reason=ctrl.reason, n_hold=int(ctrl.n_hold), theta=float(theta))
    if not run.dropped and ctrl.t_hold_end > 0:
        t0, t1 = ctrl.window()
        out["ll"] = BV.probe_ll_v2(run.obs, t0, t1, c.sigma, c.lat, ctrl.prior, c.noisefree)
        run.info.update(probe_hold=int(ctrl.n_hold), probe_reason=ctrl.reason, probe_settled=bool(ctrl.settled_end))
    run.kf = None
    return out
