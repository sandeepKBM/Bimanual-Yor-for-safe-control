"""tamp/people_pred.py -- time-indexed PEOPLE PREDICTION for the segment planner, and the timed-wait probe.  SIMULATION ONLY.

predict_people(bel, t): the belief with every person moved to its predicted position at time t from now,  c_i(t) = c_i + v_hat_i t, and its position error grown by
the velocity-estimate error, e_p_i += (e_v + e_v_i) t.  The gatekeeper rows then see (nominal + backup of a segment, accel disc growth from the SEGMENT start,
unchanged) a person centred at c_i(t_start) with radius r0 + e_p_i + ....

ASSUMPTIONS (read before trusting a "certified"):
  * constant-velocity prediction: v_hat is the belief's people_vel (sim: vel_source "state" = the walker's true current velocity).  A person who changes velocity
    between now and t (starts, stops, turns, yields) is NOT covered -- that is the gatekeeper's per-tick job (T1 is unchanged);
  * NO a_p t^2/2 growth is added to t_start: planning here is FEASIBILITY ("is there a time-consistent crossing under the current track"); safety is the gatekeeper's
    per-tick check from the actual state, which re-certifies every segment at its boundary against the then-current belief;
  * velocity-error growth e_v * t IS added (a plan 10 s ahead should not trust the velocity to the cm).
"""
from __future__ import annotations

import numpy as np


def predict_people(bel, t: float, ev_scale: float = 1.0):
    """Belief with people at c + v_hat t and e_p grown by e_v t.  t <= 0 or no people -> bel unchanged (identity)."""
    t = float(t)
    if t <= 0.0 or not len(bel.people):
        return bel
    n = len(bel.people)
    vel = bel.people_vel if len(bel.people_vel) == n else tuple(np.zeros(2) for _ in range(n))
    ppl = tuple(np.asarray(c, float) + np.asarray(v, float)[:2] * t for c, v in zip(bel.people, vel))
    ep0 = bel.people_e_p if len(bel.people_e_p) == n else (0.0,) * n
    ev_i = bel.people_e_v if len(bel.people_e_v) == n else (0.0,) * n
    ep = tuple(float(a) + float(ev_scale) * (float(bel.e_v) + float(b)) * t for a, b in zip(ep0, ev_i))
    return bel.with_(people=ppl, people_e_p=ep)


def pred_people(bel, t: float, people=None):
    """PLAN-TIME belief for person_model="pred" (tamp/plan_cert.py): the people observed at time 0, seen by a trajectory that starts t seconds later.  Each person is moved
    to c + v_hat t and gets stale += t, so the checker's prediction disc grows with the TOTAL elapsed time  r0 + e_v t_e + a_p t_e^2 / 2,  t_e = t + (time into the
    trajectory)  (the checker advances the centre by v_hat for the time into the trajectory itself).  people: optional [(c, v_hat), ...] replacing the belief's own
    (v_hat = 0 for a bare position); None keeps bel.people / bel.people_vel.  Always returns a belief with person_model "pred" (also for t = 0)."""
    if people is None:
        ppl = [np.asarray(c, float)[:2] for c in bel.people]
        vel = [np.asarray(v, float)[:2] for v in bel.people_vel] if len(bel.people_vel) == len(ppl) else [np.zeros(2) for _ in ppl]
    else:
        ppl = [np.asarray(p[0], float)[:2] for p in people]
        vel = [np.asarray(p[1], float)[:2] if len(p) > 1 and p[1] is not None else np.zeros(2) for p in people]
    n, t = len(ppl), max(0.0, float(t))
    st0 = bel.people_stale if people is None and len(bel.people_stale) == n else (0.0,) * n
    return bel.with_(people=tuple(c + v * t for c, v in zip(ppl, vel)), people_vel=tuple(vel), people_stale=tuple(float(s) + t for s in st0),
                     people_e_p=bel.people_e_p if people is None else (), people_e_v=bel.people_e_v if people is None else (), person_model="pred")


TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)       # s: timed-wait start offsets probed, smallest first


def probe_wait(certify_fn, taus=TAUS):
    """smallest tau in `taus` for which certify_fn(tau) is truthy -> (tau or None, [(tau, ok), ...] probes actually run)."""
    probes = []
    for tau in taus:
        ok = bool(certify_fn(float(tau)))
        probes.append((float(tau), ok))
        if ok:
            return float(tau), probes
    return None, probes
