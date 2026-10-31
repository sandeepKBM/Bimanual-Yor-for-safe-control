"""tamp/plan_cert.py -- the PLAN-TIME certifier (user decisions 2026-10-24..26: the PLANNER plans safely; the runtime layer keeps only cup + actuator rows).  SIMULATION ONLY.

certify_plan(traj, world, gk, people, t0)  =  gatekeeper.checks.check1 + check2 on `traj` with
    * cup + hw rows           (the world's payload belief: cup lever arm, payload mu_lo; hands empty -> no cup row, as runtime_belief_at),
    * wall rows               (check1's EDT rows on the world's occupancy grid, pose box e_xy of the world),
    * person rows             person_model="pred": ONLY the prediction disc disc(c + v_hat t_e, r0 + e_v t_e + a_p t_e^2/2) of every person (no speed disc, no worst-case
                              1.5 m/s) + the separating-motion exit rule for a start inside a keep-out (gatekeeper/checks.py docstring, user decision 10-26),
                              check2 adds "rest_person" (prediction disc at the rest sample) instead of the passive row.
`people` = [(c, v_hat), ...] observed at time 0; `t0` = delay [s] between that observation and the START of the trajectory (the people are advanced by v_hat t0 and their
error has already grown for t0).  Always evaluated with runtime_rows="all" -- whatever the runtime layer runs, the planner's certificate has walls and people.
e_v, a_p, e_p, latency_s, d_person, d_wall come from the GKConfig / world (CLI: run_tamp --pred-ev --pred-ap).

certify_segments(trajs, ...) chains timed segments as the planner / executor do (tamp/segments.py): segment j+1 starts from the END OF THE NOMINAL PART of segment j
(pose at sample round(t_sw / h), time += t_sw); the last segment is checked in full (nominal + backup to rest).

Return of both: (ok, min_margin, reason, margins).  ok = every row of check1 and check2 holds; min_margin / reason = the first failing row, else the tightest INFORMATIVE row
(rest_speed / hw_slew are 0 by construction and excluded from the tightest-row search); margins = row -> worst margin (m, m/s^2; "person_exit" m per step).
ASSUMPTIONS: constant-velocity prediction of every person (v_hat = the observation's velocity estimate); the error growth is the checker's e_v, a_p (no cap); a route
is assumed to start t0 after the observation (action durations before it are not modelled).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ..gatekeeper.checks import check1, check2, world_poses
from ..gatekeeper.traj import Traj
from . import people_pred as PP
from .world import World, runtime_belief_at

TIGHT_BY_CONSTRUCTION = ("rest_speed", "hw_slew")


def world_people(world: World):
    """[(c, v_hat)] of the world's current people belief (world.bel_people), [] if none."""
    b = world.bel_people
    if b is None or not len(b.people):
        return []
    vel = b.people_vel if len(b.people_vel) == len(b.people) else tuple(np.zeros(2) for _ in b.people)
    return [(np.asarray(c, float)[:2], np.asarray(v, float)[:2]) for c, v in zip(b.people, vel)]


def plan_belief(world: World, pose, holding: bool, item: str | None = None, people=None, t: float = 0.0, gk=None):
    """the belief the plan-time certifier uses for a trajectory that starts at map pose `pose`, `t` s after the people observation: the world's payload/wall belief (no
    people) + the people (default: world.bel_people) predicted by tamp.people_pred.pred_people, person_model "pred".  e_v / a_p are read from `gk` (default world.gk)."""
    g = world.gk if gk is None else gk
    bel = runtime_belief_at(world, pose, holding, item, people=False).with_(e_v=float(g.e_v), a_p=float(g.a_p))
    return PP.pred_people(bel, t, world_people(world) if people is None else people)


def _plan_gk(gk):
    return gk if getattr(gk, "runtime_rows", "all") == "all" else dataclasses.replace(gk, runtime_rows="all")


def _verdict(o1, m1, r1, g1, o2, m2, r2, g2):
    mg = dict(g1)
    mg.update(g2)
    if not o1:
        return False, m1, r1, mg
    if not o2:
        return False, m2, r2, mg
    rows = {k: v for k, v in mg.items() if k not in TIGHT_BY_CONSTRUCTION}
    if not rows:
        return True, 0.0, "none", mg
    k = min(rows, key=rows.get)
    return True, float(rows[k]), k, mg


def certify_plan(traj: Traj, world: World, gk, people=None, t0: float = 0.0, pose0=None, holding: bool = True, item: str | None = "tray", bel=None, spec=None):
    """-> (ok, min_margin, reason, margins).  pose0: map pose of the trajectory's first sample (default world.q0); holding / item: payload (tray in hand by default; holding=False
    = hands empty, no cup row).  bel: optional prebuilt plan belief (overrides everything above).
    spec: OPT-IN (default None = the hard-coded rows above, bit-identical): a tamp.spec_default.Spec -- the person / wall rows are then read from the constraint spec
    (tamp/spec_cert.certify_spec; margins = the checks.py cup / hw rows + one entry per spec atom)."""
    if spec is not None:
        from .spec_cert import certify_spec
        return certify_spec(traj, spec, world, gk, people, t0, pose0, holding, item, bel)
    gk = _plan_gk(gk)
    if bel is None:
        pose0 = world.q0.xyth() if pose0 is None else pose0
        bel = plan_belief(world, pose0, holding, item if holding else None, people, t0, gk)
    o1, m1, r1, g1 = check1(traj, bel, gk)
    o2, m2, r2, g2 = check2(traj, bel, gk)
    return _verdict(o1, m1, r1, g1, o2, m2, r2, g2)


def certify_segments(trajs, world: World, gk, people=None, t0: float = 0.0, pose0=None, holding: bool = True, item: str | None = "tray"):
    """chain `trajs` (timed segments) from pose0 at delay t0; see the module docstring.  -> (ok, min_margin, reason, margins); reason is "seg j: <row>"."""
    gk = _plan_gk(gk)
    pose = tuple(float(x) for x in (world.q0.xyth() if pose0 is None else pose0))
    t = float(t0)
    agg, worst, who, first_fail = {}, math.inf, "none", None
    ok_all = True
    for j, tr in enumerate(trajs):
        ok, m, r, mg = certify_plan(tr, world, gk, people, t, pose, holding, item)
        for k, v in mg.items():
            agg[k] = min(agg.get(k, math.inf), float(v))
        if not ok:
            ok_all = False
            if first_fail is None:
                first_fail = (float(m), f"seg {j}: {r}")
        elif m < worst:
            worst, who = float(m), f"seg {j}: {r}"
        if j < len(trajs) - 1:                                          # next segment starts at the end of this one's nominal part
            n = tr.N if tr.t_sw <= 0 else min(tr.N, int(round(tr.t_sw / tr.h)))
            pose = tuple(float(x) for x in world_poses(tr.poses()[n:n + 1], np.asarray(pose, float))[0])
            t += n * tr.h
    if not ok_all:
        return False, first_fail[0], first_fail[1], agg
    return True, (0.0 if worst == math.inf else worst), who, agg
