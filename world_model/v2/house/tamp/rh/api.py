"""tamp/rh/api.py -- shared interface for RECEDING-HORIZON base planners (lattice, MPPI).  SIMULATION ONLY.

User decisions (2026-10-26): the PLANNER plans safely (no low-level person/wall guardrails; runtime checks only the cup and actuator limits,
GKConfig.runtime_rows="cup"); people are modelled by their PREDICTION (constant velocity + error growing e_v = 0.15 m/s, a_p = 0), not the
1.5 m/s worst case.  Whole routes cannot be certified against a growing error, so planners certify only a SHORT stretch and replan.

Contract (both planners):
  * a Proposer turns the current state into candidate trajectories, best-first.  Each candidate is a gatekeeper Traj (body twist profile on the
    grid h from the CURRENT state): a nominal part of at most RHConfig.horizon_s, then a cup-safe backup that ends at rest (Traj.t_sw = backup start).
  * admit() certifies candidates with tamp/plan_cert.certify_plan (person_model="pred", e_v, a_p from RHConfig; walls, cup, hw) and returns the
    first one that passes.  Proposers never certify themselves: the same admit() is used for every planner, so results are comparable.
  * the loop (tamp/rh/loop.py) executes the admitted Traj under runtime_rows="cup", replans RHConfig.lead_s before the nominal part ends from the
    predicted state, keeps the committed Traj if no candidate is admitted (it brakes to rest by construction), and counts every replan / refusal / wait.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from ...gatekeeper.traj import Traj


@dataclass
class RHConfig:
    horizon_s: float = 1.5          # max nominal (driving) time per candidate before its cup-safe backup [s]  (ASSUMPTION, swept 1.0-2.0)
    lead_s: float = 0.3             # replan this long before the committed nominal part ends [s]
    e_v: float = 0.15               # people prediction error growth [m/s]  (user decision 10-26)
    a_p: float = 0.0                # people prediction acceleration term [m/s^2]  (user decision 10-26)
    v_max: float = 0.3              # base speed cap [m/s] (same as segments.SPEEDS[0]; do not raise without asking)
    wait_s: float = 0.5             # at rest with no admitted candidate: replan every wait_s [s] (counted as WAIT)
    max_wait_s: float = 60.0        # give up after this much total waiting [s] (reported, not a failure of safety)


@dataclass
class State:
    pose: tuple                      # map pose (x, y, yaw) of sample 0
    v: tuple                         # body twist (vx, vy, wz) at sample 0
    t: float                         # sim time of sample 0
    people: list                     # [(c_xy, v_hat_xy), ...] observed at time t (truth or tracker, as configured)


@dataclass
class Candidate:
    traj: Traj                       # starts at State (Traj.t0 = State.t); nominal <= horizon_s, then backup to rest
    progress: float                  # route length gained [m] at the END of the nominal part (V(start) - V(end)); ranking only
    source: str                      # e.g. "lattice:arc+0.3", "mppi:k17"
    meta: dict = field(default_factory=dict)


class Proposer(Protocol):
    name: str

    def propose(self, state: State, world, goal_xy: np.ndarray, route_xy: np.ndarray | None, cfg: RHConfig) -> Sequence[Candidate]:
        """candidates from `state`, best-first (the loop admits the first that certifies).  route_xy: optional guide polyline (scene route)."""
        ...


def plan_gk(gk, cfg: RHConfig):
    """planner-side checker config: prediction person model with the RHConfig error growth (walls/people/cup/hw all ON)."""
    from dataclasses import replace
    return replace(gk, person_model="pred", e_v=cfg.e_v, a_p=cfg.a_p, runtime_rows="all")


def admit(cands: Sequence[Candidate], state: State, world, gk, cfg: RHConfig, holding: bool = True, t0: float = 0.0):
    """-> (Candidate or None, log).  log = [(source, ok, min_margin, reason)] for every candidate tried (in order).
    t0 = age of the people observation at the trajectory start [s] (e.g. lead_s when replanning from the predicted end of the committed nominal):
    the prediction error grows over it.  t0 = 0 is the original behaviour."""
    from ..plan_cert import certify_plan
    g = plan_gk(gk, cfg)
    log = []
    for c in cands:
        ok, m, r, _ = certify_plan(c.traj, world, g, people=state.people, t0=float(t0), pose0=state.pose, holding=holding)
        log.append((c.source, bool(ok), float(m), str(r)))
        if ok:
            return c, log
    return None, log
