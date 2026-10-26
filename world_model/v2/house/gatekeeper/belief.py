"""gatekeeper/belief.py -- set-valued belief used by the checkers.  SIMULATION ONLY.

Every number that is NOT already a project constant (teleop/base.py, shield.ShieldCfg, geom.py) is a named ASSUMPTION field of GKConfig.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..teleop import base as BS
from ..teleop import geom as GM

G = 9.81


@dataclass
class GKConfig:
    # ---- gatekeeper schedule ----
    k_ticks: int = 5                                   # gatekeeper step every K sim ticks (TICK=0.02 s -> 0.1 s)
    t_switch: tuple = (1.0, 0.6, 0.3, 0.1)             # candidate switch times T_S [s], tried in this (descending) order
    h: float = 0.05                                    # s, trajectory / checker time grid
    t_back_max: float = 6.0                            # s, backup that has not reached rest by then is rejected ("no_rest")
    steady_frac: float = 0.60                          # held (steady) twists are limited to |centripetal cup accel| <= steady_frac * L so the backup keeps tangential headroom (sqrt(1-0.6^2)=0.8 L)
    plan_frac: float = 0.97                            # planner uses plan_frac * a_cup_max (headroom so the checker, at a_cup_max, passes)
    # ---- ASSUMPTIONS (not measured; state in every report) ----
    v_p_max: float = 1.5                               # ASSUMPTION m/s, max person speed (reachable disc grows at this rate)
    e_p: float = 0.0                                   # ASSUMPTION m, person tracking error added to the disc radius (0 = truth people source)
    latency_s: float = 0.0                             # ASSUMPTION s, age of the people observation (disc already grown by v_p_max*latency)
    cup_margin: float = 0.90                           # ASSUMPTION <=1, a_cup_max = cup_margin * ratio(mu_lo) * g (unmodelled tray tilt / slosh)
    # ---- OPT-IN cup-row gating (default "always" == the original: cup row at the cup's live position even with empty hands, bit-identical) ----
    cup_row: str = "always"                            # "always" | "attached": cup row only while the cup rides on the robot (see cup_attached)
    k_pose: float = 1.0                                # pose box half-width = k_pose * sim.loc.sigma_xy (+ ShieldCfg.loc_margin); 0 when pose source is truth
    th_over_xy: float = 0.15                           # ASSUMPTION rad/m, e_th = th_over_xy * e_xy (LocCfg defaults: vio_drift_yaw 0.003 / sig_drift_xy 0.02)
    fp_step: float = 0.03                              # m, footprint boundary sampling step (half of it is added to the clearance requirement)
    sweep_slack: bool = True                           # add the rigorous inter-sample sweep allowance 0.5*h*(speed of boundary point [+ v_p_max])
    falsify_tol: float = 0.01                          # ASSUMPTION m, numerical tolerance (open-loop pose drift, EDT quantisation) before a committed backup is declared falsified
    tol: float = 1e-9
    regime2_fallback: str = "escape"                   # ShieldCfg.fallback used by the internal Shield in regime 2 ("brake" = the emergency brake, "escape")
    # ---- OPT-IN improvement A: velocity-aware person reachable set (default "speed" == the original speed-only disc, bit-identical) ----
    person_model: str = "speed"                        # "speed" | "accel" | "pred" ("pred" = plan-time model: prediction disc only + separating-motion exit rule, checks.py)
    e_v: float = 0.2                                   # ASSUMPTION m/s, error of the person velocity estimate v_hat (only used by "accel" / "pred")
    a_p: float = 2.0                                   # ASSUMPTION m/s^2, max person acceleration (only used by "accel" / "pred")
    vel_source: str = "state"                          # "state" = velocity of sim.people_state (truth source); "fd" = finite difference of the observed positions over the last gk steps
    # ---- OPT-IN improvement B: certified candidate set (default "hold" == only u_op is tried, bit-identical) ----
    candidates: str = "hold"                           # "hold" | "set"
    cand_t_switch: tuple = (0.6, 0.1)                  # T_S tried for every deviation candidate (descending); hold-u_op keeps t_switch
    cand_max_checks: int = 24                          # cap on (candidate, T_S) plan+check attempts per gatekeeper step (excluding the hold-u_op attempts)
    cand_scales: tuple = (0.75, 0.5, 0.25)             # u_op scaling candidates
    cand_rot_deg: tuple = (20.0, 40.0)                 # heading rotation of the translational part, +/-
    cand_lat_frac: float = 0.5                         # lateral sidestep component = +/- cand_lat_frac * V_STICK_y added to u_op
    cand_spin_frac: float = 0.5                        # pure rotation = +/- cand_spin_frac * V_STICK_wz
    cand_backoff_frac: float = 0.4                     # back-off speed = cand_backoff_frac * V_STICK along the binding direction
    cand_time_budget_ms: float = 0.0                   # 0 = off (deterministic); >0 stops the candidate search after this many ms
    # ---- OPT-IN people source (default "truth" == sim.people_state, bit-identical).  "tracker": the gatekeeper owns a people_track.PerceptionStack (CPU
    #      ray-cast head camera -> KF tracks); the belief is built from the TRACKS (see build_belief); sim.people_state is used for the metric audit only ----
    people_source: str = "truth"                       # "truth" | "tracker"
    perception: str = "depth"                          # run_pt.parse_cond condition; CPU-only (mj_multiRay, no GL): "depth" | "seg" | "param:SIG:LAT:DROP"
    conformal: object = None                           # None -> e_p = k_sigma*sqrt(lam_max(P_pp)); else path to a ConformalMargin json (or a ConformalMargin): e_p = radius(bin)
    k_sigma: float = 2.0                               # confidence multiplier for e_p (no conformal) and e_v
    seed: int = 0                                      # perception noise seed
    audit_match_r: float = 1.0                         # m, a true person is "tracked" if some track centre is within this distance (audit only)
    audit_near_r: float = 3.0                          # m, an untracked true person counts in n_untracked_near if within this distance of the robot footprint
    # ---- OPT-IN theory fixes (defaults == current behaviour, bit-identical) ----
    keep_rule: str = "recertify"                       # "recertify" (re-run Checkers 1+2 on the kept trajectory under the CURRENT belief) | "monitor" (T1 KEEP rule: falsification monitor, see monitor.py) [GAP-2]
    belief_filter: str = "none"                        # "none" | "intersect": per-person set-membership filter, previous predicted disc ∩ observation disc (see monitor.py)
    cup_chord: bool = False                            # True: Checker 1 cup row subtracts the inter-sample chord bound (h^2/8)(2|alpha||a_b| + 2 alpha^2 |r|) per segment [GAP-1]
    # ---- OPT-IN segment-mode additions (defaults == current behaviour, bit-identical; tamp/segments.py, gatekeeper/ready.py) ----
    c2_ready: str = "off"                              # "off" | "rank": C2-ready ranking of admitted segment candidates (gatekeeper/ready.py); never admits/rejects
    c2_ready_dmin: float = 0.05                        # m, ASSUMPTION: minimum route progress of the next move for a rest pose to count as ready
    c2_ready_speeds: tuple = (0.3, 0.2, 0.12)          # m/s speeds the readiness move may use (== tamp.segments.SPEEDS; after a stop the planner replans and picks the speed)
    seg_offsets: tuple = ()                            # m, lateral route offsets (left normal +) tried at segment boundaries besides the current one; () = off
    seg_k_lat: float = 1.5                             # 1/s lateral offset tracking gain (vy)
    seg_v_lat_max: float = 0.15                        # m/s cap on the lateral tracking speed
    seg_offset_taper: float = 0.6                      # m: the offset tapers linearly to 0 over the last seg_offset_taper m before s_stop (route ends on the route)
    c2_ready_scope: str = "both"                       # "both" = today: --c2-ready rank runs in the planner's route streams AND at the executor's segment boundaries | "gate" = boundaries only (planner solves keep the cheap first-passing rule)
    c2_ready_tube_h: float = 0.0                       # s, tracked-tube term of the ranking (tamp/segments.py, boundaries only): off-tube rest poses first; 0 = off (bit-identical)
    c2_ready_tube_clear: float = 0.0                   # m, extra clearance beyond d_person that the rest pose must keep to the tracked tube over [t_rest, t_rest + tube_h] (0 = only "not touching"; the hop-usable value is ~0.5 m, see tamp/hop_e3.py)
    # ---- OPT-IN plan-prediction monitor (tamp/plan_monitor.py; default "off" == no monitor, bit-identical) ----
    plan_monitor: str = "off"                          # "off" | "on": flag a segment route whose people prediction (constant velocity from plan time) is falsified
    plan_eps_p: float = 0.30                           # m, position deviation from the plan's prediction that counts as a violation
    plan_eps_v: float = 0.20                           # m/s, velocity deviation that counts as a violation
    plan_r_rel: float = 2.0                            # m, a violation counts only within this distance of the remaining planned route (== the planner's near_m)
    plan_confirm: int = 2                              # consecutive violating gatekeeper steps before the flag
    plan_decision: str = "timing"                      # after a flag: "timing" = keep driving the old route while t_ready + t_stop + margin < t_conf, else brake | "brake" = always brake (coast on the committed trajectory)
    plan_conf_margin: float = 0.5                      # s, t_margin of the timing test
    # ---- OPT-IN "planner owns safety" mode (user decision 10-24..26; defaults == current behaviour, bit-identical) ----
    runtime_rows: str = "all"                          # "all" | "cup": cup = the RUNTIME checkers evaluate only the cup + hw rows (no wall / person rows), people part of the falsification monitor is off,
                                                       #   and regime 2 (internal Shield) is never entered: a pose falsification commits the cup-safe brake and reports the route refused (checks.py docstring)
    plan_cert: str = "off"                             # "off" | "pred": tamp.streams.Streams default for the plan-time certifier (tamp/plan_cert.py: cup + hw + walls + people with person_model="pred" over the WHOLE route)


@dataclass
class Belief:
    """worst-case sets.  Enlarging any of e_xy, e_th, e_p, v_p_max, e_v, a_p, latency_s or lowering mu_lo only SHRINKS what is admitted (monotone)."""
    e_xy: float = 0.0                                   # m, translation error half-width of the pose box (Euclidean)
    e_th: float = 0.0                                   # rad, heading error half-width
    mu_lo: float = 0.18                                 # friction set [mu_lo, mu_hi]; the cup limit uses mu_lo
    mu_hi: float = 0.35
    kappa: float = 0.80
    theta_spill: float = 0.30
    cup_margin: float = 0.90
    d_wall: float = 0.10                                # m buffer to static obstacles
    d_wall_dock: float = 0.03                           # m buffer inside dock_radius of a dock (as the Shield)
    dock_radius: float = 0.8
    docks: tuple = ()                                   # [(x, y)]
    d_person: float = 0.10                              # m, extra clearance beyond the person disc
    e_p: float = 0.0
    v_p_max: float = 1.5
    latency_s: float = 0.0
    people: tuple = ()                                  # world xy of each tracked person (current observation)
    people_e_p: tuple = ()                              # per-person ADDITIONAL position error (tracker source); () = none -> only the global e_p
    people_e_v: tuple = ()                              # per-person ADDITIONAL velocity-estimate error; () = none -> only the global e_v
    people_stale: tuple = ()                            # per-person ADDITIONAL observation age [s] (time since the track's last update); () = none -> only latency_s
    people_vel: tuple = ()                              # world velocity estimate v_hat of each person (() = unknown -> zeros); only used when person_model == "accel"
    person_model: str = "speed"                         # "speed": disc(c, r0 + v_p t);  "accel": also disc(c + v_hat t, r0 + e_v t + a_p t^2/2), margin = max of the two;  "pred": ONLY the second disc (+ exit rule)
    e_v: float = 0.2                                    # m/s  velocity-estimate error half-width
    a_p: float = 2.0                                    # m/s^2 person acceleration bound
    pose_wall: np.ndarray = field(default_factory=lambda: np.zeros(3))      # map-frame pose the static-obstacle check is anchored at (estimate)
    pose_people: np.ndarray = field(default_factory=lambda: np.zeros(3))    # pose the relative people check is anchored at (people are sensed relative to the robot)
    grid: object = None                                 # OccGrid (occupancy EDT)
    r_cup: np.ndarray = field(default_factory=lambda: np.zeros(2))          # cup position in the base frame
    fp: dict = field(default_factory=lambda: dict(GM.FP))
    a_hw: np.ndarray = field(default_factory=lambda: BS.A_HW.copy())
    v_cap: np.ndarray = field(default_factory=lambda: BS.V_STICK.copy())

    @property
    def ratio(self):
        return min(self.kappa * self.mu_lo, math.tan(self.theta_spill))

    @property
    def a_cup_max(self):
        """m/s^2, bound on |cup inertial acceleration| (no-slip disc of the flat tray, worst case mu = mu_lo, times cup_margin)."""
        return self.cup_margin * self.ratio * G

    def with_(self, **kw):
        d = dict(self.__dict__); d.update(kw)
        return Belief(**d)


CUP_FREE_MARGIN = 1e3      # cup not on the robot: cup row inactive (a_cup_max huge -> only the hardware slew box limits acceleration); same value as tamp.world.FREE_CUP_MARGIN


cup_attached = GM.cup_attached     # shared rule (teleop/geom.py), also used by the Shield


from ..ocs.state import WorldState, RobotState
from world_model.v2.tray_rep.sets.ir import Pose
@dataclass
class RobotInfo:
    """what the belief needs from the ROBOT itself (proprioception / limits / map -- not world facts): pose estimate, controller-frame pose estimate,
    hardware slew limits, localisation sigma (None = pose source is truth), occupancy grid (the map the wall check reads)."""
    pose: np.ndarray                                    # (3,) pose the relative people check is anchored at (sim.pose())
    ctrl_pose: np.ndarray                               # (3,) map-frame pose the static-obstacle check is anchored at (sim.ctrl_pose() if present, else pose)
    a_hw: np.ndarray                                    # (3,) hardware acceleration limits
    loc_sigma: float | None = None                      # sim.loc.sigma_xy, None when the pose source is truth
    grid: object = None                                 # OccGrid (sim.occ_ctrl)


def robot_info_from_sim(sim) -> RobotInfo:
    """read the robot's own proprioception / map from the (sim) robot: allowed, it is not a world fact."""
    loc = getattr(sim, "loc", None)
    pose_t = sim.pose()
    pose_c = sim.ctrl_pose() if hasattr(sim, "ctrl_pose") else pose_t
    return RobotInfo(pose=pose_t, ctrl_pose=pose_c, a_hw=sim.a_hw, loc_sigma=float(loc.sigma_xy) if loc is not None else None, grid=sim.occ_ctrl)


def belief_from_ws(ws, robot: RobotInfo, shield_cfg, gk: GKConfig, fp=None, docks=(), people_vel=None, e_v=None, cup: str = "cup") -> Belief:
    """Belief from a WorldState (facts) + the robot's own info (proprioception, limits, map).  NO simulator access:
      cup position in the base frame  <- ws.world_pose(cup) and ws.robot (base xy + yaw)
      cup on the robot?               <- ws.attached(cup, default=True)   (unknown = attached, the conservative rule of geom.cup_attached)
      people                          <- the PERSON objects of ws (position, vel = v_hat); if ws.meta["people_estimated"] (tracker / VLM-perception producers)
                                         the per-person pos_err / vel_err / stale fields are used and the global e_v is 0 -- the legacy `tracks=` mapping.
    people_vel / e_v: explicit overrides, same semantics as build_belief."""
    from ..ocs.consumers import person_ids
    sig = 0.0 if robot.loc_sigma is None else float(robot.loc_sigma)
    e_xy = gk.k_pose * sig + shield_cfg.loc_margin + (shield_cfg.loc_margin_k * sig if robot.loc_sigma is not None else 0.0)
    est = bool(ws.meta.get("people_estimated", False))
    pids = person_ids(ws)
    P = [ws.objects[i] for i in pids]
    if not est and people_vel is None and gk.person_model in ("accel", "pred") and gk.vel_source == "state":
        people_vel = tuple(np.zeros(2) if o.vel is None else np.asarray(o.vel, float)[:2].copy() for o in P)
    R = GM.rot(ws.robot.yaw if ws.robot.yaw is not None else math.atan2(ws.robot.base.R[1][0], ws.robot.base.R[0][0]))
    cup_xy = np.asarray(ws.world_pose(cup).p, float)[:2]
    r_c = R.T @ (cup_xy - np.array([ws.robot.base.p[0], ws.robot.base.p[1]]))
    cup_margin = gk.cup_margin
    if getattr(gk, "cup_row", "always") == "attached" and not ws.attached(cup, default=True):
        r_c, cup_margin = np.zeros(2), CUP_FREE_MARGIN               # the cup is not on the robot: base motion cannot slide it
    ppl = tuple(np.asarray(o.pose.p, float)[:2].copy() for o in P)
    if est:
        extra = dict(people_vel=tuple(np.zeros(2) if o.vel is None else np.asarray(o.vel, float)[:2].copy() for o in P),
                     e_v=0.0 if e_v is None else float(e_v),
                     people_e_p=tuple(float(o.pos_err or 0.0) for o in P), people_e_v=tuple(float(o.vel_err or 0.0) for o in P),
                     people_stale=tuple(float(o.stale or 0.0) for o in P))
    else:
        extra = dict(people_vel=tuple(people_vel) if people_vel is not None else (), e_v=gk.e_v if e_v is None else float(e_v))
    return Belief(e_xy=e_xy, e_th=gk.th_over_xy * e_xy, mu_lo=shield_cfg.mu_lo, mu_hi=shield_cfg.mu_hi, kappa=shield_cfg.kappa,
                  theta_spill=shield_cfg.theta_spill, cup_margin=cup_margin, d_wall=shield_cfg.d_wall, d_wall_dock=shield_cfg.d_wall_dock,
                  dock_radius=shield_cfg.dock_radius, docks=tuple(tuple(d) for d in docks), d_person=shield_cfg.d_person,
                  e_p=gk.e_p, v_p_max=gk.v_p_max, latency_s=gk.latency_s,
                  person_model=gk.person_model, a_p=gk.a_p, people=ppl, **extra,
                  pose_wall=np.asarray(robot.ctrl_pose, float).copy(), pose_people=np.asarray(robot.pose, float).copy(), grid=robot.grid,
                  r_cup=np.asarray(r_c, float), fp=dict(fp or GM.FP), a_hw=np.asarray(robot.a_hw, float).copy())


def build_belief(sim, shield_cfg, gk: GKConfig, fp=None, docks=(), people_vel=None, e_v=None, tracks=None) -> Belief:
    """Belief from the live sim (offline): the facts come from SimFactsProducer (truth) -> WorldState -> belief_from_ws; tracks (tracker people source)
    replace the PERSON objects via tracks_to_people.  Bit-identical to _build_belief_legacy (gatekeeper/tests/test_belief_from_ws.py)."""
    from ..ocs.producers import SimFactsProducer, tracks_to_people
    ws = WorldState(t=0.0, robot=RobotState(base=Pose(np.eye(3), np.zeros(3))))
    SimFactsProducer().update(ws, sim, 0.0)
    if tracks is not None:
        tracks_to_people(ws, tracks)
    return belief_from_ws(ws, robot_info_from_sim(sim), shield_cfg, gk, fp=fp, docks=docks, people_vel=people_vel, e_v=e_v)


def _build_belief_legacy(sim, shield_cfg, gk: GKConfig, fp=None, docks=(), people_vel=None, e_v=None, tracks=None) -> Belief:
    """Belief from the live sim (offline): pose box from sim.loc (0 if pose source is truth), people from sim.people_state, occupancy sim.occ_ctrl.
    tracks (tracker people source; list of dicts xy, v, e_p, e_v, stale): the people come from the TRACKS instead of sim.people_state:
        centre c = track mean xy, v_hat = track mean velocity, per-person e_p / e_v (added to the global gk.e_p / 0) and stale = time since the last update
        (added to gk.latency_s for the radius growth of THAT person only; the accel disc centre c + v_hat t is NOT advanced by it, c is already predicted to now)."""
    loc = getattr(sim, "loc", None)
    sig = float(loc.sigma_xy) if loc is not None else 0.0
    e_xy = gk.k_pose * sig + shield_cfg.loc_margin + (shield_cfg.loc_margin_k * sig if loc is not None else 0.0)
    if tracks is None and people_vel is None and gk.person_model in ("accel", "pred") and gk.vel_source == "state":
        people_vel = tuple(np.asarray(v, float).copy() for _, v in (sim.people_state or []))
    pose_t = sim.pose()
    pose_c = sim.ctrl_pose() if hasattr(sim, "ctrl_pose") else pose_t
    R = GM.rot(sim.base.yaw)
    cup_xy = sim.d.xpos[sim.cup][:2]
    r_c = R.T @ (cup_xy - np.array([sim.base.x, sim.base.y]))
    cup_margin = gk.cup_margin
    if getattr(gk, "cup_row", "always") == "attached" and not cup_attached(sim):
        r_c, cup_margin = np.zeros(2), CUP_FREE_MARGIN               # the cup is not on the robot: base motion cannot slide it
    if tracks is not None:
        ppl = tuple(np.asarray(t["xy"], float).copy() for t in tracks)
        extra = dict(people_vel=tuple(np.asarray(t["v"], float).copy() for t in tracks), e_v=0.0 if e_v is None else float(e_v),
                     people_e_p=tuple(float(t["e_p"]) for t in tracks), people_e_v=tuple(float(t["e_v"]) for t in tracks),
                     people_stale=tuple(float(t["stale"]) for t in tracks))
    else:
        ppl = tuple(np.asarray(xy, float).copy() for xy, _ in (sim.people_state or []))
        extra = dict(people_vel=tuple(people_vel) if people_vel is not None else (), e_v=gk.e_v if e_v is None else float(e_v))
    return Belief(e_xy=e_xy, e_th=gk.th_over_xy * e_xy, mu_lo=shield_cfg.mu_lo, mu_hi=shield_cfg.mu_hi, kappa=shield_cfg.kappa,
                  theta_spill=shield_cfg.theta_spill, cup_margin=cup_margin, d_wall=shield_cfg.d_wall, d_wall_dock=shield_cfg.d_wall_dock,
                  dock_radius=shield_cfg.dock_radius, docks=tuple(tuple(d) for d in docks), d_person=shield_cfg.d_person,
                  e_p=gk.e_p, v_p_max=gk.v_p_max, latency_s=gk.latency_s,
                  person_model=gk.person_model, a_p=gk.a_p, people=ppl, **extra,
                  pose_wall=np.asarray(pose_c, float).copy(), pose_people=np.asarray(pose_t, float).copy(), grid=sim.occ_ctrl,
                  r_cup=np.asarray(r_c, float), fp=dict(fp or GM.FP), a_hw=np.asarray(sim.a_hw, float).copy())
