"""carry.py -- h5 teleop carry: the sampling planner as a COMMAND PROPOSER in front of the unchanged shield.  SIMULATION ONLY.

    operator stick (sloppy, closed-loop human model) --intent--> MPPI planner --proposal--> teleop.shield.Shield.filter (hard base QP) --> sim

The planner's cost tracks the operator's stick held over the horizon (shared autonomy), plus wall clearance (the shield's EDT), person discs,
mu-robust (CVaR) cup spill risk and smoothness.  The shield is NOT modified; it still filters every proposal (and may be bypassed with
`shield="off"` to expose the planner's own cup-slip behaviour).  `set_mu` sets the SIM's cup-tray sliding friction (truth).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import dynamics as DY, fields as FL, cost as CO, planner as PL
from ..teleop import base as BS, geom as GM, operator as OP, shield as SH
from ..teleop.sim import TeleopSim, TICK, Cmd


def set_mu(sim, mu):
    """cup <-> tray sliding friction (MuJoCo max-combines the pair, so BOTH bodies' geoms are set).  Returns n geoms changed."""
    m = sim.m
    n = 0
    for b in (sim.cup, sim.tray):
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) == int(b) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_friction[g, 0] = mu
                n += 1
    return n


class CarryPlanner:
    """owns the CarryProblem + SamplingPlanner for one TeleopSim.  `propose(sim, cmd)` -> a Cmd whose twist is the planner's first step."""

    def __init__(self, sim, K=1024, H=25, dt=0.1, dts=None, mu_mode="robust", mu_nominal=0.25, cfg_kw=None, use_tilt=False, method="mppi", n_people=4, seed=0,
                 use_graph=True, follow_pred=True, dock_passthrough=True, d_wall=0.10, d_wall_dock=0.03, docks=(), sigma=(0.12, 0.12, 0.25, 0.02, 0.02)):
        self.sim = sim
        g = sim.occ_ctrl
        pts = GM.boundary_pts(step=0.05)
        ck = dict(margin=d_wall, d0=0.20, person_r=GM.PERSON_CLEAR_R + 0.10, mu_nominal=None if mu_mode == "robust" else mu_nominal, w_cbf=10.0)
        ck.update(cfg_kw or {})
        self.cfg = CO.CostCfg(**ck)
        self.d_wall, self.d_wall_dock, self.docks = d_wall, d_wall_dock, [np.asarray(x, float) for x in docks]
        dts = dts if dts is not None else tuple([0.02] * 4 + [0.04] * 3 + [0.1] * (H - 7))
        model = DY.BaseModel(dt=dt, dts=dts, frame="body", a_hw=tuple(float(x) for x in sim.a_hw), v_max=tuple(float(x) for x in BS.V_STICK), use_tilt=use_tilt)
        self.field = FL.Grid2DField(g)
        self.prob = CO.CarryProblem(model, H, pts, np.zeros(len(pts)), np.zeros(len(pts)), self.field, self.cfg, mode="intent", r_cup=(0, 0),
                                    n_people=n_people, v_scale=tuple(float(x) for x in BS.V_STICK))
        self.pl = PL.SamplingPlanner(self.prob, K=K, method=method, sigma=sigma, plan_dt=TICK, use_graph=use_graph, seed=seed, prim_speed=(0.7, 0.7))
        self.use_tilt = use_tilt
        self.n_people = n_people
        self.warmup_ms = self.pl.warmup()
        self.last_u = np.zeros(3)
        self.last_a = np.zeros(2)
        self.follow_pred = follow_pred
        self.dock_passthrough = dock_passthrough
        self.n_pass = 0
        self.t_last = None
        self.log = []

    def _state(self, cmd):
        sim = self.sim
        d = sim.d
        R = GM.rot(sim.base.yaw)
        r_c = R.T @ (d.xpos[sim.cup][:2] - np.array([sim.base.x, sim.base.y]))
        c = d.xmat[sim.tray].reshape(3, 3)[:, 2]
        pose = sim.pose()
        near = any(np.hypot(pose[0] - x[0], pose[1] - x[1]) < 0.8 for x in self.docks)
        ppl = [[xy[0], xy[1], v[0], v[1]] for xy, v in (sim.people_state if sim.with_people else [])]
        return dict(pose=pose, v=sim.base.v.copy(), r_cup=r_c, c_world=c[:2].copy(), margin_add=-(self.d_wall - self.d_wall_dock) if near else 0.0, people=ppl,
                    u_int=np.asarray(cmd.twist, float))

    def propose(self, cmd):
        st = self._state(cmd)
        if self.dock_passthrough and st["margin_add"] < 0:          # inside the dock radius: the operator's creep/dock is forwarded untouched (the shield still filters it)
            self.n_pass += 1
            self.last_u = np.asarray(cmd.twist, float).copy()
            self.last_a = np.zeros(2)
            self.log.append((0.0, 0.0, 0.0))
            return cmd
        S = self.prob.S
        self.prob.set_state(pose=st["pose"], v=st["v"], u_prev=self.last_u, a_prev=self.last_a, r_cup=st["r_cup"], u_int=st["u_int"], people=st["people"], margin_add=st["margin_add"],
                            d0=(0.02 if st["margin_add"] < 0 else self.cfg.d0), c_world=st["c_world"])
        U, info = self.pl.plan(elapsed=TICK)
        u = U[0, :3].astype(float)
        if self.follow_pred:          # command the PREDICTED velocity profile one tick ahead (plan accelerations, not step targets): keeps the cup jerk planned
            v0 = st["v"]; v1 = self.pl.v_pred[0]
            u = v0 + (v1 - v0) * min(1.0, TICK / float(self.prob.dtv[0]))
        self.last_a = self.pl.a_pred[0].copy()
        self.last_u = U[0, :3].copy() if not self.use_tilt else U[0, :self.prob.nu].copy()
        jog = np.asarray(cmd.jog, float).copy()
        if self.use_tilt:                                         # tilt rates (c_x', c_y') -> jog angular rates (wx = -c_y', wy = c_x')
            jog[3], jog[4] = -U[0, 4], U[0, 3]
        self.log.append((info["ms"], info["cost"], info["h_min"]))
        return Cmd(twist=u, lift_v=cmd.lift_v, jog=jog, grasp=cmd.grasp, release=cmd.release)


def run_episode(mode="planner", seed=0, mu_true=None, task="carry", scene="h5", max_s=45.0, shield_mode="on", K=1024, mu_mode="robust", mu_nominal=0.25,
                use_tilt=False, cfg_kw=None, tail_s=1.0, with_people=True, H=25, dt=0.1, follow_pred=True, dock_passthrough=True, verbose=False):
    """mode: "shield" (operator -> shield, the existing system) | "planner" (operator -> planner -> shield) | "raw" (operator only)."""
    from ..teleop import run as RU
    carry = task == "carry"
    sim = TeleopSim(scene, grasped=carry, seed=seed, with_people=with_people, counter=not carry, desk=not carry)
    if mu_true is not None:
        set_mu(sim, float(mu_true))
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(seed, task, sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], back_route=sc.back_route, back_mode=sc.back_mode)
    use_shield = (mode != "raw") and shield_mode == "on"
    shield = SH.Shield(None, occ=sim.occ_ctrl, a_hw=sim.a_hw) if use_shield else SH.Passthrough()
    docks = [wp["cafe_counter_pick"][:2], wp["lab_desk_place"][:2]] if not carry else [wp["cafe_counter_pick"][:2]]
    if use_shield:
        shield.docks = docks
    cp = None
    if mode == "planner":
        cp = CarryPlanner(sim, K=K, H=H, dt=dt, follow_pred=follow_pred, dock_passthrough=dock_passthrough, mu_mode=mu_mode, mu_nominal=mu_nominal, use_tilt=use_tilt, cfg_kw=cfg_kw, docks=docks, method="mppi")
    rows, done_t, t_wall = [], None, time.time()
    for k in range(int(max_s / TICK)):
        obs = sim.observe()
        cmd = op.step(obs)
        prop = cp.propose(cmd) if cp is not None else cmd
        safe, info = shield.filter(sim, prop)
        o = sim.tick(safe)
        o.update(info)
        u_hw = np.clip(cmd.twist, obs["v"] - sim.a_hw * TICK, obs["v"] + sim.a_hw * TICK)
        o["interv_vs_raw"] = float(np.linalg.norm((u_hw - safe.twist) / BS.V_STICK))
        o["interv_prop"] = float(np.linalg.norm((prop.twist - safe.twist) / BS.V_STICK))     # what the SHIELD changed in the proposal
        o["u_op"] = cmd.twist.copy(); o["u_safe"] = safe.twist.copy(); o["phase"] = op.phase
        rows.append(o)
        if op.done and done_t is None:
            done_t = o["t"]
        if done_t is not None and o["t"] - done_t >= tail_s:
            break
    summ = RU.summarize(sim, op, shield, rows, use_shield, seed, task, done_t, [0.0], time.time() - t_wall)
    ip = np.array([r["interv_prop"] for r in rows])
    pk = np.array(wp["cafe_counter_pick"][:2], float)
    dd = np.array([np.hypot(*(r["pose"][:2] - pk)) for r in rows]); tt = np.array([r["t"] for r in rows])
    t_dock = float(tt[np.argmax(dd < 0.8)]) if (dd < 0.8).any() else None
    summ.update(mode=mode, shield_mode=shield_mode if mode != "raw" else "off", mu_true=mu_true, mu_mode=mu_mode if mode == "planner" else None,
                shield_changed_proposal_frac_gt2pct=float((ip > 0.02).mean()), shield_changed_proposal_mean=float(ip.mean()),
                t_dock_entry_s=t_dock, mean_speed=float(np.mean([np.hypot(*r["v"][:2]) for r in rows])), cup_slip_max_m=summ["max_cup_slip_on_tray_m"])
    if cp is not None:
        lg = np.array(cp.log)
        summ.update(planner_warmup_ms=cp.warmup_ms, planner_ms_median=float(np.median(lg[:, 0])), planner_ms_p95=float(np.percentile(lg[:, 0], 95)), planner_K=K,
                    planner_summary=cp.pl.summary())
    return summ, rows
